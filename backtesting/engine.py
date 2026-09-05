"""
backtesting/engine.py
=====================
Deterministic Backtesting Engine for daily, long-only trading strategies.

Design Principles
-----------------
* **Strict Look-Ahead Bias Prevention**:
  - Signals evaluated at bar `t` close NEVER execute at bar `t`.
    They execute at bar `t+1` Open.
  - Signal providers receive only ``data.iloc[: current_index + 1]`` to
    structurally prevent future peeking.
  - Final-bar entry or exit signals are ignored (no bar `t+1` exists).
* **Deterministic Execution Order** per bar `i`:
  1. Execute any pending order (queued by bar `i-1` signal) at bar `i` Open.
  2. Check active-position risk controls: max holding period, then
     stop-loss / take-profit using bar `i` Open, High, and Low.
  3. Snapshot portfolio equity at bar `i` Close.
  4. Evaluate strategy signals for bar `i` (to be executed on bar `i+1`).
* **Conservative Stop/Target Rule**:
  - If both stop-loss and take-profit levels are touched on the same candle,
    stop-loss executes first (conservative deterministic tie-break).
* **End of Dataset Accounting**:
  - If a position is still open on the final bar, it is force-closed at
    the final Close price (with sell slippage) and labeled END_OF_DATA.
    This is an accounting close, not a signal-based fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Protocol

import pandas as pd

from backtesting.execution import TradeExecutor
from backtesting.metrics import calculate_metrics
from backtesting.models import (
    BacktestConfig,
    BacktestResult,
    EquityPoint,
    ExitReason,
)
from backtesting.portfolio import Portfolio
from data.schema import validate_ohlcv


# ---------------------------------------------------------------------------
# Strategy signal protocol
# ---------------------------------------------------------------------------


class StrategySignalProvider(Protocol):
    """Protocol defining the strategy evaluation interface.

    The engine calls these methods once per bar using historical data sliced
    strictly up to ``current_index`` inclusive.  Implementations MUST NOT
    access rows beyond ``data.iloc[current_index]``.

    Sprint 3's ``CompiledStrategy`` implements this protocol.
    """

    def should_enter(self, current_index: int, data: pd.DataFrame) -> bool:
        """Return True when entry conditions are satisfied at ``current_index``."""
        ...

    def should_exit(self, current_index: int, data: pd.DataFrame) -> bool:
        """Return True when exit conditions are satisfied at ``current_index``."""
        ...

    def get_stop_price(
        self,
        entry_price: float,
        current_index: int,
        data: pd.DataFrame,
    ) -> Optional[float]:
        """Return the stop-loss price for an entry at ``entry_price``, or None."""
        ...

    def get_take_profit_price(
        self,
        entry_price: float,
        stop_price: Optional[float],
        current_index: int,
        data: pd.DataFrame,
    ) -> Optional[float]:
        """Return the take-profit price for an entry at ``entry_price``, or None."""
        ...


# ---------------------------------------------------------------------------
# Internal pending-order state
# ---------------------------------------------------------------------------


@dataclass
class _PendingOrder:
    """An order queued by yesterday's signal, waiting for next-bar execution.

    Attributes
    ----------
    is_entry:
        True when the pending order is a BUY; False for SELL.
    signal_date:
        The bar date on which the signal was generated.  Stored in the
        Trade record so execution timing can be audited.
    """

    is_entry: bool
    signal_date: date


# ---------------------------------------------------------------------------
# Data validation helper
# ---------------------------------------------------------------------------


def _validate_and_normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the one canonical OHLCV contract, then normalize column case."""
    validated = validate_ohlcv(data)
    validated.columns = [str(column).lower() for column in validated.columns]
    return validated


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Deterministic, look-ahead-safe backtesting engine.

    Parameters
    ----------
    config:
        Immutable configuration for the backtest run.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.executor = TradeExecutor(
            commission_pct=config.commission_pct,
            slippage_pct=config.slippage_pct,
        )

    def run(
        self,
        data: pd.DataFrame,
        signal_provider: StrategySignalProvider,
        *,
        evaluation_start: pd.Timestamp | datetime | None = None,
    ) -> BacktestResult:
        """Run a deterministic backtest.

        Parameters
        ----------
        data:
            Historical OHLCV DataFrame.  Column names are case-insensitive.
            Must have a DatetimeIndex or a ``Date`` column.
        signal_provider:
            An object implementing :class:`StrategySignalProvider`.  Each
            method receives only historical data up to the current bar.

        Returns
        -------
        BacktestResult
            Fully populated result including trades, equity curve, and metrics.
        """
        df = _validate_and_normalize_data(data)
        n_bars = len(df)
        if evaluation_start is None:
            start_index = 0
        else:
            boundary = pd.Timestamp(evaluation_start)
            if boundary not in df.index:
                raise ValueError("evaluation_start must identify an exact market-data bar.")
            location = df.index.get_loc(boundary)
            if not isinstance(location, int):
                raise ValueError("evaluation_start must identify exactly one bar.")
            start_index = location

        portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            position_size_pct=self.config.position_size_pct,
            executor=self.executor,
        )

        equity_curve: list[EquityPoint] = []
        pending_order: Optional[_PendingOrder] = None

        dates = [
            ts.date() if isinstance(ts, (pd.Timestamp, datetime)) else ts
            for ts in df.index
        ]

        for i in range(start_index, n_bars):
            curr_date: date = dates[i]
            open_p = float(df["open"].iloc[i])
            high_p = float(df["high"].iloc[i])
            low_p = float(df["low"].iloc[i])
            close_p = float(df["close"].iloc[i])

            # -----------------------------------------------------------
            # Step 1 — Execute pending order at today's Open
            # -----------------------------------------------------------
            if pending_order is not None:
                if pending_order.is_entry:
                    if not portfolio.is_invested():
                        # Apply slippage first. Risk features are then resolved
                        # strictly from information available before this Open.
                        entry_fill = self.executor.calculate_buy_fill(open_p)
                        risk_index = i - 1
                        hist = df.iloc[:i]
                        stop_p = signal_provider.get_stop_price(
                            entry_fill, risk_index, hist
                        )
                        target_p = signal_provider.get_take_profit_price(
                            entry_fill, stop_p, risk_index, hist
                        )
                        portfolio.open_position(
                            entry_date=curr_date,
                            raw_price=open_p,
                            override_fill_price=entry_fill,
                            stop_price=stop_p,
                            target_price=target_p,
                            entry_signal_date=pending_order.signal_date,
                        )
                else:  # pending exit (signal-based)
                    if portfolio.is_invested():
                        portfolio.close_position(
                            exit_date=curr_date,
                            raw_price=open_p,
                            exit_reason=ExitReason.SIGNAL,
                            exit_signal_date=pending_order.signal_date,
                        )
                pending_order = None

            # -----------------------------------------------------------
            # Step 2 — Evaluate risk controls on active position
            # -----------------------------------------------------------
            if portfolio.is_invested() and portfolio.position is not None:
                pos = portfolio.position

                # Maximum holding period check (takes priority over intraday triggers)
                if (
                    self.config.maximum_holding_days is not None
                    and (curr_date - pos.entry_date).days >= self.config.maximum_holding_days
                ):
                    portfolio.close_position(
                        exit_date=curr_date,
                        raw_price=open_p,
                        exit_reason=ExitReason.MAX_HOLDING_PERIOD,
                    )
                else:
                    # Intraday stop-loss / take-profit evaluation using bar H/L
                    trigger = self.executor.evaluate_stop_and_target(
                        open_price=open_p,
                        high_price=high_p,
                        low_price=low_p,
                        stop_price=pos.stop_price,
                        target_price=pos.target_price,
                    )
                    if trigger is not None:
                        exit_reason, fill_price = trigger
                        # fill_price already includes sell slippage from TradeExecutor
                        portfolio.close_position(
                            exit_date=curr_date,
                            raw_price=open_p,  # informational; overridden below
                            exit_reason=exit_reason,
                            override_fill_price=fill_price,
                        )

            # -----------------------------------------------------------
            # Step 3 — Snapshot equity at Close
            # -----------------------------------------------------------
            equity_curve.append(
                EquityPoint(
                    date=curr_date,
                    equity=portfolio.get_equity(close_p),
                    cash=portfolio.cash,
                    is_invested=portfolio.is_invested(),
                )
            )

            # -----------------------------------------------------------
            # Step 4 — Evaluate signals for tomorrow's open
            #          (only if a next bar exists)
            # -----------------------------------------------------------
            if i < n_bars - 1:
                hist_slice = df.iloc[: i + 1]
                if not portfolio.is_invested() and pending_order is None:
                    if signal_provider.should_enter(i, hist_slice):
                        pending_order = _PendingOrder(is_entry=True, signal_date=curr_date)
                elif portfolio.is_invested() and pending_order is None:
                    if signal_provider.should_exit(i, hist_slice):
                        pending_order = _PendingOrder(is_entry=False, signal_date=curr_date)

        # -------------------------------------------------------------------
        # Step 5 — End-of-data accounting close (position still open at EOF)
        # -------------------------------------------------------------------
        if portfolio.is_invested():
            last_date = dates[-1]
            last_close = float(df["close"].iloc[-1])
            portfolio.close_position(
                exit_date=last_date,
                raw_price=last_close,
                exit_reason=ExitReason.END_OF_DATA,
            )
            if equity_curve:
                # Rewrite the last equity point now that the position is closed
                equity_curve[-1] = EquityPoint(
                    date=last_date,
                    equity=portfolio.get_equity(last_close),
                    cash=portfolio.cash,
                    is_invested=False,
                )

        # -------------------------------------------------------------------
        # Step 6 — Calculate performance metrics and assemble result
        # -------------------------------------------------------------------
        metrics = calculate_metrics(
            initial_capital=self.config.initial_capital,
            trades=portfolio.closed_trades,
            equity_curve=equity_curve,
        )

        return BacktestResult(
            config=self.config,
            start_date=dates[start_index],
            end_date=dates[-1],
            initial_capital=self.config.initial_capital,
            final_equity=equity_curve[-1].equity if equity_curve else self.config.initial_capital,
            metrics=metrics,
            trades=portfolio.closed_trades,
            equity_curve=equity_curve,
        )
