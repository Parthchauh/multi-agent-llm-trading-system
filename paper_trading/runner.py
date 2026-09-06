"""paper_trading/runner.py
========================
Paper trading engine executing frozen strategies bar-by-bar on live/incoming data.

Workflow:
Frozen Strategy -> Incoming Market Bar -> Compiler -> Signal -> Next-Bar Execution -> Portfolio -> Journal
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from backtesting.models import BacktestConfig
from data.schema import validate_ohlcv
from paper_trading.journal import PaperJournal
from paper_trading.models import (
    PaperFill,
    PaperPortfolioState,
    PaperTradeSignal,
    SignalType,
)
from paper_trading.portfolio import PaperPortfolio
from strategies.compiler import StrategyCompiler
from strategies.schema import StrategySchema
from strategies.validator import StrategyValidator


class PaperTradingRunner:
    """Executes a strictly frozen strategy over incoming chronological bars."""

    def __init__(
        self,
        strategy: StrategySchema,
        *,
        config: BacktestConfig | None = None,
        journal_path: Path | str | None = None,
    ) -> None:
        if not isinstance(strategy, StrategySchema):
            raise TypeError("strategy must be a canonical StrategySchema instance.")
        validation = StrategyValidator().validate(strategy)
        if not validation.is_valid:
            raise ValueError(
                "PaperTradingRunner requires a semantically valid StrategySchema: "
                + "; ".join(validation.errors)
            )
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.portfolio = PaperPortfolio(initial_capital=self.config.initial_capital)
        self.journal = PaperJournal(output_path=journal_path)
        self.compiler = StrategyCompiler()
        self._pending_signal: PaperTradeSignal | None = None
        self._previous_history: pd.DataFrame | None = None
        self.strategy_id = strategy.metadata.name or "strategy-frozen"

    def process_bar(
        self,
        symbol: str,
        historical_bar_series: pd.DataFrame,
    ) -> tuple[PaperFill | None, PaperPortfolioState]:
        """Process one new chronological bar.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL')
            historical_bar_series: DataFrame of OHLCV up to and including the latest bar.

        Returns:
            Tuple of (optional executed fill on this bar, updated portfolio state).
        """
        clean_data = validate_ohlcv(historical_bar_series)
        if self._previous_history is not None:
            prior_bars = clean_data.iloc[:-1]
            if (
                not prior_bars.index.equals(self._previous_history.index)
                or not prior_bars.equals(self._previous_history)
            ):
                raise ValueError(
                    "Incoming paper-trading history must append exactly one unrevised bar."
                )
        current_bar = clean_data.iloc[-1]
        current_time = clean_data.index[-1].to_pydatetime()
        current_open = float(current_bar["Open"])
        current_close = float(current_bar["Close"])

        executed_fill: PaperFill | None = None

        # 1. Execute pending signal queued from prior bar's close at current Open
        if self._pending_signal is not None and self._pending_signal.symbol == symbol:
            sig = self._pending_signal
            self._pending_signal = None

            if sig.signal_type == SignalType.ENTER_LONG and not self.portfolio.get_open_position(symbol):
                # Apply slippage
                fill_price = current_open * (1.0 + self.config.slippage_pct)
                target_alloc = self.portfolio.cash * (self.config.position_size_pct / 100.0)
                quantity = target_alloc / fill_price
                commission = fill_price * quantity * self.config.commission_pct
                slippage_cost = (fill_price - current_open) * quantity

                if target_alloc <= self.portfolio.cash:
                    fill = PaperFill(
                        fill_id=str(uuid.uuid4())[:8],
                        symbol=symbol,
                        fill_time=current_time,
                        signal_type=SignalType.ENTER_LONG,
                        fill_price=round(fill_price, 4),
                        quantity=round(quantity, 4),
                        commission=round(commission, 4),
                        slippage=round(slippage_cost, 4),
                    )
                    self.portfolio.process_fill(fill, strategy_id=self.strategy_id)
                    self.journal.record_fill(fill)
                    executed_fill = fill

            elif sig.signal_type == SignalType.EXIT_LONG and self.portfolio.get_open_position(symbol):
                pos = self.portfolio.get_open_position(symbol)
                assert pos is not None
                fill_price = current_open * (1.0 - self.config.slippage_pct)
                commission = fill_price * pos.quantity * self.config.commission_pct
                slippage_cost = (current_open - fill_price) * pos.quantity

                fill = PaperFill(
                    fill_id=str(uuid.uuid4())[:8],
                    symbol=symbol,
                    fill_time=current_time,
                    signal_type=SignalType.EXIT_LONG,
                    fill_price=round(fill_price, 4),
                    quantity=round(pos.quantity, 4),
                    commission=round(commission, 4),
                    slippage=round(slippage_cost, 4),
                )
                self.portfolio.process_fill(fill, strategy_id=self.strategy_id)
                self.journal.record_fill(fill)
                executed_fill = fill

        # 2. Recompile strategy over available history to check for new signal at Close
        compiled = self.compiler.compile(self.strategy, clean_data)
        has_entry = bool(compiled.entry_signals.iloc[-1])
        has_exit = bool(compiled.exit_signals.iloc[-1])

        # Queue signal for next-bar Open fill
        if has_entry and not self.portfolio.get_open_position(symbol):
            self._pending_signal = PaperTradeSignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_time=current_time,
                signal_type=SignalType.ENTER_LONG,
                intended_execution_time=current_time,
                target_position_size_pct=self.config.position_size_pct,
                rationale="Entry condition satisfied on bar close",
            )
            self.journal.record_signal(self._pending_signal)
        elif has_exit and self.portfolio.get_open_position(symbol):
            self._pending_signal = PaperTradeSignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_time=current_time,
                signal_type=SignalType.EXIT_LONG,
                intended_execution_time=current_time,
                target_position_size_pct=0.0,
                rationale="Exit condition satisfied on bar close",
            )
            self.journal.record_signal(self._pending_signal)

        # 3. Snapshot portfolio state
        state = self.portfolio.snapshot(current_time, {symbol: current_close})
        self.journal.record_snapshot(state)
        self._previous_history = clean_data.copy(deep=True)
        return executed_fill, state
