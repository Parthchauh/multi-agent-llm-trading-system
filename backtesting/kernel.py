"""Shared chronological bar execution for historical and paper adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from backtesting.models import ExitReason


@dataclass(frozen=True)
class PendingOrder:
    is_entry: bool
    signal_date: date


class UnifiedExecutionKernel:
    @staticmethod
    def execute(*, config, executor, portfolio, pending_order, signal_provider, df, i):
        curr_date = df.index[i].date()
        open_p = float(df["open"].iloc[i])
        high_p = float(df["high"].iloc[i])
        low_p = float(df["low"].iloc[i])
        # -----------------------------------------------------------
        # Step 1 — Execute pending order at today's Open
        # -----------------------------------------------------------
        if pending_order is not None:
            if pending_order.is_entry:
                if not portfolio.is_invested():
                    # Apply slippage first. Risk features are then resolved
                    # strictly from information available before this Open.
                    entry_fill = executor.calculate_buy_fill(open_p)
                    risk_index = i - 1
                    hist = df.iloc[:i]
                    stop_p = signal_provider.get_stop_price(entry_fill, risk_index, hist)
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
                config.maximum_holding_days is not None
                and (curr_date - pos.entry_date).days >= config.maximum_holding_days
            ):
                portfolio.close_position(
                    exit_date=curr_date,
                    raw_price=open_p,
                    exit_reason=ExitReason.MAX_HOLDING_PERIOD,
                )
            else:
                # Intraday stop-loss / take-profit evaluation using bar H/L
                trigger = executor.evaluate_stop_and_target(
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
