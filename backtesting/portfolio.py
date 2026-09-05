"""
backtesting/portfolio.py
========================
Portfolio state manager tracking cash, open positions, realized trades, and equity.

Design Constraints
------------------
* Long-only, single asset.
* Maximum of one active position at any time.
* Whole shares only (no fractional shares in Sprint 2).
* No leverage (cannot spend more cash than available including commission).
"""

from __future__ import annotations

import math
from datetime import date
from typing import List, Optional

from backtesting.execution import TradeExecutor
from backtesting.models import ExitReason, Position, Trade, TradeSide


class Portfolio:
    """Tracks cash, open position, closed trades, and total equity.

    Parameters
    ----------
    initial_capital:
        Starting cash balance.
    position_size_pct:
        Percentage of available cash to allocate per trade (0 < pct <= 100).
    executor:
        TradeExecutor instance for commission and slippage calculations.
    """

    def __init__(
        self,
        initial_capital: float,
        position_size_pct: float,
        executor: TradeExecutor,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.position_size_pct = float(position_size_pct)
        self.executor = executor

        self.position: Optional[Position] = None
        self.closed_trades: List[Trade] = []

    def is_invested(self) -> bool:
        """Return True if there is an active open position."""
        return self.position is not None

    def calculate_order_quantity(self, fill_price: float) -> int:
        """Calculate the integer number of whole shares to buy without exceeding allocated cash.

        Cash allocated = cash * (position_size_pct / 100.0)
        Effective cost per share = fill_price * (1 + commission_pct)
        """
        if self.cash <= 0 or fill_price <= 0:
            return 0

        allocated_cash = self.cash * (self.position_size_pct / 100.0)
        effective_cost_per_share = fill_price * (1.0 + self.executor.commission_pct)

        shares = int(math.floor(allocated_cash / effective_cost_per_share))
        return max(0, shares)

    def open_position(
        self,
        entry_date: date,
        raw_price: float,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None,
        entry_signal_date: Optional[date] = None,
        override_fill_price: Optional[float] = None,
    ) -> Optional[Position]:
        """Open a new long position if not already invested and cash allows.

        Returns
        -------
        Optional[Position]
            The opened position, or None if already invested or insufficient cash.
        """
        if self.is_invested():
            return None

        fill_price = (
            override_fill_price
            if override_fill_price is not None
            else self.executor.calculate_buy_fill(raw_price)
        )
        quantity = self.calculate_order_quantity(fill_price)

        if quantity <= 0:
            return None

        trade_value = fill_price * quantity
        commission = self.executor.calculate_commission(trade_value)
        total_cost = trade_value + commission

        # Deduct cash
        self.cash -= total_cost

        self.position = Position(
            entry_date=entry_date,
            entry_price=fill_price,
            quantity=quantity,
            stop_price=stop_price,
            target_price=target_price,
            entry_signal_date=entry_signal_date,
            commission_paid_entry=commission,
        )
        return self.position

    def close_position(
        self,
        exit_date: date,
        raw_price: float,
        exit_reason: ExitReason,
        exit_signal_date: Optional[date] = None,
        override_fill_price: Optional[float] = None,
    ) -> Optional[Trade]:
        """Close an active position and record the completed trade.

        Parameters
        ----------
        exit_date:
            Execution date of the exit.
        raw_price:
            Underlying price before sell slippage (if override_fill_price is None).
        exit_reason:
            Reason for exit (STOP_LOSS, TAKE_PROFIT, SIGNAL, etc.).
        exit_signal_date:
            Date the signal was generated (if applicable).
        override_fill_price:
            Pre-computed fill price (e.g. from TradeExecutor.evaluate_stop_and_target).
            If provided, raw_price sell slippage is not re-applied.

        Returns
        -------
        Optional[Trade]
            The completed Trade record, or None if no position was open.
        """
        if not self.is_invested() or self.position is None:
            return None

        pos = self.position

        if override_fill_price is not None:
            fill_price = override_fill_price
        else:
            fill_price = self.executor.calculate_sell_fill(raw_price)

        exit_value = fill_price * pos.quantity
        commission_exit = self.executor.calculate_commission(exit_value)
        net_proceeds = exit_value - commission_exit

        # Add proceeds to cash
        self.cash += net_proceeds

        total_commission = pos.commission_paid_entry + commission_exit
        gross_pnl = (fill_price - pos.entry_price) * pos.quantity
        net_pnl = gross_pnl - total_commission
        invested_capital = pos.entry_price * pos.quantity
        return_pct = (net_pnl / invested_capital) * 100.0 if invested_capital > 0 else 0.0
        holding_days = (exit_date - pos.entry_date).days

        trade = Trade(
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=exit_date,
            exit_price=fill_price,
            quantity=pos.quantity,
            gross_pnl=gross_pnl,
            commission_paid=total_commission,
            net_pnl=net_pnl,
            return_pct=return_pct,
            exit_reason=exit_reason,
            holding_days=max(0, holding_days),
            entry_signal_date=pos.entry_signal_date,
            exit_signal_date=exit_signal_date,
            side=TradeSide.LONG,
        )

        self.closed_trades.append(trade)
        self.position = None
        return trade

    def get_equity(self, current_close_price: float) -> float:
        """Calculate total current portfolio equity (cash + mark-to-market position value)."""
        if self.position is not None:
            position_value = self.position.quantity * current_close_price
            return self.cash + position_value
        return self.cash
