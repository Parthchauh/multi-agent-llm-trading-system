"""paper_trading/portfolio.py
===========================
Deterministic paper portfolio tracking cash, fills, and mark-to-market equity.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime

from paper_trading.models import (
    PaperFill,
    PaperPortfolioState,
    PaperPosition,
    PositionStatus,
    SignalType,
)


class PaperPortfolio:
    """Manages paper trading cash balance and active positions deterministically."""

    def __init__(self, initial_capital: float = 100_000.0) -> None:
        if not math.isfinite(initial_capital) or initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.realized_pnl = 0.0
        self.positions: dict[str, PaperPosition] = {}
        self.closed_positions: list[PaperPosition] = []
        self.fills: list[PaperFill] = []

    def get_open_position(self, symbol: str) -> PaperPosition | None:
        return self.positions.get(symbol.upper())

    def process_fill(self, fill: PaperFill, *, strategy_id: str) -> PaperPosition | None:
        """Process a simulated fill and update cash and positions."""
        if any(existing.fill_id == fill.fill_id for existing in self.fills):
            raise ValueError("Duplicate fill identifier.")
        symbol = fill.symbol.upper()

        if fill.signal_type == SignalType.ENTER_LONG:
            if symbol in self.positions:
                raise RuntimeError(f"Position already open for {symbol}; concurrent positions disabled.")
            cost = fill.total_cost
            if cost > self.cash:
                raise ValueError(f"Insufficient cash for fill: required={cost:.2f}, available={self.cash:.2f}")
            self.cash -= cost
            self.fills.append(fill)
            pos = PaperPosition(
                position_id=str(uuid.uuid4())[:8],
                strategy_id=strategy_id,
                symbol=symbol,
                entry_time=fill.fill_time,
                entry_price=fill.fill_price,
                quantity=fill.quantity,
                status=PositionStatus.OPEN,
                realized_pnl=-fill.commission,
                unrealized_pnl=0.0,
            )
            self.positions[symbol] = pos
            return pos

        elif fill.signal_type == SignalType.EXIT_LONG:
            if symbol not in self.positions:
                raise RuntimeError(f"No open position to exit for {symbol}.")
            pos = self.positions[symbol]
            if fill.quantity != pos.quantity:
                raise ValueError("Exit quantity must equal the held quantity; partial exits unsupported.")
            self.positions.pop(symbol)
            self.fills.append(fill)
            gross_proceeds = fill.fill_price * pos.quantity
            net_proceeds = gross_proceeds - fill.commission
            trade_pnl = (fill.fill_price - pos.entry_price) * pos.quantity - fill.commission + pos.realized_pnl
            self.cash += net_proceeds
            self.realized_pnl += trade_pnl
            closed_pos = PaperPosition(
                position_id=pos.position_id,
                strategy_id=pos.strategy_id,
                symbol=symbol,
                entry_time=pos.entry_time,
                entry_price=pos.entry_price,
                quantity=pos.quantity,
                status=PositionStatus.CLOSED,
                exit_time=fill.fill_time,
                exit_price=fill.fill_price,
                realized_pnl=trade_pnl,
                unrealized_pnl=0.0,
            )
            self.closed_positions.append(closed_pos)
            return closed_pos

        raise ValueError('Only ENTER_LONG and EXIT_LONG fills are supported.')

    def snapshot(self, timestamp: datetime, current_prices: dict[str, float]) -> PaperPortfolioState:
        """Generate an exact mark-to-market snapshot of portfolio equity."""
        holdings_val = 0.0
        unrealized = 0.0
        for symbol, pos in self.positions.items():
            price = current_prices.get(symbol, pos.entry_price)
            val = price * pos.quantity
            holdings_val += val
            unrealized += (price - pos.entry_price) * pos.quantity + pos.realized_pnl

        total_equity = self.cash + holdings_val
        return PaperPortfolioState(
            timestamp=timestamp,
            cash=round(self.cash, 4),
            holdings_value=round(holdings_val, 4),
            total_equity=round(total_equity, 4),
            realized_pnl=round(self.realized_pnl, 4),
            unrealized_pnl=round(unrealized, 4),
            open_positions_count=len(self.positions),
        )
