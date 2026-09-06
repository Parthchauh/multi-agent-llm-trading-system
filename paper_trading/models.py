"""paper_trading/models.py
========================
Strict, frozen Pydantic contracts for simulated paper trading.

No real brokerage execution or external side-effects are permitted.
Every model enforces strict validation and immutable record-keeping.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class SignalType(str, Enum):
    ENTER_LONG = "ENTER_LONG"
    EXIT_LONG = "EXIT_LONG"
    HOLD = "HOLD"


class PaperTradeSignal(_FrozenStrictModel):
    """Immutable signal emitted by a frozen strategy for simulated next-bar execution."""

    strategy_id: str = Field(min_length=1, max_length=128)
    symbol: str = Field(min_length=1, max_length=16)
    signal_time: datetime
    signal_type: SignalType
    intended_execution_time: datetime
    target_position_size_pct: float = Field(default=100.0, ge=0.0, le=100.0)
    rationale: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def execution_must_follow_signal(self) -> "PaperTradeSignal":
        if self.intended_execution_time < self.signal_time:
            raise ValueError("intended_execution_time cannot be earlier than signal_time.")
        return self


class PaperFill(_FrozenStrictModel):
    """Deterministic fill record from a simulated execution."""

    fill_id: str = Field(min_length=1, max_length=128)
    symbol: str = Field(min_length=1, max_length=16)
    fill_time: datetime
    signal_type: SignalType
    fill_price: float = Field(gt=0.0)
    quantity: float = Field(gt=0.0)
    commission: float = Field(ge=0.0)
    slippage: float = Field(ge=0.0)

    @property
    def gross_value(self) -> float:
        return self.fill_price * self.quantity

    @property
    def total_cost(self) -> float:
        return self.gross_value + self.commission + self.slippage


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class PaperPosition(_FrozenStrictModel):
    """Paper trading position with deterministic accounting."""

    position_id: str = Field(min_length=1, max_length=128)
    strategy_id: str = Field(min_length=1, max_length=128)
    symbol: str = Field(min_length=1, max_length=16)
    entry_time: datetime
    entry_price: float = Field(gt=0.0)
    quantity: float = Field(gt=0.0)
    status: PositionStatus = PositionStatus.OPEN
    exit_time: datetime | None = None
    exit_price: float | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.quantity


class PaperPortfolioState(_FrozenStrictModel):
    """Snapshot of simulated paper account equity and balances."""

    timestamp: datetime
    cash: float = Field(ge=0.0)
    holdings_value: float = Field(ge=0.0)
    total_equity: float = Field(ge=0.0)
    realized_pnl: float
    unrealized_pnl: float
    open_positions_count: int = Field(ge=0)
