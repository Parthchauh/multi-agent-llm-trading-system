"""
backtesting/models.py
=====================
Immutable typed data models for the backtesting engine.

All models are either frozen Pydantic models (for validation-heavy configs)
or plain dataclasses (for high-volume result records where construction
speed matters more than schema enforcement).

Design constraints
------------------
* No LLM, no LangGraph, no Claude.
* No eval(), no exec().
* JSON-serialisable — the Performance Analyst agent (Sprint 5) will consume
  BacktestResult as a dict/JSON string.
* Fully typed — every field has an explicit Python type annotation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SignalType(str, Enum):
    """The type of trading signal generated on a given bar.

    Values
    ------
    BUY:
        Strategy entry condition became True.
    EXIT:
        Strategy exit condition became True while a position is open.
    HOLD:
        No actionable signal; maintain current state.
    """

    BUY = "BUY"
    EXIT = "EXIT"
    HOLD = "HOLD"


class TradeSide(str, Enum):
    """Direction of a trade.

    Sprint 2 supports long-only, so only LONG is used.
    SHORT is reserved for future sprints.
    """

    LONG = "LONG"
    SHORT = "SHORT"  # reserved


class ExitReason(str, Enum):
    """The reason a trade was closed.

    Values
    ------
    STOP_LOSS:
        Price touched or gapped through the stop-loss level.
    TAKE_PROFIT:
        Price touched or gapped through the take-profit level.
    SIGNAL:
        The strategy's exit condition became True.
    MAX_HOLDING_PERIOD:
        The position was held for the maximum allowed calendar days.
    END_OF_DATA:
        The dataset ended with an open position; forced close at final close
        price.  This is an accounting close, not a signal-based fill.
    """

    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    SIGNAL = "SIGNAL"
    MAX_HOLDING_PERIOD = "MAX_HOLDING_PERIOD"
    END_OF_DATA = "END_OF_DATA"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class BacktestConfig(BaseModel):
    """Immutable configuration for a single backtest run.

    Attributes
    ----------
    initial_capital:
        Starting cash in currency units (e.g. USD).  Must be > 0.
    commission_pct:
        Round-trip commission rate expressed as a decimal fraction.
        Example: ``0.001`` = 0.10% charged on BOTH entry and exit legs
        (charged separately each time).  Must be >= 0.
    slippage_pct:
        Slippage expressed as a decimal fraction applied to each fill.
        Example: ``0.0005`` = 0.05%.  For buys, fill_price = raw * (1 +
        slippage_pct).  For sells, fill_price = raw * (1 - slippage_pct).
        Must be >= 0.
    position_size_pct:
        Percentage of *current* available cash to allocate per trade.
        Example: ``10.0`` = 10%.  Range: (0, 100].
    maximum_holding_days:
        Optional hard cap on trade duration in calendar days.  If a
        position is held longer than this, it is force-closed on the next
        bar's open.  Must be > 0 if provided.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    initial_capital: float = Field(
        ...,
        description="Starting cash in currency units.  Must be > 0.",
    )
    commission_pct: float = Field(
        default=0.001,
        description="Commission rate per trade leg as decimal fraction (e.g. 0.001 = 0.1%).",
    )
    slippage_pct: float = Field(
        default=0.0005,
        description="Slippage per fill as decimal fraction (e.g. 0.0005 = 0.05%).",
    )
    position_size_pct: float = Field(
        default=10.0,
        description="Percentage of available capital to allocate per trade.  Range: (0, 100].",
    )
    maximum_holding_days: Optional[int] = Field(
        default=None,
        description="Hard cap on trade duration in calendar days.  None = no cap.  Must be > 0.",
    )

    @field_validator(
        "initial_capital",
        "commission_pct",
        "slippage_pct",
        "position_size_pct",
        "maximum_holding_days",
        mode="before",
    )
    @classmethod
    def numeric_configuration_must_not_be_boolean(cls, value: object) -> object:
        """Do not silently reinterpret a JSON boolean as a numeric setting."""
        if isinstance(value, bool):
            raise ValueError("Backtest numeric configuration values must not be boolean.")
        return value

    @field_validator("initial_capital")
    @classmethod
    def capital_positive(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"initial_capital must be a finite number. Got: {v}.")
        if v <= 0:
            raise ValueError(f"initial_capital must be > 0. Got: {v}.")
        return v

    @field_validator("commission_pct", "slippage_pct")
    @classmethod
    def cost_non_negative(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"commission_pct / slippage_pct must be a finite number. Got: {v}.")
        if v < 0:
            raise ValueError(f"commission_pct / slippage_pct must be >= 0. Got: {v}.")
        if v > 0.50:
            raise ValueError(
                f"commission_pct / slippage_pct of {v:.4f} exceeds 50% per leg — "
                "this is almost certainly a configuration error. "
                "Use a decimal fraction (e.g. 0.001 for 0.1%)."
            )
        return v

    @field_validator("position_size_pct")
    @classmethod
    def size_in_range(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"position_size_pct must be a finite number. Got: {v}.")
        if not (0 < v <= 100):
            raise ValueError(
                f"position_size_pct must be in (0, 100]. Got: {v}."
            )
        return v

    @field_validator("maximum_holding_days")
    @classmethod
    def holding_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError(
                f"maximum_holding_days must be > 0. Got: {v}."
            )
        return v


# ---------------------------------------------------------------------------
# Mid-flight data structures
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """A single-bar trading signal returned by a StrategySignalProvider.

    Attributes
    ----------
    bar_index:
        Zero-based index of the bar that generated the signal.
    signal_type:
        The action indicated by the signal (BUY / EXIT / HOLD).
    """

    bar_index: int
    signal_type: SignalType


@dataclass
class Position:
    """An open (unfilled) long position currently held by the Portfolio.

    Attributes
    ----------
    entry_date:
        Calendar date the position was opened (the execution bar date).
    entry_price:
        Actual fill price (after slippage) at entry.
    quantity:
        Number of whole shares held.  Always an integer >= 1.
    stop_price:
        Stop-loss trigger price.  None if no stop is configured.
    target_price:
        Take-profit trigger price.  None if no target is configured.
    entry_signal_date:
        Calendar date the BUY signal was generated (the *signal* bar, not
        the *execution* bar).  Used to prove next-bar execution.
    commission_paid_entry:
        Commission charged at entry in currency units.
    """

    entry_date: date
    entry_price: float
    quantity: int
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    entry_signal_date: Optional[date] = None
    commission_paid_entry: float = 0.0


# ---------------------------------------------------------------------------
# Completed trade record
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    """An immutable record of one completed round-trip trade.

    All prices are fill prices (slippage already applied).
    All PnL figures are in currency units.

    Attributes
    ----------
    entry_date:
        Execution date of the buy (next bar after signal).
    entry_price:
        Fill price at entry (after buy slippage).
    exit_date:
        Execution date of the sell.
    exit_price:
        Fill price at exit (after sell slippage).
    quantity:
        Number of whole shares traded.
    gross_pnl:
        ``(exit_price - entry_price) * quantity``.  Before commissions.
    commission_paid:
        Total commission charged (entry + exit legs).
    net_pnl:
        ``gross_pnl - commission_paid``.
    return_pct:
        ``net_pnl / (entry_price * quantity) * 100``.
    exit_reason:
        Why the trade was closed (:class:`ExitReason`).
    holding_days:
        Number of calendar days from entry_date to exit_date (inclusive).
    entry_signal_date:
        Bar date that generated the entry signal.  Proves next-bar fill.
    exit_signal_date:
        Bar date that generated the exit signal.  None for stop/target/EOD.
    side:
        Direction of the trade.  Always :attr:`TradeSide.LONG` in Sprint 2.
    """

    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    quantity: int
    gross_pnl: float
    commission_paid: float
    net_pnl: float
    return_pct: float
    exit_reason: ExitReason
    holding_days: int
    entry_signal_date: Optional[date] = None
    exit_signal_date: Optional[date] = None
    side: TradeSide = TradeSide.LONG

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary of this trade."""
        return {
            "entry_date": self.entry_date.isoformat(),
            "entry_price": round(self.entry_price, 6),
            "exit_date": self.exit_date.isoformat(),
            "exit_price": round(self.exit_price, 6),
            "quantity": self.quantity,
            "gross_pnl": round(self.gross_pnl, 6),
            "commission_paid": round(self.commission_paid, 6),
            "net_pnl": round(self.net_pnl, 6),
            "return_pct": round(self.return_pct, 4),
            "exit_reason": self.exit_reason.value,
            "holding_days": self.holding_days,
            "entry_signal_date": (
                self.entry_signal_date.isoformat()
                if self.entry_signal_date
                else None
            ),
            "exit_signal_date": (
                self.exit_signal_date.isoformat()
                if self.exit_signal_date
                else None
            ),
            "side": self.side.value,
        }


# ---------------------------------------------------------------------------
# Equity curve point
# ---------------------------------------------------------------------------


@dataclass
class EquityPoint:
    """A single point on the portfolio equity curve.

    Attributes
    ----------
    date:
        Calendar date this equity snapshot was taken (end of bar).
    equity:
        Total portfolio value: cash + mark-to-market value of any open
        position.
    cash:
        Uninvested cash at end of bar.
    is_invested:
        True if a position was open at end of bar (used for exposure calc).
    """

    date: date
    equity: float
    cash: float
    is_invested: bool

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "equity": round(self.equity, 4),
            "cash": round(self.cash, 4),
            "is_invested": self.is_invested,
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class BacktestMetrics:
    """Aggregated performance metrics computed from a completed backtest.

    All percentage values are expressed as percentages (e.g. ``15.2`` = 15.2%).
    Ratios (Sharpe, Sortino, Calmar, profit factor, payoff) are plain floats.
    Undefined / not-computable metrics use ``None`` rather than NaN or inf.

    Attributes
    ----------
    total_return_pct:
        ``(final_equity / initial_capital - 1) * 100``.
    cagr_pct:
        Compound Annual Growth Rate.  ``None`` if backtest duration < 2 days.
    max_drawdown_pct:
        Maximum peak-to-trough drawdown expressed as a positive percentage
        magnitude.  E.g. a 15.2% drawdown is reported as ``15.2``.
    sharpe_ratio:
        Annualised Sharpe ratio using daily equity returns.
        Risk-free rate = 0.  ``None`` if standard deviation of returns = 0.
    sortino_ratio:
        Annualised Sortino ratio using only downside daily returns.
        ``None`` if no downside observations exist.
    win_rate_pct:
        ``winning_trades / number_of_trades * 100``.
        ``None`` if no trades were made.
    profit_factor:
        ``net_profits / abs(net_losses)`` using the same net-PnL basis as
        winning and losing trade classification.
        ``None`` if there are no losing trades or no trades at all.
    expectancy:
        Mean net PnL per trade in currency units.
        ``None`` if no trades were made.
    number_of_trades:
        Total completed round-trip trades.
    winning_trades:
        Trades with net_pnl > 0.
    losing_trades:
        Trades with net_pnl <= 0.
    average_win:
        Mean net_pnl of winning trades.  ``None`` if no winning trades.
    average_loss:
        Mean net_pnl of losing trades (a negative number).
        ``None`` if no losing trades.
    largest_win:
        Maximum net_pnl among winning trades (net_pnl > 0).
        ``None`` if no winning trades exist.
    largest_loss:
        Minimum net_pnl among losing trades (net_pnl <= 0), typically negative.
        ``None`` if no losing trades exist.
    average_holding_days:
        Mean calendar days held per trade.  ``None`` if no trades.
    exposure_pct:
        Percentage of total backtest bars with an open position at bar close.
        Note: positions opened and closed intraday may not be counted.
    calmar_ratio:
        CAGR divided by maximum drawdown before output rounding.
        ``None`` when CAGR is not computable or maximum drawdown is zero, so a
        flat/no-drawdown path is not represented as an artificial infinity.
    payoff_ratio:
        Average net winning-trade PnL divided by the absolute average net
        losing-trade PnL.  ``None`` when either side is absent or the average
        loss is zero.
    turnover_pct:
        Full-period, non-annualised gross executed turnover: the sum of entry
        and exit fill notionals divided by average finite end-of-bar equity,
        expressed as a percentage.  Thus, a fully invested round trip is
        approximately 200%, because both legs are included.  It is ``0.0``
        when no completed trades exist and ``None`` when completed trades
        exist but a finite, positive denominator or trade notional cannot be
        formed.
    max_drawdown_duration_days:
        Longest peak-to-recovery (or peak-to-final-observation when still
        unrecovered) drawdown duration in calendar days.  It is ``0`` when
        there is no drawdown or fewer than two finite equity observations.
    max_drawdown_duration_bars:
        Longest peak-to-recovery (or peak-to-final-observation) drawdown
        duration measured as elapsed equity-curve bar intervals.  It is ``0``
        under the same degenerate conditions as
        ``max_drawdown_duration_days``.
    metrics_version:
        Version string identifying the metric computation semantics.
        "1.2" retains the version 1.1 net-PnL profit-factor semantics and adds
        Calmar, payoff, turnover, and drawdown-duration metrics.
    """

    total_return_pct: float
    cagr_pct: Optional[float]
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    win_rate_pct: Optional[float]
    profit_factor: Optional[float]
    expectancy: Optional[float]
    number_of_trades: int
    winning_trades: int
    losing_trades: int
    average_win: Optional[float]
    average_loss: Optional[float]
    largest_win: Optional[float]
    largest_loss: Optional[float]
    average_holding_days: Optional[float]
    exposure_pct: float
    metrics_version: str = "1.2"
    # Appended after the pre-1.2 fields to preserve positional construction
    # compatibility for downstream consumers of this public dataclass.
    calmar_ratio: Optional[float] = None
    payoff_ratio: Optional[float] = None
    turnover_pct: Optional[float] = None
    max_drawdown_duration_days: int = 0
    max_drawdown_duration_bars: int = 0

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary of all metrics."""
        return {
            "total_return_pct": self.total_return_pct,
            "cagr_pct": self.cagr_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "win_rate_pct": self.win_rate_pct,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "number_of_trades": self.number_of_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "average_holding_days": self.average_holding_days,
            "exposure_pct": self.exposure_pct,
            "calmar_ratio": self.calmar_ratio,
            "payoff_ratio": self.payoff_ratio,
            "turnover_pct": self.turnover_pct,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "max_drawdown_duration_bars": self.max_drawdown_duration_bars,
            "metrics_version": self.metrics_version,
        }


# ---------------------------------------------------------------------------
# Top-level backtest result
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """The complete output of one BacktestEngine.run() call.

    Attributes
    ----------
    config:
        The configuration used for this run.
    start_date:
        First date in the input dataset.
    end_date:
        Last date in the input dataset.
    initial_capital:
        Cash at start of backtest (convenience alias for config.initial_capital).
    final_equity:
        Portfolio value at end of last bar (cash + mark-to-market).
    metrics:
        Aggregated performance metrics.
    trades:
        Ordered list of all completed trades.
    equity_curve:
        Daily portfolio equity snapshots (one per bar in the dataset).
    """

    config: BacktestConfig
    start_date: date
    end_date: date
    initial_capital: float
    final_equity: float
    metrics: BacktestMetrics
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a fully JSON-serialisable dictionary of the result.

        This is the payload that the LLM Performance Analyst will consume.
        """
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "final_equity": round(self.final_equity, 4),
            "metrics": self.metrics.to_dict(),
            "trades": [t.to_dict() for t in self.trades],
            "equity_curve": [ep.to_dict() for ep in self.equity_curve],
            "config": {
                "commission_pct": self.config.commission_pct,
                "slippage_pct": self.config.slippage_pct,
                "position_size_pct": self.config.position_size_pct,
                "maximum_holding_days": self.config.maximum_holding_days,
            },
        }
