"""Strict immutable contracts for deterministic Sprint 6 risk controls.

The models in this module are intentionally independent of LLMs, LangGraph,
and market-data providers.  They record only facts supplied by the validated
backtest and deterministic sizing helpers.  A policy deliberately does *not*
include a liquidity rule: the current OHLCV/backtester contracts do not expose
an execution-quality or market-depth measure suitable for enforcing one.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _FrozenStrictModel(BaseModel):
    """Shared external-data boundary for risk artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def _finite_number(value: float | int | None, field_name: str) -> float | int | None:
    """Reject bool/NaN/infinity instead of quietly accepting unsafe policy data."""

    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number or None.")
    return value


class RiskRule(str, Enum):
    """Deterministic rules enforceable with current backtest artifacts."""

    MAX_DRAWDOWN_PCT = "MAX_DRAWDOWN_PCT"
    MAX_POSITION_PCT = "MAX_POSITION_PCT"
    MAX_PORTFOLIO_EXPOSURE_PCT = "MAX_PORTFOLIO_EXPOSURE_PCT"
    MAX_LOSS_PER_TRADE_PCT = "MAX_LOSS_PER_TRADE_PCT"
    MAX_TURNOVER_PCT = "MAX_TURNOVER_PCT"
    MINIMUM_TRADE_COUNT = "MINIMUM_TRADE_COUNT"
    MAX_CONCURRENT_POSITIONS = "MAX_CONCURRENT_POSITIONS"


class ViolationSeverity(str, Enum):
    """Severity is structured so orchestration never infers it from prose."""

    ERROR = "ERROR"
    WARNING = "WARNING"


class RiskPolicy(_FrozenStrictModel):
    """Immutable deterministic limits for one long-only backtest.

    All configured percentages are expressed in percentage points, so ``5.0``
    means five percent.  Every optional threshold is disabled when ``None``.

    ``max_portfolio_exposure_pct`` is evaluated as a single-position limit.
    This is correct for the current engine because it cannot hold more than
    one long position.  The model rejects ``max_concurrent_positions > 1``
    rather than pretending multi-position portfolio control exists.

    Liquidity is intentionally not a field.  Current OHLCV data does not
    establish executable depth, spread, or participation-rate capacity.
    ``extra='forbid'`` means an attempted ``minimum_liquidity`` field is
    rejected instead of being silently ignored.
    """

    policy_id: str = Field(default="risk-policy-v1", min_length=1, max_length=128)
    max_drawdown_pct: float | None = None
    max_position_pct: float | None = None
    max_portfolio_exposure_pct: float | None = None
    max_loss_per_trade_pct: float | None = None
    max_turnover_pct: float | None = None
    minimum_trade_count: int | None = None
    max_concurrent_positions: int = 1

    @field_validator("policy_id")
    @classmethod
    def policy_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("policy_id must be non-empty.")
        return value

    @field_validator(
        "max_drawdown_pct",
        "max_position_pct",
        "max_portfolio_exposure_pct",
        "max_loss_per_trade_pct",
        "max_turnover_pct",
        mode="before",
    )
    @classmethod
    def thresholds_must_be_finite_non_negative(
        cls,
        value: float | None,
        info,
    ) -> float | None:
        checked = _finite_number(value, info.field_name)
        if checked is not None and float(checked) < 0.0:
            raise ValueError(f"{info.field_name} must be non-negative when configured.")
        return checked  # type: ignore[return-value]

    @field_validator("minimum_trade_count", "max_concurrent_positions", mode="before")
    @classmethod
    def integer_policy_values_must_not_be_boolean(
        cls,
        value: int | None,
        info,
    ) -> int | None:
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be an integer, not a boolean.")
        return value

    @field_validator("minimum_trade_count")
    @classmethod
    def minimum_trade_count_must_be_non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("minimum_trade_count must be non-negative when configured.")
        return value

    @field_validator("max_concurrent_positions")
    @classmethod
    def concurrency_must_match_current_long_only_engine(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_concurrent_positions must be at least 1.")
        if value > 1:
            raise ValueError(
                "max_concurrent_positions > 1 is unsupported: the current "
                "long-only backtester has a single open-position model."
            )
        return value

    @model_validator(mode="after")
    def position_limits_must_not_claim_leverage_support(self) -> "RiskPolicy":
        for field_name in ("max_position_pct", "max_portfolio_exposure_pct"):
            value = getattr(self, field_name)
            if value is not None and value > 100.0:
                raise ValueError(
                    f"{field_name} cannot exceed 100 for the non-leveraged current engine."
                )
        return self


class EffectivePositionSize(_FrozenStrictModel):
    """Auditable evidence used to apply single-position allocation limits.

    ``configured_position_size_pct`` comes from the immutable
    :class:`backtesting.models.BacktestConfig`.  The completed-trade figure is
    measured against initial capital and captures cash growth or rounding.  A
    conservative effective value is the larger known value; it is never
    inferred from LLM output or a future market bar.
    """

    configured_position_size_pct: float | None = None
    max_entry_notional_pct_initial_capital: float | None = None
    effective_position_size_pct: float | None = None

    @field_validator(
        "configured_position_size_pct",
        "max_entry_notional_pct_initial_capital",
        "effective_position_size_pct",
    )
    @classmethod
    def values_must_be_non_negative_finite(
        cls,
        value: float | None,
        info,
    ) -> float | None:
        checked = _finite_number(value, info.field_name)
        if checked is not None and float(checked) < 0.0:
            raise ValueError(f"{info.field_name} must be non-negative when defined.")
        return checked  # type: ignore[return-value]

    @model_validator(mode="after")
    def effective_size_must_not_understate_known_values(self) -> "EffectivePositionSize":
        known = [
            value
            for value in (
                self.configured_position_size_pct,
                self.max_entry_notional_pct_initial_capital,
            )
            if value is not None
        ]
        if known and self.effective_position_size_pct is None:
            raise ValueError("effective_position_size_pct is required when size evidence is available.")
        if known and self.effective_position_size_pct is not None:
            if self.effective_position_size_pct + 1e-12 < max(known):
                raise ValueError("effective_position_size_pct must not understate known size evidence.")
        return self


class RiskMetrics(_FrozenStrictModel):
    """Execution-derived metrics that can be checked against a RiskPolicy."""

    initial_capital: float = Field(gt=0)
    max_drawdown_pct: float = Field(ge=0)
    position_size: EffectivePositionSize
    max_loss_per_trade_pct: float = Field(ge=0)
    turnover_pct: float | None = None
    trade_count: int = Field(ge=0)
    max_concurrent_positions: int = Field(ge=0)

    @field_validator(
        "initial_capital",
        "max_drawdown_pct",
        "max_loss_per_trade_pct",
        "turnover_pct",
    )
    @classmethod
    def numeric_metrics_must_be_finite(
        cls,
        value: float | None,
        info,
    ) -> float | None:
        checked = _finite_number(value, info.field_name)
        if checked is not None and float(checked) < 0.0:
            raise ValueError(f"{info.field_name} must be non-negative when defined.")
        return checked  # type: ignore[return-value]

    @field_validator("trade_count", "max_concurrent_positions", mode="before")
    @classmethod
    def count_metrics_must_not_be_boolean(cls, value: int, info) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be an integer, not a boolean.")
        return value


class RiskViolation(_FrozenStrictModel):
    """One deterministic policy failure with observed fact and configured limit."""

    rule: RiskRule
    observed_value: float | int | None
    limit: float | int | None
    severity: ViolationSeverity = ViolationSeverity.ERROR
    detail: str = Field(default="", max_length=500)

    @field_validator("observed_value", "limit")
    @classmethod
    def violation_values_must_be_finite(
        cls,
        value: float | int | None,
        info,
    ) -> float | int | None:
        return _finite_number(value, info.field_name)


class RiskAssessment(_FrozenStrictModel):
    """Immutable, structured risk decision owned by deterministic code.

    ``passed`` is constrained to equal ``not violations``.  An LLM-facing
    Risk Manager can add interpretation elsewhere, but cannot construct an
    inconsistent assessment that turns a deterministic violation into a pass.
    """

    passed: bool
    violations: tuple[RiskViolation, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: RiskMetrics

    @field_validator("warnings")
    @classmethod
    def warning_text_must_not_be_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("warnings must not contain blank text.")
        return value

    @model_validator(mode="after")
    def pass_flag_must_match_violations(self) -> "RiskAssessment":
        if self.passed != (not self.violations):
            raise ValueError("RiskAssessment.passed must equal whether violations are empty.")
        return self
