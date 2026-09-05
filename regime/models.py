"""Frozen Pydantic contracts for market-regime analysis."""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrendState(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class VolatilityState(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class MomentumState(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class CompositeRegime(str, Enum):
    BULLISH_TREND = "bullish_trend"
    BEARISH_TREND = "bearish_trend"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MIXED = "mixed"


class AssessmentSource(str, Enum):
    LLM = "LLM"
    DETERMINISTIC_FALLBACK = "DETERMINISTIC_FALLBACK"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RegimeFeatures(_FrozenModel):
    """Compact authoritative feature snapshot supplied to regime analysis."""

    symbol: str = Field(min_length=1, max_length=32)
    timestamp: datetime
    observations: int = Field(ge=1)
    close: float = Field(gt=0)
    close_vs_sma20_pct: float | None = None
    close_vs_sma50_pct: float | None = None
    close_vs_sma200_pct: float | None = None
    sma20_slope_pct: float | None = None
    sma50_slope_pct: float | None = None
    sma50_vs_sma200_pct: float | None = None
    return_20d_pct: float | None = None
    return_60d_pct: float | None = None
    realized_volatility_20d_pct: float | None = Field(default=None, ge=0)
    atr_14_pct: float | None = Field(default=None, ge=0)
    distance_from_high_20d_pct: float | None = Field(default=None, le=0)
    recent_max_drawdown_60d_pct: float | None = Field(default=None, le=0)
    volume_trend_20d_pct: float | None = None

    @field_validator(
        "close",
        "close_vs_sma20_pct",
        "close_vs_sma50_pct",
        "close_vs_sma200_pct",
        "sma20_slope_pct",
        "sma50_slope_pct",
        "sma50_vs_sma200_pct",
        "return_20d_pct",
        "return_60d_pct",
        "realized_volatility_20d_pct",
        "atr_14_pct",
        "distance_from_high_20d_pct",
        "recent_max_drawdown_60d_pct",
        "volume_trend_20d_pct",
        mode="before",
    )
    @classmethod
    def values_must_not_be_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Regime feature values must be numeric, not boolean.")
        return value

    @field_validator(
        "close",
        "close_vs_sma20_pct",
        "close_vs_sma50_pct",
        "close_vs_sma200_pct",
        "sma20_slope_pct",
        "sma50_slope_pct",
        "sma50_vs_sma200_pct",
        "return_20d_pct",
        "return_60d_pct",
        "realized_volatility_20d_pct",
        "atr_14_pct",
        "distance_from_high_20d_pct",
        "recent_max_drawdown_60d_pct",
        "volume_trend_20d_pct",
    )
    @classmethod
    def values_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Regime feature values must be finite or None.")
        return value


class RegimeAssessmentDraft(_FrozenModel):
    """Decision fields an LLM may propose; source is assigned by the system."""

    regime: CompositeRegime
    trend_state: TrendState
    volatility_state: VolatilityState
    momentum_state: MomentumState
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=2000)
    preferred_strategy_characteristics: tuple[str, ...] = Field(
        default_factory=tuple, max_length=6
    )
    avoid_strategy_characteristics: tuple[str, ...] = Field(
        default_factory=tuple, max_length=6
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def confidence_must_not_be_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Regime confidence must be numeric, not boolean.")
        return value


class RegimeAssessment(RegimeAssessmentDraft):
    """Validated regime assessment with an authoritative provenance marker."""

    source: AssessmentSource


class RegimeAgentOutcome(_FrozenModel):
    """Regime-agent result that records, rather than hides, fallback failures."""

    assessment: RegimeAssessment
    fallback_reason: str | None = Field(default=None, max_length=1000)
