"""Strict, deterministic contracts for Sprint 5 evaluation.

This module deliberately contains no agent, graph, provider, or execution
dependencies.  It records the provenance required to compare backtest
candidates safely, but it does not run a backtest or generate a strategy.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvaluationError(ValueError):
    """Base error for deterministic evaluation-policy violations."""


class EvaluationScopeError(EvaluationError):
    """Raised when an evaluation is used outside its declared data scope."""


class MixedEvaluationContextError(EvaluationError):
    """Raised when candidates do not share an exactly comparable context."""


class UndefinedMetricError(EvaluationError):
    """Raised when a required metric is undefined and cannot be scored safely."""


class EvaluationScope(str, Enum):
    """Chronological partition on which a candidate was evaluated."""

    VALIDATION = "VALIDATION"
    TEST = "TEST"
    FINAL_HOLDOUT = "FINAL_HOLDOUT"


class UndefinedMetricAction(str, Enum):
    """Explicit policy for a metric that is mathematically undefined."""

    REJECT = "REJECT"
    ALLOW = "ALLOW"


class ScoreMetric(str, Enum):
    """Metrics which may be gated or used as transparent score components."""

    TOTAL_RETURN_PCT = "total_return_pct"
    CAGR_PCT = "cagr_pct"
    MAX_DRAWDOWN_PCT = "max_drawdown_pct"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE_PCT = "win_rate_pct"
    EXPECTANCY = "expectancy"
    NUMBER_OF_TRADES = "number_of_trades"
    EXPOSURE_PCT = "exposure_pct"
    CALMAR_RATIO = "calmar_ratio"
    PAYOFF_RATIO = "payoff_ratio"
    TURNOVER_PCT = "turnover_pct"
    MAX_DRAWDOWN_DURATION_DAYS = "max_drawdown_duration_days"
    MAX_DRAWDOWN_DURATION_BARS = "max_drawdown_duration_bars"


# ``MetricName`` is kept as a readable alias for consumers which do not use
# the metric specifically for ranking.  It is the same enum, not a duplicate
# set of values that could drift from the scorer.
MetricName = ScoreMetric


class GateComparator(str, Enum):
    """The threshold comparison used by a viability gate."""

    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="


class GateStatus(str, Enum):
    """Auditable outcome of one configured viability gate."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    UNDEFINED = "UNDEFINED"


class ScoreDirection(str, Enum):
    """Whether a larger or smaller observed value is preferred."""

    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"


class _FrozenStrictModel(BaseModel):
    """Shared immutable boundary for externally supplied evaluation facts."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def _require_non_blank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty.")
    return value


def _require_finite(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number or None.")
    return value


class EvaluationContext(_FrozenStrictModel):
    """Immutable identity of the data, configuration, and metric semantics.

    All candidates in a ranking must share this full identity.  In particular,
    matching only a symbol is insufficient: a different historical slice,
    transaction-cost configuration, or metric version is a different
    experiment and cannot be ranked together.
    """

    scope: EvaluationScope
    dataset_id: str = Field(min_length=1, max_length=256)
    dataset_fingerprint: str = Field(min_length=1, max_length=256)
    backtest_config_fingerprint: str = Field(min_length=1, max_length=256)
    metrics_version: str = Field(min_length=1, max_length=64)

    @field_validator(
        "dataset_id",
        "dataset_fingerprint",
        "backtest_config_fingerprint",
        "metrics_version",
    )
    @classmethod
    def identifiers_must_not_be_blank(cls, value: str, info) -> str:
        return _require_non_blank(value, info.field_name)

    @property
    def comparability_key(self) -> tuple[str, str, str, str, str]:
        """Stable identity used to reject invalid cross-experiment rankings."""

        return (
            self.scope.value,
            self.dataset_id,
            self.dataset_fingerprint,
            self.backtest_config_fingerprint,
            self.metrics_version,
        )


class EvaluationMetrics(_FrozenStrictModel):
    """Finite metric snapshot with ``None`` reserved for undefined values.

    This mirrors the deterministic backtester without depending on its
    dataclass.  The distinction between ``None`` and ``0.0`` is preserved so
    gates and ranking can never quietly treat an undefined Sharpe, profit
    factor, or CAGR as a neutral performance result.
    """

    total_return_pct: float | None = None
    cagr_pct: float | None = None
    max_drawdown_pct: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    profit_factor: float | None = None
    win_rate_pct: float | None = None
    expectancy: float | None = None
    number_of_trades: int | None = None
    exposure_pct: float | None = None
    calmar_ratio: float | None = None
    payoff_ratio: float | None = None
    turnover_pct: float | None = None
    max_drawdown_duration_days: int | None = None
    max_drawdown_duration_bars: int | None = None

    @field_validator(
        "total_return_pct",
        "cagr_pct",
        "max_drawdown_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "profit_factor",
        "win_rate_pct",
        "expectancy",
        "exposure_pct",
        "calmar_ratio",
        "payoff_ratio",
        "turnover_pct",
    )
    @classmethod
    def optional_metrics_must_be_finite(cls, value: float | None, info) -> float | None:
        return _require_finite(value, info.field_name)

    @field_validator(
        "number_of_trades",
        "max_drawdown_duration_days",
        "max_drawdown_duration_bars",
    )
    @classmethod
    def count_metrics_must_be_non_negative(cls, value: int | None, info) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or value < 0:
            raise ValueError(f"{info.field_name} must be a non-negative integer or None.")
        return value

    @model_validator(mode="after")
    def bounded_metrics_must_have_valid_ranges(self) -> "EvaluationMetrics":
        if self.max_drawdown_pct is not None and self.max_drawdown_pct < 0:
            raise ValueError("max_drawdown_pct must be non-negative when defined.")
        if self.profit_factor is not None and self.profit_factor < 0:
            raise ValueError("profit_factor must be non-negative when defined.")
        if self.payoff_ratio is not None and self.payoff_ratio < 0:
            raise ValueError("payoff_ratio must be non-negative when defined.")
        if self.turnover_pct is not None and self.turnover_pct < 0:
            raise ValueError("turnover_pct must be non-negative when defined.")
        for name in ("win_rate_pct", "exposure_pct"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 100:
                raise ValueError(f"{name} must be in [0, 100] when defined.")
        return self

    def value_for(self, metric: ScoreMetric) -> float | int | None:
        """Return a metric without inventing a value for an undefined field."""

        return getattr(self, metric.value)

    @classmethod
    def from_backtest_metrics(cls, metrics: object) -> "EvaluationMetrics":
        """Copy a deterministic backtest metric artifact without importing its engine.

        This narrow adapter keeps the evaluation package free of execution
        dependencies while preserving every metric that can participate in a
        Sprint 5 gate or ranking policy.  Missing fields are a caller error;
        silently substituting values would make experiments incomparable.
        """

        fields = tuple(cls.model_fields)
        try:
            payload = {field: getattr(metrics, field) for field in fields}
        except AttributeError as exc:
            raise TypeError(
                "metrics must expose the complete BacktestMetrics-compatible public shape."
            ) from exc
        return cls(**payload)


class CandidateEvaluation(_FrozenStrictModel):
    """One immutable candidate's deterministic metric snapshot."""

    candidate_id: str = Field(min_length=1, max_length=256)
    context: EvaluationContext
    metrics: EvaluationMetrics
    strategy_hash: str | None = Field(default=None, max_length=256)

    @field_validator("candidate_id")
    @classmethod
    def candidate_id_must_not_be_blank(cls, value: str) -> str:
        return _require_non_blank(value, "candidate_id")

    @field_validator("strategy_hash")
    @classmethod
    def strategy_hash_must_not_be_blank_when_present(cls, value: str | None) -> str | None:
        if value is not None:
            return _require_non_blank(value, "strategy_hash")
        return value


class ViabilityPolicy(_FrozenStrictModel):
    """Explicit non-optimising acceptance thresholds for one evaluation scope."""

    policy_id: str = Field(default="default-viability-v1", min_length=1, max_length=128)
    required_scope: EvaluationScope = EvaluationScope.VALIDATION
    minimum_trade_count: int | None = Field(default=1, ge=0)
    minimum_total_return_pct: float | None = None
    minimum_sharpe_ratio: float | None = None
    minimum_sortino_ratio: float | None = None
    minimum_profit_factor: float | None = None
    minimum_calmar_ratio: float | None = None
    minimum_payoff_ratio: float | None = None
    maximum_drawdown_pct: float | None = Field(default=None, ge=0)
    maximum_turnover_pct: float | None = Field(default=None, ge=0)
    maximum_drawdown_duration_days: int | None = Field(default=None, ge=0)
    maximum_drawdown_duration_bars: int | None = Field(default=None, ge=0)
    undefined_metric_action: UndefinedMetricAction = UndefinedMetricAction.REJECT

    @field_validator("policy_id")
    @classmethod
    def policy_id_must_not_be_blank(cls, value: str) -> str:
        return _require_non_blank(value, "policy_id")

    @field_validator("minimum_trade_count", mode="before")
    @classmethod
    def policy_trade_count_must_not_be_boolean(cls, value: int | None) -> int | None:
        if isinstance(value, bool):
            raise ValueError("minimum_trade_count must not be a boolean.")
        return value

    @field_validator(
        "minimum_total_return_pct",
        "minimum_sharpe_ratio",
        "minimum_sortino_ratio",
        "minimum_profit_factor",
        "minimum_calmar_ratio",
        "minimum_payoff_ratio",
        "maximum_drawdown_pct",
        "maximum_turnover_pct",
    )
    @classmethod
    def policy_thresholds_must_be_finite(cls, value: float | None, info) -> float | None:
        return _require_finite(value, info.field_name)

    @field_validator(
        "maximum_drawdown_duration_days",
        "maximum_drawdown_duration_bars",
        mode="before",
    )
    @classmethod
    def policy_duration_thresholds_must_not_be_boolean(cls, value: int | None) -> int | None:
        if isinstance(value, bool):
            raise ValueError("Drawdown duration thresholds must not be boolean.")
        return value


class MetricGateResult(_FrozenStrictModel):
    """One threshold decision, including an explicit undefined state."""

    metric: ScoreMetric
    comparator: GateComparator
    threshold: float
    observed: float | None
    status: GateStatus
    detail: str = Field(default="", max_length=500)

    @field_validator("threshold", "observed")
    @classmethod
    def gate_numbers_must_be_finite(cls, value: float | None, info) -> float | None:
        return _require_finite(value, info.field_name)


class GateResult(_FrozenStrictModel):
    """Complete auditable viability outcome for one candidate."""

    candidate_id: str = Field(min_length=1, max_length=256)
    context: EvaluationContext
    policy_id: str = Field(min_length=1, max_length=128)
    is_viable: bool
    checks: tuple[MetricGateResult, ...] = ()
    failed_metrics: tuple[ScoreMetric, ...] = ()
    undefined_metrics: tuple[ScoreMetric, ...] = ()

    @field_validator("candidate_id", "policy_id")
    @classmethod
    def gate_identifiers_must_not_be_blank(cls, value: str, info) -> str:
        return _require_non_blank(value, info.field_name)


class ScoreComponentPolicy(_FrozenStrictModel):
    """One fixed, inspectable term in the ranking formula.

    ``contribution = weight * sign(direction) * observed / scale``.  The
    formula intentionally uses no peer-relative normalization, so adding or
    removing another candidate cannot change an existing candidate's score.
    """

    metric: ScoreMetric
    weight: float = Field(gt=0)
    direction: ScoreDirection
    scale: float = Field(default=1.0, gt=0)

    @field_validator("weight", "scale")
    @classmethod
    def score_policy_numbers_must_be_finite(cls, value: float, info) -> float:
        result = _require_finite(value, info.field_name)
        assert result is not None
        return result


def _default_score_components() -> tuple[ScoreComponentPolicy, ...]:
    """Conservative default: reward return/Sharpe and penalise drawdown."""

    return (
        ScoreComponentPolicy(
            metric=ScoreMetric.TOTAL_RETURN_PCT,
            weight=1.0,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            scale=100.0,
        ),
        ScoreComponentPolicy(
            metric=ScoreMetric.SHARPE_RATIO,
            weight=1.0,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            scale=1.0,
        ),
        ScoreComponentPolicy(
            metric=ScoreMetric.MAX_DRAWDOWN_PCT,
            weight=0.5,
            direction=ScoreDirection.LOWER_IS_BETTER,
            scale=100.0,
        ),
    )


class RankingPolicy(_FrozenStrictModel):
    """Immutable formula and deterministic tie-break rule for comparable runs."""

    policy_id: str = Field(default="default-ranking-v1", min_length=1, max_length=128)
    required_scope: EvaluationScope = EvaluationScope.VALIDATION
    components: tuple[ScoreComponentPolicy, ...] = Field(default_factory=_default_score_components)
    tie_breakers: tuple[ScoreMetric, ...] = (
        ScoreMetric.SHARPE_RATIO,
        ScoreMetric.TOTAL_RETURN_PCT,
        ScoreMetric.MAX_DRAWDOWN_PCT,
    )

    @field_validator("policy_id")
    @classmethod
    def ranking_policy_id_must_not_be_blank(cls, value: str) -> str:
        return _require_non_blank(value, "policy_id")

    @model_validator(mode="after")
    def ranking_formula_must_be_unambiguous(self) -> "RankingPolicy":
        if not self.components:
            raise ValueError("RankingPolicy requires at least one score component.")
        component_metrics = [component.metric for component in self.components]
        if len(component_metrics) != len(set(component_metrics)):
            raise ValueError("RankingPolicy components must use unique metrics.")
        if len(self.tie_breakers) != len(set(self.tie_breakers)):
            raise ValueError("RankingPolicy tie_breakers must be unique.")
        return self

    def direction_for(self, metric: ScoreMetric) -> ScoreDirection:
        """Resolve tie-break direction from a configured component or metric semantics."""

        for component in self.components:
            if component.metric is metric:
                return component.direction
        if metric in {
            ScoreMetric.MAX_DRAWDOWN_PCT,
            ScoreMetric.TURNOVER_PCT,
            ScoreMetric.MAX_DRAWDOWN_DURATION_DAYS,
            ScoreMetric.MAX_DRAWDOWN_DURATION_BARS,
        }:
            return ScoreDirection.LOWER_IS_BETTER
        return ScoreDirection.HIGHER_IS_BETTER


class ScoreComponent(_FrozenStrictModel):
    """Auditable numeric contribution for one candidate and one score term."""

    metric: ScoreMetric
    raw_value: float
    normalized_value: float
    weight: float
    direction: ScoreDirection
    scale: float
    contribution: float

    @field_validator(
        "raw_value",
        "normalized_value",
        "weight",
        "scale",
        "contribution",
    )
    @classmethod
    def score_numbers_must_be_finite(cls, value: float, info) -> float:
        result = _require_finite(value, info.field_name)
        assert result is not None
        return result


class RankedCandidate(_FrozenStrictModel):
    """One candidate after deterministic scoring and stable tie-breaking."""

    rank: int = Field(ge=1)
    candidate: CandidateEvaluation
    total_score: float
    components: tuple[ScoreComponent, ...]

    @field_validator("total_score")
    @classmethod
    def total_score_must_be_finite(cls, value: float) -> float:
        result = _require_finite(value, "total_score")
        assert result is not None
        return result


class RankingResult(_FrozenStrictModel):
    """Fully reproducible ranking for candidates sharing one context."""

    context: EvaluationContext
    policy_id: str = Field(min_length=1, max_length=128)
    ranked_candidates: tuple[RankedCandidate, ...]
    excluded_candidate_ids: tuple[str, ...] = ()

    @field_validator("policy_id")
    @classmethod
    def result_policy_id_must_not_be_blank(cls, value: str) -> str:
        return _require_non_blank(value, "policy_id")

    @property
    def winner(self) -> RankedCandidate | None:
        """Return the stable winner, or ``None`` if no candidates were ranked."""

        return self.ranked_candidates[0] if self.ranked_candidates else None
