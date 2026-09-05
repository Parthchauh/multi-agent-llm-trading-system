"""Pure deterministic in-sample refinement acceptance for Sprint 6.

No agent prose or raw model output appears in the decision.  This policy is a
deliberately conservative validation-only guard until Sprint 7 can introduce
chronologically protected out-of-sample evidence.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evaluation.models import (
    CandidateEvaluation,
    EvaluationContext,
    EvaluationScope,
    GateResult,
    ScoreMetric,
)
from evaluation.research_metrics import StrategyComplexity, measure_strategy_complexity
from tradingagents.research.lineage import StrategyVersionRecord


class _FrozenStrictModel(BaseModel):
    """Immutable deterministic acceptance contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class RefinementQualityLabel(str, Enum):
    """Scope label that prevents in-sample acceptance from implying generalization."""

    IN_SAMPLE_ONLY = "IN_SAMPLE_ONLY"


class RefinementAcceptanceStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class RefinementRejectionReason(str, Enum):
    """Structured reasons retained even when an LLM recommends acceptance."""

    NOT_DIRECT_CHILD = "NOT_DIRECT_CHILD"
    NON_VALIDATION_SCOPE = "NON_VALIDATION_SCOPE"
    EVALUATION_CONTEXT_MISMATCH = "EVALUATION_CONTEXT_MISMATCH"
    PARENT_CANDIDATE_MISMATCH = "PARENT_CANDIDATE_MISMATCH"
    CHILD_CANDIDATE_MISMATCH = "CHILD_CANDIDATE_MISMATCH"
    STRATEGY_FINGERPRINT_MISMATCH = "STRATEGY_FINGERPRINT_MISMATCH"
    CHILD_GATE_CANDIDATE_MISMATCH = "CHILD_GATE_CANDIDATE_MISMATCH"
    CHILD_GATE_CONTEXT_MISMATCH = "CHILD_GATE_CONTEXT_MISMATCH"
    CHILD_NOT_VIABLE = "CHILD_NOT_VIABLE"
    IMPROVEMENT_METRIC_UNDEFINED = "IMPROVEMENT_METRIC_UNDEFINED"
    INSUFFICIENT_VALIDATION_IMPROVEMENT = "INSUFFICIENT_VALIDATION_IMPROVEMENT"
    COMPLEXITY_INCREASE_EXCEEDED = "COMPLEXITY_INCREASE_EXCEEDED"
    DRAWDOWN_UNDEFINED = "DRAWDOWN_UNDEFINED"
    DRAWDOWN_DEGRADATION_EXCEEDED = "DRAWDOWN_DEGRADATION_EXCEEDED"


# CAGR and raw total return are intentionally absent.  A refinement cannot be
# accepted merely because it found a larger in-sample return number.
_SAFE_IMPROVEMENT_METRICS = frozenset(
    {
        ScoreMetric.SHARPE_RATIO,
        ScoreMetric.SORTINO_RATIO,
        ScoreMetric.CALMAR_RATIO,
        ScoreMetric.PROFIT_FACTOR,
        ScoreMetric.PAYOFF_RATIO,
        ScoreMetric.EXPECTANCY,
    }
)


class RefinementAcceptancePolicy(_FrozenStrictModel):
    """Explicit validation-only controls for a proposed child strategy.

    Complexity is measured by the transparent structural score returned by
    :func:`complexity_score`: condition count + unique indicators + maximum
    nested-group depth + numeric parameters.  It is a safety cap, not a claim
    that simpler strategies are economically superior.
    """

    policy_id: str = Field(default="sprint6-refinement-v1", min_length=1, max_length=128)
    improvement_metric: ScoreMetric = ScoreMetric.SHARPE_RATIO
    allowed_improvement_metrics: tuple[ScoreMetric, ...] = tuple(
        sorted(_SAFE_IMPROVEMENT_METRICS, key=lambda metric: metric.value)
    )
    minimum_improvement: float = Field(default=0.0001, gt=0)
    max_complexity_increase: int = Field(default=2, ge=0, le=10_000)
    max_drawdown_degradation_pct: float = Field(default=1.0, ge=0)

    @field_validator("policy_id")
    @classmethod
    def policy_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("policy_id must be non-empty.")
        return value

    @field_validator("minimum_improvement", "max_drawdown_degradation_pct")
    @classmethod
    def policy_floats_must_be_finite(cls, value: float, info) -> float:
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be finite and must not be boolean.")
        return value

    @field_validator("max_complexity_increase", mode="before")
    @classmethod
    def complexity_limit_must_not_be_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("max_complexity_increase must be an integer, not a boolean.")
        return value

    @model_validator(mode="after")
    def improvement_metric_must_be_explicitly_safe(self) -> "RefinementAcceptancePolicy":
        if not self.allowed_improvement_metrics:
            raise ValueError("allowed_improvement_metrics must contain at least one metric.")
        if len(set(self.allowed_improvement_metrics)) != len(self.allowed_improvement_metrics):
            raise ValueError("allowed_improvement_metrics must not contain duplicates.")
        unsafe = set(self.allowed_improvement_metrics) - _SAFE_IMPROVEMENT_METRICS
        if unsafe:
            raise ValueError(
                "Raw return and CAGR metrics cannot be used as refinement acceptance metrics."
            )
        if self.improvement_metric not in self.allowed_improvement_metrics:
            raise ValueError("improvement_metric must be present in allowed_improvement_metrics.")
        return self


class RefinementAcceptanceResult(_FrozenStrictModel):
    """Full deterministic decision record for one parent-to-child comparison."""

    status: RefinementAcceptanceStatus
    quality_label: RefinementQualityLabel = RefinementQualityLabel.IN_SAMPLE_ONLY
    policy_id: str = Field(min_length=1, max_length=128)
    context: EvaluationContext
    parent_strategy_id: str = Field(min_length=1, max_length=128)
    child_strategy_id: str = Field(min_length=1, max_length=128)
    improvement_metric: ScoreMetric
    parent_metric_value: float | None
    child_metric_value: float | None
    metric_improvement: float | None
    parent_complexity: StrategyComplexity
    child_complexity: StrategyComplexity
    complexity_delta: int
    parent_drawdown_pct: float | None
    child_drawdown_pct: float | None
    drawdown_degradation_pct: float | None
    child_is_viable: bool
    rejection_reasons: tuple[RefinementRejectionReason, ...] = ()

    @field_validator(
        "parent_metric_value",
        "child_metric_value",
        "metric_improvement",
        "parent_drawdown_pct",
        "child_drawdown_pct",
        "drawdown_degradation_pct",
    )
    @classmethod
    def optional_numbers_must_be_finite(cls, value: float | None, info) -> float | None:
        if value is None:
            return value
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be finite or None.")
        return value

    @model_validator(mode="after")
    def status_must_agree_with_reasons(self) -> "RefinementAcceptanceResult":
        if self.status is RefinementAcceptanceStatus.ACCEPTED and self.rejection_reasons:
            raise ValueError("An accepted refinement cannot include rejection reasons.")
        if self.status is RefinementAcceptanceStatus.REJECTED and not self.rejection_reasons:
            raise ValueError("A rejected refinement must include at least one reason.")
        return self

    @property
    def accepted(self) -> bool:
        """Convenience predicate; the immutable status remains authoritative."""

        return self.status is RefinementAcceptanceStatus.ACCEPTED


def complexity_score(complexity: StrategyComplexity) -> int:
    """Return the documented structural score used for the acceptance cap."""

    return (
        complexity.condition_count
        + complexity.unique_indicator_count
        + complexity.maximum_group_depth
        + complexity.numeric_parameter_count
    )


def assess_refinement_acceptance(
    *,
    parent: StrategyVersionRecord,
    child: StrategyVersionRecord,
    parent_evaluation: CandidateEvaluation,
    child_evaluation: CandidateEvaluation,
    child_gate: GateResult,
    policy: RefinementAcceptancePolicy | None = None,
) -> RefinementAcceptanceResult:
    """Deterministically accept or reject one immutable refinement child.

    This function is intentionally side-effect free.  It accepts no LLM
    recommendation and always labels a successful decision ``IN_SAMPLE_ONLY``.
    It cannot use test or final-holdout metrics because both parent and child
    evaluation contexts must be the validation partition.
    """

    effective_policy = policy or RefinementAcceptancePolicy()
    reasons: list[RefinementRejectionReason] = []

    if (
        child.lineage.parent_strategy_id != parent.strategy_id
        or child.lineage.generation != parent.lineage.generation + 1
        or child.strategy_id == parent.strategy_id
    ):
        reasons.append(RefinementRejectionReason.NOT_DIRECT_CHILD)

    parent_context = parent_evaluation.context
    child_context = child_evaluation.context
    if (
        parent_context.scope is not EvaluationScope.VALIDATION
        or child_context.scope is not EvaluationScope.VALIDATION
    ):
        reasons.append(RefinementRejectionReason.NON_VALIDATION_SCOPE)
    if parent_context.comparability_key != child_context.comparability_key:
        reasons.append(RefinementRejectionReason.EVALUATION_CONTEXT_MISMATCH)

    if parent_evaluation.candidate_id != parent.strategy_id:
        reasons.append(RefinementRejectionReason.PARENT_CANDIDATE_MISMATCH)
    if child_evaluation.candidate_id != child.strategy_id:
        reasons.append(RefinementRejectionReason.CHILD_CANDIDATE_MISMATCH)
    if not _matches_optional_fingerprint(parent_evaluation, parent):
        reasons.append(RefinementRejectionReason.STRATEGY_FINGERPRINT_MISMATCH)
    if not _matches_optional_fingerprint(child_evaluation, child):
        reasons.append(RefinementRejectionReason.STRATEGY_FINGERPRINT_MISMATCH)

    if child_gate.candidate_id != child.strategy_id:
        reasons.append(RefinementRejectionReason.CHILD_GATE_CANDIDATE_MISMATCH)
    if child_gate.context.comparability_key != child_context.comparability_key:
        reasons.append(RefinementRejectionReason.CHILD_GATE_CONTEXT_MISMATCH)
    if not child_gate.is_viable:
        reasons.append(RefinementRejectionReason.CHILD_NOT_VIABLE)

    parent_value = _metric_value(parent_evaluation, effective_policy.improvement_metric)
    child_value = _metric_value(child_evaluation, effective_policy.improvement_metric)
    metric_improvement = (
        child_value - parent_value
        if parent_value is not None and child_value is not None
        else None
    )
    if metric_improvement is None:
        reasons.append(RefinementRejectionReason.IMPROVEMENT_METRIC_UNDEFINED)
    elif metric_improvement < effective_policy.minimum_improvement:
        reasons.append(RefinementRejectionReason.INSUFFICIENT_VALIDATION_IMPROVEMENT)

    parent_complexity = measure_strategy_complexity(parent.strategy)
    child_complexity = measure_strategy_complexity(child.strategy)
    complexity_delta = complexity_score(child_complexity) - complexity_score(parent_complexity)
    if complexity_delta > effective_policy.max_complexity_increase:
        reasons.append(RefinementRejectionReason.COMPLEXITY_INCREASE_EXCEEDED)

    parent_drawdown = _metric_value(parent_evaluation, ScoreMetric.MAX_DRAWDOWN_PCT)
    child_drawdown = _metric_value(child_evaluation, ScoreMetric.MAX_DRAWDOWN_PCT)
    drawdown_degradation = (
        child_drawdown - parent_drawdown
        if parent_drawdown is not None and child_drawdown is not None
        else None
    )
    if drawdown_degradation is None:
        reasons.append(RefinementRejectionReason.DRAWDOWN_UNDEFINED)
    elif drawdown_degradation > effective_policy.max_drawdown_degradation_pct:
        reasons.append(RefinementRejectionReason.DRAWDOWN_DEGRADATION_EXCEEDED)

    # Preserve first-occurrence order for compact, deterministic audit output.
    unique_reasons = tuple(dict.fromkeys(reasons))
    return RefinementAcceptanceResult(
        status=(
            RefinementAcceptanceStatus.ACCEPTED
            if not unique_reasons
            else RefinementAcceptanceStatus.REJECTED
        ),
        policy_id=effective_policy.policy_id,
        context=child_context,
        parent_strategy_id=parent.strategy_id,
        child_strategy_id=child.strategy_id,
        improvement_metric=effective_policy.improvement_metric,
        parent_metric_value=parent_value,
        child_metric_value=child_value,
        metric_improvement=metric_improvement,
        parent_complexity=parent_complexity,
        child_complexity=child_complexity,
        complexity_delta=complexity_delta,
        parent_drawdown_pct=parent_drawdown,
        child_drawdown_pct=child_drawdown,
        drawdown_degradation_pct=drawdown_degradation,
        child_is_viable=child_gate.is_viable,
        rejection_reasons=unique_reasons,
    )


def _metric_value(candidate: CandidateEvaluation, metric: ScoreMetric) -> float | None:
    value = candidate.metrics.value_for(metric)
    return None if value is None else float(value)


def _matches_optional_fingerprint(
    candidate: CandidateEvaluation,
    record: StrategyVersionRecord,
) -> bool:
    """Validate supplied semantic hashes without requiring redundant storage."""

    return (
        candidate.strategy_hash is None
        or candidate.strategy_hash == record.lineage.execution_fingerprint
    )


# A compact alias for orchestration nodes; it retains the complete auditable
# result rather than returning a lossy Boolean.
evaluate_refinement_acceptance = assess_refinement_acceptance
