"""Deterministic viability gates for Sprint 5 candidate evaluation."""

from __future__ import annotations

from typing import Iterable

from evaluation.models import (
    CandidateEvaluation,
    EvaluationScopeError,
    GateComparator,
    GateResult,
    GateStatus,
    MetricGateResult,
    ScoreMetric,
    UndefinedMetricAction,
    ViabilityPolicy,
)


def _configured_thresholds(
    policy: ViabilityPolicy,
) -> Iterable[tuple[ScoreMetric, GateComparator, float]]:
    """Yield configured gates in a stable, audit-friendly order."""

    rules: tuple[tuple[ScoreMetric, GateComparator, float | int | None], ...] = (
        (
            ScoreMetric.NUMBER_OF_TRADES,
            GateComparator.GREATER_THAN_OR_EQUAL,
            policy.minimum_trade_count,
        ),
        (
            ScoreMetric.TOTAL_RETURN_PCT,
            GateComparator.GREATER_THAN_OR_EQUAL,
            policy.minimum_total_return_pct,
        ),
        (
            ScoreMetric.SHARPE_RATIO,
            GateComparator.GREATER_THAN_OR_EQUAL,
            policy.minimum_sharpe_ratio,
        ),
        (
            ScoreMetric.SORTINO_RATIO,
            GateComparator.GREATER_THAN_OR_EQUAL,
            policy.minimum_sortino_ratio,
        ),
        (
            ScoreMetric.PROFIT_FACTOR,
            GateComparator.GREATER_THAN_OR_EQUAL,
            policy.minimum_profit_factor,
        ),
        (
            ScoreMetric.CALMAR_RATIO,
            GateComparator.GREATER_THAN_OR_EQUAL,
            policy.minimum_calmar_ratio,
        ),
        (
            ScoreMetric.PAYOFF_RATIO,
            GateComparator.GREATER_THAN_OR_EQUAL,
            policy.minimum_payoff_ratio,
        ),
        (
            ScoreMetric.MAX_DRAWDOWN_PCT,
            GateComparator.LESS_THAN_OR_EQUAL,
            policy.maximum_drawdown_pct,
        ),
        (
            ScoreMetric.TURNOVER_PCT,
            GateComparator.LESS_THAN_OR_EQUAL,
            policy.maximum_turnover_pct,
        ),
        (
            ScoreMetric.MAX_DRAWDOWN_DURATION_DAYS,
            GateComparator.LESS_THAN_OR_EQUAL,
            policy.maximum_drawdown_duration_days,
        ),
        (
            ScoreMetric.MAX_DRAWDOWN_DURATION_BARS,
            GateComparator.LESS_THAN_OR_EQUAL,
            policy.maximum_drawdown_duration_bars,
        ),
    )
    for metric, comparator, threshold in rules:
        if threshold is not None:
            yield metric, comparator, float(threshold)


def evaluate_viability(
    candidate: CandidateEvaluation,
    policy: ViabilityPolicy,
) -> GateResult:
    """Apply policy thresholds without inventing values for undefined metrics.

    A policy can explicitly allow an undefined metric, but the result still
    records it in ``undefined_metrics`` and its individual check is marked
    ``UNDEFINED``.  The default policy rejects it.  Scope misuse is a caller
    error rather than an ordinary losing score, and therefore raises before a
    candidate can be accidentally selected from the wrong partition.
    """

    if candidate.context.scope is not policy.required_scope:
        raise EvaluationScopeError(
            "Candidate evaluation scope "
            f"{candidate.context.scope.value!r} does not match viability policy scope "
            f"{policy.required_scope.value!r}."
        )

    checks: list[MetricGateResult] = []
    failed_metrics: list[ScoreMetric] = []
    undefined_metrics: list[ScoreMetric] = []

    for metric, comparator, threshold in _configured_thresholds(policy):
        observed = candidate.metrics.value_for(metric)
        if observed is None:
            undefined_metrics.append(metric)
            checks.append(
                MetricGateResult(
                    metric=metric,
                    comparator=comparator,
                    threshold=threshold,
                    observed=None,
                    status=GateStatus.UNDEFINED,
                    detail="Metric is undefined; no numeric substitute was applied.",
                )
            )
            continue

        observed_float = float(observed)
        passed = (
            observed_float >= threshold
            if comparator is GateComparator.GREATER_THAN_OR_EQUAL
            else observed_float <= threshold
        )
        checks.append(
            MetricGateResult(
                metric=metric,
                comparator=comparator,
                threshold=threshold,
                observed=observed_float,
                status=GateStatus.PASSED if passed else GateStatus.FAILED,
                detail="Threshold satisfied." if passed else "Threshold not satisfied.",
            )
        )
        if not passed:
            failed_metrics.append(metric)

    undefined_rejected = (
        bool(undefined_metrics)
        and policy.undefined_metric_action is UndefinedMetricAction.REJECT
    )
    return GateResult(
        candidate_id=candidate.candidate_id,
        context=candidate.context,
        policy_id=policy.policy_id,
        is_viable=not failed_metrics and not undefined_rejected,
        checks=tuple(checks),
        failed_metrics=tuple(failed_metrics),
        undefined_metrics=tuple(undefined_metrics),
    )


# Readable aliases for orchestration code.  They preserve the one canonical
# implementation and do not introduce a second, potentially divergent policy
# path.
apply_viability_gates = evaluate_viability
evaluate_candidate = evaluate_viability
