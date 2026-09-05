"""Transparent, deterministic ranking for comparable Sprint 5 candidates."""

from __future__ import annotations

import math
from collections.abc import Iterable

from evaluation.models import (
    CandidateEvaluation,
    EvaluationError,
    EvaluationScopeError,
    GateResult,
    MixedEvaluationContextError,
    RankedCandidate,
    RankingPolicy,
    RankingResult,
    ScoreComponent,
    ScoreDirection,
    UndefinedMetricError,
)


def _validate_comparable_contexts(
    candidates: tuple[CandidateEvaluation, ...],
    policy: RankingPolicy,
) -> None:
    """Reject every form of mixed experiment context before scoring."""

    if not candidates:
        raise EvaluationError("At least one candidate is required for ranking.")

    reference = candidates[0].context
    for candidate in candidates[1:]:
        if candidate.context.comparability_key != reference.comparability_key:
            raise MixedEvaluationContextError(
                "Candidates cannot be ranked across different evaluation scope, dataset, "
                "configuration, or metric-version identities."
            )
    if reference.scope is not policy.required_scope:
        raise EvaluationScopeError(
            "Ranking policy requires scope "
            f"{policy.required_scope.value!r}, but candidates were evaluated on "
            f"{reference.scope.value!r}."
        )

    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise EvaluationError("Candidate IDs must be unique within one ranking.")


def score_candidate(
    candidate: CandidateEvaluation,
    policy: RankingPolicy,
) -> tuple[float, tuple[ScoreComponent, ...]]:
    """Return the fixed-formula score and every numeric contribution.

    Required score metrics may not be ``None``.  This intentionally raises
    instead of treating an undefined ratio as zero, neutral, or favourable.
    """

    components: list[ScoreComponent] = []
    total = 0.0
    for rule in policy.components:
        observed = candidate.metrics.value_for(rule.metric)
        if observed is None:
            raise UndefinedMetricError(
                f"Candidate {candidate.candidate_id!r} has undefined required score metric "
                f"{rule.metric.value!r}."
            )
        raw_value = float(observed)
        normalized_value = raw_value / rule.scale
        sign = 1.0 if rule.direction is ScoreDirection.HIGHER_IS_BETTER else -1.0
        contribution = rule.weight * sign * normalized_value
        if not math.isfinite(normalized_value) or not math.isfinite(contribution):
            raise UndefinedMetricError(
                f"Candidate {candidate.candidate_id!r} produced a non-finite score "
                f"component for {rule.metric.value!r}."
            )
        total += contribution
        if not math.isfinite(total):
            raise UndefinedMetricError(
                f"Candidate {candidate.candidate_id!r} produced a non-finite total score."
            )
        components.append(
            ScoreComponent(
                metric=rule.metric,
                raw_value=raw_value,
                normalized_value=normalized_value,
                weight=rule.weight,
                direction=rule.direction,
                scale=rule.scale,
                contribution=contribution,
            )
        )
    return total, tuple(components)


def _tie_key(
    candidate: CandidateEvaluation,
    policy: RankingPolicy,
) -> tuple[object, ...]:
    """Construct a deterministic sort suffix after the numeric total score.

    An undefined *tie-break-only* metric sorts after a defined one.  It is not
    converted to zero; its absence remains a distinct, less-preferred state.
    """

    values: list[object] = []
    for metric in policy.tie_breakers:
        observed = candidate.metrics.value_for(metric)
        if observed is None:
            values.append((1, 0.0))
            continue
        direction = policy.direction_for(metric)
        value = float(observed)
        ordered_value = -value if direction is ScoreDirection.HIGHER_IS_BETTER else value
        values.append((0, ordered_value))
    # Candidate id is the final stable key.  It removes insertion-order and
    # hash-order dependence when every numeric criterion is identical.
    values.append(candidate.candidate_id)
    return tuple(values)


def rank_candidates(
    candidates: Iterable[CandidateEvaluation],
    policy: RankingPolicy | None = None,
    *,
    gate_results: Iterable[GateResult] | None = None,
) -> RankingResult:
    """Rank comparable candidates using an immutable, inspectable formula.

    When ``gate_results`` are supplied, only candidates with a matching,
    viable result are eligible.  Non-viable candidates are retained as audit
    identifiers in ``excluded_candidate_ids`` rather than being quietly
    scored.  It always rejects candidates with mixed scope, dataset, data
    fingerprint, backtest configuration, or metric version before computing
    any score.
    """

    materialized = tuple(candidates)
    effective_policy = policy or RankingPolicy()
    # Validate the full submitted population before filtering.  Otherwise a
    # caller could hide a mismatched dataset/configuration behind a failed
    # gate, which would leave the audit population internally inconsistent.
    _validate_comparable_contexts(materialized, effective_policy)
    excluded_candidate_ids: tuple[str, ...] = ()
    if gate_results is not None:
        materialized, excluded_candidate_ids = _select_viable_candidates(
            materialized,
            tuple(gate_results),
        )

    scored: list[tuple[CandidateEvaluation, float, tuple[ScoreComponent, ...]]] = []
    for candidate in materialized:
        total, components = score_candidate(candidate, effective_policy)
        scored.append((candidate, total, components))

    ordered = sorted(
        scored,
        key=lambda item: (-item[1], *_tie_key(item[0], effective_policy)),
    )
    ranked = tuple(
        RankedCandidate(
            rank=index,
            candidate=candidate,
            total_score=total,
            components=components,
        )
        for index, (candidate, total, components) in enumerate(ordered, start=1)
    )
    return RankingResult(
        context=materialized[0].context,
        policy_id=effective_policy.policy_id,
        ranked_candidates=ranked,
        excluded_candidate_ids=excluded_candidate_ids,
    )


def _select_viable_candidates(
    candidates: tuple[CandidateEvaluation, ...],
    gate_results: tuple[GateResult, ...],
) -> tuple[tuple[CandidateEvaluation, ...], tuple[str, ...]]:
    """Require a matched gate audit before allowing a candidate into ranking."""

    if not candidates:
        raise EvaluationError("At least one candidate is required for ranking.")
    result_by_id: dict[str, GateResult] = {}
    for result in gate_results:
        if result.candidate_id in result_by_id:
            raise EvaluationError("Gate results must contain unique candidate IDs.")
        result_by_id[result.candidate_id] = result

    candidate_ids = {candidate.candidate_id for candidate in candidates}
    orphaned_ids = sorted(set(result_by_id) - candidate_ids)
    if orphaned_ids:
        raise EvaluationError(
            "Gate results contain candidate IDs absent from the submitted ranking: "
            + ", ".join(orphaned_ids)
        )

    eligible: list[CandidateEvaluation] = []
    excluded: list[str] = []
    for candidate in candidates:
        result = result_by_id.get(candidate.candidate_id)
        if result is None:
            raise EvaluationError(
                f"Candidate {candidate.candidate_id!r} has no viability gate result."
            )
        if result.context.comparability_key != candidate.context.comparability_key:
            raise MixedEvaluationContextError(
                f"Gate result for candidate {candidate.candidate_id!r} has a different "
                "evaluation context."
            )
        if result.is_viable:
            eligible.append(candidate)
        else:
            excluded.append(candidate.candidate_id)

    if not eligible:
        raise EvaluationError("No viable candidates are available for ranking.")
    return tuple(eligible), tuple(excluded)


def rank_viable_candidates(
    candidates: Iterable[CandidateEvaluation],
    gate_results: Iterable[GateResult],
    policy: RankingPolicy | None = None,
) -> RankingResult:
    """Convenience entry point that requires viability evidence by design."""

    return rank_candidates(candidates, policy, gate_results=gate_results)


# Short alias for callers that use the module as a scoring service.
rank = rank_candidates
