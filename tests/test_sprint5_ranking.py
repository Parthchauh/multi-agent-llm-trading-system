"""Focused tests for comparable, deterministic Sprint 5 ranking."""

from __future__ import annotations

import pytest

from evaluation.gates import evaluate_viability
from evaluation.models import (
    CandidateEvaluation,
    EvaluationContext,
    EvaluationError,
    EvaluationMetrics,
    EvaluationScope,
    EvaluationScopeError,
    MixedEvaluationContextError,
    RankingPolicy,
    ScoreComponentPolicy,
    ScoreDirection,
    ScoreMetric,
    UndefinedMetricError,
    ViabilityPolicy,
)
from evaluation.ranking import rank_candidates, rank_viable_candidates


def _context(
    *,
    scope: EvaluationScope = EvaluationScope.VALIDATION,
    dataset_fingerprint: str = "dataset-sha256-a",
    config_fingerprint: str = "config-sha256-a",
) -> EvaluationContext:
    return EvaluationContext(
        scope=scope,
        dataset_id="validation-2024-q1",
        dataset_fingerprint=dataset_fingerprint,
        backtest_config_fingerprint=config_fingerprint,
        metrics_version="1.1",
    )


def _candidate(
    candidate_id: str,
    *,
    context: EvaluationContext | None = None,
    total_return_pct: float | None = 10.0,
    sharpe_ratio: float | None = 1.0,
    max_drawdown_pct: float | None = 8.0,
    number_of_trades: int | None = 10,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_id=candidate_id,
        strategy_hash=f"strategy-{candidate_id}",
        context=context or _context(),
        metrics=EvaluationMetrics(
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            number_of_trades=number_of_trades,
        ),
    )


def _return_only_policy() -> RankingPolicy:
    return RankingPolicy(
        policy_id="return-only",
        components=(
            ScoreComponentPolicy(
                metric=ScoreMetric.TOTAL_RETURN_PCT,
                weight=1.0,
                direction=ScoreDirection.HIGHER_IS_BETTER,
                scale=100.0,
            ),
        ),
        tie_breakers=(ScoreMetric.SHARPE_RATIO,),
    )


@pytest.mark.parametrize(
    "other_context",
    [
        _context(scope=EvaluationScope.TEST),
        _context(dataset_fingerprint="dataset-sha256-b"),
        _context(config_fingerprint="config-sha256-b"),
    ],
)
def test_mixed_scope_dataset_or_configuration_is_rejected(
    other_context: EvaluationContext,
) -> None:
    with pytest.raises(MixedEvaluationContextError, match="different evaluation"):
        rank_candidates(
            [_candidate("a"), _candidate("b", context=other_context)],
            _return_only_policy(),
        )


def test_final_holdout_is_rejected_by_default_validation_ranking_policy() -> None:
    with pytest.raises(EvaluationScopeError, match="requires scope"):
        rank_candidates(
            [_candidate("holdout", context=_context(scope=EvaluationScope.FINAL_HOLDOUT))]
        )


def test_equal_numeric_scores_have_a_stable_candidate_id_tie_break() -> None:
    # Reverse input order deliberately; the ranking must not depend on it.
    result = rank_candidates(
        [_candidate("candidate-z"), _candidate("candidate-a")],
        _return_only_policy(),
    )

    assert [item.candidate.candidate_id for item in result.ranked_candidates] == [
        "candidate-a",
        "candidate-z",
    ]
    assert [item.rank for item in result.ranked_candidates] == [1, 2]


def test_undefined_required_score_metric_is_rejected_not_scored_as_zero() -> None:
    with pytest.raises(UndefinedMetricError, match="undefined required score metric"):
        rank_candidates([_candidate("undefined", total_return_pct=None)], _return_only_policy())


def test_only_candidates_with_matching_viability_passes_are_ranked() -> None:
    viable = _candidate("viable", total_return_pct=4.0)
    rejected = _candidate("rejected", total_return_pct=-10.0)
    gates = [
        evaluate_viability(viable, ViabilityPolicy(minimum_total_return_pct=0.0)),
        evaluate_viability(rejected, ViabilityPolicy(minimum_total_return_pct=0.0)),
    ]

    result = rank_viable_candidates(
        [rejected, viable],
        gates,
        _return_only_policy(),
    )

    assert [item.candidate.candidate_id for item in result.ranked_candidates] == ["viable"]
    assert result.excluded_candidate_ids == ("rejected",)


def test_viable_ranking_requires_gate_evidence_for_every_candidate() -> None:
    candidate = _candidate("unproven")
    with pytest.raises(EvaluationError, match="no viability gate result"):
        rank_viable_candidates([candidate], [], _return_only_policy())


def test_mixed_context_is_rejected_even_when_one_candidate_fails_its_gate() -> None:
    viable = _candidate("viable", total_return_pct=5.0)
    mismatched = _candidate(
        "mismatched",
        context=_context(dataset_fingerprint="dataset-sha256-b"),
        total_return_pct=-5.0,
    )
    policy = ViabilityPolicy(minimum_total_return_pct=0.0)

    with pytest.raises(MixedEvaluationContextError, match="different evaluation"):
        rank_viable_candidates(
            [viable, mismatched],
            [evaluate_viability(viable, policy), evaluate_viability(mismatched, policy)],
            _return_only_policy(),
        )
