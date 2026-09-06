"""Focused Sprint 6 tests for immutable lineage and refinement acceptance."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from evaluation.gates import evaluate_viability
from evaluation.models import (
    CandidateEvaluation,
    EvaluationContext,
    EvaluationMetrics,
    EvaluationScope,
    GateResult,
    ScoreMetric,
    ViabilityPolicy,
)
from strategies.schema import StrategySchema
from tradingagents.research import (
    ChildStrategyValidationError,
    RefinementAcceptancePolicy,
    RefinementAcceptanceStatus,
    RefinementQualityLabel,
    RefinementRejectionReason,
    assess_refinement_acceptance,
    create_initial_strategy_record,
    promote_refinement_strategy,
)


def _raw_strategy(*, entry_threshold: float = 55.0) -> dict[str, object]:
    return {
        "metadata": {
            "name": "Lineage test strategy",
            "description": "Structured test fixture",
            "market_type": "equity",
            "timeframe": "1d",
        },
        "entry": {
            "long_conditions": {
                "logic": "ALL",
                "conditions": [
                    {"indicator": "rsi_14", "operator": ">", "value": entry_threshold},
                ],
            }
        },
        "exit": {
            "exit_conditions": {
                "logic": "ANY",
                "conditions": [
                    {"indicator": "rsi_14", "operator": "<", "value": 45.0},
                ],
            }
        },
        "stop_loss": {"type": "percentage", "value": 5.0},
        "take_profit": {"type": "percentage", "value": 10.0},
        "position_sizing": {"type": "fixed_percentage", "value": 10.0},
    }


def _parent_record():
    return create_initial_strategy_record(
        StrategySchema.model_validate(_raw_strategy()),
        strategy_id="strategy-parent-v1",
        revision_reason="Initial research proposal.",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def _child_record(parent, *, raw: dict[str, object] | None = None):
    return promote_refinement_strategy(
        parent,
        raw or _raw_strategy(entry_threshold=56.0),
        revision_reason="Increase RSI entry threshold after deterministic review.",
        created_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )


def _context(
    scope: EvaluationScope = EvaluationScope.VALIDATION,
    *,
    fingerprint: str = "dataset-fingerprint-a",
) -> EvaluationContext:
    return EvaluationContext(
        scope=scope,
        dataset_id="lineage-validation-set",
        dataset_fingerprint=fingerprint,
        backtest_config_fingerprint="backtest-config-a",
        metrics_version="1.2",
    )


def _candidate(
    record,
    *,
    context: EvaluationContext,
    sharpe: float = 1.0,
    drawdown: float = 8.0,
    total_return: float = 10.0,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_id=record.strategy_id,
        strategy_hash=record.lineage.execution_fingerprint,
        context=context,
        metrics=EvaluationMetrics(
            total_return_pct=total_return,
            max_drawdown_pct=drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=1.2,
            profit_factor=1.3,
            number_of_trades=12,
        ),
    )


def _passing_gate(candidate: CandidateEvaluation):
    return evaluate_viability(
        candidate,
        ViabilityPolicy(
            minimum_trade_count=1,
            minimum_sharpe_ratio=0.0,
            maximum_drawdown_pct=50.0,
        ),
    )


def test_lineage_is_immutable_and_child_versions_are_hierarchical() -> None:
    parent = _parent_record()
    child = _child_record(parent)

    assert parent.strategy_id == "strategy-parent-v1"
    assert parent.lineage.parent_strategy_id is None
    assert parent.lineage.generation == 0
    assert child.strategy_id != parent.strategy_id
    assert child.lineage.parent_strategy_id == parent.strategy_id
    assert child.lineage.generation == 1
    assert child.lineage.created_at.tzinfo is timezone.utc

    with pytest.raises(ValidationError):
        parent.lineage.strategy_id = "overwrite-parent"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        child.strategy = parent.strategy  # type: ignore[misc]


def test_invalid_raw_child_is_rejected_before_a_new_version_is_created() -> None:
    parent = _parent_record()
    parent_snapshot = parent.model_dump(mode="json")
    invalid_raw = _raw_strategy()
    invalid_raw["entry"] = {
        "long_conditions": {
            "logic": "ALL",
            "conditions": [{"indicator": "not_registered", "operator": ">", "value": 3.0}],
        }
    }

    with pytest.raises(ChildStrategyValidationError):
        _child_record(parent, raw=invalid_raw)

    # Promotion is construct-only: an invalid request cannot alter v1.
    assert parent.model_dump(mode="json") == parent_snapshot


def test_valid_refinement_is_accepted_only_as_in_sample_evidence() -> None:
    parent = _parent_record()
    child = _child_record(parent)
    context = _context()
    parent_eval = _candidate(parent, context=context, sharpe=1.0, drawdown=8.0)
    child_eval = _candidate(child, context=context, sharpe=1.25, drawdown=8.5)

    result = assess_refinement_acceptance(
        parent=parent,
        child=child,
        parent_evaluation=parent_eval,
        child_evaluation=child_eval,
        child_gate=_passing_gate(child_eval),
    )

    assert result.status is RefinementAcceptanceStatus.ACCEPTED
    assert result.accepted
    assert result.quality_label is RefinementQualityLabel.IN_SAMPLE_ONLY
    assert result.metric_improvement == pytest.approx(0.25)
    assert result.drawdown_degradation_pct == pytest.approx(0.5)
    assert result.rejection_reasons == ()


def test_cross_context_and_final_holdout_evidence_are_rejected() -> None:
    parent = _parent_record()
    child = _child_record(parent)
    parent_eval = _candidate(parent, context=_context(), sharpe=1.0)
    mismatched_child_eval = _candidate(
        child,
        context=_context(fingerprint="dataset-fingerprint-b"),
        sharpe=1.2,
    )

    cross_context = assess_refinement_acceptance(
        parent=parent,
        child=child,
        parent_evaluation=parent_eval,
        child_evaluation=mismatched_child_eval,
        child_gate=_passing_gate(mismatched_child_eval),
    )
    assert not cross_context.accepted
    assert RefinementRejectionReason.EVALUATION_CONTEXT_MISMATCH in cross_context.rejection_reasons

    holdout = _context(EvaluationScope.FINAL_HOLDOUT)
    holdout_parent = _candidate(parent, context=holdout, sharpe=1.0)
    holdout_child = _candidate(child, context=holdout, sharpe=1.2)
    holdout_gate = GateResult(
        candidate_id=child.strategy_id,
        context=holdout,
        policy_id="holdout-must-not-select",
        is_viable=True,
    )
    holdout_result = assess_refinement_acceptance(
        parent=parent,
        child=child,
        parent_evaluation=holdout_parent,
        child_evaluation=holdout_child,
        child_gate=holdout_gate,
    )
    assert not holdout_result.accepted
    assert RefinementRejectionReason.NON_VALIDATION_SCOPE in holdout_result.rejection_reasons


def test_viability_drawdown_and_complexity_limits_are_authoritative() -> None:
    parent = _parent_record()
    child = _child_record(parent)
    context = _context()
    parent_eval = _candidate(parent, context=context, sharpe=1.0, drawdown=8.0)
    child_eval = _candidate(child, context=context, sharpe=1.3, drawdown=11.0)

    rejected_gate = evaluate_viability(
        child_eval,
        ViabilityPolicy(minimum_total_return_pct=20.0),
    )
    result = assess_refinement_acceptance(
        parent=parent,
        child=child,
        parent_evaluation=parent_eval,
        child_evaluation=child_eval,
        child_gate=rejected_gate,
        policy=RefinementAcceptancePolicy(max_drawdown_degradation_pct=1.0),
    )
    assert not result.accepted
    assert RefinementRejectionReason.CHILD_NOT_VIABLE in result.rejection_reasons
    assert RefinementRejectionReason.DRAWDOWN_DEGRADATION_EXCEEDED in result.rejection_reasons

    complex_raw = deepcopy(_raw_strategy(entry_threshold=56.0))
    complex_raw["entry"]["long_conditions"]["conditions"].extend(  # type: ignore[index]
        [
            {"indicator": "close", "operator": ">", "value": "ema_20"},
            {"indicator": "sma_20", "operator": ">", "value": "sma_50"},
        ]
    )
    complex_child = _child_record(parent, raw=complex_raw)
    complex_eval = _candidate(complex_child, context=context, sharpe=1.4, drawdown=8.0)
    complexity_result = assess_refinement_acceptance(
        parent=parent,
        child=complex_child,
        parent_evaluation=parent_eval,
        child_evaluation=complex_eval,
        child_gate=_passing_gate(complex_eval),
        policy=RefinementAcceptancePolicy(max_complexity_increase=0),
    )
    assert not complexity_result.accepted
    assert complexity_result.complexity_delta > 0
    assert RefinementRejectionReason.COMPLEXITY_INCREASE_EXCEEDED in (
        complexity_result.rejection_reasons
    )


def test_acceptance_policy_cannot_select_by_raw_cagr_alone() -> None:
    with pytest.raises(ValidationError, match="CAGR"):
        RefinementAcceptancePolicy(
            improvement_metric=ScoreMetric.CAGR_PCT,
            allowed_improvement_metrics=(ScoreMetric.CAGR_PCT,),
        )


def test_refinement_constructs_a_child_without_mutating_parent_schema() -> None:
    parent = _parent_record()
    parent_snapshot = parent.model_dump(mode="json")
    child = _child_record(parent)

    assert child.strategy.entry.long_conditions.conditions[0].value == 56.0
    assert parent.strategy.entry.long_conditions.conditions[0].value == 55.0
    assert parent.model_dump(mode="json") == parent_snapshot
