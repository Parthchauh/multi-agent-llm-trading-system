"""Focused contract tests for deterministic Sprint 5 viability gates."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evaluation.gates import evaluate_viability
from evaluation.models import (
    CandidateEvaluation,
    EvaluationContext,
    EvaluationMetrics,
    EvaluationScope,
    EvaluationScopeError,
    GateStatus,
    ScoreMetric,
    UndefinedMetricAction,
    ViabilityPolicy,
)


def _context(scope: EvaluationScope = EvaluationScope.VALIDATION) -> EvaluationContext:
    return EvaluationContext(
        scope=scope,
        dataset_id="validation-2024-q1",
        dataset_fingerprint="dataset-sha256-a",
        backtest_config_fingerprint="config-sha256-a",
        metrics_version="1.1",
    )


def _candidate(**metric_overrides: object) -> CandidateEvaluation:
    payload = {
        "total_return_pct": 12.0,
        "max_drawdown_pct": 8.0,
        "sharpe_ratio": 1.25,
        "sortino_ratio": 1.5,
        "profit_factor": 1.4,
        "number_of_trades": 12,
    }
    payload.update(metric_overrides)
    return CandidateEvaluation(
        candidate_id="candidate-a",
        strategy_hash="strategy-sha256-a",
        context=_context(),
        metrics=EvaluationMetrics(**payload),
    )


def test_viability_gate_reports_each_threshold_and_accepts_candidate() -> None:
    policy = ViabilityPolicy(
        minimum_trade_count=10,
        minimum_total_return_pct=5.0,
        minimum_sharpe_ratio=1.0,
        minimum_profit_factor=1.1,
        maximum_drawdown_pct=10.0,
    )

    result = evaluate_viability(_candidate(), policy)

    assert result.is_viable
    assert not result.failed_metrics
    assert not result.undefined_metrics
    assert [check.metric for check in result.checks] == [
        ScoreMetric.NUMBER_OF_TRADES,
        ScoreMetric.TOTAL_RETURN_PCT,
        ScoreMetric.SHARPE_RATIO,
        ScoreMetric.PROFIT_FACTOR,
        ScoreMetric.MAX_DRAWDOWN_PCT,
    ]
    assert all(check.status is GateStatus.PASSED for check in result.checks)


def test_undefined_metric_is_never_coerced_to_zero_or_passed_silently() -> None:
    result = evaluate_viability(
        _candidate(sharpe_ratio=None),
        ViabilityPolicy(minimum_sharpe_ratio=1.0),
    )

    assert not result.is_viable
    assert result.undefined_metrics == (ScoreMetric.SHARPE_RATIO,)
    check = next(check for check in result.checks if check.metric is ScoreMetric.SHARPE_RATIO)
    assert check.observed is None
    assert check.status is GateStatus.UNDEFINED


def test_policy_can_explicitly_allow_an_undefined_metric_but_keeps_its_audit_record() -> None:
    result = evaluate_viability(
        _candidate(sharpe_ratio=None),
        ViabilityPolicy(
            minimum_sharpe_ratio=1.0,
            undefined_metric_action=UndefinedMetricAction.ALLOW,
        ),
    )

    assert result.is_viable
    assert result.undefined_metrics == (ScoreMetric.SHARPE_RATIO,)
    assert next(
        check for check in result.checks if check.metric is ScoreMetric.SHARPE_RATIO
    ).status is GateStatus.UNDEFINED


def test_wrong_scope_is_rejected_before_candidate_selection() -> None:
    candidate = _candidate()
    test_context = _context(EvaluationScope.TEST)
    candidate = candidate.model_copy(update={"context": test_context})

    with pytest.raises(EvaluationScopeError, match="scope"):
        evaluate_viability(candidate, ViabilityPolicy())


def test_policy_rejects_boolean_threshold_and_contract_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="boolean"):
        ViabilityPolicy(minimum_trade_count=True)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvaluationContext(
            scope=EvaluationScope.VALIDATION,
            dataset_id="validation",
            dataset_fingerprint="data",
            backtest_config_fingerprint="config",
            metrics_version="1.1",
            undeclared="not permitted",
        )


def test_expanded_execution_metrics_can_be_explicitly_gated() -> None:
    result = evaluate_viability(
        _candidate(
            calmar_ratio=0.7,
            payoff_ratio=1.5,
            turnover_pct=185.0,
            max_drawdown_duration_days=8,
            max_drawdown_duration_bars=5,
        ),
        ViabilityPolicy(
            minimum_calmar_ratio=0.5,
            minimum_payoff_ratio=1.2,
            maximum_turnover_pct=200.0,
            maximum_drawdown_duration_days=10,
            maximum_drawdown_duration_bars=6,
        ),
    )

    assert result.is_viable
    assert [check.metric for check in result.checks] == [
        ScoreMetric.NUMBER_OF_TRADES,
        ScoreMetric.CALMAR_RATIO,
        ScoreMetric.PAYOFF_RATIO,
        ScoreMetric.TURNOVER_PCT,
        ScoreMetric.MAX_DRAWDOWN_DURATION_DAYS,
        ScoreMetric.MAX_DRAWDOWN_DURATION_BARS,
    ]
