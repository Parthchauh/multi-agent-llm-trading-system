"""Tests for deterministic Sprint 5 research-system metrics."""

from __future__ import annotations

from types import SimpleNamespace

from evaluation.research_metrics import (
    measure_strategy_complexity,
    strategy_execution_fingerprint,
    summarize_research_runs,
)
from regime.models import AssessmentSource
from strategies.schema import StrategyMetadata
from tests.conftest_helpers import make_simple_strategy


def _run(*, status, strategy=None, events=(), attempts=0, source=None):
    assessment = None if source is None else SimpleNamespace(source=source)
    return SimpleNamespace(
        status=status,
        strategy=strategy,
        events=events,
        metadata=SimpleNamespace(generation_attempts=attempts),
        regime_assessment=assessment,
    )


def test_execution_fingerprint_ignores_descriptive_metadata() -> None:
    original = make_simple_strategy()
    renamed = original.model_copy(
        update={
            "metadata": StrategyMetadata(
                name="Renamed strategy",
                description="Different prose only",
                market_type="equity",
                timeframe="1d",
            )
        }
    )
    assert strategy_execution_fingerprint(original) == strategy_execution_fingerprint(renamed)


def test_complexity_counts_conditions_indicators_and_parameters() -> None:
    complexity = measure_strategy_complexity(make_simple_strategy())
    assert complexity.condition_count == 2
    assert complexity.unique_indicator_count == 3
    assert complexity.maximum_group_depth == 1
    assert complexity.numeric_parameter_count == 4


def test_research_summary_tracks_validity_failure_and_fallback_rates() -> None:
    strategy = make_simple_strategy()
    events = (
        SimpleNamespace(stage="compile_strategy", outcome="succeeded"),
        SimpleNamespace(stage="backtest", outcome="succeeded"),
    )
    runs = [
        _run(
            status="COMPLETED",
            strategy=strategy,
            events=events,
            attempts=1,
            source=AssessmentSource.LLM,
        ),
        _run(
            status="FAILED_VALIDATION",
            attempts=2,
            source=AssessmentSource.DETERMINISTIC_FALLBACK,
        ),
        _run(
            status="FAILED_COMPILATION",
            strategy=strategy,
            attempts=1,
            source=AssessmentSource.LLM,
        ),
    ]
    summary = summarize_research_runs(runs)
    assert summary.run_count == 3
    assert summary.valid_strategy_rate == 0.6667
    assert summary.compilation_success_rate == 0.5
    assert summary.backtest_success_rate == 1.0
    assert summary.failure_rate == 0.6667
    assert summary.unique_execution_strategy_rate == 0.5
    assert summary.average_condition_count == 2.0
    assert summary.average_generation_attempts == 1.3333
    assert summary.regime_fallback_rate == 0.3333
    assert summary.strategy_generation_call_count == 4
    assert [(item.status, item.count) for item in summary.failure_counts] == [
        ("FAILED_COMPILATION", 1),
        ("FAILED_VALIDATION", 1),
    ]


def test_empty_summary_reports_unknown_rates_not_false_zeros() -> None:
    summary = summarize_research_runs([])
    assert summary.run_count == 0
    assert summary.valid_strategy_rate is None
    assert summary.failure_rate is None
    assert summary.token_usage is None
