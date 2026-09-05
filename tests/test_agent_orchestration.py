"""Routing and authoritative-boundary tests for the Sprint 4 LangGraph."""

from __future__ import annotations
from typing import Any
import pytest
from backtesting.models import BacktestConfig
from regime.models import AssessmentSource
from strategies.compiler import StrategyCompiler
from tradingagents.agents.regime_analyst import RegimeAnalystAgent
from tradingagents.agents.strategy_generator import StrategyGeneratorAgent
from tradingagents.graph.research_graph import Sprint4ResearchGraph
from tradingagents.graph.research_models import ResearchRunStatus
from tests.conftest_helpers import make_dataset_split, make_simple_strategy
from tests.test_regime_agent import QueueStructuredClient, _valid_assessment
from tests.test_strategy_generator_agent import _draft


def _config():
    return BacktestConfig(initial_capital=100_000.0, commission_pct=0.001, slippage_pct=0.0005, position_size_pct=50.0)


def _graph(strategy_responses, *, regime_responses=None, max_attempts=3, compiler=None, engine_factory=None):
    regime_client = QueueStructuredClient(regime_responses if regime_responses is not None else [_valid_assessment()])
    strategy_client = QueueStructuredClient(strategy_responses)
    return Sprint4ResearchGraph(
        regime_agent=RegimeAnalystAgent(regime_client),
        strategy_generator=StrategyGeneratorAgent(strategy_client, max_attempts=max_attempts),
        compiler=compiler,
        engine_factory=engine_factory,
    )


def test_happy_path_reaches_authoritative_metrics():
    result = _graph([_draft()]).run(symbol="TEST", dataset_split=make_dataset_split(), backtest_config=_config())
    assert result.status is ResearchRunStatus.COMPLETED
    assert result.strategy is not None
    assert result.backtest_result is not None
    assert result.metadata.generation_attempts == 1
    assert result.metadata.backtest_config.position_size_pct == result.strategy.position_sizing.value


def test_invalid_candidate_routes_to_retry_then_continues():
    invalid = make_simple_strategy().model_dump(mode="json")
    invalid["entry"]["long_conditions"]["conditions"][0]["indicator"] = "bad_alpha"
    result = _graph([_draft(invalid), _draft()], max_attempts=2).run(symbol="TEST", dataset_split=make_dataset_split(), backtest_config=_config())
    assert result.status is ResearchRunStatus.COMPLETED
    assert result.metadata.generation_attempts == 2
    outcomes = [(e.stage, e.outcome) for e in result.events]
    assert ("validate_strategy", "rejected") in outcomes
    assert [record.outcome for record in result.strategy_attempts] == [
        "validation_rejected",
        "accepted",
    ]
    assert result.metadata.regime_sampling.effective_temperature == 0.0
    assert result.metadata.strategy_sampling.effective_seed == 7


class CountingCompiler:
    def __init__(self, *, fail=False):
        self.calls = 0
        self.fail = fail
        self.inner = StrategyCompiler()

    def compile(self, strategy, data):
        self.calls += 1
        if self.fail:
            raise RuntimeError("compiler deliberately failed")
        return self.inner.compile(strategy, data)


def test_exhausted_invalid_candidate_never_reaches_compiler():
    invalid = make_simple_strategy().model_dump(mode="json")
    invalid["entry"]["long_conditions"]["conditions"][0]["operator"] = "eval"
    compiler = CountingCompiler()
    result = _graph([_draft(invalid), _draft(invalid)], max_attempts=2, compiler=compiler).run(symbol="TEST", dataset_split=make_dataset_split(), backtest_config=_config())
    assert result.status is ResearchRunStatus.FAILED_VALIDATION
    assert compiler.calls == 0
    assert result.backtest_result is None


def test_regime_llm_failure_falls_back_and_pipeline_continues():
    result = _graph([_draft()], regime_responses=[TimeoutError("regime timeout")]).run(symbol="TEST", dataset_split=make_dataset_split(), backtest_config=_config())
    assert result.status is ResearchRunStatus.COMPLETED
    assert result.regime_assessment is not None
    assert result.regime_assessment.source is AssessmentSource.DETERMINISTIC_FALLBACK
    assert result.warnings


def test_compiler_failure_terminates_before_backtest():
    engine_calls = 0

    def engine_factory(config):
        nonlocal engine_calls
        engine_calls += 1
        raise AssertionError("engine must not be constructed")

    compiler = CountingCompiler(fail=True)
    result = _graph([_draft()], compiler=compiler, engine_factory=engine_factory).run(symbol="TEST", dataset_split=make_dataset_split(), backtest_config=_config())
    assert result.status is ResearchRunStatus.FAILED_COMPILATION
    assert compiler.calls == 1
    assert engine_calls == 0


def test_backtester_failure_is_typed_in_status():
    class FailingEngine:
        def __init__(self, config):
            self.config = config

        def run(self, data, compiled, *, evaluation_start=None):
            raise RuntimeError("backtester deliberately failed")

    result = _graph([_draft()], engine_factory=FailingEngine).run(symbol="TEST", dataset_split=make_dataset_split(), backtest_config=_config())
    assert result.status is ResearchRunStatus.FAILED_BACKTEST
    assert result.backtest_result is None
    assert "RuntimeError" in result.errors[0]


def test_generation_provider_exhaustion_terminates_gracefully():
    result = _graph([TimeoutError("one"), TimeoutError("two")], max_attempts=2).run(symbol="TEST", dataset_split=make_dataset_split(), backtest_config=_config())
    assert result.status is ResearchRunStatus.FAILED_GENERATION
    assert result.metadata.generation_attempts == 2
    assert result.backtest_result is None
    assert [record.outcome for record in result.strategy_attempts] == [
        "generation_failed",
        "generation_failed",
    ]
    serialized = result.model_dump_json()
    assert "one" not in serialized
    assert "two" not in serialized


def test_invalid_market_data_stops_before_agents():
    """Verify that missing Volume is caught by validate_ohlcv (the data boundary guard)."""
    from data.schema import MissingMarketDataColumnsError, validate_ohlcv
    good_split = make_dataset_split()
    bad_df = good_split.generation_view().context.observations.drop(columns=["Volume"])
    with pytest.raises(MissingMarketDataColumnsError):
        validate_ohlcv(bad_df)
