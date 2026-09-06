"""Authoritative-boundary tests for the composed Sprint 6 lifecycle."""

from __future__ import annotations

import json

import pytest

from backtesting.models import BacktestConfig, BacktestMetrics, BacktestResult
from evaluation.models import ViabilityPolicy
from risk import RiskPolicy
from strategies.compiler import StrategyCompiler
from tests.conftest_helpers import make_dataset_split, make_simple_strategy
from tests.test_regime_agent import QueueStructuredClient, _valid_assessment
from tests.test_strategy_generator_agent import _draft
from tradingagents.agents.regime_analyst import RegimeAnalystAgent
from tradingagents.agents.sprint6_agents import (
    PerformanceAnalystAgent,
    RefinementAgent,
    RiskManagerAgent,
    StrategyCriticAgent,
)
from tradingagents.agents.strategy_generator import StrategyGeneratorAgent
from tradingagents.graph.research_graph import Sprint4ResearchGraph
from tradingagents.graph.sprint6_lifecycle import (
    Sprint6LifecycleConfig,
    Sprint6ResearchGraph,
    Sprint6RunStatus,
)


def _config() -> BacktestConfig:
    return BacktestConfig(
        initial_capital=100_000.0,
        commission_pct=0.001,
        slippage_pct=0.0005,
        position_size_pct=50.0,
    )


def _initial_graph() -> Sprint4ResearchGraph:
    return Sprint4ResearchGraph(
        regime_agent=RegimeAnalystAgent(QueueStructuredClient([_valid_assessment()])),
        strategy_generator=StrategyGeneratorAgent(QueueStructuredClient([_draft()]), max_attempts=1),
    )


class _StaticEngine:
    """A deterministic result fixture that records every S6 execution input."""

    frames = []

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(self, data, compiled, *, evaluation_start=None) -> BacktestResult:
        type(self).frames.append(data.copy(deep=True))
        metrics = BacktestMetrics(
            total_return_pct=2.0,
            cagr_pct=2.0,
            max_drawdown_pct=10.0,
            sharpe_ratio=0.6,
            sortino_ratio=0.5,
            win_rate_pct=50.0,
            profit_factor=1.2,
            expectancy=10.0,
            number_of_trades=2,
            winning_trades=1,
            losing_trades=1,
            average_win=20.0,
            average_loss=-10.0,
            largest_win=20.0,
            largest_loss=-10.0,
            average_holding_days=2.0,
            exposure_pct=25.0,
            turnover_pct=50.0,
        )
        return BacktestResult(
            config=self.config,
            start_date=data.index[0].date(),
            end_date=data.index[-1].date(),
            initial_capital=self.config.initial_capital,
            final_equity=self.config.initial_capital * 1.02,
            metrics=metrics,
        )


class _NeverCalled:
    def __init__(self) -> None:
        self.calls = 0

    def review(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("LLM review must not run after deterministic rejection")

    def analyze(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("LLM analysis must not run after deterministic rejection")

    def propose(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("Refinement must not run after deterministic rejection")


class _CountingCompiler:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.inner = StrategyCompiler()
        self.calls = 0
        self.fail_after = fail_after

    def compile(self, strategy, data):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("deliberate child compiler failure")
        return self.inner.compile(strategy, data)


def _review_agents(*, raw_child: dict | None = None):
    raw = raw_child or make_simple_strategy(entry_value=56.0).model_dump(mode="json")
    risk = RiskManagerAgent(
        QueueStructuredClient(
            [{"key_risks": ["Use deterministic limits."], "confidence": 0.2}]
        )
    )
    performance = PerformanceAnalystAgent(
        QueueStructuredClient(
            [
                {
                    "strengths": ["Metrics are supplied deterministically."],
                    "weaknesses": [],
                    "performance_concentration": "Not recalculated by the agent.",
                    "trade_quality": "Not recalculated by the agent.",
                    "warnings": [],
                    "confidence": 0.2,
                }
            ]
        )
    )
    critic = StrategyCriticAgent(
        QueueStructuredClient(
            [
                {
                    "accepted_assumptions": [],
                    "rejected_assumptions": [],
                    "structural_problems": ["One bounded child is requested."],
                    "suggested_changes": [],
                    "refinement_priority": "LOW",
                    "confidence": 0.2,
                }
            ]
        )
    )
    refiner = RefinementAgent(
        QueueStructuredClient(
            [
                {
                    "parent_strategy_id": "placeholder",
                    "revision_reason": "Bounded lifecycle test revision.",
                    "changes": [],
                    "revised_strategy": raw,
                    "confidence": 0.2,
                }
            ]
        )
    )
    # The parent ID is derived at runtime; replace the queue client's validate
    # hook with a small schema-preserving resolver for this one test fixture.
    def invoke(schema, messages):
        item = refiner.client.responses.pop(0)
        prompt_data = json.loads(messages[1][1].split("\n", 1)[1])
        item["parent_strategy_id"] = prompt_data["parent_strategy_id"]
        return schema.model_validate(item)

    refiner.client.invoke = invoke  # type: ignore[method-assign]
    return risk, performance, critic, refiner


def _lifecycle(*, config: Sprint6LifecycleConfig, compiler=None, agents=()):
    kwargs = {}
    if agents:
        kwargs.update(
            risk_manager=agents[0],
            performance_analyst=agents[1],
            critic=agents[2],
            refinement_agent=agents[3],
        )
    return Sprint6ResearchGraph(
        initial_graph=_initial_graph(),
        config=config,
        compiler=compiler,
        engine_factory=_StaticEngine,
        **kwargs,
    )


def test_lifecycle_stays_on_validation_data_and_can_finish_without_refinement() -> None:
    _StaticEngine.frames = []
    split = make_dataset_split()
    result = _lifecycle(
        config=Sprint6LifecycleConfig(
            max_debate_rounds=0,
            max_refinement_rounds=0,
            viability_policy=ViabilityPolicy(minimum_trade_count=0),
        )
    ).run(symbol="TEST", dataset_split=split, backtest_config=_config())

    assert result.status is Sprint6RunStatus.COMPLETED
    assert result.selected_record is not None
    assert len(result.strategy_records) == 1
    assert _StaticEngine.frames[0].index[-1] < split.partition_fingerprints[2].start


def test_failed_deterministic_risk_stops_before_any_llm_risk_override() -> None:
    _StaticEngine.frames = []
    never = _NeverCalled()
    result = _lifecycle(
        config=Sprint6LifecycleConfig(
            max_debate_rounds=0,
            max_refinement_rounds=1,
            viability_policy=ViabilityPolicy(minimum_trade_count=0),
            risk_policy=RiskPolicy(max_drawdown_pct=5.0),
        ),
        agents=(never, never, never, never),
    ).run(symbol="TEST", dataset_split=make_dataset_split(), backtest_config=_config())

    assert result.status is Sprint6RunStatus.RISK_REJECTED
    assert result.selected_record is not None
    assert never.calls == 0


def test_invalid_refinement_never_reaches_compiler_or_backtester() -> None:
    invalid = make_simple_strategy().model_dump(mode="json")
    invalid["entry"]["long_conditions"]["conditions"][0]["indicator"] = "not_supported"
    compiler = _CountingCompiler()
    result = _lifecycle(
        config=Sprint6LifecycleConfig(
            max_debate_rounds=0,
            max_refinement_rounds=1,
            viability_policy=ViabilityPolicy(minimum_trade_count=0),
        ),
        compiler=compiler,
        agents=_review_agents(raw_child=invalid),
    ).run(symbol="TEST", dataset_split=make_dataset_split(), backtest_config=_config())

    assert result.status is Sprint6RunStatus.FAILED_REFINEMENT_VALIDATION
    assert compiler.calls == 1  # Root only; the invalid child remains a raw draft.
    assert len(result.critic_reports) == 1
    assert result.selected_record is not None


def test_child_compiler_failure_halts_bounded_refinement() -> None:
    compiler = _CountingCompiler(fail_after=1)
    result = _lifecycle(
        config=Sprint6LifecycleConfig(
            max_debate_rounds=0,
            max_refinement_rounds=1,
            viability_policy=ViabilityPolicy(minimum_trade_count=0),
        ),
        compiler=compiler,
        agents=_review_agents(),
    ).run(symbol="TEST", dataset_split=make_dataset_split(), backtest_config=_config())

    assert result.status is Sprint6RunStatus.FAILED_REFINEMENT_EXECUTION
    assert compiler.calls == 2
    assert len(result.strategy_records) == 1


@pytest.mark.parametrize("value", [-1, 6, True])
def test_refinement_loop_policy_is_hard_bounded(value) -> None:
    with pytest.raises(Exception):
        Sprint6LifecycleConfig(max_refinement_rounds=value)
