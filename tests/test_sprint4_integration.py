"""Golden end-to-end Sprint 4 integration and quantitative-authority test."""

from __future__ import annotations

from backtesting.models import BacktestConfig, BacktestMetrics
from tradingagents.agents.regime_analyst import RegimeAnalystAgent
from tradingagents.agents.strategy_generator import StrategyGeneratorAgent
from tradingagents.graph.research_graph import Sprint4ResearchGraph
from tradingagents.graph.research_models import ResearchRunStatus
from tests.conftest_helpers import make_crossover_strategy, make_dataset_split
from tests.test_regime_agent import QueueStructuredClient, _valid_assessment
from tests.test_strategy_generator_agent import _draft


def test_golden_market_to_metrics_pipeline() -> None:
    strategy = make_crossover_strategy().model_dump(mode="json")
    strategy["entry"]["long_conditions"] = {
        "logic": "ALL",
        "conditions": [
            {"indicator": "ema_20", "operator": ">", "value": "ema_50"},
            {"indicator": "rsi_14", "operator": ">", "value": 55.0},
        ],
    }
    regime_client = QueueStructuredClient([_valid_assessment()])
    strategy_client = QueueStructuredClient([_draft(strategy)])
    graph = Sprint4ResearchGraph(
        regime_agent=RegimeAnalystAgent(regime_client),
        strategy_generator=StrategyGeneratorAgent(strategy_client, max_attempts=2),
    )

    result = graph.run(
        symbol="SYNTH",
        dataset_split=make_dataset_split(n=400),
        backtest_config=BacktestConfig(
            initial_capital=100_000.0,
            commission_pct=0.001,
            slippage_pct=0.0005,
        ),
    )

    assert result.status is ResearchRunStatus.COMPLETED
    assert result.strategy is not None
    assert result.strategy.entry.long_conditions.logic == "ALL"
    assert result.backtest_result is not None
    assert isinstance(result.backtest_metrics, BacktestMetrics)
    assert result.backtest_metrics == result.backtest_result.metrics
    assert result.metadata.regime_prompt_version == "1.0"
    assert result.metadata.strategy_prompt_version == "1.0"
    assert result.metadata.symbol == "SYNTH"
    assert [event.stage for event in result.events] == [
        "load_data",
        "compute_regime",
        "analyze_regime",
        "generate_strategy",
        "validate_strategy",
        "compile_strategy",
        "backtest",
    ]
