"""Sprint 7 chronological isolation, robustness, and final-test tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backtesting.engine import BacktestEngine
from backtesting.models import BacktestConfig
from data.research_split import FrozenStrategySelection
from evaluation.gates import evaluate_viability
from evaluation.models import (
    CandidateEvaluation,
    EvaluationContext,
    EvaluationMetrics,
    EvaluationScope,
    ViabilityPolicy,
)
from research_validation import (
    CostScenario,
    DatasetPartitions,
    DeterministicRobustnessEvaluator,
    FinalTestAlreadyExecutedError,
    FinalTestExecutor,
    MonteCarloConfig,
    MonteCarloMethod,
    OOSPartitionError,
    ParameterPerturbation,
    ParameterTarget,
    RobustnessScore,
    TimeSeriesSplitConfig,
    TradeSkipConfig,
    ValidationCandidate,
    WalkForwardConfig,
    WalkForwardEvaluator,
    WalkForwardPlan,
    assess_overfitting_risk,
    generate_parameter_variants,
    monte_carlo_trade_resampling,
    regime_stability,
    score_robustness,
    select_validation_candidate,
    subperiod_stability,
    trade_skip_robustness,
)
from risk import RiskPolicy, evaluate_risk
from strategies.compiler import StrategyCompiler
from tests.conftest_helpers import make_ohlcv, make_simple_strategy


def _config() -> BacktestConfig:
    return BacktestConfig(
        initial_capital=100_000.0,
        commission_pct=0.001,
        slippage_pct=0.0005,
        position_size_pct=20.0,
    )


def test_chronological_partitions_seal_the_final_test_from_agent_access() -> None:
    frame = make_ohlcv(n=200)
    partitions = DatasetPartitions.from_frame(frame, dataset_id="oos-fixture")
    agent = partitions.agent_access()
    validation = partitions.validation_access()
    fingerprints = partitions.partition_fingerprints

    assert agent.train.fingerprint.end < validation.validation.fingerprint.start
    assert fingerprints[1].end < fingerprints[2].start
    assert not hasattr(agent, "final_test")
    assert not hasattr(validation, "final_test")
    with pytest.raises(OOSPartitionError):
        partitions.release_final_test("not-frozen")  # type: ignore[arg-type]

    frame.loc[frame.index[0], "Close"] *= 9
    assert agent.train.observations.iloc[0]["Close"] != frame.iloc[0]["Close"]


def test_split_config_and_walk_forward_windows_are_strictly_chronological() -> None:
    with pytest.raises(Exception):
        TimeSeriesSplitConfig(train_fraction=0.7, validation_fraction=0.2, test_fraction=0.2)
    with pytest.raises(Exception):
        TimeSeriesSplitConfig(train_fraction=True)  # type: ignore[arg-type]

    frame = make_ohlcv(n=150)
    folds = WalkForwardPlan.build(
        frame, WalkForwardConfig(train_size=40, validation_size=20, step_size=20)
    )
    assert len(folds) == 5
    for fold in folds:
        assert fold.train_end == fold.validation_start
        assert fold.train_frame.index[-1] < fold.validation_frame.index[0]


def test_walk_forward_results_ignore_future_perturbation_for_earlier_fold() -> None:
    frame = make_ohlcv(n=180)
    evaluator = WalkForwardEvaluator()
    config = WalkForwardConfig(train_size=60, validation_size=30, step_size=30, expanding=True)
    baseline = evaluator.evaluate(
        strategy_id="wf-1",
        strategy=make_simple_strategy(),
        data=frame,
        walk_forward=config,
        backtest_config=_config(),
        viability_policy=ViabilityPolicy(minimum_trade_count=0),
    )
    perturbed = frame.copy(deep=True)
    cutoff = perturbed.index[90]
    perturbed.loc[perturbed.index > cutoff, ["Open", "High", "Low", "Close"]] *= 10
    rerun = evaluator.evaluate(
        strategy_id="wf-1",
        strategy=make_simple_strategy(),
        data=perturbed,
        walk_forward=config,
        backtest_config=_config(),
        viability_policy=ViabilityPolicy(minimum_trade_count=0),
    )

    assert baseline.folds[0].evaluation.metrics == rerun.folds[0].evaluation.metrics
    assert baseline.folds[0].viability == rerun.folds[0].viability


def test_cost_delay_and_parameter_sensitivity_are_bounded_and_deterministic() -> None:
    frame = make_ohlcv(n=160)
    strategy = make_simple_strategy()
    evaluator = DeterministicRobustnessEvaluator()
    report = evaluator.cost_sensitivity(
        strategy=strategy,
        data=frame,
        base_config=_config(),
        scenarios=(
            CostScenario(name="base", commission_pct=0.001, slippage_pct=0.0005),
            CostScenario(name="stressed", commission_pct=0.01, slippage_pct=0.01),
        ),
    )
    delayed = evaluator.entry_delay_sensitivity(
        strategy=strategy, data=frame, backtest_config=_config(), additional_delay_bars=1
    )
    variants = generate_parameter_variants(
        strategy,
        ParameterPerturbation(
            target=ParameterTarget.ENTRY_CONDITION_VALUE,
            condition_index=0,
            deltas=(-5.0, 5.0),
        ),
    )

    assert [outcome.scenario.name for outcome in report.outcomes] == ["base", "stressed"]
    assert delayed.additional_delay_bars == 1
    assert len(variants.variants) == 2
    assert all(variant.strategy is not None for variant in variants.variants)
    with pytest.raises(Exception):
        ParameterPerturbation(
            target=ParameterTarget.ENTRY_CONDITION_VALUE,
            condition_index=0,
            deltas=(-101.0,),
        )


def test_trade_resampling_regime_and_subperiod_reports_are_seeded_and_disclosed() -> None:
    frame = make_ohlcv(n=200)
    compiled = StrategyCompiler().compile(make_simple_strategy(), frame)
    result = BacktestEngine(_config()).run(frame, compiled)
    skipped_first = trade_skip_robustness(
        result, TradeSkipConfig(skip_fraction=0.5, repetitions=5, seed=17)
    )
    skipped_second = trade_skip_robustness(
        result, TradeSkipConfig(skip_fraction=0.5, repetitions=5, seed=17)
    )
    monte = monte_carlo_trade_resampling(
        result,
        MonteCarloConfig(
            method=MonteCarloMethod.TRADE_BOOTSTRAP, repetitions=10, seed=9
        ),
    )
    regimes = regime_stability(
        result, {trade.entry_date: "TRENDING_BULL" for trade in result.trades}
    )
    stability = subperiod_stability(result, periods=4)

    assert skipped_first == skipped_second
    assert "Completed trade net PnL" in monte.assumptions[0]
    assert regimes.unclassified_trade_count == 0
    assert len(stability.period_returns_pct) in {0, 4}


def test_validation_selection_precedes_exactly_one_final_test() -> None:
    frame = make_ohlcv(n=200)
    partitions = DatasetPartitions.from_frame(frame, dataset_id="selection-fixture")
    validation = partitions.validation_access().validation
    strategy = make_simple_strategy()
    result = BacktestEngine(_config()).run(
        validation.observations, StrategyCompiler().compile(strategy, validation.observations)
    )
    context = EvaluationContext(
        scope=EvaluationScope.VALIDATION,
        dataset_id=validation.fingerprint.partition_id,
        dataset_fingerprint=validation.fingerprint.data_sha256,
        backtest_config_fingerprint="cfg-selection",
        metrics_version=result.metrics.metrics_version,
    )
    evaluation = CandidateEvaluation(
        candidate_id="candidate-v1",
        context=context,
        metrics=EvaluationMetrics.from_backtest_metrics(result.metrics),
    )
    candidate = ValidationCandidate(
        strategy_id="candidate-v1",
        strategy=strategy,
        evaluation=evaluation,
        viability=evaluate_viability(evaluation, ViabilityPolicy(minimum_trade_count=0)),
        risk_assessment=evaluate_risk(result, RiskPolicy()),
    )
    selection = select_validation_candidate([candidate])
    executor = FinalTestExecutor()
    final = executor.execute(
        partitions=partitions,
        selection=selection,
        strategy=strategy,
        backtest_config=_config(),
    )

    assert final.final_test_partition_id.endswith("final_test")
    with pytest.raises(FinalTestAlreadyExecutedError):
        executor.execute(
            partitions=partitions,
            selection=selection,
            strategy=strategy,
            backtest_config=_config(),
        )

    forged = FrozenStrategySelection(
        strategy_id="candidate-v1",
        strategy_hash="not-the-semantic-fingerprint",
        frozen_at=datetime.now(timezone.utc),
    )
    with pytest.raises(Exception):
        executor.execute(
            partitions=partitions,
            selection=selection.model_copy(update={"frozen_selection": forged}),
            strategy=strategy,
            backtest_config=_config(),
        )


def test_overfitting_diagnostics_flag_disclosed_risks_only() -> None:
    frame = make_ohlcv(n=160)
    result = BacktestEngine(_config()).run(
        frame, StrategyCompiler().compile(make_simple_strategy(), frame)
    )
    report = assess_overfitting_risk(
        strategy=make_simple_strategy(),
        backtest_metrics=result.metrics,
        train_total_return_pct=40.0,
        validation_total_return_pct=1.0,
        final_test_total_return_pct=-20.0,
        refinement_generation=4,
        minimum_trade_count=100,
    )

    assert "LOW_TRADE_COUNT" in report.flags
    assert "EXTREME_TRAIN_VALIDATION_GAP" in report.flags
    assert "EXTREME_VALIDATION_TEST_GAP" in report.flags
    assert "EXCESSIVE_REFINEMENT_GENERATIONS" in report.flags
    assert report.includes_final_test_evidence is True
    with pytest.raises(Exception, match="Final-test"):
        score_robustness(
            walk_forward_pass_rate=1.0,
            cost_sensitivity=None,
            trade_skip=None,
            overfitting=report,
        )


def test_validation_candidate_rejects_a_non_validation_robustness_score() -> None:
    with pytest.raises(Exception, match="final-test"):
        ValidationCandidate(
            strategy_id="candidate-v1",
            strategy=make_simple_strategy(),
            evaluation=CandidateEvaluation(
                candidate_id="candidate-v1",
                context=EvaluationContext(
                    scope=EvaluationScope.VALIDATION,
                    dataset_id="validation",
                    dataset_fingerprint="data",
                    backtest_config_fingerprint="config",
                    metrics_version="1.2",
                ),
                metrics=EvaluationMetrics(number_of_trades=1),
            ),
            viability=evaluate_viability(
                CandidateEvaluation(
                    candidate_id="candidate-v1",
                    context=EvaluationContext(
                        scope=EvaluationScope.VALIDATION,
                        dataset_id="validation",
                        dataset_fingerprint="data",
                        backtest_config_fingerprint="config",
                        metrics_version="1.2",
                    ),
                    metrics=EvaluationMetrics(number_of_trades=1),
                ),
                ViabilityPolicy(minimum_trade_count=0),
            ),
            risk_assessment=evaluate_risk(
                BacktestEngine(_config()).run(
                    make_ohlcv(n=80),
                    StrategyCompiler().compile(make_simple_strategy(), make_ohlcv(n=80)),
                ),
                RiskPolicy(),
            ),
            robustness=RobustnessScore(
                score=100.0,
                contributions=(("forged", 100.0),),
                validation_only=False,
            ),
        )
