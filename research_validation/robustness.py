"""Deterministic Sprint 7 robustness experiments and transparent diagnostics.

The module operates only on validated schemas, canonical OHLCV, and completed
backtests.  It deliberately implements sensitivity experiments rather than a
parameter optimiser: every variation is supplied explicitly, bounded, and
recorded alongside its degradation from the configured base case.
"""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Callable, Iterable, Mapping
from datetime import date
from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backtesting.engine import BacktestEngine
from backtesting.models import BacktestConfig, BacktestMetrics, BacktestResult
from data.schema import validate_ohlcv
from evaluation.gates import evaluate_viability
from evaluation.models import (
    CandidateEvaluation,
    EvaluationContext,
    EvaluationMetrics,
    GateResult,
    ViabilityPolicy,
)
from evaluation.research_metrics import measure_strategy_complexity
from strategies.compiler import CompiledStrategy, StrategyCompiler
from strategies.schema import StrategySchema
from strategies.validator import StrategyValidator


class RobustnessError(ValueError):
    """Raised for invalid bounded sensitivity or resampling configurations."""


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be finite numeric, not boolean.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


class CostScenario(_FrozenStrictModel):
    name: str = Field(min_length=1, max_length=64)
    commission_pct: float = Field(ge=0.0, le=100.0)
    slippage_pct: float = Field(ge=0.0, le=100.0)

    @field_validator("commission_pct", "slippage_pct", mode="before")
    @classmethod
    def costs_are_finite(cls, value: object, info) -> float:
        return _finite(value, name=info.field_name)


class SensitivityOutcome(_FrozenStrictModel):
    scenario: CostScenario
    metrics: EvaluationMetrics
    viability: GateResult | None = None
    total_return_degradation_pct: float | None
    sharpe_degradation: float | None
    drawdown_degradation_pct: float | None


class CostSensitivityReport(_FrozenStrictModel):
    base_metrics: EvaluationMetrics
    outcomes: tuple[SensitivityOutcome, ...] = Field(min_length=1, max_length=16)


class EntryDelayReport(_FrozenStrictModel):
    additional_delay_bars: int = Field(ge=1, le=5)
    base_metrics: EvaluationMetrics
    delayed_metrics: EvaluationMetrics
    total_return_degradation_pct: float | None
    sharpe_degradation: float | None
    drawdown_degradation_pct: float | None


class ParameterTarget(str, Enum):
    ENTRY_CONDITION_VALUE = "ENTRY_CONDITION_VALUE"
    EXIT_CONDITION_VALUE = "EXIT_CONDITION_VALUE"
    STOP_LOSS_VALUE = "STOP_LOSS_VALUE"
    TAKE_PROFIT_VALUE = "TAKE_PROFIT_VALUE"
    POSITION_SIZE_VALUE = "POSITION_SIZE_VALUE"
    MAX_HOLDING_DAYS = "MAX_HOLDING_DAYS"


class ParameterPerturbation(_FrozenStrictModel):
    """One explicit small sensitivity neighborhood, not an optimiser search."""

    target: ParameterTarget
    deltas: tuple[float, ...] = Field(min_length=1, max_length=4)
    condition_index: int | None = Field(default=None, ge=0, le=20)

    @field_validator("deltas", mode="before")
    @classmethod
    def deltas_are_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("deltas")
    @classmethod
    def deltas_are_small_finite_numbers(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        checked = tuple(_finite(value, name="delta") for value in values)
        if any(abs(value) > 100.0 for value in checked):
            raise ValueError("Parameter perturbation deltas are capped at absolute 100.")
        if len(set(checked)) != len(checked):
            raise ValueError("Parameter perturbation deltas must be unique.")
        return checked

    @model_validator(mode="after")
    def condition_index_required_only_for_condition_target(self) -> "ParameterPerturbation":
        is_condition = self.target in {
            ParameterTarget.ENTRY_CONDITION_VALUE,
            ParameterTarget.EXIT_CONDITION_VALUE,
        }
        if is_condition and self.condition_index is None:
            raise ValueError("condition_index is required for condition perturbations.")
        if not is_condition and self.condition_index is not None:
            raise ValueError("condition_index only applies to condition perturbations.")
        return self


class ParameterVariant(_FrozenStrictModel):
    target: ParameterTarget
    delta: float
    strategy: StrategySchema | None = None
    rejected_reason: str | None = None

    @model_validator(mode="after")
    def either_valid_variant_or_rejection(self) -> "ParameterVariant":
        if (self.strategy is None) == (self.rejected_reason is None):
            raise ValueError("Parameter variant must contain exactly one of strategy or rejected_reason.")
        return self


class ParameterSensitivityReport(_FrozenStrictModel):
    variants: tuple[ParameterVariant, ...] = Field(max_length=32)


class TradeSkipConfig(_FrozenStrictModel):
    skip_fraction: float = Field(gt=0.0, lt=1.0)
    repetitions: int = Field(default=100, ge=1, le=1_000)
    seed: int = Field(default=0, ge=0)

    @field_validator("skip_fraction", mode="before")
    @classmethod
    def skip_fraction_is_finite(cls, value: object) -> float:
        return _finite(value, name="skip_fraction")


class TradeResampleMetrics(_FrozenStrictModel):
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    trade_sharpe_ratio: float | None
    included_trade_count: int = Field(ge=0)


class TradeSkipOutcome(_FrozenStrictModel):
    repetition: int = Field(ge=1)
    skipped_trade_indices: tuple[int, ...]
    metrics: TradeResampleMetrics


class TradeSkipReport(_FrozenStrictModel):
    config: TradeSkipConfig
    baseline: TradeResampleMetrics
    outcomes: tuple[TradeSkipOutcome, ...]
    viability_survival_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class MonteCarloMethod(str, Enum):
    TRADE_ORDER_PERMUTATION = "TRADE_ORDER_PERMUTATION"
    TRADE_BOOTSTRAP = "TRADE_BOOTSTRAP"


class MonteCarloConfig(_FrozenStrictModel):
    method: MonteCarloMethod
    repetitions: int = Field(default=500, ge=1, le=5_000)
    seed: int = Field(default=0, ge=0)


class MonteCarloReport(_FrozenStrictModel):
    config: MonteCarloConfig
    mean_total_return_pct: float | None
    median_total_return_pct: float | None
    worst_total_return_pct: float | None
    best_total_return_pct: float | None
    median_max_drawdown_pct: float | None
    assumptions: tuple[str, ...]


class RegimePerformance(_FrozenStrictModel):
    regime: str = Field(min_length=1, max_length=80)
    trade_count: int = Field(ge=0)
    total_net_pnl: float
    hit_rate_pct: float | None
    expectancy: float | None


class RegimeStabilityReport(_FrozenStrictModel):
    segments: tuple[RegimePerformance, ...]
    unclassified_trade_count: int = Field(ge=0)


class SubperiodStabilityReport(_FrozenStrictModel):
    period_returns_pct: tuple[float, ...]
    positive_profit_concentration: float | None
    warning: str | None = None


class OverfittingRiskReport(_FrozenStrictModel):
    flags: tuple[str, ...]
    trade_count: int = Field(ge=0)
    complexity_score: int = Field(ge=0)
    train_validation_gap_pct: float | None
    validation_test_gap_pct: float | None
    parameter_instability_rate: float | None
    cost_sensitivity_rate: float | None
    profit_concentration: float | None
    refinement_generation: int = Field(ge=0)
    includes_final_test_evidence: bool = False


class RobustnessScore(_FrozenStrictModel):
    score: float = Field(ge=0.0, le=100.0)
    contributions: tuple[tuple[str, float], ...]
    validation_only: bool = True


class RobustnessReport(_FrozenStrictModel):
    cost_sensitivity: CostSensitivityReport | None = None
    slippage_sensitivity: CostSensitivityReport | None = None
    parameter_sensitivity: ParameterSensitivityReport | None = None
    delay_sensitivity: EntryDelayReport | None = None
    trade_skip: TradeSkipReport | None = None
    monte_carlo: MonteCarloReport | None = None
    regime_stability: RegimeStabilityReport | None = None
    subperiod_stability: SubperiodStabilityReport | None = None
    overfitting_risk: OverfittingRiskReport | None = None
    score: RobustnessScore | None = None


def _metric_delta(base: float | None, other: float | None) -> float | None:
    return None if base is None or other is None else base - other


def _drawdown_delta(base: float | None, other: float | None) -> float | None:
    return None if base is None or other is None else other - base


def _candidate_metrics(result: BacktestResult) -> EvaluationMetrics:
    return EvaluationMetrics.from_backtest_metrics(result.metrics)


def _scenario_config(base: BacktestConfig, scenario: CostScenario) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=base.initial_capital,
        commission_pct=scenario.commission_pct,
        slippage_pct=scenario.slippage_pct,
        position_size_pct=base.position_size_pct,
        maximum_holding_days=base.maximum_holding_days,
    )


class _DelayedSignals:
    """Add only a causal signal delay; fills remain under the existing engine."""

    def __init__(self, compiled: CompiledStrategy, additional_delay_bars: int) -> None:
        self._compiled = compiled
        self._delay = additional_delay_bars

    def should_enter(self, current_index: int, data: pd.DataFrame) -> bool:
        delayed_index = current_index - self._delay
        return delayed_index >= 0 and self._compiled.should_enter(
            delayed_index, data.iloc[: delayed_index + 1]
        )

    def should_exit(self, current_index: int, data: pd.DataFrame) -> bool:
        delayed_index = current_index - self._delay
        return delayed_index >= 0 and self._compiled.should_exit(
            delayed_index, data.iloc[: delayed_index + 1]
        )

    def get_stop_price(self, entry_price: float, current_index: int, data: pd.DataFrame):
        return self._compiled.get_stop_price(entry_price, current_index, data)

    def get_take_profit_price(self, entry_price: float, stop_price, current_index: int, data):
        return self._compiled.get_take_profit_price(entry_price, stop_price, current_index, data)


class DeterministicRobustnessEvaluator:
    """Execute bounded cost/slippage/delay experiments through canonical services."""

    def __init__(self, compiler: StrategyCompiler | None = None) -> None:
        self.compiler = compiler or StrategyCompiler()

    def cost_sensitivity(
        self,
        *,
        strategy: StrategySchema,
        data: pd.DataFrame,
        base_config: BacktestConfig,
        scenarios: Iterable[CostScenario],
        viability_context: EvaluationContext | None = None,
        viability_policy: ViabilityPolicy | None = None,
    ) -> CostSensitivityReport:
        frame = validate_ohlcv(data)
        compiled = self.compiler.compile(strategy, frame)
        base = _candidate_metrics(BacktestEngine(base_config).run(frame, compiled))
        outcomes: list[SensitivityOutcome] = []
        for scenario in tuple(scenarios):
            config = _scenario_config(base_config, scenario)
            metrics = _candidate_metrics(BacktestEngine(config).run(frame, compiled))
            viability = None
            if viability_context is not None and viability_policy is not None:
                candidate = CandidateEvaluation(
                    candidate_id=f"sensitivity:{scenario.name}",
                    context=viability_context,
                    metrics=metrics,
                )
                viability = evaluate_viability(candidate, viability_policy)
            outcomes.append(
                SensitivityOutcome(
                    scenario=scenario,
                    metrics=metrics,
                    viability=viability,
                    total_return_degradation_pct=_metric_delta(
                        base.total_return_pct, metrics.total_return_pct
                    ),
                    sharpe_degradation=_metric_delta(base.sharpe_ratio, metrics.sharpe_ratio),
                    drawdown_degradation_pct=_drawdown_delta(
                        base.max_drawdown_pct, metrics.max_drawdown_pct
                    ),
                )
            )
        if not outcomes:
            raise RobustnessError("At least one deterministic cost scenario is required.")
        return CostSensitivityReport(base_metrics=base, outcomes=tuple(outcomes))

    def entry_delay_sensitivity(
        self,
        *,
        strategy: StrategySchema,
        data: pd.DataFrame,
        backtest_config: BacktestConfig,
        additional_delay_bars: int = 1,
    ) -> EntryDelayReport:
        if isinstance(additional_delay_bars, bool) or not 1 <= additional_delay_bars <= 5:
            raise RobustnessError("additional_delay_bars must be an integer from 1 through 5.")
        frame = validate_ohlcv(data)
        compiled = self.compiler.compile(strategy, frame)
        base = _candidate_metrics(BacktestEngine(backtest_config).run(frame, compiled))
        delayed = _candidate_metrics(
            BacktestEngine(backtest_config).run(
                frame, _DelayedSignals(compiled, additional_delay_bars)
            )
        )
        return EntryDelayReport(
            additional_delay_bars=additional_delay_bars,
            base_metrics=base,
            delayed_metrics=delayed,
            total_return_degradation_pct=_metric_delta(base.total_return_pct, delayed.total_return_pct),
            sharpe_degradation=_metric_delta(base.sharpe_ratio, delayed.sharpe_ratio),
            drawdown_degradation_pct=_drawdown_delta(
                base.max_drawdown_pct, delayed.max_drawdown_pct
            ),
        )

    def slippage_sensitivity(
        self,
        *,
        strategy: StrategySchema,
        data: pd.DataFrame,
        base_config: BacktestConfig,
        scenarios: Iterable[CostScenario],
        viability_context: EvaluationContext | None = None,
        viability_policy: ViabilityPolicy | None = None,
    ) -> CostSensitivityReport:
        """Run scenarios whose callers vary slippage while holding other costs explicit.

        The same canonical implementation is used for cost and slippage stress;
        scenario fields make every unchanged or changed assumption visible.
        """

        return self.cost_sensitivity(
            strategy=strategy,
            data=data,
            base_config=base_config,
            scenarios=scenarios,
            viability_context=viability_context,
            viability_policy=viability_policy,
        )


def generate_parameter_variants(
    strategy: StrategySchema, perturbation: ParameterPerturbation
) -> ParameterSensitivityReport:
    """Create at most four validated semantic neighbors for one declared parameter."""

    raw = strategy.model_dump(mode="json")
    variants: list[ParameterVariant] = []
    for delta in perturbation.deltas:
        candidate = copy.deepcopy(raw)
        try:
            _apply_delta(candidate, perturbation, delta)
            validation = StrategyValidator().validate_dict(candidate)
            if not validation.is_valid:
                raise RobustnessError("Canonical strategy validation rejected this perturbation.")
            variants.append(
                ParameterVariant(
                    target=perturbation.target,
                    delta=delta,
                    strategy=StrategySchema.model_validate(candidate),
                )
            )
        except (KeyError, TypeError, ValueError, RobustnessError) as exc:
            variants.append(
                ParameterVariant(
                    target=perturbation.target,
                    delta=delta,
                    rejected_reason=type(exc).__name__,
                )
            )
    return ParameterSensitivityReport(variants=tuple(variants))


def _apply_delta(raw: dict, perturbation: ParameterPerturbation, delta: float) -> None:
    target = perturbation.target
    if target is ParameterTarget.ENTRY_CONDITION_VALUE:
        condition = raw["entry"]["long_conditions"]["conditions"][perturbation.condition_index]
        _adjust_numeric_field(condition, "value", delta)
    elif target is ParameterTarget.EXIT_CONDITION_VALUE:
        condition = raw["exit"]["exit_conditions"]["conditions"][perturbation.condition_index]
        _adjust_numeric_field(condition, "value", delta)
    elif target is ParameterTarget.STOP_LOSS_VALUE:
        _adjust_numeric_field(raw["stop_loss"], "value", delta)
    elif target is ParameterTarget.TAKE_PROFIT_VALUE:
        _adjust_numeric_field(raw["take_profit"], "value", delta)
    elif target is ParameterTarget.POSITION_SIZE_VALUE:
        _adjust_numeric_field(raw["position_sizing"], "value", delta)
    elif target is ParameterTarget.MAX_HOLDING_DAYS:
        current = raw["exit"].get("maximum_holding_days")
        if not isinstance(current, int) or isinstance(current, bool):
            raise RobustnessError("maximum_holding_days must be explicitly configured to perturb.")
        adjusted = current + delta
        if not adjusted.is_integer():
            raise RobustnessError("maximum_holding_days perturbation must produce an integer.")
        raw["exit"]["maximum_holding_days"] = int(adjusted)
    else:  # pragma: no cover - exhaustive enum protection
        raise RobustnessError("Unsupported parameter perturbation target.")


def _adjust_numeric_field(container: dict, field: str, delta: float) -> None:
    current = container[field]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise RobustnessError("Only numeric schema parameters may be perturbed.")
    container[field] = float(current) + delta


def _resample_metrics(initial_capital: float, pnl_values: Iterable[float]) -> TradeResampleMetrics:
    equity = float(initial_capital)
    peak = equity
    worst_drawdown = 0.0
    returns: list[float] = []
    included = 0
    for pnl in pnl_values:
        if not math.isfinite(pnl):
            raise RobustnessError("Trade PnL must be finite for resampling.")
        before = equity
        equity += pnl
        if before > 0:
            returns.append(pnl / before)
        peak = max(peak, equity)
        if peak > 0:
            worst_drawdown = max(worst_drawdown, (peak - equity) / peak * 100.0)
        included += 1
    trade_sharpe = None
    if len(returns) > 1:
        std = statistics.stdev(returns)
        if std > 0:
            trade_sharpe = statistics.fmean(returns) / std * math.sqrt(len(returns))
    return TradeResampleMetrics(
        final_equity=equity,
        total_return_pct=(equity / initial_capital - 1.0) * 100.0,
        max_drawdown_pct=worst_drawdown,
        trade_sharpe_ratio=trade_sharpe,
        included_trade_count=included,
    )


def trade_skip_robustness(
    result: BacktestResult,
    config: TradeSkipConfig,
    *,
    viability_predicate: Callable[[TradeResampleMetrics], bool] | None = None,
) -> TradeSkipReport:
    """Randomly skip whole trades under a seeded, auditable sequence model.

    This is not a replacement execution backtest: it measures post-trade
    sequence fragility from completed deterministic trades.  Its Sharpe field
    is explicitly a *trade* Sharpe, never the engine's daily-equity Sharpe.
    """

    pnl = tuple(float(trade.net_pnl) for trade in result.trades)
    baseline = _resample_metrics(result.initial_capital, pnl)
    count_to_skip = math.floor(len(pnl) * config.skip_fraction)
    rng = np.random.default_rng(config.seed)
    outcomes: list[TradeSkipOutcome] = []
    survived = 0
    for repetition in range(1, config.repetitions + 1):
        skipped = tuple(sorted(rng.choice(len(pnl), size=count_to_skip, replace=False).tolist())) if count_to_skip else ()
        skip_set = set(skipped)
        metrics = _resample_metrics(
            result.initial_capital, (value for index, value in enumerate(pnl) if index not in skip_set)
        )
        if viability_predicate is not None and viability_predicate(metrics):
            survived += 1
        outcomes.append(
            TradeSkipOutcome(repetition=repetition, skipped_trade_indices=skipped, metrics=metrics)
        )
    return TradeSkipReport(
        config=config,
        baseline=baseline,
        outcomes=tuple(outcomes),
        viability_survival_rate=(survived / config.repetitions if viability_predicate else None),
    )


def monte_carlo_trade_resampling(result: BacktestResult, config: MonteCarloConfig) -> MonteCarloReport:
    """Perform only disclosed permutation/bootstrap trade resampling."""

    pnl = np.asarray([float(trade.net_pnl) for trade in result.trades], dtype=float)
    if not np.isfinite(pnl).all():
        raise RobustnessError("Trade PnL must be finite for Monte Carlo resampling.")
    rng = np.random.default_rng(config.seed)
    metrics: list[TradeResampleMetrics] = []
    for _ in range(config.repetitions):
        if config.method is MonteCarloMethod.TRADE_ORDER_PERMUTATION:
            sampled = rng.permutation(pnl)
        else:
            sampled = rng.choice(pnl, size=len(pnl), replace=True)
        metrics.append(_resample_metrics(result.initial_capital, sampled.tolist()))
    returns = [item.total_return_pct for item in metrics]
    drawdowns = [item.max_drawdown_pct for item in metrics]
    assumptions = (
        "Completed trade net PnL values are the only resampled observations.",
        "Resampling does not fabricate intraday paths, fills, or independent returns.",
        "Outputs are distributional diagnostics, not forecasts.",
    )
    return MonteCarloReport(
        config=config,
        mean_total_return_pct=(statistics.fmean(returns) if returns else None),
        median_total_return_pct=(statistics.median(returns) if returns else None),
        worst_total_return_pct=(min(returns) if returns else None),
        best_total_return_pct=(max(returns) if returns else None),
        median_max_drawdown_pct=(statistics.median(drawdowns) if drawdowns else None),
        assumptions=assumptions,
    )


def regime_stability(
    result: BacktestResult, regime_by_entry_date: Mapping[date, str]
) -> RegimeStabilityReport:
    """Summarize trades only where a caller supplies timestamped regime labels."""

    grouped: dict[str, list[float]] = {}
    unclassified = 0
    for trade in result.trades:
        regime = regime_by_entry_date.get(trade.entry_date)
        if not isinstance(regime, str) or not regime.strip():
            unclassified += 1
            continue
        grouped.setdefault(regime.strip(), []).append(float(trade.net_pnl))
    segments = tuple(
        RegimePerformance(
            regime=regime,
            trade_count=len(pnl),
            total_net_pnl=sum(pnl),
            hit_rate_pct=(sum(value > 0 for value in pnl) / len(pnl) * 100.0 if pnl else None),
            expectancy=(statistics.fmean(pnl) if pnl else None),
        )
        for regime, pnl in sorted(grouped.items())
    )
    return RegimeStabilityReport(segments=segments, unclassified_trade_count=unclassified)


def subperiod_stability(result: BacktestResult, *, periods: int = 4) -> SubperiodStabilityReport:
    if isinstance(periods, bool) or not 2 <= periods <= 24:
        raise RobustnessError("periods must be an integer from 2 through 24.")
    values = np.asarray([point.equity for point in result.equity_curve], dtype=float)
    if len(values) < periods or not np.isfinite(values).all():
        return SubperiodStabilityReport(period_returns_pct=(), positive_profit_concentration=None)
    chunks = np.array_split(values, periods)
    returns = tuple((chunk[-1] / chunk[0] - 1.0) * 100.0 for chunk in chunks if chunk[0] > 0)
    positive = [value for value in returns if value > 0]
    concentration = max(positive) / sum(positive) if positive and sum(positive) > 0 else None
    return SubperiodStabilityReport(
        period_returns_pct=returns,
        positive_profit_concentration=concentration,
        warning=(
            "Most positive subperiod profit is concentrated in one period."
            if concentration is not None and concentration > 0.5
            else None
        ),
    )


def assess_overfitting_risk(
    *,
    strategy: StrategySchema,
    backtest_metrics: BacktestMetrics,
    train_total_return_pct: float | None = None,
    validation_total_return_pct: float | None = None,
    final_test_total_return_pct: float | None = None,
    parameter_variants: ParameterSensitivityReport | None = None,
    cost_sensitivity: CostSensitivityReport | None = None,
    subperiod: SubperiodStabilityReport | None = None,
    refinement_generation: int = 0,
    minimum_trade_count: int = 10,
) -> OverfittingRiskReport:
    """Produce disclosed warning flags without inventing an optimisation score."""

    if isinstance(refinement_generation, bool) or refinement_generation < 0:
        raise RobustnessError("refinement_generation must be a non-negative integer.")
    flags: list[str] = []
    complexity = measure_strategy_complexity(strategy)
    complexity_score = (
        complexity.condition_count
        + complexity.unique_indicator_count
        + complexity.maximum_group_depth
        + complexity.numeric_parameter_count
    )
    if backtest_metrics.number_of_trades < minimum_trade_count:
        flags.append("LOW_TRADE_COUNT")
    if complexity_score >= 12:
        flags.append("HIGH_STRATEGY_COMPLEXITY")
    train_validation_gap = _metric_delta(train_total_return_pct, validation_total_return_pct)
    validation_test_gap = _metric_delta(validation_total_return_pct, final_test_total_return_pct)
    if train_validation_gap is not None and abs(train_validation_gap) > 15.0:
        flags.append("EXTREME_TRAIN_VALIDATION_GAP")
    if validation_test_gap is not None and abs(validation_test_gap) > 15.0:
        flags.append("EXTREME_VALIDATION_TEST_GAP")
    parameter_instability = None
    if parameter_variants is not None and parameter_variants.variants:
        parameter_instability = sum(item.strategy is None for item in parameter_variants.variants) / len(
            parameter_variants.variants
        )
        if parameter_instability > 0.5:
            flags.append("PARAMETER_INSTABILITY")
    cost_rate = None
    if cost_sensitivity is not None:
        viable = [item.viability for item in cost_sensitivity.outcomes if item.viability is not None]
        if viable:
            cost_rate = sum(not item.is_viable for item in viable) / len(viable)
            if cost_rate > 0.0:
                flags.append("TRANSACTION_COST_SENSITIVITY")
    concentration = None if subperiod is None else subperiod.positive_profit_concentration
    if concentration is not None and concentration > 0.5:
        flags.append("TEMPORAL_PROFIT_CONCENTRATION")
    if refinement_generation > 3:
        flags.append("EXCESSIVE_REFINEMENT_GENERATIONS")
    return OverfittingRiskReport(
        flags=tuple(flags),
        trade_count=backtest_metrics.number_of_trades,
        complexity_score=complexity_score,
        train_validation_gap_pct=train_validation_gap,
        validation_test_gap_pct=validation_test_gap,
        parameter_instability_rate=parameter_instability,
        cost_sensitivity_rate=cost_rate,
        profit_concentration=concentration,
        refinement_generation=refinement_generation,
        includes_final_test_evidence=final_test_total_return_pct is not None,
    )


def score_robustness(
    *,
    walk_forward_pass_rate: float | None,
    cost_sensitivity: CostSensitivityReport | None,
    trade_skip: TradeSkipReport | None,
    overfitting: OverfittingRiskReport | None,
) -> RobustnessScore:
    """A transparent, bounded score for ranking—not a model-generated number."""

    contributions: list[tuple[str, float]] = []
    if walk_forward_pass_rate is not None:
        contributions.append(("walk_forward_pass_rate", max(0.0, min(40.0, walk_forward_pass_rate * 40))))
    if cost_sensitivity is not None:
        viable = [item.viability for item in cost_sensitivity.outcomes if item.viability is not None]
        rate = 1.0 if not viable else sum(item.is_viable for item in viable) / len(viable)
        contributions.append(("cost_viability_rate", rate * 25.0))
    if trade_skip is not None and trade_skip.viability_survival_rate is not None:
        contributions.append(("trade_skip_survival_rate", trade_skip.viability_survival_rate * 20.0))
    if overfitting is not None:
        if overfitting.includes_final_test_evidence:
            raise RobustnessError(
                "Final-test evidence cannot contribute to a validation robustness score."
            )
        contributions.append(("overfitting_flag_penalty", -min(15.0, len(overfitting.flags) * 3.0)))
    raw_score = sum(value for _, value in contributions)
    return RobustnessScore(score=max(0.0, min(100.0, raw_score)), contributions=tuple(contributions))


__all__ = [
    "CostScenario",
    "CostSensitivityReport",
    "DeterministicRobustnessEvaluator",
    "EntryDelayReport",
    "MonteCarloConfig",
    "MonteCarloMethod",
    "MonteCarloReport",
    "OverfittingRiskReport",
    "ParameterPerturbation",
    "ParameterSensitivityReport",
    "ParameterTarget",
    "RobustnessError",
    "RobustnessReport",
    "RobustnessScore",
    "TradeSkipConfig",
    "TradeSkipReport",
    "assess_overfitting_risk",
    "generate_parameter_variants",
    "monte_carlo_trade_resampling",
    "regime_stability",
    "score_robustness",
    "subperiod_stability",
    "trade_skip_robustness",
]
