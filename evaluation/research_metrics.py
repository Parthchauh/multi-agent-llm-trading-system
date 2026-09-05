"""Deterministic metrics for evaluating the research system itself."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from regime.models import AssessmentSource
from strategies.schema import ConditionGroup, StrategySchema


class StrategyComplexity(BaseModel):
    """Structural complexity diagnostics; they are not an optimization target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    condition_count: int = Field(ge=0)
    unique_indicator_count: int = Field(ge=0)
    maximum_group_depth: int = Field(ge=0)
    numeric_parameter_count: int = Field(ge=0)


class FailureCount(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str = Field(min_length=1)
    count: int = Field(ge=1)


class ResearchSystemMetrics(BaseModel):
    """Aggregate, transparent statistics for completed and failed research runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_count: int = Field(ge=0)
    completed_run_count: int = Field(ge=0)
    valid_strategy_count: int = Field(ge=0)
    compiled_strategy_count: int = Field(ge=0)
    valid_strategy_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    compilation_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    backtest_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    failure_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    unique_execution_strategy_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    average_condition_count: float | None = Field(default=None, ge=0.0)
    average_generation_attempts: float | None = Field(default=None, ge=0.0)
    regime_fallback_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    strategy_generation_call_count: int = Field(ge=0)
    token_usage: int | None = Field(
        default=None,
        description="None when provider telemetry was not retained in the run artifact.",
    )
    failure_counts: tuple[FailureCount, ...] = Field(default_factory=tuple)


def strategy_execution_fingerprint(strategy: StrategySchema) -> str:
    """Hash executable strategy semantics without descriptive metadata.

    Names and prose must not make otherwise identical strategies appear unique.
    """
    payload = strategy.model_dump(mode="json")
    payload.pop("metadata", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def measure_strategy_complexity(strategy: StrategySchema) -> StrategyComplexity:
    """Count declarative structure without inferring economic quality."""
    indicators: set[str] = set()
    condition_count = 0
    numeric_parameters = 3  # stop loss, take profit, and position sizing

    def visit(group: ConditionGroup, depth: int) -> int:
        nonlocal condition_count, numeric_parameters
        maximum = depth
        for item in group.conditions:
            if isinstance(item, ConditionGroup):
                maximum = max(maximum, visit(item, depth + 1))
                continue
            condition_count += 1
            indicators.add(item.indicator)
            if isinstance(item.value, str):
                indicators.add(item.value)
            else:
                numeric_parameters += 1
        return maximum

    maximum_depth = max(
        visit(strategy.entry.long_conditions, 1),
        visit(strategy.exit.exit_conditions, 1),
    )
    if strategy.exit.maximum_holding_days is not None:
        numeric_parameters += 1
    return StrategyComplexity(
        condition_count=condition_count,
        unique_indicator_count=len(indicators),
        maximum_group_depth=maximum_depth,
        numeric_parameter_count=numeric_parameters,
    )


def summarize_research_runs(runs: Iterable[Any]) -> ResearchSystemMetrics:
    """Summarize run artifacts without importing or controlling their graph.

    The function intentionally relies on the small public run-artifact shape
    (``status``, ``strategy``, ``events``, ``metadata`` and
    ``regime_assessment``), which keeps the deterministic evaluation package
    decoupled from LangGraph and provider clients.
    """
    materialized = tuple(runs)
    run_count = len(materialized)
    statuses = [_enum_value(getattr(run, "status", "UNKNOWN")) for run in materialized]
    completed = sum(status == "COMPLETED" for status in statuses)
    strategies = [
        strategy
        for run in materialized
        if isinstance((strategy := getattr(run, "strategy", None)), StrategySchema)
    ]
    valid_count = len(strategies)
    compiled_count = sum(
        any(
            getattr(event, "stage", None) == "compile_strategy"
            and getattr(event, "outcome", None) == "succeeded"
            for event in getattr(run, "events", ())
        )
        for run in materialized
    )
    attempted_backtests = sum(
        any(getattr(event, "stage", None) == "backtest" for event in getattr(run, "events", ()))
        for run in materialized
    )
    fallback_count = sum(
        _enum_value(getattr(getattr(run, "regime_assessment", None), "source", None))
        == AssessmentSource.DETERMINISTIC_FALLBACK.value
        for run in materialized
    )
    complexities = [measure_strategy_complexity(strategy) for strategy in strategies]
    fingerprints = {strategy_execution_fingerprint(strategy) for strategy in strategies}
    generation_calls = sum(
        int(getattr(getattr(run, "metadata", None), "generation_attempts", 0))
        for run in materialized
    )
    failures = Counter(status for status in statuses if status != "COMPLETED")

    return ResearchSystemMetrics(
        run_count=run_count,
        completed_run_count=completed,
        valid_strategy_count=valid_count,
        compiled_strategy_count=compiled_count,
        valid_strategy_rate=_rate(valid_count, run_count),
        compilation_success_rate=_rate(compiled_count, valid_count),
        backtest_success_rate=_rate(completed, attempted_backtests),
        failure_rate=_rate(run_count - completed, run_count),
        unique_execution_strategy_rate=_rate(len(fingerprints), valid_count),
        average_condition_count=(
            round(sum(item.condition_count for item in complexities) / valid_count, 4)
            if valid_count
            else None
        ),
        average_generation_attempts=(
            round(generation_calls / run_count, 4) if run_count else None
        ),
        regime_fallback_rate=_rate(fallback_count, run_count),
        strategy_generation_call_count=generation_calls,
        failure_counts=tuple(
            FailureCount(status=status, count=count)
            for status, count in sorted(failures.items())
        ),
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 4)


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))
