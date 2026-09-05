"""Deterministic benchmark strategies for Sprint 5.

The benchmarks in this module are deliberately small, declarative, and
repeatable.  Every technical baseline follows the same protected path as a
strategy proposed by an agent:

``raw definition -> StrategyValidator -> StrategyCompiler -> BacktestEngine``

Buy-and-hold is the sole exception to the declarative compiler path because it
has no indicator rule.  It is still evaluated exclusively by
``BacktestEngine`` through a one-shot, history-bounded signal provider; this
module never calculates portfolio returns itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from backtesting.engine import BacktestEngine
from backtesting.models import BacktestConfig, BacktestResult
from data.schema import validate_ohlcv
from strategies.compiler import StrategyCompiler
from strategies.schema import StrategySchema
from strategies.validator import StrategyValidator


class BaselineKind(str, Enum):
    """The fixed benchmark family available to every comparable experiment."""

    BUY_AND_HOLD = "buy_and_hold"
    SMA_CROSSOVER = "sma_crossover"
    RSI_MEAN_REVERSION = "rsi_mean_reversion"
    MOMENTUM = "momentum"


class BaselineDefinitionError(ValueError):
    """Raised only if a maintained baseline fails the canonical strategy gate."""


@dataclass(frozen=True)
class BaselineRun:
    """One benchmark result plus its auditable declarative definition.

    ``strategy`` is ``None`` for buy-and-hold because it intentionally has no
    indicator expression.  All other kinds expose the exact validated schema
    that was compiled for the run.
    """

    kind: BaselineKind
    result: BacktestResult
    strategy: StrategySchema | None


@dataclass(frozen=True)
class BuyAndHoldSignalProvider:
    """One entry signal at a known evaluation bar, with no discretionary exit.

    The provider never receives or stores the full market frame.  It accepts
    an entry only when the engine supplies the historical window ending at the
    requested bar, so it cannot use unseen bars to alter the signal.  The
    engine provides next-bar execution and end-of-data accounting.
    """

    entry_index: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.entry_index, bool)
            or not isinstance(self.entry_index, int)
            or self.entry_index < 0
        ):
            raise ValueError("entry_index must be a non-negative integer.")

    def should_enter(self, current_index: int, data: pd.DataFrame) -> bool:
        return (
            current_index == self.entry_index
            and current_index >= 0
            and len(data) == current_index + 1
        )

    def should_exit(self, current_index: int, data: pd.DataFrame) -> bool:
        return False

    def get_stop_price(
        self,
        entry_price: float,
        current_index: int,
        data: pd.DataFrame,
    ) -> None:
        return None

    def get_take_profit_price(
        self,
        entry_price: float,
        stop_price: float | None,
        current_index: int,
        data: pd.DataFrame,
    ) -> None:
        return None


# Schema v1 requires an explicit stop and target.  These fixed, published
# values keep benchmark mechanics transparent while allowing exit rules—not
# an unreported manual PnL calculation—to remain the primary baseline logic.
_BASELINE_STOP_LOSS_PCT = 20.0
_BASELINE_TAKE_PROFIT_PCT = 40.0


def build_sma_crossover_strategy(config: BacktestConfig) -> StrategySchema:
    """Build the validated 20/50-day SMA crossover benchmark."""
    return _validated_strategy(
        _strategy_payload(
            config=config,
            name="Deterministic SMA 20/50 Crossover",
            description=(
                "Enter when the 20-day SMA crosses above the 50-day SMA; "
                "exit on the reciprocal cross."
            ),
            entry={
                "logic": "ALL",
                "conditions": [
                    {
                        "indicator": "sma_20",
                        "operator": "crosses_above",
                        "value": "sma_50",
                    }
                ],
            },
            exit={
                "logic": "ALL",
                "conditions": [
                    {
                        "indicator": "sma_20",
                        "operator": "crosses_below",
                        "value": "sma_50",
                    }
                ],
            },
        )
    )


def build_rsi_mean_reversion_strategy(config: BacktestConfig) -> StrategySchema:
    """Build the validated RSI(14) 30/70 mean-reversion benchmark."""
    return _validated_strategy(
        _strategy_payload(
            config=config,
            name="Deterministic RSI 30/70 Mean Reversion",
            description=(
                "Enter when the causal RSI(14) is at or below 30; exit when "
                "it is at or above 70."
            ),
            entry={
                "logic": "ALL",
                "conditions": [
                    {"indicator": "rsi_14", "operator": "<=", "value": 30.0}
                ],
            },
            exit={
                "logic": "ALL",
                "conditions": [
                    {"indicator": "rsi_14", "operator": ">=", "value": 70.0}
                ],
            },
        )
    )


def build_momentum_strategy(config: BacktestConfig) -> StrategySchema:
    """Build a validated, trend-filtered momentum benchmark.

    Momentum is defined transparently as price and fast trend agreement:
    ``close > sma_50`` and ``sma_20 > sma_50``.  The position exits when
    either relation reverses.
    """
    return _validated_strategy(
        _strategy_payload(
            config=config,
            name="Deterministic Trend-Filtered Momentum",
            description=(
                "Enter when close and the 20-day SMA are both above the "
                "50-day SMA; exit when either condition reverses."
            ),
            entry={
                "logic": "ALL",
                "conditions": [
                    {"indicator": "close", "operator": ">", "value": "sma_50"},
                    {"indicator": "sma_20", "operator": ">", "value": "sma_50"},
                ],
            },
            exit={
                "logic": "ANY",
                "conditions": [
                    {"indicator": "close", "operator": "<", "value": "sma_50"},
                    {"indicator": "sma_20", "operator": "<", "value": "sma_50"},
                ],
            },
        )
    )


def run_baseline(
    kind: BaselineKind | str,
    data: pd.DataFrame,
    config: BacktestConfig,
    *,
    evaluation_start: pd.Timestamp | datetime | None = None,
) -> BaselineRun:
    """Run one deterministic benchmark over canonical OHLCV data.

    ``data``, ``config``, and ``evaluation_start`` are shared verbatim by
    every baseline through :func:`run_all_baselines`, which makes comparison
    windows and execution assumptions explicit and equivalent.
    """
    baseline_kind = BaselineKind(kind)
    _require_backtest_config(config)
    market_data = validate_ohlcv(data)
    boundary, start_index = _resolve_evaluation_start(market_data, evaluation_start)

    if baseline_kind is BaselineKind.BUY_AND_HOLD:
        result = BacktestEngine(config).run(
            market_data,
            BuyAndHoldSignalProvider(entry_index=start_index),
            evaluation_start=boundary,
        )
        return BaselineRun(kind=baseline_kind, result=result, strategy=None)

    strategy = _strategy_builder_for(baseline_kind)(config)
    compiled = StrategyCompiler().compile(strategy, market_data)
    result = BacktestEngine(config).run(
        market_data,
        compiled,
        evaluation_start=boundary,
    )
    return BaselineRun(kind=baseline_kind, result=result, strategy=strategy)


def run_all_baselines(
    data: pd.DataFrame,
    config: BacktestConfig,
    *,
    evaluation_start: pd.Timestamp | datetime | None = None,
) -> tuple[BaselineRun, ...]:
    """Run all fixed baselines in a stable, documented order.

    No baseline is allowed a different sample window or execution
    configuration.  Callers that need a mapping can key this tuple by
    :attr:`BaselineRun.kind` without changing evaluation semantics.
    """
    return tuple(
        run_baseline(
            kind,
            data,
            config,
            evaluation_start=evaluation_start,
        )
        for kind in BaselineKind
    )


def _strategy_payload(
    *,
    config: BacktestConfig,
    name: str,
    description: str,
    entry: dict[str, Any],
    exit: dict[str, Any],
) -> dict[str, Any]:
    """Create only plain schema data; parsing remains centralized below."""
    return {
        "metadata": {
            "name": name,
            "description": description,
            "market_type": "equity",
            "timeframe": "1d",
        },
        "entry": {"long_conditions": entry},
        "exit": {
            "exit_conditions": exit,
            "maximum_holding_days": config.maximum_holding_days,
        },
        "stop_loss": {"type": "percentage", "value": _BASELINE_STOP_LOSS_PCT},
        "take_profit": {
            "type": "percentage",
            "value": _BASELINE_TAKE_PROFIT_PCT,
        },
        "position_sizing": {
            "type": "fixed_percentage",
            "value": config.position_size_pct,
        },
    }


def _validated_strategy(raw_strategy: dict[str, Any]) -> StrategySchema:
    """Enforce the canonical raw-dict validation boundary before compilation."""
    validation = StrategyValidator().validate_dict(raw_strategy)
    if not validation.is_valid:
        details = "; ".join(validation.errors) or "unknown validation error"
        raise BaselineDefinitionError(
            "A maintained deterministic baseline failed strategy validation: "
            f"{details}"
        )
    return StrategySchema.model_validate(raw_strategy)


def _strategy_builder_for(kind: BaselineKind):
    builders = {
        BaselineKind.SMA_CROSSOVER: build_sma_crossover_strategy,
        BaselineKind.RSI_MEAN_REVERSION: build_rsi_mean_reversion_strategy,
        BaselineKind.MOMENTUM: build_momentum_strategy,
    }
    try:
        return builders[kind]
    except KeyError as exc:
        raise ValueError(f"{kind.value!r} has no declarative strategy definition.") from exc


def _resolve_evaluation_start(
    market_data: pd.DataFrame,
    evaluation_start: pd.Timestamp | datetime | None,
) -> tuple[pd.Timestamp | None, int]:
    """Mirror the engine's exact-boundary rule before constructing buy-and-hold."""
    if evaluation_start is None:
        return None, 0
    boundary = pd.Timestamp(evaluation_start)
    if boundary not in market_data.index:
        raise ValueError("evaluation_start must identify an exact market-data bar.")
    location = market_data.index.get_loc(boundary)
    if not isinstance(location, int):
        raise ValueError("evaluation_start must identify exactly one bar.")
    return boundary, location


def _require_backtest_config(config: BacktestConfig) -> None:
    if not isinstance(config, BacktestConfig):
        raise TypeError("config must be a BacktestConfig instance.")


__all__ = [
    "BaselineDefinitionError",
    "BaselineKind",
    "BaselineRun",
    "BuyAndHoldSignalProvider",
    "build_momentum_strategy",
    "build_rsi_mean_reversion_strategy",
    "build_sma_crossover_strategy",
    "run_all_baselines",
    "run_baseline",
]
