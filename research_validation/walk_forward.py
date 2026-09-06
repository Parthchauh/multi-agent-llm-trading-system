"""Deterministic chronological walk-forward evaluation for validated strategies."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backtesting.engine import BacktestEngine
from backtesting.models import BacktestConfig
from data.schema import validate_ohlcv
from evaluation.gates import evaluate_viability
from evaluation.models import (
    CandidateEvaluation,
    EvaluationContext,
    EvaluationMetrics,
    EvaluationScope,
    GateResult,
    ViabilityPolicy,
)
from strategies.compiler import StrategyCompiler
from strategies.schema import StrategySchema


class WalkForwardError(ValueError):
    """Raised for malformed, overlapping, or non-chronological walk-forward work."""


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class WalkForwardConfig(_FrozenStrictModel):
    train_size: int = Field(ge=2)
    validation_size: int = Field(ge=2)
    step_size: int = Field(ge=1)
    expanding: bool = False

    @field_validator("train_size", "validation_size", "step_size", mode="before")
    @classmethod
    def sizes_must_not_be_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Walk-forward sizes must be integers, not boolean.")
        return value


@dataclass(frozen=True)
class WalkForwardFold:
    fold_number: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    _data: pd.DataFrame

    @property
    def train_frame(self) -> pd.DataFrame:
        return self._data.iloc[self.train_start : self.train_end].copy(deep=True)

    @property
    def validation_frame(self) -> pd.DataFrame:
        return self._data.iloc[self.validation_start : self.validation_end].copy(deep=True)


class WalkForwardPlan:
    """A fully deterministic sequence of non-overlapping validation folds."""

    @staticmethod
    def build(data: pd.DataFrame, config: WalkForwardConfig) -> tuple[WalkForwardFold, ...]:
        frame = validate_ohlcv(data)
        if not isinstance(config, WalkForwardConfig):
            raise WalkForwardError("config must be a WalkForwardConfig.")
        folds: list[WalkForwardFold] = []
        train_start = 0
        train_end = config.train_size
        fold_number = 1
        while train_end + config.validation_size <= len(frame):
            validation_start = train_end
            validation_end = validation_start + config.validation_size
            folds.append(
                WalkForwardFold(
                    fold_number=fold_number,
                    train_start=train_start,
                    train_end=train_end,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    _data=frame,
                )
            )
            fold_number += 1
            train_end += config.step_size
            if config.expanding:
                continue
            train_start += config.step_size
        if not folds:
            raise WalkForwardError("Not enough chronological data for one walk-forward fold.")
        return tuple(folds)


class WalkForwardFoldResult(_FrozenStrictModel):
    fold_number: int = Field(ge=1)
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    strategy_id: str
    evaluation: CandidateEvaluation
    viability: GateResult
    score: float | None = None


class WalkForwardAggregate(_FrozenStrictModel):
    metric: str
    mean: float | None
    median: float | None
    variance: float | None
    worst_fold: float | None
    best_fold: float | None
    pass_rate: float


class WalkForwardReport(_FrozenStrictModel):
    folds: tuple[WalkForwardFoldResult, ...] = Field(min_length=1)
    aggregate: WalkForwardAggregate


def _config_fingerprint(config: BacktestConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class WalkForwardEvaluator:
    """Run one already-selected schema through causally compiled folds only."""

    def __init__(self, compiler: StrategyCompiler | None = None) -> None:
        self.compiler = compiler or StrategyCompiler()

    def evaluate(
        self,
        *,
        strategy_id: str,
        strategy: StrategySchema,
        data: pd.DataFrame,
        walk_forward: WalkForwardConfig,
        backtest_config: BacktestConfig,
        viability_policy: ViabilityPolicy | None = None,
    ) -> WalkForwardReport:
        policy = viability_policy or ViabilityPolicy()
        folds = WalkForwardPlan.build(data, walk_forward)
        results: list[WalkForwardFoldResult] = []
        values: list[float] = []
        for fold in folds:
            train = fold.train_frame
            validation = fold.validation_frame
            # Feature warm-up is training history only; no later bar can enter compilation.
            feature_frame = pd.concat([train, validation])
            compiled = self.compiler.compile(strategy, feature_frame)
            result = BacktestEngine(backtest_config).run(
                feature_frame, compiled, evaluation_start=validation.index[0]
            )
            context = EvaluationContext(
                scope=EvaluationScope.VALIDATION,
                dataset_id=f"walk-forward:{fold.fold_number}",
                dataset_fingerprint=hashlib.sha256(
                    pd.util.hash_pandas_object(validation, index=True).to_numpy().tobytes()
                ).hexdigest(),
                backtest_config_fingerprint=_config_fingerprint(backtest_config),
                metrics_version=result.metrics.metrics_version,
            )
            evaluation = CandidateEvaluation(
                candidate_id=strategy_id,
                context=context,
                metrics=EvaluationMetrics.from_backtest_metrics(result.metrics),
            )
            viability = evaluate_viability(evaluation, policy)
            value = evaluation.metrics.total_return_pct
            if value is not None:
                values.append(float(value))
            results.append(
                WalkForwardFoldResult(
                    fold_number=fold.fold_number,
                    train_start=train.index[0].isoformat(),
                    train_end=train.index[-1].isoformat(),
                    validation_start=validation.index[0].isoformat(),
                    validation_end=validation.index[-1].isoformat(),
                    strategy_id=strategy_id,
                    evaluation=evaluation,
                    viability=viability,
                    score=value,
                )
            )
        return WalkForwardReport(
            folds=tuple(results),
            aggregate=WalkForwardAggregate(
                metric="total_return_pct",
                mean=(statistics.fmean(values) if values else None),
                median=(statistics.median(values) if values else None),
                variance=(statistics.pvariance(values) if len(values) > 1 else 0.0 if values else None),
                worst_fold=(min(values) if values else None),
                best_fold=(max(values) if values else None),
                pass_rate=sum(item.viability.is_viable for item in results) / len(results),
            ),
        )


__all__ = [
    "WalkForwardConfig",
    "WalkForwardError",
    "WalkForwardEvaluator",
    "WalkForwardFold",
    "WalkForwardFoldResult",
    "WalkForwardPlan",
    "WalkForwardReport",
]
