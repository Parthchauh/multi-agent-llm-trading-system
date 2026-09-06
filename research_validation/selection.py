"""Validation-only candidate selection and one-time final-test execution."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backtesting.engine import BacktestEngine
from backtesting.models import BacktestConfig, BacktestResult
from data.research_split import FrozenStrategySelection
from evaluation.models import CandidateEvaluation, EvaluationScope, GateResult
from evaluation.research_metrics import measure_strategy_complexity, strategy_execution_fingerprint
from research_validation.partitions import DatasetPartitions, FinalTestAccess
from research_validation.robustness import RobustnessScore
from risk import RiskAssessment
from strategies.compiler import StrategyCompiler
from strategies.schema import StrategySchema


class SelectionError(ValueError):
    """Raised when a candidate or final test violates the OOS information policy."""


class FinalTestAlreadyExecutedError(SelectionError):
    """Raised when a final-test result would be re-run for an already frozen selection."""


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ValidationCandidate(_FrozenStrictModel):
    """All selection facts, each explicitly derived before final-test release."""

    strategy_id: str = Field(min_length=1, max_length=128)
    strategy: StrategySchema
    evaluation: CandidateEvaluation
    viability: GateResult
    risk_assessment: RiskAssessment
    robustness: RobustnessScore | None = None

    @field_validator("strategy_id")
    @classmethod
    def strategy_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("strategy_id must not be blank.")
        return value

    @field_validator("robustness")
    @classmethod
    def robustness_must_be_validation_only(
        cls, value: RobustnessScore | None
    ) -> RobustnessScore | None:
        if value is not None and not value.validation_only:
            raise ValueError("Validation selection cannot use final-test robustness evidence.")
        return value


class CandidateSelection(_FrozenStrictModel):
    selected_strategy_id: str = Field(min_length=1, max_length=128)
    selection_score: float
    selection_reason: str = Field(min_length=1, max_length=500)
    frozen_selection: FrozenStrategySelection


class FinalTestResult(_FrozenStrictModel):
    selection: FrozenStrategySelection
    final_test_partition_id: str
    result: BacktestResult


def _complexity_score(strategy: StrategySchema) -> int:
    complexity = measure_strategy_complexity(strategy)
    return (
        complexity.condition_count
        + complexity.unique_indicator_count
        + complexity.maximum_group_depth
        + complexity.numeric_parameter_count
    )


def select_validation_candidate(candidates: list[ValidationCandidate]) -> CandidateSelection:
    """Freeze the best viable/risk-passing validation candidate deterministically.

    No final-test object is accepted here.  Ranking uses validation performance,
    deterministic robustness, and a documented complexity penalty; ties are
    resolved by stable strategy ID rather than an LLM preference.
    """

    eligible: list[tuple[float, ValidationCandidate]] = []
    for candidate in candidates:
        if candidate.evaluation.context.scope is not EvaluationScope.VALIDATION:
            raise SelectionError("Candidate selection accepts validation evidence only.")
        if candidate.evaluation.candidate_id != candidate.strategy_id:
            raise SelectionError("Candidate evaluation identity must match strategy_id.")
        if candidate.viability.candidate_id != candidate.strategy_id:
            raise SelectionError("Viability identity must match strategy_id.")
        if not candidate.viability.is_viable or not candidate.risk_assessment.passed:
            continue
        sharpe = candidate.evaluation.metrics.sharpe_ratio or 0.0
        robustness = 0.0 if candidate.robustness is None else candidate.robustness.score / 100.0
        score = sharpe + robustness - _complexity_score(candidate.strategy) * 0.01
        eligible.append((score, candidate))
    if not eligible:
        raise SelectionError("No candidate passed deterministic validation and risk gates.")
    score, selected = sorted(eligible, key=lambda item: (-item[0], item[1].strategy_id))[0]
    fingerprint = strategy_execution_fingerprint(selected.strategy)
    return CandidateSelection(
        selected_strategy_id=selected.strategy_id,
        selection_score=score,
        selection_reason=(
            "Validation-only deterministic score: Sharpe + robustness score fraction "
            "- documented structural complexity penalty."
        ),
        frozen_selection=FrozenStrategySelection(
            strategy_id=selected.strategy_id,
            strategy_hash=fingerprint,
            frozen_at=datetime.now(timezone.utc),
        ),
    )


class FinalTestExecutor:
    """Release and execute each selected final-test identity exactly once per process."""

    def __init__(self, compiler: StrategyCompiler | None = None) -> None:
        self.compiler = compiler or StrategyCompiler()
        self._executed: set[tuple[str, str]] = set()

    def execute(
        self,
        *,
        partitions: DatasetPartitions,
        selection: CandidateSelection,
        strategy: StrategySchema,
        backtest_config: BacktestConfig,
    ) -> FinalTestResult:
        if selection.selected_strategy_id != selection.frozen_selection.strategy_id:
            raise SelectionError("Frozen selection ID does not match selected strategy ID.")
        if strategy_execution_fingerprint(strategy) != selection.frozen_selection.strategy_hash:
            raise SelectionError("Final-test strategy does not match the frozen strategy hash.")
        access: FinalTestAccess = partitions.release_final_test(selection.frozen_selection)
        key = (access.final_test.fingerprint.data_sha256, selection.frozen_selection.strategy_hash)
        if key in self._executed:
            raise FinalTestAlreadyExecutedError(
                "Final test was already executed for this frozen strategy and partition."
            )
        frame = access.final_test.observations
        compiled = self.compiler.compile(strategy, frame)
        result = BacktestEngine(backtest_config).run(frame, compiled)
        self._executed.add(key)
        return FinalTestResult(
            selection=selection.frozen_selection,
            final_test_partition_id=access.final_test.fingerprint.partition_id,
            result=result,
        )


__all__ = [
    "CandidateSelection",
    "FinalTestAlreadyExecutedError",
    "FinalTestExecutor",
    "FinalTestResult",
    "SelectionError",
    "ValidationCandidate",
    "select_validation_candidate",
]
