"""Deterministic evaluation, benchmark, and research-system contracts.

Sprint 5 intentionally provides policies, metrics, and baseline utilities;
it does not introduce an experiment runner or let an LLM select a result.
"""

from evaluation.baselines import (
    BaselineDefinitionError,
    BaselineKind,
    BaselineRun,
    BuyAndHoldSignalProvider,
    run_all_baselines,
    run_baseline,
)
from evaluation.experiments import (
    ExperimentArchitecture,
    ExperimentConfig,
    InformationSet,
    PromptVersion,
)
from evaluation.gates import evaluate_viability
from evaluation.models import (
    CandidateEvaluation,
    EvaluationContext,
    EvaluationMetrics,
    EvaluationScope,
    RankingPolicy,
    ViabilityPolicy,
)
from evaluation.platform import (
    ExperimentReport,
    ExperimentRunner,
    FailureType,
    ReproducibilityRecord,
    ResearchEvent,
    ResearchPlatformError,
    ResearchRun,
    SQLiteResearchRepository,
    report_markdown,
)
from evaluation.ranking import rank_candidates, rank_viable_candidates
from evaluation.research_metrics import (
    measure_strategy_complexity,
    strategy_execution_fingerprint,
    summarize_research_runs,
)

__all__ = [
    "BaselineDefinitionError",
    "BaselineKind",
    "BaselineRun",
    "BuyAndHoldSignalProvider",
    "CandidateEvaluation",
    "EvaluationContext",
    "EvaluationMetrics",
    "EvaluationScope",
    "ExperimentArchitecture",
    "ExperimentConfig",
    "InformationSet",
    "PromptVersion",
    "RankingPolicy",
    "ViabilityPolicy",
    "evaluate_viability",
    "measure_strategy_complexity",
    "rank_candidates",
    "rank_viable_candidates",
    "run_all_baselines",
    "run_baseline",
    "strategy_execution_fingerprint",
    "summarize_research_runs",
    "ExperimentReport",
    "ExperimentRunner",
    "FailureType",
    "ResearchEvent",
    "ResearchPlatformError",
    "ResearchRun",
    "ReproducibilityRecord",
    "SQLiteResearchRepository",
    "report_markdown",
]
