"""Bounded Sprint 6 lifecycle around the authoritative Sprint 4 graph.

The legacy ``TradingAgentsGraph`` is deliberately not imported here.  Initial
generation remains in :class:`Sprint4ResearchGraph`; every refinement is then
promoted through the same Sprint 1 validation and Sprint 3 compiler boundary
before its deterministic backtest is considered.  Review agents are strictly
interpretive and cannot override risk, viability, or acceptance decisions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backtesting.engine import BacktestEngine
from backtesting.models import BacktestConfig, BacktestResult
from data.research_split import ResearchDatasetSplit
from evaluation.gates import evaluate_viability
from evaluation.models import (
    CandidateEvaluation,
    EvaluationContext,
    EvaluationMetrics,
    EvaluationScope,
    GateResult,
    ViabilityPolicy,
)
from evaluation.research_metrics import measure_strategy_complexity
from risk import (
    RiskAssessment,
    RiskPolicy,
    SizingPolicy,
    derive_backtest_config,
    derive_position_size,
    derive_sizing_input_from_ohlcv,
    evaluate_risk,
)
from strategies.compiler import StrategyCompiler
from tradingagents.agents.sprint6_agents import (
    BearCase,
    BearResearcherAgent,
    BoundedDebateOrchestrator,
    BullCase,
    BullResearcherAgent,
    CriticReport,
    DebateSession,
    DebateSynthesizerAgent,
    PerformanceAnalysis,
    PerformanceAnalystAgent,
    RefinementAgent,
    RefinementProposalDraft,
    RiskManagerAgent,
    RiskReview,
    StrategyCriticAgent,
)
from tradingagents.graph.research_graph import Sprint4ResearchGraph
from tradingagents.graph.research_models import ResearchRunResult, ResearchRunStatus
from tradingagents.research import (
    RefinementAcceptancePolicy,
    RefinementAcceptanceResult,
    StrategyVersionRecord,
    assess_refinement_acceptance,
    create_initial_strategy_record,
    promote_refinement_strategy,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Sprint6RunStatus(str, Enum):
    """Terminal status of the bounded lifecycle, independent of legacy actions."""

    COMPLETED = "COMPLETED"
    FAILED_INITIAL_PIPELINE = "FAILED_INITIAL_PIPELINE"
    FAILED_SIZING = "FAILED_SIZING"
    FAILED_EXECUTION = "FAILED_EXECUTION"
    VIABILITY_REJECTED = "VIABILITY_REJECTED"
    RISK_REJECTED = "RISK_REJECTED"
    FAILED_REVIEW = "FAILED_REVIEW"
    FAILED_REFINEMENT_VALIDATION = "FAILED_REFINEMENT_VALIDATION"
    FAILED_REFINEMENT_EXECUTION = "FAILED_REFINEMENT_EXECUTION"
    REFINEMENT_REJECTED = "REFINEMENT_REJECTED"


class _ExecutionStageFailure(RuntimeError):
    """Internal, sanitized stage marker for deterministic execution failures."""

    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = stage
        self.cause_type = type(cause).__name__
        super().__init__(f"{stage} failed: {self.cause_type}")


class Sprint6Event(_FrozenStrictModel):
    """A compact sanitized lifecycle event; no provider payloads are retained."""

    timestamp: datetime
    stage: str = Field(min_length=1, max_length=80)
    outcome: str = Field(min_length=1, max_length=40)
    detail: str = Field(default="", max_length=500)


class Sprint6LifecycleConfig(_FrozenStrictModel):
    """Hard-bounded policy bundle for one controlled refinement lifecycle."""

    max_debate_rounds: int = Field(default=1, ge=0, le=3)
    max_refinement_rounds: int = Field(default=1, ge=0, le=5)
    viability_policy: ViabilityPolicy = Field(default_factory=ViabilityPolicy)
    risk_policy: RiskPolicy = Field(default_factory=RiskPolicy)
    sizing_policy: SizingPolicy | None = None
    refinement_acceptance_policy: RefinementAcceptancePolicy = Field(
        default_factory=RefinementAcceptancePolicy
    )

    @field_validator("max_debate_rounds", "max_refinement_rounds", mode="before")
    @classmethod
    def loop_limits_must_not_be_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Lifecycle loop limits must be integers, not boolean.")
        return value


class StrategyExecutionRecord(_FrozenStrictModel):
    """One deterministic validation-partition execution of a versioned strategy."""

    version: StrategyVersionRecord
    effective_backtest_config: BacktestConfig
    backtest_result: BacktestResult
    evaluation: CandidateEvaluation
    viability: GateResult
    risk_assessment: RiskAssessment


class Sprint6RunResult(_FrozenStrictModel):
    """Immutable evidence for an initial strategy and all bounded refinements."""

    status: Sprint6RunStatus
    initial_run: ResearchRunResult
    strategy_records: tuple[StrategyExecutionRecord, ...] = ()
    selected_strategy_id: str | None = None
    debate_sessions: tuple[DebateSession, ...] = ()
    risk_reviews: tuple[RiskReview, ...] = ()
    performance_analyses: tuple[PerformanceAnalysis, ...] = ()
    critic_reports: tuple[CriticReport, ...] = ()
    refinement_proposals: tuple[RefinementProposalDraft, ...] = ()
    refinement_acceptance: tuple[RefinementAcceptanceResult, ...] = ()
    events: tuple[Sprint6Event, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def selected_record(self) -> StrategyExecutionRecord | None:
        if self.selected_strategy_id is None:
            return None
        for record in self.strategy_records:
            if record.version.strategy_id == self.selected_strategy_id:
                return record
        return None


class Sprint6ResearchGraph:
    """Compose controlled S4 generation with bounded S6 risk/review/refinement.

    The first S4 run establishes a canonical validated strategy.  This class
    reruns that strategy with a deterministic pre-run sizing decision before
    treating any resulting metric as Sprint-6 authority.  The extra run keeps
    the protected S4 graph backward compatible while ensuring configured risk
    sizing governs the record considered by debate, review, and refinement.
    """

    def __init__(
        self,
        *,
        initial_graph: Sprint4ResearchGraph,
        config: Sprint6LifecycleConfig | None = None,
        compiler: StrategyCompiler | None = None,
        engine_factory: Callable[[BacktestConfig], Any] | None = None,
        bull_researcher: BullResearcherAgent | None = None,
        bear_researcher: BearResearcherAgent | None = None,
        debate_synthesizer: DebateSynthesizerAgent | None = None,
        risk_manager: RiskManagerAgent | None = None,
        performance_analyst: PerformanceAnalystAgent | None = None,
        critic: StrategyCriticAgent | None = None,
        refinement_agent: RefinementAgent | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.initial_graph = initial_graph
        self.config = config or Sprint6LifecycleConfig()
        self.compiler = compiler or StrategyCompiler()
        self.engine_factory = engine_factory or BacktestEngine
        self.bull_researcher = bull_researcher
        self.bear_researcher = bear_researcher
        self.debate_synthesizer = debate_synthesizer
        self.risk_manager = risk_manager
        self.performance_analyst = performance_analyst
        self.critic = critic
        self.refinement_agent = refinement_agent
        self.clock = clock
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        if self.config.max_debate_rounds and any(
            agent is None
            for agent in (
                self.bull_researcher,
                self.bear_researcher,
                self.debate_synthesizer,
            )
        ):
            raise ValueError("Configured debate rounds require Bull, Bear, and synthesis agents.")
        if self.config.max_refinement_rounds and any(
            agent is None
            for agent in (
                self.risk_manager,
                self.performance_analyst,
                self.critic,
                self.refinement_agent,
            )
        ):
            raise ValueError(
                "Configured refinement rounds require Risk Manager, Performance Analyst, "
                "Critic, and Refinement agents."
            )

    def _event(self, stage: str, outcome: str, detail: str = "") -> Sprint6Event:
        return Sprint6Event(
            timestamp=self.clock(), stage=stage, outcome=outcome, detail=detail
        )

    def run(
        self,
        *,
        symbol: str,
        dataset_split: ResearchDatasetSplit,
        backtest_config: BacktestConfig,
    ) -> Sprint6RunResult:
        """Run S4 generation once, then bounded deterministic Sprint-6 controls."""

        initial = self.initial_graph.run(
            symbol=symbol,
            dataset_split=dataset_split,
            backtest_config=backtest_config,
        )
        if initial.status is not ResearchRunStatus.COMPLETED or initial.strategy is None:
            return Sprint6RunResult(
                status=Sprint6RunStatus.FAILED_INITIAL_PIPELINE,
                initial_run=initial,
                events=(self._event("initial_pipeline", "failed", initial.status.value),),
                errors=(initial.status.value,),
            )

        try:
            root_version = create_initial_strategy_record(initial.strategy)
            root_execution = self._execute_version(
                root_version, dataset_split=dataset_split, base_config=backtest_config
            )
        except Exception as exc:
            stage = exc.stage if isinstance(exc, _ExecutionStageFailure) else "lineage"
            status = (
                Sprint6RunStatus.FAILED_SIZING
                if stage == "risk_sizing"
                else Sprint6RunStatus.FAILED_EXECUTION
            )
            return Sprint6RunResult(
                status=status,
                initial_run=initial,
                events=(self._event(stage, "failed", type(exc).__name__),),
                errors=(type(exc).__name__,),
            )

        records: list[StrategyExecutionRecord] = [root_execution]
        events: list[Sprint6Event] = [self._event("risk_sizing", "succeeded")]
        sessions: list[DebateSession] = []
        reviews: list[RiskReview] = []
        analyses: list[PerformanceAnalysis] = []
        critics: list[CriticReport] = []
        proposals: list[RefinementProposalDraft] = []
        acceptance_results: list[RefinementAcceptanceResult] = []
        if not root_execution.viability.is_viable:
            events.append(self._event("viability", "rejected"))
            return self._result(
                Sprint6RunStatus.VIABILITY_REJECTED,
                initial,
                records,
                root_execution,
                events,
                errors=("ViabilityRejected",),
                sessions=sessions,
                reviews=reviews,
                analyses=analyses,
                critics=critics,
                proposals=proposals,
                acceptance_results=acceptance_results,
            )
        if not root_execution.risk_assessment.passed:
            events.append(self._event("risk_assessment", "rejected"))
            return self._result(
                Sprint6RunStatus.RISK_REJECTED,
                initial,
                records,
                root_execution,
                events,
                errors=("RiskAssessmentRejected",),
                sessions=sessions,
                reviews=reviews,
                analyses=analyses,
                critics=critics,
                proposals=proposals,
                acceptance_results=acceptance_results,
            )

        selected = root_execution
        events.append(self._event("risk_assessment", "passed"))
        for refinement_round in range(1, self.config.max_refinement_rounds + 1):
            try:
                session, review, analysis, critic_report = self._review_strategy(
                    selected, initial=initial
                )
            except Exception as exc:
                events.append(self._event("review", "failed", type(exc).__name__))
                return self._result(
                    Sprint6RunStatus.FAILED_REVIEW,
                    initial,
                    records,
                    selected,
                    events,
                    errors=(type(exc).__name__,),
                    sessions=sessions,
                    reviews=reviews,
                    analyses=analyses,
                    critics=critics,
                    proposals=proposals,
                    acceptance_results=acceptance_results,
                )
            sessions.append(session)
            reviews.append(review)
            analyses.append(analysis)
            critics.append(critic_report)
            events.append(self._event("review", "succeeded", f"round={refinement_round}"))

            try:
                assert self.refinement_agent is not None
                proposal = self.refinement_agent.propose(
                    selected.version.strategy_id,
                    selected.version.strategy,
                    critic_report,
                    risk_review=review,
                )
                child_version = promote_refinement_strategy(
                    selected.version,
                    proposal.revised_strategy,
                    revision_reason=proposal.revision_reason,
                )
                proposals.append(proposal)
            except Exception as exc:
                events.append(
                    self._event("refinement_validation", "failed", type(exc).__name__)
                )
                return self._result(
                    Sprint6RunStatus.FAILED_REFINEMENT_VALIDATION,
                    initial,
                    records,
                    selected,
                    events,
                    errors=(type(exc).__name__,),
                    sessions=sessions,
                    reviews=reviews,
                    analyses=analyses,
                    critics=critics,
                    proposals=proposals,
                    acceptance_results=acceptance_results,
                )

            try:
                child_execution = self._execute_version(
                    child_version, dataset_split=dataset_split, base_config=backtest_config
                )
            except Exception as exc:
                stage = exc.stage if isinstance(exc, _ExecutionStageFailure) else "execution"
                events.append(self._event(stage, "failed", type(exc).__name__))
                return self._result(
                    Sprint6RunStatus.FAILED_REFINEMENT_EXECUTION,
                    initial,
                    records,
                    selected,
                    events,
                    errors=(type(exc).__name__,),
                    sessions=sessions,
                    reviews=reviews,
                    analyses=analyses,
                    critics=critics,
                    proposals=proposals,
                    acceptance_results=acceptance_results,
                )

            records.append(child_execution)
            if not child_execution.viability.is_viable:
                events.append(self._event("viability", "rejected", "refinement"))
                return self._result(
                    Sprint6RunStatus.VIABILITY_REJECTED,
                    initial,
                    records,
                    selected,
                    events,
                    errors=("ViabilityRejected",),
                    sessions=sessions,
                    reviews=reviews,
                    analyses=analyses,
                    critics=critics,
                    proposals=proposals,
                    acceptance_results=acceptance_results,
                )
            if not child_execution.risk_assessment.passed:
                events.append(self._event("risk_assessment", "rejected", "refinement"))
                return self._result(
                    Sprint6RunStatus.RISK_REJECTED,
                    initial,
                    records,
                    selected,
                    events,
                    errors=("RiskAssessmentRejected",),
                    sessions=sessions,
                    reviews=reviews,
                    analyses=analyses,
                    critics=critics,
                    proposals=proposals,
                    acceptance_results=acceptance_results,
                )

            acceptance = assess_refinement_acceptance(
                parent=selected.version,
                child=child_execution.version,
                parent_evaluation=selected.evaluation,
                child_evaluation=child_execution.evaluation,
                child_gate=child_execution.viability,
                policy=self.config.refinement_acceptance_policy,
            )
            acceptance_results.append(acceptance)
            if not acceptance.accepted:
                events.append(self._event("refinement_acceptance", "rejected"))
                return self._result(
                    Sprint6RunStatus.REFINEMENT_REJECTED,
                    initial,
                    records,
                    selected,
                    events,
                    sessions=sessions,
                    reviews=reviews,
                    analyses=analyses,
                    critics=critics,
                    proposals=proposals,
                    acceptance_results=acceptance_results,
                )
            selected = child_execution
            events.append(self._event("refinement_acceptance", "accepted"))

        return self._result(
            Sprint6RunStatus.COMPLETED,
            initial,
            records,
            selected,
            events,
            sessions=sessions,
            reviews=reviews,
            analyses=analyses,
            critics=critics,
            proposals=proposals,
            acceptance_results=acceptance_results,
        )

    def _result(
        self,
        status: Sprint6RunStatus,
        initial: ResearchRunResult,
        records: list[StrategyExecutionRecord],
        selected: StrategyExecutionRecord,
        events: list[Sprint6Event],
        *,
        errors: tuple[str, ...] = (),
        sessions: list[DebateSession] | None = None,
        reviews: list[RiskReview] | None = None,
        analyses: list[PerformanceAnalysis] | None = None,
        critics: list[CriticReport] | None = None,
        proposals: list[RefinementProposalDraft] | None = None,
        acceptance_results: list[RefinementAcceptanceResult] | None = None,
    ) -> Sprint6RunResult:
        return Sprint6RunResult(
            status=status,
            initial_run=initial,
            strategy_records=tuple(records),
            selected_strategy_id=selected.version.strategy_id,
            debate_sessions=tuple(sessions or ()),
            risk_reviews=tuple(reviews or ()),
            performance_analyses=tuple(analyses or ()),
            critic_reports=tuple(critics or ()),
            refinement_proposals=tuple(proposals or ()),
            refinement_acceptance=tuple(acceptance_results or ()),
            events=tuple(events),
            errors=errors,
        )

    def _execute_version(
        self,
        version: StrategyVersionRecord,
        *,
        dataset_split: ResearchDatasetSplit,
        base_config: BacktestConfig,
    ) -> StrategyExecutionRecord:
        generation = dataset_split.generation_view()
        development = dataset_split.development_view()
        try:
            effective_config = self._effective_config(
                version, generation.context.feature_frame, base_config
            )
        except Exception as exc:
            raise _ExecutionStageFailure("risk_sizing", exc) from exc
        try:
            compiled = self.compiler.compile(
                version.strategy, development.validation.feature_frame
            )
        except Exception as exc:
            raise _ExecutionStageFailure("compilation", exc) from exc
        try:
            engine = self.engine_factory(effective_config)
            result = engine.run(
                development.validation.feature_frame,
                compiled,
                evaluation_start=development.validation.evaluation_start,
            )
        except Exception as exc:
            raise _ExecutionStageFailure("backtest", exc) from exc
        try:
            context = EvaluationContext(
                scope=EvaluationScope.VALIDATION,
                dataset_id=development.validation.fingerprint.partition_id,
                dataset_fingerprint=development.validation.fingerprint.data_sha256,
                backtest_config_fingerprint=_config_fingerprint(effective_config),
                metrics_version=result.metrics.metrics_version,
            )
            evaluation = CandidateEvaluation(
                candidate_id=version.strategy_id,
                strategy_hash=version.lineage.execution_fingerprint,
                context=context,
                metrics=EvaluationMetrics.from_backtest_metrics(result.metrics),
            )
            viability = evaluate_viability(evaluation, self.config.viability_policy)
            assessment = evaluate_risk(result, self.config.risk_policy)
        except Exception as exc:
            raise _ExecutionStageFailure("evaluation", exc) from exc
        return StrategyExecutionRecord(
            version=version,
            effective_backtest_config=effective_config,
            backtest_result=result,
            evaluation=evaluation,
            viability=viability,
            risk_assessment=assessment,
        )

    def _effective_config(
        self,
        version: StrategyVersionRecord,
        context_data: pd.DataFrame,
        base_config: BacktestConfig,
    ) -> BacktestConfig:
        position_caps = [
            value
            for value in (
                self.config.risk_policy.max_position_pct,
                self.config.risk_policy.max_portfolio_exposure_pct,
            )
            if value is not None
        ]
        strategy_position_pct = min(
            [version.strategy.position_sizing.value, *position_caps]
        )
        strategy_config = BacktestConfig.model_validate(
            {
                **base_config.model_dump(),
                "position_size_pct": strategy_position_pct,
                "maximum_holding_days": version.strategy.exit.maximum_holding_days,
            }
        )
        policy = self.config.sizing_policy
        if policy is None:
            return strategy_config
        risk_cap = min(
            value
            for value in (
                policy.maximum_position_pct,
                self.config.risk_policy.max_position_pct,
                self.config.risk_policy.max_portfolio_exposure_pct,
            )
            if value is not None
        )
        policy = policy.model_copy(update={"maximum_position_pct": risk_cap})
        input_ = derive_sizing_input_from_ohlcv(
            context_data,
            as_of_index=len(context_data) - 1,
            capital=strategy_config.initial_capital,
        )
        decision = derive_position_size(policy, input_)
        return derive_backtest_config(strategy_config, decision)

    def _review_strategy(
        self,
        execution: StrategyExecutionRecord,
        *,
        initial: ResearchRunResult,
    ) -> tuple[DebateSession, RiskReview, PerformanceAnalysis, CriticReport]:
        if initial.regime_assessment is None or initial.strategy_proposal is None:
            raise ValueError("Completed initial run lacks required structured regime/proposal artifacts.")
        if self.config.max_debate_rounds:
            assert self.bull_researcher is not None
            assert self.bear_researcher is not None
            assert self.debate_synthesizer is not None
            debate = BoundedDebateOrchestrator(
                self.bull_researcher,
                self.bear_researcher,
                self.debate_synthesizer,
                max_rounds=self.config.max_debate_rounds,
            ).run(
                execution.version.strategy,
                initial.regime_assessment,
                initial.strategy_proposal,
                self.config.risk_policy,
            )
            last_round = debate.rounds[-1]
            bull_case, bear_case = last_round.bull_case, last_round.bear_case
        else:
            debate = DebateSession()
            bull_case = BullCase(
                supporting_arguments=("Debate disabled by bounded lifecycle configuration.",),
                confidence=0.0,
            )
            bear_case = BearCase(
                objections=("Debate disabled by bounded lifecycle configuration.",),
                confidence=0.0,
            )
        assert self.risk_manager is not None
        assert self.performance_analyst is not None
        assert self.critic is not None
        review = self.risk_manager.review(
            execution.version.strategy,
            execution.risk_assessment,
            execution.backtest_result.metrics,
            initial.regime_assessment,
            bull_case,
            bear_case,
            debate_session=debate,
        )
        analysis = self.performance_analyst.analyze(
            execution.backtest_result.metrics, execution.evaluation
        )
        critic_report = self.critic.review(
            execution.version.strategy,
            bull_case,
            bear_case,
            review,
            execution.evaluation,
            initial.regime_assessment,
            measure_strategy_complexity(execution.version.strategy),
            performance_analysis=analysis,
            debate_session=debate,
        )
        return debate, review, analysis, critic_report


def _config_fingerprint(config: BacktestConfig) -> str:
    """Stable provenance identity for comparable validation evaluation records."""

    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "Sprint6Event",
    "Sprint6LifecycleConfig",
    "Sprint6ResearchGraph",
    "Sprint6RunResult",
    "Sprint6RunStatus",
    "StrategyExecutionRecord",
]
