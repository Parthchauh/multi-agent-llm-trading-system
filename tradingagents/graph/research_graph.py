"""Controlled LangGraph orchestration around the deterministic quant core."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from langgraph.graph import END, START, StateGraph

from backtesting.engine import BacktestEngine
from backtesting.models import BacktestConfig
from data.research_split import ResearchDatasetSplit
from data.schema import validate_ohlcv
from regime.features import RegimeFeatureEngine
from strategies.compiler import StrategyCompiler
from tradingagents.agents.prompts.sprint4 import (
    REGIME_ANALYST_PROMPT_VERSION,
    STRATEGY_GENERATOR_PROMPT_VERSION,
)
from tradingagents.agents.regime_analyst import RegimeAnalystAgent
from tradingagents.agents.strategy_generator import StrategyGeneratorAgent
from tradingagents.graph.research_models import (
    ResearchEvent,
    ResearchRunMetadata,
    ResearchRunResult,
    ResearchRunStatus,
    SamplingMetadata,
    StrategyAttemptRecord,
)
from tradingagents.graph.research_state import ResearchState

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_error(exc: Exception) -> str:
    """Return a stable error code without persisting provider exception text."""
    return type(exc).__name__


def _sampling_metadata(
    client: Any | None,
    *,
    fallback_provider: str,
    fallback_model: str,
) -> SamplingMetadata:
    """Capture requested and observed client settings without assuming support."""
    provider = getattr(client, "provider", fallback_provider)
    model = getattr(client, "model", fallback_model)
    temperature = getattr(client, "temperature", None)
    seed = getattr(client, "seed", None)
    return SamplingMetadata(
        provider=provider,
        model=model,
        requested_temperature=getattr(client, "requested_temperature", temperature),
        effective_temperature=getattr(client, "effective_temperature", temperature),
        requested_seed=getattr(client, "requested_seed", seed),
        effective_seed=getattr(client, "effective_seed", seed),
    )


class Sprint4ResearchGraph:
    """Run regime analysis, strategy validation, compilation, and backtesting."""

    def __init__(
        self,
        *,
        regime_agent: RegimeAnalystAgent,
        strategy_generator: StrategyGeneratorAgent,
        feature_engine: RegimeFeatureEngine | None = None,
        compiler: StrategyCompiler | None = None,
        engine_factory: Callable[[BacktestConfig], Any] | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.regime_agent = regime_agent
        self.strategy_generator = strategy_generator
        self.feature_engine = feature_engine or RegimeFeatureEngine()
        self.compiler = compiler or StrategyCompiler()
        self.engine_factory = engine_factory or BacktestEngine
        self.clock = clock
        self.graph = self._build_graph()

    def _event(self, stage: str, outcome: str, detail: str = "") -> list[ResearchEvent]:
        event = ResearchEvent(
            timestamp=self.clock(), stage=stage, outcome=outcome, detail=detail
        )
        logger.info("research_event stage=%s outcome=%s", stage, outcome)
        return [event]

    def _build_graph(self):
        workflow = StateGraph(ResearchState)
        workflow.add_node("load_data", self._load_data_node)
        workflow.add_node("compute_regime", self._compute_regime_node)
        workflow.add_node("analyze_regime", self._analyze_regime_node)
        workflow.add_node("generate_strategy", self._generate_strategy_node)
        workflow.add_node("validate_strategy", self._validate_strategy_node)
        workflow.add_node("compile_strategy", self._compile_strategy_node)
        workflow.add_node("backtest", self._backtest_node)

        workflow.add_edge(START, "load_data")
        workflow.add_conditional_edges(
            "load_data",
            self._route_after_data,
            {"compute": "compute_regime", "end": END},
        )
        workflow.add_conditional_edges(
            "compute_regime",
            self._route_after_regime_features,
            {"analyze": "analyze_regime", "end": END},
        )
        workflow.add_edge("analyze_regime", "generate_strategy")
        workflow.add_conditional_edges(
            "generate_strategy",
            self._route_after_generation,
            {
                "validate": "validate_strategy",
                "retry": "generate_strategy",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "validate_strategy",
            self._route_after_validation,
            {
                "compile": "compile_strategy",
                "retry": "generate_strategy",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "compile_strategy",
            self._route_after_compilation,
            {"backtest": "backtest", "end": END},
        )
        workflow.add_edge("backtest", END)
        return workflow.compile()

    def _load_data_node(self, state: ResearchState) -> dict[str, Any]:
        try:
            context_data = validate_ohlcv(state["context_data"])
            evaluation_data = validate_ohlcv(state["evaluation_data"])
            return {
                "context_data": context_data,
                "evaluation_data": evaluation_data,
                "events": self._event(
                    "load_data",
                    "succeeded",
                    f"context_bars={len(context_data)}; evaluation_bars={len(evaluation_data)}",
                ),
            }
        except Exception as exc:
            error = _safe_error(exc)
            return {
                "status": ResearchRunStatus.FAILED_DATA,
                "errors": [error],
                "events": self._event("load_data", "failed", error),
            }

    @staticmethod
    def _route_after_data(state: ResearchState) -> str:
        return "end" if state.get("status") is ResearchRunStatus.FAILED_DATA else "compute"

    def _compute_regime_node(self, state: ResearchState) -> dict[str, Any]:
        try:
            features = self.feature_engine.snapshot(
                state["context_data"], symbol=state["symbol"]
            )
            return {
                "regime_features": features,
                "events": self._event("compute_regime", "succeeded"),
            }
        except Exception as exc:
            error = _safe_error(exc)
            return {
                "status": ResearchRunStatus.FAILED_REGIME,
                "errors": [error],
                "events": self._event("compute_regime", "failed", error),
            }

    @staticmethod
    def _route_after_regime_features(state: ResearchState) -> str:
        return "end" if state.get("status") is ResearchRunStatus.FAILED_REGIME else "analyze"

    def _analyze_regime_node(self, state: ResearchState) -> dict[str, Any]:
        outcome = self.regime_agent.assess(state["regime_features"])
        update: dict[str, Any] = {
            "regime_assessment": outcome.assessment,
            "events": self._event(
                "analyze_regime", "succeeded", f"source={outcome.assessment.source.value}"
            ),
        }
        if outcome.fallback_reason:
            update["warnings"] = [
                "Regime LLM fallback used: " + outcome.fallback_reason
            ]
        return update

    def _generate_strategy_node(self, state: ResearchState) -> dict[str, Any]:
        attempt = state.get("generation_attempt", 0) + 1
        feedback = state.get("validation_feedback", ())
        try:
            draft = self.strategy_generator.request_candidate(
                state["regime_assessment"],
                state["regime_features"],
                attempt=attempt,
                validation_errors=feedback,
            )
            return {
                "generation_attempt": attempt,
                "strategy_candidate": draft,
                "validation_feedback": (),
                "events": self._event(
                    "generate_strategy", "succeeded", f"attempt={attempt}"
                ),
            }
        except Exception as exc:
            error = _safe_error(exc)
            exhausted = attempt >= self.strategy_generator.max_attempts
            update: dict[str, Any] = {
                "generation_attempt": attempt,
                "strategy_candidate": None,
                "validation_feedback": (error,),
                "strategy_attempts": [
                    StrategyAttemptRecord(
                        attempt=attempt,
                        outcome="generation_failed",
                        feedback=(error,),
                    )
                ],
                "events": self._event(
                    "generate_strategy", "failed", f"attempt={attempt}; {error}"
                ),
            }
            if exhausted:
                update["status"] = ResearchRunStatus.FAILED_GENERATION
                update["errors"] = [error]
            return update

    def _route_after_generation(self, state: ResearchState) -> str:
        if state.get("strategy_candidate") is not None:
            return "validate"
        if state.get("status") is ResearchRunStatus.FAILED_GENERATION:
            return "end"
        return "retry"

    def _validate_strategy_node(self, state: ResearchState) -> dict[str, Any]:
        proposal, errors = self.strategy_generator.validate_candidate(
            state["strategy_candidate"], attempt=state["generation_attempt"]
        )
        if proposal is not None:
            return {
                "strategy_proposal": proposal,
                "validated_strategy": proposal.strategy,
                "validation_feedback": (),
                "strategy_attempts": [
                    StrategyAttemptRecord(
                        attempt=state["generation_attempt"], outcome="accepted"
                    )
                ],
                "events": self._event(
                    "validate_strategy", "succeeded", f"attempt={state['generation_attempt']}"
                ),
            }

        exhausted = state["generation_attempt"] >= self.strategy_generator.max_attempts
        audit_feedback = (f"validation_rejected:{len(errors)}",)
        update: dict[str, Any] = {
            "strategy_candidate": None,
            "validation_feedback": errors,
            "strategy_attempts": [
                StrategyAttemptRecord(
                    attempt=state["generation_attempt"],
                    outcome="validation_rejected",
                    feedback=audit_feedback,
                )
            ],
            "events": self._event(
                "validate_strategy",
                "rejected",
                f"attempt={state['generation_attempt']}; errors={len(errors)}",
            ),
        }
        if exhausted:
            update["status"] = ResearchRunStatus.FAILED_VALIDATION
            update["errors"] = list(audit_feedback)
        return update

    @staticmethod
    def _route_after_validation(state: ResearchState) -> str:
        if state.get("validated_strategy") is not None:
            return "compile"
        if state.get("status") is ResearchRunStatus.FAILED_VALIDATION:
            return "end"
        return "retry"

    def _compile_strategy_node(self, state: ResearchState) -> dict[str, Any]:
        try:
            compiled = self.compiler.compile(
                state["validated_strategy"], state["evaluation_data"]
            )
            return {
                "compiled_strategy": compiled,
                "events": self._event("compile_strategy", "succeeded"),
            }
        except Exception as exc:
            error = _safe_error(exc)
            return {
                "status": ResearchRunStatus.FAILED_COMPILATION,
                "errors": [error],
                "events": self._event("compile_strategy", "failed", error),
            }

    @staticmethod
    def _route_after_compilation(state: ResearchState) -> str:
        return (
            "end"
            if state.get("status") is ResearchRunStatus.FAILED_COMPILATION
            else "backtest"
        )

    def _backtest_node(self, state: ResearchState) -> dict[str, Any]:
        effective_config = state["backtest_config"]
        try:
            strategy = state["validated_strategy"]
            base_config = state["backtest_config"]
            effective_config = BacktestConfig.model_validate(
                {
                    **base_config.model_dump(),
                    "position_size_pct": strategy.position_sizing.value,
                    "maximum_holding_days": strategy.exit.maximum_holding_days,
                }
            )
            engine = self.engine_factory(effective_config)
            result = engine.run(
                state["evaluation_data"],
                state["compiled_strategy"],
                evaluation_start=state["evaluation_start"],
            )
            return {
                "backtest_config": effective_config,
                "backtest_result": result,
                "status": ResearchRunStatus.COMPLETED,
                "events": self._event("backtest", "succeeded"),
            }
        except Exception as exc:
            error = _safe_error(exc)
            return {
                "backtest_config": effective_config,
                "status": ResearchRunStatus.FAILED_BACKTEST,
                "errors": [error],
                "events": self._event("backtest", "failed", error),
            }

    def run(
        self,
        *,
        symbol: str,
        dataset_split: ResearchDatasetSplit,
        backtest_config: BacktestConfig,
    ) -> ResearchRunResult:
        generation = dataset_split.generation_view()
        development = dataset_split.development_view()
        context_data = generation.context.observations
        evaluation_data = development.validation.feature_frame
        split_id = ":".join(
            fingerprint.partition_id
            for fingerprint in dataset_split.partition_fingerprints
        )
        initial: ResearchState = {
            "symbol": symbol.strip().upper(),
            "context_data": context_data,
            "evaluation_data": evaluation_data,
            "evaluation_start": development.validation.evaluation_start,
            "split_id": split_id,
            "backtest_config": backtest_config,
            "generation_attempt": 0,
            "validation_feedback": (),
            "strategy_attempts": [],
            "status": ResearchRunStatus.RUNNING,
            "errors": [],
            "warnings": [],
            "events": [],
        }
        try:
            state = self.graph.invoke(
                initial,
                config={
                    "recursion_limit": max(
                        25, self.strategy_generator.max_attempts * 4 + 10
                    )
                },
            )
        except Exception as exc:
            state = dict(initial)
            state["status"] = ResearchRunStatus.FAILED_PIPELINE
            state["errors"] = [_safe_error(exc)]

        scored = development.validation.observations
        data_start = scored.index[0].to_pydatetime()
        data_end = scored.index[-1].to_pydatetime()
        regime_sampling = _sampling_metadata(
            self.regime_agent.client,
            fallback_provider="deterministic",
            fallback_model="deterministic-fallback",
        )
        strategy_sampling = _sampling_metadata(
            self.strategy_generator.client,
            fallback_provider="unknown",
            fallback_model="unknown",
        )
        metadata = ResearchRunMetadata(
            timestamp=self.clock(),
            symbol=symbol.strip().upper(),
            data_start=data_start,
            data_end=data_end,
            regime_provider=regime_sampling.provider,
            regime_model=regime_sampling.model,
            strategy_provider=strategy_sampling.provider,
            strategy_model=strategy_sampling.model,
            temperature=strategy_sampling.effective_temperature,
            seed=strategy_sampling.effective_seed,
            regime_sampling=regime_sampling,
            strategy_sampling=strategy_sampling,
            regime_prompt_version=REGIME_ANALYST_PROMPT_VERSION,
            strategy_prompt_version=STRATEGY_GENERATOR_PROMPT_VERSION,
            generation_attempts=state.get("generation_attempt", 0),
            research_split_id=state.get("split_id", split_id),
            decision_timestamp=generation.decision_timestamp,
            backtest_config=state.get("backtest_config", backtest_config),
        )
        status = state.get("status", ResearchRunStatus.FAILED_PIPELINE)
        if status is ResearchRunStatus.RUNNING:
            status = ResearchRunStatus.FAILED_PIPELINE
        return ResearchRunResult(
            status=status,
            regime_features=state.get("regime_features"),
            regime_assessment=state.get("regime_assessment"),
            strategy=state.get("validated_strategy"),
            strategy_proposal=state.get("strategy_proposal"),
            strategy_attempts=tuple(state.get("strategy_attempts", [])),
            backtest_result=state.get("backtest_result"),
            errors=tuple(state.get("errors", [])),
            warnings=tuple(state.get("warnings", [])),
            events=tuple(state.get("events", [])),
            metadata=metadata,
        )
