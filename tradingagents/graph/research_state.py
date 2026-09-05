"""Typed LangGraph state for the controlled Sprint 4 research workflow."""

from __future__ import annotations

import operator
from typing import Annotated

import pandas as pd
from typing_extensions import TypedDict

from backtesting.models import BacktestConfig, BacktestResult
from regime.models import RegimeAssessment, RegimeFeatures
from strategies.compiler import CompiledStrategy
from strategies.schema import StrategySchema
from tradingagents.agents.strategy_generator import (
    StrategyProposal,
    StrategyProposalDraft,
)
from tradingagents.graph.research_models import (
    ResearchEvent,
    ResearchRunStatus,
    StrategyAttemptRecord,
)


class ResearchState(TypedDict, total=False):
    symbol: str
    context_data: pd.DataFrame
    evaluation_data: pd.DataFrame
    evaluation_start: pd.Timestamp
    split_id: str
    backtest_config: BacktestConfig
    regime_features: RegimeFeatures
    regime_assessment: RegimeAssessment
    strategy_candidate: StrategyProposalDraft | None
    strategy_proposal: StrategyProposal
    validated_strategy: StrategySchema
    compiled_strategy: CompiledStrategy
    backtest_result: BacktestResult
    generation_attempt: int
    validation_feedback: tuple[str, ...]
    strategy_attempts: Annotated[list[StrategyAttemptRecord], operator.add]
    status: ResearchRunStatus
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    events: Annotated[list[ResearchEvent], operator.add]
