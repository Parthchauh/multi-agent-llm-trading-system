"""Structured Sprint 4 research-run status, metadata, events, and result."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backtesting.models import BacktestConfig, BacktestMetrics, BacktestResult
from regime.models import RegimeAssessment, RegimeFeatures
from strategies.schema import StrategySchema
from tradingagents.agents.strategy_generator import StrategyProposal


class ResearchRunStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED_DATA = "FAILED_DATA"
    FAILED_REGIME = "FAILED_REGIME"
    FAILED_GENERATION = "FAILED_GENERATION"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    FAILED_COMPILATION = "FAILED_COMPILATION"
    FAILED_BACKTEST = "FAILED_BACKTEST"
    FAILED_PIPELINE = "FAILED_PIPELINE"


class ResearchEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    stage: str = Field(min_length=1, max_length=80)
    outcome: str = Field(min_length=1, max_length=40)
    detail: str = Field(default="", max_length=1000)


class SamplingMetadata(BaseModel):
    """Requested versus observed sampling controls for one agent client."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    requested_temperature: float | None = None
    effective_temperature: float | None = None
    requested_seed: int | None = None
    effective_seed: int | None = None


class StrategyAttemptRecord(BaseModel):
    """Sanitized audit record for one bounded generation/validation attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt: int = Field(ge=1)
    outcome: Literal["generation_failed", "validation_rejected", "accepted"]
    feedback: tuple[str, ...] = Field(default_factory=tuple, max_length=20)


class ResearchRunMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    symbol: str
    data_start: datetime | None
    data_end: datetime | None
    regime_provider: str
    regime_model: str
    strategy_provider: str
    strategy_model: str
    temperature: float | None
    seed: int | None
    regime_sampling: SamplingMetadata
    strategy_sampling: SamplingMetadata
    regime_prompt_version: str
    strategy_prompt_version: str
    generation_attempts: int = Field(ge=0)
    research_split_id: str
    decision_timestamp: datetime
    compiler_version: str = "sprint3-1.0"
    backtest_config: BacktestConfig


class ResearchRunResult(BaseModel):
    """Compact run artifact; authoritative metrics remain inside BacktestResult."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ResearchRunStatus
    regime_features: RegimeFeatures | None = None
    regime_assessment: RegimeAssessment | None = None
    strategy: StrategySchema | None = None
    strategy_proposal: StrategyProposal | None = None
    strategy_attempts: tuple[StrategyAttemptRecord, ...] = ()
    backtest_result: BacktestResult | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    events: tuple[ResearchEvent, ...] = ()
    metadata: ResearchRunMetadata

    @property
    def backtest_metrics(self) -> BacktestMetrics | None:
        """Expose only metrics calculated by the deterministic backtester."""
        return None if self.backtest_result is None else self.backtest_result.metrics
