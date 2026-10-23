"""Strict, reproducible experiment definitions for the research platform.

The configuration deliberately records the *conditions* of a comparison
(data window, requested LLM settings, deterministic cost assumptions, and
partition boundaries).  It does not itself execute an architecture: callers
must still inject an authoritative executor which preserves the validation,
compiler, backtest, risk, and holdout boundaries.
"""

from __future__ import annotations

import math
from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExperimentArchitecture(str, Enum):
    """Architectural treatments that can be compared once their roles exist."""

    SINGLE_AGENT = "single_agent"
    MULTI_AGENT_NO_DEBATE = "multi_agent_no_debate"
    MULTI_AGENT_DEBATE = "multi_agent_debate"
    MULTI_AGENT_RISK = "multi_agent_risk"
    MULTI_AGENT_FULL = "multi_agent_full"


class InformationSet(str, Enum):
    """Controlled evidence subsets for future ablation studies."""

    TECHNICAL_ONLY = "technical_only"
    FUNDAMENTAL_ONLY = "fundamental_only"
    NEWS_ONLY = "news_only"
    TECHNICAL_FUNDAMENTAL = "technical_fundamental"
    TECHNICAL_NEWS = "technical_news"
    ALL_ANALYSTS = "all_analysts"


class PromptVersion(BaseModel):
    """One prompt identity recorded in an experiment definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str = Field(min_length=1, max_length=80)
    version: str = Field(min_length=1, max_length=80)
    content_hash: str | None = Field(
        default=None,
        min_length=16,
        max_length=128,
        description="Optional immutable hash of the prompt content used for this role.",
    )


class _FrozenStrictConfig(BaseModel):
    """Common immutable, closed contract used in persisted experiment metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class TransactionCostConfig(_FrozenStrictConfig):
    """Deterministic execution assumptions that must match compared treatments."""

    initial_capital: float = Field(default=100_000.0, gt=0)
    commission_pct: float = Field(default=0.001, ge=0, le=1)
    slippage_pct: float = Field(default=0.0005, ge=0, le=1)
    position_size_pct: float = Field(default=10.0, gt=0, le=100)
    maximum_holding_days: int | None = Field(default=None, gt=0)

    @field_validator(
        "initial_capital",
        "commission_pct",
        "slippage_pct",
        "position_size_pct",
        "maximum_holding_days",
        mode="before",
    )
    @classmethod
    def numeric_values_must_not_be_booleans(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Transaction-cost configuration values must not be boolean.")
        return value

    @field_validator(
        "initial_capital", "commission_pct", "slippage_pct", "position_size_pct"
    )
    @classmethod
    def numeric_values_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Transaction-cost configuration values must be finite.")
        return value


class ChronologicalSplitConfig(_FrozenStrictConfig):
    """Recorded temporal boundaries for a sealed chronological experiment split.

    Actual leakage protection remains enforced by ``ResearchDatasetSplit``;
    this model makes the requested boundary policy reproducible in persisted
    experiment metadata.
    """

    context_end: date | None = None
    validation_end: date | None = None
    test_end: date | None = None

    @model_validator(mode="after")
    def boundaries_must_be_strictly_chronological(self) -> "ChronologicalSplitConfig":
        boundaries = (self.context_end, self.validation_end, self.test_end)
        supplied = tuple(boundary for boundary in boundaries if boundary is not None)
        if supplied and len(supplied) != len(boundaries):
            raise ValueError(
                "Chronological split boundaries must be supplied together or omitted together."
            )
        if supplied and not (self.context_end < self.validation_end < self.test_end):
            raise ValueError("Chronological split boundaries must be strictly increasing.")
        return self


class ModelRequestConfig(_FrozenStrictConfig):
    """Provider-neutral requested sampling/transport settings, never credentials."""

    temperature: float | None = Field(default=None, ge=0, le=2)
    timeout_seconds: int | None = Field(default=None, gt=0)
    max_retries: int | None = Field(default=None, ge=0, le=20)

    @field_validator("temperature", "timeout_seconds", "max_retries", mode="before")
    @classmethod
    def requested_settings_must_not_be_booleans(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Model request settings must not be boolean.")
        return value


class RobustnessConfig(_FrozenStrictConfig):
    """Requested deterministic robustness experiments for one research run.

    These flags are declarative: a runner records only experiments that its
    configured authoritative executor actually performs.  They are never a
    substitute for the Sprint 7 implementation or final-holdout protection.
    """

    walk_forward: bool = False
    cost_sensitivity: bool = False
    slippage_sensitivity: bool = False
    parameter_perturbation: bool = False
    entry_delay: bool = False
    trade_skipping: bool = False
    trade_bootstrap: bool = False
    regime_stability: bool = False


class ExperimentConfig(BaseModel):
    """Reproducible experiment definition, deliberately without a runner.

    Sprint 5 stores comparable treatments but does not claim that the
    debate, risk, or refinement architectures are executable yet. Those
    capabilities are introduced in Sprints 6–8.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(min_length=1, max_length=160)
    architecture: ExperimentArchitecture
    provider: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=160)
    dataset_id: str = Field(min_length=1, max_length=240)
    dataset_fingerprint: str | None = Field(default=None, min_length=16, max_length=128)
    start: date | None = None
    end: date | None = None
    symbols: tuple[str, ...] = Field(min_length=1, max_length=500)
    prompt_versions: tuple[PromptVersion, ...] = Field(default_factory=tuple)
    information_set: InformationSet = InformationSet.TECHNICAL_ONLY
    development_only: bool = False
    seed: int | None = None
    model_settings: ModelRequestConfig = Field(default_factory=ModelRequestConfig)
    transaction_costs: TransactionCostConfig = Field(default_factory=TransactionCostConfig)
    split_config: ChronologicalSplitConfig | None = None
    robustness_config: RobustnessConfig = Field(default_factory=RobustnessConfig)
    strategy_schema_version: str = Field(default="strategy-schema-v1", min_length=1, max_length=80)
    repetitions: int = Field(default=1, ge=1, le=10_000)
    debate_rounds: int = Field(default=0, ge=0, le=3)
    refinement_rounds: int = Field(default=0, ge=0, le=5)

    @field_validator("symbols")
    @classmethod
    def symbols_must_be_normalized_and_unique(
        cls, symbols: tuple[str, ...]
    ) -> tuple[str, ...]:
        normalized = tuple(symbol.strip().upper() for symbol in symbols)
        if any(not symbol for symbol in normalized):
            raise ValueError("Experiment symbols cannot be blank.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Experiment symbols must be unique.")
        return normalized

    @field_validator("seed", mode="before")
    @classmethod
    def seed_must_not_be_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Experiment seed must be an integer, not a boolean.")
        return value

    @model_validator(mode="after")
    def architecture_rounds_must_be_coherent(self) -> "ExperimentConfig":
        no_debate = {
            ExperimentArchitecture.SINGLE_AGENT,
            ExperimentArchitecture.MULTI_AGENT_NO_DEBATE,
        }
        if self.architecture in no_debate and self.debate_rounds != 0:
            raise ValueError("This architecture must set debate_rounds to zero.")
        if (
            self.architecture
            in {
                ExperimentArchitecture.MULTI_AGENT_DEBATE,
                ExperimentArchitecture.MULTI_AGENT_RISK,
                ExperimentArchitecture.MULTI_AGENT_FULL,
            }
            and self.debate_rounds == 0
        ):
            raise ValueError("A debate architecture requires at least one debate round.")
        if (
            self.architecture is ExperimentArchitecture.MULTI_AGENT_FULL
            and self.refinement_rounds == 0
        ):
            raise ValueError(
                "The full architecture requires at least one refinement round."
            )
        if (
            self.architecture is not ExperimentArchitecture.MULTI_AGENT_FULL
            and self.refinement_rounds != 0
        ):
            raise ValueError("Only the full architecture may configure refinement rounds.")
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("Experiment start must be earlier than end.")
        if self.split_config is not None:
            if self.start is not None and self.split_config.context_end < self.start:
                raise ValueError("The context boundary cannot precede the experiment start.")
            if self.end is not None and self.split_config.test_end >= self.end:
                raise ValueError("The final holdout must begin before the experiment end.")
        return self

    @property
    def is_runnable_in_current_sprint(self) -> bool:
        """Sprint 8 supplies a runner; callers still inject an authoritative executor."""
        return True

    @property
    def implementation_note(self) -> str:
        if self.architecture is ExperimentArchitecture.SINGLE_AGENT:
            return (
                "A controlled single-generator path can run through the Sprint 8 "
                "experiment runner with a configured authoritative executor."
            )
        return (
            "This treatment is runnable through the Sprint 8 experiment runner "
            "when its bounded authoritative architecture executor is configured."
        )
