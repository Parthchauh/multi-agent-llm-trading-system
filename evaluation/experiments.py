"""Declarative, non-executing experiment configurations for Sprint 5."""

from __future__ import annotations

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
    symbols: tuple[str, ...] = Field(min_length=1, max_length=500)
    prompt_versions: tuple[PromptVersion, ...] = Field(default_factory=tuple)
    information_set: InformationSet = InformationSet.TECHNICAL_ONLY
    seed: int | None = None
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
        return self

    @property
    def is_runnable_in_current_sprint(self) -> bool:
        """False until Sprint 8 supplies the persistent experiment runner."""
        return False

    @property
    def implementation_note(self) -> str:
        if self.architecture is ExperimentArchitecture.SINGLE_AGENT:
            return (
                "A controlled single-generator path exists, but no Sprint 5 "
                "experiment runner has been wired."
            )
        return (
            "This is a declarative treatment definition; its required agent "
            "roles are scheduled for later sprints."
        )
