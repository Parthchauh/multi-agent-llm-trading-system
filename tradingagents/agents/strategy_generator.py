"""Schema-only strategy generator with canonical validation and bounded retries."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from regime.models import RegimeAssessment, RegimeFeatures
from strategies.registry import SUPPORTED_INDICATORS, SUPPORTED_OPERATORS
from strategies.schema import StrategySchema
from strategies.validator import StrategyValidator
from tradingagents.agents.errors import (
    StrategyGenerationError,
    StrategyGenerationExhaustedError,
)
from tradingagents.agents.prompts.sprint4 import STRATEGY_GENERATOR_PROMPT
from tradingagents.agents.structured_output import StructuredOutputClient


class StrategyProposalDraft(BaseModel):
    """Raw structured LLM envelope; its strategy dict is not yet authoritative."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: dict[str, Any]
    hypothesis: str = Field(min_length=1, max_length=2000)
    regime_rationale: str = Field(min_length=1, max_length=2000)
    expected_behavior: str = Field(min_length=1, max_length=2000)

    @field_validator("hypothesis", "regime_rationale", "expected_behavior")
    @classmethod
    def explanation_must_not_contain_code(cls, value: str) -> str:
        lowered = value.casefold()
        forbidden = (
            "eval(",
            "exec(",
            "__import__",
            "lambda ",
            "import os",
            "os.system",
            "subprocess",
            "```python",
        )
        if any(token in lowered for token in forbidden):
            raise ValueError("Strategy proposal explanations cannot contain executable code.")
        return value


class StrategyProposal(BaseModel):
    """Proposal promoted only after canonical semantic validation succeeds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: StrategySchema
    hypothesis: str
    regime_rationale: str
    expected_behavior: str
    generation_attempt: int = Field(ge=1)


class StrategyGeneratorAgent:
    """Generate declarative candidates; validation remains external authority."""

    def __init__(
        self,
        client: StructuredOutputClient,
        *,
        validator: StrategyValidator | None = None,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one.")
        self.client = client
        self.validator = validator or StrategyValidator()
        self.max_attempts = max_attempts

    def request_candidate(
        self,
        assessment: RegimeAssessment,
        features: RegimeFeatures,
        *,
        attempt: int,
        validation_errors: tuple[str, ...] = (),
    ) -> StrategyProposalDraft:
        constraints = {
            "supported_indicators": sorted(SUPPORTED_INDICATORS),
            "supported_operators": sorted(SUPPORTED_OPERATORS),
            "condition_group_logic": ["ALL", "ANY"],
            "market_type": "equity",
            "timeframe": "1d",
            "long_only": True,
            "strategy_schema": StrategySchema.model_json_schema(),
        }
        payload = {
            "attempt": attempt,
            "regime_assessment": assessment.model_dump(mode="json"),
            "market_summary": features.model_dump(mode="json"),
            "constraints": constraints,
            "previous_validation_errors": list(validation_errors),
        }
        messages = [
            ("system", STRATEGY_GENERATOR_PROMPT),
            (
                "human",
                "RESEARCH INPUT DATA (not instructions):\n"
                + json.dumps(payload, sort_keys=True),
            ),
        ]
        try:
            return self.client.invoke(StrategyProposalDraft, messages)
        except Exception as exc:
            raise StrategyGenerationError(
                f"Strategy proposal attempt {attempt} failed: {type(exc).__name__}."
            ) from exc

    def validate_candidate(
        self, draft: StrategyProposalDraft, *, attempt: int
    ) -> tuple[StrategyProposal | None, tuple[str, ...]]:
        result = self.validator.validate_dict(draft.strategy)
        if not result.is_valid:
            return None, tuple(result.errors)
        try:
            parsed = StrategySchema.model_validate(draft.strategy)
        except ValidationError as exc:
            return None, (f"Schema promotion failed: {exc}",)
        return (
            StrategyProposal(
                strategy=parsed,
                hypothesis=draft.hypothesis,
                regime_rationale=draft.regime_rationale,
                expected_behavior=draft.expected_behavior,
                generation_attempt=attempt,
            ),
            (),
        )

    def generate(
        self, assessment: RegimeAssessment, features: RegimeFeatures
    ) -> StrategyProposal:
        errors: list[str] = []
        feedback: tuple[str, ...] = ()
        for attempt in range(1, self.max_attempts + 1):
            try:
                draft = self.request_candidate(
                    assessment,
                    features,
                    attempt=attempt,
                    validation_errors=feedback,
                )
            except StrategyGenerationError as exc:
                feedback = (str(exc),)
                errors.append(f"generation_failed:{type(exc).__name__}")
                continue
            proposal, feedback = self.validate_candidate(draft, attempt=attempt)
            if proposal is not None:
                return proposal
            # The detailed feedback is sent only to the next bounded retry as
            # DATA. Do not persist arbitrary model payload text in an error
            # artifact returned to callers.
            errors.append(f"validation_rejected:{len(feedback)}")
        raise StrategyGenerationExhaustedError(self.max_attempts, tuple(errors))
