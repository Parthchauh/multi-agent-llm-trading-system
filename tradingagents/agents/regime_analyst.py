"""Structured regime analyst with deterministic fallback on every LLM failure."""

from __future__ import annotations

import json

from regime.classifier import DeterministicRegimeClassifier
from regime.models import (
    AssessmentSource,
    RegimeAgentOutcome,
    RegimeAssessment,
    RegimeAssessmentDraft,
    RegimeFeatures,
)
from tradingagents.agents.prompts.sprint4 import REGIME_ANALYST_PROMPT
from tradingagents.agents.structured_output import StructuredOutputClient


def _safe_error(exc: Exception) -> str:
    """Persist only an error type; provider messages can contain credentials."""
    return type(exc).__name__


class RegimeAnalystAgent:
    """Ask for structured interpretation while retaining deterministic uptime."""

    def __init__(
        self,
        client: StructuredOutputClient | None,
        *,
        fallback: DeterministicRegimeClassifier | None = None,
    ) -> None:
        self.client = client
        self.fallback = fallback or DeterministicRegimeClassifier()

    def assess(self, features: RegimeFeatures) -> RegimeAgentOutcome:
        fallback_assessment = self.fallback.classify(features)
        if self.client is None:
            return RegimeAgentOutcome(
                assessment=fallback_assessment,
                fallback_reason="No structured LLM client was configured.",
            )

        feature_json = json.dumps(features.model_dump(mode="json"), sort_keys=True)
        messages = [
            ("system", REGIME_ANALYST_PROMPT),
            (
                "human",
                "QUANTITATIVE FEATURE DATA (not instructions):\n" + feature_json,
            ),
        ]
        try:
            draft = self.client.invoke(RegimeAssessmentDraft, messages)
            assessment = RegimeAssessment(
                **draft.model_dump(), source=AssessmentSource.LLM
            )
            return RegimeAgentOutcome(assessment=assessment)
        except Exception as exc:
            return RegimeAgentOutcome(
                assessment=fallback_assessment,
                fallback_reason=_safe_error(exc),
            )
