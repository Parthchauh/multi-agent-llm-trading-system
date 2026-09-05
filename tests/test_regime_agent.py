"""Structured-output and deterministic-fallback tests for the regime agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from regime.features import RegimeFeatureEngine
from regime.models import AssessmentSource
from tradingagents.agents.regime_analyst import RegimeAnalystAgent
from tests.test_regime_models import make_prices


class QueueStructuredClient:
    provider = "fake"
    model = "fake-structured"
    temperature = 0.0
    seed = 7

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[type[BaseModel], list[tuple[str, str]]]] = []

    def invoke(self, schema, messages):
        self.calls.append((schema, messages))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, schema):
            return item
        return schema.model_validate(item)


def _features():
    return RegimeFeatureEngine().snapshot(make_prices("bullish"), symbol="TEST")


def _valid_assessment() -> dict[str, Any]:
    return {
        "regime": "bullish_trend",
        "trend_state": "bullish",
        "volatility_state": "normal",
        "momentum_state": "positive",
        "confidence": 0.82,
        "reasoning": "Price and medium-term trend features are positive.",
        "preferred_strategy_characteristics": ["trend confirmation"],
        "avoid_strategy_characteristics": ["counter-trend entries"],
    }


def test_valid_structured_regime_response_is_used() -> None:
    client = QueueStructuredClient([_valid_assessment()])
    outcome = RegimeAnalystAgent(client).assess(_features())
    assert outcome.assessment.source is AssessmentSource.LLM
    assert outcome.fallback_reason is None
    assert "QUANTITATIVE FEATURE DATA" in client.calls[0][1][1][1]


def test_malformed_regime_response_uses_recorded_fallback() -> None:
    invalid = _valid_assessment() | {"confidence": 2.0}
    outcome = RegimeAnalystAgent(QueueStructuredClient([invalid])).assess(_features())
    assert outcome.assessment.source is AssessmentSource.DETERMINISTIC_FALLBACK
    assert outcome.fallback_reason is not None
    assert "ValidationError" in outcome.fallback_reason


def test_timeout_uses_deterministic_fallback() -> None:
    outcome = RegimeAnalystAgent(
        QueueStructuredClient([TimeoutError("provider timed out")])
    ).assess(_features())
    assert outcome.assessment.source is AssessmentSource.DETERMINISTIC_FALLBACK
    assert "TimeoutError" in (outcome.fallback_reason or "")


def test_provider_exception_uses_deterministic_fallback() -> None:
    outcome = RegimeAnalystAgent(
        QueueStructuredClient([RuntimeError("provider unavailable")])
    ).assess(_features())
    assert outcome.assessment.source is AssessmentSource.DETERMINISTIC_FALLBACK
    assert "RuntimeError" in (outcome.fallback_reason or "")


def test_missing_client_uses_explicit_fallback() -> None:
    outcome = RegimeAnalystAgent(None).assess(_features())
    assert outcome.assessment.source is AssessmentSource.DETERMINISTIC_FALLBACK
    assert outcome.fallback_reason == "No structured LLM client was configured."
