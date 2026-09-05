"""Bounded retry, whitelist, and adversarial tests for strategy generation."""

from __future__ import annotations

import json
from typing import Any

import pytest

from regime.classifier import DeterministicRegimeClassifier
from regime.features import RegimeFeatureEngine
from strategies.registry import SUPPORTED_INDICATORS, SUPPORTED_OPERATORS
from tradingagents.agents.errors import StrategyGenerationExhaustedError
from tradingagents.agents.strategy_generator import StrategyGeneratorAgent
from tests.conftest_helpers import make_simple_strategy
from tests.test_regime_agent import QueueStructuredClient
from tests.test_regime_models import make_prices


def _context():
    features = RegimeFeatureEngine().snapshot(make_prices("bullish"), symbol="TEST")
    return features, DeterministicRegimeClassifier().classify(features)


def _draft(strategy: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "strategy": strategy or make_simple_strategy().model_dump(mode="json"),
        "hypothesis": "Momentum persists when trend and RSI agree.",
        "regime_rationale": "The assessment indicates a positive trend.",
        "expected_behavior": "Enter only after confirmation and exit on weakness.",
    }


def test_valid_first_attempt_strategy() -> None:
    features, assessment = _context()
    proposal = StrategyGeneratorAgent(
        QueueStructuredClient([_draft()]), max_attempts=2
    ).generate(assessment, features)
    assert proposal.generation_attempt == 1
    assert proposal.strategy.metadata.market_type == "equity"


def test_invalid_indicator_retries_then_succeeds() -> None:
    features, assessment = _context()
    invalid = make_simple_strategy().model_dump(mode="json")
    invalid["entry"]["long_conditions"]["conditions"][0]["indicator"] = "alpha_999"
    client = QueueStructuredClient([_draft(invalid), _draft()])
    proposal = StrategyGeneratorAgent(client, max_attempts=2).generate(
        assessment, features
    )
    assert proposal.generation_attempt == 2
    retry_payload = json.loads(client.calls[1][1][1][1].split("\n", 1)[1])
    assert retry_payload["previous_validation_errors"]
    assert "alpha_999" in retry_payload["previous_validation_errors"][0]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["entry"]["long_conditions"]["conditions"][0].update(
            operator="eval"
        ),
        lambda payload: payload["position_sizing"].update(value=-10),
        lambda payload: payload["entry"]["long_conditions"]["conditions"][0].update(
            value="__import__('os')"
        ),
        lambda payload: payload.update(code="os.system('whoami')"),
    ],
)
def test_adversarial_strategy_is_rejected_before_promotion(mutate) -> None:
    features, assessment = _context()
    payload = make_simple_strategy().model_dump(mode="json")
    mutate(payload)
    generator = StrategyGeneratorAgent(
        QueueStructuredClient([_draft(payload)]), max_attempts=1
    )
    with pytest.raises(StrategyGenerationExhaustedError):
        generator.generate(assessment, features)


def test_retry_exhaustion_is_typed_and_bounded() -> None:
    features, assessment = _context()
    invalid = make_simple_strategy().model_dump(mode="json")
    invalid["entry"]["long_conditions"]["conditions"][0]["operator"] = "contains"
    client = QueueStructuredClient([_draft(invalid), _draft(invalid)])
    with pytest.raises(StrategyGenerationExhaustedError) as captured:
        StrategyGeneratorAgent(client, max_attempts=2).generate(assessment, features)
    assert captured.value.attempts == 2
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "responses",
    [
        [{"hypothesis": "missing fields"}],
        [TimeoutError("timeout")],
        [RuntimeError("provider failure")],
        [_draft() | {"authoritative_metrics": {"sharpe": 99}}],
    ],
)
def test_malformed_or_provider_failure_exhausts_safely(responses) -> None:
    features, assessment = _context()
    with pytest.raises(StrategyGenerationExhaustedError):
        StrategyGeneratorAgent(
            QueueStructuredClient(responses), max_attempts=1
        ).generate(assessment, features)


def test_prompt_uses_live_repository_whitelists() -> None:
    features, assessment = _context()
    client = QueueStructuredClient([_draft()])
    StrategyGeneratorAgent(client, max_attempts=1).generate(assessment, features)
    payload = json.loads(client.calls[0][1][1][1].split("\n", 1)[1])
    assert payload["constraints"]["supported_indicators"] == sorted(
        SUPPORTED_INDICATORS
    )
    assert payload["constraints"]["supported_operators"] == sorted(
        SUPPORTED_OPERATORS
    )
    assert payload["constraints"]["long_only"] is True


def test_executable_code_in_explanation_is_rejected() -> None:
    features, assessment = _context()
    malicious = _draft() | {"hypothesis": "import os; os.system('whoami')"}
    with pytest.raises(StrategyGenerationExhaustedError):
        StrategyGeneratorAgent(
            QueueStructuredClient([malicious]), max_attempts=1
        ).generate(assessment, features)
