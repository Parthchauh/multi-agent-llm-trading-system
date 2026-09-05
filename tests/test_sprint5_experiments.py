"""Tests for declarative Sprint 5 architecture experiment configurations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evaluation.experiments import ExperimentArchitecture, ExperimentConfig


def _config(**overrides) -> ExperimentConfig:
    payload = {
        "experiment_id": "benchmark-001",
        "architecture": ExperimentArchitecture.SINGLE_AGENT,
        "provider": "fake",
        "model": "fake-model",
        "dataset_id": "synthetic-v1",
        "symbols": ("test",),
    }
    payload.update(overrides)
    return ExperimentConfig(**payload)


def test_single_agent_config_is_normalized_but_not_claimed_runnable() -> None:
    config = _config()
    assert config.symbols == ("TEST",)
    assert not config.is_runnable_in_current_sprint
    assert "no Sprint 5 experiment runner" in config.implementation_note


def test_no_debate_treatments_reject_debate_rounds() -> None:
    with pytest.raises(ValidationError, match="debate_rounds"):
        _config(debate_rounds=1)


def test_debate_treatment_requires_a_bounded_round() -> None:
    with pytest.raises(ValidationError, match="requires at least one debate"):
        _config(architecture=ExperimentArchitecture.MULTI_AGENT_DEBATE)
    config = _config(
        architecture=ExperimentArchitecture.MULTI_AGENT_DEBATE,
        debate_rounds=1,
    )
    assert not config.is_runnable_in_current_sprint


def test_full_treatment_requires_bounded_refinement() -> None:
    with pytest.raises(ValidationError, match="refinement round"):
        _config(
            architecture=ExperimentArchitecture.MULTI_AGENT_FULL,
            debate_rounds=1,
        )
    config = _config(
        architecture=ExperimentArchitecture.MULTI_AGENT_FULL,
        debate_rounds=2,
        refinement_rounds=1,
    )
    assert not config.is_runnable_in_current_sprint


def test_duplicate_or_boolean_config_values_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        _config(symbols=("TEST", "test"))
    with pytest.raises(ValidationError, match="not a boolean"):
        _config(seed=True)
