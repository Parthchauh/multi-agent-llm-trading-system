"""Regression tests for undeclared and adversarial LLM strategy fields."""

from __future__ import annotations

import math

import pytest

from backtesting.models import BacktestConfig
from strategies.validator import StrategyValidator
from tests.conftest_helpers import make_simple_strategy


def test_undeclared_strategy_code_field_is_rejected() -> None:
    payload = make_simple_strategy().model_dump()
    payload["code"] = "__import__('os').system('whoami')"
    result = StrategyValidator().validate_dict(payload)
    assert not result.is_valid
    assert "extra" in result.errors[0].lower()


def test_undeclared_nested_callable_field_is_rejected() -> None:
    payload = make_simple_strategy().model_dump()
    payload["entry"]["long_conditions"]["conditions"][0]["callable"] = "eval"
    result = StrategyValidator().validate_dict(payload)
    assert not result.is_valid


def test_malicious_comparison_rhs_is_rejected_before_compilation() -> None:
    payload = make_simple_strategy().model_dump()
    payload["entry"]["long_conditions"]["conditions"][0]["value"] = (
        "__import__('os')"
    )
    result = StrategyValidator().validate_dict(payload)
    assert not result.is_valid
    assert "not recognised" in result.errors[0]


@pytest.mark.parametrize("invalid_value", [True, math.nan, math.inf, -math.inf])
def test_non_finite_or_boolean_condition_threshold_is_rejected(invalid_value) -> None:
    payload = make_simple_strategy().model_dump()
    payload["entry"]["long_conditions"]["conditions"][0]["value"] = invalid_value
    result = StrategyValidator().validate_dict(payload)
    assert not result.is_valid


def test_boolean_position_size_is_rejected_instead_of_coerced() -> None:
    payload = make_simple_strategy().model_dump()
    payload["position_sizing"]["value"] = True
    assert not StrategyValidator().validate_dict(payload).is_valid


def test_boolean_backtest_configuration_is_rejected_instead_of_coerced() -> None:
    with pytest.raises(ValueError, match="must not be boolean"):
        BacktestConfig(initial_capital=True)
