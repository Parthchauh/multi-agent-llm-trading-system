"""Unit tests for the explicit, look-ahead-safe operator registry."""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.exceptions import OperatorResolutionError, SignalAlignmentError
from strategies.operator_registry import OPERATOR_REGISTRY, apply_operator
from strategies.registry import SUPPORTED_OPERATORS


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)))


def test_registry_exactly_matches_schema_allow_list() -> None:
    assert set(OPERATOR_REGISTRY) == set(SUPPORTED_OPERATORS)


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        (">", [False, False, True]),
        (">=", [False, True, True]),
        ("<", [True, False, False]),
        ("<=", [True, True, False]),
        ("==", [False, True, False]),
    ],
)
def test_comparison_operators(operator: str, expected: list[bool]) -> None:
    result = apply_operator(operator, _series([1, 2, 3]), _series([2, 2, 2]))
    assert result.tolist() == expected
    assert result.dtype == bool


def test_crossover_uses_only_current_and_previous_bars() -> None:
    result = apply_operator(
        "crosses_above", _series([1, 3, 4, 1]), _series([2, 2, 2, 2])
    )
    assert result.tolist() == [False, True, False, False]


def test_crossunder_uses_only_current_and_previous_bars() -> None:
    result = apply_operator(
        "crosses_below", _series([3, 1, 0, 4]), _series([2, 2, 2, 2])
    )
    assert result.tolist() == [False, True, False, False]


def test_cross_with_nan_operand_is_false() -> None:
    result = apply_operator(
        "crosses_above", _series([float("nan"), 3]), _series([2, 2])
    )
    assert result.tolist() == [False, False]


def test_invalid_operator_raises_typed_error() -> None:
    with pytest.raises(OperatorResolutionError, match="Unsupported operator"):
        apply_operator("contains", _series([1]), _series([1]))


def test_misaligned_operands_raise_typed_error() -> None:
    lhs = pd.Series([1.0], index=pd.date_range("2024-01-01", periods=1))
    rhs = pd.Series([1.0], index=pd.date_range("2024-02-01", periods=1))
    with pytest.raises(SignalAlignmentError, match="different indexes"):
        apply_operator(">", lhs, rhs)
