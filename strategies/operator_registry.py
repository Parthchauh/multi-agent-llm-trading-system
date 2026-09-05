"""Explicit, vectorized operator registry used by the strategy compiler.

Every operator receives two equally indexed numeric Series and returns a
boolean Series on that same index.  Cross operations use only the current and
immediately previous row.  Missing operands always produce ``False``.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Mapping

import pandas as pd

from strategies.exceptions import OperatorResolutionError, SignalAlignmentError

OperatorFunction = Callable[[pd.Series, pd.Series], pd.Series]


def _valid_now(lhs: pd.Series, rhs: pd.Series) -> pd.Series:
    return lhs.notna() & rhs.notna()


def _gt(lhs: pd.Series, rhs: pd.Series) -> pd.Series:
    return _valid_now(lhs, rhs) & lhs.gt(rhs)


def _gte(lhs: pd.Series, rhs: pd.Series) -> pd.Series:
    return _valid_now(lhs, rhs) & lhs.ge(rhs)


def _lt(lhs: pd.Series, rhs: pd.Series) -> pd.Series:
    return _valid_now(lhs, rhs) & lhs.lt(rhs)


def _lte(lhs: pd.Series, rhs: pd.Series) -> pd.Series:
    return _valid_now(lhs, rhs) & lhs.le(rhs)


def _eq(lhs: pd.Series, rhs: pd.Series) -> pd.Series:
    return _valid_now(lhs, rhs) & lhs.eq(rhs)


def _crosses_above(lhs: pd.Series, rhs: pd.Series) -> pd.Series:
    lhs_previous = lhs.shift(1)
    rhs_previous = rhs.shift(1)
    valid = _valid_now(lhs, rhs) & _valid_now(lhs_previous, rhs_previous)
    return valid & lhs.gt(rhs) & lhs_previous.le(rhs_previous)


def _crosses_below(lhs: pd.Series, rhs: pd.Series) -> pd.Series:
    lhs_previous = lhs.shift(1)
    rhs_previous = rhs.shift(1)
    valid = _valid_now(lhs, rhs) & _valid_now(lhs_previous, rhs_previous)
    return valid & lhs.lt(rhs) & lhs_previous.ge(rhs_previous)


OPERATOR_REGISTRY: Mapping[str, OperatorFunction] = MappingProxyType(
    {
        ">": _gt,
        ">=": _gte,
        "<": _lt,
        "<=": _lte,
        "==": _eq,
        "crosses_above": _crosses_above,
        "crosses_below": _crosses_below,
    }
)


def resolve_operator(name: str) -> OperatorFunction:
    """Return an allow-listed operator or raise a typed resolution error."""
    try:
        return OPERATOR_REGISTRY[name]
    except KeyError as exc:
        raise OperatorResolutionError(
            f"Unsupported operator {name!r}; allowed operators are "
            f"{sorted(OPERATOR_REGISTRY)}."
        ) from exc


def apply_operator(name: str, lhs: pd.Series, rhs: pd.Series) -> pd.Series:
    """Apply an operator while enforcing exact index and boolean alignment."""
    if not lhs.index.equals(rhs.index):
        raise SignalAlignmentError(
            f"Operator {name!r} received operands with different indexes."
        )
    result = resolve_operator(name)(lhs, rhs)
    if not result.index.equals(lhs.index):
        raise SignalAlignmentError(
            f"Operator {name!r} changed the operand index during evaluation."
        )
    return result.fillna(False).astype(bool)
