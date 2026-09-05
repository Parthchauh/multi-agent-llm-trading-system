"""
strategies/registry.py
======================
Centralized registry of supported indicators and operators.

This module is the single source of truth for every indicator name and
operator symbol that the strategy system accepts.  The schema and the
validator both import from here; the compiler (Sprint 3) will also use
this registry to map indicator names to DataFrame column computations.

Design constraints
------------------
* No imports from LLM, agent, or graph modules — zero circular-import risk.
* No executable code, no ``eval()``, no ``exec()``.
* All public names are plain Python sets / frozensets so they can be
  serialised, iterated, and tested without any special handling.
"""

from __future__ import annotations

from typing import FrozenSet

# ---------------------------------------------------------------------------
# Supported indicators
# ---------------------------------------------------------------------------

# Raw price / volume columns that come directly from OHLCV data.
PRICE_INDICATORS: FrozenSet[str] = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
)

# Simple Moving Averages
SMA_INDICATORS: FrozenSet[str] = frozenset(
    {
        "sma_20",
        "sma_50",
        "sma_200",
    }
)

# Exponential Moving Averages
EMA_INDICATORS: FrozenSet[str] = frozenset(
    {
        "ema_10",
        "ema_20",
        "ema_50",
    }
)

# Momentum indicators
MOMENTUM_INDICATORS: FrozenSet[str] = frozenset(
    {
        "rsi_14",
        "macd",
        "macd_signal",
    }
)

# Volatility indicators
VOLATILITY_INDICATORS: FrozenSet[str] = frozenset(
    {
        "atr_14",
        "bollinger_upper",
        "bollinger_middle",
        "bollinger_lower",
    }
)

# Volume-derived indicators
VOLUME_INDICATORS: FrozenSet[str] = frozenset(
    {
        "volume_sma_20",
    }
)

# Master set — union of every category above.
SUPPORTED_INDICATORS: FrozenSet[str] = (
    PRICE_INDICATORS
    | SMA_INDICATORS
    | EMA_INDICATORS
    | MOMENTUM_INDICATORS
    | VOLATILITY_INDICATORS
    | VOLUME_INDICATORS
)

# ---------------------------------------------------------------------------
# Supported operators
# ---------------------------------------------------------------------------

# Standard numeric comparison operators.
COMPARISON_OPERATORS: FrozenSet[str] = frozenset(
    {
        ">",
        ">=",
        "<",
        "<=",
        "==",
    }
)

# Crossover operators — the value side MUST be another indicator, never a
# number.  The validator enforces this constraint.
CROSSOVER_OPERATORS: FrozenSet[str] = frozenset(
    {
        "crosses_above",
        "crosses_below",
    }
)

# Master operator set.
SUPPORTED_OPERATORS: FrozenSet[str] = COMPARISON_OPERATORS | CROSSOVER_OPERATORS

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def is_supported_indicator(name: str) -> bool:
    """Return ``True`` if *name* is a known indicator or price column.

    Parameters
    ----------
    name:
        The indicator name to check (case-sensitive).

    Returns
    -------
    bool
        ``True`` when *name* is in :data:`SUPPORTED_INDICATORS`.

    Examples
    --------
    >>> is_supported_indicator("rsi_14")
    True
    >>> is_supported_indicator("fantasy_index")
    False
    """
    return name in SUPPORTED_INDICATORS


def is_supported_operator(operator: str) -> bool:
    """Return ``True`` if *operator* is a recognised condition operator.

    Parameters
    ----------
    operator:
        The operator symbol to check (e.g. ``">"``, ``"crosses_above"``).

    Returns
    -------
    bool
        ``True`` when *operator* is in :data:`SUPPORTED_OPERATORS`.

    Examples
    --------
    >>> is_supported_operator(">")
    True
    >>> is_supported_operator("approximately")
    False
    """
    return operator in SUPPORTED_OPERATORS


def is_crossover_operator(operator: str) -> bool:
    """Return ``True`` if *operator* is a crossover-type operator.

    Crossover operators require the value side of a condition to be another
    indicator name (not a numeric literal).

    Parameters
    ----------
    operator:
        The operator symbol to check.

    Returns
    -------
    bool
        ``True`` when *operator* is in :data:`CROSSOVER_OPERATORS`.
    """
    return operator in CROSSOVER_OPERATORS


def is_comparison_operator(operator: str) -> bool:
    """Return ``True`` if *operator* is a numeric comparison operator.

    Parameters
    ----------
    operator:
        The operator symbol to check.

    Returns
    -------
    bool
        ``True`` when *operator* is in :data:`COMPARISON_OPERATORS`.
    """
    return operator in COMPARISON_OPERATORS
