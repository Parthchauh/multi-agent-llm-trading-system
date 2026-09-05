"""Typed failures for the deterministic Sprint 6 risk boundary.

This package deliberately has no dependency on LLM agents or orchestration.
Failures here describe invalid deterministic inputs or an unsupported request;
they are not prose opinions about whether a strategy is attractive.
"""

from __future__ import annotations


class RiskError(ValueError):
    """Base class for deterministic risk-domain errors."""


class RiskDataError(RiskError):
    """Raised when a backtest artifact cannot support a risk calculation."""


class InsufficientRiskDataError(RiskDataError):
    """Raised when causal market history is too short for a sizing input."""


class SizingError(RiskError):
    """Raised when a deterministic sizing policy cannot produce a decision."""


class UnsupportedRiskConstraintError(RiskError):
    """Raised for a risk constraint the current deterministic engine cannot enforce."""
