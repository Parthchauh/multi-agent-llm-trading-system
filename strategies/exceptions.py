"""Typed failures raised by deterministic strategy compilation."""

from __future__ import annotations


class StrategyCompilationError(ValueError):
    """Base class for a strategy that cannot be compiled safely."""


class InvalidStrategyTypeError(StrategyCompilationError, TypeError):
    """Raised when compiler input is not a validated ``StrategySchema``."""


class StrategySemanticValidationError(StrategyCompilationError):
    """Raised when a structurally valid strategy fails semantic validation."""


class IndicatorResolutionError(StrategyCompilationError):
    """Raised when an indicator name is absent from the allow-list."""


class IndicatorComputationError(StrategyCompilationError):
    """Raised when an allow-listed indicator cannot be calculated."""


class OperatorResolutionError(StrategyCompilationError):
    """Raised when an operator is absent from the allow-list."""


class SignalAlignmentError(StrategyCompilationError):
    """Raised when an indicator or signal index is not exactly aligned."""
