"""Typed failures for immutable Sprint 6 strategy-version records."""

from __future__ import annotations


class StrategyLineageError(ValueError):
    """Base error for invalid immutable strategy lineage operations."""


class ChildStrategyValidationError(StrategyLineageError):
    """Raised before a raw refinement child can become a strategy version."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = errors
        super().__init__(
            "Refined strategy failed the canonical StrategyValidator validation path."
        )


class InvalidLineageTransitionError(StrategyLineageError):
    """Raised when a requested parent-to-child version transition is invalid."""
