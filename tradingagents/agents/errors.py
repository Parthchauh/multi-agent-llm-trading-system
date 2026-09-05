"""Typed failures for the controlled Sprint 4 agent layer."""

from __future__ import annotations


class AgentError(RuntimeError):
    """Base class for controlled agent-layer failures."""


class LLMProviderError(AgentError):
    """Raised when an external model provider cannot complete a request."""


class StructuredOutputError(AgentError):
    """Raised when a provider response cannot satisfy its output contract."""


class RegimeAnalysisError(AgentError):
    """Raised when structured regime analysis fails before fallback."""


class StrategyGenerationError(AgentError):
    """Raised when one strategy-generation attempt fails."""


class StrategyGenerationExhaustedError(StrategyGenerationError):
    """Raised after the bounded strategy-generation budget is exhausted."""

    def __init__(self, attempts: int, errors: tuple[str, ...]) -> None:
        self.attempts = attempts
        self.errors = errors
        summary = "; ".join(errors[-3:]) if errors else "unknown generation failure"
        super().__init__(
            f"Strategy generation exhausted after {attempts} attempts: {summary}"
        )
