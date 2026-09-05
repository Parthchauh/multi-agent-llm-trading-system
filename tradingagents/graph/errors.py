"""Typed deterministic orchestration failures."""


class ResearchPipelineError(RuntimeError):
    """Base error for a controlled research-pipeline failure."""


class ResearchCompilationError(ResearchPipelineError):
    """Raised when deterministic strategy compilation fails."""


class ResearchBacktestError(ResearchPipelineError):
    """Raised when the authoritative backtest cannot complete."""
