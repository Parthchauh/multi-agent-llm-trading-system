"""
strategies/__init__.py
======================
Public API for the strategy system.

Sprint 1  — Schema and semantic validation.
Sprint 3  — Indicator computation and strategy compilation.

Example usage::

    from strategies import (
        StrategySchema,
        StrategyValidator,
        StrategyCompiler,
        CompiledStrategy,
    )
"""

# Sprint 1 — Schema models
from strategies.schema import (
    Condition,
    ConditionGroup,
    EntryRules,
    ExitRules,
    PositionSizing,
    StopLoss,
    StrategyMetadata,
    StrategySchema,
    TakeProfit,
)

# Sprint 1 — Semantic validation
from strategies.validator import StrategyValidator, ValidationResult

# Sprint 1 — Registry constants and helpers
from strategies.registry import (
    SUPPORTED_INDICATORS,
    SUPPORTED_OPERATORS,
    COMPARISON_OPERATORS,
    CROSSOVER_OPERATORS,
    is_supported_indicator,
    is_supported_operator,
    is_crossover_operator,
    is_comparison_operator,
)

# Sprint 3 — Indicator computation
from strategies.indicators import (
    INDICATOR_REGISTRY,
    IndicatorSpec,
    REGISTRY_TO_STOCKSTATS,
    collect_required_indicators,
    compute_indicators,
)

# Sprint 3 — Strategy compiler
from strategies.compiler import (
    CompiledStrategy,
    StrategyCompiler,
)
from strategies.exceptions import (
    IndicatorComputationError,
    IndicatorResolutionError,
    InvalidStrategyTypeError,
    OperatorResolutionError,
    SignalAlignmentError,
    StrategyCompilationError,
    StrategySemanticValidationError,
)
from strategies.operator_registry import (
    OPERATOR_REGISTRY,
    apply_operator,
    resolve_operator,
)

__all__ = [
    # Schema models
    "Condition",
    "ConditionGroup",
    "EntryRules",
    "ExitRules",
    "PositionSizing",
    "StopLoss",
    "StrategyMetadata",
    "StrategySchema",
    "TakeProfit",
    # Validation
    "StrategyValidator",
    "ValidationResult",
    # Registry
    "SUPPORTED_INDICATORS",
    "SUPPORTED_OPERATORS",
    "COMPARISON_OPERATORS",
    "CROSSOVER_OPERATORS",
    "is_supported_indicator",
    "is_supported_operator",
    "is_crossover_operator",
    "is_comparison_operator",
    # Indicators (Sprint 3)
    "REGISTRY_TO_STOCKSTATS",
    "INDICATOR_REGISTRY",
    "IndicatorSpec",
    "collect_required_indicators",
    "compute_indicators",
    # Compiler (Sprint 3)
    "CompiledStrategy",
    "StrategyCompiler",
    # Safe execution registries and typed failures (Sprint 3)
    "OPERATOR_REGISTRY",
    "apply_operator",
    "resolve_operator",
    "StrategyCompilationError",
    "InvalidStrategyTypeError",
    "StrategySemanticValidationError",
    "IndicatorResolutionError",
    "IndicatorComputationError",
    "OperatorResolutionError",
    "SignalAlignmentError",
]
