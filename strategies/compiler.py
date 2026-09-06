"""Deterministic bridge from validated strategy schemas to backtest signals.

The compiler resolves only allow-listed indicators and operators. It emits
aligned boolean Series and implements Sprint 2's ``StrategySignalProvider``
protocol. No expression parsing or arbitrary code execution is used.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from data.schema import validate_ohlcv
from strategies.exceptions import (
    IndicatorResolutionError,
    InvalidStrategyTypeError,
    SignalAlignmentError,
    StrategySemanticValidationError,
)
from strategies.indicators import collect_required_indicators, compute_indicators
from strategies.operator_registry import apply_operator
from strategies.schema import Condition, ConditionGroup, StrategySchema
from strategies.validator import StrategyValidator


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class CompiledStrategy:
    """Precomputed long-entry/exit signals plus Sprint 2 risk callbacks."""

    def __init__(self, strategy: StrategySchema, indicator_df: pd.DataFrame) -> None:
        if not isinstance(strategy, StrategySchema):
            raise InvalidStrategyTypeError(
                "CompiledStrategy requires a validated StrategySchema instance."
            )
        if not isinstance(indicator_df, pd.DataFrame):
            raise SignalAlignmentError("indicator_df must be a pandas DataFrame.")
        if not indicator_df.index.is_unique:
            raise SignalAlignmentError("Indicator index must contain unique timestamps.")

        self._strategy = strategy
        self._indicator_df = indicator_df.copy(deep=True)
        self._n_bars = len(indicator_df)
        self._assert_required_columns()
        self._long_entry = self._compile_group(strategy.entry.long_conditions)
        self._long_exit = self._compile_group(strategy.exit.exit_conditions)

    def _assert_required_columns(self) -> None:
        required = collect_required_indicators(self._strategy)
        missing = sorted(required - set(self._indicator_df.columns))
        if missing:
            raise IndicatorResolutionError(
                f"Compiled indicator frame is missing required columns: {missing}."
            )

    def _compile_group(self, group: ConditionGroup) -> pd.Series:
        parts: list[pd.Series] = []
        for node in group.conditions:
            if isinstance(node, ConditionGroup):
                parts.append(self._compile_group(node))
            else:
                parts.append(self._compile_condition(node))

        result = parts[0].copy()
        for part in parts[1:]:
            if not part.index.equals(result.index):
                raise SignalAlignmentError("Nested condition signals are not aligned.")
            result = result & part if group.logic == "ALL" else result | part
        return result.fillna(False).astype(bool).rename(None)

    def _compile_condition(self, condition: Condition) -> pd.Series:
        lhs = pd.to_numeric(
            self._indicator_df[condition.indicator], errors="coerce"
        ).astype(float)
        if isinstance(condition.value, str):
            rhs = pd.to_numeric(
                self._indicator_df[condition.value], errors="coerce"
            ).astype(float)
        else:
            rhs = pd.Series(
                float(condition.value),
                index=self._indicator_df.index,
                dtype=float,
            )
        return apply_operator(condition.operator, lhs, rhs)

    def should_enter(self, current_index: int, data: pd.DataFrame) -> bool:
        """Return the compiled long-entry signal for ``current_index``."""
        if current_index < 0 or current_index >= self._n_bars:
            return False
        return bool(self._long_entry.iloc[current_index])

    def should_exit(self, current_index: int, data: pd.DataFrame) -> bool:
        """Return the compiled long-exit signal for ``current_index``."""
        if current_index < 0 or current_index >= self._n_bars:
            return False
        return bool(self._long_exit.iloc[current_index])

    def get_stop_price(
        self,
        entry_price: float,
        current_index: int,
        data: pd.DataFrame,
    ) -> Optional[float]:
        """Return the deterministic percentage or ATR stop for a new fill."""
        stop = self._strategy.stop_loss
        if stop.type == "percentage":
            return entry_price * (1.0 - stop.value / 100.0)
        if current_index < 0 or current_index >= self._n_bars:
            return None
        atr = _safe_float(self._indicator_df["atr_14"].iloc[current_index])
        return None if atr is None else entry_price - stop.value * atr

    def get_take_profit_price(
        self,
        entry_price: float,
        stop_price: Optional[float],
        current_index: int,
        data: pd.DataFrame,
    ) -> Optional[float]:
        """Return the deterministic percentage or risk/reward target."""
        target = self._strategy.take_profit
        if target.type == "percentage":
            return entry_price * (1.0 + target.value / 100.0)
        if stop_price is None:
            return None
        risk = entry_price - stop_price
        return None if risk <= 0 else entry_price + target.value * risk

    @property
    def long_entry(self) -> pd.Series:
        """Copy of the aligned long-entry boolean signal."""
        return self._long_entry.copy(deep=True)

    @property
    def entry_signals(self) -> pd.Series:
        """Compatibility alias for :attr:`long_entry`, returned defensively."""

        return self.long_entry

    @property
    def long_exit(self) -> pd.Series:
        """Copy of the aligned long-exit boolean signal."""
        return self._long_exit.copy(deep=True)

    @property
    def exit_signals(self) -> pd.Series:
        """Compatibility alias for :attr:`long_exit`, returned defensively."""

        return self.long_exit

    @property
    def signals(self) -> pd.DataFrame:
        """Aligned signal table suitable for auditing or export."""
        return pd.DataFrame(
            {"long_entry": self._long_entry, "long_exit": self._long_exit},
            index=self._indicator_df.index,
        )

    @property
    def strategy_name(self) -> str:
        return self._strategy.metadata.name

    @property
    def n_bars(self) -> int:
        return self._n_bars


class StrategyCompiler:
    """Compile a validated ``StrategySchema`` against canonical OHLCV data."""

    def compile(self, strategy: StrategySchema, data: pd.DataFrame) -> CompiledStrategy:
        if not isinstance(strategy, StrategySchema):
            raise InvalidStrategyTypeError(
                "StrategyCompiler accepts only an already parsed StrategySchema; "
                "raw dict/JSON input must pass through StrategyValidator.validate_dict()."
            )

        semantic_result = StrategyValidator().validate(strategy)
        if not semantic_result.is_valid:
            raise StrategySemanticValidationError(
                "Strategy failed semantic validation: "
                + "; ".join(semantic_result.errors)
            )

        market_data = validate_ohlcv(data)
        required = collect_required_indicators(strategy)
        indicator_df = compute_indicators(market_data, required, strict=True)
        if not indicator_df.index.equals(data.index):
            raise SignalAlignmentError(
                "Indicator computation did not preserve the original DataFrame index."
            )
        return CompiledStrategy(strategy=strategy, indicator_df=indicator_df)
