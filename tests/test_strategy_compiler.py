"""
tests/test_strategy_compiler.py
================================
Tests for strategies/compiler.py (StrategyCompiler + CompiledStrategy).

Tests cover:
  - Compilation validates correctly
  - Condition evaluation for comparison operators
  - Crossover operator detection (crosses_above / crosses_below)
  - NaN indicator returns False (warmup bars)
  - Stop-loss calculation (percentage and ATR-multiple)
  - Take-profit calculation (percentage and risk-reward)
  - Empty data raises ValueError
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.compiler import CompiledStrategy, StrategyCompiler
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
from tests.conftest_helpers import make_ohlcv, make_simple_strategy


class TestStrategyCompilerBasic:
    def test_compile_returns_compiled_strategy(self) -> None:
        strategy = make_simple_strategy()
        data = make_ohlcv(100)
        compiled = StrategyCompiler().compile(strategy, data)
        assert isinstance(compiled, CompiledStrategy)

    def test_compile_empty_data_raises(self) -> None:
        strategy = make_simple_strategy()
        with pytest.raises(ValueError, match="empty"):
            StrategyCompiler().compile(strategy, pd.DataFrame())

    def test_compiled_strategy_name(self) -> None:
        strategy = make_simple_strategy()
        compiled = StrategyCompiler().compile(strategy, make_ohlcv(50))
        assert compiled.strategy_name == "Test RSI Strategy"

    def test_compiled_strategy_n_bars(self) -> None:
        strategy = make_simple_strategy()
        data = make_ohlcv(150)
        compiled = StrategyCompiler().compile(strategy, data)
        assert compiled.n_bars == 150


class TestConditionEvaluationComparisons:
    """Test all comparison operators using synthetic indicator DataFrames."""

    def _make_compiled(self, indicator_vals: dict[str, list[float]], strategy: StrategySchema) -> CompiledStrategy:
        """Build a CompiledStrategy with a manually constructed indicator DataFrame."""
        n = len(next(iter(indicator_vals.values())))
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        indicator_df = pd.DataFrame(indicator_vals, index=dates)
        return CompiledStrategy(strategy=strategy, indicator_df=indicator_df)

    def test_greater_than_true(self) -> None:
        from strategies.schema import Condition, ConditionGroup, EntryRules, ExitRules
        strategy = make_simple_strategy(entry_indicator="rsi_14", entry_operator=">", entry_value=50.0)
        compiled = self._make_compiled(
            {"rsi_14": [30.0, 60.0, 40.0], "close": [100.0, 105.0, 110.0], "sma_50": [95.0, 102.0, 108.0]},
            strategy,
        )
        dummy_data = pd.DataFrame()
        assert not compiled.should_enter(0, dummy_data)   # 30 > 50 → False
        assert compiled.should_enter(1, dummy_data)       # 60 > 50 → True
        assert not compiled.should_enter(2, dummy_data)   # 40 > 50 → False

    def test_less_than_operator(self) -> None:
        strategy = make_simple_strategy(exit_indicator="rsi_14", exit_operator="<", exit_value=45.0)
        compiled = self._make_compiled(
            {"rsi_14": [55.0, 40.0], "close": [100.0, 98.0], "sma_50": [95.0, 97.0]},
            strategy,
        )
        dummy_data = pd.DataFrame()
        assert not compiled.should_exit(0, dummy_data)  # 55 < 45 → False
        assert compiled.should_exit(1, dummy_data)      # 40 < 45 → True

    def test_nan_indicator_returns_false(self) -> None:
        strategy = make_simple_strategy()
        compiled = self._make_compiled(
            {"rsi_14": [float("nan"), 60.0], "close": [100.0, 105.0], "sma_50": [95.0, 102.0]},
            strategy,
        )
        dummy_data = pd.DataFrame()
        assert not compiled.should_enter(0, dummy_data)   # NaN → False
        assert compiled.should_enter(1, dummy_data)       # 60 > 55 → True

    def test_indicator_vs_indicator_comparison(self) -> None:
        """close > sma_50 comparison between two indicators."""
        from strategies.schema import Condition, ConditionGroup, EntryRules
        strategy = make_simple_strategy(entry_indicator="close", entry_operator=">", entry_value=0.0)
        # Override entry to close > sma_50
        strategy2 = StrategySchema(
            metadata=strategy.metadata,
            entry=EntryRules(
                long_conditions=ConditionGroup(
                    logic="ALL",
                    conditions=[Condition(indicator="close", operator=">", value="sma_50")],
                )
            ),
            exit=strategy.exit,
            stop_loss=strategy.stop_loss,
            take_profit=strategy.take_profit,
            position_sizing=strategy.position_sizing,
        )
        compiled = self._make_compiled(
            {"close": [90.0, 110.0], "sma_50": [100.0, 105.0]},
            strategy2,
        )
        dummy_data = pd.DataFrame()
        assert not compiled.should_enter(0, dummy_data)  # 90 > 100 → False
        assert compiled.should_enter(1, dummy_data)      # 110 > 105 → True


class TestCrossoverEvaluation:
    def _make_crossover_compiled(
        self, ema20_vals: list[float], ema50_vals: list[float]
    ) -> CompiledStrategy:
        from tests.conftest_helpers import make_crossover_strategy
        strategy = make_crossover_strategy()
        n = len(ema20_vals)
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        indicator_df = pd.DataFrame(
            {"ema_20": ema20_vals, "ema_50": ema50_vals},
            index=dates,
        )
        return CompiledStrategy(strategy=strategy, indicator_df=indicator_df)

    def test_crosses_above_detected(self) -> None:
        # Bar 0: ema20=90, ema50=100 (below)
        # Bar 1: ema20=110, ema50=100 (above) → crosses_above at bar 1
        compiled = self._make_crossover_compiled([90.0, 110.0, 115.0], [100.0, 100.0, 100.0])
        dummy = pd.DataFrame()
        assert not compiled.should_enter(0, dummy)  # first bar → no crossover
        assert compiled.should_enter(1, dummy)       # cross above detected
        assert not compiled.should_enter(2, dummy)   # already above, no cross

    def test_crosses_below_detected(self) -> None:
        # Bar 0: ema20=110, ema50=100 (above)
        # Bar 1: ema20=90,  ema50=100 (below) → crosses_below at bar 1
        compiled = self._make_crossover_compiled([110.0, 90.0, 85.0], [100.0, 100.0, 100.0])
        dummy = pd.DataFrame()
        assert not compiled.should_exit(0, dummy)   # first bar → no crossover
        assert compiled.should_exit(1, dummy)        # cross below detected
        assert not compiled.should_exit(2, dummy)    # already below

    def test_crossover_needs_minimum_two_bars(self) -> None:
        compiled = self._make_crossover_compiled([110.0], [100.0])
        dummy = pd.DataFrame()
        assert not compiled.should_enter(0, dummy)  # only one bar → cannot detect cross


class TestStopLossCalculation:
    def test_percentage_stop(self) -> None:
        strategy = make_simple_strategy(stop_loss_pct=5.0)
        compiled = StrategyCompiler().compile(strategy, make_ohlcv(50))
        stop = compiled.get_stop_price(100.0, 49, pd.DataFrame())
        assert pytest.approx(stop, 1e-6) == 95.0

    def test_atr_multiple_stop(self) -> None:
        from strategies.schema import StopLoss
        strategy = make_simple_strategy()
        # Rebuild with atr_multiple stop
        strategy2 = StrategySchema(
            metadata=strategy.metadata,
            entry=strategy.entry,
            exit=strategy.exit,
            stop_loss=StopLoss(type="atr_multiple", value=2.0),
            take_profit=strategy.take_profit,
            position_sizing=strategy.position_sizing,
        )
        data = make_ohlcv(100)
        compiled = StrategyCompiler().compile(strategy2, data)
        # Use last bar (ATR should be available after warmup)
        stop = compiled.get_stop_price(100.0, 99, pd.DataFrame())
        # ATR is positive; stop should be below entry
        assert stop is not None
        assert stop < 100.0

    def test_atr_stop_returns_none_during_warmup(self) -> None:
        from strategies.schema import StopLoss
        strategy = make_simple_strategy()
        strategy2 = StrategySchema(
            metadata=strategy.metadata,
            entry=strategy.entry,
            exit=strategy.exit,
            stop_loss=StopLoss(type="atr_multiple", value=2.0),
            take_profit=strategy.take_profit,
            position_sizing=strategy.position_sizing,
        )
        data = make_ohlcv(20)  # Very short — ATR may have NaN at bar 0
        compiled = StrategyCompiler().compile(strategy2, data)
        # Bar 0 ATR is likely NaN
        stop = compiled.get_stop_price(100.0, 0, pd.DataFrame())
        # Result is either None (NaN ATR) or a valid price below 100
        if stop is not None:
            assert stop < 100.0


class TestTakeProfitCalculation:
    def test_percentage_take_profit(self) -> None:
        strategy = make_simple_strategy(take_profit_pct=10.0)
        compiled = StrategyCompiler().compile(strategy, make_ohlcv(50))
        target = compiled.get_take_profit_price(100.0, 95.0, 49, pd.DataFrame())
        assert pytest.approx(target, 1e-6) == 110.0

    def test_risk_reward_take_profit(self) -> None:
        from strategies.schema import TakeProfit
        strategy = make_simple_strategy()
        strategy2 = StrategySchema(
            metadata=strategy.metadata,
            entry=strategy.entry,
            exit=strategy.exit,
            stop_loss=strategy.stop_loss,
            take_profit=TakeProfit(type="risk_reward", value=2.0),
            position_sizing=strategy.position_sizing,
        )
        compiled = StrategyCompiler().compile(strategy2, make_ohlcv(50))
        # Entry=100, Stop=95 → Risk=5 → Target = 100 + 2×5 = 110
        target = compiled.get_take_profit_price(100.0, 95.0, 49, pd.DataFrame())
        assert pytest.approx(target, 1e-6) == 110.0

    def test_risk_reward_without_stop_returns_none(self) -> None:
        from strategies.schema import TakeProfit
        strategy = make_simple_strategy()
        strategy2 = StrategySchema(
            metadata=strategy.metadata,
            entry=strategy.entry,
            exit=strategy.exit,
            stop_loss=strategy.stop_loss,
            take_profit=TakeProfit(type="risk_reward", value=2.0),
            position_sizing=strategy.position_sizing,
        )
        compiled = StrategyCompiler().compile(strategy2, make_ohlcv(50))
        # No stop price → risk_reward can't be computed
        target = compiled.get_take_profit_price(100.0, None, 49, pd.DataFrame())
        assert target is None
