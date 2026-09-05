"""Sprint 3.5 contract hardening tests.

Verifies:
- NaN/Inf rejected from all authoritative numeric model fields
- ConditionGroup.conditions is a tuple (immutable)
- BacktestConfig uses extra=forbid
- Inclusive-bound contradiction detector fix (P2-01)
"""
from __future__ import annotations
import math
import pytest
from pydantic import ValidationError
from strategies.schema import (
    Condition, ConditionGroup, StopLoss, TakeProfit, PositionSizing,
    StrategySchema, StrategyMetadata, EntryRules, ExitRules,
)
from strategies.validator import StrategyValidator, _ConditionContradict
from backtesting.models import BacktestConfig


# ---------------------------------------------------------------------------
# Schema NaN / Inf guards
# ---------------------------------------------------------------------------

class TestStopLossNanGuard:
    def test_nan_stop_loss_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            StopLoss(type="percentage", value=float("nan"))

    def test_inf_stop_loss_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            StopLoss(type="percentage", value=float("inf"))

    def test_neg_inf_stop_loss_rejected(self):
        with pytest.raises(ValidationError):
            StopLoss(type="percentage", value=float("-inf"))

    def test_valid_stop_loss_passes(self):
        s = StopLoss(type="percentage", value=5.0)
        assert s.value == 5.0


class TestTakeProfitNanGuard:
    def test_nan_take_profit_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            TakeProfit(type="percentage", value=float("nan"))

    def test_inf_take_profit_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            TakeProfit(type="percentage", value=float("inf"))

    def test_valid_take_profit_passes(self):
        t = TakeProfit(type="percentage", value=10.0)
        assert t.value == 10.0


class TestPositionSizingNanGuard:
    def test_nan_position_size_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            PositionSizing(type="fixed_percentage", value=float("nan"))

    def test_inf_position_size_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            PositionSizing(type="fixed_percentage", value=float("inf"))

    def test_valid_position_size_passes(self):
        p = PositionSizing(type="fixed_percentage", value=20.0)
        assert p.value == 20.0


class TestBacktestConfigGuards:
    def test_nan_initial_capital_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            BacktestConfig(initial_capital=float("nan"))

    def test_inf_initial_capital_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            BacktestConfig(initial_capital=float("inf"))

    def test_nan_commission_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            BacktestConfig(initial_capital=10000.0, commission_pct=float("nan"))

    def test_inf_slippage_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            BacktestConfig(initial_capital=10000.0, slippage_pct=float("inf"))

    def test_nan_position_size_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            BacktestConfig(initial_capital=10000.0, position_size_pct=float("nan"))

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            BacktestConfig(initial_capital=10000.0, unknown_field=42)

    def test_valid_config_passes(self):
        cfg = BacktestConfig(initial_capital=50000.0, commission_pct=0.001, slippage_pct=0.0005)
        assert cfg.initial_capital == 50000.0

    def test_commission_above_50pct_rejected(self):
        with pytest.raises(ValidationError, match="50%"):
            BacktestConfig(initial_capital=10000.0, commission_pct=0.99)


# ---------------------------------------------------------------------------
# ConditionGroup immutability
# ---------------------------------------------------------------------------

class TestConditionGroupImmutability:
    def test_conditions_is_tuple(self):
        cg = ConditionGroup(logic="ALL", conditions=[
            Condition(indicator="rsi_14", operator=">", value=50.0)
        ])
        assert isinstance(cg.conditions, tuple)

    def test_cannot_append_to_conditions(self):
        cg = ConditionGroup(logic="ALL", conditions=[
            Condition(indicator="rsi_14", operator=">", value=50.0)
        ])
        with pytest.raises((AttributeError, TypeError)):
            cg.conditions.append(  # type: ignore[attr-defined]
                Condition(indicator="close", operator=">", value=100.0)
            )

    def test_cannot_mutate_model_field(self):
        cg = ConditionGroup(logic="ALL", conditions=[
            Condition(indicator="rsi_14", operator=">", value=50.0)
        ])
        with pytest.raises((TypeError, ValidationError)):
            cg.conditions = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Contradiction detection — inclusive bound fix (P2-01)
# ---------------------------------------------------------------------------

class TestContradictionDetector:
    def _make_cond(self, indicator, op, value):
        return Condition(indicator=indicator, operator=op, value=value)

    def test_inclusive_equal_bounds_not_contradictory(self):
        """RSI >= 50 AND RSI <= 50 is valid (x=50 satisfies both)."""
        checker = _ConditionContradict()
        checker.add(self._make_cond("rsi_14", ">=", 50.0))
        checker.add(self._make_cond("rsi_14", "<=", 50.0))
        assert checker.contradictions() == []

    def test_strict_equal_bounds_are_contradictory(self):
        """RSI > 50 AND RSI < 50 is impossible."""
        checker = _ConditionContradict()
        checker.add(self._make_cond("rsi_14", ">", 50.0))
        checker.add(self._make_cond("rsi_14", "<", 50.0))
        assert len(checker.contradictions()) == 1

    def test_mixed_strict_inclusive_equal_bounds_contradictory(self):
        """RSI > 50 AND RSI <= 50 is impossible (nothing satisfies strict > and inclusive <=)."""
        checker = _ConditionContradict()
        checker.add(self._make_cond("rsi_14", ">", 50.0))
        checker.add(self._make_cond("rsi_14", "<=", 50.0))
        assert len(checker.contradictions()) == 1

    def test_mixed_inclusive_strict_equal_bounds_contradictory(self):
        """RSI >= 50 AND RSI < 50 is impossible."""
        checker = _ConditionContradict()
        checker.add(self._make_cond("rsi_14", ">=", 50.0))
        checker.add(self._make_cond("rsi_14", "<", 50.0))
        assert len(checker.contradictions()) == 1

    def test_clear_numeric_contradiction_caught(self):
        """RSI > 70 AND RSI < 30 is impossible."""
        checker = _ConditionContradict()
        checker.add(self._make_cond("rsi_14", ">", 70.0))
        checker.add(self._make_cond("rsi_14", "<", 30.0))
        assert len(checker.contradictions()) == 1

    def test_valid_range_not_contradictory(self):
        """RSI > 30 AND RSI < 70 is valid."""
        checker = _ConditionContradict()
        checker.add(self._make_cond("rsi_14", ">", 30.0))
        checker.add(self._make_cond("rsi_14", "<", 70.0))
        assert checker.contradictions() == []

    def test_indicator_vs_indicator_contradiction(self):
        """close > ema_50 AND close < ema_50 is contradictory."""
        checker = _ConditionContradict()
        checker.add(self._make_cond("close", ">", "ema_50"))
        checker.add(self._make_cond("close", "<", "ema_50"))
        assert len(checker.contradictions()) == 1

    def test_validator_accepts_inclusive_bounds_strategy(self):
        """Full strategy validator should not reject RSI >= 50 AND RSI <= 70."""
        from tests.conftest_helpers import make_simple_strategy
        strategy = make_simple_strategy()
        # Modify to RSI >= 50 AND RSI <= 70 — valid range
        from strategies.schema import ConditionGroup, Condition, EntryRules
        entry = EntryRules(
            long_conditions=ConditionGroup(
                logic="ALL",
                conditions=[
                    Condition(indicator="rsi_14", operator=">=", value=50.0),
                    Condition(indicator="rsi_14", operator="<=", value=70.0),
                ],
            )
        )
        import copy
        patched = strategy.model_copy(update={"entry": entry})
        result = StrategyValidator().validate(patched)
        assert result.is_valid, f"Should be valid, got errors: {result.errors}"
