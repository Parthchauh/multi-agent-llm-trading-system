"""
tests/test_strategy_schema.py
=============================
Pytest tests for strategies/schema.py.

Tests cover:
* A complete valid strategy parses without error.
* Field-level validation errors (bad position size, bad stop loss, etc.).
* Pydantic-enforced enum constraints (market_type, timeframe).
* Crossover operator constraints (value must be indicator, not number).
* JSON serialisation roundtrip.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

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


# ---------------------------------------------------------------------------
# Fixtures — reusable valid building blocks
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_entry() -> EntryRules:
    return EntryRules(
        long_conditions=ConditionGroup(
            logic="ALL",
            conditions=[
                Condition(indicator="ema_20", operator="crosses_above", value="ema_50"),
                Condition(indicator="rsi_14", operator=">", value=55),
            ],
        )
    )


@pytest.fixture()
def valid_exit() -> ExitRules:
    return ExitRules(
        exit_conditions=ConditionGroup(
            logic="ANY",
            conditions=[
                Condition(indicator="rsi_14", operator="<", value=45),
            ],
        ),
        maximum_holding_days=30,
    )


@pytest.fixture()
def valid_strategy(valid_entry: EntryRules, valid_exit: ExitRules) -> StrategySchema:
    return StrategySchema(
        metadata=StrategyMetadata(
            name="EMA RSI Momentum",
            description="Long-only momentum strategy",
            market_type="equity",
            timeframe="1d",
        ),
        entry=valid_entry,
        exit=valid_exit,
        stop_loss=StopLoss(type="percentage", value=5),
        take_profit=TakeProfit(type="percentage", value=10),
        position_sizing=PositionSizing(type="fixed_percentage", value=10),
    )


# ---------------------------------------------------------------------------
# Tests: full valid strategy
# ---------------------------------------------------------------------------


class TestValidStrategy:
    def test_valid_strategy_parses(self, valid_strategy: StrategySchema) -> None:
        """A fully-populated valid strategy should parse without error."""
        assert valid_strategy.metadata.name == "EMA RSI Momentum"
        assert valid_strategy.metadata.market_type == "equity"
        assert valid_strategy.metadata.timeframe == "1d"

    def test_valid_strategy_entry_has_two_conditions(self, valid_strategy: StrategySchema) -> None:
        assert len(valid_strategy.entry.long_conditions.conditions) == 2

    def test_valid_strategy_exit_has_max_holding(self, valid_strategy: StrategySchema) -> None:
        assert valid_strategy.exit.maximum_holding_days == 30

    def test_valid_strategy_stop_loss(self, valid_strategy: StrategySchema) -> None:
        assert valid_strategy.stop_loss.type == "percentage"
        assert valid_strategy.stop_loss.value == 5

    def test_valid_strategy_take_profit(self, valid_strategy: StrategySchema) -> None:
        assert valid_strategy.take_profit.type == "percentage"
        assert valid_strategy.take_profit.value == 10

    def test_valid_strategy_position_sizing(self, valid_strategy: StrategySchema) -> None:
        assert valid_strategy.position_sizing.type == "fixed_percentage"
        assert valid_strategy.position_sizing.value == 10


# ---------------------------------------------------------------------------
# Tests: JSON serialisation / deserialisation
# ---------------------------------------------------------------------------


class TestJsonSerialisation:
    def test_model_dump_returns_dict(self, valid_strategy: StrategySchema) -> None:
        d = valid_strategy.model_dump()
        assert isinstance(d, dict)
        assert "metadata" in d
        assert "entry" in d
        assert "exit" in d

    def test_model_dump_json_returns_string(self, valid_strategy: StrategySchema) -> None:
        s = valid_strategy.model_dump_json()
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["metadata"]["name"] == "EMA RSI Momentum"

    def test_roundtrip_via_model_validate(self, valid_strategy: StrategySchema) -> None:
        """Serialise to dict and re-parse — result must be equal."""
        d = valid_strategy.model_dump()
        reconstructed = StrategySchema.model_validate(d)
        assert reconstructed == valid_strategy

    def test_roundtrip_via_json_string(self, valid_strategy: StrategySchema) -> None:
        """Serialise to JSON string and re-parse — result must be equal."""
        s = valid_strategy.model_dump_json()
        reconstructed = StrategySchema.model_validate_json(s)
        assert reconstructed == valid_strategy


# ---------------------------------------------------------------------------
# Tests: PositionSizing validation
# ---------------------------------------------------------------------------


class TestPositionSizing:
    def test_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be > 0"):
            PositionSizing(type="fixed_percentage", value=0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be > 0"):
            PositionSizing(type="fixed_percentage", value=-10)

    def test_above_100_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be <= 100"):
            PositionSizing(type="fixed_percentage", value=101)

    def test_exactly_100_is_valid(self) -> None:
        ps = PositionSizing(type="fixed_percentage", value=100)
        assert ps.value == 100

    def test_fractional_value_is_valid(self) -> None:
        ps = PositionSizing(type="fixed_percentage", value=2.5)
        assert ps.value == 2.5


# ---------------------------------------------------------------------------
# Tests: StrategyMetadata validation
# ---------------------------------------------------------------------------


class TestStrategyMetadata:
    def test_invalid_market_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            StrategyMetadata(
                name="Test",
                market_type="crypto",   # not supported in Sprint 1
                timeframe="1d",
            )

    def test_invalid_timeframe_raises(self) -> None:
        with pytest.raises(ValidationError):
            StrategyMetadata(
                name="Test",
                market_type="equity",
                timeframe="1h",          # not supported in Sprint 1
            )

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            StrategyMetadata(
                name="",
                market_type="equity",
                timeframe="1d",
            )

    def test_valid_metadata_parses(self) -> None:
        m = StrategyMetadata(name="My Strategy", market_type="equity", timeframe="1d")
        assert m.name == "My Strategy"


# ---------------------------------------------------------------------------
# Tests: StopLoss validation
# ---------------------------------------------------------------------------


class TestStopLoss:
    def test_negative_percentage_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be > 0"):
            StopLoss(type="percentage", value=-5)

    def test_zero_percentage_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be > 0"):
            StopLoss(type="percentage", value=0)

    def test_over_100_percentage_raises(self) -> None:
        with pytest.raises(ValidationError):
            StopLoss(type="percentage", value=101)

    def test_valid_percentage_stop_loss(self) -> None:
        sl = StopLoss(type="percentage", value=5)
        assert sl.value == 5

    def test_valid_atr_multiple_stop_loss(self) -> None:
        sl = StopLoss(type="atr_multiple", value=2)
        assert sl.type == "atr_multiple"

    def test_zero_atr_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be > 0"):
            StopLoss(type="atr_multiple", value=0)


# ---------------------------------------------------------------------------
# Tests: TakeProfit validation
# ---------------------------------------------------------------------------


class TestTakeProfit:
    def test_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be > 0"):
            TakeProfit(type="percentage", value=0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be > 0"):
            TakeProfit(type="percentage", value=-10)

    def test_valid_percentage(self) -> None:
        tp = TakeProfit(type="percentage", value=10)
        assert tp.value == 10

    def test_valid_risk_reward(self) -> None:
        tp = TakeProfit(type="risk_reward", value=3)
        assert tp.type == "risk_reward"


# ---------------------------------------------------------------------------
# Tests: ExitRules — maximum_holding_days
# ---------------------------------------------------------------------------


class TestExitRulesHoldingPeriod:
    def _base_exit_conditions(self) -> ConditionGroup:
        return ConditionGroup(
            logic="ANY",
            conditions=[
                Condition(indicator="rsi_14", operator="<", value=45),
            ],
        )

    def test_zero_holding_days_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExitRules(
                exit_conditions=self._base_exit_conditions(),
                maximum_holding_days=0,
            )

    def test_negative_holding_days_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExitRules(
                exit_conditions=self._base_exit_conditions(),
                maximum_holding_days=-5,
            )

    def test_none_holding_days_is_valid(self) -> None:
        er = ExitRules(
            exit_conditions=self._base_exit_conditions(),
            maximum_holding_days=None,
        )
        assert er.maximum_holding_days is None

    def test_positive_holding_days_is_valid(self) -> None:
        er = ExitRules(
            exit_conditions=self._base_exit_conditions(),
            maximum_holding_days=10,
        )
        assert er.maximum_holding_days == 10


# ---------------------------------------------------------------------------
# Tests: Condition — indicator and operator validation
# ---------------------------------------------------------------------------


class TestConditionValidation:
    def test_unsupported_indicator_raises(self) -> None:
        with pytest.raises(ValidationError, match="Unsupported indicator"):
            Condition(indicator="fake_rsi", operator=">", value=50)

    def test_unsupported_operator_raises(self) -> None:
        with pytest.raises(ValidationError, match="Unsupported operator"):
            Condition(indicator="rsi_14", operator="approximately", value=50)

    def test_crossover_with_numeric_value_raises(self) -> None:
        """crosses_above 50 is invalid — value must be another indicator."""
        with pytest.raises(ValidationError, match="indicator name"):
            Condition(indicator="ema_20", operator="crosses_above", value=50)

    def test_crossover_with_unsupported_indicator_value_raises(self) -> None:
        with pytest.raises(ValidationError, match="supported indicator"):
            Condition(indicator="ema_20", operator="crosses_above", value="mystery_line")

    def test_crossover_with_indicator_value_is_valid(self) -> None:
        cond = Condition(indicator="ema_20", operator="crosses_above", value="ema_50")
        assert cond.value == "ema_50"

    def test_comparison_with_numeric_value_is_valid(self) -> None:
        cond = Condition(indicator="rsi_14", operator=">", value=55)
        assert cond.value == 55

    def test_comparison_with_indicator_value_is_valid(self) -> None:
        cond = Condition(indicator="close", operator=">", value="ema_20")
        assert cond.value == "ema_20"

    def test_all_comparison_operators_accepted(self) -> None:
        for op in (">", ">=", "<", "<=", "=="):
            cond = Condition(indicator="rsi_14", operator=op, value=50)
            assert cond.operator == op

    def test_all_crossover_operators_accepted(self) -> None:
        for op in ("crosses_above", "crosses_below"):
            cond = Condition(indicator="ema_20", operator=op, value="ema_50")
            assert cond.operator == op


# ---------------------------------------------------------------------------
# Tests: ConditionGroup
# ---------------------------------------------------------------------------


class TestConditionGroup:
    def test_empty_conditions_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConditionGroup(logic="ALL", conditions=[])

    def test_invalid_logic_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConditionGroup(
                logic="AND",     # must be 'ALL' or 'ANY'
                conditions=[Condition(indicator="rsi_14", operator=">", value=50)],
            )

    def test_all_logic_is_valid(self) -> None:
        g = ConditionGroup(
            logic="ALL",
            conditions=[Condition(indicator="rsi_14", operator=">", value=50)],
        )
        assert g.logic == "ALL"

    def test_any_logic_is_valid(self) -> None:
        g = ConditionGroup(
            logic="ANY",
            conditions=[Condition(indicator="rsi_14", operator="<", value=30)],
        )
        assert g.logic == "ANY"


# ---------------------------------------------------------------------------
# Tests: supported indicators from registry are all valid in Condition
# ---------------------------------------------------------------------------


class TestAllRegistryIndicatorsAccepted:
    def test_all_registry_indicators_parse_in_condition(self) -> None:
        """Every indicator in SUPPORTED_INDICATORS must parse without error."""
        from strategies.registry import SUPPORTED_INDICATORS

        for indicator in SUPPORTED_INDICATORS:
            cond = Condition(indicator=indicator, operator=">", value=1.0)
            assert cond.indicator == indicator
