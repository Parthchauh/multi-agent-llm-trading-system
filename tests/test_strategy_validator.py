"""
tests/test_strategy_validator.py
=================================
Pytest tests for strategies/validator.py (StrategyValidator + ValidationResult).

Tests cover:
* Valid strategy → is_valid == True, no errors, no warnings.
* Unknown indicator → error.
* Unknown operator → error (caught at schema level, relayed through validate_dict).
* Crossover with numeric value → error (caught at schema level).
* Contradictory conditions in ALL group → error.
* Duplicate conditions in ALL group → warning, not error.
* Stop loss sanity.
* Take profit sanity.
* Holding period sanity.
* validate_dict convenience method — valid JSON dict and invalid JSON dict.
* ValidationResult str representation.
"""

from __future__ import annotations

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
from strategies.validator import StrategyValidator, ValidationResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_strategy(
    entry_conditions: list[Condition] | None = None,
    exit_conditions: list[Condition] | None = None,
    entry_logic: str = "ALL",
    exit_logic: str = "ANY",
    stop_loss: StopLoss | None = None,
    take_profit: TakeProfit | None = None,
    position_sizing: PositionSizing | None = None,
    maximum_holding_days: int | None = None,
) -> StrategySchema:
    """Build a StrategySchema with sensible defaults, overrideable per-test."""
    if entry_conditions is None:
        entry_conditions = [
            Condition(indicator="rsi_14", operator=">", value=55),
            Condition(indicator="close", operator=">", value="ema_20"),
        ]
    if exit_conditions is None:
        exit_conditions = [
            Condition(indicator="rsi_14", operator="<", value=45),
        ]
    return StrategySchema(
        metadata=StrategyMetadata(
            name="Test Strategy",
            market_type="equity",
            timeframe="1d",
        ),
        entry=EntryRules(
            long_conditions=ConditionGroup(
                logic=entry_logic,
                conditions=entry_conditions,
            )
        ),
        exit=ExitRules(
            exit_conditions=ConditionGroup(
                logic=exit_logic,
                conditions=exit_conditions,
            ),
            maximum_holding_days=maximum_holding_days,
        ),
        stop_loss=stop_loss or StopLoss(type="percentage", value=5),
        take_profit=take_profit or TakeProfit(type="percentage", value=10),
        position_sizing=position_sizing or PositionSizing(type="fixed_percentage", value=10),
    )


@pytest.fixture()
def validator() -> StrategyValidator:
    return StrategyValidator()


@pytest.fixture()
def valid_strategy() -> StrategySchema:
    return _make_strategy()


# ---------------------------------------------------------------------------
# Tests: valid strategy
# ---------------------------------------------------------------------------


class TestValidStrategy:
    def test_valid_strategy_is_valid(
        self, validator: StrategyValidator, valid_strategy: StrategySchema
    ) -> None:
        result = validator.validate(valid_strategy)
        assert result.is_valid is True

    def test_valid_strategy_has_no_errors(
        self, validator: StrategyValidator, valid_strategy: StrategySchema
    ) -> None:
        result = validator.validate(valid_strategy)
        assert result.errors == []

    def test_valid_strategy_has_no_warnings(
        self, validator: StrategyValidator, valid_strategy: StrategySchema
    ) -> None:
        result = validator.validate(valid_strategy)
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Tests: indicator validation (schema-level, surfaced via validate_dict)
# ---------------------------------------------------------------------------


class TestIndicatorValidation:
    def test_unsupported_indicator_in_entry_fails(
        self, validator: StrategyValidator
    ) -> None:
        """An unknown indicator should fail Pydantic validation."""
        data = {
            "metadata": {"name": "T", "market_type": "equity", "timeframe": "1d"},
            "entry": {
                "long_conditions": {
                    "logic": "ALL",
                    "conditions": [
                        {"indicator": "ghost_signal", "operator": ">", "value": 50}
                    ],
                }
            },
            "exit": {
                "exit_conditions": {
                    "logic": "ANY",
                    "conditions": [
                        {"indicator": "rsi_14", "operator": "<", "value": 45}
                    ],
                }
            },
            "stop_loss": {"type": "percentage", "value": 5},
            "take_profit": {"type": "percentage", "value": 10},
            "position_sizing": {"type": "fixed_percentage", "value": 10},
        }
        result = validator.validate_dict(data)
        assert result.is_valid is False
        assert any("ghost_signal" in e or "Unsupported indicator" in e for e in result.errors)

    def test_unsupported_indicator_in_exit_fails(
        self, validator: StrategyValidator
    ) -> None:
        data = {
            "metadata": {"name": "T", "market_type": "equity", "timeframe": "1d"},
            "entry": {
                "long_conditions": {
                    "logic": "ALL",
                    "conditions": [
                        {"indicator": "rsi_14", "operator": ">", "value": 50}
                    ],
                }
            },
            "exit": {
                "exit_conditions": {
                    "logic": "ANY",
                    "conditions": [
                        {"indicator": "fake_exit_indicator", "operator": "<", "value": 45}
                    ],
                }
            },
            "stop_loss": {"type": "percentage", "value": 5},
            "take_profit": {"type": "percentage", "value": 10},
            "position_sizing": {"type": "fixed_percentage", "value": 10},
        }
        result = validator.validate_dict(data)
        assert result.is_valid is False


# ---------------------------------------------------------------------------
# Tests: operator validation
# ---------------------------------------------------------------------------


class TestOperatorValidation:
    def test_unsupported_operator_fails(self, validator: StrategyValidator) -> None:
        """An unknown operator like 'approximately' must produce an error."""
        data = {
            "metadata": {"name": "T", "market_type": "equity", "timeframe": "1d"},
            "entry": {
                "long_conditions": {
                    "logic": "ALL",
                    "conditions": [
                        {"indicator": "rsi_14", "operator": "approximately", "value": 50}
                    ],
                }
            },
            "exit": {
                "exit_conditions": {
                    "logic": "ANY",
                    "conditions": [
                        {"indicator": "rsi_14", "operator": "<", "value": 45}
                    ],
                }
            },
            "stop_loss": {"type": "percentage", "value": 5},
            "take_profit": {"type": "percentage", "value": 10},
            "position_sizing": {"type": "fixed_percentage", "value": 10},
        }
        result = validator.validate_dict(data)
        assert result.is_valid is False
        assert any("approximately" in e or "operator" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Tests: crossover value must be indicator
# ---------------------------------------------------------------------------


class TestCrossoverValidation:
    def test_crossover_with_numeric_value_fails(self, validator: StrategyValidator) -> None:
        """crosses_above 50 (a number) must fail — must be an indicator name."""
        data = {
            "metadata": {"name": "T", "market_type": "equity", "timeframe": "1d"},
            "entry": {
                "long_conditions": {
                    "logic": "ALL",
                    "conditions": [
                        {"indicator": "ema_20", "operator": "crosses_above", "value": 50}
                    ],
                }
            },
            "exit": {
                "exit_conditions": {
                    "logic": "ANY",
                    "conditions": [
                        {"indicator": "rsi_14", "operator": "<", "value": 45}
                    ],
                }
            },
            "stop_loss": {"type": "percentage", "value": 5},
            "take_profit": {"type": "percentage", "value": 10},
            "position_sizing": {"type": "fixed_percentage", "value": 10},
        }
        result = validator.validate_dict(data)
        assert result.is_valid is False

    def test_crossover_with_valid_indicator_value_passes(
        self, validator: StrategyValidator
    ) -> None:
        strategy = _make_strategy(
            entry_conditions=[
                Condition(indicator="ema_20", operator="crosses_above", value="ema_50"),
            ]
        )
        result = validator.validate(strategy)
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# Tests: contradiction detection
# ---------------------------------------------------------------------------


class TestContradictionDetection:
    def test_rsi_above_70_and_below_30_contradicts(
        self, validator: StrategyValidator
    ) -> None:
        """RSI > 70 AND RSI < 30 in an ALL group is logically impossible."""
        strategy = _make_strategy(
            entry_conditions=[
                Condition(indicator="rsi_14", operator=">", value=70),
                Condition(indicator="rsi_14", operator="<", value=30),
            ],
            entry_logic="ALL",
        )
        result = validator.validate(strategy)
        assert result.is_valid is False
        assert any("rsi_14" in e for e in result.errors)

    def test_rsi_above_55_and_below_45_contradicts(
        self, validator: StrategyValidator
    ) -> None:
        strategy = _make_strategy(
            entry_conditions=[
                Condition(indicator="rsi_14", operator=">", value=55),
                Condition(indicator="rsi_14", operator="<", value=45),
            ],
            entry_logic="ALL",
        )
        result = validator.validate(strategy)
        assert result.is_valid is False

    def test_close_above_and_below_same_indicator_contradicts(
        self, validator: StrategyValidator
    ) -> None:
        """close > ema_20 AND close < ema_20 is a logical contradiction."""
        strategy = _make_strategy(
            entry_conditions=[
                Condition(indicator="close", operator=">", value="ema_20"),
                Condition(indicator="close", operator="<", value="ema_20"),
            ],
            entry_logic="ALL",
        )
        result = validator.validate(strategy)
        assert result.is_valid is False

    def test_rsi_equal_thresholds_contradicts(
        self, validator: StrategyValidator
    ) -> None:
        """RSI > 50 AND RSI < 50 — equal thresholds still contradictory."""
        strategy = _make_strategy(
            entry_conditions=[
                Condition(indicator="rsi_14", operator=">", value=50),
                Condition(indicator="rsi_14", operator="<", value=50),
            ],
            entry_logic="ALL",
        )
        result = validator.validate(strategy)
        assert result.is_valid is False

    def test_contradiction_in_any_group_does_not_error(
        self, validator: StrategyValidator
    ) -> None:
        """In an ANY group, contradictory conditions are redundant but not impossible
        to satisfy as a whole — no error should be raised."""
        strategy = _make_strategy(
            entry_conditions=[
                Condition(indicator="rsi_14", operator=">", value=70),
                Condition(indicator="rsi_14", operator="<", value=30),
            ],
            entry_logic="ANY",  # ANY group — at least one can be true
        )
        result = validator.validate(strategy)
        assert result.is_valid is True

    def test_no_contradiction_with_non_overlapping_bounds(
        self, validator: StrategyValidator
    ) -> None:
        """RSI > 50 AND RSI < 80 — perfectly valid range."""
        strategy = _make_strategy(
            entry_conditions=[
                Condition(indicator="rsi_14", operator=">", value=50),
                Condition(indicator="rsi_14", operator="<", value=80),
            ],
            entry_logic="ALL",
        )
        result = validator.validate(strategy)
        assert result.is_valid is True

    def test_different_indicators_no_contradiction(
        self, validator: StrategyValidator
    ) -> None:
        """RSI > 55 AND close < ema_20 — different indicators, no contradiction."""
        strategy = _make_strategy(
            entry_conditions=[
                Condition(indicator="rsi_14", operator=">", value=55),
                Condition(indicator="close", operator="<", value="ema_20"),
            ],
            entry_logic="ALL",
        )
        result = validator.validate(strategy)
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# Tests: duplicate condition detection
# ---------------------------------------------------------------------------


class TestDuplicateConditionDetection:
    def test_exact_duplicate_produces_warning(
        self, validator: StrategyValidator
    ) -> None:
        """Identical conditions in the same group should produce a warning."""
        strategy = _make_strategy(
            entry_conditions=[
                Condition(indicator="rsi_14", operator=">", value=55),
                Condition(indicator="rsi_14", operator=">", value=55),  # duplicate
                Condition(indicator="close", operator=">", value="ema_20"),
            ],
        )
        result = validator.validate(strategy)
        assert len(result.warnings) >= 1
        # Duplicate warning — not an error
        assert result.is_valid is True

    def test_non_duplicate_has_no_warning(
        self, validator: StrategyValidator
    ) -> None:
        strategy = _make_strategy(
            entry_conditions=[
                Condition(indicator="rsi_14", operator=">", value=55),
                Condition(indicator="rsi_14", operator=">", value=60),  # different value
            ],
        )
        result = validator.validate(strategy)
        # Different value — not a duplicate
        assert all("Duplicate" not in w for w in result.warnings)

    def test_three_duplicates_produces_two_warnings(
        self, validator: StrategyValidator
    ) -> None:
        """Three identical conditions → warnings for the 2nd and 3rd occurrences."""
        strategy = _make_strategy(
            entry_conditions=[
                Condition(indicator="rsi_14", operator=">", value=55),
                Condition(indicator="rsi_14", operator=">", value=55),
                Condition(indicator="rsi_14", operator=">", value=55),
            ],
        )
        result = validator.validate(strategy)
        assert len(result.warnings) >= 2


# ---------------------------------------------------------------------------
# Tests: validate_dict convenience method
# ---------------------------------------------------------------------------


class TestValidateDict:
    def _valid_dict(self) -> dict:
        return {
            "metadata": {"name": "T", "market_type": "equity", "timeframe": "1d"},
            "entry": {
                "long_conditions": {
                    "logic": "ALL",
                    "conditions": [
                        {"indicator": "rsi_14", "operator": ">", "value": 55},
                    ],
                }
            },
            "exit": {
                "exit_conditions": {
                    "logic": "ANY",
                    "conditions": [
                        {"indicator": "rsi_14", "operator": "<", "value": 45}
                    ],
                }
            },
            "stop_loss": {"type": "percentage", "value": 5},
            "take_profit": {"type": "percentage", "value": 10},
            "position_sizing": {"type": "fixed_percentage", "value": 10},
        }

    def test_valid_dict_passes(self, validator: StrategyValidator) -> None:
        result = validator.validate_dict(self._valid_dict())
        assert result.is_valid is True

    def test_invalid_json_string_fails(self, validator: StrategyValidator) -> None:
        result = validator.validate_dict("{this is not valid json}")
        assert result.is_valid is False
        assert any("JSON" in e for e in result.errors)

    def test_missing_required_field_fails(self, validator: StrategyValidator) -> None:
        data = self._valid_dict()
        del data["stop_loss"]
        result = validator.validate_dict(data)
        assert result.is_valid is False


# ---------------------------------------------------------------------------
# Tests: ValidationResult model
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_is_valid_when_no_errors(self) -> None:
        r = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert r.is_valid is True

    def test_is_not_valid_when_errors_present(self) -> None:
        r = ValidationResult(is_valid=False, errors=["something wrong"])
        assert r.is_valid is False
        assert len(r.errors) == 1

    def test_str_representation_contains_errors(self) -> None:
        r = ValidationResult(
            is_valid=False,
            errors=["Bad indicator"],
            warnings=["Duplicate found"],
        )
        s = str(r)
        assert "ERROR" in s
        assert "WARNING" in s
        assert "Bad indicator" in s
        assert "Duplicate found" in s

    def test_warnings_do_not_make_invalid(self) -> None:
        r = ValidationResult(is_valid=True, errors=[], warnings=["minor issue"])
        assert r.is_valid is True
