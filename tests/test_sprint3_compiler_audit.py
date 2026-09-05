"""Integration and negative tests for the completed Sprint 3 compiler."""

from __future__ import annotations

import pandas as pd
import pytest

from data.schema import DuplicateTimestampError, MissingMarketDataColumnsError
from strategies.compiler import CompiledStrategy, StrategyCompiler
from strategies.exceptions import (
    IndicatorResolutionError,
    InvalidStrategyTypeError,
    OperatorResolutionError,
    StrategySemanticValidationError,
)
from strategies.schema import Condition, ConditionGroup, EntryRules
from tests.conftest_helpers import make_ohlcv, make_simple_strategy


def test_compiler_accepts_only_strategy_schema() -> None:
    with pytest.raises(InvalidStrategyTypeError, match="StrategySchema"):
        StrategyCompiler().compile({}, make_ohlcv(30))  # type: ignore[arg-type]


def test_compiled_signals_are_aligned_boolean_series() -> None:
    data = make_ohlcv(100)
    compiled = StrategyCompiler().compile(make_simple_strategy(), data)
    assert compiled.long_entry.index.equals(data.index)
    assert compiled.long_exit.index.equals(data.index)
    assert compiled.long_entry.dtype == bool
    assert compiled.long_exit.dtype == bool
    assert compiled.signals.columns.tolist() == ["long_entry", "long_exit"]


def test_nested_and_or_groups_compile_correctly() -> None:
    base = make_simple_strategy()
    nested = ConditionGroup(
        logic="ALL",
        conditions=[
            Condition(indicator="close", operator=">", value="sma_50"),
            ConditionGroup(
                logic="ANY",
                conditions=[
                    Condition(indicator="rsi_14", operator=">", value=50),
                    Condition(indicator="rsi_14", operator="<", value=30),
                ],
            ),
        ],
    )
    strategy = base.model_copy(
        update={"entry": EntryRules(long_conditions=nested)}
    )
    index = pd.date_range("2024-01-01", periods=4)
    indicators = pd.DataFrame(
        {
            "close": [5.0, 5.0, 11.0, 11.0],
            "sma_50": [10.0, 10.0, 10.0, 10.0],
            "rsi_14": [20.0, 60.0, 60.0, 20.0],
        },
        index=index,
    )
    compiled = CompiledStrategy(strategy, indicators)
    assert compiled.long_entry.tolist() == [False, False, True, True]


def test_nested_groups_roundtrip_from_llm_style_dict() -> None:
    payload = {
        "logic": "ALL",
        "conditions": [
            {"indicator": "close", "operator": ">", "value": "sma_50"},
            {
                "logic": "ANY",
                "conditions": [
                    {"indicator": "rsi_14", "operator": ">", "value": 60},
                    {"indicator": "rsi_14", "operator": "<", "value": 30},
                ],
            },
        ],
    }
    parsed = ConditionGroup.model_validate(payload)
    assert isinstance(parsed.conditions[1], ConditionGroup)
    assert ConditionGroup.model_validate_json(parsed.model_dump_json()) == parsed


def test_warmup_rows_emit_false_not_ambiguous_signals() -> None:
    base = make_simple_strategy()
    group = ConditionGroup(
        logic="ALL",
        conditions=[Condition(indicator="close", operator=">", value="sma_200")],
    )
    strategy = base.model_copy(update={"entry": EntryRules(long_conditions=group)})
    compiled = StrategyCompiler().compile(strategy, make_ohlcv(210))
    assert not compiled.long_entry.iloc[:199].any()


def test_compilation_does_not_mutate_market_data() -> None:
    data = make_ohlcv(60)
    original = data.copy(deep=True)
    StrategyCompiler().compile(make_simple_strategy(), data)
    pd.testing.assert_frame_equal(data, original)


def test_missing_volume_is_rejected_before_indicator_calculation() -> None:
    with pytest.raises(MissingMarketDataColumnsError, match="Volume"):
        StrategyCompiler().compile(
            make_simple_strategy(), make_ohlcv(30).drop(columns="Volume")
        )


def test_duplicate_timestamps_are_rejected() -> None:
    data = make_ohlcv(30)
    data.index = pd.DatetimeIndex([data.index[0], data.index[0], *data.index[2:]])
    with pytest.raises(DuplicateTimestampError):
        StrategyCompiler().compile(make_simple_strategy(), data)


def test_invalid_indicator_cannot_fail_silently() -> None:
    base = make_simple_strategy()
    invalid = Condition.model_construct(
        indicator="future_alpha", operator=">", value=0.0
    )
    group = ConditionGroup.model_construct(logic="ALL", conditions=[invalid])
    entry = EntryRules.model_construct(long_conditions=group)
    strategy = base.model_copy(update={"entry": entry})
    with pytest.raises(IndicatorResolutionError, match="future_alpha"):
        StrategyCompiler().compile(strategy, make_ohlcv(30))


def test_invalid_operator_cannot_fail_silently() -> None:
    base = make_simple_strategy()
    invalid = Condition.model_construct(
        indicator="close", operator="approximately", value=100.0
    )
    group = ConditionGroup.model_construct(logic="ALL", conditions=[invalid])
    entry = EntryRules.model_construct(long_conditions=group)
    strategy = base.model_copy(update={"entry": entry})
    with pytest.raises(OperatorResolutionError, match="approximately"):
        StrategyCompiler().compile(strategy, make_ohlcv(30))


def test_semantic_validation_cannot_be_bypassed() -> None:
    base = make_simple_strategy()
    contradictory = ConditionGroup(
        logic="ALL",
        conditions=[
            Condition(indicator="rsi_14", operator=">", value=70),
            Condition(indicator="rsi_14", operator="<", value=30),
        ],
    )
    strategy = base.model_copy(
        update={"entry": EntryRules(long_conditions=contradictory)}
    )
    with pytest.raises(StrategySemanticValidationError, match="semantic"):
        StrategyCompiler().compile(strategy, make_ohlcv(30))


def test_future_data_changes_do_not_change_prior_signals() -> None:
    base = make_simple_strategy()
    group = ConditionGroup(
        logic="ALL",
        conditions=[Condition(indicator="close", operator=">", value="sma_20")],
    )
    strategy = base.model_copy(update={"entry": EntryRules(long_conditions=group)})
    original = make_ohlcv(220)
    changed = original.copy(deep=True)
    changed.iloc[120:, changed.columns.get_indexer(["Open", "High", "Low", "Close"])] += 500

    first = StrategyCompiler().compile(strategy, original)
    second = StrategyCompiler().compile(strategy, changed)
    pd.testing.assert_series_equal(
        first.long_entry.iloc[:120], second.long_entry.iloc[:120]
    )


def test_repeated_compilation_is_deterministic() -> None:
    strategy = make_simple_strategy()
    data = make_ohlcv(120)
    first = StrategyCompiler().compile(strategy, data)
    second = StrategyCompiler().compile(strategy, data)
    pd.testing.assert_frame_equal(first.signals, second.signals)
