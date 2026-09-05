"""
tests/test_indicators.py
========================
Tests for strategies/indicators.py.

Validates indicator name mapping, computation correctness, and NaN behaviour.
"""

from __future__ import annotations

import math
import pytest
import pandas as pd
import numpy as np

from strategies.indicators import (
    REGISTRY_TO_STOCKSTATS,
    collect_required_indicators,
    compute_indicators,
)
from strategies.indicator_registry import sma
from strategies.registry import SUPPORTED_INDICATORS


class TestRegistryMapping:
    def test_all_supported_indicators_have_mapping(self) -> None:
        """Every indicator in SUPPORTED_INDICATORS must have a stockstats mapping."""
        assert set(REGISTRY_TO_STOCKSTATS.keys()) == SUPPORTED_INDICATORS

    def test_no_extra_mappings(self) -> None:
        """There must be no entries in the map that aren't in SUPPORTED_INDICATORS."""
        extras = set(REGISTRY_TO_STOCKSTATS.keys()) - SUPPORTED_INDICATORS
        assert extras == set()


def _make_ohlcv(n: int = 250, start_price: float = 100.0) -> pd.DataFrame:
    """Generate a simple trending OHLCV DataFrame with n bars."""
    rng = np.random.default_rng(42)
    closes = start_price + np.cumsum(rng.normal(0.1, 1.0, n))
    opens = closes * (1 + rng.normal(0, 0.002, n))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0.001, 0.01, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0.001, 0.01, n))
    volumes = rng.integers(100_000, 500_000, n).astype(float)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=dates,
    )


class TestComputeIndicators:
    def test_empty_data_returns_empty(self) -> None:
        result = compute_indicators(pd.DataFrame(), {"rsi_14"})
        assert result.empty

    def test_price_indicators_returned_directly(self) -> None:
        df = _make_ohlcv(50)
        result = compute_indicators(df, {"close", "open", "volume"})
        assert "close" in result.columns
        assert "open" in result.columns
        assert "volume" in result.columns
        # Close values should match (case-normalised)
        pd.testing.assert_series_equal(
            result["close"].reset_index(drop=True),
            df["Close"].reset_index(drop=True),
            check_names=False,
        )

    def test_sma_computed_correctly(self) -> None:
        """SMA(20) at bar 25 should equal the mean of close prices for bars 6..25."""
        df = _make_ohlcv(60)
        result = compute_indicators(df, {"sma_20"})
        assert "sma_20" in result.columns

        closes = df["Close"].values
        # Bar 25 (0-indexed) — SMA20 = mean of closes[6:26]
        expected_sma = float(np.mean(closes[6:26]))
        computed_sma = float(result["sma_20"].iloc[25])
        assert pytest.approx(computed_sma, rel=1e-4) == expected_sma

    def test_explicit_series_indicator_preserves_index(self) -> None:
        close = _make_ohlcv(25)["Close"]
        result = sma(close, period=20)
        assert result.index.equals(close.index)
        assert result.iloc[:19].isna().all()

    def test_rsi_within_valid_range(self) -> None:
        df = _make_ohlcv(100)
        result = compute_indicators(df, {"rsi_14"})
        assert "rsi_14" in result.columns
        valid_rsi = result["rsi_14"].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_warmup_bars_are_nan(self) -> None:
        """SMA(200) requires 200 bars of data; earlier bars must be NaN."""
        df = _make_ohlcv(210)
        result = compute_indicators(df, {"sma_200"})
        # Bars 0..198 should be NaN; bar 199 onwards should be valid
        assert math.isnan(float(result["sma_200"].iloc[0]))
        assert math.isnan(float(result["sma_200"].iloc[100]))
        assert not math.isnan(float(result["sma_200"].iloc[200]))

    def test_unknown_indicator_skipped(self) -> None:
        df = _make_ohlcv(30)
        result = compute_indicators(df, {"close", "fantasy_index_99"})
        assert "close" in result.columns
        assert "fantasy_index_99" not in result.columns

    def test_atr_computed_positive(self) -> None:
        df = _make_ohlcv(50)
        result = compute_indicators(df, {"atr_14"})
        assert "atr_14" in result.columns
        valid_atr = result["atr_14"].dropna()
        assert (valid_atr > 0).all()

    def test_bollinger_bands_ordering(self) -> None:
        """Upper band ≥ Middle band ≥ Lower band at every non-NaN bar."""
        df = _make_ohlcv(60)
        result = compute_indicators(df, {"bollinger_upper", "bollinger_middle", "bollinger_lower"})
        valid = result.dropna()
        assert (valid["bollinger_upper"] >= valid["bollinger_middle"]).all()
        assert (valid["bollinger_middle"] >= valid["bollinger_lower"]).all()

    def test_macd_and_signal_computed(self) -> None:
        df = _make_ohlcv(100)
        result = compute_indicators(df, {"macd", "macd_signal"})
        assert "macd" in result.columns
        assert "macd_signal" in result.columns


class TestCollectRequiredIndicators:
    def test_collects_entry_and_exit_indicators(self) -> None:
        """collect_required_indicators should find indicators from entry and exit groups."""
        from tests.conftest_helpers import make_simple_strategy
        strategy = make_simple_strategy()
        required = collect_required_indicators(strategy)
        # Strategy uses rsi_14 in entry and close in exit
        assert "rsi_14" in required
        assert "close" in required

    def test_atr_added_for_atr_multiple_stop(self) -> None:
        from strategies.schema import (
            Condition, ConditionGroup, EntryRules, ExitRules,
            PositionSizing, StopLoss, StrategyMetadata, StrategySchema, TakeProfit,
        )
        strategy = StrategySchema(
            metadata=StrategyMetadata(name="ATR Stop Test", market_type="equity", timeframe="1d"),
            entry=EntryRules(
                long_conditions=ConditionGroup(
                    logic="ALL",
                    conditions=[Condition(indicator="close", operator=">", value="sma_50")],
                )
            ),
            exit=ExitRules(
                exit_conditions=ConditionGroup(
                    logic="ANY",
                    conditions=[Condition(indicator="close", operator="<", value="sma_50")],
                )
            ),
            stop_loss=StopLoss(type="atr_multiple", value=2.0),
            take_profit=TakeProfit(type="percentage", value=10.0),
            position_sizing=PositionSizing(type="fixed_percentage", value=10.0),
        )
        required = collect_required_indicators(strategy)
        assert "atr_14" in required

    def test_atr_not_added_for_percentage_stop(self) -> None:
        from tests.conftest_helpers import make_simple_strategy
        strategy = make_simple_strategy()  # uses percentage stop
        required = collect_required_indicators(strategy)
        # percentage stop doesn't need ATR
        assert "atr_14" not in required
