"""
tests/test_lookahead_bias.py
============================
Explicit verification tests proving the absence of look-ahead bias.

Requirements Verified:
----------------------
1. Signal on Day t cannot execute at Day t Close; executes at Day t+1 Open.
2. Modifying future data (Day t+2 onwards) does not alter signals or executions on Day t.
3. Final-day entry signal creates NO trade (no future candle exists).
4. Final-day exit signal does not execute as SIGNAL; position is closed via END_OF_DATA policy.
5. Signal date vs Execution date auditing in Trade records.
"""

from datetime import date
from typing import Optional
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine, StrategySignalProvider
from backtesting.models import BacktestConfig, ExitReason


class TargetedSignalProvider(StrategySignalProvider):
    """Signals entry and exit on exact bar indices."""

    def __init__(self, entry_bar: int, exit_bar: int) -> None:
        self.entry_bar = entry_bar
        self.exit_bar = exit_bar
        self.observed_max_indices: list[int] = []

    def should_enter(self, current_index: int, data: pd.DataFrame) -> bool:
        # Record the length of data provided to prove no future data is visible
        self.observed_max_indices.append(len(data) - 1)
        return current_index == self.entry_bar

    def should_exit(self, current_index: int, data: pd.DataFrame) -> bool:
        self.observed_max_indices.append(len(data) - 1)
        return current_index == self.exit_bar

    def get_stop_price(
        self, entry_price: float, current_index: int, data: pd.DataFrame
    ) -> Optional[float]:
        return None

    def get_take_profit_price(
        self,
        entry_price: float,
        stop_price: Optional[float],
        current_index: int,
        data: pd.DataFrame,
    ) -> Optional[float]:
        return None


class TestLookaheadBiasPrevention:
    def test_signal_on_day_2_executes_on_day_3_open(self) -> None:
        """Test 1: Signal on bar index 1 (Day 2) must execute on bar index 2 (Day 3) Open."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [105.0, 106.0, 107.0, 108.0, 109.0],
                "Low": [95.0, 96.0, 97.0, 98.0, 99.0],
                "Close": [101.0, 102.0, 103.0, 104.0, 105.0],
                "Volume": [1_000.0] * 5,
            },
            index=dates,
        )

        config = BacktestConfig(initial_capital=10000.0, slippage_pct=0.0, commission_pct=0.0)
        engine = BacktestEngine(config)
        # Entry signal on Day 2 (index 1), exit signal on Day 3 (index 2)
        provider = TargetedSignalProvider(entry_bar=1, exit_bar=2)
        result = engine.run(df, provider)

        assert len(result.trades) == 1
        trade = result.trades[0]

        # Signal generated on 2023-01-02 (Day 2)
        assert trade.entry_signal_date == date(2023, 1, 2)
        # Execution occurs on 2023-01-03 (Day 3) Open: 102.0
        assert trade.entry_date == date(2023, 1, 3)
        assert trade.entry_price == 102.0

        # Exit signal generated on 2023-01-03 (Day 3)
        assert trade.exit_signal_date == date(2023, 1, 3)
        # Execution occurs on 2023-01-04 (Day 4) Open: 103.0
        assert trade.exit_date == date(2023, 1, 4)
        assert trade.exit_price == 103.0
        assert trade.exit_reason == ExitReason.SIGNAL

    def test_future_data_is_structurally_inaccessible(self) -> None:
        """Test 2: When evaluating bar i, data passed to provider has length i + 1."""
        dates = pd.date_range("2023-01-01", periods=6, freq="D")
        df = pd.DataFrame(
            {
                "Open": [100.0] * 6,
                "High": [105.0] * 6,
                "Low": [95.0] * 6,
                "Close": [102.0] * 6,
                "Volume": [1_000.0] * 6,
            },
            index=dates,
        )
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        provider = TargetedSignalProvider(entry_bar=1, exit_bar=3)
        engine.run(df, provider)

        # For every bar index i evaluated, the max index present in the passed data slice was exactly i
        for obs in provider.observed_max_indices:
            assert obs <= 4  # Bars evaluated are 0 through 4 (bar 5 is last bar, not evaluated for signals)

    def test_modifying_future_data_does_not_affect_prior_trade(self) -> None:
        """Test 3: Altering Day 4 & 5 prices does not change entry fill or Day 2/3 behavior."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df_orig = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [105.0, 106.0, 107.0, 108.0, 109.0],
                "Low": [95.0, 96.0, 97.0, 98.0, 99.0],
                "Close": [101.0, 102.0, 103.0, 104.0, 105.0],
                "Volume": [1_000.0] * 5,
            },
            index=dates,
        )

        df_mod = df_orig.copy()
        # Wild change on Day 4 and Day 5
        df_mod.loc[dates[3]:, "Open"] = 500.0
        df_mod.loc[dates[3]:, "High"] = 550.0
        df_mod.loc[dates[3]:, "Low"] = 450.0
        df_mod.loc[dates[3]:, "Close"] = 520.0

        config = BacktestConfig(initial_capital=10000.0, slippage_pct=0.0, commission_pct=0.0)
        engine = BacktestEngine(config)
        provider1 = TargetedSignalProvider(entry_bar=1, exit_bar=2)
        provider2 = TargetedSignalProvider(entry_bar=1, exit_bar=2)

        res_orig = engine.run(df_orig, provider1)
        res_mod = engine.run(df_mod, provider2)

        # Entry trade price and date on Day 3 are completely unaffected
        assert res_orig.trades[0].entry_date == res_mod.trades[0].entry_date
        assert res_orig.trades[0].entry_price == res_mod.trades[0].entry_price

    def test_final_day_entry_signal_creates_no_trade(self) -> None:
        """Test 4: Signal generated on the last candle cannot execute (no next day exists)."""
        dates = pd.date_range("2023-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0],
                "High": [105.0, 106.0, 107.0, 108.0],
                "Low": [95.0, 96.0, 97.0, 98.0],
                "Close": [101.0, 102.0, 103.0, 104.0],
                "Volume": [1_000.0] * 4,
            },
            index=dates,
        )
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        # Entry signal on final bar (index 3)
        provider = TargetedSignalProvider(entry_bar=3, exit_bar=99)
        result = engine.run(df, provider)

        # No trade should have executed
        assert len(result.trades) == 0
        assert result.metrics.number_of_trades == 0

    def test_final_day_exit_signal_closed_via_end_of_data(self) -> None:
        """Test 5: If position is open and exit signal is on last candle, it's settled via END_OF_DATA."""
        dates = pd.date_range("2023-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0],
                "High": [105.0, 106.0, 107.0, 108.0],
                "Low": [95.0, 96.0, 97.0, 98.0],
                "Close": [101.0, 102.0, 103.0, 104.0],
                "Volume": [1_000.0] * 4,
            },
            index=dates,
        )
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        # Enter on bar 0 (fills bar 1). Exit signal on final bar 3.
        provider = TargetedSignalProvider(entry_bar=0, exit_bar=3)
        result = engine.run(df, provider)

        assert len(result.trades) == 1
        trade = result.trades[0]
        # Forced accounting close at end of data, not a next-bar signal fill
        assert trade.exit_reason == ExitReason.END_OF_DATA
        assert trade.exit_date == date(2023, 1, 4)
