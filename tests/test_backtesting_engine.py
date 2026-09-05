"""
tests/test_backtesting_engine.py
================================
Tests for backtesting/engine.py (BacktestEngine).
"""

from datetime import date
from typing import Optional
import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine, StrategySignalProvider
from backtesting.models import BacktestConfig, ExitReason


class DummySignalProvider(StrategySignalProvider):
    """Simple test provider: buy on day 1 (index 1), exit on day 4 (index 4)."""

    def __init__(
        self,
        entry_bar: int = 1,
        exit_bar: int = 4,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ) -> None:
        self.entry_bar = entry_bar
        self.exit_bar = exit_bar
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def should_enter(self, current_index: int, data: pd.DataFrame) -> bool:
        return current_index == self.entry_bar

    def should_exit(self, current_index: int, data: pd.DataFrame) -> bool:
        return current_index == self.exit_bar

    def get_stop_price(
        self, entry_price: float, current_index: int, data: pd.DataFrame
    ) -> Optional[float]:
        if self.stop_loss_pct is not None:
            return entry_price * (1.0 - self.stop_loss_pct)
        return None

    def get_take_profit_price(
        self,
        entry_price: float,
        stop_price: Optional[float],
        current_index: int,
        data: pd.DataFrame,
    ) -> Optional[float]:
        if self.take_profit_pct is not None:
            return entry_price * (1.0 + self.take_profit_pct)
        return None


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """7 days of synthetic OHLCV data."""
    dates = pd.date_range("2023-01-01", periods=7, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 105.0, 107.0, 110.0, 108.0, 106.0],
            "High": [103.0, 106.0, 108.0, 111.0, 112.0, 110.0, 109.0],
            "Low": [99.0, 101.0, 104.0, 106.0, 107.0, 105.0, 104.0],
            "Close": [102.0, 105.0, 107.0, 110.0, 108.0, 106.0, 108.0],
            "Volume": [1000, 1200, 1100, 1500, 1300, 1000, 1100],
        },
        index=dates,
    )
    return df


class TestBacktestEngine:
    def test_end_to_end_backtest_run(self, sample_ohlcv: pd.DataFrame) -> None:
        config = BacktestConfig(
            initial_capital=10000.0,
            commission_pct=0.001,
            slippage_pct=0.0005,
            position_size_pct=50.0,
        )
        engine = BacktestEngine(config)
        # Entry signal on bar 1 (2023-01-02), executes bar 2 (2023-01-03) Open: 105.0
        # Exit signal on bar 4 (2023-01-05), executes bar 5 (2023-01-06) Open: 108.0
        provider = DummySignalProvider(entry_bar=1, exit_bar=4)
        result = engine.run(sample_ohlcv, provider)

        assert result.initial_capital == 10000.0
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.entry_date == date(2023, 1, 3)
        assert trade.exit_date == date(2023, 1, 6)
        assert trade.exit_reason == ExitReason.SIGNAL
        assert trade.net_pnl > 0  # Profitable trade: bought at ~105, sold at ~108
        assert len(result.equity_curve) == 7
        assert result.metrics.number_of_trades == 1
        assert result.metrics.win_rate_pct == 100.0

    def test_maximum_holding_days_triggers_exit(self, sample_ohlcv: pd.DataFrame) -> None:
        config = BacktestConfig(
            initial_capital=10000.0,
            maximum_holding_days=2,  # max hold 2 calendar days
        )
        engine = BacktestEngine(config)
        # Enter bar 1 -> fills bar 2 (2023-01-03).
        # On bar 4 (2023-01-05), days held = (2023-01-05 - 2023-01-03) = 2 days >= 2 -> force close
        provider = DummySignalProvider(entry_bar=1, exit_bar=99)  # never signal exit
        result = engine.run(sample_ohlcv, provider)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == ExitReason.MAX_HOLDING_PERIOD
        assert trade.exit_date == date(2023, 1, 5)

    def test_stop_loss_triggered_intraday(self) -> None:
        dates = pd.date_range("2023-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 98.0, 95.0],
                "High": [102.0, 101.0, 99.0, 96.0],
                "Low": [99.0, 99.0, 92.0, 90.0],  # Bar 2 low is 92.0 -> triggers 5% stop at 95.0
                "Close": [100.0, 100.0, 94.0, 91.0],
                "Volume": [1000.0, 1000.0, 1000.0, 1000.0],
            },
            index=dates,
        )
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        # Enter bar 0 -> fills bar 1 at open 100.0. Stop loss at 5% = 95.0
        provider = DummySignalProvider(entry_bar=0, exit_bar=99, stop_loss_pct=0.05)
        result = engine.run(df, provider)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == ExitReason.STOP_LOSS
        assert trade.exit_date == date(2023, 1, 3)

    def test_end_of_data_forces_close(self, sample_ohlcv: pd.DataFrame) -> None:
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        # Enter bar 1 -> fills bar 2. Never exit. Should force close at end of data (bar 6).
        provider = DummySignalProvider(entry_bar=1, exit_bar=99)
        result = engine.run(sample_ohlcv, provider)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == ExitReason.END_OF_DATA
        assert trade.exit_date == date(2023, 1, 7)

    def test_deterministic_output(self, sample_ohlcv: pd.DataFrame) -> None:
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        provider = DummySignalProvider(entry_bar=1, exit_bar=4)

        res1 = engine.run(sample_ohlcv, provider)
        res2 = engine.run(sample_ohlcv, provider)

        assert res1.final_equity == res2.final_equity
        assert res1.metrics.total_return_pct == res2.metrics.total_return_pct
        assert len(res1.trades) == len(res2.trades)


class TestDataValidation:
    def test_empty_dataframe_raises(self) -> None:
        engine = BacktestEngine(BacktestConfig(initial_capital=10000.0))
        provider = DummySignalProvider()
        with pytest.raises(ValueError, match="at least one bar"):
            engine.run(pd.DataFrame(), provider)

    def test_missing_columns_raises(self) -> None:
        engine = BacktestEngine(BacktestConfig(initial_capital=10000.0))
        provider = DummySignalProvider()
        df = pd.DataFrame({"Close": [10, 20]}, index=pd.date_range("2023-01-01", periods=2))
        with pytest.raises(ValueError, match="Missing required OHLCV columns"):
            engine.run(df, provider)

    def test_non_monotonic_dates_raises(self) -> None:
        engine = BacktestEngine(BacktestConfig(initial_capital=10000.0))
        provider = DummySignalProvider()
        dates = [pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-01")]  # descending
        df = pd.DataFrame(
            {"Open": [10.0, 20.0], "High": [12.0, 22.0], "Low": [9.0, 19.0],
             "Close": [11.0, 21.0], "Volume": [100.0, 200.0]},
            index=dates,
        )
        with pytest.raises(ValueError, match="ascending"):
            engine.run(df, provider)

    def test_nan_values_raise(self) -> None:
        engine = BacktestEngine(BacktestConfig(initial_capital=10000.0))
        provider = DummySignalProvider()
        dates = pd.date_range("2023-01-01", periods=2)
        df = pd.DataFrame(
            {"Open": [10.0, np.nan], "High": [12.0, 22.0], "Low": [9.0, 19.0],
             "Close": [11.0, 21.0], "Volume": [100.0, 200.0]},
            index=dates,
        )
        with pytest.raises(ValueError, match="finite numeric"):
            engine.run(df, provider)

    def test_impossible_candle_raises(self) -> None:
        engine = BacktestEngine(BacktestConfig(initial_capital=10000.0))
        provider = DummySignalProvider()
        dates = pd.date_range("2023-01-01", periods=2)
        df = pd.DataFrame(
            {
                "Open": [10, 20], "High": [8, 22], "Low": [9, 19],
                "Close": [11, 21], "Volume": [100.0, 100.0],
            },  # High < Low on bar 0
            index=dates,
        )
        with pytest.raises(ValueError, match="High cannot be below Low"):
            engine.run(df, provider)
