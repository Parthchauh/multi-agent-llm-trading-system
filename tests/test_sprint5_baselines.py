"""Tests for the deterministic, engine-backed Sprint 5 baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backtesting.models import BacktestConfig, ExitReason
from evaluation.baselines import (
    BaselineKind,
    BuyAndHoldSignalProvider,
    build_momentum_strategy,
    build_rsi_mean_reversion_strategy,
    build_sma_crossover_strategy,
    run_all_baselines,
    run_baseline,
)
from strategies.compiler import StrategyCompiler
from strategies.validator import StrategyValidator


def _benchmark_ohlcv(n_bars: int = 320) -> pd.DataFrame:
    """A deterministic oscillating trend with enough history for SMA warm-up."""
    index = pd.date_range("2020-01-02", periods=n_bars, freq="B")
    steps = np.arange(n_bars, dtype=float)
    close = 100.0 + 0.12 * steps + 8.0 * np.sin(steps / 7.0)
    open_price = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_price, close) + 1.0
    low = np.minimum(open_price, close) - 1.0
    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": 100_000.0 + steps,
        },
        index=index,
    )


@pytest.fixture()
def config() -> BacktestConfig:
    return BacktestConfig(
        initial_capital=50_000.0,
        commission_pct=0.001,
        slippage_pct=0.0005,
        position_size_pct=35.0,
    )


@pytest.mark.parametrize(
    "builder",
    [
        build_sma_crossover_strategy,
        build_rsi_mean_reversion_strategy,
        build_momentum_strategy,
    ],
)
def test_declarative_baselines_pass_the_canonical_validator(
    builder, config: BacktestConfig
) -> None:
    strategy = builder(config)

    assert StrategyValidator().validate(strategy).is_valid
    assert strategy.position_sizing.value == config.position_size_pct
    assert strategy.exit.maximum_holding_days == config.maximum_holding_days


def test_all_baselines_share_the_same_window_and_execution_config(
    config: BacktestConfig,
) -> None:
    data = _benchmark_ohlcv()
    evaluation_start = data.index[100]

    runs = run_all_baselines(data, config, evaluation_start=evaluation_start)

    assert [run.kind for run in runs] == list(BaselineKind)
    assert all(run.result.config == config for run in runs)
    assert all(run.result.start_date == evaluation_start.date() for run in runs)
    assert runs[0].kind is BaselineKind.BUY_AND_HOLD
    assert runs[0].strategy is None
    assert all(run.strategy is not None for run in runs[1:])


def test_buy_and_hold_uses_the_engine_next_bar_fill_and_eod_close() -> None:
    data = _benchmark_ohlcv(8)
    config = BacktestConfig(
        initial_capital=10_000.0,
        commission_pct=0.0,
        slippage_pct=0.0,
        position_size_pct=100.0,
    )
    evaluation_start = data.index[1]

    run = run_baseline(
        BaselineKind.BUY_AND_HOLD,
        data,
        config,
        evaluation_start=evaluation_start,
    )

    assert len(run.result.trades) == 1
    trade = run.result.trades[0]
    assert trade.entry_signal_date == evaluation_start.date()
    assert trade.entry_date == data.index[2].date()
    assert trade.exit_date == data.index[-1].date()
    assert trade.exit_reason is ExitReason.END_OF_DATA


def test_buy_and_hold_provider_accepts_only_its_bounded_history_window() -> None:
    provider = BuyAndHoldSignalProvider(entry_index=3)
    history = _benchmark_ohlcv(4)

    assert provider.should_enter(3, history)
    assert not provider.should_enter(2, history.iloc[:3])
    assert not provider.should_enter(3, _benchmark_ohlcv(12))
    assert not provider.should_exit(3, history)


def test_baseline_runs_are_deterministic(config: BacktestConfig) -> None:
    data = _benchmark_ohlcv()
    first = run_all_baselines(data, config, evaluation_start=data.index[80])
    second = run_all_baselines(data, config, evaluation_start=data.index[80])

    assert [run.result.to_dict() for run in first] == [run.result.to_dict() for run in second]


def test_declarative_baseline_signals_do_not_depend_on_future_market_rows(
    config: BacktestConfig,
) -> None:
    original = _benchmark_ohlcv()
    changed_future = original.copy(deep=True)
    split_index = 240
    future_close = changed_future["Close"].iloc[split_index:] * 1.8
    changed_future.loc[future_close.index, "Open"] = future_close
    changed_future.loc[future_close.index, "Close"] = future_close
    changed_future.loc[future_close.index, "High"] = future_close * 1.01
    changed_future.loc[future_close.index, "Low"] = future_close * 0.99

    for builder in (
        build_sma_crossover_strategy,
        build_rsi_mean_reversion_strategy,
        build_momentum_strategy,
    ):
        strategy = builder(config)
        before = StrategyCompiler().compile(strategy, original).signals
        after = StrategyCompiler().compile(strategy, changed_future).signals
        pdt.assert_frame_equal(before.iloc[:split_index], after.iloc[:split_index])
