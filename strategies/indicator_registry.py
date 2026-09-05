"""Explicit causal indicator functions and their immutable allow-list."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Callable, Mapping

import pandas as pd

IndicatorFunction = Callable[..., pd.Series]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def identity(series: pd.Series) -> pd.Series:
    """Return a numeric copy of an explicit OHLCV input Series."""
    return _numeric(series).copy(deep=True)


def sma(series: pd.Series, *, period: int) -> pd.Series:
    """Causal simple moving average with a full-window warm-up."""
    return _numeric(series).rolling(period, min_periods=period).mean()


def ema(series: pd.Series, *, period: int) -> pd.Series:
    """Causal exponential moving average with an explicit span."""
    return _numeric(series).ewm(
        span=period, adjust=False, min_periods=period
    ).mean()


def rsi(series: pd.Series, *, period: int = 14) -> pd.Series:
    """Wilder-style relative strength index using historical deltas only."""
    delta = _numeric(series).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = gain.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    average_loss = loss.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    relative_strength = average_gain / average_loss
    result = 100.0 - (100.0 / (1.0 + relative_strength))
    result = result.where(average_loss.ne(0.0), 100.0)
    both_flat = average_gain.eq(0.0) & average_loss.eq(0.0)
    return result.where(~both_flat, 50.0)


def macd_line(
    series: pd.Series, *, fast_period: int = 12, slow_period: int = 26
) -> pd.Series:
    """MACD line from explicit causal fast and slow EMAs."""
    return ema(series, period=fast_period) - ema(series, period=slow_period)


def macd_signal(
    series: pd.Series,
    *,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.Series:
    """Signal EMA of the causal MACD line."""
    line = macd_line(series, fast_period=fast_period, slow_period=slow_period)
    return line.ewm(
        span=signal_period, adjust=False, min_periods=signal_period
    ).mean()


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    period: int = 14,
) -> pd.Series:
    """Wilder-style average true range using only prior close and current bar."""
    high_values = _numeric(high)
    low_values = _numeric(low)
    previous_close = _numeric(close).shift(1)
    true_range = pd.concat(
        [
            high_values - low_values,
            (high_values - previous_close).abs(),
            (low_values - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()


def bollinger_middle(series: pd.Series, *, period: int = 20) -> pd.Series:
    return sma(series, period=period)


def bollinger_upper(
    series: pd.Series, *, period: int = 20, deviations: float = 2.0
) -> pd.Series:
    values = _numeric(series)
    middle = values.rolling(period, min_periods=period).mean()
    deviation = values.rolling(period, min_periods=period).std(ddof=0)
    return middle + deviations * deviation


def bollinger_lower(
    series: pd.Series, *, period: int = 20, deviations: float = 2.0
) -> pd.Series:
    values = _numeric(series)
    middle = values.rolling(period, min_periods=period).mean()
    deviation = values.rolling(period, min_periods=period).std(ddof=0)
    return middle - deviations * deviation


@dataclass(frozen=True)
class IndicatorSpec:
    """Inputs, deterministic function, and minimum bars for one indicator."""

    inputs: tuple[str, ...]
    function: IndicatorFunction
    warmup_bars: int


INDICATOR_REGISTRY: Mapping[str, IndicatorSpec] = MappingProxyType(
    {
        "open": IndicatorSpec(("open",), identity, 1),
        "high": IndicatorSpec(("high",), identity, 1),
        "low": IndicatorSpec(("low",), identity, 1),
        "close": IndicatorSpec(("close",), identity, 1),
        "volume": IndicatorSpec(("volume",), identity, 1),
        "sma_20": IndicatorSpec(("close",), partial(sma, period=20), 20),
        "sma_50": IndicatorSpec(("close",), partial(sma, period=50), 50),
        "sma_200": IndicatorSpec(("close",), partial(sma, period=200), 200),
        "ema_10": IndicatorSpec(("close",), partial(ema, period=10), 10),
        "ema_20": IndicatorSpec(("close",), partial(ema, period=20), 20),
        "ema_50": IndicatorSpec(("close",), partial(ema, period=50), 50),
        "rsi_14": IndicatorSpec(("close",), partial(rsi, period=14), 15),
        "macd": IndicatorSpec(("close",), macd_line, 26),
        "macd_signal": IndicatorSpec(("close",), macd_signal, 34),
        "atr_14": IndicatorSpec(
            ("high", "low", "close"), partial(atr, period=14), 14
        ),
        "bollinger_upper": IndicatorSpec(("close",), bollinger_upper, 20),
        "bollinger_middle": IndicatorSpec(("close",), bollinger_middle, 20),
        "bollinger_lower": IndicatorSpec(("close",), bollinger_lower, 20),
        "volume_sma_20": IndicatorSpec(
            ("volume",), partial(sma, period=20), 20
        ),
    }
)
