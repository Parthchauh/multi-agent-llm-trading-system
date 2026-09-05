"""Causal market-regime feature engine built from explicit Pandas operations."""

from __future__ import annotations

import math

import pandas as pd

from data.schema import validate_ohlcv
from regime.exceptions import RegimeFeatureError
from regime.models import RegimeFeatures
from strategies.indicator_registry import atr

FEATURE_COLUMNS: tuple[str, ...] = (
    "close",
    "sma_20",
    "sma_50",
    "sma_200",
    "close_vs_sma20_pct",
    "close_vs_sma50_pct",
    "close_vs_sma200_pct",
    "sma20_slope_pct",
    "sma50_slope_pct",
    "sma50_vs_sma200_pct",
    "return_20d_pct",
    "return_60d_pct",
    "realized_volatility_20d_pct",
    "atr_14_pct",
    "distance_from_high_20d_pct",
    "recent_max_drawdown_60d_pct",
    "volume_trend_20d_pct",
)


class RegimeFeatureEngine:
    """Compute an aligned feature frame and compact latest-bar snapshot."""

    def __init__(self, *, annualization_factor: int = 252) -> None:
        if annualization_factor <= 0:
            raise RegimeFeatureError("annualization_factor must be positive.")
        self._annualization_factor = annualization_factor

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        market = validate_ohlcv(data)
        close = market["Close"].astype(float)
        high = market["High"].astype(float)
        low = market["Low"].astype(float)
        volume = market["Volume"].astype(float)

        sma_20 = close.rolling(20, min_periods=20).mean()
        sma_50 = close.rolling(50, min_periods=50).mean()
        sma_200 = close.rolling(200, min_periods=200).mean()
        daily_return = close.pct_change(fill_method=None)
        rolling_high_20 = close.rolling(20, min_periods=20).max()
        rolling_high_60 = close.rolling(60, min_periods=60).max()
        drawdown_from_60d_high = (close / rolling_high_60 - 1.0) * 100.0
        volume_sma_20 = volume.rolling(20, min_periods=20).mean()
        atr_14 = atr(high, low, close, period=14)

        result = pd.DataFrame(index=market.index)
        result["close"] = close
        result["sma_20"] = sma_20
        result["sma_50"] = sma_50
        result["sma_200"] = sma_200
        result["close_vs_sma20_pct"] = (close / sma_20 - 1.0) * 100.0
        result["close_vs_sma50_pct"] = (close / sma_50 - 1.0) * 100.0
        result["close_vs_sma200_pct"] = (close / sma_200 - 1.0) * 100.0
        result["sma20_slope_pct"] = sma_20.pct_change(
            periods=5, fill_method=None
        ) * 100.0
        result["sma50_slope_pct"] = sma_50.pct_change(
            periods=10, fill_method=None
        ) * 100.0
        result["sma50_vs_sma200_pct"] = (sma_50 / sma_200 - 1.0) * 100.0
        result["return_20d_pct"] = close.pct_change(
            periods=20, fill_method=None
        ) * 100.0
        result["return_60d_pct"] = close.pct_change(
            periods=60, fill_method=None
        ) * 100.0
        result["realized_volatility_20d_pct"] = (
            daily_return.rolling(20, min_periods=20).std(ddof=0)
            * math.sqrt(self._annualization_factor)
            * 100.0
        )
        result["atr_14_pct"] = atr_14 / close * 100.0
        result["distance_from_high_20d_pct"] = (
            close / rolling_high_20 - 1.0
        ) * 100.0
        result["recent_max_drawdown_60d_pct"] = drawdown_from_60d_high.rolling(
            60, min_periods=60
        ).min()
        result["volume_trend_20d_pct"] = (volume / volume_sma_20 - 1.0) * 100.0

        if not result.index.equals(data.index):
            raise RegimeFeatureError("Regime features changed the market-data index.")
        return result.loc[:, FEATURE_COLUMNS]

    def snapshot(self, data: pd.DataFrame, *, symbol: str) -> RegimeFeatures:
        features = self.compute(data)
        latest = features.iloc[-1]

        def optional_float(name: str) -> float | None:
            value = latest[name]
            if pd.isna(value):
                return None
            number = float(value)
            return number if math.isfinite(number) else None

        timestamp = features.index[-1]
        return RegimeFeatures(
            symbol=symbol.strip().upper(),
            timestamp=timestamp.to_pydatetime(),
            observations=len(features),
            close=float(latest["close"]),
            close_vs_sma20_pct=optional_float("close_vs_sma20_pct"),
            close_vs_sma50_pct=optional_float("close_vs_sma50_pct"),
            close_vs_sma200_pct=optional_float("close_vs_sma200_pct"),
            sma20_slope_pct=optional_float("sma20_slope_pct"),
            sma50_slope_pct=optional_float("sma50_slope_pct"),
            sma50_vs_sma200_pct=optional_float("sma50_vs_sma200_pct"),
            return_20d_pct=optional_float("return_20d_pct"),
            return_60d_pct=optional_float("return_60d_pct"),
            realized_volatility_20d_pct=optional_float(
                "realized_volatility_20d_pct"
            ),
            atr_14_pct=optional_float("atr_14_pct"),
            distance_from_high_20d_pct=optional_float(
                "distance_from_high_20d_pct"
            ),
            recent_max_drawdown_60d_pct=optional_float(
                "recent_max_drawdown_60d_pct"
            ),
            volume_trend_20d_pct=optional_float("volume_trend_20d_pct"),
        )
