"""Tests for the provider-independent Sprint 3 OHLCV contract and cache."""

from __future__ import annotations

import pandas as pd
import pytest

from data.cache import InMemoryMarketDataCache
from data.provider import CachedMarketDataProvider, YahooFinanceProvider
from data.schema import (
    DuplicateTimestampError,
    InvalidMarketDataValueError,
    MarketDataContract,
    MarketDataIndexError,
    MissingMarketDataColumnsError,
    validate_ohlcv,
)


def _frame(index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    index = index if index is not None else pd.date_range("2024-01-01", periods=3)
    return pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "High": [11.0, 12.0, 13.0],
            "Low": [9.0, 10.0, 11.0],
            "Close": [10.5, 11.5, 12.5],
            "Volume": [100.0, 110.0, 120.0],
        },
        index=index,
    )


def test_canonicalizes_columns_without_mutating_input() -> None:
    source = _frame().rename(columns=str.lower)
    original = source.copy(deep=True)
    result = validate_ohlcv(source)
    assert result.columns.tolist() == ["Open", "High", "Low", "Close", "Volume"]
    assert result.index.equals(source.index)
    pd.testing.assert_frame_equal(source, original)


def test_timezone_aware_index_is_preserved() -> None:
    index = pd.date_range("2024-01-01", periods=3, tz="UTC")
    result = validate_ohlcv(
        _frame(index), contract=MarketDataContract(require_timezone=True)
    )
    assert result.index.tz is not None
    assert result.index.equals(index)


def test_timezone_can_be_required() -> None:
    with pytest.raises(MarketDataIndexError, match="timezone-aware"):
        validate_ohlcv(_frame(), contract=MarketDataContract(require_timezone=True))


def test_missing_ohlcv_field_raises_typed_error() -> None:
    with pytest.raises(MissingMarketDataColumnsError, match="Volume"):
        validate_ohlcv(_frame().drop(columns="Volume"))


def test_non_datetime_index_is_rejected() -> None:
    data = _frame()
    data.index = pd.RangeIndex(len(data))
    with pytest.raises(MarketDataIndexError, match="DatetimeIndex"):
        validate_ohlcv(data)


def test_duplicate_timestamp_is_rejected() -> None:
    data = _frame(pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"]))
    with pytest.raises(DuplicateTimestampError, match="duplicate"):
        validate_ohlcv(data)


def test_descending_timestamps_are_rejected() -> None:
    with pytest.raises(MarketDataIndexError, match="ascending"):
        validate_ohlcv(_frame().sort_index(ascending=False))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.assign(Close=[10.5, float("nan"), 12.5]),
        lambda data: data.assign(Volume=[100.0, -1.0, 120.0]),
        lambda data: data.assign(High=[8.0, 12.0, 13.0]),
    ],
)
def test_invalid_values_are_rejected(mutation) -> None:
    with pytest.raises(InvalidMarketDataValueError):
        validate_ohlcv(mutation(_frame()))


class _CountingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def fetch(self, symbol, start, end, *, interval="1d") -> pd.DataFrame:
        self.calls += 1
        return _frame()


def test_cache_is_provider_independent_and_copy_safe() -> None:
    source = _CountingProvider()
    provider = CachedMarketDataProvider(source, InMemoryMarketDataCache())
    first = provider.fetch("aapl", "2024-01-01", "2024-02-01")
    first.iloc[0, 0] = 999.0
    second = provider.fetch("AAPL", "2024-01-01", "2024-02-01")
    assert source.calls == 1
    assert second.iloc[0]["Open"] == 10.0


def test_yahoo_adapter_returns_validated_canonical_data() -> None:
    calls: list[tuple[str, dict]] = []

    def downloader(symbol: str, **kwargs) -> pd.DataFrame:
        calls.append((symbol, kwargs))
        return _frame().rename(columns=str.lower)

    result = YahooFinanceProvider(downloader).fetch(
        " aapl ", "2024-01-01", "2024-02-01"
    )
    assert calls[0][0] == "AAPL"
    assert result.columns[:5].tolist() == ["Open", "High", "Low", "Close", "Volume"]
