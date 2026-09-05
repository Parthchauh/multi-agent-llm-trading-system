"""Provider-independent market-data API and a validated Yahoo adapter."""

from __future__ import annotations

from datetime import date, datetime
from typing import Callable, Protocol, runtime_checkable

import pandas as pd

from data.cache import MarketDataCache, MarketDataCacheKey
from data.schema import MarketDataContract, validate_ohlcv

DateLike = date | datetime | str


class MarketDataProviderError(RuntimeError):
    """Raised when a provider cannot return the requested validated data."""


@runtime_checkable
class MarketDataProvider(Protocol):
    """Vendor-neutral OHLCV provider contract."""

    def fetch(
        self,
        symbol: str,
        start: DateLike,
        end: DateLike,
        *,
        interval: str = "1d",
    ) -> pd.DataFrame: ...


class YahooFinanceProvider:
    """Validated adapter around the project's existing yfinance dependency."""

    def __init__(
        self,
        downloader: Callable[..., pd.DataFrame] | None = None,
        *,
        contract: MarketDataContract | None = None,
    ) -> None:
        if downloader is None:
            import yfinance

            downloader = yfinance.download
        self._downloader = downloader
        self._contract = contract or MarketDataContract()

    def fetch(
        self,
        symbol: str,
        start: DateLike,
        end: DateLike,
        *,
        interval: str = "1d",
    ) -> pd.DataFrame:
        clean_symbol = symbol.strip().upper()
        if not clean_symbol:
            raise MarketDataProviderError("A non-empty ticker symbol is required.")
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        if start_ts >= end_ts:
            raise MarketDataProviderError("Provider start must be earlier than end.")
        try:
            frame = self._downloader(
                clean_symbol,
                start=start_ts,
                end=end_ts,
                interval=interval,
                auto_adjust=False,
                progress=False,
                multi_level_index=False,
            )
        except Exception as exc:
            raise MarketDataProviderError(
                f"Yahoo Finance failed for {clean_symbol!r}: {exc}"
            ) from exc
        if not isinstance(frame, pd.DataFrame):
            raise MarketDataProviderError(
                f"Yahoo Finance returned {type(frame).__name__}, expected DataFrame."
            )
        return validate_ohlcv(frame, contract=self._contract)


class CachedMarketDataProvider:
    """Cache decorator that can wrap any ``MarketDataProvider`` implementation."""

    def __init__(self, provider: MarketDataProvider, cache: MarketDataCache) -> None:
        self._provider = provider
        self._cache = cache

    def fetch(
        self,
        symbol: str,
        start: DateLike,
        end: DateLike,
        *,
        interval: str = "1d",
    ) -> pd.DataFrame:
        key = MarketDataCacheKey(symbol, start, end, interval)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        frame = self._provider.fetch(symbol, start, end, interval=interval)
        validated = validate_ohlcv(frame)
        self._cache.set(key, validated)
        return validated.copy(deep=True)
