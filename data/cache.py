"""Optional, provider-independent cache primitives for validated OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from threading import RLock
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class MarketDataCacheKey:
    symbol: str
    start: date | datetime | str
    end: date | datetime | str
    interval: str = "1d"

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "start", str(pd.Timestamp(self.start)))
        object.__setattr__(self, "end", str(pd.Timestamp(self.end)))


class MarketDataCache(Protocol):
    def get(self, key: MarketDataCacheKey) -> pd.DataFrame | None: ...

    def set(self, key: MarketDataCacheKey, data: pd.DataFrame) -> None: ...


class InMemoryMarketDataCache:
    """Thread-safe cache that never exposes its stored mutable DataFrames."""

    def __init__(self) -> None:
        self._items: dict[MarketDataCacheKey, pd.DataFrame] = {}
        self._lock = RLock()

    def get(self, key: MarketDataCacheKey) -> pd.DataFrame | None:
        with self._lock:
            item = self._items.get(key)
            return None if item is None else item.copy(deep=True)

    def set(self, key: MarketDataCacheKey, data: pd.DataFrame) -> None:
        with self._lock:
            self._items[key] = data.copy(deep=True)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
