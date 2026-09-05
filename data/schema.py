"""Canonical validation and normalization for deterministic OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CANONICAL_OHLCV_COLUMNS: tuple[str, ...] = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
)
_CANONICAL_BY_CASEFOLD = {name.casefold(): name for name in CANONICAL_OHLCV_COLUMNS}


class MarketDataValidationError(ValueError):
    """Base class for invalid market data."""


class EmptyMarketDataError(MarketDataValidationError):
    """Raised when no market bars are supplied."""


class MissingMarketDataColumnsError(MarketDataValidationError):
    """Raised when one or more canonical OHLCV fields are absent."""


class MarketDataIndexError(MarketDataValidationError):
    """Raised when the timestamp index is not valid or ordered."""


class DuplicateTimestampError(MarketDataIndexError):
    """Raised when two bars have the same timestamp."""


class InvalidMarketDataValueError(MarketDataValidationError):
    """Raised when an OHLCV value is missing, non-finite, or impossible."""


@dataclass(frozen=True)
class MarketDataContract:
    """Configuration for validating a provider-independent OHLCV frame."""

    require_timezone: bool = False
    allow_extra_columns: bool = True


def validate_ohlcv(
    data: pd.DataFrame,
    *,
    contract: MarketDataContract | None = None,
) -> pd.DataFrame:
    """Return a canonical, validated copy while preserving the input index.

    Column matching is case-insensitive.  The returned core columns use
    ``Open, High, Low, Close, Volume`` and retain any explicitly allowed extra
    columns.  The source DataFrame is never mutated.
    """
    if not isinstance(data, pd.DataFrame):
        raise MarketDataValidationError(
            f"OHLCV data must be a pandas DataFrame, got {type(data).__name__}."
        )
    if data.empty:
        raise EmptyMarketDataError("OHLCV data cannot be empty; at least one bar is required.")

    contract = contract or MarketDataContract()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise MarketDataIndexError("OHLCV data must use a DatetimeIndex.")
    if data.index.hasnans:
        raise MarketDataIndexError("OHLCV DatetimeIndex contains missing timestamps.")
    if data.index.has_duplicates:
        duplicates = data.index[data.index.duplicated()].unique()
        raise DuplicateTimestampError(
            f"OHLCV data contains duplicate timestamps: {list(duplicates[:5])}."
        )
    if not data.index.is_monotonic_increasing:
        raise MarketDataIndexError("OHLCV timestamps must be strictly ascending.")
    if contract.require_timezone and data.index.tz is None:
        raise MarketDataIndexError("OHLCV timestamps must be timezone-aware.")

    rename: dict[object, str] = {}
    seen: dict[str, object] = {}
    for column in data.columns:
        folded = str(column).casefold()
        if folded in _CANONICAL_BY_CASEFOLD:
            canonical = _CANONICAL_BY_CASEFOLD[folded]
            if canonical in seen:
                raise MarketDataValidationError(
                    f"Multiple columns map to canonical field {canonical!r}: "
                    f"{seen[canonical]!r} and {column!r}."
                )
            seen[canonical] = column
            rename[column] = canonical

    missing = [name for name in CANONICAL_OHLCV_COLUMNS if name not in seen]
    if missing:
        raise MissingMarketDataColumnsError(
            f"Missing required OHLCV columns: {missing}."
        )

    if not contract.allow_extra_columns:
        extras = [column for column in data.columns if column not in rename]
        if extras:
            raise MarketDataValidationError(
                f"Unexpected market-data columns: {list(map(str, extras))}."
            )

    normalized = data.rename(columns=rename).copy(deep=True)
    core = normalized.loc[:, CANONICAL_OHLCV_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    if core.isna().any().any() or not np.isfinite(core.to_numpy(dtype=float)).all():
        raise InvalidMarketDataValueError(
            "OHLCV columns must contain only finite numeric values."
        )
    if (core["Volume"] < 0).any():
        raise InvalidMarketDataValueError("OHLCV Volume cannot be negative.")
    if (core[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise InvalidMarketDataValueError("OHLC prices must be greater than zero.")
    if (core["High"] < core["Low"]).any():
        raise InvalidMarketDataValueError("OHLCV High cannot be below Low.")
    if (core["High"] < core[["Open", "Close"]].max(axis=1)).any():
        raise InvalidMarketDataValueError("OHLCV High cannot be below Open or Close.")
    if (core["Low"] > core[["Open", "Close"]].min(axis=1)).any():
        raise InvalidMarketDataValueError("OHLCV Low cannot be above Open or Close.")

    for column in CANONICAL_OHLCV_COLUMNS:
        normalized[column] = core[column]
    ordered = list(CANONICAL_OHLCV_COLUMNS) + [
        column for column in normalized.columns if column not in CANONICAL_OHLCV_COLUMNS
    ]
    return normalized.loc[:, ordered]
