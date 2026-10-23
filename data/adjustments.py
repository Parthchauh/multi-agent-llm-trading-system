"""Normalize OHLCV using explicit point-in-time vendor adjustment factors.

No dividend amount or split ratio is guessed from price jumps. Factors must be
supplied under the provider's declared methodology and availability timestamps.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from data.schema import validate_ohlcv
from data.snapshots import canonical_bytes


def normalize_with_factors(raw: pd.DataFrame, factors: pd.DataFrame, *, as_of: pd.Timestamp,
                           basis: str, source: str, method_version: str) -> tuple[pd.DataFrame, dict]:
    frame = validate_ohlcv(raw)
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None or frame.index.tz is None:
        raise ValueError('Normalization requires timezone-aware bars and as-of timestamp.')
    if frame.index[-1] > cutoff:
        raise ValueError('Market observations extend beyond the point-in-time cutoff.')
    if basis not in {'split_adjusted', 'total_return_adjusted'} or not source or not method_version:
        raise ValueError('A named vendor adjustment policy and version are required.')
    if not isinstance(factors, pd.DataFrame) or not factors.index.equals(frame.index):
        raise ValueError('Adjustment factors must match every raw bar exactly.')
    if set(factors.columns) != {'price_factor', 'volume_factor', 'available_at'}:
        raise ValueError('Require price_factor, volume_factor and available_at columns.')
    values = factors[['price_factor', 'volume_factor']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Adjustment factors must be positive and finite.')
    available = [pd.Timestamp(value) for value in factors['available_at']]
    if any(stamp.tzinfo is None or pd.isna(stamp) or stamp > cutoff for stamp in available):
        raise ValueError('Factors are missing, timezone-ambiguous or unavailable at the as-of cutoff.')
    adjusted = frame.copy(deep=True)
    adjusted[['Open', 'High', 'Low', 'Close']] = adjusted[['Open', 'High', 'Low', 'Close']].mul(factors['price_factor'], axis=0)
    adjusted['Volume'] = adjusted['Volume']*factors['volume_factor']
    adjusted = validate_ohlcv(adjusted)
    trace = {'basis': basis, 'source': source, 'method_version': method_version,
             'as_of': cutoff.isoformat(),
             'price_factors': values[:, 0].tolist(), 'volume_factors': values[:, 1].tolist(),
             'available_at': [stamp.isoformat() for stamp in available],
             'point_in_time_vendor_claim_independently_verified': False,
             'limitations': ['Vendor factors and licensed availability must be independently reviewed.',
                             'Rebuild factors at each fold cutoff; latest adjusted history is not past-known history.']}
    trace['sha256'] = hashlib.sha256(canonical_bytes(trace)).hexdigest()
    return adjusted, trace
