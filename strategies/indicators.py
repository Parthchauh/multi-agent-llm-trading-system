"""
strategies/indicators.py
========================
Maps registry indicator names to explicit causal Pandas computations.

This module is the compatibility-facing translation layer between strategy
indicator names and :mod:`strategies.indicator_registry` functions. The
legacy stockstats-name mapping remains public for backward compatibility but
is not used to execute indicators.

Design constraints
------------------
* No LLM, no agent, no graph imports.
* Receives a plain pandas OHLCV DataFrame; returns a plain pandas DataFrame.
* Never modifies the input DataFrame.
* All computations are deterministic and reproducible given identical input.

Column name mapping
-------------------
Registry name       → stockstats column accessed
──────────────────────────────────────────────────
open / high / …     → raw column (no computation)
sma_20              → close_20_sma
sma_50              → close_50_sma
sma_200             → close_200_sma
ema_10              → close_10_ema
ema_20              → close_20_ema
ema_50              → close_50_ema
rsi_14              → rsi
macd                → macd
macd_signal         → macds
atr_14              → atr
bollinger_upper     → boll_ub
bollinger_middle    → boll
bollinger_lower     → boll_lb
volume_sma_20       → volume_20_sma
"""

from __future__ import annotations

from typing import Set

import pandas as pd

from strategies.exceptions import IndicatorComputationError, IndicatorResolutionError
from strategies.indicator_registry import INDICATOR_REGISTRY, IndicatorSpec
from strategies.registry import SUPPORTED_INDICATORS

# ---------------------------------------------------------------------------
# Column name mapping
# ---------------------------------------------------------------------------

#: Legacy registry-to-stockstats mapping retained as a stable public API.
REGISTRY_TO_STOCKSTATS: dict[str, str] = {
    # Raw OHLCV price columns (no computation needed)
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    # Simple Moving Averages (stockstats: close_{period}_sma)
    "sma_20": "close_20_sma",
    "sma_50": "close_50_sma",
    "sma_200": "close_200_sma",
    # Exponential Moving Averages (stockstats: close_{period}_ema)
    "ema_10": "close_10_ema",
    "ema_20": "close_20_ema",
    "ema_50": "close_50_ema",
    # Momentum
    "rsi_14": "rsi",       # stockstats default RSI period = 14
    "macd": "macd",        # MACD line (fast_ema - slow_ema)
    "macd_signal": "macds",  # MACD signal line (EMA of MACD)
    # Volatility
    "atr_14": "atr",          # Average True Range (default 14-period)
    "bollinger_upper": "boll_ub",
    "bollinger_middle": "boll",
    "bollinger_lower": "boll_lb",
    # Volume
    "volume_sma_20": "volume_20_sma",
}


# Sanity check — every SUPPORTED_INDICATORS entry must have a mapping.
assert set(REGISTRY_TO_STOCKSTATS.keys()) == SUPPORTED_INDICATORS, (
    "REGISTRY_TO_STOCKSTATS is out of sync with SUPPORTED_INDICATORS. "
    "Please update indicators.py whenever registry.py changes."
)
assert set(INDICATOR_REGISTRY) == SUPPORTED_INDICATORS, (
    "INDICATOR_REGISTRY is out of sync with SUPPORTED_INDICATORS."
)


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------


def compute_indicators(
    data: pd.DataFrame,
    required: Set[str],
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """Compute the required indicators for all bars in *data*.

    Parameters
    ----------
    data:
        Raw OHLCV DataFrame.  Column names are case-insensitive.
        Expected columns: Open/open, High/high, Low/low, Close/close, Volume/volume.
    required:
        Set of registry indicator names (from ``SUPPORTED_INDICATORS``) to
        compute. Unknown names are skipped only in legacy non-strict mode.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the same DatetimeIndex as *data* and one column
        per requested indicator, named using the registry name (e.g. ``"rsi_14"``,
        ``"ema_20"``).  Bars that fall in the warmup period for a given
        indicator will have ``NaN`` values — callers should treat NaN as
        "condition cannot be evaluated → False".

    Notes
    -----
    * *data* is never mutated.
    * Explicit Pandas functions receive lowercase input Series.
    * Unknown names retain legacy skip behaviour unless ``strict=True``.
      The strategy compiler always enables strict mode.
    """
    if data.empty:
        return pd.DataFrame()

    # Normalize once for the explicit indicator functions.
    df = data.copy()
    df.columns = [str(c).lower() for c in df.columns]

    result: dict[str, pd.Series] = {}
    for reg_name in sorted(required):
        if reg_name not in SUPPORTED_INDICATORS:
            if strict:
                raise IndicatorResolutionError(
                    f"Unsupported indicator {reg_name!r}; allowed indicators are "
                    f"{sorted(SUPPORTED_INDICATORS)}."
                )
            continue

        spec = INDICATOR_REGISTRY[reg_name]
        missing_inputs = [column for column in spec.inputs if column not in df.columns]
        if missing_inputs:
            if strict:
                raise IndicatorComputationError(
                    f"Indicator {reg_name!r} requires missing input columns "
                    f"{missing_inputs}."
                )
            series = pd.Series(float("nan"), index=df.index, dtype=float)
        else:
            try:
                series = spec.function(*(df[column] for column in spec.inputs))
            except Exception as exc:
                if strict:
                    raise IndicatorComputationError(
                        f"Failed to compute indicator {reg_name!r}: {exc}"
                    ) from exc
                series = pd.Series(float("nan"), index=df.index, dtype=float)

        if not isinstance(series, pd.Series) or not series.index.equals(df.index):
            if strict:
                raise IndicatorComputationError(
                    f"Indicator {reg_name!r} did not preserve the market-data index."
                )
            series = pd.Series(float("nan"), index=df.index, dtype=float)

        series = pd.to_numeric(series, errors="coerce").astype(float)
        if spec.warmup_bars > 1:
            series.iloc[: spec.warmup_bars - 1] = float("nan")
        result[reg_name] = series

    return pd.DataFrame(result, index=df.index)


def collect_required_indicators(strategy: object) -> set[str]:
    """Collect every indicator name referenced by a StrategySchema.

    Traverses entry conditions, exit conditions, and stop-loss type to build
    the complete set of registry indicator names that must be computed before
    the strategy can be evaluated.

    Parameters
    ----------
    strategy:
        A ``strategies.schema.StrategySchema`` instance.  Typed as ``object``
        to avoid a circular import; duck-typed to access ``.entry``,
        ``.exit``, ``.stop_loss``.

    Returns
    -------
    set[str]
        Registry indicator names needed to fully evaluate the strategy.
    """
    indicators: set[str] = set()

    def _collect_from_group(group: object) -> None:
        for node in group.conditions:  # type: ignore[union-attr]
            if hasattr(node, "conditions"):
                _collect_from_group(node)
                continue
            indicators.add(node.indicator)
            if isinstance(node.value, str):  # indicator vs indicator comparison
                indicators.add(node.value)

    _collect_from_group(strategy.entry.long_conditions)  # type: ignore[union-attr]
    _collect_from_group(strategy.exit.exit_conditions)   # type: ignore[union-attr]

    # ATR-based stop loss requires atr_14 at entry time
    if strategy.stop_loss.type == "atr_multiple":          # type: ignore[union-attr]
        indicators.add("atr_14")

    return indicators
