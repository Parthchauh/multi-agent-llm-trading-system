"""Causality, alignment, and repeatability tests for regime features."""

from __future__ import annotations

import pandas as pd

from regime.features import FEATURE_COLUMNS, RegimeFeatureEngine
from tests.test_regime_models import make_prices


def test_feature_frame_is_aligned_and_does_not_mutate_input() -> None:
    data = make_prices("bullish")
    original = data.copy(deep=True)
    result = RegimeFeatureEngine().compute(data)
    assert result.index.equals(data.index)
    assert result.columns.tolist() == list(FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(data, original)


def test_warmup_nan_handling_is_causal_and_explicit() -> None:
    result = RegimeFeatureEngine().compute(make_prices("bullish", n=210))
    assert result["sma_20"].iloc[:19].isna().all()
    assert result["sma_50"].iloc[:49].isna().all()
    assert result["sma_200"].iloc[:199].isna().all()
    assert pd.notna(result["sma_200"].iloc[199])


def test_repeated_feature_calculation_is_deterministic() -> None:
    data = make_prices("high_volatility")
    engine = RegimeFeatureEngine()
    pd.testing.assert_frame_equal(engine.compute(data), engine.compute(data))


def test_future_perturbation_does_not_change_prior_features() -> None:
    data = make_prices("bullish", n=300)
    changed = data.copy(deep=True)
    cutoff = 180
    price_columns = ["Open", "High", "Low", "Close"]
    changed.loc[changed.index[cutoff + 1 :], price_columns] *= 4.0
    changed.loc[changed.index[cutoff + 1 :], "Volume"] *= 10.0

    engine = RegimeFeatureEngine()
    baseline = engine.compute(data)
    perturbed = engine.compute(changed)
    pd.testing.assert_frame_equal(
        baseline.iloc[: cutoff + 1], perturbed.iloc[: cutoff + 1]
    )


def test_snapshot_is_compact_and_timestamped_at_latest_bar() -> None:
    data = make_prices("bearish")
    snapshot = RegimeFeatureEngine().snapshot(data, symbol=" xyz ")
    assert snapshot.symbol == "XYZ"
    assert snapshot.observations == len(data)
    assert snapshot.timestamp == data.index[-1].to_pydatetime()
