"""Contract and deterministic fallback tests for Sprint 4 regime models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from regime.classifier import DeterministicRegimeClassifier
from regime.features import RegimeFeatureEngine
from regime.models import (
    AssessmentSource,
    CompositeRegime,
    RegimeAssessment,
    TrendState,
    VolatilityState,
)


def make_prices(kind: str, n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    if kind == "bullish":
        close = np.linspace(100.0, 180.0, n) + np.sin(np.arange(n) / 7.0)
    elif kind == "bearish":
        close = np.linspace(180.0, 100.0, n) + np.sin(np.arange(n) / 7.0)
    elif kind == "sideways":
        close = 100.0 + np.sin(np.arange(n) / 5.0)
    elif kind == "high_volatility":
        returns = rng.normal(0.0002, 0.045, n)
        close = 100.0 * np.cumprod(1.0 + np.clip(returns, -0.12, 0.12))
    elif kind == "low_volatility":
        close = 100.0 * np.cumprod(np.full(n, 1.0001))
    else:
        raise ValueError(kind)
    open_values = close * 0.999
    high = np.maximum(open_values, close) * 1.005
    low = np.minimum(open_values, close) * 0.995
    return pd.DataFrame(
        {
            "Open": open_values,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.linspace(100_000, 120_000, n),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def test_assessment_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        RegimeAssessment(
            regime="mixed",
            trend_state="neutral",
            volatility_state="normal",
            momentum_state="neutral",
            confidence=1.1,
            reasoning="invalid confidence",
            source="LLM",
        )


def test_assessment_confidence_rejects_boolean_coercion() -> None:
    with pytest.raises(ValidationError, match="not boolean"):
        RegimeAssessment(
            regime="mixed",
            trend_state="neutral",
            volatility_state="normal",
            momentum_state="neutral",
            confidence=True,
            reasoning="invalid confidence type",
            source="LLM",
        )


@pytest.mark.parametrize(
    ("kind", "expected_trend"),
    [("bullish", TrendState.BULLISH), ("bearish", TrendState.BEARISH)],
)
def test_fallback_detects_directional_trends(kind: str, expected_trend: TrendState) -> None:
    snapshot = RegimeFeatureEngine().snapshot(make_prices(kind), symbol="TEST")
    assessment = DeterministicRegimeClassifier().classify(snapshot)
    assert assessment.trend_state is expected_trend
    assert assessment.source is AssessmentSource.DETERMINISTIC_FALLBACK


def test_fallback_detects_sideways_market() -> None:
    snapshot = RegimeFeatureEngine().snapshot(make_prices("sideways"), symbol="TEST")
    assessment = DeterministicRegimeClassifier().classify(snapshot)
    assert assessment.trend_state is TrendState.NEUTRAL
    assert assessment.regime in {
        CompositeRegime.RANGE_BOUND,
        CompositeRegime.LOW_VOLATILITY,
    }


def test_fallback_detects_high_volatility() -> None:
    snapshot = RegimeFeatureEngine().snapshot(
        make_prices("high_volatility"), symbol="TEST"
    )
    assessment = DeterministicRegimeClassifier().classify(snapshot)
    assert assessment.volatility_state is VolatilityState.HIGH
    assert assessment.regime is CompositeRegime.HIGH_VOLATILITY


def test_fallback_detects_low_volatility() -> None:
    snapshot = RegimeFeatureEngine().snapshot(
        make_prices("low_volatility"), symbol="TEST"
    )
    assessment = DeterministicRegimeClassifier().classify(snapshot)
    assert assessment.volatility_state is VolatilityState.LOW


def test_insufficient_warmup_is_explicit_not_backfilled() -> None:
    snapshot = RegimeFeatureEngine().snapshot(
        make_prices("sideways", n=10), symbol="TEST"
    )
    assert snapshot.close_vs_sma20_pct is None
    assert snapshot.return_60d_pct is None
    assessment = DeterministicRegimeClassifier().classify(snapshot)
    assert assessment.trend_state is TrendState.NEUTRAL
