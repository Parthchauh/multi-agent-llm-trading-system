"""Transparent deterministic fallback classifier for regime-agent failures."""

from __future__ import annotations

from regime.models import (
    AssessmentSource,
    CompositeRegime,
    MomentumState,
    RegimeAssessment,
    RegimeFeatures,
    TrendState,
    VolatilityState,
)


class DeterministicRegimeClassifier:
    """Classify compact regime dimensions with inspectable threshold rules."""

    def __init__(
        self,
        *,
        low_volatility_pct: float = 12.0,
        high_volatility_pct: float = 30.0,
        momentum_threshold_pct: float = 2.0,
        trend_deadband_pct: float = 1.5,
    ) -> None:
        if not 0 <= low_volatility_pct < high_volatility_pct:
            raise ValueError("Volatility thresholds must satisfy 0 <= low < high.")
        self.low_volatility_pct = low_volatility_pct
        self.high_volatility_pct = high_volatility_pct
        self.momentum_threshold_pct = momentum_threshold_pct
        if trend_deadband_pct < 0:
            raise ValueError("trend_deadband_pct cannot be negative.")
        self.trend_deadband_pct = trend_deadband_pct

    def classify(self, features: RegimeFeatures) -> RegimeAssessment:
        trend_votes: list[int] = []
        for value in (
            features.close_vs_sma200_pct,
            features.sma50_vs_sma200_pct,
            features.sma50_slope_pct,
            features.return_60d_pct,
        ):
            if value is not None:
                trend_votes.append(
                    1
                    if value > self.trend_deadband_pct
                    else -1
                    if value < -self.trend_deadband_pct
                    else 0
                )

        vote_sum = sum(trend_votes)
        if len(trend_votes) >= 3 and vote_sum >= 2:
            trend = TrendState.BULLISH
        elif len(trend_votes) >= 3 and vote_sum <= -2:
            trend = TrendState.BEARISH
        else:
            trend = TrendState.NEUTRAL

        return_20 = features.return_20d_pct
        return_60 = features.return_60d_pct
        if (
            return_20 is not None
            and return_60 is not None
            and return_20 > self.momentum_threshold_pct
            and return_60 > 0
        ):
            momentum = MomentumState.POSITIVE
        elif (
            return_20 is not None
            and return_60 is not None
            and return_20 < -self.momentum_threshold_pct
            and return_60 < 0
        ):
            momentum = MomentumState.NEGATIVE
        else:
            momentum = MomentumState.NEUTRAL

        volatility_value = features.realized_volatility_20d_pct
        if volatility_value is None:
            volatility = VolatilityState.NORMAL
        elif volatility_value < self.low_volatility_pct:
            volatility = VolatilityState.LOW
        elif volatility_value > self.high_volatility_pct:
            volatility = VolatilityState.HIGH
        else:
            volatility = VolatilityState.NORMAL

        if volatility is VolatilityState.HIGH:
            regime = CompositeRegime.HIGH_VOLATILITY
        elif trend is TrendState.BULLISH:
            regime = CompositeRegime.BULLISH_TREND
        elif trend is TrendState.BEARISH:
            regime = CompositeRegime.BEARISH_TREND
        elif momentum is MomentumState.NEUTRAL and volatility is VolatilityState.LOW:
            regime = CompositeRegime.LOW_VOLATILITY
        elif trend is TrendState.NEUTRAL and momentum is MomentumState.NEUTRAL:
            regime = CompositeRegime.RANGE_BOUND
        else:
            regime = CompositeRegime.MIXED

        preferred, avoid = self._characteristics(regime)
        available_vote_ratio = min(1.0, len(trend_votes) / 4.0)
        agreement = abs(vote_sum) / max(1, len(trend_votes))
        confidence = round(min(0.95, 0.40 + 0.25 * available_vote_ratio + 0.25 * agreement), 3)
        reasoning = (
            f"Deterministic rules classified trend={trend.value} from "
            f"{len(trend_votes)} long-term votes (sum={vote_sum}), "
            f"momentum={momentum.value} from 20/60-day returns, and "
            f"volatility={volatility.value} from annualized 20-day volatility."
        )
        return RegimeAssessment(
            regime=regime,
            trend_state=trend,
            volatility_state=volatility,
            momentum_state=momentum,
            confidence=confidence,
            reasoning=reasoning,
            preferred_strategy_characteristics=preferred,
            avoid_strategy_characteristics=avoid,
            source=AssessmentSource.DETERMINISTIC_FALLBACK,
        )

    @staticmethod
    def _characteristics(regime: CompositeRegime) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if regime is CompositeRegime.BULLISH_TREND:
            return ("trend confirmation", "positive momentum"), ("short selling",)
        if regime is CompositeRegime.BEARISH_TREND:
            return ("defensive entry filters", "strict risk limits"), ("aggressive dip buying",)
        if regime is CompositeRegime.HIGH_VOLATILITY:
            return ("multiple confirmations", "volatility-aware stops"), ("tight unfiltered entries",)
        if regime in (CompositeRegime.RANGE_BOUND, CompositeRegime.LOW_VOLATILITY):
            return ("mean reversion", "oscillator confirmation"), ("unconfirmed breakouts",)
        return ("multiple confirmations",), ("single-indicator conviction",)
