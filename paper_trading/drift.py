"""Predeclared deterministic drift thresholds; no statistical or release authority."""
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class DriftPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', allow_inf_nan=False)
    window: int = Field(default=30, ge=30, le=10000)
    minimum_completed_trades: int = Field(default=20, ge=20)
    maximum_mean_bar_return_gap_pct: float = Field(default=.5, gt=0.)
    maximum_drawdown_gap_pct: float = Field(default=5., gt=0.)


def measure_drift(actual_equity, reference_equity, *, completed_trades: int, policy: DriftPolicy) -> dict:
    actual, reference = np.asarray(actual_equity, dtype=float), np.asarray(reference_equity, dtype=float)
    if (actual.ndim != 1 or reference.ndim != 1 or not np.isfinite(actual).all()
            or not np.isfinite(reference).all() or (actual <= 0).any() or (reference <= 0).any()):
        raise ValueError('Drift requires finite positive one-dimensional equity paths.')
    if isinstance(completed_trades, bool) or not isinstance(completed_trades, int) or completed_trades < 0:
        raise ValueError('Completed trade count must be a non-negative integer.')
    base = {'policy': policy.model_dump(), 'observations': max(0, len(actual)-1),
            'completed_trades': completed_trades, 'paper_eligible': False,
            'limitations': ['Thresholds are research policy defaults, not validated investment limits.',
                            'Distribution comparison does not establish statistical certification.']}
    if min(len(actual), len(reference)) <= policy.window or completed_trades < policy.minimum_completed_trades:
        return {**base, 'status': 'INSUFFICIENT_EVIDENCE'}
    actual, reference = actual[-policy.window-1:], reference[-policy.window-1:]
    actual_returns = actual[1:]/actual[:-1]-1
    reference_returns = reference[1:]/reference[:-1]-1
    return_gap = float((actual_returns.mean()-reference_returns.mean())*100)
    actual_dd = float(np.max(1-actual/np.maximum.accumulate(actual))*100)
    reference_dd = float(np.max(1-reference/np.maximum.accumulate(reference))*100)
    dd_gap = actual_dd-reference_dd
    breached = (abs(return_gap) > policy.maximum_mean_bar_return_gap_pct
                or dd_gap > policy.maximum_drawdown_gap_pct)
    return {**base, 'status': 'DRIFT_THRESHOLD_BREACH' if breached else 'NO_THRESHOLD_BREACH_OBSERVED',
            'mean_bar_return_gap_pct': return_gap, 'drawdown_gap_pct': dd_gap}
