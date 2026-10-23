"""Reproducible research diagnostics, never promotion authority.

PSR/DSR methodology references are recorded in docs/prelaunch/research-methodology.md.
Returns and reference Sharpe must use the same observation frequency.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def _returns(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError('Returns must be a finite one-dimensional sequence above -100%.')
    return values


def sharpe_diagnostics(returns, *, reference_sharpe=0., trial_sharpes=None,
                       independent_trials=None) -> dict:
    values = _returns(returns)
    if not math.isfinite(reference_sharpe):
        raise ValueError('Reference Sharpe must be finite and non-annualized.')
    report = {'observations': len(values), 'frequency': 'per_observation',
              'psr': None, 'dsr': None, 'status': 'INSUFFICIENT_EVIDENCE',
              'assumptions': ['PSR uses asymptotic sampling and moment estimates.',
                              'Serial dependence can invalidate the PSR approximation.',
                              'DSR needs declared independent trials and their Sharpe dispersion.'],
              'promotion_eligible': False}
    if len(values) < 20 or values.max() == values.min():
        return report
    # Sharpe and standardized moments are scale-invariant; normalize first to
    # avoid overflow when a malformed unit convention produces extreme values.
    scaled = values/max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    centered = scaled-scaled.mean()
    m2 = float(np.mean(centered**2))
    skew = float(np.mean(centered**3)/m2**1.5)
    kurtosis = float(np.mean(centered**4)/m2**2)
    sharpe = float(scaled.mean()/scaled.std(ddof=1))
    variance_term = 1-skew*sharpe+(kurtosis-1)*sharpe**2/4
    if not math.isfinite(variance_term) or variance_term <= 0:
        return report
    standard_error = math.sqrt(variance_term/(len(values)-1))
    report.update(status='DIAGNOSTIC_ONLY', sharpe=sharpe, skewness=skew,
                  kurtosis=kurtosis, psr=NormalDist().cdf((sharpe-reference_sharpe)/standard_error))
    report['lag1_autocorrelation'] = (float(np.corrcoef(scaled[:-1], scaled[1:])[0, 1])
        if np.std(scaled[:-1]) > 0 and np.std(scaled[1:]) > 0 else None)
    if independent_trials is None:
        report['dsr_unavailable_reason'] = 'INDEPENDENT_SEARCH_TRIALS_NOT_ESTABLISHED'
        return report
    if isinstance(independent_trials, bool) or not isinstance(independent_trials, int) or independent_trials < 1:
        raise ValueError('Independent trial count must be a positive integer.')
    trials = np.asarray(trial_sharpes, dtype=float)
    if trials.ndim != 1 or len(trials) < independent_trials or not np.isfinite(trials).all():
        raise ValueError('Supply finite, comparable trial Sharpe estimates for the declared search.')
    if independent_trials > 1 and len(trials) < 2:
        raise ValueError('Trial dispersion cannot be inferred from one result.')
    if independent_trials == 1:
        benchmark = 0.
    else:
        sigma = float(trials.std(ddof=1))
        gamma = 0.5772156649015329
        benchmark = sigma*((1-gamma)*NormalDist().inv_cdf(1-1/independent_trials)
                          +gamma*NormalDist().inv_cdf(1-1/(independent_trials*math.e)))
    report.update(independent_trials=independent_trials, search_results=len(trials),
                  deflated_reference_sharpe=benchmark,
                  dsr=NormalDist().cdf((sharpe-benchmark)/standard_error))
    return report


def block_bootstrap(returns, *, seed=42, repetitions=1000, block_size=5) -> dict:
    values = _returns(returns)
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or not 100 <= repetitions <= 10000:
        raise ValueError('Bootstrap repetitions must be in [100, 10000].')
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError('Block size must be a positive integer.')
    if len(values) < max(20, block_size*2):
        return {'status': 'INSUFFICIENT_EVIDENCE', 'observations': len(values),
                'promotion_eligible': False}
    rng = np.random.default_rng(seed)
    means, totals, drawdowns = [], [], []
    blocks = math.ceil(len(values)/block_size)
    for _ in range(repetitions):
        starts = rng.integers(0, len(values), size=blocks)
        indices = ((starts[:, None]+np.arange(block_size)) % len(values)).ravel()[:len(values)]
        sample = values[indices]
        equity = np.r_[1., np.cumprod(1+sample)]
        means.append(float(sample.mean()))
        totals.append(float((equity[-1]-1)*100))
        drawdowns.append(float(np.max(1-equity/np.maximum.accumulate(equity))*100))
    if not np.isfinite(totals).all():
        raise ValueError('Bootstrap compounded values overflowed; inspect return units.')
    return {'status': 'DIAGNOSTIC_ONLY', 'observations': len(values), 'seed': seed,
            'repetitions': repetitions, 'block_size': block_size,
            'mean_return_ci95': np.quantile(means, [.025, .975]).tolist(),
            'total_return_pct_ci95': np.quantile(totals, [.025, .975]).tolist(),
            'max_drawdown_pct_ci95': np.quantile(drawdowns, [.025, .975]).tolist(),
            'assumptions': ['Circular contiguous block resampling.',
                            'Within-block dependence is retained; across-block dependence is not.',
                            'Block length must be specified before interpreting results.'],
            'promotion_eligible': False}


def trade_path_distribution(pnl_values, initial_capital, *, seed=42, repetitions=1000,
                            bootstrap=False) -> dict:
    values = np.asarray(pnl_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not math.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError('Require finite trade P&L and positive initial capital.')
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or not 100 <= repetitions <= 10000:
        raise ValueError('Repetitions must be in [100, 10000].')
    if len(values) < 20:
        return {'status': 'INSUFFICIENT_EVIDENCE', 'trade_count': len(values),
                'promotion_eligible': False}
    rng = np.random.default_rng(seed)
    returns, drawdowns, ruin = [], [], 0
    for _ in range(repetitions):
        sample = rng.choice(values, len(values), replace=True) if bootstrap else rng.permutation(values)
        equity = np.r_[initial_capital, initial_capital+np.cumsum(sample)]
        ruin += int(np.min(equity) <= 0)
        returns.append(float((equity[-1]/initial_capital-1)*100))
        drawdowns.append(float(np.max(1-equity/np.maximum.accumulate(equity))*100))
    return {'status': 'DIAGNOSTIC_ONLY', 'method': 'TRADE_BOOTSTRAP' if bootstrap else 'TRADE_ORDER_PERMUTATION',
            'trade_count': len(values), 'seed': seed, 'repetitions': repetitions,
            'return_pct_ci95': np.quantile(returns, [.025, .975]).tolist(),
            'drawdown_pct_distribution': np.quantile(drawdowns, [0., .05, .5, .95, 1.]).tolist(),
            'ruin_fraction': ruin/repetitions,
            'assumptions': ['Fixed completed net P&L is resampled; sizing and market paths are not replayed.'],
            'promotion_eligible': False}
