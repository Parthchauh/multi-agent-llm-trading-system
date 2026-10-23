import numpy as np
import pytest

from research_validation.uncertainty import (
    block_bootstrap,
    sharpe_diagnostics,
    trade_path_distribution,
)


def test_symmetric_zero_mean_psr_is_half_and_dsr_requires_search_evidence():
    values = [-.01, .01]*50
    report = sharpe_diagnostics(values)
    assert report['psr'] == .5
    assert report['dsr'] is None
    assert report['dsr_unavailable_reason'] == 'INDEPENDENT_SEARCH_TRIALS_NOT_ESTABLISHED'
    one = sharpe_diagnostics(values, trial_sharpes=[0.], independent_trials=1)
    assert one['dsr'] == .5
    many = sharpe_diagnostics(values, trial_sharpes=[-.1, .1]*50, independent_trials=100)
    assert many['dsr'] < .5


def test_bootstrap_constant_returns_has_hand_computable_degenerate_interval():
    report = block_bootstrap([.01]*20, repetitions=100, block_size=2)
    assert report['total_return_pct_ci95'] == pytest.approx([22.0190039947967]*2)
    assert report['max_drawdown_pct_ci95'] == [0., 0.]
    assert report['promotion_eligible'] is False
    assert report == block_bootstrap([.01]*20, repetitions=100, block_size=2)


def test_trade_ordering_preserves_total_pnl_but_changes_drawdown():
    report = trade_path_distribution([10., -5.]*10, 1000., repetitions=100)
    assert report['return_pct_ci95'] == pytest.approx([5., 5.])
    assert report['drawdown_pct_distribution'][0] < report['drawdown_pct_distribution'][-1]
    assert report['ruin_fraction'] == 0.


@pytest.mark.parametrize('bad', [[float('nan')]*30, [-1.]*30, np.ones((2, 3))])
def test_bad_returns_are_rejected(bad):
    with pytest.raises(ValueError):
        block_bootstrap(bad)


def test_insufficient_observations_are_not_significance():
    assert sharpe_diagnostics([.01]*2)['status'] == 'INSUFFICIENT_EVIDENCE'
    assert block_bootstrap([.01]*2)['status'] == 'INSUFFICIENT_EVIDENCE'
    assert sharpe_diagnostics([.01]*100)['status'] == 'INSUFFICIENT_EVIDENCE'


def test_nonzero_mean_psr_matches_independent_symmetric_moment_calculation():
    # Mean=.01, sample variance=.0004*100/99, skew=0, Pearson kurtosis=1.
    # Thus the PSR normal z-score is exactly 4.95.
    assert sharpe_diagnostics([-.01, .03]*50)['psr'] == pytest.approx(.9999996288, abs=1e-9)


@pytest.mark.parametrize('repetitions', [True, 100.5, 99, 10001])
def test_invalid_resampling_budget_is_rejected(repetitions):
    with pytest.raises(ValueError):
        block_bootstrap([.01]*30, repetitions=repetitions)
    with pytest.raises(ValueError):
        trade_path_distribution([1.]*30, 1000., repetitions=repetitions)
