import pytest

from paper_trading.drift import DriftPolicy, measure_drift


def test_independent_flat_path_and_twenty_percent_drawdown_drift_expectations():
    policy = DriftPolicy()
    flat = [100.]*31
    same = measure_drift(flat, flat, completed_trades=20, policy=policy)
    assert same['status'] == 'NO_THRESHOLD_BREACH_OBSERVED'
    assert same['mean_bar_return_gap_pct'] == 0.
    assert same['drawdown_gap_pct'] == 0.
    fallen = measure_drift([100.]*30+[80.], flat, completed_trades=20, policy=policy)
    assert fallen['status'] == 'DRIFT_THRESHOLD_BREACH'
    assert fallen['drawdown_gap_pct'] == pytest.approx(20.)
    assert fallen['mean_bar_return_gap_pct'] == pytest.approx(-2/3)
    assert fallen['paper_eligible'] is False


def test_low_trade_or_bar_count_is_never_a_drift_pass():
    for equity, trades in (([100.]*31, 19), ([100.]*30, 20)):
        assert measure_drift(equity, [100.]*31, completed_trades=trades,
                             policy=DriftPolicy())['status'] == 'INSUFFICIENT_EVIDENCE'
    with pytest.raises(ValueError):
        measure_drift([100., 0.], [100., 100.], completed_trades=20, policy=DriftPolicy())
