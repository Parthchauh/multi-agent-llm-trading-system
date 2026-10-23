import pandas as pd
import pytest

from data.adjustments import normalize_with_factors


def test_literal_two_for_one_split_preserves_price_volume_units_without_guessing():
    index = pd.date_range('2020-01-01', periods=3, tz='UTC')
    raw = pd.DataFrame({c: [100., 102., 51.] for c in ('Open', 'High', 'Low', 'Close')}, index=index)
    raw['Volume'] = [100., 100., 200.]
    factors = pd.DataFrame({'price_factor': [.5, .5, 1.], 'volume_factor': [2., 2., 1.],
        'available_at': [index[-1]]*3}, index=index)
    adjusted, trace = normalize_with_factors(raw, factors, as_of=index[-1], basis='split_adjusted',
                                             source='recorded_vendor', method_version='v1')
    assert adjusted.Close.tolist() == [50., 51., 51.]
    assert adjusted.Volume.tolist() == [200., 200., 200.]
    assert raw.Close.tolist() == [100., 102., 51.]
    assert trace['point_in_time_vendor_claim_independently_verified'] is False
    with pytest.raises(ValueError, match='cutoff'):
        normalize_with_factors(raw.iloc[:2], factors.iloc[:2], as_of=index[1], basis='split_adjusted',
                               source='recorded_vendor', method_version='v1')


def test_vendor_cash_dividend_total_return_factors_adjust_prices_without_adjusting_volume():
    index = pd.date_range('2020-01-01', periods=2, tz='UTC')
    raw = pd.DataFrame({c: [100., 98.] for c in ('Open', 'High', 'Low', 'Close')}, index=index)
    raw['Volume'] = [100., 100.]
    factors = pd.DataFrame({'price_factor': [.98, 1.], 'volume_factor': [1., 1.],
        'available_at': [index[-1]]*2}, index=index)
    adjusted, _ = normalize_with_factors(raw, factors, as_of=index[-1], basis='total_return_adjusted',
                                        source='recorded_vendor', method_version='dividend_fixture_v1')
    assert adjusted.Close.tolist() == [98., 98.]
    assert adjusted.Volume.tolist() == [100., 100.]


@pytest.mark.parametrize('factor', [0., -1., float('nan'), float('inf')])
def test_impossible_adjustment_factors_are_rejected(factor):
    index = pd.date_range('2020-01-01', periods=2, tz='UTC')
    raw = pd.DataFrame({c: [100., 100.] for c in ('Open', 'High', 'Low', 'Close', 'Volume')}, index=index)
    factors = pd.DataFrame({'price_factor': [factor, 1.], 'volume_factor': [1., 1.],
                            'available_at': [index[-1]]*2}, index=index)
    with pytest.raises(ValueError):
        normalize_with_factors(raw, factors, as_of=index[-1], basis='split_adjusted', source='test', method_version='v1')
