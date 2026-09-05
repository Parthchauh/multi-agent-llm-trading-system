"""Sprint 3.5 metrics consistency tests.

Verifies:
- profit_factor uses net PnL (consistent with win/loss classification)
- largest_win is None when no winning trades
- largest_loss is None when no losing trades
- metrics_version == "1.2" while retaining the version 1.1 net-PnL
  profit-factor semantics
- all-zero-trade edge cases return None safely
"""
from __future__ import annotations

from datetime import date

from backtesting.metrics import calculate_metrics
from backtesting.models import EquityPoint, ExitReason, Trade, TradeSide


def _trade(net_pnl, gross_pnl=None, holding_days=5):
    gross = gross_pnl if gross_pnl is not None else net_pnl + 1.0
    commission = gross - net_pnl
    entry_p = 100.0
    exit_p = entry_p + (gross / 10)
    return Trade(
        entry_date=date(2023, 1, 2),
        entry_price=entry_p,
        exit_date=date(2023, 1, 9),
        exit_price=exit_p,
        quantity=10,
        gross_pnl=gross,
        commission_paid=commission,
        net_pnl=net_pnl,
        return_pct=(net_pnl / (entry_p * 10)) * 100,
        exit_reason=ExitReason.SIGNAL,
        holding_days=holding_days,
        side=TradeSide.LONG,
    )


def _eq(equity=10000.0, invested=False):
    return EquityPoint(date=date(2023, 1, 2), equity=equity, cash=equity, is_invested=invested)


class TestMetricsVersion:
    def test_metrics_version_is_1_2(self):
        m = calculate_metrics(10000.0, [], [_eq()])
        assert m.metrics_version == "1.2"


class TestProfitFactorNetBasis:
    def test_profit_factor_uses_net_pnl(self):
        """A trade that is gross-positive but net-negative should count as a LOSS."""
        # gross=+50, commission=60, net=-10 → should be counted as a losing trade
        gross_win_net_loss = _trade(net_pnl=-10.0, gross_pnl=50.0)
        # pure winning trade
        winner = _trade(net_pnl=100.0, gross_pnl=110.0)
        m = calculate_metrics(10000.0, [gross_win_net_loss, winner], [_eq(), _eq()])
        # wins = [100.0], losses = [-10.0]
        # profit_factor = 100.0 / 10.0 = 10.0
        assert m.profit_factor is not None
        assert abs(m.profit_factor - 10.0) < 0.01
        assert m.winning_trades == 1
        assert m.losing_trades == 1

    def test_profit_factor_none_when_no_losses(self):
        m = calculate_metrics(10000.0, [_trade(net_pnl=50.0)], [_eq()])
        assert m.profit_factor is None

    def test_profit_factor_zero_when_no_wins(self):
        m = calculate_metrics(10000.0, [_trade(net_pnl=-50.0)], [_eq()])
        # net_profits = 0, net_losses_abs = 50 → profit_factor = 0/50 = 0.0
        assert m.profit_factor == 0.0


class TestLargestWinLoss:
    def test_largest_win_none_when_all_losing(self):
        trades = [_trade(net_pnl=-20.0), _trade(net_pnl=-30.0)]
        m = calculate_metrics(10000.0, trades, [_eq(), _eq()])
        assert m.largest_win is None

    def test_largest_loss_none_when_all_winning(self):
        trades = [_trade(net_pnl=20.0), _trade(net_pnl=30.0)]
        m = calculate_metrics(10000.0, trades, [_eq(), _eq()])
        assert m.largest_loss is None

    def test_largest_win_correct_when_mixed(self):
        trades = [_trade(net_pnl=50.0), _trade(net_pnl=-20.0), _trade(net_pnl=30.0)]
        m = calculate_metrics(10000.0, trades, [_eq(), _eq(), _eq()])
        assert m.largest_win == 50.0

    def test_largest_loss_correct_when_mixed(self):
        trades = [_trade(net_pnl=50.0), _trade(net_pnl=-20.0), _trade(net_pnl=-35.0)]
        m = calculate_metrics(10000.0, trades, [_eq(), _eq(), _eq()])
        assert m.largest_loss == -35.0


class TestZeroTradeEdgeCases:
    def test_no_trades_returns_all_none(self):
        m = calculate_metrics(10000.0, [], [_eq()])
        assert m.win_rate_pct is None
        assert m.profit_factor is None
        assert m.expectancy is None
        assert m.average_win is None
        assert m.average_loss is None
        assert m.largest_win is None
        assert m.largest_loss is None
        assert m.average_holding_days is None
        assert m.number_of_trades == 0
        assert m.winning_trades == 0
        assert m.losing_trades == 0

    def test_empty_equity_curve_returns_zeros(self):
        m = calculate_metrics(10000.0, [], [])
        assert m.total_return_pct == 0.0
        assert m.max_drawdown_pct == 0.0
        assert m.sharpe_ratio is None
        assert m.sortino_ratio is None
        assert m.exposure_pct == 0.0
