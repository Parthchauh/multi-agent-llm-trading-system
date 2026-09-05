"""
tests/test_metrics.py
=====================
Tests for backtesting/metrics.py with hand-calculated numerical validation.
"""

from datetime import date
import pytest
from backtesting.metrics import calculate_metrics
from backtesting.models import EquityPoint, ExitReason, Trade, TradeSide


class TestMetricsCalculation:
    def test_empty_backtest_metrics(self) -> None:
        metrics = calculate_metrics(initial_capital=10000.0, trades=[], equity_curve=[])
        assert metrics.total_return_pct == 0.0
        assert metrics.max_drawdown_pct == 0.0
        assert metrics.sharpe_ratio is None
        assert metrics.sortino_ratio is None
        assert metrics.cagr_pct is None
        assert metrics.number_of_trades == 0
        assert metrics.win_rate_pct is None
        assert metrics.profit_factor is None

    def test_hand_calculated_drawdown_and_return(self) -> None:
        # Initial 10,000. Equity: 10,000 -> 12,000 (peak) -> 9,000 (trough) -> 11,000 (final)
        # Peak = 12,000, Trough = 9,000 -> Drawdown = (9,000 - 12,000)/12,000 = -25% -> reported as 25.0%
        # Total return = (11,000 / 10,000 - 1) * 100 = 10.0%
        equity_curve = [
            EquityPoint(date=date(2023, 1, 1), equity=10000.0, cash=10000.0, is_invested=False),
            EquityPoint(date=date(2023, 1, 2), equity=12000.0, cash=0.0, is_invested=True),
            EquityPoint(date=date(2023, 1, 3), equity=9000.0, cash=0.0, is_invested=True),
            EquityPoint(date=date(2023, 1, 4), equity=11000.0, cash=11000.0, is_invested=False),
        ]
        metrics = calculate_metrics(initial_capital=10000.0, trades=[], equity_curve=equity_curve)
        assert pytest.approx(metrics.total_return_pct, 1e-4) == 10.0
        assert pytest.approx(metrics.max_drawdown_pct, 1e-4) == 25.0
        assert metrics.exposure_pct == 50.0  # 2 out of 4 days invested

    def test_hand_calculated_trade_statistics(self) -> None:
        # Trade 1: +500 gross, 10 comm -> +490 net (Win)
        # Trade 2: -200 gross, 10 comm -> -210 net (Loss)
        # Trade 3: +300 gross, 10 comm -> +290 net (Win)
        t1 = Trade(
            entry_date=date(2023, 1, 1),
            entry_price=100.0,
            exit_date=date(2023, 1, 5),
            exit_price=105.0,
            quantity=100,
            gross_pnl=500.0,
            commission_paid=10.0,
            net_pnl=490.0,
            return_pct=4.9,
            exit_reason=ExitReason.TAKE_PROFIT,
            holding_days=4,
        )
        t2 = Trade(
            entry_date=date(2023, 1, 6),
            entry_price=100.0,
            exit_date=date(2023, 1, 8),
            exit_price=98.0,
            quantity=100,
            gross_pnl=-200.0,
            commission_paid=10.0,
            net_pnl=-210.0,
            return_pct=-2.1,
            exit_reason=ExitReason.STOP_LOSS,
            holding_days=2,
        )
        t3 = Trade(
            entry_date=date(2023, 1, 9),
            entry_price=100.0,
            exit_date=date(2023, 1, 15),
            exit_price=103.0,
            quantity=100,
            gross_pnl=300.0,
            commission_paid=10.0,
            net_pnl=290.0,
            return_pct=2.9,
            exit_reason=ExitReason.SIGNAL,
            holding_days=6,
        )

        equity_curve = [
            EquityPoint(date=date(2023, 1, 1), equity=10000.0, cash=10000.0, is_invested=True),
            EquityPoint(date=date(2023, 1, 15), equity=10570.0, cash=10570.0, is_invested=False),
        ]

        metrics = calculate_metrics(
            initial_capital=10000.0,
            trades=[t1, t2, t3],
            equity_curve=equity_curve,
        )

        assert metrics.number_of_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        # Win rate: 2/3 = 66.6667%
        assert pytest.approx(metrics.win_rate_pct, 1e-4) == 66.6667
        # Profit factor v1.1: net profits (490 + 290) / net loss (210) = 3.7143.
        assert pytest.approx(metrics.profit_factor, 1e-4) == 3.7143
        # Expectancy: (490 - 210 + 290) / 3 = 570 / 3 = 190.0
        assert pytest.approx(metrics.expectancy, 1e-4) == 190.0
        # Average win: (490 + 290) / 2 = 390.0
        assert pytest.approx(metrics.average_win, 1e-4) == 390.0
        # Average loss: -210.0
        assert pytest.approx(metrics.average_loss, 1e-4) == -210.0
        # Largest win & loss
        assert metrics.largest_win == 490.0
        assert metrics.largest_loss == -210.0
        # Average holding days: (4 + 2 + 6) / 3 = 4.0
        assert pytest.approx(metrics.average_holding_days, 1e-2) == 4.0

    def test_no_losing_trades_profit_factor_none(self) -> None:
        t1 = Trade(
            entry_date=date(2023, 1, 1),
            entry_price=100.0,
            exit_date=date(2023, 1, 5),
            exit_price=105.0,
            quantity=100,
            gross_pnl=500.0,
            commission_paid=10.0,
            net_pnl=490.0,
            return_pct=4.9,
            exit_reason=ExitReason.TAKE_PROFIT,
            holding_days=4,
        )
        metrics = calculate_metrics(initial_capital=10000.0, trades=[t1], equity_curve=[])
        assert metrics.profit_factor is None
        assert metrics.average_loss is None
