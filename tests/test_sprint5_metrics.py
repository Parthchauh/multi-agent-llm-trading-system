"""Sprint 5 tests for execution-derived deterministic performance metrics."""

from __future__ import annotations

from datetime import date

from backtesting.metrics import calculate_metrics
from backtesting.models import EquityPoint, ExitReason, Trade, TradeSide
from evaluation.models import EvaluationMetrics


def _equity(day: int, equity: float, *, month: int = 1, year: int = 2023) -> EquityPoint:
    return EquityPoint(
        date=date(year, month, day),
        equity=equity,
        cash=equity,
        is_invested=False,
    )


def _trade(
    *,
    entry_price: float = 100.0,
    exit_price: float = 100.0,
    quantity: int = 1,
    net_pnl: float = 0.0,
    gross_pnl: float | None = None,
) -> Trade:
    gross = net_pnl if gross_pnl is None else gross_pnl
    return Trade(
        entry_date=date(2023, 1, 1),
        entry_price=entry_price,
        exit_date=date(2023, 1, 2),
        exit_price=exit_price,
        quantity=quantity,
        gross_pnl=gross,
        commission_paid=gross - net_pnl,
        net_pnl=net_pnl,
        return_pct=net_pnl / (entry_price * quantity) * 100.0,
        exit_reason=ExitReason.SIGNAL,
        holding_days=1,
        side=TradeSide.LONG,
    )


class TestExpandedPerformanceMetrics:
    def test_hand_calculated_calmar_payoff_turnover_and_duration(self) -> None:
        # Peak Jan 1 -> drawdown Jan 2 -> recovery Jan 5: 4 calendar days,
        # 2 elapsed equity-curve bars.  The final 10% gain occurs after 365 days.
        equity_curve = [
            _equity(1, 100.0),
            _equity(2, 80.0),
            _equity(5, 100.0),
            _equity(1, 110.0, year=2024),
        ]
        trades = [
            _trade(entry_price=100.0, exit_price=120.0, net_pnl=20.0),
            _trade(entry_price=100.0, exit_price=90.0, net_pnl=-10.0),
        ]

        metrics = calculate_metrics(100.0, trades, equity_curve)

        expected_cagr = round(((110.0 / 100.0) ** (365.25 / 365.0) - 1.0) * 100.0, 4)
        assert metrics.cagr_pct == expected_cagr
        assert metrics.max_drawdown_pct == 20.0
        assert metrics.calmar_ratio == round(expected_cagr / 20.0, 4)
        assert metrics.payoff_ratio == 2.0
        # Gross two-sided executed notional = 100 + 120 + 100 + 90 = 410;
        # average end-of-bar equity = (100 + 80 + 100 + 110) / 4 = 97.5.
        assert metrics.turnover_pct == round((410.0 / 97.5) * 100.0, 4)
        assert metrics.max_drawdown_duration_days == 4
        assert metrics.max_drawdown_duration_bars == 2

        serialized = metrics.to_dict()
        assert serialized["calmar_ratio"] == metrics.calmar_ratio
        assert serialized["payoff_ratio"] == metrics.payoff_ratio
        assert serialized["turnover_pct"] == metrics.turnover_pct
        assert serialized["max_drawdown_duration_days"] == 4
        assert serialized["max_drawdown_duration_bars"] == 2
        assert serialized["metrics_version"] == "1.2"

    def test_unrecovered_drawdown_ends_at_final_observation(self) -> None:
        equity_curve = [
            _equity(1, 100.0),
            _equity(3, 90.0),
            _equity(10, 80.0),
        ]

        metrics = calculate_metrics(100.0, [], equity_curve)

        assert metrics.max_drawdown_duration_days == 9
        assert metrics.max_drawdown_duration_bars == 2

    def test_no_drawdown_has_zero_duration_and_undefined_calmar(self) -> None:
        equity_curve = [
            _equity(1, 100.0),
            _equity(5, 100.0),
            _equity(10, 110.0),
        ]

        metrics = calculate_metrics(100.0, [], equity_curve)

        assert metrics.max_drawdown_duration_days == 0
        assert metrics.max_drawdown_duration_bars == 0
        assert metrics.calmar_ratio is None


class TestExpandedMetricDegenerateCases:
    def test_no_completed_trades_has_zero_turnover_and_no_payoff(self) -> None:
        metrics = calculate_metrics(100.0, [], [])

        assert metrics.turnover_pct == 0.0
        assert metrics.payoff_ratio is None
        assert metrics.calmar_ratio is None
        assert metrics.max_drawdown_duration_days == 0
        assert metrics.max_drawdown_duration_bars == 0

    def test_turnover_is_undefined_without_a_positive_equity_denominator(self) -> None:
        metrics = calculate_metrics(100.0, [_trade(net_pnl=5.0)], [])

        assert metrics.turnover_pct is None

    def test_payoff_uses_net_pnl_and_requires_a_nonzero_loss(self) -> None:
        gross_positive_net_loss = _trade(net_pnl=-10.0, gross_pnl=50.0)
        winner = _trade(net_pnl=100.0, gross_pnl=110.0)
        metrics = calculate_metrics(
            100.0,
            [gross_positive_net_loss, winner],
            [_equity(1, 100.0), _equity(2, 100.0)],
        )
        assert metrics.payoff_ratio == 10.0

        zero_net_pnl = calculate_metrics(
            100.0,
            [_trade(net_pnl=100.0), _trade(net_pnl=0.0)],
            [_equity(1, 100.0), _equity(2, 100.0)],
        )
        assert zero_net_pnl.payoff_ratio is None


def test_evaluation_snapshot_copies_the_complete_backtest_metric_contract() -> None:
    backtest_metrics = calculate_metrics(
        100.0,
        [_trade(entry_price=100.0, exit_price=110.0, net_pnl=10.0)],
        [_equity(1, 100.0), _equity(2, 110.0)],
    )

    snapshot = EvaluationMetrics.from_backtest_metrics(backtest_metrics)

    assert snapshot.calmar_ratio == backtest_metrics.calmar_ratio
    assert snapshot.payoff_ratio == backtest_metrics.payoff_ratio
    assert snapshot.turnover_pct == backtest_metrics.turnover_pct
    assert snapshot.max_drawdown_duration_days == backtest_metrics.max_drawdown_duration_days
    assert snapshot.max_drawdown_duration_bars == backtest_metrics.max_drawdown_duration_bars
