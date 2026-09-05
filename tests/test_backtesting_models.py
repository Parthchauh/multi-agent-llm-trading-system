"""
tests/test_backtesting_models.py
================================
Tests for backtesting/models.py.
"""

from datetime import date
import pytest
from pydantic import ValidationError

from backtesting.models import (
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    EquityPoint,
    ExitReason,
    Position,
    Signal,
    SignalType,
    Trade,
    TradeSide,
)


class TestBacktestConfig:
    def test_valid_config(self) -> None:
        cfg = BacktestConfig(
            initial_capital=100000.0,
            commission_pct=0.001,
            slippage_pct=0.0005,
            position_size_pct=20.0,
            maximum_holding_days=30,
        )
        assert cfg.initial_capital == 100000.0
        assert cfg.commission_pct == 0.001
        assert cfg.slippage_pct == 0.0005
        assert cfg.position_size_pct == 20.0
        assert cfg.maximum_holding_days == 30

    def test_negative_capital_raises(self) -> None:
        with pytest.raises(ValidationError):
            BacktestConfig(initial_capital=-500)

    def test_zero_capital_raises(self) -> None:
        with pytest.raises(ValidationError):
            BacktestConfig(initial_capital=0)

    def test_invalid_position_size_pct(self) -> None:
        with pytest.raises(ValidationError):
            BacktestConfig(initial_capital=1000, position_size_pct=0)
        with pytest.raises(ValidationError):
            BacktestConfig(initial_capital=1000, position_size_pct=105)

    def test_negative_commission_slippage(self) -> None:
        with pytest.raises(ValidationError):
            BacktestConfig(initial_capital=1000, commission_pct=-0.01)
        with pytest.raises(ValidationError):
            BacktestConfig(initial_capital=1000, slippage_pct=-0.01)

    def test_invalid_holding_days(self) -> None:
        with pytest.raises(ValidationError):
            BacktestConfig(initial_capital=1000, maximum_holding_days=0)
        with pytest.raises(ValidationError):
            BacktestConfig(initial_capital=1000, maximum_holding_days=-5)


class TestTradeModel:
    def test_trade_to_dict(self) -> None:
        trade = Trade(
            entry_date=date(2023, 1, 1),
            entry_price=100.0,
            exit_date=date(2023, 1, 10),
            exit_price=110.0,
            quantity=50,
            gross_pnl=500.0,
            commission_paid=10.5,
            net_pnl=489.5,
            return_pct=9.79,
            exit_reason=ExitReason.TAKE_PROFIT,
            holding_days=9,
            entry_signal_date=date(2022, 12, 31),
            exit_signal_date=None,
            side=TradeSide.LONG,
        )
        d = trade.to_dict()
        assert d["entry_date"] == "2023-01-01"
        assert d["exit_date"] == "2023-01-10"
        assert d["entry_price"] == 100.0
        assert d["exit_price"] == 110.0
        assert d["quantity"] == 50
        assert d["net_pnl"] == 489.5
        assert d["exit_reason"] == "TAKE_PROFIT"
        assert d["entry_signal_date"] == "2022-12-31"
        assert d["exit_signal_date"] is None


class TestEquityPoint:
    def test_equity_point_to_dict(self) -> None:
        pt = EquityPoint(
            date=date(2023, 1, 5),
            equity=105000.25,
            cash=80000.0,
            is_invested=True,
        )
        d = pt.to_dict()
        assert d["date"] == "2023-01-05"
        assert d["equity"] == 105000.25
        assert d["is_invested"] is True
