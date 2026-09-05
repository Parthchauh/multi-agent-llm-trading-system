"""
tests/test_portfolio.py
=======================
Tests for backtesting/portfolio.py (Portfolio).
"""

from datetime import date
import pytest
from backtesting.execution import TradeExecutor
from backtesting.models import ExitReason
from backtesting.portfolio import Portfolio


class TestPortfolio:
    def test_initial_state(self) -> None:
        executor = TradeExecutor()
        portfolio = Portfolio(initial_capital=10000.0, position_size_pct=20.0, executor=executor)
        assert portfolio.cash == 10000.0
        assert portfolio.position is None
        assert not portfolio.is_invested()
        assert portfolio.get_equity(100.0) == 10000.0

    def test_calculate_order_quantity_whole_shares(self) -> None:
        executor = TradeExecutor(commission_pct=0.001, slippage_pct=0.0)
        portfolio = Portfolio(initial_capital=10000.0, position_size_pct=20.0, executor=executor)
        # Allocated = 2000. Cost per share = 100 * 1.001 = 100.10. 2000 / 100.10 = 19.98 -> 19 shares
        qty = portfolio.calculate_order_quantity(100.0)
        assert qty == 19
        assert isinstance(qty, int)

    def test_open_position_deducts_cash(self) -> None:
        executor = TradeExecutor(commission_pct=0.001, slippage_pct=0.001)
        portfolio = Portfolio(initial_capital=10000.0, position_size_pct=50.0, executor=executor)
        # Buy fill at 100 * 1.001 = 100.10. Cost per share with comm: 100.10 * 1.001 = 100.2001
        # Allocated = 5000. Qty = floor(5000 / 100.2001) = 49 shares
        # Trade value = 49 * 100.10 = 4904.90. Comm = 4904.90 * 0.001 = 4.9049. Total cost = 4909.8049
        pos = portfolio.open_position(
            entry_date=date(2023, 1, 2),
            raw_price=100.0,
            stop_price=95.0,
            target_price=110.0,
            entry_signal_date=date(2023, 1, 1),
        )
        assert pos is not None
        assert pos.quantity == 49
        assert portfolio.is_invested()
        assert pytest.approx(portfolio.cash, 1e-4) == 10000.0 - 4909.8049
        # Equity at close 102.0: Cash + 49 * 102
        assert pytest.approx(portfolio.get_equity(102.0), 1e-4) == portfolio.cash + (49 * 102.0)

    def test_cannot_open_second_position_while_invested(self) -> None:
        executor = TradeExecutor()
        portfolio = Portfolio(initial_capital=10000.0, position_size_pct=10.0, executor=executor)
        pos1 = portfolio.open_position(entry_date=date(2023, 1, 2), raw_price=50.0)
        assert pos1 is not None
        # Attempt second open
        pos2 = portfolio.open_position(entry_date=date(2023, 1, 3), raw_price=50.0)
        assert pos2 is None

    def test_close_position_realizes_pnl_and_adds_cash(self) -> None:
        executor = TradeExecutor(commission_pct=0.0, slippage_pct=0.0)
        portfolio = Portfolio(initial_capital=10000.0, position_size_pct=100.0, executor=executor)
        # 100 shares at 100 = 10,000 cost. Cash becomes 0.
        portfolio.open_position(entry_date=date(2023, 1, 2), raw_price=100.0)
        assert portfolio.cash == 0.0

        # Close at 120. Exit value = 12,000. Cash becomes 12,000.
        trade = portfolio.close_position(
            exit_date=date(2023, 1, 12),
            raw_price=120.0,
            exit_reason=ExitReason.TAKE_PROFIT,
        )
        assert trade is not None
        assert not portfolio.is_invested()
        assert portfolio.cash == 12000.0
        assert trade.gross_pnl == 2000.0
        assert trade.net_pnl == 2000.0
        assert trade.return_pct == 20.0
        assert trade.holding_days == 10
        assert len(portfolio.closed_trades) == 1
