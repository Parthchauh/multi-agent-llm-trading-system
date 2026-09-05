"""
tests/test_execution.py
=======================
Tests for backtesting/execution.py (TradeExecutor).
"""

import pytest
from backtesting.execution import TradeExecutor
from backtesting.models import ExitReason


class TestTradeExecutor:
    def test_buy_slippage_worsens_price(self) -> None:
        executor = TradeExecutor(commission_pct=0.001, slippage_pct=0.001)  # 0.1% slippage
        # Buy: 100 * (1 + 0.001) = 100.10
        fill = executor.calculate_buy_fill(100.0)
        assert pytest.approx(fill, 1e-6) == 100.10

    def test_sell_slippage_worsens_price(self) -> None:
        executor = TradeExecutor(commission_pct=0.001, slippage_pct=0.001)  # 0.1% slippage
        # Sell: 100 * (1 - 0.001) = 99.90
        fill = executor.calculate_sell_fill(100.0)
        assert pytest.approx(fill, 1e-6) == 99.90

    def test_commission_calculation(self) -> None:
        executor = TradeExecutor(commission_pct=0.002, slippage_pct=0.0005)  # 0.2% commission
        comm = executor.calculate_commission(10000.0)
        assert pytest.approx(comm, 1e-6) == 20.0

    def test_gap_down_below_stop_fills_at_open(self) -> None:
        executor = TradeExecutor(slippage_pct=0.001)
        # Stop at 95. Open gaps down to 90.
        res = executor.evaluate_stop_and_target(
            open_price=90.0,
            high_price=92.0,
            low_price=88.0,
            stop_price=95.0,
            target_price=110.0,
        )
        assert res is not None
        reason, fill_price = res
        assert reason == ExitReason.STOP_LOSS
        # Fill should be open price with sell slippage: 90 * (1 - 0.001) = 89.91
        assert pytest.approx(fill_price, 1e-6) == 89.91

    def test_gap_up_above_target_fills_at_open(self) -> None:
        executor = TradeExecutor(slippage_pct=0.001)
        # Target at 110. Open gaps up to 115.
        res = executor.evaluate_stop_and_target(
            open_price=115.0,
            high_price=118.0,
            low_price=114.0,
            stop_price=95.0,
            target_price=110.0,
        )
        assert res is not None
        reason, fill_price = res
        assert reason == ExitReason.TAKE_PROFIT
        # Fill should be open price with sell slippage: 115 * (1 - 0.001) = 114.885
        assert pytest.approx(fill_price, 1e-6) == 114.885

    def test_stop_hit_intraday_fills_at_stop_price(self) -> None:
        executor = TradeExecutor(slippage_pct=0.001)
        # Open 100, Low 94, High 102. Stop at 95. Target at 110.
        res = executor.evaluate_stop_and_target(
            open_price=100.0,
            high_price=102.0,
            low_price=94.0,
            stop_price=95.0,
            target_price=110.0,
        )
        assert res is not None
        reason, fill_price = res
        assert reason == ExitReason.STOP_LOSS
        # Fill at stop price: 95 * (1 - 0.001) = 94.905
        assert pytest.approx(fill_price, 1e-6) == 94.905

    def test_target_hit_intraday_fills_at_target_price(self) -> None:
        executor = TradeExecutor(slippage_pct=0.001)
        # Open 100, High 112, Low 98. Stop at 95. Target at 110.
        res = executor.evaluate_stop_and_target(
            open_price=100.0,
            high_price=112.0,
            low_price=98.0,
            stop_price=95.0,
            target_price=110.0,
        )
        assert res is not None
        reason, fill_price = res
        assert reason == ExitReason.TAKE_PROFIT
        # Fill at target price: 110 * (1 - 0.001) = 109.89
        assert pytest.approx(fill_price, 1e-6) == 109.89

    def test_stop_and_target_both_hit_same_bar_stop_wins(self) -> None:
        executor = TradeExecutor(slippage_pct=0.001)
        # Open 100, High 115, Low 90. Stop at 95. Target at 110.
        res = executor.evaluate_stop_and_target(
            open_price=100.0,
            high_price=115.0,
            low_price=90.0,
            stop_price=95.0,
            target_price=110.0,
        )
        assert res is not None
        reason, fill_price = res
        assert reason == ExitReason.STOP_LOSS
        # Conservative rule: stop loss fill at 95 * (1 - 0.001) = 94.905
        assert pytest.approx(fill_price, 1e-6) == 94.905

    def test_neither_hit_returns_none(self) -> None:
        executor = TradeExecutor()
        res = executor.evaluate_stop_and_target(
            open_price=100.0,
            high_price=105.0,
            low_price=98.0,
            stop_price=95.0,
            target_price=110.0,
        )
        assert res is None
