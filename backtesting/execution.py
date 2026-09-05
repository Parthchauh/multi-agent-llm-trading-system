"""
backtesting/execution.py
========================
Deterministic trade execution logic with realistic slippage, commission,
and gap calculations.

Responsibilities
----------------
* Calculate executable entry price with buy slippage:
    fill_price = raw_price * (1 + slippage_pct)
* Calculate executable exit price with sell slippage:
    fill_price = raw_price * (1 - slippage_pct)
* Calculate commissions on trade value:
    commission = trade_value * commission_pct
* Handle price gaps for stops and targets
* Evaluate intraday / gap stop-loss and take-profit triggers with conservative
  same-bar tie-breaking (Stop Loss executes before Take Profit).
"""

from __future__ import annotations

from typing import Optional, Tuple

from backtesting.models import ExitReason


class TradeExecutor:
    """Calculates executable fills, commissions, and stops/targets.

    Parameters
    ----------
    commission_pct:
        Decimal fraction commission per trade leg (e.g. 0.001 = 0.1%).
    slippage_pct:
        Decimal fraction slippage applied to each fill (e.g. 0.0005 = 0.05%).
    """

    def __init__(self, commission_pct: float = 0.001, slippage_pct: float = 0.0005) -> None:
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def calculate_buy_fill(self, raw_price: float) -> float:
        """Calculate executed buy price with slippage (worsened higher)."""
        return raw_price * (1.0 + self.slippage_pct)

    def calculate_sell_fill(self, raw_price: float) -> float:
        """Calculate executed sell price with slippage (worsened lower)."""
        return raw_price * (1.0 - self.slippage_pct)

    def calculate_commission(self, trade_value: float) -> float:
        """Calculate commission charged on a transaction value."""
        return trade_value * self.commission_pct

    def evaluate_stop_and_target(
        self,
        open_price: float,
        high_price: float,
        low_price: float,
        stop_price: Optional[float],
        target_price: Optional[float],
    ) -> Optional[Tuple[ExitReason, float]]:
        """Evaluate if an active position's stop loss or take profit was hit on this bar.

        Conservative Execution Rules (Long-Only):
        ------------------------------------------
        1. Gap down on Open:
           If open <= stop_price:
               Stop hit on open gap! Fill at open price with sell slippage.
        2. Gap up on Open:
           If open >= target_price (when target defined):
               Target hit on open gap! Fill at open price with sell slippage.
        3. Intraday touch (both touched on same day):
           If low <= stop_price AND high >= target_price:
               Conservative rule: STOP LOSS executes first. Fill at stop_price with sell slippage.
        4. Stop Loss touched intraday:
           If low <= stop_price:
               Stop hit intraday. Fill at stop_price with sell slippage.
        5. Take Profit touched intraday:
           If target_price is not None and high >= target_price:
               Target hit intraday. Fill at target_price with sell slippage.

        Returns
        -------
        Optional[Tuple[ExitReason, float]]
            (ExitReason, fill_price) or None if neither triggered.
        """
        # 1. Gap below stop on open
        if stop_price is not None and open_price <= stop_price:
            fill = self.calculate_sell_fill(open_price)
            return ExitReason.STOP_LOSS, fill

        # 2. Gap above target on open
        if target_price is not None and open_price >= target_price:
            fill = self.calculate_sell_fill(open_price)
            return ExitReason.TAKE_PROFIT, fill

        # Check intraday triggers
        stop_hit = stop_price is not None and low_price <= stop_price
        target_hit = target_price is not None and high_price >= target_price

        # 3. Both hit on the same candle -> Conservative rule: Stop loss wins
        if stop_hit and target_hit:
            fill = self.calculate_sell_fill(stop_price)  # type: ignore[arg-type]
            return ExitReason.STOP_LOSS, fill

        # 4. Stop loss hit intraday
        if stop_hit:
            fill = self.calculate_sell_fill(stop_price)  # type: ignore[arg-type]
            return ExitReason.STOP_LOSS, fill

        # 5. Take profit hit intraday
        if target_hit:
            fill = self.calculate_sell_fill(target_price)  # type: ignore[arg-type]
            return ExitReason.TAKE_PROFIT, fill

        return None
