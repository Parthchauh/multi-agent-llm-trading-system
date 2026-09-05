"""
backtesting/metrics.py
======================
Deterministic financial metrics calculations for backtest results.

All calculations handle edge cases (zero trades, zero standard deviation,
no losing trades, no drawdown) safely without crashing or returning infinity.

Metric definitions are deliberately execution-derived: no annualisation is
applied to turnover, and every trade statistic uses completed, net-PnL trades.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from backtesting.models import BacktestMetrics, EquityPoint, Trade


def _finite_equity_observations(
    equity_curve: List[EquityPoint],
) -> list[tuple[int, EquityPoint]]:
    """Return finite equity points with their original bar positions.

    The backtest engine normally guarantees finite equity.  Keeping the
    original positions here nevertheless makes duration accounting defensive:
    an invalid observation cannot silently compress elapsed bar counts.
    """
    return [
        (bar_index, point)
        for bar_index, point in enumerate(equity_curve)
        if math.isfinite(point.equity)
    ]


def _max_drawdown_duration(
    observations: list[tuple[int, EquityPoint]],
) -> tuple[int, int]:
    """Calculate longest peak-to-recovery drawdown duration.

    A drawdown begins at the most recent running-equity high and ends when the
    equity returns to that high or makes a new high.  If it remains unrecovered
    at the last finite observation, its duration ends there.  Durations are
    measured as elapsed bar intervals and calendar days, so a one-bar decline
    from a peak has duration ``1`` bar; a path with no decline has ``0``.

    Calendar-day and bar durations are maximised independently.  That avoids
    treating irregularly sampled data as daily data while still exposing both
    useful, deterministic representations of the longest underwater periods.
    """
    if len(observations) < 2:
        return 0, 0

    peak_bar_index, peak_point = observations[0]
    peak_equity = peak_point.equity
    peak_date = peak_point.date
    is_underwater = False
    max_duration_bars = 0
    max_duration_days = 0

    def record_duration(bar_index: int, point: EquityPoint) -> None:
        nonlocal max_duration_bars, max_duration_days
        max_duration_bars = max(max_duration_bars, bar_index - peak_bar_index)
        # Backtest equity curves are chronological.  Clamp defensively so an
        # invalid caller cannot emit a negative duration into a result payload.
        max_duration_days = max(
            max_duration_days,
            max(0, (point.date - peak_date).days),
        )

    for bar_index, point in observations[1:]:
        if point.equity < peak_equity:
            is_underwater = True
            # Covers an unrecovered drawdown whose final point is this bar.
            record_duration(bar_index, point)
            continue

        if is_underwater:
            # Include the recovery bar in peak-to-recovery duration.
            record_duration(bar_index, point)

        # Equality is recovery, and becomes the reference peak for a later
        # drawdown.  This measures duration from the most recent high-water
        # mark rather than from an arbitrarily older equal high.
        peak_bar_index = bar_index
        peak_point = point
        peak_equity = point.equity
        peak_date = point.date
        is_underwater = False

    return max_duration_bars, max_duration_days


def _calculate_turnover_pct(
    trades: List[Trade],
    equity_curve: List[EquityPoint],
) -> Optional[float]:
    """Return full-period gross executed turnover, without annualisation.

    Turnover is the sum of every completed trade's entry and exit fill
    notional, divided by arithmetic mean finite end-of-bar equity, multiplied
    by 100.  Both legs intentionally count, so one round trip using the entire
    portfolio is approximately 200% turnover.  No trades is a true zero;
    completed trades without a valid positive denominator or notional are
    undefined and therefore return ``None`` rather than a fabricated value.
    """
    if not trades:
        return 0.0

    finite_equities = [
        point.equity for point in equity_curve if math.isfinite(point.equity)
    ]
    if not finite_equities:
        return None

    average_equity = float(np.mean(finite_equities))
    if not math.isfinite(average_equity) or average_equity <= 0.0:
        return None

    executed_notional = 0.0
    for trade in trades:
        try:
            entry_price = float(trade.entry_price)
            exit_price = float(trade.exit_price)
            quantity = float(trade.quantity)
        except (TypeError, ValueError):
            return None
        if (
            not math.isfinite(entry_price)
            or not math.isfinite(exit_price)
            or not math.isfinite(quantity)
            or entry_price <= 0.0
            or exit_price <= 0.0
            or quantity <= 0.0
        ):
            return None
        executed_notional += (entry_price + exit_price) * quantity

    if not math.isfinite(executed_notional):
        return None
    return round((executed_notional / average_equity) * 100.0, 4)


def calculate_metrics(
    initial_capital: float,
    trades: List[Trade],
    equity_curve: List[EquityPoint],
) -> BacktestMetrics:
    """Compute comprehensive performance metrics from trades and equity curve.

    Parameters
    ----------
    initial_capital:
        Initial starting cash.
    trades:
        List of completed round-trip Trade objects.
    equity_curve:
        List of daily EquityPoint snapshots.

    Returns
    -------
    BacktestMetrics

    Notes
    -----
    Metric semantics (version 1.2):
    - profit_factor: sum(net_pnl of net-winning trades) /
                     abs(sum(net_pnl of net-losing trades)).
      Uses the SAME net PnL basis as win/loss classification and expectancy.
      Gross-positive but net-negative trades are counted as losses here.
    - largest_win: maximum net_pnl among net-winning trades. None if no winners.
    - largest_loss: minimum net_pnl among net-losing trades. None if no losers.
    - exposure_pct: fraction of end-of-bar snapshots where a position was open.
      Note: same-day (open-to-close) positions may not be reflected here.
    - holding_days: calendar days from entry_date to exit_date (exclusive end).
    - calmar_ratio: CAGR / maximum drawdown.  Undefined (``None``) when CAGR
      is unavailable or maximum drawdown is zero.
    - payoff_ratio: average net win / absolute average net loss.  Undefined
      when either trade side is absent or the average loss is zero.
    - turnover_pct: full-period, two-sided executed notional / average
      end-of-bar equity.  It is deliberately not annualised.
    - max_drawdown_duration_days / _bars: longest peak-to-recovery (or
      peak-to-final-observation) interval; both are zero when no drawdown is
      observed.
    """
    if not equity_curve:
        final_equity = initial_capital
        total_return_pct = 0.0
        max_drawdown_pct = 0.0
        sharpe_ratio = None
        sortino_ratio = None
        cagr_pct = None
        cagr_raw = None
        calmar_ratio = None
        exposure_pct = 0.0
        max_drawdown_duration_bars = 0
        max_drawdown_duration_days = 0
    else:
        final_equity = equity_curve[-1].equity
        total_return_pct = ((final_equity / initial_capital) - 1.0) * 100.0

        # Equity array and returns
        finite_observations = _finite_equity_observations(equity_curve)
        equities = np.array(
            [point.equity for _, point in finite_observations],
            dtype=float,
        )

        # Max Drawdown calculation
        if len(equities) > 0:
            peaks = np.maximum.accumulate(equities)
            # Guard against zero-peak equity (pathological case)
            with np.errstate(invalid="ignore", divide="ignore"):
                drawdowns = np.where(peaks > 0, (equities - peaks) / peaks, 0.0)
            max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
            max_drawdown_pct = abs(max_dd) * 100.0
        else:
            max_drawdown_pct = 0.0
        (
            max_drawdown_duration_bars,
            max_drawdown_duration_days,
        ) = _max_drawdown_duration(finite_observations)

        # Daily returns for Sharpe and Sortino
        if len(equities) > 1:
            with np.errstate(invalid="ignore", divide="ignore"):
                daily_returns = np.where(
                    equities[:-1] > 0,
                    np.diff(equities) / equities[:-1],
                    0.0,
                )
            daily_returns = daily_returns[np.isfinite(daily_returns)]

            if len(daily_returns) > 0:
                mean_ret = float(np.mean(daily_returns))
                std_ret = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0

                # Annualized Sharpe (252 trading days, rf = 0)
                if std_ret > 1e-12:
                    sharpe_ratio = round((mean_ret / std_ret) * math.sqrt(252), 4)
                else:
                    sharpe_ratio = None

                # Annualized Sortino (downside deviation from 0)
                negative_returns = daily_returns[daily_returns < 0.0]
                if len(negative_returns) > 0:
                    downside_std = math.sqrt(
                        float(np.sum(negative_returns ** 2)) / len(daily_returns)
                    )
                    if downside_std > 1e-12:
                        sortino_ratio = round((mean_ret / downside_std) * math.sqrt(252), 4)
                    else:
                        sortino_ratio = None
                else:
                    sortino_ratio = None
            else:
                sharpe_ratio = None
                sortino_ratio = None

            # CAGR
            total_days = (equity_curve[-1].date - equity_curve[0].date).days
            if total_days >= 2 and final_equity > 0:
                years = total_days / 365.25
                cagr = ((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0
                cagr_raw = cagr
                cagr_pct = round(cagr, 4)
            else:
                cagr_raw = None
                cagr_pct = None
        else:
            sharpe_ratio = None
            sortino_ratio = None
            cagr_raw = None
            cagr_pct = None

        # Exposure %
        invested_bars = sum(1 for pt in equity_curve if pt.is_invested)
        exposure_pct = round((invested_bars / len(equity_curve)) * 100.0, 4)

        if cagr_raw is not None and max_drawdown_pct > 0.0:
            calmar_ratio = round(cagr_raw / max_drawdown_pct, 4)
        else:
            calmar_ratio = None

    # Trade statistics
    number_of_trades = len(trades)
    if number_of_trades == 0:
        win_rate_pct = None
        profit_factor = None
        expectancy = None
        winning_trades = 0
        losing_trades = 0
        average_win = None
        average_loss = None
        largest_win = None
        largest_loss = None
        average_holding_days = None
        payoff_ratio = None
    else:
        # Net PnL basis for all classifications (version 1.1)
        wins = [t.net_pnl for t in trades if t.net_pnl > 0]
        losses = [t.net_pnl for t in trades if t.net_pnl <= 0]

        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate_pct = round((winning_trades / number_of_trades) * 100.0, 4)

        # Profit factor: net profits / abs(net losses) — consistent with win/loss basis
        net_profits = sum(wins)
        net_losses_abs = abs(sum(losses))
        if net_losses_abs > 0:
            profit_factor = round(net_profits / net_losses_abs, 4)
        else:
            profit_factor = None  # No losing trades → undefined (infinite)

        expectancy = round(float(np.mean([t.net_pnl for t in trades])), 4)
        average_win_raw = float(np.mean(wins)) if wins else None
        average_loss_raw = float(np.mean(losses)) if losses else None
        average_win = round(average_win_raw, 4) if average_win_raw is not None else None
        average_loss = round(average_loss_raw, 4) if average_loss_raw is not None else None

        if (
            average_win_raw is not None
            and average_loss_raw is not None
            and average_loss_raw < 0.0
        ):
            payoff_ratio = round(average_win_raw / abs(average_loss_raw), 4)
        else:
            payoff_ratio = None

        # largest_win: None if no winning trades (avoids returning a negative value)
        largest_win = round(max(wins), 4) if wins else None
        # largest_loss: None if no losing trades (avoids returning None incorrectly)
        largest_loss = round(min(losses), 4) if losses else None

        average_holding_days = round(float(np.mean([t.holding_days for t in trades])), 2)

    turnover_pct = _calculate_turnover_pct(trades, equity_curve)

    return BacktestMetrics(
        total_return_pct=round(total_return_pct, 4),
        cagr_pct=cagr_pct,
        max_drawdown_pct=round(max_drawdown_pct, 4),
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        win_rate_pct=win_rate_pct,
        profit_factor=profit_factor,
        expectancy=expectancy,
        number_of_trades=number_of_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        average_win=average_win,
        average_loss=average_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        average_holding_days=average_holding_days,
        exposure_pct=exposure_pct,
        metrics_version="1.2",
        calmar_ratio=calmar_ratio,
        payoff_ratio=payoff_ratio,
        turnover_pct=turnover_pct,
        max_drawdown_duration_days=max_drawdown_duration_days,
        max_drawdown_duration_bars=max_drawdown_duration_bars,
    )
