"""Focused deterministic-contract tests for Sprint 6 risk controls."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from backtesting.metrics import calculate_metrics
from backtesting.models import (
    BacktestConfig,
    BacktestResult,
    EquityPoint,
    ExitReason,
    Trade,
    TradeSide,
)
from risk import (
    RiskAssessment,
    RiskPolicy,
    RiskRule,
    RiskViolation,
    SizingError,
    SizingInput,
    SizingMethod,
    SizingPolicy,
    derive_backtest_config,
    derive_position_size,
    derive_sizing_input_from_ohlcv,
    evaluate_risk,
)


def _trade(*, entry_price: float = 250.0, exit_price: float = 150.0) -> Trade:
    return Trade(
        entry_date=date(2024, 1, 2),
        entry_price=entry_price,
        exit_date=date(2024, 1, 3),
        exit_price=exit_price,
        quantity=1,
        gross_pnl=exit_price - entry_price,
        commission_paid=0.0,
        net_pnl=exit_price - entry_price,
        return_pct=((exit_price - entry_price) / entry_price) * 100.0,
        exit_reason=ExitReason.SIGNAL,
        holding_days=1,
        side=TradeSide.LONG,
    )


def _result(*, position_size_pct: float = 25.0) -> BacktestResult:
    config = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.0,
        slippage_pct=0.0,
        position_size_pct=position_size_pct,
    )
    trades = [_trade()]
    equity_curve = [
        EquityPoint(date=date(2024, 1, 1), equity=1000.0, cash=1000.0, is_invested=False),
        EquityPoint(date=date(2024, 1, 2), equity=850.0, cash=750.0, is_invested=True),
        EquityPoint(date=date(2024, 1, 3), equity=900.0, cash=900.0, is_invested=False),
    ]
    return BacktestResult(
        config=config,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
        initial_capital=1000.0,
        final_equity=900.0,
        metrics=calculate_metrics(1000.0, trades, equity_curve),
        trades=trades,
        equity_curve=equity_curve,
    )


class TestRiskPolicyContracts:
    @pytest.mark.parametrize(
        "payload",
        [
            {"max_drawdown_pct": True},
            {"max_position_pct": float("inf")},
            {"minimum_trade_count": True},
            {"max_concurrent_positions": 0},
            {"max_concurrent_positions": 2},
            {"minimum_liquidity": 1_000_000},
        ],
    )
    def test_invalid_or_unsupported_policy_inputs_are_rejected(self, payload: dict) -> None:
        with pytest.raises(ValidationError):
            RiskPolicy(**payload)

    def test_deterministic_assessment_flags_every_supported_violation(self) -> None:
        assessment = evaluate_risk(
            _result(),
            RiskPolicy(
                max_drawdown_pct=10.0,
                max_position_pct=20.0,
                max_portfolio_exposure_pct=20.0,
                max_loss_per_trade_pct=5.0,
                max_turnover_pct=20.0,
                minimum_trade_count=2,
            ),
        )

        assert assessment.passed is False
        assert {violation.rule for violation in assessment.violations} == {
            RiskRule.MAX_DRAWDOWN_PCT,
            RiskRule.MAX_POSITION_PCT,
            RiskRule.MAX_PORTFOLIO_EXPOSURE_PCT,
            RiskRule.MAX_LOSS_PER_TRADE_PCT,
            RiskRule.MAX_TURNOVER_PCT,
            RiskRule.MINIMUM_TRADE_COUNT,
        }
        assert assessment.metrics.max_drawdown_pct == 15.0
        assert assessment.metrics.position_size.effective_position_size_pct == 25.0
        assert assessment.metrics.max_loss_per_trade_pct == 10.0
        assert assessment.metrics.trade_count == 1

    def test_assessment_is_deterministic_and_pass_flag_cannot_override_violations(self) -> None:
        policy = RiskPolicy(max_drawdown_pct=20.0, minimum_trade_count=1)
        first = evaluate_risk(_result(), policy)
        second = evaluate_risk(_result(), policy)

        assert first == second
        assert first.passed is True
        with pytest.raises(ValidationError, match="passed"):
            RiskAssessment(
                passed=True,
                violations=(
                    RiskViolation(
                        rule=RiskRule.MAX_DRAWDOWN_PCT,
                        observed_value=15.0,
                        limit=10.0,
                    ),
                ),
                metrics=first.metrics,
            )


class TestDeterministicSizing:
    def test_fixed_fraction_is_capped_and_whole_share_deterministic(self) -> None:
        decision = derive_position_size(
            SizingPolicy(
                method=SizingMethod.FIXED_FRACTION,
                fixed_fraction_pct=60.0,
                maximum_position_pct=25.0,
            ),
            SizingInput(as_of_index=20, capital=1000.0, entry_price=120.0),
        )

        assert decision.requested_position_pct == 60.0
        assert decision.capped_position_pct == 25.0
        assert decision.quantity == 2
        assert decision.actual_notional == 240.0
        assert decision.actual_position_pct == 24.0

    def test_atr_and_volatility_sizing_are_bounded_without_llm_arithmetic(self) -> None:
        atr_decision = derive_position_size(
            SizingPolicy(
                method=SizingMethod.ATR_RISK,
                risk_per_trade_pct=2.0,
                atr_stop_multiple=2.0,
                maximum_position_pct=30.0,
            ),
            SizingInput(as_of_index=20, capital=1000.0, entry_price=100.0, atr=5.0),
        )
        volatility_decision = derive_position_size(
            SizingPolicy(
                method=SizingMethod.VOLATILITY_ADJUSTED,
                base_position_pct=50.0,
                target_volatility_pct=10.0,
                maximum_position_pct=30.0,
            ),
            SizingInput(
                as_of_index=20,
                capital=1000.0,
                entry_price=100.0,
                realised_volatility_pct=5.0,
            ),
        )

        assert atr_decision.per_share_risk == 10.0
        assert atr_decision.risk_budget == 20.0
        assert atr_decision.quantity == 2
        assert atr_decision.actual_position_pct == 20.0
        assert volatility_decision.requested_position_pct == 100.0
        assert volatility_decision.capped_position_pct == 30.0
        assert volatility_decision.actual_position_pct == 30.0

    def test_backtest_config_bridge_preserves_execution_costs_and_rejects_zero_share(self) -> None:
        base = BacktestConfig(
            initial_capital=1000.0,
            commission_pct=0.001,
            slippage_pct=0.002,
            position_size_pct=10.0,
            maximum_holding_days=5,
        )
        decision = derive_position_size(
            SizingPolicy(
                method=SizingMethod.FIXED_FRACTION,
                fixed_fraction_pct=20.0,
            ),
            SizingInput(as_of_index=5, capital=1000.0, entry_price=100.0),
        )
        derived = derive_backtest_config(base, decision)

        assert derived.position_size_pct == 20.0
        assert derived.commission_pct == base.commission_pct
        assert derived.slippage_pct == base.slippage_pct
        assert derived.maximum_holding_days == base.maximum_holding_days

        zero_share = derive_position_size(
            SizingPolicy(
                method=SizingMethod.FIXED_FRACTION,
                fixed_fraction_pct=1.0,
            ),
            SizingInput(as_of_index=5, capital=100.0, entry_price=10_000.0),
        )
        with pytest.raises(SizingError, match="zero-share"):
            derive_backtest_config(base, zero_share)


def test_causal_ohlcv_sizing_ignores_future_perturbation() -> None:
    rows = 50
    close = np.linspace(100.0, 130.0, rows) + np.sin(np.arange(rows))
    frame = pd.DataFrame(
        {
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
        },
        index=pd.date_range("2024-01-01", periods=rows, freq="D"),
    )
    cutoff = 30
    baseline_input = derive_sizing_input_from_ohlcv(
        frame,
        as_of_index=cutoff,
        capital=1000.0,
        atr_lookback=5,
        volatility_lookback=10,
    )
    perturbed = frame.copy(deep=True)
    perturbed.loc[perturbed.index[cutoff + 1 :], ["High", "Low", "Close"]] *= 100.0
    perturbed_input = derive_sizing_input_from_ohlcv(
        perturbed,
        as_of_index=cutoff,
        capital=1000.0,
        atr_lookback=5,
        volatility_lookback=10,
    )
    policy = SizingPolicy(
        method=SizingMethod.VOLATILITY_ADJUSTED,
        base_position_pct=25.0,
        target_volatility_pct=15.0,
        maximum_position_pct=50.0,
    )

    assert baseline_input == perturbed_input
    assert derive_position_size(policy, baseline_input) == derive_position_size(
        policy,
        perturbed_input,
    )
