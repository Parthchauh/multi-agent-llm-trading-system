"""Pure deterministic risk evaluation over completed backtest artifacts.

The risk package never recalculates PnL, never consumes LLM text, and never
executes a generated expression.  It turns facts already produced by the
authoritative backtester into an immutable :class:`RiskAssessment`.
"""

from __future__ import annotations

import math

from backtesting.models import BacktestConfig, BacktestResult
from risk.exceptions import RiskDataError
from risk.models import (
    EffectivePositionSize,
    RiskAssessment,
    RiskMetrics,
    RiskPolicy,
    RiskRule,
    RiskViolation,
)


def _require_finite_non_negative(value: object, label: str) -> float:
    """Read one deterministic numeric fact without accepting malformed data."""

    if isinstance(value, bool):
        raise RiskDataError(f"{label} must be numeric, not boolean.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RiskDataError(f"{label} must be a finite number.") from exc
    if not math.isfinite(number) or number < 0.0:
        raise RiskDataError(f"{label} must be a finite non-negative number.")
    return number


def configured_position_size_from_backtest_config(config: BacktestConfig) -> float:
    """Return the validated pre-run position allocation percentage.

    This is intentionally a configuration helper, rather than a second order
    sizing system.  The caller must select and validate the configuration
    *before* calling ``BacktestEngine.run``.  The engine's currently supported
    long-only allocation is a fraction of available cash and is the only
    source of configured position-size evidence for risk evaluation.
    """

    if not isinstance(config, BacktestConfig):
        raise RiskDataError("config must be a validated BacktestConfig.")
    size = _require_finite_non_negative(config.position_size_pct, "position_size_pct")
    if size <= 0.0 or size > 100.0:
        # BacktestConfig already enforces this.  Keep the guard local so a
        # handcrafted/mutated object cannot weaken a risk report.
        raise RiskDataError("position_size_pct must be in (0, 100].")
    return size


def derive_effective_position_size(result: BacktestResult) -> EffectivePositionSize:
    """Derive conservative single-position size evidence from a completed run.

    The configured percentage is known before execution.  The maximum filled
    entry notional is also measured against *initial* capital after the run.
    Taking the larger value prevents a growing cash balance or share rounding
    from making a risk report claim a smaller allocation than either source
    establishes.  This function deliberately uses completed trade records
    only; it does not inspect market bars or future data.
    """

    if not isinstance(result, BacktestResult):
        raise RiskDataError("result must be a BacktestResult produced by the deterministic engine.")

    initial_capital = _require_finite_non_negative(result.initial_capital, "initial_capital")
    if initial_capital <= 0.0:
        raise RiskDataError("initial_capital must be positive.")
    configured = configured_position_size_from_backtest_config(result.config)

    observed_max: float | None = None
    for index, trade in enumerate(result.trades):
        entry_price = _require_finite_non_negative(
            trade.entry_price,
            f"trades[{index}].entry_price",
        )
        quantity = _require_finite_non_negative(
            trade.quantity,
            f"trades[{index}].quantity",
        )
        if entry_price <= 0.0 or quantity <= 0.0:
            raise RiskDataError(
                f"trades[{index}] must have positive entry_price and quantity."
            )
        entry_notional_pct = (entry_price * quantity / initial_capital) * 100.0
        if not math.isfinite(entry_notional_pct):
            raise RiskDataError(f"trades[{index}] entry notional is not finite.")
        observed_max = (
            entry_notional_pct
            if observed_max is None
            else max(observed_max, entry_notional_pct)
        )

    effective = max(
        configured,
        observed_max if observed_max is not None else configured,
    )
    return EffectivePositionSize(
        configured_position_size_pct=configured,
        max_entry_notional_pct_initial_capital=observed_max,
        effective_position_size_pct=effective,
    )


def _derive_risk_metrics_and_warnings(
    result: BacktestResult,
) -> tuple[RiskMetrics, tuple[str, ...]]:
    """Return risk facts plus non-fatal deterministic artifact-consistency notes."""

    if not isinstance(result, BacktestResult):
        raise RiskDataError("result must be a BacktestResult produced by the deterministic engine.")

    initial_capital = _require_finite_non_negative(result.initial_capital, "initial_capital")
    if initial_capital <= 0.0:
        raise RiskDataError("initial_capital must be positive.")
    max_drawdown_pct = _require_finite_non_negative(
        result.metrics.max_drawdown_pct,
        "metrics.max_drawdown_pct",
    )
    turnover_raw = result.metrics.turnover_pct
    turnover_pct = (
        None
        if turnover_raw is None
        else _require_finite_non_negative(turnover_raw, "metrics.turnover_pct")
    )

    maximum_loss = 0.0
    for index, trade in enumerate(result.trades):
        if isinstance(trade.net_pnl, bool):
            raise RiskDataError(f"trades[{index}].net_pnl must be numeric, not boolean.")
        try:
            net_pnl = float(trade.net_pnl)
        except (TypeError, ValueError) as exc:
            raise RiskDataError(f"trades[{index}].net_pnl must be finite.") from exc
        if not math.isfinite(net_pnl):
            raise RiskDataError(f"trades[{index}].net_pnl must be finite.")
        maximum_loss = max(maximum_loss, max(0.0, -net_pnl))

    # The engine is structurally single-position.  A completed trade or any
    # invested equity snapshot establishes one observed concurrent position;
    # no inference about unavailable intrabar data is made.
    has_position = bool(result.trades) or any(
        point.is_invested for point in result.equity_curve
    )
    observed_concurrency = 1 if has_position else 0
    trade_count = len(result.trades)
    warnings: list[str] = []
    reported_trade_count = result.metrics.number_of_trades
    if reported_trade_count != trade_count:
        warnings.append(
            "BacktestMetrics.number_of_trades did not match completed trade records; "
            "risk evaluation used completed trade records."
        )

    metrics = RiskMetrics(
        initial_capital=initial_capital,
        max_drawdown_pct=max_drawdown_pct,
        position_size=derive_effective_position_size(result),
        max_loss_per_trade_pct=(maximum_loss / initial_capital) * 100.0,
        turnover_pct=turnover_pct,
        trade_count=trade_count,
        max_concurrent_positions=observed_concurrency,
    )
    return metrics, tuple(warnings)


def derive_risk_metrics(result: BacktestResult) -> RiskMetrics:
    """Derive all currently enforceable risk facts from a completed backtest."""

    metrics, _ = _derive_risk_metrics_and_warnings(result)
    return metrics


def _limit_violation(
    rule: RiskRule,
    observed: float | int | None,
    limit: float | int,
    *,
    detail: str,
) -> RiskViolation:
    return RiskViolation(
        rule=rule,
        observed_value=observed,
        limit=limit,
        detail=detail,
    )


def evaluate_risk(result: BacktestResult, policy: RiskPolicy) -> RiskAssessment:
    """Apply a strict RiskPolicy to one deterministic backtest result.

    A missing metric required by a configured rule is treated as a violation,
    not as a passing value.  This makes the returned ``passed`` flag suitable
    as an authoritative graph gate: an LLM Risk Manager may explain the result
    but cannot clear an assessment with any violation.
    """

    if not isinstance(policy, RiskPolicy):
        raise RiskDataError("policy must be a validated RiskPolicy.")

    metrics, warnings = _derive_risk_metrics_and_warnings(result)
    violations: list[RiskViolation] = []

    if (
        policy.max_drawdown_pct is not None
        and metrics.max_drawdown_pct > policy.max_drawdown_pct
    ):
        violations.append(
            _limit_violation(
                RiskRule.MAX_DRAWDOWN_PCT,
                metrics.max_drawdown_pct,
                policy.max_drawdown_pct,
                detail="Maximum drawdown exceeded the configured limit.",
            )
        )

    effective_size = metrics.position_size.effective_position_size_pct
    if (
        policy.max_position_pct is not None
        and (effective_size is None or effective_size > policy.max_position_pct)
    ):
        violations.append(
            _limit_violation(
                RiskRule.MAX_POSITION_PCT,
                effective_size,
                policy.max_position_pct,
                detail="Effective single-position allocation exceeded the configured limit.",
            )
        )

    if (
        policy.max_portfolio_exposure_pct is not None
        and (
            effective_size is None
            or effective_size > policy.max_portfolio_exposure_pct
        )
    ):
        violations.append(
            _limit_violation(
                RiskRule.MAX_PORTFOLIO_EXPOSURE_PCT,
                effective_size,
                policy.max_portfolio_exposure_pct,
                detail=(
                    "Single-position portfolio exposure exceeded the configured limit."
                ),
            )
        )

    if (
        policy.max_loss_per_trade_pct is not None
        and metrics.max_loss_per_trade_pct > policy.max_loss_per_trade_pct
    ):
        violations.append(
            _limit_violation(
                RiskRule.MAX_LOSS_PER_TRADE_PCT,
                metrics.max_loss_per_trade_pct,
                policy.max_loss_per_trade_pct,
                detail="Largest completed net loss exceeded the configured initial-capital limit.",
            )
        )

    if policy.max_turnover_pct is not None:
        if metrics.turnover_pct is None or metrics.turnover_pct > policy.max_turnover_pct:
            violations.append(
                _limit_violation(
                    RiskRule.MAX_TURNOVER_PCT,
                    metrics.turnover_pct,
                    policy.max_turnover_pct,
                    detail=(
                        "Turnover was unavailable or exceeded the configured limit; "
                        "no substitute value was used."
                    ),
                )
            )

    if (
        policy.minimum_trade_count is not None
        and metrics.trade_count < policy.minimum_trade_count
    ):
        violations.append(
            _limit_violation(
                RiskRule.MINIMUM_TRADE_COUNT,
                metrics.trade_count,
                policy.minimum_trade_count,
                detail="Completed trade count was below the configured minimum.",
            )
        )

    if metrics.max_concurrent_positions > policy.max_concurrent_positions:
        violations.append(
            _limit_violation(
                RiskRule.MAX_CONCURRENT_POSITIONS,
                metrics.max_concurrent_positions,
                policy.max_concurrent_positions,
                detail="Observed concurrent positions exceeded the configured engine-compatible limit.",
            )
        )

    return RiskAssessment(
        passed=not violations,
        violations=tuple(violations),
        warnings=warnings,
        metrics=metrics,
    )


# Naming aliases make the authoritative implementation easy to discover while
# preserving exactly one risk-evaluation path.
assess_risk = evaluate_risk
assess_backtest_risk = evaluate_risk
