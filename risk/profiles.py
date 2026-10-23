"""Versioned complete launch profiles; thresholds are research policy, not advice."""

import hashlib

from risk.models import RiskPolicy

MANDATORY = (
    "max_drawdown_pct",
    "max_position_pct",
    "max_loss_per_trade_pct",
    "max_turnover_pct",
    "max_portfolio_exposure_pct",
    "minimum_trade_count",
    "max_concurrent_positions",
)


def require_complete_policy(policy: RiskPolicy) -> RiskPolicy:
    policy = RiskPolicy.model_validate(policy.model_dump())
    missing = [field for field in MANDATORY if getattr(policy, field) is None]
    if missing:
        raise ValueError("Incomplete launch risk policy: " + ", ".join(missing))
    if policy.minimum_trade_count < 20:
        raise ValueError("Launch risk policy requires at least 20 completed trades.")
    if policy.max_drawdown_pct > 100.0 or policy.max_loss_per_trade_pct > 100.0:
        raise ValueError("Unleveraged loss and drawdown limits cannot exceed 100 percent.")
    return policy


def risk_profile(name: str = "research_standard") -> RiskPolicy:
    limits = {
        "research_conservative": (10.0, 10.0, 1.0, 1000.0, 30),
        "research_standard": (20.0, 20.0, 2.0, 2000.0, 20),
        "paper_beta": (10.0, 10.0, 1.0, 1000.0, 30),
    }
    if name not in limits:
        raise ValueError(f"Unknown risk profile: {name}")
    drawdown, position, loss, turnover, trades = limits[name]
    return require_complete_policy(
        RiskPolicy(
            policy_id=f"{name}:v1",
            max_drawdown_pct=drawdown,
            max_position_pct=position,
            max_portfolio_exposure_pct=position,
            max_loss_per_trade_pct=loss,
            max_turnover_pct=turnover,
            minimum_trade_count=trades,
            max_concurrent_positions=1,
        )
    )


def policy_hash(policy: RiskPolicy) -> str:
    return hashlib.sha256(policy.model_dump_json().encode()).hexdigest()
