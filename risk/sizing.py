"""Deterministic, causal position-sizing helpers for Sprint 6.

The existing backtester remains the sole execution engine.  This module
produces a typed allocation decision before a run and can derive a replacement
immutable ``BacktestConfig`` from that decision.  It never asks an LLM to do
arithmetic and never accepts generated Python code.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backtesting.models import BacktestConfig
from risk.exceptions import InsufficientRiskDataError, SizingError

if TYPE_CHECKING:
    from pandas import DataFrame


class _FrozenStrictModel(BaseModel):
    """Shared immutable boundary for externally supplied sizing facts."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def _finite(value: float | int | None, field_name: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number or None.")
    return value


class SizingMethod(str, Enum):
    """Supported deterministic allocation methods."""

    FIXED_FRACTION = "FIXED_FRACTION"
    ATR_RISK = "ATR_RISK"
    VOLATILITY_ADJUSTED = "VOLATILITY_ADJUSTED"


class SizingPolicy(_FrozenStrictModel):
    """Immutable inputs to one deterministic sizing calculation.

    The three methods have deliberately narrow, explicit semantics:

    * ``FIXED_FRACTION`` allocates ``fixed_fraction_pct`` of capital.
    * ``ATR_RISK`` sizes shares so an ``atr_stop_multiple`` adverse move uses
      at most ``risk_per_trade_pct`` of capital, then caps the notional at
      ``maximum_position_pct``.
    * ``VOLATILITY_ADJUSTED`` scales ``base_position_pct`` by
      ``target_volatility_pct / realised_volatility_pct``, then applies the
      same maximum-position cap.

    The policy is a pre-backtest input.  It is not a claim that the current
    engine can dynamically rebalance an open position.
    """

    method: SizingMethod
    maximum_position_pct: float = Field(default=100.0)
    fixed_fraction_pct: float | None = None
    risk_per_trade_pct: float | None = None
    atr_stop_multiple: float | None = None
    target_volatility_pct: float | None = None
    base_position_pct: float | None = None

    @field_validator(
        "maximum_position_pct",
        "fixed_fraction_pct",
        "risk_per_trade_pct",
        "atr_stop_multiple",
        "target_volatility_pct",
        "base_position_pct",
        mode="before",
    )
    @classmethod
    def numeric_policy_values_must_not_be_boolean(
        cls,
        value: float | None,
        info,
    ) -> float | None:
        return _finite(value, info.field_name)  # type: ignore[return-value]

    @model_validator(mode="after")
    def method_parameters_must_be_complete_and_valid(self) -> "SizingPolicy":
        if not 0.0 < self.maximum_position_pct <= 100.0:
            raise ValueError("maximum_position_pct must be in (0, 100].")

        positive_fields = (
            "fixed_fraction_pct",
            "risk_per_trade_pct",
            "atr_stop_multiple",
            "target_volatility_pct",
            "base_position_pct",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be > 0 when configured.")

        requirements: dict[SizingMethod, tuple[str, ...]] = {
            SizingMethod.FIXED_FRACTION: ("fixed_fraction_pct",),
            SizingMethod.ATR_RISK: ("risk_per_trade_pct", "atr_stop_multiple"),
            SizingMethod.VOLATILITY_ADJUSTED: (
                "target_volatility_pct",
                "base_position_pct",
            ),
        }
        missing = [
            name for name in requirements[self.method] if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                f"{self.method.value} requires policy fields: {', '.join(missing)}."
            )
        return self


class SizingInput(_FrozenStrictModel):
    """Facts observable on or before the sizing decision bar.

    ``as_of_index`` identifies the final included market bar.  ``entry_price``
    is a reference price for an allocation target, not a promise of a future
    fill; the authoritative backtester still applies its next-bar execution,
    slippage, commission, and integer-share rules.
    """

    as_of_index: int = Field(ge=0)
    capital: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    atr: float | None = None
    realised_volatility_pct: float | None = None

    @field_validator("as_of_index", mode="before")
    @classmethod
    def as_of_index_must_not_be_boolean(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("as_of_index must be an integer, not a boolean.")
        return value

    @field_validator("capital", "entry_price", "atr", "realised_volatility_pct")
    @classmethod
    def values_must_be_finite(cls, value: float | None, info) -> float | None:
        checked = _finite(value, info.field_name)
        if checked is not None and float(checked) < 0.0:
            raise ValueError(f"{info.field_name} must be non-negative when defined.")
        return checked  # type: ignore[return-value]


class PositionSizingDecision(_FrozenStrictModel):
    """Fully auditable deterministic allocation result for one entry decision."""

    method: SizingMethod
    as_of_index: int = Field(ge=0)
    capital: float = Field(gt=0)
    reference_price: float = Field(gt=0)
    requested_position_pct: float = Field(ge=0)
    capped_position_pct: float = Field(ge=0, le=100)
    actual_position_pct: float = Field(ge=0, le=100)
    quantity: int = Field(ge=0)
    target_notional: float = Field(ge=0)
    actual_notional: float = Field(ge=0)
    risk_budget: float | None = Field(default=None, ge=0)
    per_share_risk: float | None = Field(default=None, ge=0)
    warnings: tuple[str, ...] = ()

    @field_validator(
        "capital",
        "reference_price",
        "requested_position_pct",
        "capped_position_pct",
        "actual_position_pct",
        "target_notional",
        "actual_notional",
        "risk_budget",
        "per_share_risk",
    )
    @classmethod
    def numeric_decision_values_must_be_finite(
        cls,
        value: float | None,
        info,
    ) -> float | None:
        checked = _finite(value, info.field_name)
        if checked is not None and float(checked) < 0.0:
            raise ValueError(f"{info.field_name} must be non-negative when defined.")
        return checked  # type: ignore[return-value]

    @field_validator("as_of_index", "quantity", mode="before")
    @classmethod
    def integer_decision_values_must_not_be_boolean(cls, value: int, info) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be an integer, not a boolean.")
        return value

    @field_validator("warnings")
    @classmethod
    def warnings_must_not_be_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not warning.strip() for warning in value):
            raise ValueError("warnings must not contain blank text.")
        return value

    @model_validator(mode="after")
    def decision_must_not_overstate_its_actual_allocation(self) -> "PositionSizingDecision":
        if self.capped_position_pct > self.requested_position_pct + 1e-12:
            raise ValueError("capped_position_pct cannot exceed requested_position_pct.")
        if self.actual_notional > self.target_notional + 1e-8:
            raise ValueError("actual_notional cannot exceed target_notional.")
        return self


def _floor_quantity(notional: float, price: float) -> int:
    """Return an integer quantity without permitting leverage or fractional shares."""

    if not math.isfinite(notional) or not math.isfinite(price) or notional < 0 or price <= 0:
        raise SizingError("notional and price must be finite, with price > 0.")
    return max(0, math.floor(notional / price))


def derive_position_size(
    policy: SizingPolicy,
    sizing_input: SizingInput,
) -> PositionSizingDecision:
    """Calculate one reproducible allocation using only causal scalar inputs."""

    if not isinstance(policy, SizingPolicy):
        raise SizingError("policy must be a validated SizingPolicy.")
    if not isinstance(sizing_input, SizingInput):
        raise SizingError("sizing_input must be a validated SizingInput.")

    capital = sizing_input.capital
    price = sizing_input.entry_price
    requested_pct: float
    risk_budget: float | None = None
    per_share_risk: float | None = None

    if policy.method is SizingMethod.FIXED_FRACTION:
        assert policy.fixed_fraction_pct is not None
        requested_pct = policy.fixed_fraction_pct
    elif policy.method is SizingMethod.ATR_RISK:
        if sizing_input.atr is None or sizing_input.atr <= 0.0:
            raise SizingError("ATR_RISK requires a positive causal atr input.")
        assert policy.risk_per_trade_pct is not None
        assert policy.atr_stop_multiple is not None
        risk_budget = capital * (policy.risk_per_trade_pct / 100.0)
        per_share_risk = sizing_input.atr * policy.atr_stop_multiple
        if not math.isfinite(per_share_risk) or per_share_risk <= 0.0:
            raise SizingError("ATR_RISK produced an invalid per-share risk.")
        # ``risk_budget / per_share_risk`` is already a share count, unlike
        # the notional-based branches handled by ``_floor_quantity``.
        requested_notional = max(0, math.floor(risk_budget / per_share_risk)) * price
        requested_pct = (requested_notional / capital) * 100.0
    elif policy.method is SizingMethod.VOLATILITY_ADJUSTED:
        if (
            sizing_input.realised_volatility_pct is None
            or sizing_input.realised_volatility_pct <= 0.0
        ):
            raise SizingError(
                "VOLATILITY_ADJUSTED requires a positive causal realised_volatility_pct input."
            )
        assert policy.target_volatility_pct is not None
        assert policy.base_position_pct is not None
        requested_pct = (
            policy.base_position_pct
            * policy.target_volatility_pct
            / sizing_input.realised_volatility_pct
        )
    else:  # pragma: no cover - exhaustive enum guard for future extensions
        raise SizingError(f"Unsupported deterministic sizing method: {policy.method!r}.")

    if not math.isfinite(requested_pct) or requested_pct < 0.0:
        raise SizingError("Sizing calculation produced an invalid requested allocation.")

    capped_pct = min(requested_pct, policy.maximum_position_pct)
    target_notional = capital * (capped_pct / 100.0)
    quantity = _floor_quantity(target_notional, price)
    actual_notional = quantity * price
    actual_pct = (actual_notional / capital) * 100.0
    warnings: tuple[str, ...] = ()
    if quantity == 0:
        warnings = (
            "Sizing target is below one whole share at the supplied reference price.",
        )

    return PositionSizingDecision(
        method=policy.method,
        as_of_index=sizing_input.as_of_index,
        capital=capital,
        reference_price=price,
        requested_position_pct=requested_pct,
        capped_position_pct=capped_pct,
        actual_position_pct=actual_pct,
        quantity=quantity,
        target_notional=target_notional,
        actual_notional=actual_notional,
        risk_budget=risk_budget,
        per_share_risk=per_share_risk,
        warnings=warnings,
    )


def derive_backtest_config(
    base_config: BacktestConfig,
    decision: PositionSizingDecision,
) -> BacktestConfig:
    """Create a validated immutable engine config from a pre-run decision.

    This is the only bridge from Sprint 6 sizing into the existing engine.  It
    preserves every cost and execution setting from ``base_config`` and updates
    only the already-supported ``position_size_pct`` field.  It intentionally
    refuses a zero-share decision rather than silently substituting a size.
    """

    if not isinstance(base_config, BacktestConfig):
        raise SizingError("base_config must be a validated BacktestConfig.")
    if not isinstance(decision, PositionSizingDecision):
        raise SizingError("decision must be a PositionSizingDecision.")
    if decision.quantity <= 0 or decision.actual_position_pct <= 0.0:
        raise SizingError("Cannot derive BacktestConfig from a zero-share sizing decision.")
    return BacktestConfig(
        initial_capital=base_config.initial_capital,
        commission_pct=base_config.commission_pct,
        slippage_pct=base_config.slippage_pct,
        position_size_pct=decision.actual_position_pct,
        maximum_holding_days=base_config.maximum_holding_days,
    )


def derive_sizing_input_from_ohlcv(
    data: "DataFrame",
    *,
    as_of_index: int,
    capital: float,
    atr_lookback: int = 14,
    volatility_lookback: int = 20,
) -> SizingInput:
    """Compute causal ATR and realised volatility from a bounded history.

    Only rows ``0`` through ``as_of_index`` (inclusive) are read.  The helper
    requires completed lookback windows and raises instead of forward/back
    filling missing values.  Volatility is annualised from daily simple returns
    using ``sqrt(252)``; callers using a different bar frequency must supply
    their own explicitly defined scalar ``SizingInput``.
    """

    if isinstance(as_of_index, bool) or not isinstance(as_of_index, int) or as_of_index < 0:
        raise InsufficientRiskDataError("as_of_index must be a non-negative integer.")
    if (
        isinstance(atr_lookback, bool)
        or not isinstance(atr_lookback, int)
        or atr_lookback <= 1
    ):
        raise InsufficientRiskDataError("atr_lookback must be an integer greater than one.")
    if (
        isinstance(volatility_lookback, bool)
        or not isinstance(volatility_lookback, int)
        or volatility_lookback <= 1
    ):
        raise InsufficientRiskDataError(
            "volatility_lookback must be an integer greater than one."
        )
    if not isinstance(data, pd.DataFrame):
        raise InsufficientRiskDataError("data must be a pandas DataFrame with High, Low, Close.")
    required_columns = ("High", "Low", "Close")
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise InsufficientRiskDataError(
            f"OHLCV data is missing required sizing columns: {', '.join(missing)}."
        )
    if as_of_index >= len(data):
        raise InsufficientRiskDataError("as_of_index exceeds available OHLCV history.")

    # This is the temporal boundary: no later row can influence any derived
    # input, even if the caller retains a larger DataFrame in memory.
    history = data.iloc[: as_of_index + 1]
    required_observations = max(atr_lookback, volatility_lookback + 1)
    if len(history) < required_observations:
        raise InsufficientRiskDataError(
            "Insufficient completed history for requested ATR/volatility lookbacks."
        )

    series: dict[str, pd.Series] = {}
    for column in required_columns:
        source = history[column]
        if pd.api.types.is_bool_dtype(source):
            raise InsufficientRiskDataError(f"OHLCV column {column} must not be boolean.")
        numeric = pd.to_numeric(source, errors="coerce").astype(float)
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
            raise InsufficientRiskDataError(
                f"OHLCV column {column} contains missing or non-finite values in the causal window."
            )
        series[column] = numeric

    high, low, close = series["High"], series["Low"], series["Close"]
    if (high < low).any() or (close <= 0.0).any():
        raise InsufficientRiskDataError("Causal OHLCV window has invalid High/Low/Close values.")

    previous_close = close.shift(1)
    true_range = pd.concat(
        (high - low, (high - previous_close).abs(), (low - previous_close).abs()),
        axis=1,
    ).max(axis=1)
    atr = float(true_range.iloc[-atr_lookback:].mean())
    returns = close.pct_change().iloc[1:]
    recent_returns = returns.iloc[-volatility_lookback:]
    realised_volatility_pct = float(recent_returns.std(ddof=1) * math.sqrt(252) * 100.0)
    if not math.isfinite(atr) or atr <= 0.0:
        raise InsufficientRiskDataError("Causal ATR is undefined or non-positive.")
    if not math.isfinite(realised_volatility_pct) or realised_volatility_pct < 0.0:
        raise InsufficientRiskDataError("Causal realised volatility is invalid.")

    return SizingInput(
        as_of_index=as_of_index,
        capital=capital,
        entry_price=float(close.iloc[-1]),
        atr=atr,
        realised_volatility_pct=realised_volatility_pct,
    )
