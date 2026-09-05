"""
strategies/schema.py — Pydantic v2 strategy data models.
strategies/schema.py
====================
Pydantic v2 models that define the complete, declarative structure of a
trading strategy.

Design principles
-----------------
* **No executable code.**  Every field is a plain data value (string, number,
  enum).  There is no Python expression, no ``eval()``, no ``exec()``.
* **LLM-safe.**  An LLM can generate a valid strategy by producing a JSON
  object that matches these models.  The JSON is validated by Pydantic before
  any further processing happens.
* **Registry-integrated.**  Allowed indicator names and operator symbols are
  imported from :mod:`strategies.registry` so that schema and registry stay
  in sync automatically.
* **Sprint-scoped.**  Sprint 1 intentionally restricts market types to
  ``"equity"`` and timeframes to ``"1d"``.  Future sprints can widen the
  ``Literal`` unions without breaking existing strategies.

All models use Pydantic v2 (``model_config``, ``model_validator``,
``field_validator``).
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strategies.registry import (
    SUPPORTED_INDICATORS,
    SUPPORTED_OPERATORS,
    CROSSOVER_OPERATORS,
)

# ---------------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------------

# The right-hand side of a condition can be either a numeric value or the
# name of another supported indicator.
ConditionValue = Union[float, str]

# ---------------------------------------------------------------------------
# Condition models
# ---------------------------------------------------------------------------


class Condition(BaseModel):
    """A single, atomic trading condition.

    A condition compares an *indicator* to a *value* using an *operator*.
    The value may be a numeric literal or the name of a second indicator,
    depending on the operator used.

    Attributes
    ----------
    indicator:
        The left-hand-side indicator.  Must be present in
        :data:`strategies.registry.SUPPORTED_INDICATORS`.
    operator:
        The comparison or crossover operator.  Must be present in
        :data:`strategies.registry.SUPPORTED_OPERATORS`.
    value:
        The right-hand-side operand.  For crossover operators this must be
        another indicator name (``str``); for comparison operators this may
        be either a ``float`` or an indicator name.

    Examples
    --------
    Numeric comparison::

        Condition(indicator="rsi_14", operator=">", value=55)

    Indicator-vs-indicator comparison::

        Condition(indicator="close", operator=">", value="ema_20")

    Crossover::

        Condition(indicator="ema_20", operator="crosses_above", value="ema_50")
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    indicator: str = Field(
        ...,
        description="Left-hand indicator name (must exist in SUPPORTED_INDICATORS).",
    )
    operator: str = Field(
        ...,
        description="Comparison or crossover operator.",
    )
    value: ConditionValue = Field(
        ...,
        description=(
            "Right-hand operand — a numeric value or another indicator name. "
            "Crossover operators require an indicator name."
        ),
    )

    @field_validator("value", mode="before")
    @classmethod
    def condition_value_must_not_be_boolean(cls, value: object) -> object:
        """Reject booleans before Pydantic can coerce ``True`` to ``1.0``."""
        if isinstance(value, bool):
            raise ValueError("Condition value must be a number or supported indicator, not a boolean.")
        return value

    @field_validator("indicator")
    @classmethod
    def indicator_must_be_supported(cls, v: str) -> str:
        """Reject indicator names that are not in the registry."""
        if v not in SUPPORTED_INDICATORS:
            raise ValueError(
                f"Unsupported indicator '{v}'. "
                f"Supported indicators: {sorted(SUPPORTED_INDICATORS)}"
            )
        return v

    @field_validator("operator")
    @classmethod
    def operator_must_be_supported(cls, v: str) -> str:
        """Reject operator symbols that are not in the registry."""
        if v not in SUPPORTED_OPERATORS:
            raise ValueError(
                f"Unsupported operator '{v}'. "
                f"Supported operators: {sorted(SUPPORTED_OPERATORS)}"
            )
        return v

    @model_validator(mode="after")
    def crossover_value_must_be_indicator(self) -> "Condition":
        """Crossover operators require the value to be an indicator name, not a number."""
        if self.operator in CROSSOVER_OPERATORS:
            if not isinstance(self.value, str):
                raise ValueError(
                    f"Operator '{self.operator}' requires the value to be an "
                    f"indicator name (str), not a numeric value. "
                    f"Got: {self.value!r}."
                )
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("Condition numeric value must be a finite number.")
        if isinstance(self.value, str) and self.value not in SUPPORTED_INDICATORS:
            raise ValueError(
                f"Condition value '{self.value}' is not recognised as a supported indicator. "
                f"Supported indicators: {sorted(SUPPORTED_INDICATORS)}"
            )
        return self


class ConditionGroup(BaseModel):
    """A logical group of conditions combined with AND (``ALL``) or OR (``ANY``).

    Attributes
    ----------
    logic:
        ``"ALL"`` means every condition must be true (logical AND).
        ``"ANY"`` means at least one condition must be true (logical OR).
    conditions:
        An ordered list of :class:`Condition` instances.  At least one
        condition is required.

    Examples
    --------
    All conditions must hold::

        ConditionGroup(
            logic="ALL",
            conditions=[
                Condition(indicator="rsi_14", operator=">", value=55),
                Condition(indicator="close", operator=">", value="ema_20"),
            ],
        )

    Any condition triggers::

        ConditionGroup(
            logic="ANY",
            conditions=[
                Condition(indicator="rsi_14", operator="<", value=30),
                Condition(indicator="close", operator="crosses_below", value="sma_50"),
            ],
        )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    logic: Literal["ALL", "ANY"] = Field(
        ...,
        description="Logical combinator: 'ALL' (AND) or 'ANY' (OR).",
    )
    conditions: tuple[Union[Condition, "ConditionGroup"], ...] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered tuple of atomic conditions or nested condition groups. "
            "At least one item is required. Tuple enforces deep immutability."
        ),
    )


# ---------------------------------------------------------------------------
# Entry / Exit rules
# ---------------------------------------------------------------------------


class EntryRules(BaseModel):
    """Entry rules for a long-only strategy.

    Attributes
    ----------
    long_conditions:
        A :class:`ConditionGroup` that, when satisfied, triggers a BUY signal.

    Notes
    -----
    Short-selling is not supported in Sprint 1.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    long_conditions: ConditionGroup = Field(
        ...,
        description="Conditions required to enter a long position.",
    )


class ExitRules(BaseModel):
    """Exit rules for a long-only strategy.

    Attributes
    ----------
    exit_conditions:
        A :class:`ConditionGroup` that, when satisfied, triggers a SELL signal.
    maximum_holding_days:
        Optional.  If supplied, the position is force-closed after this many
        calendar days regardless of whether exit conditions are met.
        Must be a positive integer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    exit_conditions: ConditionGroup = Field(
        ...,
        description="Conditions required to exit a long position.",
    )
    maximum_holding_days: Annotated[int, Field(gt=0)] | None = Field(
        default=None,
        description=(
            "Optional maximum holding period in calendar days. "
            "Position is force-closed after this many days. Must be > 0."
        ),
    )

    @field_validator("maximum_holding_days", mode="before")
    @classmethod
    def holding_days_must_not_be_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("maximum_holding_days must be an integer, not a boolean.")
        return value


# ---------------------------------------------------------------------------
# Risk management models
# ---------------------------------------------------------------------------


class StopLoss(BaseModel):
    """Stop-loss configuration.

    Attributes
    ----------
    type:
        ``"percentage"``  — stop is placed *value* percent below the entry
        price (e.g. ``value=5`` → stop at entry × 0.95).

        ``"atr_multiple"`` — stop is placed *value* × ATR(14) below the
        entry price.
    value:
        The numeric magnitude of the stop.  Must be strictly positive.

    Examples
    --------
    Five-percent stop::

        StopLoss(type="percentage", value=5)

    Two-ATR stop::

        StopLoss(type="atr_multiple", value=2)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["percentage", "atr_multiple"] = Field(
        ...,
        description="Stop-loss method: 'percentage' or 'atr_multiple'.",
    )
    value: float = Field(
        ...,
        description="Stop-loss magnitude.  Must be > 0.",
    )

    @field_validator("value", mode="before")
    @classmethod
    def value_must_not_be_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("StopLoss value must be a number, not a boolean.")
        return value

    @field_validator("value")
    @classmethod
    def value_must_be_positive(cls, v: float) -> float:
        """Reject non-positive and non-finite stop-loss values."""
        if not math.isfinite(v):
            raise ValueError(
                f"StopLoss value must be a finite number. Got: {v}."
            )
        if v <= 0:
            raise ValueError(
                f"StopLoss value must be > 0. Got: {v}."
            )
        return v

    @model_validator(mode="after")
    def percentage_stop_within_bounds(self) -> "StopLoss":
        """For 'percentage' type only: reject values > 100 (would stop out immediately).

        'atr_multiple' values are not bounded by 100 — an ATR of 2.5× is
        perfectly valid and has no logical upper bound relative to entry price.
        """
        if self.type == "percentage" and self.value > 100:
            raise ValueError(
                f"StopLoss percentage value {self.value} exceeds 100 %. "
                "This would stop out immediately. Did you mean a decimal?"
            )
        return self


class TakeProfit(BaseModel):
    """Take-profit configuration.

    Attributes
    ----------
    type:
        ``"percentage"``  — target is placed *value* percent above the entry
        price.

        ``"risk_reward"`` — target is placed at entry + *value* × (entry −
        stop_loss).  Requires a stop loss to be defined on the parent
        strategy.
    value:
        The numeric magnitude of the target.  Must be strictly positive.

    Examples
    --------
    Ten-percent target::

        TakeProfit(type="percentage", value=10)

    Three-to-one risk/reward::

        TakeProfit(type="risk_reward", value=3)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["percentage", "risk_reward"] = Field(
        ...,
        description="Take-profit method: 'percentage' or 'risk_reward'.",
    )
    value: float = Field(
        ...,
        description="Take-profit magnitude.  Must be > 0.",
    )

    @field_validator("value", mode="before")
    @classmethod
    def value_must_not_be_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("TakeProfit value must be a number, not a boolean.")
        return value

    @field_validator("value")
    @classmethod
    def value_must_be_positive(cls, v: float) -> float:
        """Reject non-positive and non-finite take-profit values."""
        if not math.isfinite(v):
            raise ValueError(
                f"TakeProfit value must be a finite number. Got: {v}."
            )
        if v <= 0:
            raise ValueError(
                f"TakeProfit value must be > 0. Got: {v}."
            )
        return v


class PositionSizing(BaseModel):
    """Position sizing configuration.

    Attributes
    ----------
    type:
        ``"fixed_percentage"`` — invest *value* percent of available capital
        per trade.
    value:
        Percentage of capital to allocate per trade.
        Must be in the range ``(0, 100]``.

    Examples
    --------
    Ten percent of capital per trade::

        PositionSizing(type="fixed_percentage", value=10)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["fixed_percentage"] = Field(
        ...,
        description="Position sizing method. Currently only 'fixed_percentage' is supported.",
    )
    value: float = Field(
        ...,
        description="Percentage of capital per trade.  Range: (0, 100].",
    )

    @field_validator("value", mode="before")
    @classmethod
    def value_must_not_be_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("PositionSizing value must be a number, not a boolean.")
        return value

    @field_validator("value")
    @classmethod
    def value_in_valid_range(cls, v: float) -> float:
        """Reject position sizes that are zero, negative, non-finite, or above 100 %."""
        if not math.isfinite(v):
            raise ValueError(
                f"PositionSizing value must be a finite number. Got: {v}."
            )
        if v <= 0:
            raise ValueError(
                f"PositionSizing value must be > 0. Got: {v}."
            )
        if v > 100:
            raise ValueError(
                f"PositionSizing value must be <= 100 (percent). Got: {v}."
            )
        return v


# ---------------------------------------------------------------------------
# Strategy metadata
# ---------------------------------------------------------------------------


class StrategyMetadata(BaseModel):
    """Descriptive metadata attached to a strategy.

    Attributes
    ----------
    name:
        Human-readable strategy name.  Must be non-empty.
    description:
        Short description of the strategy's intent.
    market_type:
        The market class this strategy is designed for.
        Sprint 1 supports ``"equity"`` only.
    timeframe:
        The candle / bar frequency.
        Sprint 1 supports ``"1d"`` (daily) only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable strategy name.",
    )
    description: str = Field(
        default="",
        description="Short description of the strategy's intent.",
    )
    market_type: Literal["equity"] = Field(
        ...,
        description="Market class.  Sprint 1 supports 'equity' only.",
    )
    timeframe: Literal["1d"] = Field(
        ...,
        description="Bar frequency.  Sprint 1 supports '1d' (daily) only.",
    )


# ---------------------------------------------------------------------------
# Top-level strategy schema
# ---------------------------------------------------------------------------


class StrategySchema(BaseModel):
    """The complete, declarative representation of a trading strategy.

    A :class:`StrategySchema` is the *only* object that enters the backtesting
    engine.  It contains no executable code — only structured data that the
    compiler (Sprint 3) converts into deterministic logic.

    Attributes
    ----------
    metadata:
        Descriptive information about the strategy.
    entry:
        Entry rules specifying when to open a long position.
    exit:
        Exit rules specifying when to close the position.
    stop_loss:
        Stop-loss configuration applied to every open position.
    take_profit:
        Take-profit configuration applied to every open position.
    position_sizing:
        Capital allocation rules per trade.

    Examples
    --------
    Minimal valid strategy::

        strategy = StrategySchema(
            metadata=StrategyMetadata(
                name="EMA RSI Momentum",
                description="Long-only momentum strategy",
                market_type="equity",
                timeframe="1d",
            ),
            entry=EntryRules(
                long_conditions=ConditionGroup(
                    logic="ALL",
                    conditions=[
                        Condition(indicator="ema_20", operator="crosses_above", value="ema_50"),
                        Condition(indicator="rsi_14", operator=">", value=55),
                    ],
                )
            ),
            exit=ExitRules(
                exit_conditions=ConditionGroup(
                    logic="ANY",
                    conditions=[
                        Condition(indicator="rsi_14", operator="<", value=45),
                    ],
                ),
                maximum_holding_days=30,
            ),
            stop_loss=StopLoss(type="percentage", value=5),
            take_profit=TakeProfit(type="percentage", value=10),
            position_sizing=PositionSizing(type="fixed_percentage", value=10),
        )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: StrategyMetadata = Field(
        ...,
        description="Descriptive metadata for the strategy.",
    )
    entry: EntryRules = Field(
        ...,
        description="Entry rules for opening a long position.",
    )
    exit: ExitRules = Field(
        ...,
        description="Exit rules for closing the position.",
    )
    stop_loss: StopLoss = Field(
        ...,
        description="Stop-loss configuration.",
    )
    take_profit: TakeProfit = Field(
        ...,
        description="Take-profit configuration.",
    )
    position_sizing: PositionSizing = Field(
        ...,
        description="Position sizing configuration.",
    )
