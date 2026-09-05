"""
strategies/validator.py
=======================
Post-schema validation for :class:`~strategies.schema.StrategySchema`.

Pydantic enforces the *structural* contract (correct types, required fields,
registry membership for indicators and operators).  This module adds
*semantic* validation that Pydantic alone cannot express:

* Contradiction detection within ``ALL`` condition groups.
* Duplicate condition warnings.
* Crossover value sanity (belt-and-suspenders, Pydantic already checks this).
* Stop-loss and take-profit value sanity.
* Holding period sanity.

Usage
-----
::

    from strategies.schema import StrategySchema
    from strategies.validator import StrategyValidator, ValidationResult

    raw = {...}  # dict from LLM or user input
    strategy = StrategySchema.model_validate(raw)  # Pydantic validation first

    validator = StrategyValidator()
    result = validator.validate(strategy)

    if result.is_valid:
        print("Ready to compile and backtest.")
    else:
        for err in result.errors:
            print("ERROR:", err)
        for warn in result.warnings:
            print("WARN:", warn)

The validator also accepts a plain ``dict`` via :meth:`~StrategyValidator.validate_dict`,
which runs Pydantic parsing *then* semantic validation in one step.
"""

from __future__ import annotations

import json
from typing import Union

from pydantic import BaseModel, Field

from strategies.registry import (
    CROSSOVER_OPERATORS,
    COMPARISON_OPERATORS,
    is_crossover_operator,
)
from strategies.schema import (
    Condition,
    ConditionGroup,
    EntryRules,
    ExitRules,
    StrategySchema,
    StopLoss,
    TakeProfit,
)

# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """The outcome of running :class:`StrategyValidator` on a strategy.

    Attributes
    ----------
    is_valid:
        ``True`` when no errors were found.  Warnings do not affect validity.
    errors:
        A list of human-readable error messages.  The strategy must NOT be
        compiled or backtested when this list is non-empty.
    warnings:
        A list of human-readable warning messages.  These indicate suspicious
        but not necessarily invalid strategy design choices.
    """

    is_valid: bool = Field(..., description="True when no errors were found.")
    errors: list[str] = Field(
        default_factory=list,
        description="Fatal validation errors. Strategy must not proceed.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings. Strategy may still proceed.",
    )

    def __str__(self) -> str:
        lines = [f"ValidationResult(is_valid={self.is_valid})"]
        for e in self.errors:
            lines.append(f"  ERROR:   {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _condition_key(cond: Condition) -> str:
    """Return a stable string key for a condition, used for duplicate detection."""
    return f"{cond.indicator}|{cond.operator}|{cond.value}"


def _is_numeric(value: object) -> bool:
    """Return True if value is an int or float (not a string indicator name)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class _ConditionContradict:
    """Stateful helper that detects obvious contradictions inside one ALL group.

    The only contradictions detected are:
    * Same indicator, operators ``>`` and ``<``, where the ``>`` threshold
      is >= the ``<`` threshold (e.g. RSI > 70 AND RSI < 30).
    * Same indicator, operators ``>``/``>=`` and ``<``/``<=``, where the
      numeric ranges are mutually exclusive.
    * Same indicator, identical operator but opposite sides
      (e.g. close > ema_20 AND close < ema_20).

    No symbolic math is attempted — only obvious, direct contradictions.
    """

    def __init__(self) -> None:
        # Maps indicator → list of (operator, value, is_strict) triples
        # is_strict = True for strict operators (> or <); False for >= or <=
        self._lower: dict[str, list[tuple[float, bool]]] = {}  # > or >=
        self._upper: dict[str, list[tuple[float, bool]]] = {}  # < or <=
        self._indicator_lower: dict[str, list[str]] = {}  # > or >= against indicator
        self._indicator_upper: dict[str, list[str]] = {}  # < or <= against indicator

    def add(self, cond: Condition) -> None:
        indicator = cond.indicator
        op = cond.operator
        val = cond.value
        if op in (">", ">="):
            is_strict = op == ">"
            if _is_numeric(val):
                self._lower.setdefault(indicator, []).append((float(val), is_strict))
            elif isinstance(val, str):
                self._indicator_lower.setdefault(indicator, []).append(val)
        elif op in ("<", "<="):
            is_strict = op == "<"
            if _is_numeric(val):
                self._upper.setdefault(indicator, []).append((float(val), is_strict))
            elif isinstance(val, str):
                self._indicator_upper.setdefault(indicator, []).append(val)

    def contradictions(self) -> list[str]:
        """Return a list of contradiction error messages (empty if none).

        A numeric contradiction exists when no value x can satisfy all bounds
        simultaneously:
        - x > a AND x < b  (a >= b)  → impossible for any x
        - x >= a AND x < a → impossible (nothing between a and a exclusive)
        - x > a AND x <= a → impossible (nothing between a and a exclusive)
        - x >= a AND x <= a → VALID (x = a satisfies both); NOT flagged
        """
        errors: list[str] = []

        all_indicators = set(self._lower) | set(self._upper) | set(
            self._indicator_lower) | set(self._indicator_upper)

        for indicator in all_indicators:
            lowers = self._lower.get(indicator, [])
            uppers = self._upper.get(indicator, [])

            if lowers and uppers:
                # Check each (lower_value, lower_strict) against each (upper_value, upper_strict)
                for lo_val, lo_strict in lowers:
                    for hi_val, hi_strict in uppers:
                        impossible = False
                        if lo_val > hi_val:
                            # e.g. x > 70 AND x < 30 — clearly impossible
                            impossible = True
                        elif lo_val == hi_val:
                            # e.g. x > 50 AND x < 50 → impossible
                            # e.g. x > 50 AND x <= 50 → impossible
                            # e.g. x >= 50 AND x < 50 → impossible
                            # e.g. x >= 50 AND x <= 50 → VALID (x=50)
                            if lo_strict or hi_strict:
                                impossible = True
                        if impossible:
                            lo_op = ">" if lo_strict else ">="
                            hi_op = "<" if hi_strict else "<="
                            errors.append(
                                f"Contradictory numeric conditions on '{indicator}': "
                                f"condition requires {indicator} {lo_op} {lo_val} "
                                f"AND {indicator} {hi_op} {hi_val}, "
                                f"which can never both be true simultaneously."
                            )
                            break  # one error per indicator is enough
                    else:
                        continue
                    break

            # Check indicator contradictions: same indicator on both sides against same reference
            lower_inds = set(self._indicator_lower.get(indicator, []))
            upper_inds = set(self._indicator_upper.get(indicator, []))
            common_ref = lower_inds & upper_inds
            for ref in common_ref:
                errors.append(
                    f"Contradictory conditions on '{indicator}': "
                    f"condition requires {indicator} > {ref} "
                    f"AND {indicator} < {ref} simultaneously."
                )

        return errors


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class StrategyValidator:
    """Semantic validator for :class:`~strategies.schema.StrategySchema`.

    This class performs validation checks that go *beyond* what Pydantic's
    field and model validators can express.  It should be run **after**
    Pydantic has already parsed and structurally validated the strategy.

    Methods
    -------
    validate(strategy)
        Validate a :class:`StrategySchema` instance.
    validate_dict(data)
        Parse a raw dict into :class:`StrategySchema`, then validate.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, strategy: StrategySchema) -> ValidationResult:
        """Run all semantic validation rules on a parsed strategy.

        Parameters
        ----------
        strategy:
            A fully parsed :class:`~strategies.schema.StrategySchema` instance.
            Pydantic structural validation has already passed at this point.

        Returns
        -------
        ValidationResult
            Contains ``is_valid``, ``errors``, and ``warnings``.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Rule 5 + 6: Contradictions and duplicates in entry conditions
        self._validate_condition_group(
            strategy.entry.long_conditions,
            context="entry.long_conditions",
            errors=errors,
            warnings=warnings,
        )

        # Rule 5 + 6: Contradictions and duplicates in exit conditions
        self._validate_condition_group(
            strategy.exit.exit_conditions,
            context="exit.exit_conditions",
            errors=errors,
            warnings=warnings,
        )

        # Rule 7: Stop loss sanity (belt-and-suspenders beyond Pydantic)
        self._validate_stop_loss(strategy.stop_loss, errors=errors)

        # Rule 8: Take profit sanity
        self._validate_take_profit(strategy.take_profit, errors=errors)

        # Rule 9: Holding period sanity
        self._validate_holding_period(strategy.exit, errors=errors)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_dict(
        self, data: Union[dict, str]
    ) -> ValidationResult:
        """Parse a raw dictionary (or JSON string) into a StrategySchema, then validate.

        This is a convenience method that combines Pydantic parsing and
        semantic validation in one call.

        Parameters
        ----------
        data:
            Either a plain Python ``dict`` or a JSON-encoded ``str``.

        Returns
        -------
        ValidationResult
            If Pydantic parsing fails, ``is_valid`` will be ``False`` and
            ``errors`` will contain the Pydantic validation messages.
        """
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as exc:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Invalid JSON: {exc}"],
                )

        try:
            strategy = StrategySchema.model_validate(data)
        except Exception as exc:  # Pydantic ValidationError
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema validation failed: {exc}"],
            )

        return self.validate(strategy)

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    def _validate_condition_group(
        self,
        group: ConditionGroup,
        context: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Validate a ConditionGroup and every nested child group."""
        self._check_duplicates(group, context, warnings)
        if group.logic == "ALL":
            # Contradictions are only meaningful in an AND group.
            self._check_contradictions(group, context, errors)
        for index, node in enumerate(group.conditions):
            if isinstance(node, ConditionGroup):
                self._validate_condition_group(
                    node,
                    context=f"{context}.conditions[{index}]",
                    errors=errors,
                    warnings=warnings,
                )

    def _check_duplicates(
        self,
        group: ConditionGroup,
        context: str,
        warnings: list[str],
    ) -> None:
        """Warn when the same condition appears more than once in a group."""
        seen_keys: dict[str, int] = {}
        for i, cond in enumerate(group.conditions):
            if isinstance(cond, ConditionGroup):
                continue
            key = _condition_key(cond)
            if key in seen_keys:
                warnings.append(
                    f"[{context}] Duplicate condition at index {i} "
                    f"(same as index {seen_keys[key]}): "
                    f"{cond.indicator} {cond.operator} {cond.value!r}. "
                    f"This condition has no additional effect."
                )
            else:
                seen_keys[key] = i

    def _check_contradictions(
        self,
        group: ConditionGroup,
        context: str,
        errors: list[str],
    ) -> None:
        """Detect obvious logical contradictions in an ALL (AND) group."""
        checker = _ConditionContradict()
        for cond in group.conditions:
            if isinstance(cond, Condition):
                checker.add(cond)
        for msg in checker.contradictions():
            errors.append(f"[{context}] {msg}")

    def _validate_stop_loss(
        self, stop_loss: StopLoss, errors: list[str]
    ) -> None:
        """Belt-and-suspenders stop loss value checks (Pydantic also validates)."""
        if stop_loss.value <= 0:
            errors.append(
                f"StopLoss value must be > 0. Got: {stop_loss.value}."
            )

    def _validate_take_profit(
        self, take_profit: TakeProfit, errors: list[str]
    ) -> None:
        """Belt-and-suspenders take profit value checks."""
        if take_profit.value <= 0:
            errors.append(
                f"TakeProfit value must be > 0. Got: {take_profit.value}."
            )

    def _validate_holding_period(
        self, exit_rules: ExitRules, errors: list[str]
    ) -> None:
        """Validate maximum_holding_days when present."""
        if (
            exit_rules.maximum_holding_days is not None
            and exit_rules.maximum_holding_days <= 0
        ):
            errors.append(
                f"ExitRules.maximum_holding_days must be > 0. "
                f"Got: {exit_rules.maximum_holding_days}."
            )
