"""Immutable, validated strategy-version lineage for Sprint 6 refinement.

This module deliberately has no graph, agent, compiler, or backtest imports.
It records *only* a canonical :class:`StrategySchema` that already crossed the
Sprint 1 validation boundary.  A caller must still send the record's strategy
through the deterministic compiler and backtester before it is evaluated.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evaluation.research_metrics import strategy_execution_fingerprint
from strategies.schema import StrategySchema
from strategies.validator import StrategyValidator
from tradingagents.research.exceptions import (
    ChildStrategyValidationError,
    InvalidLineageTransitionError,
)


class _FrozenStrictModel(BaseModel):
    """Reject coercion and mutation at the strategy-version boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")


def _non_blank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty.")
    return value


class StrategyLineage(_FrozenStrictModel):
    """Immutable identity and ancestry for exactly one strategy version.

    ``strategy_id`` identifies this version, not a mutable logical strategy.
    A root version has generation zero and no parent.  Every refinement has a
    distinct identifier, points to a parent version, and advances generation.
    """

    strategy_id: str = Field(min_length=1, max_length=128)
    parent_strategy_id: str | None = Field(default=None, min_length=1, max_length=128)
    generation: int = Field(ge=0, le=10_000)
    created_at: datetime
    revision_reason: str = Field(min_length=1, max_length=1_000)
    execution_fingerprint: str = Field(min_length=64, max_length=64)

    @field_validator("strategy_id", "parent_strategy_id", "revision_reason")
    @classmethod
    def identifiers_and_reason_must_not_be_blank(
        cls, value: str | None, info
    ) -> str | None:
        if value is None:
            return value
        return _non_blank(value, info.field_name)

    @field_validator("generation", mode="before")
    @classmethod
    def generation_must_not_be_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("generation must be an integer, not a boolean.")
        return value

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("created_at must be timezone-aware UTC.")
        return value.astimezone(timezone.utc)

    @field_validator("execution_fingerprint")
    @classmethod
    def execution_fingerprint_must_be_sha256(cls, value: str) -> str:
        value = value.lower()
        if not _FINGERPRINT_PATTERN.fullmatch(value):
            raise ValueError("execution_fingerprint must be a lowercase SHA-256 hex digest.")
        return value

    @model_validator(mode="after")
    def ancestry_must_match_generation(self) -> "StrategyLineage":
        if self.generation == 0 and self.parent_strategy_id is not None:
            raise ValueError("A generation-zero strategy cannot have a parent_strategy_id.")
        if self.generation > 0 and self.parent_strategy_id is None:
            raise ValueError("A refined strategy must declare a parent_strategy_id.")
        if self.parent_strategy_id == self.strategy_id:
            raise ValueError("strategy_id and parent_strategy_id must be different.")
        return self


class StrategyVersionRecord(_FrozenStrictModel):
    """A non-overwritable strategy schema paired with its immutable lineage."""

    lineage: StrategyLineage
    strategy: StrategySchema

    @model_validator(mode="after")
    def strategy_must_match_lineage_and_remain_semantically_valid(
        self,
    ) -> "StrategyVersionRecord":
        actual_fingerprint = strategy_execution_fingerprint(self.strategy)
        if actual_fingerprint != self.lineage.execution_fingerprint:
            raise ValueError(
                "Strategy lineage execution_fingerprint does not match the canonical strategy."
            )
        validation = StrategyValidator().validate(self.strategy)
        if not validation.is_valid:
            raise ValueError(
                "StrategyVersionRecord requires a semantically valid StrategySchema."
            )
        return self

    @property
    def strategy_id(self) -> str:
        """Convenience identity for consumers that do not need the full lineage."""

        return self.lineage.strategy_id


def derive_strategy_id(
    *,
    parent_strategy_id: str | None,
    generation: int,
    execution_fingerprint: str,
    revision_reason: str,
) -> str:
    """Derive a stable version ID from immutable lineage facts.

    This intentionally does not use process randomness.  Replaying the same
    canonical refinement produces the same version identity, while changing a
    parent, generation, executable rules, or reason produces a separate ID.
    Persistence can therefore make replay idempotent without ever overwriting
    a parent strategy.
    """

    if generation < 0 or isinstance(generation, bool):
        raise InvalidLineageTransitionError("generation must be a non-negative integer.")
    if parent_strategy_id is not None:
        _non_blank(parent_strategy_id, "parent_strategy_id")
    _non_blank(revision_reason, "revision_reason")
    if not _FINGERPRINT_PATTERN.fullmatch(execution_fingerprint.lower()):
        raise InvalidLineageTransitionError("execution_fingerprint must be a SHA-256 digest.")
    payload = {
        "parent_strategy_id": parent_strategy_id,
        "generation": generation,
        "execution_fingerprint": execution_fingerprint.lower(),
        "revision_reason": revision_reason.strip(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"strategy-v{generation}-{digest[:24]}"


def create_initial_strategy_record(
    strategy: StrategySchema,
    *,
    revision_reason: str = "Initial generated strategy.",
    created_at: datetime | None = None,
    strategy_id: str | None = None,
) -> StrategyVersionRecord:
    """Create an immutable generation-zero record from a parsed strategy.

    Root callers provide a parsed schema, so semantic validation is repeated
    here rather than trusting construction history.  Raw LLM payloads belong
    in :func:`promote_refinement_child`, which has the stricter canonical raw
    validation sequence.
    """

    if not isinstance(strategy, StrategySchema):
        raise InvalidLineageTransitionError("strategy must be a StrategySchema instance.")
    validation = StrategyValidator().validate(strategy)
    if not validation.is_valid:
        raise ChildStrategyValidationError(tuple(validation.errors))
    fingerprint = strategy_execution_fingerprint(strategy)
    resolved_id = strategy_id or derive_strategy_id(
        parent_strategy_id=None,
        generation=0,
        execution_fingerprint=fingerprint,
        revision_reason=revision_reason,
    )
    return StrategyVersionRecord(
        lineage=StrategyLineage(
            strategy_id=resolved_id,
            parent_strategy_id=None,
            generation=0,
            created_at=created_at or datetime.now(timezone.utc),
            revision_reason=revision_reason,
            execution_fingerprint=fingerprint,
        ),
        strategy=strategy,
    )


def validate_raw_refinement_child(raw_strategy: Mapping[str, Any] | str) -> StrategySchema:
    """Cross a raw refinement proposal through Sprint 1's canonical boundary.

    ``StrategyValidator.validate_dict`` runs first.  Only after it reports a
    valid structural and semantic result do we materialize ``StrategySchema``
    with ``model_validate``.  No compiler, evaluator, or LLM-generated code
    participates in this operation.
    """

    raw_payload: object = raw_strategy
    if isinstance(raw_strategy, str):
        try:
            raw_payload = json.loads(raw_strategy)
        except json.JSONDecodeError as exc:
            raise ChildStrategyValidationError(("Invalid JSON refinement proposal.",)) from exc
    if not isinstance(raw_payload, Mapping):
        raise ChildStrategyValidationError(
            ("Refinement proposal must be a JSON object or mapping.",)
        )

    # Take a shallow mapping copy so a caller cannot mutate its outer mapping
    # during validation and schema materialization.
    canonical_payload = dict(raw_payload)
    validation = StrategyValidator().validate_dict(canonical_payload)
    if not validation.is_valid:
        raise ChildStrategyValidationError(tuple(validation.errors))
    try:
        return StrategySchema.model_validate(canonical_payload)
    except Exception as exc:  # Defensive: validate_dict should already catch this.
        raise ChildStrategyValidationError(("Schema materialization failed.",)) from exc


def promote_refinement_child(
    parent: StrategyVersionRecord,
    raw_strategy: Mapping[str, Any] | str,
    *,
    revision_reason: str,
    created_at: datetime | None = None,
) -> StrategyVersionRecord:
    """Create a fresh immutable child only after canonical validation succeeds.

    The parent is never changed.  The returned child receives generation
    ``parent.generation + 1`` and a new deterministic strategy ID derived from
    the parent and the child's executable semantics.
    """

    if not isinstance(parent, StrategyVersionRecord):
        raise InvalidLineageTransitionError("parent must be a StrategyVersionRecord.")
    _non_blank(revision_reason, "revision_reason")
    child_strategy = validate_raw_refinement_child(raw_strategy)
    child_fingerprint = strategy_execution_fingerprint(child_strategy)
    child_generation = parent.lineage.generation + 1
    child_id = derive_strategy_id(
        parent_strategy_id=parent.strategy_id,
        generation=child_generation,
        execution_fingerprint=child_fingerprint,
        revision_reason=revision_reason,
    )
    return StrategyVersionRecord(
        lineage=StrategyLineage(
            strategy_id=child_id,
            parent_strategy_id=parent.strategy_id,
            generation=child_generation,
            created_at=created_at or datetime.now(timezone.utc),
            revision_reason=revision_reason,
            execution_fingerprint=child_fingerprint,
        ),
        strategy=child_strategy,
    )


# A readable alias for workflow nodes.  It has exactly the same canonical
# validation semantics as ``promote_refinement_child``.
canonical_promote_refinement = promote_refinement_child

# Short lifecycle-oriented aliases used by orchestration code.  They preserve
# the canonical validation path rather than introducing a second constructor.
create_lineage = create_initial_strategy_record
promote_refinement_strategy = promote_refinement_child
