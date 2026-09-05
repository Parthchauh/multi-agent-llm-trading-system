"""Chronological, capability-scoped datasets for historical research.

The public views deliberately omit the final holdout. A holdout frame can
only be released with a :class:`FrozenStrategySelection`, representing the
point at which generation, refinement, and selection have ended.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd

from data.schema import CANONICAL_OHLCV_COLUMNS, validate_ohlcv


class ResearchSplitError(ValueError):
    """Raised when research partitions violate temporal isolation."""


class ResearchPartitionRole(str, Enum):
    CONTEXT = "CONTEXT"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    FINAL_HOLDOUT = "FINAL_HOLDOUT"


@dataclass(frozen=True)
class DatasetFingerprint:
    """Immutable identity and boundary metadata for one partition."""

    partition_id: str
    role: ResearchPartitionRole
    start: datetime
    end: datetime
    row_count: int
    timezone: str | None
    data_sha256: str
    columns: tuple[str, ...] = CANONICAL_OHLCV_COLUMNS


def _fingerprint(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


class ResearchPartition:
    """Validated observations plus optional non-scoring feature warm-up.

    Frames are copied on construction and on every public read so callers
    cannot mutate a validated input after its fingerprint is made.
    """

    __slots__ = ("_fingerprint", "_observations", "_warmup")

    def __init__(
        self,
        *,
        partition_id: str,
        role: ResearchPartitionRole,
        observations: pd.DataFrame,
        warmup: pd.DataFrame | None = None,
    ) -> None:
        if not partition_id or not partition_id.strip():
            raise ResearchSplitError("partition_id must be non-empty.")
        clean = validate_ohlcv(observations)
        clean_warmup = None if warmup is None else validate_ohlcv(warmup)
        if clean_warmup is not None and clean_warmup.index[-1] >= clean.index[0]:
            raise ResearchSplitError(
                "Feature warm-up must end strictly before scored observations begin."
            )
        self._observations = clean.copy(deep=True)
        self._warmup = None if clean_warmup is None else clean_warmup.copy(deep=True)
        tz = clean.index.tz
        self._fingerprint = DatasetFingerprint(
            partition_id=partition_id.strip(),
            role=role,
            start=clean.index[0].to_pydatetime(),
            end=clean.index[-1].to_pydatetime(),
            row_count=len(clean),
            timezone=None if tz is None else str(tz),
            data_sha256=_fingerprint(clean.loc[:, list(CANONICAL_OHLCV_COLUMNS)]),
        )

    @property
    def fingerprint(self) -> DatasetFingerprint:
        return self._fingerprint

    @property
    def observations(self) -> pd.DataFrame:
        """Return scored observations as a defensive copy."""
        return self._observations.copy(deep=True)

    @property
    def feature_frame(self) -> pd.DataFrame:
        """Return warm-up plus observations for causal feature computation."""
        if self._warmup is None:
            return self._observations.copy(deep=True)
        return pd.concat([self._warmup, self._observations]).copy(deep=True)

    @property
    def evaluation_start(self) -> pd.Timestamp:
        return self._observations.index[0]


@dataclass(frozen=True)
class ResearchGenerationView:
    """The only dataset capability passed to strategy-generating agents."""

    context: ResearchPartition
    decision_timestamp: datetime


@dataclass(frozen=True)
class ResearchDevelopmentView:
    """Non-holdout partitions available to deterministic development stages."""

    context: ResearchPartition
    validation: ResearchPartition
    test: ResearchPartition
    decision_timestamp: datetime


@dataclass(frozen=True)
class FrozenStrategySelection:
    """Explicit evidence that strategy identity and parameters are frozen."""

    strategy_id: str
    strategy_hash: str
    frozen_at: datetime

    def __post_init__(self) -> None:
        if not self.strategy_id.strip() or not self.strategy_hash.strip():
            raise ResearchSplitError(
                "Frozen selection requires non-empty strategy_id and strategy_hash."
            )


@dataclass(frozen=True)
class FinalHoldoutEvaluationView:
    """A holdout capability released only after strategy selection freezes."""

    selection: FrozenStrategySelection
    final_holdout: ResearchPartition


class ResearchDatasetSplit:
    """Four strictly ordered research partitions with a sealed final holdout."""

    __slots__ = (
        "_context",
        "_validation",
        "_test",
        "__final_holdout",
        "_decision_timestamp",
    )

    def __init__(
        self,
        *,
        context: ResearchPartition,
        validation: ResearchPartition,
        test: ResearchPartition,
        final_holdout: ResearchPartition,
        decision_timestamp: datetime | None = None,
    ) -> None:
        expected = (
            (context, ResearchPartitionRole.CONTEXT),
            (validation, ResearchPartitionRole.VALIDATION),
            (test, ResearchPartitionRole.TEST),
            (final_holdout, ResearchPartitionRole.FINAL_HOLDOUT),
        )
        for partition, role in expected:
            if partition.fingerprint.role is not role:
                raise ResearchSplitError(
                    f"Expected {role.value} partition, got {partition.fingerprint.role.value}."
                )
        ordered = (context, validation, test, final_holdout)
        for earlier, later in zip(ordered, ordered[1:]):
            if earlier.fingerprint.end >= later.fingerprint.start:
                raise ResearchSplitError(
                    f"{earlier.fingerprint.role.value}.end must be strictly before "
                    f"{later.fingerprint.role.value}.start."
                )
        decision = decision_timestamp or context.fingerprint.end
        decision_ts = pd.Timestamp(decision)
        context_end = pd.Timestamp(context.fingerprint.end)
        validation_start = pd.Timestamp(validation.fingerprint.start)
        if decision_ts < context_end or decision_ts >= validation_start:
            raise ResearchSplitError(
                "decision_timestamp must be at/after context end and before validation start."
            )
        self._context = context
        self._validation = validation
        self._test = test
        self.__final_holdout = final_holdout
        self._decision_timestamp = decision

    def generation_view(self) -> ResearchGenerationView:
        return ResearchGenerationView(self._context, self._decision_timestamp)

    def development_view(self) -> ResearchDevelopmentView:
        return ResearchDevelopmentView(
            self._context, self._validation, self._test, self._decision_timestamp
        )

    def release_final_holdout(
        self, selection: FrozenStrategySelection
    ) -> FinalHoldoutEvaluationView:
        if not isinstance(selection, FrozenStrategySelection):
            raise ResearchSplitError(
                "Final holdout access requires a FrozenStrategySelection."
            )
        return FinalHoldoutEvaluationView(selection, self.__final_holdout)

    @property
    def partition_fingerprints(self) -> tuple[DatasetFingerprint, ...]:
        """Expose provenance without exposing holdout observations."""
        return (
            self._context.fingerprint,
            self._validation.fingerprint,
            self._test.fingerprint,
            self.__final_holdout.fingerprint,
        )
