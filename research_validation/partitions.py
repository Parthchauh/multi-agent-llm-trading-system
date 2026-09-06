"""Capability-scoped chronological partitions for Sprint 7 OOS research.

Unlike the earlier development split, this module never exposes its final test
frame through a development or agent-facing view.  It is deliberately a new
contract: reinterpreting a legacy development ``test`` partition as a final
holdout would create an ambiguous information policy.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from data.research_split import FrozenStrategySelection
from data.schema import CANONICAL_OHLCV_COLUMNS, validate_ohlcv


class OOSPartitionError(ValueError):
    """Raised when chronological or information-boundary rules are violated."""


class PartitionRole(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    FINAL_TEST = "FINAL_TEST"


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class TimeSeriesSplitConfig(_FrozenStrictModel):
    """A complete chronological train/validation/final-test allocation."""

    train_fraction: float = Field(default=0.60, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.20, gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.20, gt=0.0, lt=1.0)
    minimum_partition_rows: int = Field(default=20, ge=1)

    @field_validator("train_fraction", "validation_fraction", "test_fraction", mode="before")
    @classmethod
    def fractions_must_be_finite_non_boolean(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Split fractions must be finite numeric values, not boolean.")
        if not math.isfinite(float(value)):
            raise ValueError("Split fractions must be finite.")
        return float(value)

    @field_validator("minimum_partition_rows", mode="before")
    @classmethod
    def row_count_must_not_be_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("minimum_partition_rows must be an integer, not boolean.")
        return value

    @model_validator(mode="after")
    def fractions_must_sum_to_one(self) -> "TimeSeriesSplitConfig":
        if not math.isclose(
            self.train_fraction + self.validation_fraction + self.test_fraction,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("train_fraction + validation_fraction + test_fraction must equal 1.")
        return self


@dataclass(frozen=True)
class PartitionFingerprint:
    partition_id: str
    role: PartitionRole
    start: datetime
    end: datetime
    row_count: int
    data_sha256: str


class DatasetPartition:
    """Validated immutable partition whose observations are defensive copies."""

    __slots__ = ("_fingerprint", "_observations")

    def __init__(self, *, partition_id: str, role: PartitionRole, observations: pd.DataFrame) -> None:
        if not isinstance(role, PartitionRole):
            raise OOSPartitionError("role must be a PartitionRole.")
        if not isinstance(partition_id, str) or not partition_id.strip():
            raise OOSPartitionError("partition_id must be non-blank.")
        frame = validate_ohlcv(observations)
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(
                frame.loc[:, list(CANONICAL_OHLCV_COLUMNS)], index=True
            ).to_numpy().tobytes()
        ).hexdigest()
        self._observations = frame.copy(deep=True)
        self._fingerprint = PartitionFingerprint(
            partition_id=partition_id.strip(),
            role=role,
            start=frame.index[0].to_pydatetime(),
            end=frame.index[-1].to_pydatetime(),
            row_count=len(frame),
            data_sha256=digest,
        )

    @property
    def fingerprint(self) -> PartitionFingerprint:
        return self._fingerprint

    @property
    def observations(self) -> pd.DataFrame:
        return self._observations.copy(deep=True)


@dataclass(frozen=True)
class AgentResearchAccess:
    """Only this training capability is permitted at generation/refinement time."""

    train: DatasetPartition


@dataclass(frozen=True)
class ValidationResearchAccess:
    """Deterministic validation capability; it deliberately lacks final-test data."""

    train: DatasetPartition
    validation: DatasetPartition


@dataclass(frozen=True)
class FinalTestAccess:
    """Final test capability released only after identity and hash are frozen."""

    selection: FrozenStrategySelection
    final_test: DatasetPartition


class DatasetPartitions:
    """Train/validation/final-test partitions with a sealed final-test member."""

    __slots__ = ("_train", "_validation", "__final_test")

    def __init__(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        final_test: DatasetPartition,
    ) -> None:
        expected = (
            (train, PartitionRole.TRAIN),
            (validation, PartitionRole.VALIDATION),
            (final_test, PartitionRole.FINAL_TEST),
        )
        for partition, role in expected:
            if partition.fingerprint.role is not role:
                raise OOSPartitionError(f"Expected {role.value} partition.")
        ordered = (train, validation, final_test)
        for earlier, later in zip(ordered, ordered[1:]):
            if earlier.fingerprint.end >= later.fingerprint.start:
                raise OOSPartitionError("Chronological partitions must be non-overlapping.")
        self._train = train
        self._validation = validation
        self.__final_test = final_test

    @classmethod
    def from_frame(
        cls,
        data: pd.DataFrame,
        *,
        dataset_id: str,
        config: TimeSeriesSplitConfig | None = None,
    ) -> "DatasetPartitions":
        effective = config or TimeSeriesSplitConfig()
        frame = validate_ohlcv(data)
        n_rows = len(frame)
        train_end = int(n_rows * effective.train_fraction)
        validation_end = train_end + int(n_rows * effective.validation_fraction)
        # Remainder belongs to final test, avoiding unassigned observations.
        lengths = (train_end, validation_end - train_end, n_rows - validation_end)
        if min(lengths) < effective.minimum_partition_rows:
            raise OOSPartitionError(
                "Chronological split leaves fewer than minimum_partition_rows in a partition."
            )
        normalized_id = dataset_id.strip()
        if not normalized_id:
            raise OOSPartitionError("dataset_id must be non-blank.")
        return cls(
            train=DatasetPartition(
                partition_id=f"{normalized_id}:train", role=PartitionRole.TRAIN,
                observations=frame.iloc[:train_end],
            ),
            validation=DatasetPartition(
                partition_id=f"{normalized_id}:validation", role=PartitionRole.VALIDATION,
                observations=frame.iloc[train_end:validation_end],
            ),
            final_test=DatasetPartition(
                partition_id=f"{normalized_id}:final_test", role=PartitionRole.FINAL_TEST,
                observations=frame.iloc[validation_end:],
            ),
        )

    def agent_access(self) -> AgentResearchAccess:
        return AgentResearchAccess(train=self._train)

    def validation_access(self) -> ValidationResearchAccess:
        return ValidationResearchAccess(train=self._train, validation=self._validation)

    def release_final_test(self, selection: FrozenStrategySelection) -> FinalTestAccess:
        if not isinstance(selection, FrozenStrategySelection):
            raise OOSPartitionError("Final test access requires a FrozenStrategySelection.")
        return FinalTestAccess(selection=selection, final_test=self.__final_test)

    @property
    def partition_fingerprints(self) -> tuple[PartitionFingerprint, ...]:
        """Expose all provenance, but never an unsealed final-test frame."""

        return (
            self._train.fingerprint,
            self._validation.fingerprint,
            self.__final_test.fingerprint,
        )


ResearchPartitionAccess = AgentResearchAccess


__all__ = [
    "AgentResearchAccess",
    "DatasetPartition",
    "DatasetPartitions",
    "FinalTestAccess",
    "OOSPartitionError",
    "PartitionFingerprint",
    "PartitionRole",
    "ResearchPartitionAccess",
    "TimeSeriesSplitConfig",
    "ValidationResearchAccess",
]
