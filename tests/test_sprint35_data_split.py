"""Sprint 3.5 data split and holdout-access control tests.

Verifies:
- ResearchDatasetSplit temporal isolation (context.end < validation.start)
- Holdout cannot be accessed without FrozenStrategySelection
- make_dataset_split() produces correct partition structure
- generation_view() only sees context data
- development_view() sees context+validation+test but NOT holdout
"""
from __future__ import annotations
import pytest
from data.research_split import (
    ResearchDatasetSplit, ResearchPartition, ResearchPartitionRole,
    FrozenStrategySelection, ResearchSplitError,
)
from tests.conftest_helpers import make_dataset_split, make_frozen_selection


class TestDatasetSplitStructure:
    def test_make_dataset_split_creates_four_partitions(self):
        split = make_dataset_split(n=400)
        fps = split.partition_fingerprints
        assert len(fps) == 4
        roles = [fp.role for fp in fps]
        assert ResearchPartitionRole.CONTEXT in roles
        assert ResearchPartitionRole.VALIDATION in roles
        assert ResearchPartitionRole.TEST in roles
        assert ResearchPartitionRole.FINAL_HOLDOUT in roles

    def test_partitions_are_temporally_ordered(self):
        split = make_dataset_split(n=400)
        fps = split.partition_fingerprints
        for earlier, later in zip(fps, fps[1:]):
            assert earlier.end < later.start, (
                f"{earlier.role.value}.end ({earlier.end}) must be "
                f"strictly before {later.role.value}.start ({later.start})"
            )

    def test_context_end_before_validation_start(self):
        split = make_dataset_split(n=400)
        fps = {fp.role: fp for fp in split.partition_fingerprints}
        assert fps[ResearchPartitionRole.CONTEXT].end < fps[ResearchPartitionRole.VALIDATION].start

    def test_generation_view_only_exposes_context(self):
        split = make_dataset_split(n=400)
        gen = split.generation_view()
        ctx_fp = split.partition_fingerprints[0]
        assert gen.context.fingerprint.partition_id == ctx_fp.partition_id
        # decision_timestamp is within [context.end, validation.start)
        assert gen.decision_timestamp >= ctx_fp.end

    def test_development_view_does_not_expose_holdout_data(self):
        split = make_dataset_split(n=400)
        dev = split.development_view()
        holdout_fp = split.partition_fingerprints[3]
        # dev only has context, validation, test — no holdout
        assert dev.context.fingerprint.role is ResearchPartitionRole.CONTEXT
        assert dev.validation.fingerprint.role is ResearchPartitionRole.VALIDATION
        assert dev.test.fingerprint.role is ResearchPartitionRole.TEST
        # Verify holdout partition_id does NOT appear in dev view fingerprints
        dev_ids = {
            dev.context.fingerprint.partition_id,
            dev.validation.fingerprint.partition_id,
            dev.test.fingerprint.partition_id,
        }
        assert holdout_fp.partition_id not in dev_ids


class TestHoldoutAccessControl:
    def test_holdout_requires_frozen_selection(self):
        split = make_dataset_split(n=400)
        selection = make_frozen_selection("strategy-abc")
        view = split.release_final_holdout(selection)
        assert view.final_holdout.fingerprint.role is ResearchPartitionRole.FINAL_HOLDOUT
        assert view.selection.strategy_id == "strategy-abc"

    def test_holdout_access_without_selection_raises(self):
        split = make_dataset_split(n=400)
        with pytest.raises((ResearchSplitError, TypeError)):
            split.release_final_holdout("not-a-selection")  # type: ignore[arg-type]

    def test_holdout_access_with_none_raises(self):
        split = make_dataset_split(n=400)
        with pytest.raises((ResearchSplitError, TypeError)):
            split.release_final_holdout(None)  # type: ignore[arg-type]

    def test_frozen_selection_requires_non_empty_id(self):
        with pytest.raises((ResearchSplitError, ValueError)):
            FrozenStrategySelection(
                strategy_id="   ",
                strategy_hash="abc123",
                frozen_at=__import__("datetime").datetime.utcnow(),
            )


class TestTemporalIsolationEnforced:
    def test_overlapping_partitions_rejected(self):
        """ResearchDatasetSplit must reject partitions whose boundaries overlap."""
        import pandas as pd
        import numpy as np
        dates_a = pd.date_range("2020-01-01", periods=100, freq="B")
        dates_b = pd.date_range("2020-01-01", periods=80, freq="B")  # starts SAME as a

        def _frame(dates):
            n = len(dates)
            rng = np.random.default_rng(1)
            closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
            return pd.DataFrame({
                "Open": closes * 0.999, "High": closes * 1.005,
                "Low": closes * 0.995, "Close": closes,
                "Volume": np.full(n, 100000.0)
            }, index=dates)

        ctx = ResearchPartition(partition_id="c", role=ResearchPartitionRole.CONTEXT, observations=_frame(dates_a))
        val = ResearchPartition(partition_id="v", role=ResearchPartitionRole.VALIDATION, observations=_frame(dates_b))

        with pytest.raises(ResearchSplitError):
            ResearchDatasetSplit(
                context=ctx,
                validation=val,
                test=val,      # would also overlap but error fires on ctx/val first
                final_holdout=val,
            )
