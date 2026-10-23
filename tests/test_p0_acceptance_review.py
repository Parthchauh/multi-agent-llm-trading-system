"""Acceptance-review regressions for capability scope and persisted exposure."""

import json
import subprocess
import sys
from datetime import datetime

import pytest

from data.research_split import (
    FrozenStrategySelection,
    ResearchDatasetSplit,
    ResearchPartition,
    ResearchPartitionRole,
)
from evaluation.certification import EvaluationLifecycle, evidence_status
from scripts.run_research_e2e import build_research_dataset_split, run_e2e_research
from tests.conftest_helpers import make_ohlcv, make_simple_strategy
from tests.test_research_e2e import MockLifecycleClient


def test_canonical_split_rejects_compatibility_and_early_release():
    split = build_research_dataset_split(make_ohlcv(300), symbol="AAA")
    selection = FrozenStrategySelection("candidate", "hash", datetime(2020, 1, 1))
    with pytest.raises(ValueError, match="legacy_access"):
        split.development_view()
    with pytest.raises(ValueError, match="lifecycle"):
        split.release_confirmatory_test(selection)
    with pytest.raises(ValueError, match="lifecycle"):
        split.release_final_holdout(selection)
    state = EvaluationLifecycle()
    state.freeze("hash")
    with pytest.raises(ValueError, match="TEST_CONFIRMATORY"):
        split.release_confirmatory_test(selection, lifecycle=state)
    state.begin_test("hash")
    split.release_confirmatory_test(selection, lifecycle=state)
    with pytest.raises(ValueError, match="already released"):
        split.release_confirmatory_test(selection, lifecycle=state)
    state.begin_holdout(test_passed=True)
    wrong = FrozenStrategySelection("other", "different-hash", datetime(2020, 1, 1))
    with pytest.raises(ValueError, match="hash"):
        split.release_final_holdout(wrong, lifecycle=state)
    split.release_final_holdout(selection, lifecycle=state)
    with pytest.raises(ValueError, match="only once"):
        split.release_final_holdout(selection, lifecycle=state)


def test_holdout_exposure_survives_separate_processes(tmp_path):
    script = """
import sys
from datetime import datetime
from types import SimpleNamespace
from evaluation.campaigns import HoldoutExposureLedger
fp = SimpleNamespace(start=datetime.fromisoformat(sys.argv[3]),
    end=datetime.fromisoformat(sys.argv[4]), data_sha256=sys.argv[2])
print(HoldoutExposureLedger(sys.argv[1]).claim(dataset_family=sys.argv[5],
    campaign_id=sys.argv[2], strategy_hash=sys.argv[2], fingerprint=fp))
"""
    path = str(tmp_path / "persistent.sqlite")

    def claim(identity, start, end, family):
        return subprocess.check_output(
            [sys.executable, "-c", script, path, identity, start, end, family], text=True
        ).strip()

    assert claim("old", "2020-01-01", "2020-01-05", "AAA:provider-a") == "UNSEEN_CONFIRMATORY"
    # New process, strategy, campaign, hash, and partly overlapping dates do not reset exposure.
    assert (
        claim("new", "2020-01-04", "2020-01-08", "AAA:provider-b")
        == "PREVIOUSLY_EXPOSED_EXPLORATORY"
    )
    assert claim("fresh", "2020-02-01", "2020-02-05", "AAA:provider-b") == "UNSEEN_CONFIRMATORY"


@pytest.mark.parametrize(
    "robustness,release", [(False, False), (True, False), (False, True), (True, True)]
)
def test_boolean_assertions_cannot_enable_certification_or_promotion(robustness, release):
    arguments = dict(
        technical=True,
        validation_passed=True,
        test_passed=True,
        holdout_passed=True,
        risk_passed=True,
        robustness_passed=robustness,
        release_gates_passed=release,
    )
    if robustness or release:
        with pytest.raises(ValueError, match="disabled"):
            evidence_status(**arguments)
    else:
        status = evidence_status(**arguments)
        assert not status["STATISTICALLY_VIABLE"]
        assert not status["DEPLOYMENT_ELIGIBLE"]
        assert not status["real_money_execution"]


def test_agents_never_receive_or_trigger_sealed_observations(monkeypatch, tmp_path):
    frame = make_ohlcv(350, seed=1)
    # A conspicuous, valid price sentinel in both sealed partitions.
    frame.iloc[245:, frame.columns.get_indexer(["Open", "Close"])] = 932871013.0
    frame.iloc[245:, frame.columns.get_indexer(["High"])] = 932871014.0
    frame.iloc[245:, frame.columns.get_indexer(["Low"])] = 932871012.0
    client = MockLifecycleClient()
    original_invoke = client.invoke
    original_features = ResearchPartition.feature_frame.fget

    def invoke(schema, messages):
        assert "93287101" not in json.dumps(messages)
        assert "test_metrics" not in json.dumps(messages)
        assert "final_holdout_metrics" not in json.dumps(messages)
        return original_invoke(schema, messages)

    def guarded_features(partition):
        if partition.fingerprint.role in {
            ResearchPartitionRole.TEST,
            ResearchPartitionRole.FINAL_HOLDOUT,
        }:
            raise AssertionError("Rejected fixture must never read sealed observations.")
        return original_features(partition)

    def forbidden_compatibility(*args):
        raise AssertionError("Canonical graph must not request a legacy development view.")

    client.invoke = invoke
    monkeypatch.setattr(ResearchPartition, "feature_frame", property(guarded_features))
    monkeypatch.setattr(ResearchDatasetSplit, "development_view", forbidden_compatibility)
    result = run_e2e_research(
        symbol="AAA", df=frame, structured_client=client, mode="SINGLE_AGENT", output_dir=tmp_path
    )
    assert result["status"] == "VIABILITY_REJECTED"
    assert len(client.calls) == 2
    assert result["test_metrics"] == result["final_holdout_metrics"] == {}
    risk = json.loads((tmp_path / "risk.json").read_text())
    assert risk["status"] == "VIABILITY_REJECTED"


def test_review_and_refinement_finish_before_confirmatory_access(monkeypatch, tmp_path, caplog):
    import numpy as np
    import pandas as pd

    base = 100.0 + np.arange(1000) * 0.1
    frame = pd.DataFrame(
        {
            "Open": base,
            "Close": base + 0.1,
            "High": base + 0.2,
            "Low": base - 0.1,
            "Volume": [10000.0] * 1000,
        },
        index=pd.date_range("2020-01-01", periods=1000, freq="B"),
    )
    frame.iloc[700:, frame.columns.get_indexer(["Open", "Close"])] = 932871013.0
    frame.iloc[700:, frame.columns.get_indexer(["High"])] = 932871014.0
    frame.iloc[700:, frame.columns.get_indexer(["Low"])] = 932871012.0
    strategy = make_simple_strategy(
        entry_indicator="close",
        entry_operator=">",
        entry_value=0.0,
        exit_indicator="close",
        exit_operator="<",
        exit_value=0.0,
        max_holding_days=1,
        position_size_pct=10.0,
    )
    client = MockLifecycleClient(raw_strategy=strategy.model_dump(mode="json"))
    child = strategy.model_dump(mode="json")
    child["entry"]["long_conditions"]["conditions"][0]["value"] = 0.1
    client.responses[-1]["revised_strategy"] = child
    invoke = client.invoke
    release = ResearchDatasetSplit.release_confirmatory_test
    released = False

    def guarded_invoke(schema, messages):
        assert not released, "An adaptive agent was called after test access."
        assert "93287101" not in json.dumps(messages)
        return invoke(schema, messages)

    def guarded_release(split, selection, **kwargs):
        nonlocal released
        assert len(client.calls) == 9  # Includes bull/bear, critic, and refiner.
        partition = release(split, selection, **kwargs)
        released = True
        return partition

    client.invoke = guarded_invoke
    monkeypatch.setattr(ResearchDatasetSplit, "release_confirmatory_test", guarded_release)
    result = run_e2e_research(
        symbol="AAA",
        df=frame,
        commission=0.0,
        slippage=0.0,
        structured_client=client,
        mode="MULTI_AGENT_FULL",
        output_dir=tmp_path,
        exposure_ledger_path=tmp_path / "ledger.sqlite",
    )
    assert released
    assert result["status"] in {"COMPLETED", "REFINEMENT_REJECTED"}
    assert result["test_metrics"]["number_of_trades"] == 0
    assert result["final_holdout_metrics"] == {}
    assert "93287101" not in (tmp_path / "agent_trace.json").read_text()
    assert "93287101" not in caplog.text


def test_custom_executor_cannot_persist_unsupported_promotion(tmp_path):
    from evaluation.experiments import ExperimentArchitecture, ExperimentConfig
    from evaluation.platform import ExperimentRunner, SQLiteResearchRepository

    config = ExperimentConfig(
        experiment_id="unsupported-promotion",
        architecture=ExperimentArchitecture.SINGLE_AGENT,
        provider="fake",
        model="fake",
        dataset_id="fixture",
        symbols=("AAA",),
    )
    report = ExperimentRunner(SQLiteResearchRepository(tmp_path / "research.sqlite")).run(
        config,
        lambda *args: {"status": "COMPLETED", "certification": {"DEPLOYMENT_ELIGIBLE": True}},
    )
    assert report.runs[0].status == "FAILED"
    assert report.runs[0].outcome_class == "RUNTIME_FAILURE"
    assert report.runs[0].result == {}
