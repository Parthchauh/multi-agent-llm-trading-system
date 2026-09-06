"""Persistence, failure evidence, and report tests for Sprint 8."""

from __future__ import annotations

import sqlite3

from evaluation.experiments import ExperimentArchitecture, ExperimentConfig
from evaluation.platform import ExperimentRunner, SQLiteResearchRepository, report_markdown


def _config(**overrides) -> ExperimentConfig:
    payload = {
        "experiment_id": "s8-platform-test",
        "architecture": ExperimentArchitecture.SINGLE_AGENT,
        "provider": "fake",
        "model": "fake-model",
        "dataset_id": "fixture-data",
        "symbols": ("aaa", "bbb"),
        "repetitions": 2,
    }
    payload.update(overrides)
    return ExperimentConfig(**payload)


def test_runner_persists_successes_failures_events_and_canonical_report(tmp_path) -> None:
    repository = SQLiteResearchRepository(tmp_path / "research.sqlite")

    def executor(config, symbol, repetition):
        if symbol == "BBB" and repetition == 2:
            raise ValueError("do not persist provider detail")
        return {
            "status": "COMPLETED",
            "strategy_id": f"{symbol}-{repetition}",
            "metric": 1.0,
            "provider_metadata": {"api_key": "must-not-persist"},
        }

    report = ExperimentRunner(repository, git_commit="abc123").run(_config(), executor)

    assert len(report.runs) == 4
    assert len(report.failures) == 1
    assert report.failures[0].failure_type == "UNKNOWN"
    assert len(repository.list_runs("s8-platform-test")) == 4
    assert len(repository.list_events("s8-platform-test")) == 8
    assert repository.load_report("s8-platform-test") == report
    markdown = report_markdown(report)
    assert "Failure Analysis" in markdown
    assert "do not persist provider detail" not in markdown
    persisted_json = "\n".join(
        run.model_dump_json() for run in repository.list_runs("s8-platform-test")
    )
    assert "must-not-persist" not in persisted_json
    assert "[REDACTED]" in persisted_json

    with sqlite3.connect(tmp_path / "research.sqlite") as connection:
        assert connection.execute("SELECT count(*) FROM experiments").fetchone()[0] == 1


def test_event_metadata_rejects_secret_fields() -> None:
    from datetime import datetime, timezone

    import pytest

    from evaluation.platform import ResearchEvent

    with pytest.raises(ValueError, match="secrets"):
        ResearchEvent(
            timestamp=datetime.now(timezone.utc), run_id="run", node="test",
            status="FAILED", event_type="TEST", metadata={"api_key": "hidden"}
        )
