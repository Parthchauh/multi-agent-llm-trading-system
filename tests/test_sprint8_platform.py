"""Persistence, failure evidence, and report tests for Sprint 8."""

from __future__ import annotations

import sqlite3

import pytest

from evaluation.experiments import (
    ChronologicalSplitConfig,
    ExperimentArchitecture,
    ExperimentConfig,
    ModelRequestConfig,
    PromptVersion,
    TransactionCostConfig,
)
from evaluation.platform import (
    ExperimentRunner,
    ResearchPlatformError,
    SQLiteResearchRepository,
    report_markdown,
)


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


def test_reproducibility_record_includes_conditions_and_stable_config_identity(tmp_path) -> None:
    config = _config(
        start="2020-01-01",
        end="2026-01-01",
        prompt_versions=(
            PromptVersion(role="strategy_generator", version="v2", content_hash="a" * 64),
        ),
        model_settings=ModelRequestConfig(temperature=0.2, timeout_seconds=30),
        transaction_costs=TransactionCostConfig(
            initial_capital=100_000.0,
            commission_pct=0.001,
            slippage_pct=0.0005,
            position_size_pct=10.0,
        ),
        split_config=ChronologicalSplitConfig(
            context_end="2023-12-31",
            validation_end="2024-12-31",
            test_end="2025-12-31",
        ),
    )
    report = ExperimentRunner(SQLiteResearchRepository(tmp_path / "research.sqlite"), git_commit="abc").run(
        config,
        lambda _config, symbol, _repetition: {"status": "COMPLETED", "strategy_id": symbol},
    )

    provenance = report.reproducibility
    assert provenance.dataset_id == "fixture-data"
    assert provenance.symbols == ("AAA", "BBB")
    assert provenance.start == "2020-01-01"
    assert provenance.end == "2026-01-01"
    assert provenance.transaction_costs.position_size_pct == 10.0
    assert provenance.model_settings.temperature == 0.2
    assert provenance.prompt_hashes == (("strategy_generator", "a" * 64),)
    assert len(provenance.experiment_config_sha256) == 64
    assert provenance.dependency_versions
    markdown = report_markdown(report)
    assert "Requested data window" in markdown
    assert "Experiment configuration SHA-256" in markdown


def test_experiment_identifier_cannot_rewrite_an_existing_treatment(tmp_path) -> None:
    repository = SQLiteResearchRepository(tmp_path / "research.sqlite")
    original = _config(model="model-a")
    repository.save_experiment(original)
    repository.save_experiment(original)  # Exact persistence is idempotent.

    with pytest.raises(ResearchPlatformError, match="different immutable configuration"):
        repository.save_experiment(_config(model="model-b"))

    with sqlite3.connect(tmp_path / "research.sqlite") as connection:
        persisted = connection.execute(
            "SELECT config_json FROM experiments WHERE experiment_id=?", (original.experiment_id,)
        ).fetchone()[0]
    assert '"model":"model-a"' in persisted
