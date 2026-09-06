"""Research CLI tests that avoid provider/network access."""

from __future__ import annotations

from typer.testing import CliRunner

from cli.research import app
from evaluation.platform import SQLiteResearchRepository


def test_experiment_command_persists_a_mocked_authoritative_result(
    monkeypatch, tmp_path
) -> None:
    def fake_e2e(**kwargs):
        return {
            "status": "COMPLETED",
            "selected_strategy_id": "strategy-test",
            "validation_metrics": {"total_return_pct": 1.0},
            "test_metrics": {"total_return_pct": 0.5},
            "output_dir": str(kwargs["output_dir"]),
        }

    monkeypatch.setattr("scripts.run_research_e2e.run_e2e_research", fake_e2e)
    database = tmp_path / "research.sqlite"
    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "--symbol",
            "aaa",
            "--architecture",
            "single_agent",
            "--database",
            str(database),
        ],
    )

    assert result.exit_code == 0, result.output
    reports = SQLiteResearchRepository(database).list_reports()
    assert len(reports) == 1
    assert reports[0].runs[0].result["final_holdout_metrics"] == {"total_return_pct": 0.5}


def test_experiment_command_refuses_unwired_architecture() -> None:
    result = CliRunner().invoke(
        app,
        ["experiment", "--symbol", "AAA", "--architecture", "multi_agent_debate"],
    )
    assert result.exit_code != 0
    assert "single_agent and multi_agent_full" in result.output
