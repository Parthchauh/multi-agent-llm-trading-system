"""Research CLI tests that avoid provider/network access."""

from __future__ import annotations

import pytest
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
    assert reports[0].runs[0].result['test_metrics'] == {'total_return_pct': 0.5}
    assert reports[0].runs[0].result['final_holdout_metrics'] == {}


@pytest.mark.parametrize('architecture,debates', [('multi_agent_no_debate', 0),
    ('multi_agent_debate', 1), ('multi_agent_risk', 1)])
def test_experiment_command_wires_controlled_treatments(monkeypatch, tmp_path, architecture, debates):
    observed = []
    def fake_e2e(**kwargs):
        observed.append(kwargs)
        return {'status': 'VIABILITY_REJECTED', 'selected_strategy_id': None,
                'validation_metrics': {}, 'test_metrics': {}, 'output_dir': str(tmp_path)}
    monkeypatch.setattr('scripts.run_research_e2e.run_e2e_research', fake_e2e)
    result = CliRunner().invoke(
        app,
        ['experiment', '--symbol', 'AAA', '--architecture', architecture, '--database', str(tmp_path/'runs.sqlite')],
    )
    assert result.exit_code == 0, result.output
    assert observed[0]['mode'] == architecture.upper()
    assert observed[0]['max_debate_rounds'] == debates
    assert observed[0]['max_refinement_rounds'] == 0


def test_canonical_cli_runs_real_local_pipeline(monkeypatch, tmp_path):
    import scripts.run_research_e2e as runner
    from tests.conftest_helpers import make_ohlcv
    from tests.test_research_e2e import MockLifecycleClient
    original = runner.run_e2e_research

    def local_fixture(**kwargs):
        return original(**kwargs, df=make_ohlcv(350, seed=1),
                        structured_client=MockLifecycleClient())

    monkeypatch.setattr(runner, 'run_e2e_research', local_fixture)
    database = tmp_path / 'research.sqlite'
    result = CliRunner().invoke(app, ['experiment', '--symbol', 'AAA', '--architecture',
        'single_agent', '--database', str(database), '--output-dir', str(tmp_path / 'runs')])
    assert result.exit_code == 0, result.output
    run = SQLiteResearchRepository(database).list_reports()[0].runs[0]
    assert run.status == 'VIABILITY_REJECTED'
    assert run.outcome_class == 'RESEARCH_REJECTION'
    report = SQLiteResearchRepository(database).list_reports()[0]
    assert report.failures == ()
    assert len(report.rejections) == 1
    assert run.result['test_metrics'] == {}
    assert run.result['final_holdout_metrics'] == {}
    assert run.result['certification']['DEPLOYMENT_ELIGIBLE'] is False
