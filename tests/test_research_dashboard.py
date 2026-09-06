"""Read-only local dashboard integration tests."""

from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from evaluation.experiments import ExperimentArchitecture, ExperimentConfig
from evaluation.platform import ExperimentRunner, SQLiteResearchRepository
from research_dashboard.server import create_dashboard_server


def _make_report(database) -> None:
    config = ExperimentConfig(
        experiment_id="dashboard-test",
        architecture=ExperimentArchitecture.SINGLE_AGENT,
        provider="fake",
        model="fake-model",
        dataset_id="fixture-data",
        symbols=("AAA",),
    )
    ExperimentRunner(SQLiteResearchRepository(database)).run(
        config,
        lambda _config, symbol, _repetition: {
            "status": "COMPLETED",
            "strategy_id": f"{symbol}-strategy",
        },
    )


def test_dashboard_serves_reports_and_rejects_writes(tmp_path) -> None:
    database = tmp_path / "research.sqlite"
    _make_report(database)
    server = create_dashboard_server(database, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(f"{base_url}/health") as response:
            assert json.load(response)["status"] == "ok"
        with urlopen(f"{base_url}/api/experiments") as response:
            reports = json.load(response)
        assert reports[0]["experiment_id"] == "dashboard-test"
        with urlopen(f"{base_url}/reports/dashboard-test.md") as response:
            assert "# Experiment dashboard-test" in response.read().decode("utf-8")
        with urlopen(f"{base_url}/") as response:
            assert "Research Dashboard" in response.read().decode("utf-8")
        with pytest.raises(HTTPError) as error:
            urlopen(Request(f"{base_url}/api/experiments", method="POST"))
        assert error.value.code == 405
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_dashboard_refuses_non_local_network_binding(tmp_path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        create_dashboard_server(tmp_path / "research.sqlite", host="0.0.0.0")
