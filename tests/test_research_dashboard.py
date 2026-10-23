"""Read-only local dashboard integration tests."""

from __future__ import annotations

import json
import socket
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from evaluation.experiments import ExperimentArchitecture, ExperimentConfig
from evaluation.platform import ExperimentRunner, SQLiteResearchRepository
from research_dashboard.server import create_dashboard_server, serve_dashboard


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
        assert reports[0]['certifications'][0]['statuses'] == {}
        with urlopen(f"{base_url}/api/evidence/dashboard-test") as response:
            details = json.load(response)
        assert details['runs'][0]['strategy'] is None
        assert details['runs'][0]['paper'] == {'status': 'NOT_OBSERVED', 'metrics': None}
        assert details['runs'][0]['evidence']['final_holdout'] is None
        with urlopen(f"{base_url}/evidence/dashboard-test") as response:
            details_html = response.read().decode('utf-8')
        assert 'textContent' in details_html
        assert 'Data, model, prompt and source identity' in details_html
        with urlopen(f"{base_url}/reports/dashboard-test.md") as response:
            assert "# Experiment dashboard-test" in response.read().decode("utf-8")
        with urlopen(f"{base_url}/") as response:
            html = response.read().decode('utf-8')
            assert 'Research Dashboard' in html
            assert 'STATISTICALLY_VIABLE' in html
            assert 'DEPLOYMENT_ELIGIBLE' in html
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


def test_dashboard_supports_explicit_ipv6_loopback_when_available(tmp_path) -> None:
    if not socket.has_ipv6:
        pytest.skip("Python was built without IPv6 support")
    try:
        server = create_dashboard_server(tmp_path / "research.sqlite", host="::1", port=0)
    except OSError:
        pytest.skip("IPv6 loopback is unavailable in this environment")
    try:
        assert server.address_family == socket.AF_INET6
        assert server.server_address[0] == "::1"
    finally:
        server.server_close()


def test_dashboard_explains_a_port_conflict(tmp_path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        with pytest.raises(RuntimeError, match="already be in use"):
            serve_dashboard(tmp_path / "research.sqlite", port=port)
