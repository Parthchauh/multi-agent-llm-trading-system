import sqlite3

import pytest

from research_service.monitoring import ServiceMonitor


def test_alerts_survive_restart_deduplicate_and_can_be_acknowledged(tmp_path):
    path = tmp_path/'metrics.sqlite'
    monitor = ServiceMonitor(path, clock=lambda: 1000.)
    assert monitor.snapshot()['status'] == 'INSUFFICIENT_TRAFFIC'
    for i in range(20):
        monitor.record(503 if i < 2 else 200, 3000. if i < 2 else 10.)
    result = monitor.snapshot()
    assert result['server_error_rate'] == .1
    assert result['latency_p95_ms'] == 3000.
    assert {a['reason'] for a in result['open_alerts']} == {'HTTP_ERROR_RATE', 'HTTP_LATENCY_P95'}
    restarted = ServiceMonitor(path, clock=lambda: 1001.)
    assert len(restarted.snapshot()['open_alerts']) == 2
    for incident in result['open_alerts']:
        restarted.acknowledge(incident['incident_id'])
    assert restarted.snapshot()['open_alerts'] == []
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM operational_alerts').fetchone()[0] == 2
    with pytest.raises(KeyError):
        restarted.acknowledge('unknown')


def test_window_future_samples_denials_and_no_traffic_are_not_uptime(tmp_path):
    path = tmp_path/'metrics.sqlite'
    old = ServiceMonitor(path, clock=lambda: 1.)
    old.record(500, 10.)
    future = ServiceMonitor(path, clock=lambda: 10000.)
    future.record(500, 10.)
    current = ServiceMonitor(path, clock=lambda: 1000.)
    current.record(403, 20.)
    current.record(429, 30.)
    result = current.snapshot(minimum_requests=2)
    assert result['requests'] == 2
    assert result['access_denials'] == result['rate_limited'] == 1
    assert result['server_error_rate'] == 0.
    assert result['queue_depth'] is result['token_spend'] is None
    assert result['external_delivery'] == 'NOT_CONFIGURED'
    empty = ServiceMonitor(path, clock=lambda: 20000.).snapshot()
    assert empty['status'] == 'INSUFFICIENT_TRAFFIC'
    assert empty['server_error_rate'] is None


@pytest.mark.parametrize('status,latency', [(True, 1.), (99, 1.), (200, float('nan')), (200, -1.)])
def test_invalid_observations_are_rejected(tmp_path, status, latency):
    with pytest.raises(ValueError):
        ServiceMonitor(tmp_path/'metrics.sqlite').record(status, latency)
