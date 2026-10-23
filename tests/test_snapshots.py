import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from backtesting.models import BacktestConfig
from cli.research import app
from data.snapshots import (
    DatasetIdentity,
    create_snapshot,
    load_snapshot,
    record_replay,
    verify_replay,
)
from tests.conftest_helpers import make_ohlcv, make_simple_strategy


def identity(**kwargs):
    return DatasetIdentity(symbol="SYNTHETIC", provider="recorded", provider_version="v1",
        query={"symbol": "SYNTHETIC", "interval": "1d"}, timezone="UTC", adjustment="raw",
        retrieved_at=datetime(2026, 10, 21, tzinfo=timezone.utc), **kwargs)


def test_snapshot_is_immutable_and_change_of_query_changes_identity(tmp_path):
    frame = make_ohlcv(70)
    digest = create_snapshot(tmp_path, frame, identity())
    assert create_snapshot(tmp_path, frame, identity()) == digest
    loaded, meta = load_snapshot(tmp_path, digest)
    pd.testing.assert_frame_equal(loaded, frame, check_freq=False)
    assert meta['quality']['point_in_time_universe_verified'] is False
    alternative = identity().model_copy(update={'query': {'symbol': 'OTHER'}})
    assert create_snapshot(tmp_path, frame, alternative) != digest
    path = tmp_path / (digest+'.json')
    path.write_text(path.read_text().replace('SYNTHETIC', 'MANIPULATED'))
    with pytest.raises(ValueError, match='digest'):
        load_snapshot(tmp_path, digest)
    with pytest.raises(ValueError, match='match'):
        create_snapshot(tmp_path, frame, identity())
    with pytest.raises(ValueError, match='identifier'):
        load_snapshot(tmp_path, '../../secrets')


def test_calendar_missing_duplicate_extra_and_timezone_rejected(tmp_path):
    frame = make_ohlcv(10)
    meta = identity(calendar_id='synthetic', calendar_version='v1', calendar_source='test')
    digest = create_snapshot(tmp_path, frame, meta, expected_sessions=frame.index)
    assert load_snapshot(tmp_path, digest)[1]['quality']['session_coverage_checked']
    for sessions in (frame.index[:-1], frame.index.append(frame.index[:1]), frame.index.tz_localize('UTC')):
        with pytest.raises(ValueError):
            create_snapshot(tmp_path, frame, meta, expected_sessions=sessions)
    with pytest.raises(ValueError, match='versioned'):
        create_snapshot(tmp_path, frame, identity(), expected_sessions=frame.index)


def test_replay_matches_and_refuses_tampered_bundle(tmp_path):
    digest = create_snapshot(tmp_path, make_ohlcv(70), identity())
    bundle = tmp_path/'replay.json'
    record_replay(snapshot_root=tmp_path, dataset_sha256=digest, strategy=make_simple_strategy(),
        config=BacktestConfig(initial_capital=10000.), source_root=Path.cwd(), output=bundle)
    assert verify_replay(bundle, tmp_path, Path.cwd())['status'] == 'REPLAY_MATCH'
    with pytest.raises(FileExistsError):
        record_replay(snapshot_root=tmp_path, dataset_sha256=digest, strategy=make_simple_strategy(),
            config=BacktestConfig(initial_capital=10000.), source_root=Path.cwd(), output=bundle)
    changed = json.loads(bundle.read_text())
    changed['config']['slippage_pct'] = .1
    bundle.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='bundle digest'):
        verify_replay(bundle, tmp_path, Path.cwd())


def test_new_commands_are_available_and_report_machine_readable_failures(tmp_path):
    runner = CliRunner()
    for command in ('snapshot-dataset', 'record-replay', 'verify-replay', 'fold-research'):
        assert runner.invoke(app, [command, '--help']).exit_code == 0
    result = runner.invoke(app, ['verify-replay', str(tmp_path/'missing.json')])
    assert result.exit_code == 2
    assert 'verification failed' in result.output
