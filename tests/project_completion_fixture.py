"""Offline development evidence: python -m tests.project_completion_fixture --output NEW_DIR."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from backtesting.models import BacktestConfig
from data.snapshots import DatasetIdentity, create_snapshot, record_replay, verify_replay
from evaluation.registry import FrozenCandidateRegistry
from paper_trading.operations import PaperSimulationStore
from research_service.backups import create_backup, restore_backup
from research_validation.campaign import CampaignConfig, FoldResearchCampaign
from risk.profiles import risk_profile
from tests.conftest_helpers import make_ohlcv, make_simple_strategy


def run(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    frame = make_ohlcv(140).tz_localize('UTC')
    frame.to_csv(output/'development-prices.csv')
    strategy = make_simple_strategy()
    config = BacktestConfig(initial_capital=10000., position_size_pct=10.)
    identity = DatasetIdentity(symbol='SYNTHETIC', provider='synthetic_recorded', provider_version='v1',
        query={'symbol': 'SYNTHETIC', 'interval': '1d'}, adjustment='raw', timezone='UTC',
        retrieved_at=datetime.now(timezone.utc), calendar_id='synthetic-business-days',
        calendar_version='v1', calendar_source='fixture_only_not_an_exchange_calendar')
    dataset_id = create_snapshot(output/'datasets', frame, identity, expected_sessions=frame.index)
    replay = record_replay(snapshot_root=output/'datasets', dataset_sha256=dataset_id,
        strategy=strategy, config=config, source_root=Path.cwd(), output=output/'replay.json')
    checked = verify_replay(output/'replay.json', output/'datasets', Path.cwd())
    campaign_settings = CampaignConfig(train_size=60, validation_size=20, test_size=20, step_size=20,
        attempts_per_fold=3, maximum_attempts=9, minimum_folds=3)
    def factory(train, fold, attempt):
        return make_simple_strategy(entry_value=49.+attempt)
    report = FoldResearchCampaign().run(development_data=frame, config=campaign_settings,
        backtest_config=config, risk_policy=risk_profile(), candidate_factory=factory)
    report['candidate_source'] = 'RECORDED_DEVELOPMENT_RESPONSES'
    (output/'campaign.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (output/'strategy.json').write_text(strategy.model_dump_json(indent=2), encoding='utf-8')
    (output/'config.json').write_text(config.model_dump_json(indent=2), encoding='utf-8')
    (output/'dataset-identity.json').write_text(identity.model_dump_json(indent=2), encoding='utf-8')
    (output/'campaign-config.json').write_text(json.dumps({'campaign': campaign_settings.model_dump(),
        'backtest': config.model_dump()}), encoding='utf-8')
    responses = [[factory(None, fold, attempt).model_dump(mode='json') for attempt in range(1, 4)]
                 for fold in range(1, 4)]
    (output/'recorded-responses.json').write_text(json.dumps(responses), encoding='utf-8')
    registry = FrozenCandidateRegistry(output/'registry.sqlite')
    candidate = registry.freeze(strategy=strategy, config=config, risk_policy=risk_profile(),
        evidence={'dataset': dataset_id, 'replay_bundle': replay['bundle_sha256']})
    store = PaperSimulationStore(output/'simulation.sqlite')
    session = store.create(registry=registry, candidate_id=candidate, symbol='SYNTHETIC', sessions=frame.index)
    for end in range(60, 66):
        response = store.process(session, frame.iloc[:end], now=frame.index[end-1].to_pydatetime())
        assert response['status'] == 'SIMULATED', response
    duplicate = PaperSimulationStore(store.path).process(session, frame.iloc[:65], now=frame.index[64].to_pydatetime())
    assert duplicate['duplicate'] is True
    store.kill(session)
    assert PaperSimulationStore(store.path).status(session)['suspended']
    drift = store.monitor_drift(session, replay)
    assert drift['status'] == 'INSUFFICIENT_EVIDENCE'
    backup = create_backup(output/'backup', {'registry.sqlite': registry.path,
        'simulation.sqlite': store.path, 'campaign.json': output/'campaign.json', 'replay.json': output/'replay.json'})
    restored = restore_backup(output/'backup', output/'restored')
    assert FrozenCandidateRegistry(output/'restored/registry.sqlite').get(candidate)['paper_eligible'] is False
    summary = {'outcome_class': 'DEVELOPMENT_RESEARCH_FIXTURE', 'runtime_failure': False,
        'replay': checked, 'folds': report['tested_folds'], 'attempts': report['attempts_used'],
        'simulation': store.status(session), 'backup_sha256': backup['sha256'], 'restore': restored,
        'drift': drift,
        'statistical_certification': False, 'paper_eligible': False, 'real_money_eligible': False,
        'historical_untouched_baseline': False}
    (output/'fixture-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
