import json
import sqlite3
from datetime import timedelta

import pandas as pd
import pytest

from backtesting.models import BacktestConfig
from evaluation.registry import FrozenCandidateRegistry
from paper_trading.operations import PaperSimulationStore
from risk.profiles import risk_profile
from tests.conftest_helpers import make_simple_strategy


class Signals:
    def should_enter(self, i, data):
        return True
    def should_exit(self, i, data):
        return False
    def get_stop_price(self, *args):
        return None
    def get_take_profit_price(self, *args):
        return None


def setup(tmp_path, monkeypatch):
    from strategies.compiler import StrategyCompiler
    monkeypatch.setattr(StrategyCompiler, 'compile', lambda *args: Signals())
    frame = pd.DataFrame({'Open': [100., 100., 110., 110.], 'High': [101., 101., 111., 111.],
        'Low': [99., 99., 109., 109.], 'Close': [100., 100., 110., 110.], 'Volume': [1000.]*4},
        index=pd.date_range('2026-10-01', periods=4, tz='UTC'))
    registry = FrozenCandidateRegistry(tmp_path/'registry.sqlite')
    candidate = registry.freeze(strategy=make_simple_strategy(),
        config=BacktestConfig(initial_capital=10000., position_size_pct=20., commission_pct=.001, slippage_pct=.001),
        risk_policy=risk_profile(), evidence={'fixture': '0'*64})
    store = PaperSimulationStore(tmp_path/'simulation.sqlite')
    session = store.create(registry=registry, candidate_id=candidate, symbol='AAA', sessions=frame.index)
    return registry, candidate, store, session, frame


def test_registry_integrity_and_registration_never_grants_eligibility(tmp_path, monkeypatch):
    registry, candidate, store, session, frame = setup(tmp_path, monkeypatch)
    assert registry.get(candidate)['paper_eligible'] is False
    with pytest.raises(PermissionError, match='disabled'):
        registry.require_paper_eligible(candidate)
    with sqlite3.connect(registry.path) as db:
        row = json.loads(db.execute('SELECT payload FROM frozen_candidates').fetchone()[0])
        row['paper_eligible'] = True
        db.execute('UPDATE frozen_candidates SET payload=?', (json.dumps(row),))
    with pytest.raises(ValueError, match='integrity'):
        registry.get(candidate)


def test_duplicate_and_restart_are_idempotent_and_kill_cancels_pending_entry(tmp_path, monkeypatch):
    _, _, store, session, frame = setup(tmp_path, monkeypatch)
    first = store.process(session, frame.iloc[:1], now=frame.index[0].to_pydatetime())
    assert first['status'] == 'SIMULATED'
    fresh = PaperSimulationStore(store.path)
    duplicate = fresh.process(session, frame.iloc[:1], now=frame.index[0].to_pydatetime())
    assert duplicate == {**first, 'duplicate': True}
    fresh.kill(session)
    killed_duplicate = fresh.process(session, frame.iloc[:1], now=frame.index[0].to_pydatetime())
    assert killed_duplicate['duplicate'] and killed_duplicate['suspended']
    result = PaperSimulationStore(store.path).process(session, frame.iloc[:2], now=frame.index[1].to_pydatetime())
    assert result['fills'] == []
    assert result['snapshot']['cash'] == 10000.
    assert result['suspended'] is True
    assert store.status(session)['accepted_bars'] == 2


def test_kill_closes_existing_position_at_next_valid_open_with_independent_golden_cash(tmp_path, monkeypatch):
    _, _, store, session, frame = setup(tmp_path, monkeypatch)
    store.process(session, frame.iloc[:1], now=frame.index[0].to_pydatetime())
    bought = store.process(session, frame.iloc[:2], now=frame.index[1].to_pydatetime())
    assert bought['fills'][0]['quantity'] == 19.
    assert bought['snapshot']['cash'] == pytest.approx(8096.1981, abs=1e-8, rel=0)
    store.kill(session)
    sold = PaperSimulationStore(store.path).process(session, frame.iloc[:3], now=frame.index[2].to_pydatetime())
    assert sold['fills'][0]['fill_price'] == pytest.approx(109.89, abs=1e-8, rel=0)
    # Public snapshots intentionally display four decimals; checkpoint cash retains precision.
    assert sold['snapshot']['cash'] == pytest.approx(10182.0202, abs=1e-8, rel=0)
    with sqlite3.connect(store.path) as db:
        payload = json.loads(db.execute('SELECT payload FROM simulations').fetchone()[0])
    assert payload['checkpoint']['cash'] == pytest.approx(10182.02019, abs=1e-8, rel=0)
    assert sold['snapshot']['open_positions_count'] == 0
    fourth = store.process(session, frame, now=frame.index[3].to_pydatetime())
    assert fourth['fills'] == []
    assert fourth['snapshot']['cash'] == sold['snapshot']['cash']


@pytest.mark.parametrize('case', ['stale', 'future', 'missing_session', 'revised_bar'])
def test_invalid_data_latches_suspension_without_changing_accepted_checkpoint(tmp_path, monkeypatch, case):
    _, _, store, session, frame = setup(tmp_path, monkeypatch)
    store.process(session, frame.iloc[:1], now=frame.index[0].to_pydatetime())
    bad = frame.iloc[:2].copy()
    now = frame.index[1].to_pydatetime()
    if case == 'stale':
        now += timedelta(days=10)
    elif case == 'future':
        now -= timedelta(days=1)
    elif case == 'missing_session':
        bad = frame.iloc[[0, 2]]
    else:
        bad.iloc[0, 3] = 100.5
    rejected = store.process(session, bad, now=now)
    assert rejected['status'] == 'REJECTED_DATA'
    assert rejected['suspended'] is True
    status = PaperSimulationStore(store.path).status(session)
    assert status['accepted_bars'] == 1
    assert len(status['incidents']) == 1


def test_risk_breach_persists_without_fabricating_paper_eligibility(tmp_path, monkeypatch):
    _, _, store, session, frame = setup(tmp_path, monkeypatch)
    store.process(session, frame.iloc[:1], now=frame.index[0].to_pydatetime())
    store.process(session, frame.iloc[:2], now=frame.index[1].to_pydatetime())
    # The 19-share position rises above its 20% portfolio-exposure policy.
    third = store.process(session, frame.iloc[:3], now=frame.index[2].to_pydatetime())
    assert third['suspended'] is True
    assert store.status(session)['incidents'][-1]['reason'] == 'RISK_LIMIT_BREACH'
    assert store.status(session)['paper_eligible'] is False


def test_ledger_divergence_latches_and_reports_runtime_failure_without_accepting_bar(tmp_path, monkeypatch):
    from paper_trading.runner import PaperTradingRunner
    _, _, store, session, frame = setup(tmp_path, monkeypatch)
    original = PaperTradingRunner.process_bar
    def divergent(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self._ledger.cash -= 1.
        return result
    monkeypatch.setattr(PaperTradingRunner, 'process_bar', divergent)
    response = store.process(session, frame.iloc[:1], now=frame.index[0].to_pydatetime())
    assert response['status'] == 'RECONCILIATION_FAILURE'
    assert response['runtime_failure'] is True
    assert store.status(session)['accepted_bars'] == 0
    assert store.status(session)['suspended'] is True
