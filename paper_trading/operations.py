"""Transactional local paper simulation controls. No market or broker connection.

Each accepted bar atomically stores its checkpoint and response. Lost responses
can be retried without adding fills. A persisted suspension permits exits but
blocks entries. It can only be cleared by creating a separately reviewed session.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtesting.models import BacktestConfig
from data.schema import validate_ohlcv
from data.snapshots import canonical_bytes
from evaluation.registry import FrozenCandidateRegistry
from paper_trading.runner import PaperTradingRunner
from risk.models import RiskPolicy
from strategies.schema import StrategySchema


class PaperSimulationStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS simulations "
                       "(session_id TEXT PRIMARY KEY, payload TEXT NOT NULL, sha256 TEXT NOT NULL)")

    @staticmethod
    def _read(db, session_id):
        row = db.execute("SELECT payload, sha256 FROM simulations WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            raise KeyError("Simulation session not found.")
        if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
            raise ValueError("Simulation state integrity check failed.")
        return json.loads(row[0])

    @staticmethod
    def _write(db, session_id, state):
        content = canonical_bytes(state)
        db.execute("UPDATE simulations SET payload=?, sha256=? WHERE session_id=?",
                   (content.decode(), hashlib.sha256(content).hexdigest(), session_id))

    def create(self, *, registry: FrozenCandidateRegistry, candidate_id: str, symbol: str,
               sessions: pd.DatetimeIndex, maximum_age_seconds: int = 259200) -> str:
        candidate = registry.get(candidate_id)
        sessions = pd.DatetimeIndex(sessions)
        if (sessions.empty or sessions.tz is None or sessions.hasnans or sessions.has_duplicates
                or not sessions.is_monotonic_increasing):
            raise ValueError("Simulation requires an ordered timezone-aware session calendar.")
        if isinstance(maximum_age_seconds, bool) or not isinstance(maximum_age_seconds, int) or not 1 <= maximum_age_seconds <= 604800:
            raise ValueError("Maximum bar age must be an integer in [1, 604800] seconds.")
        if not symbol.strip() or len(symbol) > 32:
            raise ValueError("A symbol of at most 32 characters is required.")
        state = {"schema_version": 1, "mode": "LOCAL_SIMULATION_ONLY", "candidate": candidate,
                 "symbol": symbol.upper(), "sessions": [s.isoformat() for s in sessions.tz_convert('UTC')],
                 "maximum_age_seconds": maximum_age_seconds, "checkpoint": None,
                 "last_request_sha256": None, "last_response": None, "suspended": False,
                 "incidents": [], "peak_equity": candidate['config']['initial_capital'],
                 "accepted_bars": 0, "observed_sessions": [], "paper_eligible": False,
                 "equity_history": [candidate['config']['initial_capital']],
                 "created_at": datetime.now(timezone.utc).isoformat()}
        session_id = uuid.uuid4().hex
        content = canonical_bytes(state)
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO simulations VALUES (?, ?, ?)",
                       (session_id, content.decode(), hashlib.sha256(content).hexdigest()))
        return session_id

    def status(self, session_id):
        with sqlite3.connect(self.path) as db:
            state = self._read(db, session_id)
        return {key: state[key] for key in ('mode', 'suspended', 'incidents', 'accepted_bars',
                                           'observed_sessions', 'paper_eligible', 'last_response')}

    @staticmethod
    def _incident(state, reason):
        incident = {"incident_id": uuid.uuid4().hex, "reason": reason,
                    "recorded_at": datetime.now(timezone.utc).isoformat()}
        state['suspended'] = True
        state['incidents'].append(incident)
        return incident

    def kill(self, session_id):
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            state = self._read(db, session_id)
            if not state['suspended']:
                self._incident(state, 'OPERATOR_KILL_SWITCH')
                self._write(db, session_id, state)
        return self.status(session_id)

    def process(self, session_id: str, history: pd.DataFrame, *, now: datetime) -> dict:
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("BEGIN IMMEDIATE")
            state = self._read(db, session_id)
            try:
                clean = validate_ohlcv(history)
                if now.tzinfo is None or clean.index.tz is None:
                    raise ValueError("TIMEZONE_REQUIRED")
                calendar = pd.DatetimeIndex(pd.to_datetime(state['sessions'], utc=True))
                if not clean.index.tz_convert('UTC').equals(calendar[:len(clean)]):
                    raise ValueError("SESSION_CALENDAR_MISMATCH")
                age = (pd.Timestamp(now)-clean.index[-1]).total_seconds()
                if age < 0 or age > state['maximum_age_seconds']:
                    raise ValueError("FUTURE_OR_STALE_BAR")
                request_sha = hashlib.sha256(canonical_bytes({
                    "index": [stamp.isoformat() for stamp in clean.index],
                    "columns": list(clean.columns), "values": clean.values.tolist()})).hexdigest()
                if request_sha == state['last_request_sha256']:
                    return {**state['last_response'], "duplicate": True, "suspended": state['suspended']}
                checkpoint = state['checkpoint']
                runner = (PaperTradingRunner.from_checkpoint_payload(checkpoint) if checkpoint else
                          PaperTradingRunner(StrategySchema.model_validate(state['candidate']['strategy']),
                              config=BacktestConfig.model_validate(state['candidate']['config'])))
                if checkpoint and len(clean) != len(checkpoint['history']['index'])+1:
                    raise ValueError("BAR_APPEND_REQUIRED")
                _, snapshot = runner.process_bar(state['symbol'], clean,
                    entries_enabled=not state['suspended'], force_exit=state['suspended'])
            except (ValueError, TypeError) as exc:
                # No malformed data or exception text is logged. The kill latch survives restart.
                incident = self._incident(state, 'DATA_OR_RECONCILIATION_REJECTED')
                self._write(db, session_id, state)
                return {"status": "REJECTED_DATA", "runtime_failure": False,
                        "error_type": type(exc).__name__, "incident_id": incident['incident_id'],
                        "suspended": True, "paper_eligible": False}
            paper_position = runner.portfolio.get_open_position(state['symbol'])
            ledger_position = runner._ledger.position
            if (not math.isclose(runner.portfolio.cash, runner._ledger.cash, rel_tol=0., abs_tol=1e-8)
                    or bool(paper_position) != bool(ledger_position)
                    or (paper_position is not None and paper_position.quantity != ledger_position.quantity)):
                incident = self._incident(state, 'RECONCILIATION_FAILURE')
                self._write(db, session_id, state)
                return {'status': 'RECONCILIATION_FAILURE', 'runtime_failure': True,
                        'incident_id': incident['incident_id'], 'suspended': True, 'paper_eligible': False}
            state['peak_equity'] = max(state['peak_equity'], snapshot.total_equity)
            drawdown = (1-snapshot.total_equity/state['peak_equity'])*100
            policy = RiskPolicy.model_validate(state['candidate']['risk_policy'])
            loss = max([0.] + [-p.realized_pnl for p in runner.portfolio.closed_positions])
            loss_pct = loss/runner.config.initial_capital*100
            turnover_pct = sum(f.fill_price*f.quantity for f in runner.portfolio.fills)/runner.config.initial_capital*100
            exposure_pct = snapshot.holdings_value/snapshot.total_equity*100 if snapshot.total_equity else 100.
            if (drawdown > policy.max_drawdown_pct or loss_pct > policy.max_loss_per_trade_pct
                    or turnover_pct > policy.max_turnover_pct
                    or exposure_pct > policy.max_portfolio_exposure_pct):
                if not state['suspended']:
                    self._incident(state, 'RISK_LIMIT_BREACH')
            response = {"status": "SIMULATED", "duplicate": False, "runtime_failure": False,
                        "snapshot": snapshot.model_dump(mode='json'), "drawdown_pct": drawdown,
                        "max_loss_per_trade_pct": loss_pct,
                        "turnover_pct": turnover_pct, "exposure_pct": exposure_pct,
                        "suspended": state['suspended'], "paper_eligible": False,
                        "fills": [f.model_dump(mode='json') for f in runner.bar_fills]}
            state.update(checkpoint=runner.checkpoint_payload(), last_request_sha256=request_sha,
                         last_response=response, accepted_bars=state['accepted_bars']+1)
            state['observed_sessions'].append(clean.index[-1].isoformat())
            state.setdefault('equity_history', []).append(snapshot.total_equity)
            self._write(db, session_id, state)
            return response

    def monitor_drift(self, session_id, reference_bundle, *, policy=None):
        from paper_trading.drift import DriftPolicy, measure_drift
        policy = policy or DriftPolicy()
        bundle = dict(reference_bundle)
        identity = bundle.pop('bundle_sha256')
        if hashlib.sha256(canonical_bytes(bundle)).hexdigest() != identity:
            raise ValueError('Drift reference bundle integrity check failed.')
        with sqlite3.connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            state = self._read(db, session_id)
            frozen = state['candidate']
            if (frozen['evidence'].get('replay_bundle') != identity
                    or bundle['strategy'] != frozen['strategy'] or bundle['config'] != frozen['config']):
                raise ValueError('Drift reference must match the immutable candidate evidence and configuration.')
            previous = state.get('drift_policy')
            if previous is not None and previous != policy.model_dump():
                raise ValueError('Drift policy is frozen for this session; create a new reviewed session.')
            state['drift_policy'] = policy.model_dump()
            result = bundle['result']
            reference = [result['initial_capital']]+[point['equity'] for point in result['equity_curve']]
            fills = state['checkpoint']['fills'] if state['checkpoint'] else []
            trades = sum(fill['signal_type'] == 'EXIT_LONG' for fill in fills)
            drift = measure_drift(state.get('equity_history', []), reference,
                                   completed_trades=trades, policy=policy)
            drift['mode'] = 'LOCAL_SIMULATION_ONLY'
            if drift['status'] == 'DRIFT_THRESHOLD_BREACH' and not state['suspended']:
                self._incident(state, 'DRIFT_THRESHOLD_BREACH')
            state['drift'] = drift
            self._write(db, session_id, state)
        return drift
