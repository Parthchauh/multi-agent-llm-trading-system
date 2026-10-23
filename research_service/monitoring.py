"""Local request metrics and persistent alert inbox; no external notifications."""
from __future__ import annotations

import hashlib
import math
import sqlite3
import time
from contextlib import closing
from pathlib import Path


class ServiceMonitor:
    def __init__(self, path: Path, *, clock=time.time):
        self.path = Path(path)
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS request_metrics (
                    observed_at REAL NOT NULL, status INTEGER NOT NULL, latency_ms REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS metrics_time ON request_metrics(observed_at);
                CREATE TABLE IF NOT EXISTS operational_alerts (
                    incident_id TEXT PRIMARY KEY, reason TEXT NOT NULL,
                    first_seen REAL NOT NULL, last_seen REAL NOT NULL, acknowledged_at REAL);
            ''')

    def record(self, status: int, latency_ms: float) -> None:
        if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
            raise ValueError('Invalid HTTP status.')
        if isinstance(latency_ms, bool) or not math.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError('Invalid latency.')
        with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            db.execute('INSERT INTO request_metrics VALUES (?, ?, ?)',
                       (self.clock(), status, latency_ms))
            # Bounded retention; only numeric metadata is stored, never request content.
            db.execute('DELETE FROM request_metrics WHERE observed_at < ?', (self.clock()-7*86400,))

    def snapshot(self, *, window_seconds: int = 300, minimum_requests: int = 20,
                 maximum_error_rate: float = .05, maximum_p95_ms: float = 2000.) -> dict:
        if (isinstance(window_seconds, bool) or not isinstance(window_seconds, int)
                or not 1 <= window_seconds <= 86400
                or isinstance(minimum_requests, bool) or not isinstance(minimum_requests, int)
                or minimum_requests < 1):
            raise ValueError('Invalid observation window or minimum sample.')
        if (isinstance(maximum_error_rate, bool) or not math.isfinite(maximum_error_rate)
                or not 0 <= maximum_error_rate <= 1 or isinstance(maximum_p95_ms, bool)
                or not math.isfinite(maximum_p95_ms) or maximum_p95_ms <= 0):
            raise ValueError('Invalid alert thresholds.')
        now = self.clock()
        with closing(sqlite3.connect(self.path)) as db, db:
            rows = db.execute('SELECT status, latency_ms FROM request_metrics '
                              'WHERE observed_at >= ? AND observed_at <= ?',
                              (now-window_seconds, now)).fetchall()
            count = len(rows)
            error_rate = sum(status >= 500 for status, _ in rows)/count if count else None
            p95 = sorted(latency for _, latency in rows)[math.ceil(count*.95)-1] if count else None
            reasons = []
            if count >= minimum_requests and error_rate is not None and p95 is not None:
                if error_rate > maximum_error_rate:
                    reasons.append('HTTP_ERROR_RATE')
                if p95 > maximum_p95_ms:
                    reasons.append('HTTP_LATENCY_P95')
            for reason in reasons:
                identity = hashlib.sha256(f'{reason}:{int(now//window_seconds)}:{window_seconds}'.encode()).hexdigest()
                db.execute('INSERT INTO operational_alerts VALUES (?, ?, ?, ?, NULL) '
                           'ON CONFLICT(incident_id) DO UPDATE SET last_seen=excluded.last_seen',
                           (identity, reason, now, now))
            alerts = db.execute('SELECT incident_id, reason, first_seen, last_seen FROM operational_alerts '
                                'WHERE acknowledged_at IS NULL ORDER BY first_seen, incident_id').fetchall()
        return {'status': 'ALERT' if reasons else ('INSUFFICIENT_TRAFFIC' if count < minimum_requests else 'NO_THRESHOLD_BREACH'),
                'window_seconds': window_seconds, 'requests': count, 'server_error_rate': error_rate,
                'latency_p95_ms': p95, 'access_denials': sum(s in (401, 403) for s, _ in rows),
                'rate_limited': sum(s == 429 for s, _ in rows),
                'policy': {'minimum_requests': minimum_requests, 'maximum_error_rate': maximum_error_rate,
                           'maximum_p95_ms': maximum_p95_ms},
                'open_alerts': [dict(zip(('incident_id', 'reason', 'first_seen', 'last_seen'), row)) for row in alerts],
                'queue_depth': None, 'provider_failures': None, 'token_spend': None,  # nosec B105 - missing numeric spend, not a credential
                'external_delivery': 'NOT_CONFIGURED', 'paper_eligible': False,
                'limitations': ['Request observations do not establish uptime or production capacity.',
                                'This read API has no model/job queue; provider and spend metrics are unavailable.',
                                'A local operator must inspect or export this alert inbox.']}

    def acknowledge(self, incident_id: str) -> None:
        with closing(sqlite3.connect(self.path)) as db, db:
            result = db.execute('UPDATE operational_alerts SET acknowledged_at=COALESCE(acknowledged_at, ?) '
                                'WHERE incident_id=?', (self.clock(), incident_id))
            if not result.rowcount:
                raise KeyError('Unknown operational incident.')
