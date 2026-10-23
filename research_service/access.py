"""Persistent opaque sessions, tenant-scoped RBAC and audit for research reads."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Principal:
    tenant: str
    username: str
    role: str


class AccessStore:
    ROLES = {'admin', 'researcher', 'reviewer', 'read_only'}
    ACTIONS = {'read': ROLES, 'research': {'admin', 'researcher'}, 'review': {'admin', 'reviewer'},
               'admin': {'admin'}}

    def __init__(self, path: Path, *, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        with self._db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    tenant TEXT, username TEXT, role TEXT NOT NULL, salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (tenant, username));
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, tenant TEXT, username TEXT, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS resources (resource_id TEXT PRIMARY KEY, tenant TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS rate_limits (bucket TEXT PRIMARY KEY, window INTEGER, count INTEGER);
                CREATE TABLE IF NOT EXISTS access_audit (
                    timestamp REAL, tenant TEXT, username TEXT, action TEXT, resource_id TEXT, outcome TEXT);
            ''')

    def _db(self):
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def _password(password, salt):
        return hashlib.scrypt(password.encode('utf-8'), salt=salt, n=16384, r=8, p=1, dklen=32)

    def create_user(self, tenant: str, username: str, password: str, role: str):
        """Local provisioning only. No unauthenticated HTTP registration endpoint."""
        if role not in self.ROLES or not tenant or not username or max(len(tenant), len(username)) > 128:
            raise ValueError('Invalid local user identity or role.')
        if not 16 <= len(password) <= 1024:
            raise ValueError('Password must contain 16 to 1024 characters.')
        salt = secrets.token_bytes(16)
        with self._db() as db:
            db.execute('INSERT INTO users (tenant, username, role, salt, password_hash) VALUES (?, ?, ?, ?, ?)',
                       (tenant, username, role, salt, self._password(password, salt)))
        self.audit(Principal(tenant, username, role), 'local_user_provisioning', None, 'ALLOWED')

    def claim_resource(self, resource_id: str, tenant: str):
        """Local administrative import. Ownership cannot be reassigned silently."""
        if not resource_id or not tenant:
            raise ValueError('Resource and tenant are required.')
        with self._db() as db:
            db.execute('INSERT INTO resources VALUES (?, ?)', (resource_id, tenant))
        self.audit(Principal(tenant, 'local_administrator', 'admin'), 'local_resource_assignment', resource_id, 'ALLOWED')

    def audit(self, principal, action, resource_id, outcome):
        with self._db() as db:
            db.execute('INSERT INTO access_audit VALUES (?, ?, ?, ?, ?, ?)',
                       (self.clock(), principal.tenant if principal else None,
                        principal.username if principal else None, action, resource_id, outcome))

    def rate_limit(self, bucket: str, *, maximum: int, seconds=60):
        key = hashlib.sha256(bucket.encode()).hexdigest()
        window = int(self.clock()//seconds)
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT window, count FROM rate_limits WHERE bucket=?', (key,)).fetchone()
            count = row[1]+1 if row and row[0] == window else 1
            db.execute('INSERT OR REPLACE INTO rate_limits VALUES (?, ?, ?)', (key, window, count))
        if count > maximum:
            raise PermissionError('RATE_LIMITED')

    def login(self, tenant: str, username: str, password: str, *, ttl_seconds=900):
        if any(not isinstance(v, str) for v in (tenant, username, password)) or max(len(tenant), len(username)) > 128 or len(password) > 1024:
            raise PermissionError('INVALID_CREDENTIALS')
        self.rate_limit(f'login:{tenant}:{username}', maximum=5)
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or not 1 <= ttl_seconds <= 3600:
            raise ValueError('Session TTL must be in [1, 3600] seconds.')
        with self._db() as db:
            row = db.execute('SELECT role, salt, password_hash, enabled FROM users WHERE tenant=? AND username=?',
                             (tenant, username)).fetchone()
            salt = row[1] if row else b'not-a-real-user!!'
            check = self._password(password, salt)
            if row is None or not row[3] or not hmac.compare_digest(check, row[2]):
                self.audit(None, 'login', None, 'DENIED')
                raise PermissionError('INVALID_CREDENTIALS')
            token = secrets.token_urlsafe(32)
            db.execute('INSERT INTO sessions VALUES (?, ?, ?, ?)',
                       (hashlib.sha256(token.encode()).hexdigest(), tenant, username, self.clock()+ttl_seconds))
        self.audit(Principal(tenant, username, row[0]), 'login', None, 'ALLOWED')
        return token

    def authenticate(self, token):
        if not isinstance(token, str) or not 32 <= len(token) <= 128:
            raise PermissionError('SESSION_REQUIRED')
        with self._db() as db:
            row = db.execute('''SELECT s.tenant, s.username, u.role, s.expires, u.enabled FROM sessions s
                JOIN users u ON s.tenant=u.tenant AND s.username=u.username WHERE token_hash=?''',
                             (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        if not row or row[3] <= self.clock() or not row[4]:
            raise PermissionError('SESSION_INVALID')
        return Principal(*row[:3])

    def revoke(self, token):
        with self._db() as db:
            db.execute('DELETE FROM sessions WHERE token_hash=?', (hashlib.sha256(token.encode()).hexdigest(),))

    def authorize(self, token, resource_id, action='read'):
        principal = self.authenticate(token)
        self.rate_limit(f'access:{principal.tenant}:{principal.username}', maximum=120)
        with self._db() as db:
            owner = db.execute('SELECT tenant FROM resources WHERE resource_id=?', (resource_id,)).fetchone()
        allowed = (owner is not None and owner[0] == principal.tenant
                   and principal.role in self.ACTIONS.get(action, set()))
        self.audit(principal, action, resource_id, 'ALLOWED' if allowed else 'DENIED')
        if not allowed:
            raise PermissionError('RESOURCE_OR_ROLE_DENIED')
        return principal
