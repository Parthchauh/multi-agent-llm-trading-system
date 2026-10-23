import json
import sqlite3

import pytest

from research_service.access import AccessStore

PASSWORD = 'recorded-password-123456789'


def test_roles_tenant_isolation_and_revoke_survive_restart(tmp_path):
    def clock():
        return 1000.
    access = AccessStore(tmp_path/'access.sqlite', clock=clock)
    access.claim_resource('alpha-report', 'alpha')
    access.claim_resource('beta-report', 'beta')
    for role in access.ROLES:
        access.create_user('alpha', role, PASSWORD, role)
        token = access.login('alpha', role, PASSWORD)
        restarted = AccessStore(access.path, clock=clock)
        assert restarted.authorize(token, 'alpha-report').role == role
        for action, allowed in access.ACTIONS.items():
            if role in allowed:
                assert restarted.authorize(token, 'alpha-report', action).tenant == 'alpha'
            else:
                with pytest.raises(PermissionError):
                    restarted.authorize(token, 'alpha-report', action)
        for resource in ('beta-report', 'unknown-report'):
            with pytest.raises(PermissionError):
                restarted.authorize(token, resource)
        restarted.revoke(token)
        with pytest.raises(PermissionError):
            access.authenticate(token)
    with sqlite3.connect(access.path) as db:
        audits = db.execute('SELECT * FROM access_audit').fetchall()
        assert any(row[-1] == 'DENIED' for row in audits)
        assert PASSWORD not in json.dumps(audits)
        assert token not in json.dumps(audits)
        assert db.execute('SELECT password_hash FROM users').fetchone()[0] != PASSWORD.encode()


def test_expiry_disabled_users_and_persistent_login_rate_limits(tmp_path):
    now = [1000.]
    access = AccessStore(tmp_path/'access.sqlite', clock=lambda: now[0])
    access.create_user('alpha', 'user', PASSWORD, 'researcher')
    token = access.login('alpha', 'user', PASSWORD, ttl_seconds=10)
    now[0] += 10
    with pytest.raises(PermissionError, match='INVALID'):
        access.authenticate(token)
    now[0] += 60
    token = access.login('alpha', 'user', PASSWORD)
    with sqlite3.connect(access.path) as db:
        db.execute('UPDATE users SET enabled=0')
    with pytest.raises(PermissionError):
        access.authenticate(token)
    for _ in range(4):
        with pytest.raises(PermissionError, match='CREDENTIALS'):
            access.login('alpha', 'user', 'wrong')
    with pytest.raises(PermissionError, match='RATE_LIMITED'):
        AccessStore(access.path, clock=lambda: now[0]).login('alpha', 'user', PASSWORD)


def test_ownership_cannot_be_reassigned_and_unprovisioned_sessions_fail(tmp_path):
    access = AccessStore(tmp_path/'access.sqlite')
    access.claim_resource('report', 'alpha')
    with pytest.raises(sqlite3.IntegrityError):
        access.claim_resource('report', 'beta')
    with pytest.raises(PermissionError):
        access.authorize('invalid', 'report')
    with pytest.raises(ValueError):
        access.create_user('a', 'u', 'short', 'admin')
    with pytest.raises(ValueError):
        access.create_user('a', 'u', PASSWORD, 'superuser')
