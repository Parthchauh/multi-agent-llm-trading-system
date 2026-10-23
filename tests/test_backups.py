import sqlite3

import pytest

from research_service.access import AccessStore
from research_service.backups import create_backup, restore_backup


def test_online_database_backup_restore_integrity_and_session_revocation(tmp_path):
    access = AccessStore(tmp_path/'access.sqlite')
    access.create_user('alpha', 'admin', 'backup-password-123456', 'admin')
    token = access.login('alpha', 'admin', 'backup-password-123456')
    artifact = tmp_path/'artifact.json'
    artifact.write_text('{"status":"INSUFFICIENT_EVIDENCE"}')
    backup = tmp_path/'backup'
    manifest = create_backup(backup, {'access.sqlite': access.path, 'artifact.json': artifact})
    restored = tmp_path/'restored'
    report = restore_backup(backup, restored)
    assert report['status'] == 'RESTORE_VERIFIED'
    assert report['sessions_revoked'] == 1
    assert (restored/'artifact.json').read_bytes() == artifact.read_bytes()
    with pytest.raises(PermissionError):
        AccessStore(restored/'access.sqlite').authenticate(token)
    assert access.authenticate(token).username == 'admin'
    with pytest.raises(FileExistsError):
        restore_backup(backup, restored)
    with sqlite3.connect(restored/'access.sqlite') as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert len(manifest['files']) == 2


@pytest.mark.parametrize('label', ['../escape', 'C:secret', 'CON', 'nul.db', '/root', 'trailing.'])
def test_malicious_backup_paths_rejected_before_creating_destination(tmp_path, label):
    artifact = tmp_path/'input.json'
    artifact.write_text('{}')
    with pytest.raises(ValueError):
        create_backup(tmp_path/'backup', {label: artifact})
    assert not (tmp_path/'backup').exists()


def test_tampered_backup_refuses_restore_before_creating_destination(tmp_path):
    artifact = tmp_path/'input.json'
    artifact.write_text('{}')
    backup = tmp_path/'backup'
    create_backup(backup, {'artifact.json': artifact})
    (backup/'artifact.json').write_text('{"corrupted":true}')
    with pytest.raises(ValueError, match='integrity'):
        restore_backup(backup, tmp_path/'restored')
    assert not (tmp_path/'restored').exists()
