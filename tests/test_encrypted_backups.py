import secrets
import sqlite3

import pytest

from research_service.backups import create_backup
from research_service.encrypted_backups import encrypt_backup, restore_encrypted_backup


def test_encrypted_roundtrip_revokes_sessions_and_preserves_source(tmp_path):
    database = tmp_path/'access.sqlite'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE sessions (token_hash TEXT)')
        db.execute("INSERT INTO sessions VALUES ('fixture-session')")
    original = tmp_path/'backup'
    create_backup(original, {'access.sqlite': database})
    password = secrets.token_urlsafe(32)
    envelope = tmp_path/'backup.enc'
    report = encrypt_backup(original, envelope, password)
    assert report['off_host_copy_verified'] is False
    assert b'fixture-session' not in envelope.read_bytes()
    assert password.encode() not in envelope.read_bytes()
    destination = tmp_path/'restored'
    restored = restore_encrypted_backup(envelope, destination, password)
    assert restored['encrypted_envelope_verified'] is True
    with sqlite3.connect(destination/'access.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0] == 0
    assert (original/'access.sqlite').exists()
    with pytest.raises(FileExistsError):
        encrypt_backup(original, envelope, password)
    with pytest.raises(FileExistsError):
        restore_encrypted_backup(envelope, destination, password)


def test_wrong_password_and_tampering_create_no_restore_directory(tmp_path):
    artifact = tmp_path/'artifact.txt'
    artifact.write_text('recorded fixture')
    original = tmp_path/'backup'
    create_backup(original, {'artifact.txt': artifact})
    password = secrets.token_urlsafe(32)
    envelope = tmp_path/'backup.enc'
    encrypt_backup(original, envelope, password)
    destination = tmp_path/'restored'
    with pytest.raises(ValueError, match='authentication'):
        restore_encrypted_backup(envelope, destination, secrets.token_urlsafe(32))
    assert not destination.exists()
    content = envelope.read_bytes()
    envelope.write_bytes(content[:-1]+bytes([content[-1]^1]))
    with pytest.raises(ValueError, match='authentication'):
        restore_encrypted_backup(envelope, destination, password)
    assert not destination.exists()
