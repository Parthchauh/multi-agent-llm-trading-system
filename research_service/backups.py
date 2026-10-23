"""Verified local database/artifact snapshots and restore into a new directory."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from data.snapshots import canonical_bytes


def _name(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,120}', name) or name.endswith('.'):
        raise ValueError('Backup labels must be plain safe filenames.')
    if name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        raise ValueError('Reserved backup filename.')
    return name


def create_backup(destination: Path, sources: dict[str, Path]) -> dict:
    if not sources or len(sources) > 1000 or len({_name(n).casefold() for n in sources}) != len(sources):
        raise ValueError('Require 1 to 1000 distinct named backup inputs.')
    if 'manifest.json' in {n.casefold() for n in sources}:
        raise ValueError('manifest.json is reserved.')
    for source in sources.values():
        if not Path(source).is_file() or Path(source).is_symlink():
            raise ValueError('Backup sources must be explicit regular files.')
    destination.mkdir(parents=True, exist_ok=False)
    entries = []
    for label, source in sources.items():
        target = destination/label
        source = Path(source)
        sqlite = source.suffix.lower() in {'.sqlite', '.db', '.sqlite3'}
        if sqlite:
            with sqlite3.connect(source.resolve().as_uri()+'?mode=ro', uri=True) as original:
                with sqlite3.connect(target) as copied:
                    original.backup(copied)
                    if copied.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                        raise ValueError('Database backup integrity check failed.')
        else:
            target.write_bytes(source.read_bytes())
        entries.append({'name': label, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                        'bytes': target.stat().st_size, 'sqlite': sqlite})
    manifest = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                'consistency': 'Each SQLite database is atomic; quiesce writers for a cross-database snapshot.',
                'files': entries}
    manifest['sha256'] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    (destination/'manifest.json').write_bytes(canonical_bytes(manifest))
    return manifest


def restore_backup(source: Path, destination: Path) -> dict:
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    digest = manifest.pop('sha256')
    if hashlib.sha256(canonical_bytes(manifest)).hexdigest() != digest or manifest.get('schema_version') != 1:
        raise ValueError('Backup manifest integrity check failed.')
    entries = manifest['files']
    if not entries or len(entries) > 1000 or len({_name(e['name']).casefold() for e in entries}) != len(entries):
        raise ValueError('Backup has invalid or duplicate filenames.')
    contents = {}
    for entry in entries:
        path = source/_name(entry['name'])
        if path.is_symlink():
            raise ValueError('Backup symlinks are forbidden.')
        content = path.read_bytes()
        if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError('Backup file integrity check failed.')
        if entry['sqlite']:
            with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
                if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise ValueError('Backup database integrity check failed.')
        contents[entry['name']] = content
    # Every input is verified before creating a new restore target. Existing work is never replaced.
    destination.mkdir(parents=True, exist_ok=False)
    revoked = 0
    for entry in entries:
        path = destination/entry['name']
        path.write_bytes(contents[entry['name']])
        if entry['sqlite']:
            with sqlite3.connect(path) as db:
                if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'").fetchone():
                    revoked += db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]
                    db.execute('DELETE FROM sessions')
    return {'status': 'RESTORE_VERIFIED', 'backup_sha256': digest, 'files': len(entries),
            'sessions_revoked': revoked, 'paper_eligible': False}
