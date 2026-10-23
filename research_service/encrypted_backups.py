"""Password-encrypted local backup envelope; no cloud or key escrow integration."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from data.snapshots import canonical_bytes
from research_service.backups import _name, restore_backup

MAGIC = b'TRADING-RESEARCH-BACKUP-V1\x00'
MAX_BYTES = 128 * 1024 * 1024


def _key(password: str, salt: bytes) -> bytes:
    if not isinstance(password, str) or not 16 <= len(password) <= 1024:
        raise ValueError('Use a passphrase of 16 to 1024 characters.')
    return Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(password.encode())


def encrypt_backup(source: Path, destination: Path, password: str) -> dict:
    """Encrypt a completed named-file backup. Never delete or overwrite inputs."""
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    names = ['manifest.json'] + [_name(entry['name']) for entry in manifest['files']]
    if len(names) != len(set(n.casefold() for n in names)) or len(names) > 1001:
        raise ValueError('Invalid backup file set.')
    files = {}
    total = 0
    for name in names:
        path = source/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Backup inputs must be regular files.')
        total += path.stat().st_size
        if total > MAX_BYTES//2:
            raise ValueError('Backup exceeds the bounded in-memory envelope limit.')
        files[name] = base64.b64encode(path.read_bytes()).decode('ascii')
    salt, nonce = os.urandom(16), os.urandom(12)
    header = MAGIC+salt+nonce
    ciphertext = AESGCM(_key(password, salt)).encrypt(nonce, canonical_bytes(files), header)
    envelope = header+ciphertext
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as handle:
        handle.write(envelope)
    return {'status': 'ENCRYPTED_LOCAL_BACKUP', 'sha256': hashlib.sha256(envelope).hexdigest(),
            'bytes': len(envelope), 'algorithm': 'AES-256-GCM', 'kdf': 'scrypt-N32768-r8-p1',
            'off_host_copy_verified': False, 'plaintext_source_retained': True}


def restore_encrypted_backup(source: Path, destination: Path, password: str) -> dict:
    if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_BYTES:
        raise ValueError('Invalid or oversized encrypted backup.')
    envelope = source.read_bytes()
    size = len(MAGIC)+28
    if not envelope.startswith(MAGIC) or len(envelope) < size+16:
        raise ValueError('Unsupported encrypted backup envelope.')
    salt, nonce = envelope[len(MAGIC):len(MAGIC)+16], envelope[len(MAGIC)+16:size]
    try:
        plain = AESGCM(_key(password, salt)).decrypt(nonce, envelope[size:], envelope[:size])
    except InvalidTag as exc:
        raise ValueError('Backup authentication failed; wrong password or altered bytes.') from exc
    files = json.loads(plain)
    if not isinstance(files, dict) or not 1 <= len(files) <= 1001:
        raise ValueError('Invalid decrypted file set.')
    if len({name.casefold() for name in files}) != len(files):
        raise ValueError('Duplicate decrypted filenames.')
    # Authenticate and validate names before touching the destination.
    contents = {_name(name): base64.b64decode(value, validate=True) for name, value in files.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='restore-staging-', dir=destination.parent) as staging:
        root = Path(staging)
        for name, content in contents.items():
            (root/name).write_bytes(content)
        report = restore_backup(root, destination)
    return {**report, 'encrypted_envelope_verified': True,
            'limitations': ['Temporary plaintext staging is removed but secure disk erasure is not guaranteed.',
                            'Off-host storage, passphrase custody and scheduled drills require an operator.']}
