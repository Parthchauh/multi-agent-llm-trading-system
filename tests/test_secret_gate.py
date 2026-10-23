import json
import subprocess
import sys

from scripts import check_secrets


def test_secret_gate_is_offline_and_rejects_unreviewed_findings(monkeypatch, tmp_path, capsys):
    baseline = tmp_path/'baseline.json'
    known = {'type': 'Secret Keyword', 'hashed_secret': 'a'*64, 'line_number': 1, 'is_secret': False}
    baseline.write_text(json.dumps({'results': {'fixture.py': [known]}}))
    monkeypatch.setattr(sys, 'argv', ['check_secrets', '--baseline', str(baseline)])
    monkeypatch.setattr(check_secrets, 'source_files', lambda: ['fixture.py'])
    findings = [known]
    def scan(command, **kwargs):
        assert '--no-verify' in command
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({'results': {'fixture.py': findings}}).encode())
    monkeypatch.setattr(subprocess, 'run', scan)
    assert check_secrets.main() == 0
    assert 'NO_UNREVIEWED_FINDINGS' in capsys.readouterr().out
    findings.append({**known, 'hashed_secret': 'b'*64})
    assert check_secrets.main() == 1
    output = capsys.readouterr().out
    assert 'UNREVIEWED_FINDINGS' in output
    assert 'b'*64 not in output
