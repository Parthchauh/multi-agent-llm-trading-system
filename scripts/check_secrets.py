"""Fail on secret-scanner findings absent from the explicitly reviewed baseline."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

EXCLUDED = (r'(^|[\\/])(\.git|\.venv|\.tools|\.test_tmp[^\\/]*|\.mypy_cache|\.ruff_cache|'
            r'__pycache__|results|build|dist|verification-output)([\\/]|$)'
            r'|\.sqlite|\.pptx$|\.lock$|\.secrets\.baseline$'
            r'|docs[\\/]prelaunch[\\/](acceptance-evidence|completion-evidence)[\\/]')


def source_files():
    root = Path.cwd()
    listed = subprocess.check_output(['git', '-c', f'safe.directory={root.as_posix()}',
        'ls-files', '--cached', '--others', '--exclude-standard', '-z'], stderr=subprocess.DEVNULL)
    names = listed.decode('utf-8').split('\0')
    text_suffixes = {'.py', '.toml', '.ini', '.cfg', '.json', '.yaml', '.yml', '.html',
                     '.js', '.css', '.md', '.txt', '.ps1', '.sh'}
    return sorted(set(name for name in names if name and not re.search(EXCLUDED, name)
        and Path(name).is_file() and (Path(name).suffix in text_suffixes
            or Path(name).name in {'Dockerfile', 'NOTICE', 'LICENSE'}
            or Path(name).name.startswith('.env'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('.secrets.baseline'))
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    reviewed = {(path.replace('\\', '/'), finding['type'], finding['hashed_secret'])
                for path, findings in baseline['results'].items() for finding in findings
                if finding.get('is_secret') is False}
    files = source_files()
    if not files:
        print(json.dumps({'status': 'SCAN_RUNTIME_FAILURE', 'reason': 'NO_SOURCE_FILES'}))
        return 2
    scan = {'results': {}}
    for start in range(0, len(files), 100):
        completed = subprocess.run([sys.executable, '-m', 'detect_secrets', 'scan',
            '--no-verify', *files[start:start+100]], capture_output=True, timeout=120)
        if completed.returncode:
            print(json.dumps({'status': 'SCAN_RUNTIME_FAILURE', 'exit_code': completed.returncode}))
            return 2
        scan['results'].update(json.loads(completed.stdout)['results'])
    unreviewed = []
    count = 0
    for path, findings in scan['results'].items():
        for finding in findings:
            count += 1
            if (path.replace('\\', '/'), finding['type'], finding['hashed_secret']) not in reviewed:
                unreviewed.append({'file': path, 'line': finding['line_number'], 'type': finding['type']})
    print(json.dumps({'status': 'UNREVIEWED_FINDINGS' if unreviewed else 'NO_UNREVIEWED_FINDINGS',
        'source_files': len(files), 'findings': count, 'unreviewed': unreviewed,
        'limitations': 'Pattern scanning is not independent security review.'}, indent=2))
    return 1 if unreviewed else 0


if __name__ == '__main__':
    raise SystemExit(main())
