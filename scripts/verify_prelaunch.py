"""Run recorded verification with this interpreter in an already isolated environment."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

LINT_PATHS = [
    'research_validation/campaign.py', 'research_validation/uncertainty.py',
    'research_validation/walk_forward.py', 'data/snapshots.py', 'data/provenance.py', 'evaluation/registry.py',
    'data/adjustments.py', 'evaluation/ablation.py',
    'paper_trading/operations.py', 'paper_trading/runner.py', 'research_service',
    'paper_trading/drift.py',
    'cli/reproducibility.py', 'research_dashboard/views.py',
    'tradingagents/agents/telemetry.py', 'scripts/release_gate.py', 'scripts/verify_prelaunch.py']
TYPE_PATHS = ['research_service', 'research_validation/campaign.py',
    'research_validation/uncertainty.py', 'data/snapshots.py', 'data/provenance.py', 'evaluation/registry.py',
    'data/adjustments.py', 'evaluation/ablation.py',
    'paper_trading/operations.py', 'cli/reproducibility.py']
TYPE_PATHS.append('paper_trading/drift.py')
LINT_PATHS.append('research_validation/study.py')
TYPE_PATHS.append('research_validation/study.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('verification-output'))
    parser.add_argument('--online-audit', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    python = sys.executable
    commands = {
        'secret-scan': [python, '-m', 'scripts.check_secrets'],
        'lint': [python, '-m', 'ruff', 'check', *LINT_PATHS],
        'types': [python, '-m', 'mypy', '--follow-imports=skip', '--ignore-missing-imports',
                  '--check-untyped-defs', *TYPE_PATHS],
        'security': [python, '-m', 'bandit', '-r', *TYPE_PATHS, '-f', 'json'],
        'tests': [python, '-m', 'pytest', '-m', 'not live', '-q',
                  '--basetemp='+str(args.output/'test-tmp'), '-p', 'no:cacheprovider'],
        'fixture': [python, '-m', 'tests.prelaunch_fixture', '--output', str(args.output/'fixture')],
        'promotion-disabled': [python, '-m', 'scripts.release_gate', '--assert-disabled'],
        'console-help': [str(Path(python).parent/('trading-research.exe' if sys.platform == 'win32' else 'trading-research')), '--help'],
        'sbom': [python, '-m', 'cyclonedx_py', 'environment', '--of', 'JSON',
                 '--output-file', str(args.output/'sbom.json')],
    }
    if args.online_audit:
        commands['dependency-audit'] = [python, '-m', 'pip_audit', '--local', '--format', 'json']
    results = {}
    for name, command in commands.items():
        started = time.perf_counter()
        with (args.output/(name+'.txt')).open('wb') as stream:
            try:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
                code = result.returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                stream.write(type(exc).__name__.encode())
                code = 2
        results[name] = {'command': command, 'exit_code': code,
                         'seconds': round(time.perf_counter()-started, 3)}
        (args.output/'commands.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(f'{name}: exit {code}', flush=True)
    raise SystemExit(1 if any(item['exit_code'] for item in results.values()) else 0)


if __name__ == '__main__':
    main()
