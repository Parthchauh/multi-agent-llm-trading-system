"""Run with clean-env Python -I; prove installed imports and commands outside checkout."""
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import cli.research
import data.provenance
import data.snapshots
import paper_trading.operations
import scripts.run_research_e2e

root = Path(__file__).resolve().parents[3]
outside = root/'.tools/project-completion'/('installed-outside-'+uuid.uuid4().hex[:8])
outside.mkdir(parents=True, exist_ok=False)
evidence = root/'docs/prelaunch/completion-evidence'
fixture = root/'.tools/project-completion/verified-fixture-final'
modules = (cli.research, data.provenance, data.snapshots, paper_trading.operations, scripts.run_research_e2e)
sources = {module.__name__: str(Path(module.__file__).resolve()) for module in modules}
assert all(Path(path).is_relative_to(Path(sys.prefix)) for path in sources.values()), sources
results = {'python': sys.executable, 'cwd': str(outside), 'installed_modules': sources,
           'module_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                             for name, path in sources.items()}, 'commands': {}}
installed_environment = data.provenance.environment_manifest(Path(cli.research.__file__).parents[1])
assert installed_environment['git_commit'] is None
assert installed_environment['working_tree_code_sha256'] is None
assert len(installed_environment['installed_project_code_sha256']) == 64
results['installed_environment'] = installed_environment
environment = dict(os.environ)
environment.pop('PYTHONPATH', None)
environment.pop('PYTHONHOME', None)
environment['PYTHONIOENCODING'] = 'utf-8'


def execute(name, command):
    completed = subprocess.run(command, cwd=outside, env=environment, capture_output=True, timeout=300)
    (evidence/(name+'.txt')).write_bytes(completed.stdout+completed.stderr)
    results['commands'][name] = {'command': [str(part) for part in command], 'exit_code': completed.returncode}
    assert completed.returncode == 0, (name, completed.returncode)
    return completed.stdout.decode('utf-8')


executable = Path(sys.executable).parent
canonical = executable/'trading-research.exe'
for command in ('trading-research', 'tradingagents', 'tradingagents-legacy'):
    execute('installed-'+command, [executable/(command+'.exe'), '--help'])
for command in ('snapshot-dataset', 'fold-research', 'freeze-candidate', 'simulate-paper',
                'monitor-simulation', 'serve-evidence-api', 'backup-state', 'restore-state'):
    execute('installed-help-'+command, [canonical, command, '--help'])
execute('installed-fold-research', [canonical, 'fold-research', fixture/'development-prices.csv',
    fixture/'campaign-config.json', fixture/'recorded-responses.json', outside/'campaign.json'])
assert json.loads((outside/'campaign.json').read_text())['paper_eligible'] is False
replayed = json.loads(execute('installed-replay', [canonical, 'verify-replay', fixture/'replay.json',
    '--store', fixture/'datasets']))
assert replayed['status'] == 'REPLAY_MATCH' and replayed['paper_eligible'] is False
execute('installed-snapshot', [canonical, 'snapshot-dataset', fixture/'development-prices.csv',
    fixture/'dataset-identity.json', '--store', outside/'datasets'])
with sqlite3.connect(fixture/'registry.sqlite') as db:
    identity, candidate = db.execute('SELECT candidate_id, payload FROM frozen_candidates').fetchone()
candidate = json.loads(candidate)
manifest = {key: candidate[key] for key in ('strategy', 'config', 'risk_policy', 'evidence')}
(outside/'candidate.json').write_text(json.dumps(manifest), encoding='utf-8')
registered = json.loads(execute('installed-freeze', [canonical, 'freeze-candidate', outside/'candidate.json',
    '--registry', outside/'registry.sqlite']))
assert registered['candidate_id'] == identity and registered['paper_eligible'] is False

(outside/'tests').mkdir()
for source in (root/'tests').glob('*.py'):
    shutil.copy2(source, outside/'tests'/source.name)
(outside/'pytest.ini').write_text('[pytest]\npythonpath = .\n', encoding='utf-8')
code = ('import sys,pytest; sys.path.insert(0, '+repr(str(outside))+'); '
        'raise SystemExit(pytest.main(["tests/test_research_cli.py::test_canonical_cli_runs_real_local_pipeline",'
        '"-q","--basetemp=cli-tmp","-p","no:cacheprovider"]))')
execute('installed-canonical-smoke', [sys.executable, '-I', '-c', code])
fixture_code = ('import sys; sys.path.insert(0, '+repr(str(outside))+'); '
                'from tests.prelaunch_fixture import run_fixture; run_fixture("recorded-e2e")')
execute('installed-recorded-e2e', [sys.executable, '-I', '-c', fixture_code])
fixture_result = json.loads((outside/'recorded-e2e/fixture_result.json').read_text())
assert fixture_result['outcome_class'] == 'EXPECTED_RESEARCH_REJECTION'
assert fixture_result['runtime_failure'] is False
results['recorded_fixture_result'] = fixture_result
(evidence/'installed-command-checks.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
