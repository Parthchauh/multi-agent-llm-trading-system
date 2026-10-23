import json
import subprocess
import sys
from pathlib import Path
import cli.research
import scripts.run_research_e2e

repo = Path('D:/TradingAgents')
evidence = repo / 'docs/prelaunch/acceptance-evidence'
outside = repo / '.tools/p0-acceptance/outside'
modules = {m.__name__: str(Path(m.__file__).resolve()) for m in (cli.research, scripts.run_research_e2e)}
assert all(Path(path).is_relative_to(Path(sys.prefix)) for path in modules.values()), modules
results = {'python': sys.executable, 'cwd': str(outside), 'installed_modules': modules, 'commands': {}}
for name in ('trading-research', 'tradingagents', 'tradingagents-legacy'):
    command = Path(sys.executable).parent / (name + '.exe')
    completed = subprocess.run([str(command), '--help'], capture_output=True, timeout=60)
    (evidence / (name + '-help.txt')).write_bytes(completed.stdout + completed.stderr)
    results['commands'][name] = {'exit_code': completed.returncode, 'command': [str(command), '--help']}
(evidence / 'installed-smoke.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))

