import os
import sys
from pathlib import Path
outside = Path('D:/TradingAgents/.tools/p0-acceptance/outside')
os.chdir(outside)
sys.path.insert(0, str(outside))
import scripts.run_research_e2e
assert Path(scripts.run_research_e2e.__file__).is_relative_to(Path(sys.prefix))
print('Installed runner: ' + scripts.run_research_e2e.__file__)
from tests.prelaunch_fixture import run_fixture
run_fixture(outside / 'fixture')
