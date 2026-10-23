import numpy as np
import pandas as pd
import pytest

from scripts.run_research_e2e import run_e2e_research
from tests.conftest_helpers import make_simple_strategy
from tests.test_research_e2e import MockLifecycleClient


@pytest.mark.parametrize('mode,call_count,risk_reviews,debate_rounds', [
    ('MULTI_AGENT_NO_DEBATE', 4, 0, 0), ('MULTI_AGENT_DEBATE', 7, 0, 1),
    ('MULTI_AGENT_RISK', 8, 1, 1)])
def test_recorded_treatments_execute_distinct_roles_without_refining_or_promoting(
        monkeypatch, tmp_path, mode, call_count, risk_reviews, debate_rounds):
    from data.research_split import ResearchDatasetSplit
    def forbidden(*args, **kwargs):
        raise AssertionError('Development architecture study must never release sealed data.')
    monkeypatch.setattr(ResearchDatasetSplit, 'release_confirmatory_test', forbidden)
    monkeypatch.setattr(ResearchDatasetSplit, 'release_final_holdout', forbidden)
    strategy = make_simple_strategy(entry_indicator='close', entry_operator='>', entry_value=0.,
        exit_indicator='close', exit_operator='<', exit_value=0., max_holding_days=1,
        position_size_pct=10.)
    recorded = MockLifecycleClient(raw_strategy=strategy.model_dump(mode='json')).responses
    class Client:
        provider = 'recorded'
        model = 'fixture'
        def __init__(self):
            self.roles = []
        def invoke(self, schema, messages):
            self.roles.append(schema.__name__)
            for response in recorded:
                try:
                    return schema.model_validate(response)
                except ValueError:
                    continue
            raise AssertionError('No recorded response for requested role.')
    client = Client()
    prices = 100.+np.arange(1000)*.1
    frame = pd.DataFrame({'Open': prices, 'Close': prices+.1, 'High': prices+.2,
                          'Low': prices-.1, 'Volume': [10000.]*1000},
                         index=pd.date_range('2020-01-01', periods=1000, freq='B'))
    result = run_e2e_research(symbol='SYNTHETIC', df=frame, commission=0., slippage=0.,
        structured_client=client, mode=mode, development_only=True, output_dir=tmp_path/'run',
        exposure_ledger_path=tmp_path/'exposure.sqlite')
    lifecycle = result['lifecycle_result']
    assert result['status'] == 'COMPLETED'
    assert len(client.roles) == call_count
    assert len(lifecycle.risk_reviews) == risk_reviews
    assert sum(len(session.rounds) for session in lifecycle.debate_sessions) == debate_rounds
    assert lifecycle.refinement_proposals == ()
    assert len(lifecycle.strategy_records) == 1
    assert ('RiskReview' in client.roles) == bool(risk_reviews)
    assert result['certification']['DEPLOYMENT_ELIGIBLE'] is False
    assert result['certification']['STATISTICALLY_VIABLE'] is False
    assert result['test_metrics'] == result['final_holdout_metrics'] == {}
    assert result['provenance']['development_only'] is True
