import json

import pytest

from backtesting.models import BacktestConfig
from research_validation.campaign import CampaignConfig, FoldResearchCampaign
from risk.profiles import risk_profile
from tests.conftest_helpers import make_ohlcv, make_simple_strategy


def run(frame, factory, **overrides):
    config = CampaignConfig(train_size=60, validation_size=20, test_size=20, step_size=20,
                            attempts_per_fold=3, maximum_attempts=30, minimum_folds=2,
                            **overrides)
    return FoldResearchCampaign().run(development_data=frame, config=config,
        backtest_config=BacktestConfig(initial_capital=10000.),
        risk_policy=risk_profile(), candidate_factory=factory)


def test_generation_sees_only_training_copies_and_all_attempts_are_counted():
    frame = make_ohlcv(120)
    seen = []
    def factory(train, fold, attempt):
        seen.append((fold, attempt, train.index[0], train.index[-1]))
        train.iloc[:, :] = 1.  # Candidate code cannot mutate the scoring data.
        if attempt == 1:
            return {'broken': True}
        return make_simple_strategy()
    original = frame.copy(deep=True)
    report = run(frame, factory)
    assert frame.equals(original)
    assert len(seen) == 6
    assert seen[0][2:] == (frame.index[0], frame.index[59])
    assert seen[3][2:] == (frame.index[20], frame.index[79])
    assert report['attempts_used'] == 6
    assert report['tested_folds'] == 2
    for fold in report['folds']:
        assert [a['status'] for a in fold['attempts']] == ['REJECTED', 'EVALUATED', 'DUPLICATE']
        assert fold['train']['end'] < fold['validation']['start']
        assert fold['validation']['end'] < fold['test']['start']
        # The caller's 10% cap is stricter than the strategy and policy's 20%.
        assert fold['selected']['config']['position_size_pct'] == 10.
    assert report['paper_eligible'] is False
    assert report['statistical_certification'] is False
    json.dumps(report, allow_nan=False)


def test_changing_first_test_window_does_not_change_first_selection():
    frame = make_ohlcv(120)
    def factory(train, fold, attempt):
        return make_simple_strategy(entry_value=50.+attempt)
    before = run(frame, factory)
    changed = frame.copy(deep=True)
    changed.iloc[80:100, :4] *= 2.
    after = run(changed, factory)
    assert before['folds'][0]['selected'] == after['folds'][0]['selected']
    assert before['folds'][0]['attempts'] == after['folds'][0]['attempts']
    assert before['folds'][0]['test']['sha256'] != after['folds'][0]['test']['sha256']
    assert before['folds'][1]['train']['sha256'] == after['folds'][1]['train']['sha256']


def test_budget_exhaustion_is_preflight_and_calls_no_model():
    called = []
    with pytest.raises(ValueError, match='budget'):
        FoldResearchCampaign().run(development_data=make_ohlcv(120),
            config=CampaignConfig(train_size=60, validation_size=20, test_size=20,
                step_size=20, attempts_per_fold=2, maximum_attempts=3, minimum_folds=2),
            backtest_config=BacktestConfig(initial_capital=10000.), risk_policy=risk_profile(),
            candidate_factory=lambda *args: called.append(True))
    assert called == []


def test_test_overlap_is_rejected_and_expanding_training_keeps_origin():
    with pytest.raises(ValueError, match='overlap'):
        CampaignConfig(train_size=60, validation_size=20, test_size=20, step_size=10)
    report = run(make_ohlcv(120), lambda *args: make_simple_strategy(), expanding=True)
    assert report['folds'][0]['train']['start'] == report['folds'][1]['train']['start']
    assert report['folds'][1]['train']['bars'] == 80


def test_runtime_failures_are_safe_and_distinct_from_schema_rejection():
    def factory(*args):
        raise ConnectionError('provider secret must not appear')
    report = run(make_ohlcv(120), factory)
    assert report['tested_folds'] == 0
    assert report['test_distribution']['mean'] is None
    assert report['folds'][0]['attempts'][0]['status'] == 'RUNTIME_FAILURE'
    assert 'provider secret' not in json.dumps(report)


def test_actual_train_only_agent_adapter_uses_recorded_client_and_training_timestamp():
    from research_validation.campaign import TrainOnlyAgentFactory
    from tests.test_research_e2e import MockLifecycleClient

    class RecordingClient(MockLifecycleClient):
        def __init__(self):
            super().__init__()
            self.prompts = []
        def invoke(self, schema, messages):
            self.prompts.append(messages)
            return super().invoke(schema, messages)
    client = RecordingClient()
    frame = make_ohlcv(120)
    strategy = TrainOnlyAgentFactory(client, symbol='SYNTHETIC')(frame.iloc[:60], 1, 1)
    assert strategy['metadata']['timeframe'] == '1d'
    text = json.dumps(client.prompts)
    assert frame.index[59].date().isoformat() in text
    assert frame.index[60].date().isoformat() not in text
