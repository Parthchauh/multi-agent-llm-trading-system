import pytest

from evaluation.ablation import compare_architectures
from evaluation.experiments import ExperimentArchitecture, ExperimentConfig
from evaluation.platform import ExperimentRunner, SQLiteResearchRepository


def reports(tmp_path):
    result = []
    repository = SQLiteResearchRepository(tmp_path/'research.sqlite')
    for architecture in (ExperimentArchitecture.SINGLE_AGENT, ExperimentArchitecture.MULTI_AGENT_FULL):
        full = architecture is ExperimentArchitecture.MULTI_AGENT_FULL
        config = ExperimentConfig(experiment_id=architecture.value, architecture=architecture,
            dataset_id='synthetic', dataset_fingerprint='a'*64, provider='recorded', model='test',
            symbols=('AAA',), seed=7, debate_rounds=1 if full else 0, refinement_rounds=1 if full else 0)
        result.append(ExperimentRunner(repository).run(config, lambda *args: {
            'status': 'VIABILITY_REJECTED', 'validation_metrics': {'total_return_pct': 2. if full else 1.},
            'final_holdout_metrics': {'total_return_pct': 999.},
            'invocation_audit': [{'latency_ms': 4., 'schema_valid': True, 'estimated_cost': None,
                                  'cost_basis': None}]}))
    return result


def test_descriptive_comparison_counts_rejections_and_does_not_select_on_holdout(tmp_path):
    compared = compare_architectures(reports(tmp_path))
    assert compared['status'] == 'DESCRIPTIVE_COMPARISON_ONLY'
    assert compared['scope'] == 'VALIDATION'
    assert compared['paired_runs'] == 1
    assert [r['mean_validation_return_pct'] for r in compared['treatments']] == [1., 2.]
    assert all(r['research_rejections'] == 1 for r in compared['treatments'])
    assert all(r['estimated_cost'] is None for r in compared['treatments'])
    assert '999' not in str(compared)
    assert compared['paper_eligible'] is False


def test_comparison_refuses_changed_costs_and_missing_or_different_data(tmp_path):
    values = reports(tmp_path)
    changed_costs = values[1].experiment_config.transaction_costs.model_copy(update={'slippage_pct': .02})
    changed = values[1].model_copy(update={'experiment_config': values[1].experiment_config.model_copy(
        update={'transaction_costs': changed_costs})})
    with pytest.raises(ValueError, match='controlled'):
        compare_architectures([values[0], changed])
    for digest in (None, 'b'*64):
        changed = values[1].model_copy(update={'experiment_config': values[1].experiment_config.model_copy(
            update={'dataset_fingerprint': digest})})
        with pytest.raises(ValueError, match='fingerprint'):
            compare_architectures([values[0], changed])


def test_comparison_rejects_unmatched_sampling_conditions(tmp_path):
    values = reports(tmp_path)
    config = values[1].experiment_config
    altered = values[1].model_copy(update={'experiment_config': config.model_copy(update={
        'model_settings': config.model_settings.model_copy(update={'temperature': 1.5})})})
    with pytest.raises(ValueError, match='controlled'):
        compare_architectures([values[0], altered])
