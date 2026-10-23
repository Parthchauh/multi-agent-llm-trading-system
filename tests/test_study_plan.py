import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from evaluation.experiments import (
    ChronologicalSplitConfig,
    ExperimentArchitecture,
    ExperimentConfig,
    PromptVersion,
)
from evaluation.platform import ExperimentRunner, SQLiteResearchRepository
from research_validation.study import StudyPlan, StudyRegistry


def plan():
    configs = tuple(ExperimentConfig(experiment_id=a.value, architecture=a,
        provider='recorded', model='fixture-v1', dataset_id='fixture', symbols=('AAA',),
        start=date(2020, 1, 1), end=date(2022, 1, 1), seed=7, development_only=True,
        prompt_versions=(PromptVersion(role='generator', version='v1', content_hash='b'*64),),
        split_config=ChronologicalSplitConfig(context_end=date(2020, 6, 1),
            validation_end=date(2021, 1, 1), test_end=date(2021, 6, 1)))
        for a in (ExperimentArchitecture.SINGLE_AGENT, ExperimentArchitecture.MULTI_AGENT_NO_DEBATE))
    return StudyPlan(title='Recorded fixture study', hypothesis='Compare descriptive development outcomes.',
        treatments=configs, dataset_hashes={'AAA': 'a'*64}, data_rights_reference='synthetic fixture',
        review_reference='pending independent review', primary_metric='total_return_pct',
        maximum_invocations_per_run=1, maximum_total_cost=1., currency='USD')


def run_reports(tmp_path, study, events=None):
    repository = SQLiteResearchRepository(tmp_path/'research.sqlite')
    return [ExperimentRunner(repository).run(config, lambda *args: {
        'status': 'VIABILITY_REJECTED', 'validation_metrics': {'total_return_pct': 1.},
        'provenance': {'dataset': {'sha256': 'a'*64}},
        'invocation_audit': events if events is not None else [
            {'latency_ms': 3., 'schema_valid': True, 'estimated_cost': .2, 'cost_basis': {'currency': 'USD'}}]})
        for config in study.treatments]


def test_registered_plan_survives_restart_and_checks_complete_paired_evidence(tmp_path):
    path = tmp_path/'studies.sqlite'
    study = plan()
    registered = StudyRegistry(path).register(study)
    assert StudyRegistry(path).register(study) == registered
    reports = run_reports(tmp_path, study)
    result = StudyRegistry(path).assess(registered['study_id'], reports)
    assert result['total_cost'] == pytest.approx(.4)
    assert result['budget_findings'] == []
    assert result['paper_eligible'] is False
    assert result['independent_review_verified'] is False
    assert result['budget_enforced_before_calls'] is False
    with pytest.raises(ValueError, match='every predeclared'):
        StudyRegistry(path).assess(registered['study_id'], reports[:1])


def test_retroactive_registration_and_config_or_dataset_change_are_rejected(tmp_path):
    study, registry = plan(), StudyRegistry(tmp_path/'studies.sqlite')
    identity = registry.register(study)['study_id']
    reports = run_reports(tmp_path, study)
    report = reports[0]
    run = report.runs[0]
    for changed, message in [
        (run.model_copy(update={'started_at': datetime.now(timezone.utc)-timedelta(days=1)}), 'before'),
        (run.model_copy(update={'result': {**run.result, 'provenance': {'dataset': {'sha256': 'c'*64}}}}), 'dataset')]:
        altered = report.model_copy(update={'runs': (changed,)})
        with pytest.raises(ValueError, match=message):
            registry.assess(identity, [altered, reports[1]])
    altered = report.model_copy(update={'experiment_config': report.experiment_config.model_copy(update={'seed': 8})})
    with pytest.raises(ValueError, match='configuration'):
        registry.assess(identity, [altered, reports[1]])


def test_unknown_and_exceeded_budgets_never_become_approval(tmp_path):
    study, registry = plan(), StudyRegistry(tmp_path/'studies.sqlite')
    identity = registry.register(study)['study_id']
    events = [{'latency_ms': 1., 'schema_valid': True, 'estimated_cost': 2., 'cost_basis': {'currency': 'USD'}},
              {'latency_ms': 1., 'schema_valid': False, 'estimated_cost': None}]
    result = registry.assess(identity, run_reports(tmp_path, study, events))
    assert result['total_cost'] is None
    assert result['known_cost_subtotal'] == 4.
    assert result['budget_findings'] == ['COST_BUDGET_EXCEEDED', 'COST_UNKNOWN', 'INVOCATION_BUDGET_EXCEEDED']
    assert result['statistical_certification'] is False


def test_tampering_and_unsafe_unchecked_model_copies_are_rejected(tmp_path):
    registry, study = StudyRegistry(tmp_path/'studies.sqlite'), plan()
    for updates in ({'development_only': False}, {'prompt_versions': ()}, {'seed': None}):
        changed = study.model_copy(update={'treatments': (study.treatments[0].model_copy(update=updates), study.treatments[1])})
        with pytest.raises(ValueError):
            registry.register(changed)
    identity = registry.register(study)['study_id']
    with sqlite3.connect(registry.path) as db:
        db.execute("UPDATE study_plans SET payload='{}'")
    with pytest.raises(ValueError, match='integrity'):
        registry.get(identity)
