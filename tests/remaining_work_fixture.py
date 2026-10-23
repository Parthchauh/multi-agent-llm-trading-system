"""Zero-cost recorded fixture for study governance, alerts and encrypted recovery."""
import argparse
import json
import secrets
from pathlib import Path

from research_service.backups import create_backup
from research_service.encrypted_backups import encrypt_backup, restore_encrypted_backup
from research_service.monitoring import ServiceMonitor
from research_validation.study import StudyPlan, StudyRegistry
from tests.test_study_plan import plan, run_reports


def run(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    study = StudyPlan.model_validate(plan().model_copy(update={'maximum_total_cost': 0.}).model_dump())
    registry = StudyRegistry(output/'studies.sqlite')
    registered = registry.register(study)
    reports = run_reports(output, study, events=[{
        'latency_ms': 0., 'schema_valid': True, 'estimated_cost': 0.,
        'cost_basis': {'currency': 'USD', 'revision': 'offline-recorded-fixture'}}])
    assessment = registry.assess(registered['study_id'], reports)
    (output/'study-plan.json').write_text(study.model_dump_json(indent=2), encoding='utf-8')
    (output/'study-assessment.json').write_text(json.dumps(assessment, indent=2), encoding='utf-8')
    monitor = ServiceMonitor(output/'metrics.sqlite')
    for _ in range(20):
        monitor.record(503, 2500.)
    health = monitor.snapshot()
    (output/'health.json').write_text(json.dumps(health, indent=2), encoding='utf-8')
    create_backup(output/'backup', {'studies.sqlite': output/'studies.sqlite',
                                   'metrics.sqlite': output/'metrics.sqlite'})
    # Ephemeral fixture passphrase is never stored. The encrypted sample is not a user backup.
    password = secrets.token_urlsafe(32)
    encrypted = encrypt_backup(output/'backup', output/'backup.enc', password)
    restored = restore_encrypted_backup(output/'backup.enc', output/'restored', password)
    summary = {'status': 'OFFLINE_RECORDED_FIXTURE_PASSED', 'runtime_failure': False,
               'live_provider_calls': 0, 'actual_spend': 0, 'synthetic_evidence': True,
               'study_id': registered['study_id'], 'budget_findings': assessment['budget_findings'],
               'persisted_alerts': len(ServiceMonitor(output/'restored/metrics.sqlite').snapshot()['open_alerts']),
               'encrypted_backup': encrypted, 'restore': restored,
               'statistical_certification': False, 'paper_eligible': False}
    if summary['persisted_alerts'] != 2 or assessment['budget_findings']:
        raise RuntimeError('Recorded fixture invariant failed.')
    (output/'fixture-result.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
