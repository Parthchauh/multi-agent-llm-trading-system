"""Machine-readable fail-closed release status; never an approval shortcut."""
import argparse
import json

from evaluation.certification import evidence_status


def release_status():
    statuses = evidence_status(technical=True, validation_passed=True,
        test_passed=True, holdout_passed=True, risk_passed=True)
    if statuses['STATISTICALLY_VIABLE'] or statuses['DEPLOYMENT_ELIGIBLE'] or statuses['real_money_execution']:
        raise RuntimeError('Disabled promotion boundary regressed.')
    return {'status': 'RELEASE_BLOCKED', 'outcome_class': 'EXPECTED_RELEASE_REJECTION',
            'runtime_failure': False, 'statistical_certification': False,
            'paper_eligible': False, 'real_money_eligible': False,
            'blockers': ['STATISTICAL_CERTIFICATION_DISABLED', 'INDEPENDENT_REVIEW_REQUIRED',
                         'REAL_PAPER_OBSERVATION_REQUIRED', 'EXTERNAL_RELEASE_CHECKLIST_REQUIRED']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assert-disabled', action='store_true',
                        help='Exit 0 only when promotion remains disabled; this does not approve release.')
    arguments = parser.parse_args()
    print(json.dumps(release_status(), sort_keys=True))
    raise SystemExit(0 if arguments.assert_disabled else 2)
