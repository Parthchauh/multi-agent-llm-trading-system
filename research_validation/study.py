"""Immutable development-study plans and retrospective adherence checks.

Registration is local provenance, not independent preregistration or approval.
This module never calls a provider and does not enforce a provider billing cap.
"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from data.snapshots import canonical_bytes
from evaluation.ablation import compare_architectures
from evaluation.experiments import ExperimentConfig
from evaluation.platform import ExperimentReport


class StudyPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)

    schema_version: Literal[1] = 1
    title: str = Field(min_length=1, max_length=200)
    hypothesis: str = Field(min_length=20, max_length=2000)
    treatments: tuple[ExperimentConfig, ...] = Field(min_length=2, max_length=5)
    dataset_hashes: dict[str, str]
    data_rights_reference: str = Field(min_length=1, max_length=500)
    review_reference: str = Field(min_length=1, max_length=500)
    primary_metric: Literal['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct']
    missing_outcomes: Literal['retain_and_disclose'] = 'retain_and_disclose'
    maximum_invocations_per_run: int = Field(strict=True, ge=1, le=10000)
    maximum_total_cost: float = Field(strict=True, ge=0)
    currency: str = Field(pattern=r'^[A-Z]{3}$')

    @model_validator(mode='after')
    def validate_controls(self):
        first = self.treatments[0]
        if len({c.experiment_id for c in self.treatments}) != len(self.treatments):
            raise ValueError('Experiment IDs must be distinct.')
        if len({c.architecture for c in self.treatments}) != len(self.treatments):
            raise ValueError('Architecture treatments must be distinct.')
        controls = ('provider', 'model', 'dataset_id', 'start', 'end', 'symbols', 'seed',
                    'repetitions', 'information_set', 'transaction_costs', 'split_config',
                    'model_settings', 'robustness_config', 'strategy_schema_version')
        for config in self.treatments:
            if not config.development_only:
                raise ValueError('Studies must keep confirmatory and holdout data sealed.')
            if any(getattr(config, key) != getattr(first, key) for key in controls):
                raise ValueError('Treatments must share controlled conditions.')
            if config.seed is None or config.start is None or config.end is None:
                raise ValueError('Explicit seed and dates are required.')
            if config.split_config is None or config.split_config.context_end is None:
                raise ValueError('Explicit chronological boundaries are required.')
            if not config.prompt_versions or any(not p.content_hash for p in config.prompt_versions):
                raise ValueError('Prompt versions and hashes are required.')
        if set(self.dataset_hashes) != set(first.symbols):
            raise ValueError('Every planned symbol requires a dataset hash.')
        if any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h)
               for h in self.dataset_hashes.values()):
            raise ValueError('Dataset hashes must be lowercase SHA256 digests.')
        return self


class StudyRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS study_plans '
                       '(study_id TEXT PRIMARY KEY, registered_at TEXT NOT NULL, payload TEXT NOT NULL)')

    def register(self, plan: StudyPlan) -> dict:
        # Revalidate even when supplied a model_copy with unchecked values.
        plan = StudyPlan.model_validate(plan.model_dump())
        content = canonical_bytes(plan.model_dump(mode='json'))
        identity = hashlib.sha256(content).hexdigest()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('INSERT OR IGNORE INTO study_plans VALUES (?, ?, ?)',
                       (identity, datetime.now(timezone.utc).isoformat(), content.decode()))
        return self.get(identity)

    def get(self, identity: str) -> dict:
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute('SELECT registered_at, payload FROM study_plans WHERE study_id=?',
                             (identity,)).fetchone()
        if row is None:
            raise KeyError('Study plan not found.')
        if hashlib.sha256(row[1].encode()).hexdigest() != identity:
            raise ValueError('Study plan integrity failure.')
        plan = StudyPlan.model_validate_json(row[1])
        return {'study_id': identity, 'registered_at': row[0], 'plan': plan.model_dump(mode='json'),
                'independent_review_verified': False, 'paper_eligible': False}

    def assess(self, identity: str, reports: list[ExperimentReport]) -> dict:
        registered = self.get(identity)
        plan = StudyPlan.model_validate(registered['plan'])
        expected = {c.experiment_id: c for c in plan.treatments}
        if (len(reports) != len(expected)
                or {r.experiment_config.experiment_id for r in reports} != set(expected)):
            raise ValueError('Exactly every predeclared treatment report is required.')
        started_after = datetime.fromisoformat(registered['registered_at'])
        budget_findings = []
        total_cost = 0.
        costs_known = True
        for report in reports:
            config = report.experiment_config
            if config != expected[config.experiment_id]:
                raise ValueError('Report configuration differs from frozen study plan.')
            for run in report.runs:
                if run.started_at.tzinfo is None or run.started_at < started_after:
                    raise ValueError('Study must be registered before every run starts.')
                actual_hash = ((run.result.get('provenance') or {}).get('dataset') or {}).get('sha256')
                if actual_hash != plan.dataset_hashes.get(run.symbol):
                    raise ValueError('Actual run dataset differs from frozen study plan.')
                events = run.result.get('invocation_audit')
                if events is None:
                    budget_findings.append('INVOCATION_COUNT_UNKNOWN')
                    costs_known = False
                    continue
                if len(events) > plan.maximum_invocations_per_run:
                    budget_findings.append('INVOCATION_BUDGET_EXCEEDED')
                for event in events:
                    cost, basis = event.get('estimated_cost'), event.get('cost_basis') or {}
                    if cost is None or basis.get('currency') != plan.currency:
                        costs_known = False
                    else:
                        total_cost += cost
        comparison = compare_architectures(reports)  # Also validates finite costs/metrics and all run pairs.
        if not costs_known:
            budget_findings.append('COST_UNKNOWN')
        if total_cost > plan.maximum_total_cost:
            budget_findings.append('COST_BUDGET_EXCEEDED')
        return {'study_id': identity, 'status': 'DEVELOPMENT_STUDY_AUDITED',
                'primary_metric': plan.primary_metric, 'comparison': comparison,
                'budget_findings': sorted(set(budget_findings)),
                'known_cost_subtotal': total_cost, 'total_cost': total_cost if costs_known else None,
                'currency': plan.currency, 'budget_enforced_before_calls': False,
                'independent_review_verified': False, 'statistical_certification': False,
                'paper_eligible': False,
                'limitations': ['Local registration timestamps are not external attestations.',
                                'Rights/review references are recorded claims requiring human verification.',
                                'Budget audit is retrospective; no paid calls are made by this command.']}
