"""Compare recorded architecture treatments on matched validation evidence only."""
from __future__ import annotations

import math
import statistics

from evaluation.platform import ExperimentReport


def compare_architectures(reports: list[ExperimentReport]) -> dict:
    if not 2 <= len(reports) <= 5:
        raise ValueError('Compare two to five distinct architecture treatments.')
    architectures = [r.experiment_config.architecture.value for r in reports]
    if len(set(architectures)) != len(reports):
        raise ValueError('Architecture treatments must be distinct.')
    reference = reports[0].experiment_config
    controlled_fields = ('provider', 'model', 'dataset_id', 'start', 'end', 'symbols',
                         'seed', 'repetitions', 'information_set', 'transaction_costs', 'split_config', 'development_only',
                         'model_settings', 'robustness_config', 'strategy_schema_version')
    rows, reference_datasets, reference_pairs = [], None, None
    for report in reports:
        config = report.experiment_config
        if any(getattr(config, field) != getattr(reference, field) for field in controlled_fields):
            raise ValueError('Treatments differ in controlled data/model/seed/cost conditions.')
        datasets, pairs, audits = {}, set(), []
        returns: list[float] = []
        sharpes: list[float] = []
        drawdowns: list[float] = []
        telemetry_complete = True
        for run in report.runs:
            pair = (run.symbol, run.repetition)
            if pair in pairs:
                raise ValueError('Duplicate symbol/repetition evidence.')
            pairs.add(pair)
            provenance = run.result.get('provenance') or {}
            fingerprint = (provenance.get('dataset') or {}).get('sha256') or config.dataset_fingerprint
            if not fingerprint:
                raise ValueError('Actual dataset fingerprints are required for architecture comparison.')
            datasets[pair] = fingerprint
            metrics = run.result.get('validation_metrics') or {}
            for target, name in ((returns, 'total_return_pct'), (sharpes, 'sharpe_ratio'),
                                 (drawdowns, 'max_drawdown_pct')):
                value = metrics.get(name)
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError('Validation metrics must be finite numeric values.')
                    target.append(float(value))
            audit = run.result.get('invocation_audit')
            if audit is None:
                telemetry_complete = False
            else:
                for event in audit:
                    for name in ('latency_ms', 'estimated_cost'):
                        value = event.get(name)
                        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                                  or not math.isfinite(value) or value < 0):
                            raise ValueError('Audit latency/cost must be finite non-negative numbers.')
                audits.extend(audit)
        expected = {(symbol, repetition) for symbol in config.symbols for repetition in range(1, config.repetitions+1)}
        if pairs != expected:
            raise ValueError('Every planned symbol/repetition must be reported, including failures.')
        if reference_pairs is None:
            reference_pairs, reference_datasets = pairs, datasets
        elif pairs != reference_pairs or datasets != reference_datasets:
            raise ValueError('Treatments do not share identical paired dataset fingerprints.')
        all_costs_known = telemetry_complete and bool(audits) and all(
            e.get('estimated_cost') is not None and e.get('cost_basis') for e in audits)
        currencies = {e['cost_basis']['currency'] for e in audits} if all_costs_known else set()
        row = {'architecture': config.architecture.value, 'experiment_id': config.experiment_id,
               'runs': len(report.runs), 'validation_metric_count': len(returns),
               'mean_validation_return_pct': statistics.fmean(returns) if returns else None,
               'worst_validation_return_pct': min(returns) if returns else None,
               'mean_validation_sharpe': statistics.fmean(sharpes) if sharpes else None,
               'worst_validation_drawdown_pct': max(drawdowns) if drawdowns else None,
               'runtime_failures': len(report.failures), 'research_rejections': len(report.rejections),
               'actual_model_invocations': len(audits) if telemetry_complete else None,
               'invalid_schema_invocations': sum(e.get('schema_valid') is False for e in audits) if telemetry_complete else None,
               'estimated_cost': sum(e['estimated_cost'] for e in audits) if all_costs_known and len(currencies) == 1 else None,
               'currency': next(iter(currencies)) if len(currencies) == 1 else None,
               'total_latency_ms': sum(e['latency_ms'] for e in audits) if telemetry_complete and all(e.get('latency_ms') is not None for e in audits) else None,
               'treatment': {'debate_rounds': config.debate_rounds, 'refinement_rounds': config.refinement_rounds}}
        rows.append(row)
    return {'schema_version': 1, 'status': 'DESCRIPTIVE_COMPARISON_ONLY', 'scope': 'VALIDATION',
            'treatments': rows, 'paired_runs': len(reports[0].runs),
            'limitations': ['Validation data is adaptive evidence, not confirmatory significance.',
                            'No holdout metric selects the preferred architecture.',
                            'Different model-call counts and unknown token prices are disclosed.'],
            'statistical_certification': False, 'paper_eligible': False}
