"""scripts/run_research_e2e.py
============================
Authoritative end-to-end research experiment runner for:
Multi-Agent LLM Systems for Trading Strategy Generation and Backtesting.

Path:
Historical Market Data -> Data Validation -> Market Regime -> Structured Agent Analysis
-> Strategy Generation -> Strategy Validation -> Strategy Compilation -> Backtesting
-> Evaluation -> Risk Assessment -> Critic / Refinement -> Out-of-Sample Validation
-> Baseline Comparison -> Final Structured Result & Summary.
"""

# This module is intentionally executable from outside the repository.  It
# resolves the workspace before importing project packages, so E402 is the
# appropriate scoped exception rather than a hidden runtime path dependency.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

# Add workspace to path if needed
workspace_dir = str(Path(__file__).resolve().parent.parent)
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

# A manual invocation should use the same local provider configuration as the
# application CLI.  ``override=False`` preserves an explicitly supplied
# environment variable and the key never enters logs or result artifacts.
load_dotenv(Path(workspace_dir) / ".env", override=False)

from backtesting.engine import BacktestEngine
from backtesting.models import BacktestConfig
from data.provider import MarketDataProvider, YahooFinanceProvider
from data.research_split import (
    ResearchDatasetSplit,
    ResearchPartition,
    ResearchPartitionRole,
)
from data.schema import validate_ohlcv
from evaluation.baselines import run_all_baselines
from evaluation.campaigns import HoldoutExposureLedger
from evaluation.certification import EvaluationLifecycle, evidence_status, launch_viability_policy
from evaluation.research_metrics import strategy_execution_fingerprint
from research_validation import ValidationCandidate, select_validation_candidate
from risk import (
    RiskPolicy,
)
from risk.profiles import policy_hash, require_complete_policy, risk_profile
from strategies.compiler import StrategyCompiler
from strategies.schema import StrategySchema
from tradingagents.agents.regime_analyst import RegimeAnalystAgent
from tradingagents.agents.sprint6_agents import (
    BearResearcherAgent,
    BullResearcherAgent,
    DebateSynthesizerAgent,
    PerformanceAnalystAgent,
    RefinementAgent,
    RiskManagerAgent,
    StrategyCriticAgent,
)
from tradingagents.agents.strategy_generator import StrategyGeneratorAgent
from tradingagents.agents.structured_output import (
    LangChainStructuredOutputClient,
    StructuredOutputClient,
)
from tradingagents.agents.telemetry import AuditedStructuredClient
from tradingagents.graph.research_graph import Sprint4ResearchGraph
from tradingagents.graph.sprint6_lifecycle import (
    Sprint6LifecycleConfig,
    Sprint6ResearchGraph,
    Sprint6RunResult,
    Sprint6RunStatus,
)
from tradingagents.llm_clients.factory import create_llm_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_research_e2e")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_fingerprint(data: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def build_research_dataset_split(
    df: pd.DataFrame,
    *,
    symbol: str,
    train_end: str | None = None,
    validation_end: str | None = None,
    test_end: str | None = None,
) -> ResearchDatasetSplit:
    """Partition validated OHLCV data into 4 chronological, leak-free partitions."""
    validated = validate_ohlcv(df)
    n = len(validated)
    if n < 100:
        raise ValueError(f"Dataset has only {n} rows, required minimum is 100.")

    # Convert dates to partition slice indices if provided and present
    c_idx, v_idx, t_idx = None, None, None
    if train_end:
        train_dt = pd.Timestamp(train_end)
        matches = validated.index[validated.index <= train_dt]
        if len(matches) >= 40:
            c_idx = len(matches)

    if validation_end and c_idx is not None:
        val_dt = pd.Timestamp(validation_end)
        matches = validated.index[validated.index <= val_dt]
        if len(matches) > c_idx + 20:
            v_idx = len(matches)

    if test_end and v_idx is not None:
        test_dt = pd.Timestamp(test_end)
        matches = validated.index[validated.index <= test_dt]
        # Split the remaining into test and final holdout
        remaining = len(matches) - v_idx
        if remaining >= 40:
            t_idx = v_idx + (remaining // 2)

    # Fallback to standard robust chronological fractions if dates don't match cleanly
    if c_idx is None or v_idx is None or t_idx is None or not (0 < c_idx < v_idx < t_idx < n):
        c_idx = int(n * 0.50)
        v_idx = c_idx + int(n * 0.20)
        t_idx = v_idx + int(n * 0.15)

    ctx_data = validated.iloc[:c_idx]
    val_data = validated.iloc[c_idx:v_idx]
    test_data = validated.iloc[v_idx:t_idx]
    holdout_data = validated.iloc[t_idx:]

    sym = symbol.upper()
    context_partition = ResearchPartition(
        partition_id=f"{sym}:train",
        role=ResearchPartitionRole.CONTEXT,
        observations=ctx_data,
    )
    validation_partition = ResearchPartition(
        partition_id=f"{sym}:val",
        role=ResearchPartitionRole.VALIDATION,
        observations=val_data,
        warmup=ctx_data,  # Warm-up provided for non-scoring indicator causal lookbacks
    )
    test_partition = ResearchPartition(
        partition_id=f"{sym}:test",
        role=ResearchPartitionRole.TEST,
        observations=test_data,
        warmup=validated.iloc[:v_idx],
    )
    final_holdout_partition = ResearchPartition(
        partition_id=f"{sym}:final_holdout",
        role=ResearchPartitionRole.FINAL_HOLDOUT,
        observations=holdout_data,
        warmup=validated.iloc[:t_idx],
    )

    return ResearchDatasetSplit(
        context=context_partition,
        validation=validation_partition,
        test=test_partition,
        final_holdout=final_holdout_partition,
    )


def metrics_to_dict(metrics: Any) -> dict[str, Any]:
    """Serialize BacktestMetrics or EvaluationMetrics into a clean dictionary."""
    if metrics is None:
        return {}
    if hasattr(metrics, "model_dump"):
        return metrics.model_dump(mode="json")
    fields = (
        "total_return_pct",
        "cagr_pct",
        "max_drawdown_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "win_rate_pct",
        "profit_factor",
        "expectancy",
        "number_of_trades",
        "exposure_pct",
        "turnover_pct",
        "calmar_ratio",
    )
    return {f: getattr(metrics, f, None) for f in fields}


def run_e2e_research(
    *,
    symbol: str = "AAPL",
    start_date: str = "2020-01-01",
    train_end: str = "2023-12-31",
    validation_end: str = "2024-12-31",
    test_end: str = "2025-12-31",
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    position_size_pct: float = 10.0,
    model_provider: str = "google",
    model_name: str = "gemini-2.5-flash",
    structured_client: StructuredOutputClient | None = None,
    max_debate_rounds: int = 1,
    max_refinement_rounds: int = 1,
    seed: int = 42,
    mode: str = "MULTI_AGENT_FULL",
    development_only: bool = False,
    output_dir: Path | str | None = None,
    data_provider: MarketDataProvider | None = None,
    df: pd.DataFrame | None = None,
    risk_policy: RiskPolicy | None = None,
    campaign_id: str = 'default-research-campaign',
    exposure_ledger_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute the complete authoritative multi-agent research lifecycle."""

    mode = mode.upper()
    if not isinstance(development_only, bool):
        raise TypeError('development_only must be boolean.')
    if mode not in {'SINGLE_AGENT', 'MULTI_AGENT_NO_DEBATE', 'MULTI_AGENT_DEBATE', 'MULTI_AGENT_RISK', 'MULTI_AGENT_FULL'}:
        raise ValueError('Unknown controlled architecture treatment.')
    effective_risk_policy = require_complete_policy(risk_policy) if risk_policy is not None else risk_profile()

    run_id = f"e2e_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = Path("results") / "e2e" / run_id
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("=== PHASE 1: PRE-FLIGHT INITIALIZATION ===")
    logger.info("Run ID: %s | Symbol: %s | Mode: %s", run_id, symbol, mode)

    # 1. Acquire and Validate Data
    logger.info("=== PHASE 5: DATA ACQUISITION & VALIDATION ===")
    if df is not None:
        raw_df = df
        provider_name = "in_memory"
    else:
        provider = data_provider or YahooFinanceProvider()
        provider_name = type(provider).__name__
        logger.info("Downloading historical data for %s (%s -> %s)...", symbol, start_date, test_end)
        raw_df = provider.fetch(symbol, start_date, test_end)

    validated_df = validate_ohlcv(raw_df)
    first_ts = validated_df.index[0]
    last_ts = validated_df.index[-1]
    row_count = len(validated_df)

    print("\n--- DATA VALIDATION SUMMARY ---")
    print(f"Symbol: {symbol}")
    print(f"Rows: {row_count}")
    print(f"First timestamp: {first_ts}")
    print(f"Last timestamp: {last_ts}")
    print(f"Provider: {provider_name}\n")

    # 2. Build Partitions
    dataset_split = build_research_dataset_split(
        validated_df,
        symbol=symbol,
        train_end=train_end,
        validation_end=validation_end,
        test_end=test_end,
    )

    # 3. Configure Backtest Engine
    base_backtest_config = BacktestConfig(
        initial_capital=initial_capital,
        commission_pct=commission,
        slippage_pct=slippage,
        position_size_pct=position_size_pct,
    )

    # 4. Configure LLM Client
    logger.info("=== PHASE 6: LLM CLIENT CONFIGURATION ===")
    if structured_client is None:
        logger.info("Creating provider client for %s with model %s...", model_provider, model_name)
        # The provider factory receives an API key only through its explicit
        # boundary.  The key is never included in run configuration, events,
        # artifacts, or logs.
        provider_key_names = {
            "google": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "xai": "XAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        env_key = os.environ.get(provider_key_names.get(model_provider.lower(), ""))
        base_client = create_llm_client(
            provider=model_provider,
            model=model_name,
            api_key=env_key,
            temperature=0.2,
            seed=seed,
            timeout=30,
            max_retries=2,
        )
        client: StructuredOutputClient = LangChainStructuredOutputClient.from_provider_client(base_client)
    else:
        client = structured_client
    client = AuditedStructuredClient(client, experiment_id=run_id)

    # 5. Build Multi-Agent Graphs
    regime_agent = RegimeAnalystAgent(client)
    strategy_gen_agent = StrategyGeneratorAgent(client, max_attempts=3)

    initial_graph = Sprint4ResearchGraph(
        regime_agent=regime_agent,
        strategy_generator=strategy_gen_agent,
        compiler=StrategyCompiler(),
        engine_factory=BacktestEngine,
    )

    is_multi_agent = mode != 'SINGLE_AGENT'
    debate_enabled = mode in {'MULTI_AGENT_DEBATE', 'MULTI_AGENT_RISK', 'MULTI_AGENT_FULL'}
    risk_review_enabled = mode in {'MULTI_AGENT_RISK', 'MULTI_AGENT_FULL'}
    eff_debate_rounds = max_debate_rounds if debate_enabled else 0
    eff_refinement_rounds = max_refinement_rounds if mode == 'MULTI_AGENT_FULL' else 0

    lifecycle_config = Sprint6LifecycleConfig(
        max_debate_rounds=eff_debate_rounds,
        max_refinement_rounds=eff_refinement_rounds,
        review_only=is_multi_agent and mode != 'MULTI_AGENT_FULL',
        llm_risk_review=risk_review_enabled,
        viability_policy=launch_viability_policy(),
        risk_policy=effective_risk_policy,
    )

    lifecycle_kwargs: dict[str, Any] = {}
    if is_multi_agent and eff_debate_rounds > 0:
        lifecycle_kwargs.update(
            bull_researcher=BullResearcherAgent(client),
            bear_researcher=BearResearcherAgent(client),
            debate_synthesizer=DebateSynthesizerAgent(client),
        )
    if is_multi_agent:
        lifecycle_kwargs.update(
            performance_analyst=PerformanceAnalystAgent(client),
            critic=StrategyCriticAgent(client),
        )
        if risk_review_enabled:
            lifecycle_kwargs['risk_manager'] = RiskManagerAgent(client)
        if eff_refinement_rounds:
            lifecycle_kwargs['refinement_agent'] = RefinementAgent(client)

    lifecycle = Sprint6ResearchGraph(
        initial_graph=initial_graph,
        config=lifecycle_config,
        compiler=StrategyCompiler(),
        engine_factory=BacktestEngine,
        **lifecycle_kwargs,
    )

    # 6. Execute Research Lifecycle
    logger.info("=== PHASE 7-11: RESEARCH LIFECYCLE EXECUTION ===")
    lifecycle_result: Sprint6RunResult = lifecycle.run(
        symbol=symbol,
        dataset_split=dataset_split,
        backtest_config=base_backtest_config,
    )

    logger.info("Lifecycle completed with status: %s", lifecycle_result.status.value)
    selected_record = lifecycle_result.selected_record

    # 7. Evaluate Train, Validation, and Out-of-Sample Metrics
    train_metrics_dict: dict[str, Any] = {}
    val_metrics_dict: dict[str, Any] = {}
    test_metrics_dict: dict[str, Any] = {}
    final_holdout_metrics_dict: dict[str, Any] = {}
    certification = EvaluationLifecycle(state_path=out_path / 'evaluation_state.json')
    test_passed = holdout_passed = False
    test_risk_passed = holdout_risk_passed = False
    holdout_exposure = 'NOT_EXPOSED'
    strategy_schema_dict: dict[str, Any] = {}
    selection_data: dict[str, Any] = {}
    selected_strategy: StrategySchema | None = None

    # A strategy can be present in a rejected lifecycle result for audit
    # purposes, but it is not a final candidate.  Releasing the sealed
    # holdout is allowed only once the deterministic lifecycle successfully
    # completed, or explicitly retained a viable/risk-passing parent after a
    # rejected refinement proposal.
    final_test_allowed = not development_only and lifecycle_result.status in {
        Sprint6RunStatus.COMPLETED,
        Sprint6RunStatus.REFINEMENT_REJECTED,
    }
    if selected_record is not None:
        selected_strategy = selected_record.version.strategy
        strategy_schema_dict = selected_strategy.model_dump(mode="json")
        val_metrics_dict = metrics_to_dict(selected_record.backtest_result.metrics)

        # Evaluate Train partition
        compiler = StrategyCompiler()
        train_data = dataset_split.generation_view().context.observations
        compiled_train = compiler.compile(selected_strategy, train_data)
        train_res = BacktestEngine(selected_record.effective_backtest_config).run(
            train_data, compiled_train
        )
        train_metrics_dict = metrics_to_dict(train_res.metrics)

        if final_test_allowed:
            # 8. Out-of-Sample (OOS) Holdout Evaluation (PHASE 12)
            logger.info("=== PHASE 12: OUT-OF-SAMPLE VALIDATION ===")
            # Freeze final-test access through the canonical validation-only
            # selection service.  This receives deterministic validation,
            # viability, and risk evidence only; holdout data remains sealed.
            selection = select_validation_candidate(
                [
                    ValidationCandidate(
                        strategy_id=selected_record.version.strategy_id,
                        strategy=selected_strategy,
                        evaluation=selected_record.evaluation,
                        viability=selected_record.viability,
                        risk_assessment=selected_record.risk_assessment,
                    )
                ]
            )
            selection_data = selection.model_dump(mode="json")
            certification.freeze(selection.frozen_selection.strategy_hash)
            certification.begin_test(strategy_execution_fingerprint(selected_strategy))
            test_partition = dataset_split.release_confirmatory_test(selection.frozen_selection, lifecycle=certification)
            test_frame = test_partition.feature_frame
            test_result = BacktestEngine(selected_record.effective_backtest_config).run(
                test_frame, compiler.compile(selected_strategy, test_frame),
                evaluation_start=test_partition.evaluation_start)
            test_metrics_dict = metrics_to_dict(test_result.metrics)
            from risk.evaluator import evaluate_risk
            test_risk_passed = evaluate_risk(test_result, lifecycle_config.risk_policy).passed
            test_passed = (
                test_result.metrics.number_of_trades >= 20
                and test_result.metrics.total_return_pct >= 0.0
                and test_result.metrics.sharpe_ratio is not None
                and test_result.metrics.sharpe_ratio >= 0.5
                and test_risk_passed
            )
        if final_test_allowed and test_passed:
            ledger = HoldoutExposureLedger(exposure_ledger_path or Path.cwd() / 'results' / 'holdout_exposure.sqlite')
            holdout_exposure = ledger.claim(
                dataset_family=f'{symbol.upper()}:{provider_name}', campaign_id=campaign_id,
                strategy_hash=selection.frozen_selection.strategy_hash,
                fingerprint=dataset_split.partition_fingerprints[-1])
        if final_test_allowed and test_passed and holdout_exposure == 'UNSEEN_CONFIRMATORY':
            if strategy_execution_fingerprint(selected_strategy) != certification.strategy_hash:
                raise ValueError('Frozen strategy changed before final holdout.')
            certification.begin_holdout(test_passed=test_passed)
            holdout_access = dataset_split.release_final_holdout(selection.frozen_selection, lifecycle=certification)
            holdout_data = holdout_access.final_holdout.feature_frame
            compiled_holdout = compiler.compile(selected_strategy, holdout_data)
            holdout_res = BacktestEngine(selected_record.effective_backtest_config).run(
                holdout_data,
                compiled_holdout,
                evaluation_start=holdout_access.final_holdout.evaluation_start,
            )
            final_holdout_metrics_dict = metrics_to_dict(holdout_res.metrics)
            holdout_risk_passed = evaluate_risk(holdout_res, lifecycle_config.risk_policy).passed
            holdout_passed = (
                holdout_res.metrics.number_of_trades >= 20
                and holdout_res.metrics.total_return_pct >= 0.0
                and holdout_res.metrics.sharpe_ratio is not None
                and holdout_res.metrics.sharpe_ratio >= 0.5
                and holdout_risk_passed
            )
            certification.complete()

    # 9. Baseline Comparisons (PHASE 13)
    logger.info("=== PHASE 13: BASELINE COMPARISONS ===")
    val_baselines = run_all_baselines(
        dataset_split.adaptive_view().validation.feature_frame,
        base_backtest_config,
        evaluation_start=dataset_split.adaptive_view().validation.evaluation_start,
    )
    baseline_summary = {
        b.kind.value: metrics_to_dict(b.result.metrics) for b in val_baselines
    }

    # 10. Save Artifacts (PHASE 15)
    config_dict = {
        'risk_profile': lifecycle_config.risk_policy.model_dump(mode='json'),
        'risk_profile_sha256': policy_hash(lifecycle_config.risk_policy),
        'viability_profile': lifecycle_config.viability_policy.model_dump(mode='json'),
        "run_id": run_id,
        "symbol": symbol,
        "start_date": start_date,
        "train_end": train_end,
        "validation_end": validation_end,
        "test_end": test_end,
        "initial_capital": initial_capital,
        "commission": commission,
        "slippage": slippage,
        "position_size_pct": position_size_pct,
        "model_provider": model_provider,
        "model_name": model_name,
        "max_debate_rounds": max_debate_rounds,
        "max_refinement_rounds": max_refinement_rounds,
        "mode": mode,
        "seed": seed,
    }
    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, default=str)

    with open(out_path / "strategy.json", "w", encoding="utf-8") as f:
        json.dump(strategy_schema_dict, f, indent=2, default=str)

    with open(out_path / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_metrics_dict, f, indent=2, default=str)

    with open(out_path / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics_dict, f, indent=2, default=str)

    with open(out_path / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics_dict, f, indent=2, default=str)

    with open(out_path / 'final_holdout_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(final_holdout_metrics_dict, f, indent=2, default=str)
    statuses = evidence_status(
        technical=selected_record is not None,
        validation_passed=selected_record.viability.is_viable if selected_record else False,
        test_passed=test_passed, holdout_passed=holdout_passed,
        risk_passed=bool(selected_record and selected_record.risk_assessment.passed
                         and test_risk_passed and holdout_risk_passed))
    with open(out_path / 'certification.json', 'w', encoding='utf-8') as f:
        json.dump({**statuses, 'stage': certification.stage.value,
                   'transitions': certification.transitions,
                   'holdout_exposure': holdout_exposure,
                   'campaign_id': campaign_id}, f, indent=2)
    from dataclasses import asdict
    with open(out_path / 'split_provenance.json', 'w', encoding='utf-8') as f:
        json.dump([asdict(p) for p in dataset_split.partition_fingerprints], f, indent=2, default=str)
    for name, fingerprint in zip(
        ('train', 'validation', 'test', 'final_holdout'), dataset_split.partition_fingerprints
    ):
        metric_path = out_path / f'{name}_metrics.json'
        payload = json.loads(metric_path.read_text(encoding='utf-8'))
        payload['_split'] = {'name': name, **asdict(fingerprint),
                             'strategy_hash': selected_record.version.lineage.execution_fingerprint if selected_record else None,
                             'risk_profile_sha256': policy_hash(lifecycle_config.risk_policy),
                             'cost_model': selected_record.effective_backtest_config.model_dump(mode='json') if selected_record else None}
        metric_path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')

    with open(out_path / "selection.json", "w", encoding="utf-8") as f:
        json.dump(selection_data, f, indent=2, default=str)

    lineage_data = [
        {
            "strategy_id": rec.version.strategy_id,
            "parent_strategy_id": rec.version.parent_strategy_id,
            "generation": rec.version.generation,
            "revision_reason": rec.version.revision_reason,
            "status": "ACCEPTED" if rec.version.strategy_id == lifecycle_result.selected_strategy_id else "SUPERSEDED",
            "metrics": metrics_to_dict(rec.backtest_result.metrics),
        }
        for rec in lifecycle_result.strategy_records
    ]
    with open(out_path / "lineage.json", "w", encoding="utf-8") as f:
        json.dump(lineage_data, f, indent=2, default=str)

    risk_data = {
        'profile': lifecycle_config.risk_policy.model_dump(mode='json'),
        'profile_sha256': policy_hash(lifecycle_config.risk_policy),
        'assessment': selected_record.risk_assessment.model_dump(mode='json') if selected_record else None,
        'viability_evidence': selected_record.viability.model_dump(mode='json') if selected_record else None,
        "status": lifecycle_result.status.value,
        "risk_passed": selected_record.risk_assessment.passed if selected_record else False,
        "viability_passed": selected_record.viability.is_viable if selected_record else False,
        "errors": list(lifecycle_result.errors),
    }
    with open(out_path / "risk.json", "w", encoding="utf-8") as f:
        json.dump(risk_data, f, indent=2, default=str)

    comparison_data = {
        'benchmark_metadata': {b.kind.value: b.metadata for b in val_baselines},
        "strategy": val_metrics_dict,
        "baselines": baseline_summary,
    }
    with open(out_path / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, default=str)

    agent_trace_data = {
        'invocations': client.events,
        "events": [
            {"timestamp": str(e.timestamp), "stage": e.stage, "outcome": e.outcome, "detail": e.detail}
            for e in lifecycle_result.events
        ],
        "debate_sessions_count": len(lifecycle_result.debate_sessions),
        "refinement_proposals_count": len(lifecycle_result.refinement_proposals),
    }
    with open(out_path / "agent_trace.json", "w", encoding="utf-8") as f:
        json.dump(agent_trace_data, f, indent=2, default=str)

    from data.provenance import dataset_manifest, environment_manifest
    provenance = {
        'environment': environment_manifest(workspace_dir),
        'dataset': dataset_manifest(
            validated_df, provider=provider_name,
            query={'symbol': symbol, 'start': start_date, 'end': test_end}, adjusted=None),
        'risk_profile_sha256': policy_hash(lifecycle_config.risk_policy),
        'cost_model_version': 'fill_slippage_commission:v1',
        'strategy_sha256': hashlib.sha256(json.dumps(strategy_schema_dict, sort_keys=True).encode()).hexdigest(),
        'model_provider': getattr(client, 'provider', model_provider),
        'model_id': getattr(client, 'model', model_name),
        'prompt_version': client.prompt_version,
        'development_only': development_only,
        'search_budget': {'strategies': len(lifecycle_result.strategy_records),
                          'debates': len(lifecycle_result.debate_sessions),
                          'refinements': len(lifecycle_result.refinement_proposals),
                          'generator_invocations': sum(e['agent_role'] == 'StrategyProposalDraft' for e in client.events),
                          'total_model_invocations': len(client.events),
                          'structured_failures': sum(not e['schema_valid'] for e in client.events),
                          'sdk_max_retries': 2 if structured_client is None else None},
    }
    (out_path / 'provenance.json').write_text(json.dumps(provenance, indent=2, default=str), encoding='utf-8')
    # 11. Generate Human-Readable Summary (PHASE 16)
    summary_md = f"""# Multi-Agent Research Experiment Summary: {run_id}

- **Run ID:** `{run_id}`
- **Symbol:** `{symbol}`
- **Period:** `{start_date}` to `{test_end}`
- **Architecture Mode:** `{mode}`
- **Model / Provider:** `{model_provider}` / `{model_name}`
- **Status:** `{lifecycle_result.status.value}`

## Strategy Information
- **Strategy ID:** `{lifecycle_result.selected_strategy_id or 'None'}`
- **Name:** {strategy_schema_dict.get('metadata', {}).get('name', 'N/A')}
- **Description:** {strategy_schema_dict.get('metadata', {}).get('description', 'N/A')}

## Performance Summary Across Partitions

| Metric | Train (In-Sample) | Validation | Confirmatory Test |
|---|---|---|---|
| Total Return | {train_metrics_dict.get('total_return_pct', 'N/A')}% | {val_metrics_dict.get('total_return_pct', 'N/A')}% | {test_metrics_dict.get('total_return_pct', 'N/A')}% |
| CAGR | {train_metrics_dict.get('cagr_pct', 'N/A')}% | {val_metrics_dict.get('cagr_pct', 'N/A')}% | {test_metrics_dict.get('cagr_pct', 'N/A')}% |
| Sharpe Ratio | {train_metrics_dict.get('sharpe_ratio', 'N/A')} | {val_metrics_dict.get('sharpe_ratio', 'N/A')} | {test_metrics_dict.get('sharpe_ratio', 'N/A')} |
| Sortino Ratio | {train_metrics_dict.get('sortino_ratio', 'N/A')} | {val_metrics_dict.get('sortino_ratio', 'N/A')} | {test_metrics_dict.get('sortino_ratio', 'N/A')} |
| Max Drawdown | {train_metrics_dict.get('max_drawdown_pct', 'N/A')}% | {val_metrics_dict.get('max_drawdown_pct', 'N/A')}% | {test_metrics_dict.get('max_drawdown_pct', 'N/A')}% |
| Win Rate | {train_metrics_dict.get('win_rate_pct', 'N/A')}% | {val_metrics_dict.get('win_rate_pct', 'N/A')}% | {test_metrics_dict.get('win_rate_pct', 'N/A')}% |
| Profit Factor | {train_metrics_dict.get('profit_factor', 'N/A')} | {val_metrics_dict.get('profit_factor', 'N/A')} | {test_metrics_dict.get('profit_factor', 'N/A')} |
| Trades Count | {train_metrics_dict.get('number_of_trades', 'N/A')} | {val_metrics_dict.get('number_of_trades', 'N/A')} | {test_metrics_dict.get('number_of_trades', 'N/A')} |

## Baseline Comparisons (Validation Partition)

| Strategy Kind | Total Return | Sharpe Ratio | Max Drawdown | Trades |
|---|---|---|---|---|
| Generated Strategy | {val_metrics_dict.get('total_return_pct', 'N/A')}% | {val_metrics_dict.get('sharpe_ratio', 'N/A')} | {val_metrics_dict.get('max_drawdown_pct', 'N/A')}% | {val_metrics_dict.get('number_of_trades', 'N/A')} |
| Conventional Buy & Hold (100% allocation) | {baseline_summary.get('buy_and_hold', {}).get('total_return_pct', 'N/A')}% | {baseline_summary.get('buy_and_hold', {}).get('sharpe_ratio', 'N/A')} | {baseline_summary.get('buy_and_hold', {}).get('max_drawdown_pct', 'N/A')}% | {baseline_summary.get('buy_and_hold', {}).get('number_of_trades', 'N/A')} |
| Exposure-matched Buy & Hold ({position_size_pct}% allocation) | {baseline_summary.get('exposure_matched_buy_and_hold', {}).get('total_return_pct', 'N/A')}% | {baseline_summary.get('exposure_matched_buy_and_hold', {}).get('sharpe_ratio', 'N/A')} | {baseline_summary.get('exposure_matched_buy_and_hold', {}).get('max_drawdown_pct', 'N/A')}% | {baseline_summary.get('exposure_matched_buy_and_hold', {}).get('number_of_trades', 'N/A')} |
| SMA Crossover | {baseline_summary.get('sma_crossover', {}).get('total_return_pct', 'N/A')}% | {baseline_summary.get('sma_crossover', {}).get('sharpe_ratio', 'N/A')} | {baseline_summary.get('sma_crossover', {}).get('max_drawdown_pct', 'N/A')}% | {baseline_summary.get('sma_crossover', {}).get('number_of_trades', 'N/A')} |
| RSI Mean Reversion | {baseline_summary.get('rsi_mean_reversion', {}).get('total_return_pct', 'N/A')}% | {baseline_summary.get('rsi_mean_reversion', {}).get('sharpe_ratio', 'N/A')} | {baseline_summary.get('rsi_mean_reversion', {}).get('max_drawdown_pct', 'N/A')}% | {baseline_summary.get('rsi_mean_reversion', {}).get('number_of_trades', 'N/A')} |
| Momentum | {baseline_summary.get('momentum', {}).get('total_return_pct', 'N/A')}% | {baseline_summary.get('momentum', {}).get('sharpe_ratio', 'N/A')} | {baseline_summary.get('momentum', {}).get('max_drawdown_pct', 'N/A')}% | {baseline_summary.get('momentum', {}).get('number_of_trades', 'N/A')} |

## Viability & Risk Assessment
- **Validation Performance Screening:** {'PASS' if selected_record and selected_record.viability.is_viable else 'FAIL'}
- **Validation Risk Assessment:** {'PASS' if selected_record and selected_record.risk_assessment.passed else 'FAIL'}
- **Errors / Rejections:** {', '.join(lifecycle_result.errors) if lifecycle_result.errors else 'None'}

## Strategy Lineage
- Generations produced: {len(lifecycle_result.strategy_records)}
- Debate sessions: {len(lifecycle_result.debate_sessions)}
- Refinement proposals: {len(lifecycle_result.refinement_proposals)}
"""
    with open(out_path / "summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)
        f.write('\n## Certification\n\n' + json.dumps(statuses, indent=2))
        f.write('\n\n## Final holdout\n\n' + json.dumps(final_holdout_metrics_dict, indent=2))
        f.write('\n\nTechnical indicator baselines include stop/target risk overlays. '
                'All benchmarks charge the configured costs and use next-bar whole-share fills.\n')

    return {
        "run_id": run_id,
        "output_dir": str(out_path),
        "status": lifecycle_result.status.value,
        'development_only': development_only,
        "selected_strategy_id": lifecycle_result.selected_strategy_id,
        "selected_strategy": strategy_schema_dict,
        "train_metrics": train_metrics_dict,
        "validation_metrics": val_metrics_dict,
        "test_metrics": test_metrics_dict,
        'final_holdout_metrics': final_holdout_metrics_dict,
        'certification': statuses,
        'risk': risk_data,
        'provenance': provenance,
        'lineage': lineage_data,
        'execution_config': selected_record.effective_backtest_config.model_dump(mode='json') if selected_record else None,
        'invocation_audit': client.events,
        "selection": selection_data,
        "baselines": baseline_summary,
        "lifecycle_result": lifecycle_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Authoritative Multi-Agent Research Experiment.")
    parser.add_argument("--symbol", default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--start-date", default="2020-01-01", help="Dataset start date")
    parser.add_argument("--train-end", default="2023-12-31", help="Training partition end date")
    parser.add_argument("--validation-end", default="2024-12-31", help="Validation partition end date")
    parser.add_argument("--test-end", default="2025-12-31", help="Test partition end date")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Initial portfolio capital")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission percentage")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage percentage")
    parser.add_argument(
        "--position-size-pct",
        type=float,
        default=10.0,
        help="Deterministic position size as a percentage of capital",
    )
    parser.add_argument("--model-provider", default="google", help="LLM Provider (google, openai, anthropic)")
    parser.add_argument("--model-name", default="gemini-2.5-flash", help="Model name")
    parser.add_argument("--max-debate-rounds", type=int, default=1, help="Debate rounds")
    parser.add_argument("--max-refinement-rounds", type=int, default=1, help="Refinement rounds")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--mode", default="MULTI_AGENT_FULL", choices=["MULTI_AGENT_FULL", "SINGLE_AGENT", "MULTI_AGENT_NO_DEBATE", "MULTI_AGENT_DEBATE", "MULTI_AGENT_RISK", "BOTH"])
    parser.add_argument('--development-only', action='store_true', help='Never release confirmatory test or final holdout; use for architecture studies.')
    parser.add_argument("--output-dir", default=None, help="Custom output directory")

    args = parser.parse_args()

    if args.mode == "BOTH":
        logger.info("Executing Controlled A/B Comparison: SINGLE_AGENT vs MULTI_AGENT_FULL")
        single_res = run_e2e_research(
            symbol=args.symbol,
            start_date=args.start_date,
            train_end=args.train_end,
            validation_end=args.validation_end,
            test_end=args.test_end,
            initial_capital=args.initial_capital,
            commission=args.commission,
            slippage=args.slippage,
            position_size_pct=args.position_size_pct,
            model_provider=args.model_provider,
            model_name=args.model_name,
            max_debate_rounds=0,
            max_refinement_rounds=0,
            seed=args.seed,
            mode="SINGLE_AGENT",
            development_only=args.development_only,
            output_dir=Path("results") / "e2e" / f"compare_single_{args.symbol}",
        )
        multi_res = run_e2e_research(
            symbol=args.symbol,
            start_date=args.start_date,
            train_end=args.train_end,
            validation_end=args.validation_end,
            test_end=args.test_end,
            initial_capital=args.initial_capital,
            commission=args.commission,
            slippage=args.slippage,
            position_size_pct=args.position_size_pct,
            model_provider=args.model_provider,
            model_name=args.model_name,
            max_debate_rounds=args.max_debate_rounds,
            max_refinement_rounds=args.max_refinement_rounds,
            seed=args.seed,
            mode="MULTI_AGENT_FULL",
            development_only=args.development_only,
            output_dir=Path("results") / "e2e" / f"compare_multi_{args.symbol}",
        )
        print("\n=== A/B COMPARISON: SINGLE AGENT VS MULTI-AGENT ===")
        print(f"{'Metric':<25} {'Single Agent':<15} {'Multi-Agent':<15}")
        print("-" * 55)
        for m in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct", "profit_factor", "number_of_trades"):
            s_val = single_res["validation_metrics"].get(m, "N/A")
            m_val = multi_res["validation_metrics"].get(m, "N/A")
            print(f"{m:<25} {str(s_val):<15} {str(m_val):<15}")
    else:
        run_e2e_research(
            symbol=args.symbol,
            start_date=args.start_date,
            train_end=args.train_end,
            validation_end=args.validation_end,
            test_end=args.test_end,
            initial_capital=args.initial_capital,
            commission=args.commission,
            slippage=args.slippage,
            position_size_pct=args.position_size_pct,
            model_provider=args.model_provider,
            model_name=args.model_name,
            max_debate_rounds=args.max_debate_rounds,
            max_refinement_rounds=args.max_refinement_rounds,
            seed=args.seed,
            mode=args.mode,
            development_only=args.development_only,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
