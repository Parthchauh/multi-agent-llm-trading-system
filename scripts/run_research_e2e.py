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

# Add workspace to path if needed
workspace_dir = str(Path(__file__).resolve().parent.parent)
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from backtesting.engine import BacktestEngine
from backtesting.models import BacktestConfig
from data.provider import MarketDataProvider, YahooFinanceProvider
from data.research_split import (
    FrozenStrategySelection,
    ResearchDatasetSplit,
    ResearchPartition,
    ResearchPartitionRole,
)
from data.schema import validate_ohlcv
from evaluation.baselines import run_all_baselines
from evaluation.models import (
    EvaluationScope,
    ViabilityPolicy,
)
from risk import (
    RiskPolicy,
)
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
    model_provider: str = "google",
    model_name: str = "gemini-2.5-flash",
    structured_client: StructuredOutputClient | None = None,
    max_debate_rounds: int = 1,
    max_refinement_rounds: int = 1,
    seed: int = 42,
    mode: str = "MULTI_AGENT_FULL",
    output_dir: Path | str | None = None,
    data_provider: MarketDataProvider | None = None,
    df: pd.DataFrame | None = None,
    risk_policy: RiskPolicy | None = None,
) -> dict[str, Any]:
    """Execute the complete authoritative multi-agent research lifecycle."""

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
        position_size_pct=50.0,
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
        )
        client: StructuredOutputClient = LangChainStructuredOutputClient.from_provider_client(base_client)
    else:
        client = structured_client

    # 5. Build Multi-Agent Graphs
    regime_agent = RegimeAnalystAgent(client)
    strategy_gen_agent = StrategyGeneratorAgent(client, max_attempts=3)

    initial_graph = Sprint4ResearchGraph(
        regime_agent=regime_agent,
        strategy_generator=strategy_gen_agent,
        compiler=StrategyCompiler(),
        engine_factory=BacktestEngine,
    )

    is_multi_agent = mode.upper() == "MULTI_AGENT_FULL"
    eff_debate_rounds = max_debate_rounds if is_multi_agent else 0
    eff_refinement_rounds = max_refinement_rounds if is_multi_agent else 0

    lifecycle_config = Sprint6LifecycleConfig(
        max_debate_rounds=eff_debate_rounds,
        max_refinement_rounds=eff_refinement_rounds,
        viability_policy=ViabilityPolicy(
            policy_id=f"viability-{symbol}",
            required_scope=EvaluationScope.VALIDATION,
            minimum_trade_count=1,
        ),
        risk_policy=risk_policy or RiskPolicy(),
    )

    lifecycle_kwargs: dict[str, Any] = {}
    if is_multi_agent and eff_debate_rounds > 0:
        lifecycle_kwargs.update(
            bull_researcher=BullResearcherAgent(client),
            bear_researcher=BearResearcherAgent(client),
            debate_synthesizer=DebateSynthesizerAgent(client),
        )
    if is_multi_agent and eff_refinement_rounds > 0:
        lifecycle_kwargs.update(
            risk_manager=RiskManagerAgent(client),
            performance_analyst=PerformanceAnalystAgent(client),
            critic=StrategyCriticAgent(client),
            refinement_agent=RefinementAgent(client),
        )

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
    strategy_schema_dict: dict[str, Any] = {}
    selected_strategy: StrategySchema | None = None

    # A strategy can be present in a rejected lifecycle result for audit
    # purposes, but it is not a final candidate.  Releasing the sealed
    # holdout is allowed only once the deterministic lifecycle successfully
    # completed, or explicitly retained a viable/risk-passing parent after a
    # rejected refinement proposal.
    final_test_allowed = lifecycle_result.status in {
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
            frozen_sel = FrozenStrategySelection(
                strategy_id=selected_record.version.strategy_id,
                strategy_hash=selected_record.version.strategy_hash,
                frozen_at=_utc_now(),
            )
            holdout_access = dataset_split.release_final_holdout(frozen_sel)
            holdout_data = holdout_access.final_holdout.feature_frame
            compiled_holdout = compiler.compile(selected_strategy, holdout_data)
            holdout_res = BacktestEngine(selected_record.effective_backtest_config).run(
                holdout_data,
                compiled_holdout,
                evaluation_start=holdout_access.final_holdout.evaluation_start,
            )
            test_metrics_dict = metrics_to_dict(holdout_res.metrics)

    # 9. Baseline Comparisons (PHASE 13)
    logger.info("=== PHASE 13: BASELINE COMPARISONS ===")
    val_baselines = run_all_baselines(
        dataset_split.development_view().validation.feature_frame,
        base_backtest_config,
        evaluation_start=dataset_split.development_view().validation.evaluation_start,
    )
    baseline_summary = {
        b.kind.value: metrics_to_dict(b.result.metrics) for b in val_baselines
    }

    # 10. Save Artifacts (PHASE 15)
    config_dict = {
        "run_id": run_id,
        "symbol": symbol,
        "start_date": start_date,
        "train_end": train_end,
        "validation_end": validation_end,
        "test_end": test_end,
        "initial_capital": initial_capital,
        "commission": commission,
        "slippage": slippage,
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
        "status": lifecycle_result.status.value,
        "risk_passed": selected_record.risk_assessment.passed if selected_record else False,
        "viability_passed": selected_record.viability.is_viable if selected_record else False,
        "errors": list(lifecycle_result.errors),
    }
    with open(out_path / "risk.json", "w", encoding="utf-8") as f:
        json.dump(risk_data, f, indent=2, default=str)

    comparison_data = {
        "strategy": val_metrics_dict,
        "baselines": baseline_summary,
    }
    with open(out_path / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, default=str)

    agent_trace_data = {
        "events": [
            {"timestamp": str(e.timestamp), "stage": e.stage, "outcome": e.outcome, "detail": e.detail}
            for e in lifecycle_result.events
        ],
        "debate_sessions_count": len(lifecycle_result.debate_sessions),
        "refinement_proposals_count": len(lifecycle_result.refinement_proposals),
    }
    with open(out_path / "agent_trace.json", "w", encoding="utf-8") as f:
        json.dump(agent_trace_data, f, indent=2, default=str)

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

| Metric | Train (In-Sample) | Validation | Out-of-Sample (Holdout) |
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
| Buy & Hold | {baseline_summary.get('buy_and_hold', {}).get('total_return_pct', 'N/A')}% | {baseline_summary.get('buy_and_hold', {}).get('sharpe_ratio', 'N/A')} | {baseline_summary.get('buy_and_hold', {}).get('max_drawdown_pct', 'N/A')}% | {baseline_summary.get('buy_and_hold', {}).get('number_of_trades', 'N/A')} |
| SMA Crossover | {baseline_summary.get('sma_crossover', {}).get('total_return_pct', 'N/A')}% | {baseline_summary.get('sma_crossover', {}).get('sharpe_ratio', 'N/A')} | {baseline_summary.get('sma_crossover', {}).get('max_drawdown_pct', 'N/A')}% | {baseline_summary.get('sma_crossover', {}).get('number_of_trades', 'N/A')} |
| RSI Mean Reversion | {baseline_summary.get('rsi_mean_reversion', {}).get('total_return_pct', 'N/A')}% | {baseline_summary.get('rsi_mean_reversion', {}).get('sharpe_ratio', 'N/A')} | {baseline_summary.get('rsi_mean_reversion', {}).get('max_drawdown_pct', 'N/A')}% | {baseline_summary.get('rsi_mean_reversion', {}).get('number_of_trades', 'N/A')} |
| Momentum | {baseline_summary.get('momentum', {}).get('total_return_pct', 'N/A')}% | {baseline_summary.get('momentum', {}).get('sharpe_ratio', 'N/A')} | {baseline_summary.get('momentum', {}).get('max_drawdown_pct', 'N/A')}% | {baseline_summary.get('momentum', {}).get('number_of_trades', 'N/A')} |

## Viability & Risk Assessment
- **Viability Gate:** {'PASS' if selected_record and selected_record.viability.is_viable else 'FAIL'}
- **Risk Assessment:** {'PASS' if selected_record and selected_record.risk_assessment.passed else 'FAIL'}
- **Errors / Rejections:** {', '.join(lifecycle_result.errors) if lifecycle_result.errors else 'None'}

## Strategy Lineage
- Generations produced: {len(lifecycle_result.strategy_records)}
- Debate sessions: {len(lifecycle_result.debate_sessions)}
- Refinement proposals: {len(lifecycle_result.refinement_proposals)}
"""
    with open(out_path / "summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    return {
        "run_id": run_id,
        "output_dir": str(out_path),
        "status": lifecycle_result.status.value,
        "selected_strategy_id": lifecycle_result.selected_strategy_id,
        "selected_strategy": strategy_schema_dict,
        "train_metrics": train_metrics_dict,
        "validation_metrics": val_metrics_dict,
        "test_metrics": test_metrics_dict,
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
    parser.add_argument("--model-provider", default="google", help="LLM Provider (google, openai, anthropic)")
    parser.add_argument("--model-name", default="gemini-2.5-flash", help="Model name")
    parser.add_argument("--max-debate-rounds", type=int, default=1, help="Debate rounds")
    parser.add_argument("--max-refinement-rounds", type=int, default=1, help="Refinement rounds")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--mode", default="MULTI_AGENT_FULL", choices=["MULTI_AGENT_FULL", "SINGLE_AGENT", "BOTH"])
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
            model_provider=args.model_provider,
            model_name=args.model_name,
            max_debate_rounds=0,
            max_refinement_rounds=0,
            seed=args.seed,
            mode="SINGLE_AGENT",
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
            model_provider=args.model_provider,
            model_name=args.model_name,
            max_debate_rounds=args.max_debate_rounds,
            max_refinement_rounds=args.max_refinement_rounds,
            seed=args.seed,
            mode="MULTI_AGENT_FULL",
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
            model_provider=args.model_provider,
            model_name=args.model_name,
            max_debate_rounds=args.max_debate_rounds,
            max_refinement_rounds=args.max_refinement_rounds,
            seed=args.seed,
            mode=args.mode,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
