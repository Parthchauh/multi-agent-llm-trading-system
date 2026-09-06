"""tests/test_research_e2e.py
==========================
Automated acceptance tests for the End-to-End Research and Paper-Trading lifecycle.

Covers:
- test_e2e_runner_configuration
- test_e2e_invalid_data_fails
- test_e2e_invalid_strategy_never_backtests
- test_e2e_risk_failure_not_overridden
- test_e2e_holdout_not_exposed
- test_e2e_result_files_written
- test_e2e_deterministic_quant_outputs
- test_paper_trading_pipeline
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtesting.models import BacktestConfig
from data.research_split import FrozenStrategySelection
from paper_trading.runner import PaperTradingRunner
from risk import RiskPolicy
from scripts.run_research_e2e import (
    build_research_dataset_split,
    run_e2e_research,
)
from tests.conftest_helpers import make_ohlcv, make_simple_strategy
from tests.test_regime_agent import QueueStructuredClient, _valid_assessment
from tests.test_strategy_generator_agent import _draft
from tradingagents.graph.sprint6_lifecycle import Sprint6RunStatus


class MockLifecycleClient:
    """Deterministic queue-based StructuredOutputClient for complete E2E runs."""

    provider = "fake"
    model = "fake-model"
    temperature = 0.0
    seed = 42

    def __init__(self, *, raw_strategy: dict[str, Any] | None = None) -> None:
        strat = raw_strategy or make_simple_strategy(entry_value=50.0).model_dump(mode="json")
        child_strat = make_simple_strategy(entry_value=52.0).model_dump(mode="json")

        self.responses = [
            # 1. Regime assessment
            _valid_assessment(),
            # 2. Strategy draft
            _draft(strat),
            # 3. Bull researcher
            {"arguments": ["Strong trend continuation expected"], "catalysts": ["Earnings momentum"], "confidence": 0.8},
            # 4. Bear researcher
            {"objections": ["Overextended"], "failure_modes": ["Reversal"], "invalid_assumptions": ["Low vol"], "confidence": 0.7},
            # 5. Debate synthesis
            {"round_number": 1, "points_of_agreement": ["Trend positive"], "unresolved_questions": ["Reversal timing"], "research_implications": ["Keep tight stops"], "confidence": 0.75},
            # 6. Risk manager
            {"key_risks": ["Drawdown risk"], "confidence": 0.8},
            # 7. Performance analyst
            {"strengths": ["Positive return"], "weaknesses": [], "performance_concentration": "Low", "trade_quality": "High", "warnings": [], "confidence": 0.8},
            # 8. Critic
            {"accepted_assumptions": ["Trend following"], "rejected_assumptions": [], "structural_problems": [], "suggested_changes": ["Update entry"], "refinement_priority": "LOW", "confidence": 0.8},
            # 9. Refiner
            {
                "parent_strategy_id": "placeholder",
                "revision_reason": "Refined RSI entry threshold",
                "changes": ["entry threshold adjusted"],
                "revised_strategy": child_strat,
                "confidence": 0.8,
            },
        ]
        self.calls = []

    def invoke(self, schema, messages):
        self.calls.append((schema, messages))
        item = self.responses.pop(0)
        # Adapt refiner parent strategy id if present in message payload
        if schema.__name__ == "RefinementProposalDraft":
            try:
                prompt_data = json.loads(messages[1][1].split("\n", 1)[1])
                item["parent_strategy_id"] = prompt_data["parent_strategy_id"]
            except Exception:
                pass
        if isinstance(item, schema):
            return item
        return schema.model_validate(item)


def test_e2e_runner_configuration(tmp_path: Path) -> None:
    df = make_ohlcv(n=350, seed=1)
    client = MockLifecycleClient()
    result = run_e2e_research(
        symbol="AAPL",
        df=df,
        structured_client=client,
        mode="SINGLE_AGENT",
        output_dir=tmp_path / "test_cfg",
    )
    assert result["status"] == "COMPLETED"
    assert result["selected_strategy_id"] is not None
    assert "total_return_pct" in result["validation_metrics"]
    assert "total_return_pct" in result["test_metrics"]
    config = json.loads((tmp_path / "test_cfg" / "config.json").read_text(encoding="utf-8"))
    assert config["position_size_pct"] == 10.0


def test_e2e_invalid_data_fails(tmp_path: Path) -> None:
    bad_df = pd.DataFrame({"Open": [-10.0, 20.0], "Close": [15.0, 25.0]})
    with pytest.raises(Exception):
        run_e2e_research(
            symbol="AAPL",
            df=bad_df,
            output_dir=tmp_path / "bad_data",
        )


def test_e2e_invalid_strategy_never_backtests(tmp_path: Path) -> None:
    invalid_strat = make_simple_strategy().model_dump(mode="json")
    # Illegal indicator outside allowlist
    invalid_strat["entry"]["long_conditions"]["conditions"][0]["indicator"] = "forbidden_indicator_xyz"

    client = QueueStructuredClient([
        _valid_assessment(),
        _draft(invalid_strat),
        _draft(invalid_strat),
        _draft(invalid_strat),
    ])
    df = make_ohlcv(n=300, seed=2)
    result = run_e2e_research(
        symbol="AAPL",
        df=df,
        structured_client=client,
        mode="SINGLE_AGENT",
        output_dir=tmp_path / "invalid_strat",
    )
    assert result["status"] == Sprint6RunStatus.FAILED_INITIAL_PIPELINE.value
    assert result["selected_strategy_id"] is None


def test_e2e_risk_failure_not_overridden(tmp_path: Path) -> None:
    """When risk policy rejects (e.g. max drawdown limit exceeded), LLM review is not called."""
    client = MockLifecycleClient()
    df = make_ohlcv(n=300, seed=3)
    # Set an impossible max drawdown limit so risk engine rejects
    tight_risk_policy = RiskPolicy(max_drawdown_pct=0.0001)

    result = run_e2e_research(
        symbol="AAPL",
        df=df,
        structured_client=client,
        mode="MULTI_AGENT_FULL",
        output_dir=tmp_path / "risk_rej",
        risk_policy=tight_risk_policy,
    )
    assert result["status"] == Sprint6RunStatus.RISK_REJECTED.value
    assert result["test_metrics"] == {}
    # Only the regime and generation requests occur; review agents cannot be
    # called after an authoritative deterministic risk rejection.
    assert len(client.calls) == 2


def test_e2e_holdout_not_exposed() -> None:
    """Holdout access requires a valid FrozenStrategySelection."""
    df = make_ohlcv(n=300, seed=4)
    split = build_research_dataset_split(df, symbol="AAPL")

    # Access without frozen selection must be impossible or raise
    with pytest.raises(Exception):
        split.release_final_holdout(None)  # type: ignore[arg-type]

    # Valid selection releases holdout
    sel = FrozenStrategySelection(strategy_id="strat-1", strategy_hash="hash1", frozen_at=pd.Timestamp.utcnow())
    holdout = split.release_final_holdout(sel)
    assert holdout.final_holdout.observations is not None


def test_e2e_result_files_written(tmp_path: Path) -> None:
    df = make_ohlcv(n=300, seed=5)
    client = MockLifecycleClient()
    out = tmp_path / "artifacts_test"
    run_e2e_research(
        symbol="AAPL",
        df=df,
        structured_client=client,
        mode="MULTI_AGENT_FULL",
        output_dir=out,
    )
    required_files = [
        "config.json",
        "strategy.json",
        "train_metrics.json",
        "validation_metrics.json",
        "test_metrics.json",
        "lineage.json",
        "risk.json",
        "comparison.json",
        "agent_trace.json",
        "summary.md",
    ]
    for filename in required_files:
        p = out / filename
        assert p.exists(), f"Expected artifact {filename} was not created"
        assert p.stat().st_size > 0, f"Artifact {filename} is empty"


def test_e2e_deterministic_quant_outputs(tmp_path: Path) -> None:
    """Two identical runs with deterministic mocks produce identical metrics."""
    df = make_ohlcv(n=300, seed=6)
    client1 = MockLifecycleClient()
    client2 = MockLifecycleClient()

    r1 = run_e2e_research(symbol="AAPL", df=df, structured_client=client1, mode="SINGLE_AGENT", output_dir=tmp_path / "det1")
    r2 = run_e2e_research(symbol="AAPL", df=df, structured_client=client2, mode="SINGLE_AGENT", output_dir=tmp_path / "det2")

    assert r1["validation_metrics"] == r2["validation_metrics"]
    assert r1["test_metrics"] == r2["test_metrics"]


def test_paper_trading_pipeline(tmp_path: Path) -> None:
    """Test the complete paper trading simulated execution pipeline."""
    strategy = make_simple_strategy(entry_value=50.0)
    df = make_ohlcv(n=80, seed=7)

    journal_file = tmp_path / "paper_journal.jsonl"
    runner = PaperTradingRunner(
        strategy=strategy,
        config=BacktestConfig(initial_capital=50_000.0, commission_pct=0.001, slippage_pct=0.0005, position_size_pct=40.0),
        journal_path=journal_file,
    )

    # Process bars iteratively
    fills = []
    for i in range(50, len(df)):
        sub_df = df.iloc[: i + 1]
        fill, state = runner.process_bar("AAPL", sub_df)
        if fill is not None:
            fills.append(fill)

    assert state.total_equity > 0
    assert journal_file.exists()
    assert len(runner.journal.entries) > 0
