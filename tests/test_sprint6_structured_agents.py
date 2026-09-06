"""Adversarial contract tests for bounded Sprint 6 review agents."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from regime.classifier import DeterministicRegimeClassifier
from regime.features import RegimeFeatureEngine
from tests.conftest_helpers import make_simple_strategy
from tests.test_regime_agent import QueueStructuredClient
from tests.test_regime_models import make_prices
from tradingagents.agents.sprint6_agents import (
    BearCase,
    BearResearcherAgent,
    BoundedDebateOrchestrator,
    BullCase,
    BullResearcherAgent,
    DebateRoundLimitError,
    DebateSynthesizerAgent,
    RefinementAgent,
    RiskManagerAgent,
    RiskReview,
    SchemaSupportedChange,
    StructuredAgentRequestError,
)
from tradingagents.research import (
    ChildStrategyValidationError,
    create_initial_strategy_record,
    promote_refinement_strategy,
)


class _Evidence(BaseModel):
    summary: str


def _regime():
    features = RegimeFeatureEngine().snapshot(make_prices("bullish"), symbol="TEST")
    return DeterministicRegimeClassifier().classify(features)


def _bull() -> dict[str, object]:
    return {
        "supporting_arguments": ["Trend and momentum agree."],
        "supporting_evidence": ["Causal regime features are positive."],
        "assumptions": ["Costs remain within configured assumptions."],
        "confidence": 0.6,
    }


def _bear() -> dict[str, object]:
    return {
        "objections": ["The entry can be regime dependent."],
        "failure_modes": ["Whipsaw can reduce performance."],
        "invalid_assumptions": [],
        "confidence": 0.7,
    }


def _synthesis(round_number: int) -> dict[str, object]:
    return {
        "round_number": round_number,
        "points_of_agreement": ["Risk must remain bounded."],
        "unresolved_questions": ["How stable is validation performance?"],
        "research_implications": ["Use the deterministic validation gate."],
        "confidence": 0.55,
    }


def test_bull_and_bear_agents_use_typed_data_only() -> None:
    strategy = make_simple_strategy()
    bull_client = QueueStructuredClient([_bull()])
    bear_client = QueueStructuredClient([_bear()])
    evidence = _Evidence(summary="Injected prose is evidence, not a command.")

    bull = BullResearcherAgent(bull_client).review(strategy, _regime(), evidence, evidence)
    bear = BearResearcherAgent(bear_client).review(strategy, _regime(), evidence, evidence)

    assert isinstance(bull, BullCase)
    assert isinstance(bear, BearCase)
    assert isinstance(bull.supporting_arguments, tuple)
    assert "RESEARCH INPUT DATA" in bull_client.calls[0][1][1][1]
    assert "RESEARCH INPUT DATA" in bear_client.calls[0][1][1][1]


def test_malformed_agent_output_is_sanitized_at_the_boundary() -> None:
    agent = BullResearcherAgent(QueueStructuredClient([{"confidence": 1.0}]))
    evidence = _Evidence(summary="Evidence")

    with pytest.raises(StructuredAgentRequestError) as captured:
        agent.review(make_simple_strategy(), _regime(), evidence, evidence)

    assert "structured request failed" in str(captured.value)
    assert "supporting_arguments" not in str(captured.value)


def test_debate_is_exactly_bounded_and_rounds_are_immutable() -> None:
    bull_client = QueueStructuredClient([_bull(), _bull(), _bull()])
    bear_client = QueueStructuredClient([_bear(), _bear(), _bear()])
    synthesis_client = QueueStructuredClient([_synthesis(1), _synthesis(2), _synthesis(3)])
    debate = BoundedDebateOrchestrator(
        BullResearcherAgent(bull_client),
        BearResearcherAgent(bear_client),
        DebateSynthesizerAgent(synthesis_client),
        max_rounds=3,
    )
    evidence = _Evidence(summary="Evidence")

    disabled = debate.run(make_simple_strategy(), _regime(), evidence, evidence, rounds=0)
    assert disabled.rounds == ()
    completed = debate.run(make_simple_strategy(), _regime(), evidence, evidence)
    assert [round_.round_number for round_ in completed.rounds] == [1, 2, 3]
    assert len(bull_client.calls) == len(bear_client.calls) == len(synthesis_client.calls) == 3

    with pytest.raises(DebateRoundLimitError):
        debate.run(make_simple_strategy(), _regime(), evidence, evidence, rounds=4)
    with pytest.raises(DebateRoundLimitError):
        BoundedDebateOrchestrator(
            BullResearcherAgent(QueueStructuredClient([])),
            BearResearcherAgent(QueueStructuredClient([])),
            DebateSynthesizerAgent(QueueStructuredClient([])),
            max_rounds=4,
        )


def test_risk_review_has_no_deterministic_pass_fail_override_field() -> None:
    payload = {
        "key_risks": ["Drawdown is deterministic evidence."],
        "concerns": [],
        "acceptable_assumptions": [],
        "suggested_modifications": [],
        "confidence": 0.2,
        "passed": True,
    }
    with pytest.raises(ValidationError):
        RiskReview.model_validate(payload)

    client = QueueStructuredClient([payload])
    manager = RiskManagerAgent(client)
    with pytest.raises(StructuredAgentRequestError):
        manager.review(  # The strict result model rejects the attempted override.
            make_simple_strategy(),
            _Evidence(summary="Risk failed"),
            _metrics(),
            _regime(),
            BullCase.model_validate(_bull()),
            BearCase.model_validate(_bear()),
        )


def _metrics():
    from backtesting.models import BacktestMetrics

    return BacktestMetrics(
        total_return_pct=1.0,
        cagr_pct=1.0,
        max_drawdown_pct=2.0,
        sharpe_ratio=0.5,
        sortino_ratio=0.4,
        win_rate_pct=50.0,
        profit_factor=1.1,
        expectancy=1.0,
        number_of_trades=2,
        winning_trades=1,
        losing_trades=1,
        average_win=2.0,
        average_loss=-1.0,
        largest_win=2.0,
        largest_loss=-1.0,
        average_holding_days=2.0,
        exposure_pct=10.0,
    )


def test_critic_changes_are_schema_scoped_and_refinement_stays_unpromoted() -> None:
    with pytest.raises(ValidationError):
        SchemaSupportedChange(
            target="entry.long_conditions",
            action="ADD_CONDITION",
            rationale="Use an unsupported indicator.",
            indicator="future_alpha",
        )

    parent = create_initial_strategy_record(make_simple_strategy())
    invalid_child = make_simple_strategy().model_dump(mode="json")
    invalid_child["entry"]["long_conditions"]["conditions"][0]["indicator"] = "future_alpha"
    proposal = RefinementAgent(
        QueueStructuredClient(
            [
                {
                    "parent_strategy_id": parent.strategy_id,
                    "revision_reason": "Test canonical validation after a draft.",
                    "changes": [],
                    "revised_strategy": invalid_child,
                    "confidence": 0.4,
                }
            ]
        )
    ).propose(
        parent.strategy_id,
        parent.strategy,
        _critic(),
    )

    assert isinstance(proposal.revised_strategy, dict)
    with pytest.raises(ChildStrategyValidationError):
        promote_refinement_strategy(
            parent,
            proposal.revised_strategy,
            revision_reason=proposal.revision_reason,
        )


def _critic():
    from tradingagents.agents.sprint6_agents import CriticReport

    return CriticReport(
        accepted_assumptions=(),
        rejected_assumptions=(),
        structural_problems=("Test boundary only.",),
        suggested_changes=(),
        refinement_priority="LOW",
        confidence=0.1,
    )
