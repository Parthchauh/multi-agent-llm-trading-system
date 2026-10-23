"""Hand-computed ledgers, execution parity, recovery, and certification boundaries."""

from datetime import datetime

import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from backtesting.models import BacktestConfig
from evaluation.certification import EvaluationLifecycle, evidence_status
from paper_trading.models import PaperFill, SignalType
from paper_trading.portfolio import PaperPortfolio
from paper_trading.runner import PaperTradingRunner
from risk.models import RiskPolicy
from risk.profiles import require_complete_policy, risk_profile
from tests.conftest_helpers import make_ohlcv, make_simple_strategy


@pytest.mark.parametrize("slip,fee", [(0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (1.0, 2.0)])
def test_manual_round_trip(slip, fee):
    portfolio = PaperPortfolio(1000.0)
    buy = PaperFill(
        fill_id="buy",
        symbol="AAA",
        fill_time=datetime(2020, 1, 1),
        signal_type=SignalType.ENTER_LONG,
        fill_price=100.0 + slip,
        quantity=5.0,
        commission=fee,
        slippage=slip * 5.0,
        reference_price=100.0,
    )
    portfolio.process_fill(buy, strategy_id="frozen")
    assert portfolio.cash == 1000.0 - 5.0 * (100.0 + slip) - fee
    state = portfolio.snapshot(datetime(2020, 1, 1), {"AAA": 100.0 + slip})
    assert state.total_equity - 1000.0 == pytest.approx(state.realized_pnl + state.unrealized_pnl)
    sell = PaperFill(
        fill_id="sell",
        symbol="AAA",
        fill_time=datetime(2020, 1, 2),
        signal_type=SignalType.EXIT_LONG,
        fill_price=110.0 - slip,
        quantity=5.0,
        commission=fee,
        slippage=slip * 5.0,
        reference_price=110.0,
    )
    portfolio.process_fill(sell, strategy_id="frozen")
    expected_pnl = 5.0 * (10.0 - 2.0 * slip) - 2.0 * fee
    assert portfolio.cash == 1000.0 + expected_pnl
    assert portfolio.realized_pnl == expected_pnl
    assert portfolio.closed_positions[0].realized_pnl == expected_pnl
    with pytest.raises(ValueError, match="Duplicate"):
        portfolio.process_fill(sell, strategy_id="frozen")


class ScriptedSignals:
    def should_enter(self, i, data):
        return i == 0

    def should_exit(self, i, data):
        return i == 2

    def get_stop_price(self, entry_price, i, data):
        return 95.0

    def get_take_profit_price(self, entry_price, stop, i, data):
        return 110.0


@pytest.mark.parametrize(
    "scenario", ["signal", "stop", "target", "gap", "collision", "holding", "eof", "repeated"]
)
def test_golden_execution_parity(monkeypatch, scenario):
    from strategies.compiler import StrategyCompiler

    monkeypatch.setattr(StrategyCompiler, "compile", lambda *args: ScriptedSignals())
    df = pd.DataFrame(
        {
            "Open": [100.0] * 5,
            "High": [102.0] * 5,
            "Low": [98.0] * 5,
            "Close": [101.0] * 5,
            "Volume": [1000.0] * 5,
        },
        index=pd.date_range("2020-01-01", periods=5),
    )
    if scenario == "stop":
        df.loc[df.index[2], "Low"] = 94.0
    if scenario == "target":
        df.loc[df.index[2], "High"] = 112.0
    if scenario == "gap":
        df.loc[df.index[2], ["Open", "Low", "Close"]] = [90.0, 88.0, 91.0]
    if scenario == "collision":
        df.loc[df.index[1], ["High", "Low"]] = [112.0, 94.0]
    if scenario in {"eof", "repeated"}:
        monkeypatch.setattr(ScriptedSignals, "should_exit", lambda *args: False)
    if scenario == "repeated":
        monkeypatch.setattr(ScriptedSignals, "should_enter", lambda *args: True)
    config = BacktestConfig(
        initial_capital=10000.0,
        position_size_pct=100.0,
        commission_pct=0.001,
        slippage_pct=0.001,
        maximum_holding_days=1 if scenario == "holding" else None,
    )
    backtest = BacktestEngine(config).run(df, ScriptedSignals())
    # Independent ledger constants, not values produced by either adapter.
    # 99 shares cost 9,909.9 + 9.9099 commission, leaving 80.1901 cash.
    expected = {
        "signal": (99.9, 9960.4, "SIGNAL", 4),
        "stop": (94.905, 9466.389505, "STOP_LOSS", 3),
        "target": (109.89, 10948.42099, "TAKE_PROFIT", 3),
        "gap": (89.91, 8972.37901, "STOP_LOSS", 3),
        "collision": (94.905, 9466.389505, "STOP_LOSS", 2),
        "holding": (99.9, 9960.4, "MAX_HOLDING_PERIOD", 3),
        "eof": (100.899, 10059.202099, "END_OF_DATA", 5),
        "repeated": (100.899, 10059.202099, "END_OF_DATA", 5),
    }
    exit_price, final_equity, reason, exit_day = expected[scenario]
    assert len(backtest.trades) == 1
    trade = backtest.trades[0]
    assert trade.quantity == 99
    assert trade.entry_price == pytest.approx(100.1, abs=1e-8, rel=0)
    assert trade.entry_date.isoformat() == "2020-01-02"
    assert trade.exit_date.isoformat() == f"2020-01-{exit_day:02d}"
    assert trade.exit_price == pytest.approx(exit_price, abs=1e-8, rel=0)
    assert trade.exit_reason.value == reason
    assert backtest.final_equity == pytest.approx(final_equity, abs=1e-8, rel=0)
    paper = PaperTradingRunner(make_simple_strategy(), config=config)
    for end in range(1, len(df) + 1):
        paper.process_bar("AAA", df.iloc[:end])
    paper.finalize()
    assert len(paper.portfolio.fills) == 2 * len(backtest.trades)
    assert paper.portfolio.cash == pytest.approx(backtest.final_equity, abs=1e-8, rel=0)
    for i, trade in enumerate(backtest.trades):
        buy, sell = paper.portfolio.fills[2 * i : 2 * i + 2]
        assert buy.fill_price == trade.entry_price
        assert sell.fill_price == trade.exit_price
        assert sell.exit_reason == trade.exit_reason.value
        assert buy.quantity == sell.quantity == trade.quantity
        assert paper.portfolio.closed_positions[i].realized_pnl == pytest.approx(trade.net_pnl, abs=1e-8, rel=0)
    if scenario == "collision":
        assert backtest.trades[0].exit_reason.value == "STOP_LOSS"


def test_restart_reconciles_and_detects_tampering(tmp_path):
    import json

    df = make_ohlcv(n=100, seed=10)
    runner = PaperTradingRunner(make_simple_strategy())
    for end in range(50, 80):
        runner.process_bar("AAA", df.iloc[:end])
    path = tmp_path / "checkpoint.json"
    runner.save_checkpoint(path)
    restored = PaperTradingRunner.restore_checkpoint(path)
    for end in range(80, 101):
        runner.process_bar("AAA", df.iloc[:end])
        restored.process_bar("AAA", df.iloc[:end])
    runner.finalize()
    restored.finalize()
    assert runner.portfolio.cash == restored.portfolio.cash
    assert runner.portfolio.fills == restored.portfolio.fills
    payload = json.loads(path.read_text())
    payload["cash"] += 1.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="reconciliation"):
        PaperTradingRunner.restore_checkpoint(path)


def test_launch_risk_and_state_machine():
    with pytest.raises(ValueError, match="Incomplete"):
        require_complete_policy(RiskPolicy())
    for name in ("research_standard", "research_conservative", "paper_beta"):
        assert require_complete_policy(risk_profile(name))
    lifecycle = EvaluationLifecycle()
    with pytest.raises(ValueError):
        lifecycle.begin_holdout(test_passed=True)
    lifecycle.freeze("abc")
    with pytest.raises(ValueError, match="Refinement"):
        lifecycle.require_adaptive()
    lifecycle.begin_test("abc")
    with pytest.raises(ValueError):
        lifecycle.begin_holdout(test_passed=False)
    lifecycle.begin_holdout(test_passed=True)
    with pytest.raises(ValueError):
        lifecycle.begin_holdout(test_passed=True)
    lifecycle.complete()
    assert not evidence_status(
        technical=True,
        validation_passed=True,
        test_passed=True,
        holdout_passed=True,
        risk_passed=False,
    )["DEPLOYMENT_ELIGIBLE"]


def test_benchmark_hand_calculation():
    from evaluation.baselines import BaselineKind, run_baseline

    prices = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 110.0],
            "High": [101.0, 101.0, 121.0],
            "Low": [99.0, 99.0, 109.0],
            "Close": [100.0, 100.0, 120.0],
            "Volume": [1000.0] * 3,
        },
        index=pd.date_range("2020-01-01", periods=3),
    )
    config = BacktestConfig(
        initial_capital=1000.0, position_size_pct=20.0, commission_pct=0.0, slippage_pct=0.0
    )
    full = run_baseline(BaselineKind.BUY_AND_HOLD, prices, config)
    matched = run_baseline(BaselineKind.EXPOSURE_MATCHED_BUY_AND_HOLD, prices, config)
    assert full.result.final_equity == 1200.0  # 10 shares, 20 gain per share
    assert matched.result.final_equity == 1040.0  # 2 shares, 800 idle cash
    assert full.metadata["allocation_pct"] == 100.0
    assert matched.metadata["allocation_pct"] == 20.0


def test_exposure_survives_new_strategy_and_campaign(tmp_path):
    from evaluation.campaigns import HoldoutExposureLedger
    from scripts.run_research_e2e import build_research_dataset_split

    partition = build_research_dataset_split(make_ohlcv(300), symbol="AAA").partition_fingerprints[
        -1
    ]
    path = tmp_path / "exposures.sqlite"
    first = HoldoutExposureLedger(path).claim(
        dataset_family="AAA", campaign_id="one", strategy_hash="old", fingerprint=partition
    )
    repeated = HoldoutExposureLedger(path).claim(
        dataset_family="AAA", campaign_id="two", strategy_hash="new", fingerprint=partition
    )
    assert first == "UNSEEN_CONFIRMATORY"
    assert repeated == "PREVIOUSLY_EXPOSED_EXPLORATORY"


def test_rejected_exit_does_not_mutate_ledger():
    p = PaperPortfolio(1000.0)
    fill = PaperFill(
        fill_id="entry",
        symbol="AAA",
        fill_time=datetime(2020, 1, 1),
        signal_type=SignalType.ENTER_LONG,
        fill_price=100.0,
        quantity=5.0,
        commission=1.0,
        slippage=0.0,
    )
    p.process_fill(fill, strategy_id="x")
    oversell = fill.model_copy(
        update={"fill_id": "exit", "signal_type": SignalType.EXIT_LONG, "quantity": 6.0}
    )
    with pytest.raises(ValueError, match="quantity"):
        p.process_fill(oversell, strategy_id="x")
    assert p.cash == 499.0
    assert len(p.fills) == 1
    assert p.get_open_position("AAA").quantity == 5.0


def test_successful_confirmatory_path_does_not_certify_release(tmp_path):
    import json

    import numpy as np

    from scripts.run_research_e2e import run_e2e_research
    from tests.test_research_e2e import MockLifecycleClient

    count = 1000
    base = 100.0 + np.arange(count) * 0.1
    frame = pd.DataFrame(
        {
            "Open": base,
            "High": base + 0.2,
            "Low": base - 0.1,
            "Close": base + 0.1,
            "Volume": [10000.0] * count,
        },
        index=pd.date_range("2020-01-01", periods=count, freq="B"),
    )
    strategy = make_simple_strategy(
        entry_indicator="close",
        entry_operator=">",
        entry_value=0.0,
        exit_indicator="close",
        exit_operator="<",
        exit_value=0.0,
        max_holding_days=1,
        position_size_pct=10.0,
    )
    ledger = tmp_path / "campaign.sqlite"
    outputs = []
    for number in range(2):
        output = tmp_path / str(number)
        result = run_e2e_research(
            symbol="AAA",
            df=frame,
            commission=0.0,
            slippage=0.0,
            structured_client=MockLifecycleClient(raw_strategy=strategy.model_dump(mode="json")),
            mode="SINGLE_AGENT",
            output_dir=output,
            exposure_ledger_path=ledger,
            campaign_id=f"campaign-{number}",
        )
        outputs.append(result)
    assert outputs[0]["status"] == "COMPLETED"
    assert outputs[0]["test_metrics"]["number_of_trades"] >= 20
    assert outputs[0]["final_holdout_metrics"]["number_of_trades"] >= 20
    assert outputs[1]["final_holdout_metrics"] == {}
    certification = json.loads((tmp_path / "1" / "certification.json").read_text())
    assert certification["holdout_exposure"] == "PREVIOUSLY_EXPOSED_EXPLORATORY"
    assert not outputs[0]["certification"]["DEPLOYMENT_ELIGIBLE"]
    assert not outputs[0]["certification"]["STATISTICALLY_VIABLE"]
    metadata = json.loads((tmp_path / "0" / "test_metrics.json").read_text())["_split"]
    assert metadata["name"] == "test"
    assert metadata["strategy_hash"]


@pytest.mark.parametrize(
    "field,at_limit",
    [
        ("max_drawdown_pct", 15.0),
        ("max_position_pct", 25.0),
        ("max_portfolio_exposure_pct", 25.0),
        ("max_loss_per_trade_pct", 10.0),
        # (250 + 150) / mean(1000, 850, 900) * 100, rounded per metric contract.
        ("max_turnover_pct", 43.6364),
    ],
)
def test_exact_risk_boundaries(field, at_limit):
    from risk.evaluator import evaluate_risk
    from tests.test_sprint6_risk import _result

    result = _result()
    assert evaluate_risk(result, RiskPolicy(**{field: at_limit})).passed
    failure = evaluate_risk(result, RiskPolicy(**{field: at_limit - 0.00001}))
    assert not failure.passed
    assert failure.violations[0].rule.value == field.upper()
    assert failure.policy_id == "risk-policy-v1"
    assert len(failure.policy_sha256) == 64


def test_adaptive_view_has_no_confirmatory_access():
    from scripts.run_research_e2e import build_research_dataset_split

    split = build_research_dataset_split(make_ohlcv(300), symbol="AAA")
    assert not hasattr(split.adaptive_view(), "test")
    assert not hasattr(split.adaptive_view(), "final_holdout")
    with pytest.raises(ValueError, match="frozen"):
        split.release_confirmatory_test(None)


@pytest.mark.parametrize("price", [float("inf"), float("nan")])
def test_nonfinite_fill_is_rejected(price):
    with pytest.raises(ValueError):
        PaperFill(
            fill_id="bad",
            symbol="AAA",
            fill_time=datetime(2020, 1, 1),
            signal_type=SignalType.ENTER_LONG,
            fill_price=price,
            quantity=1.0,
            commission=0.0,
            slippage=0.0,
        )
