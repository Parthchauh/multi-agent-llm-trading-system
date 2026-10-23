"""Bounded fold-specific research, separate from final confirmatory evaluation.

The candidate factory receives a copy of training bars only. It is trusted Python
application code, not a sandbox for arbitrary submitted code. Model adapters must
send only that frame's derived features. Never pass the final holdout here.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backtesting.engine import BacktestEngine
from backtesting.models import BacktestConfig
from data.schema import validate_ohlcv
from research_validation.uncertainty import (
    block_bootstrap,
    sharpe_diagnostics,
    trade_path_distribution,
)
from risk.evaluator import evaluate_risk
from risk.models import RiskPolicy
from risk.profiles import require_complete_policy
from strategies.compiler import StrategyCompiler
from strategies.schema import StrategySchema
from strategies.validator import StrategyValidator


class TrainOnlyAgentFactory:
    """Use the existing structured analyst/generator on fold training features.

    One generation attempt per call: hidden retries cannot expand search budget.
    Transport retry may repeat the same request, never repair strategy semantics.
    """

    def __init__(self, client, *, symbol):
        from regime.features import RegimeFeatureEngine
        from tradingagents.agents.regime_analyst import RegimeAnalystAgent
        from tradingagents.agents.strategy_generator import StrategyGeneratorAgent

        self.features = RegimeFeatureEngine()
        self.symbol = symbol
        self.analyst = RegimeAnalystAgent(client)
        self.generator = StrategyGeneratorAgent(client, max_attempts=1)

    def __call__(self, train, fold, attempt):
        features = self.features.snapshot(train, symbol=self.symbol)
        assessment = self.analyst.assess(features).assessment
        return self.generator.request_candidate(assessment, features, attempt=attempt).strategy


class CampaignConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    train_size: int = Field(ge=2)
    validation_size: int = Field(ge=2)
    test_size: int = Field(ge=2)
    step_size: int = Field(ge=2)
    expanding: bool = False
    attempts_per_fold: int = Field(default=3, ge=1, le=100)
    maximum_attempts: int = Field(default=30, ge=1, le=1000)
    minimum_folds: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def nonoverlap(self):
        if self.step_size < self.test_size:
            raise ValueError("Test windows must not overlap: step_size >= test_size.")
        return self


def _hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _window(frame: pd.DataFrame) -> dict:
    return {"start": frame.index[0].isoformat(), "end": frame.index[-1].isoformat(),
            "bars": len(frame), "sha256": hashlib.sha256(
                pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()}


def _distribution(values: list[float]) -> dict:
    return {"count": len(values), "values": values,
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "worst": min(values) if values else None,
            "best": max(values) if values else None,
            "variance": statistics.pvariance(values) if values else None}


class FoldResearchCampaign:
    """Train-only generation, deterministic validation selection, frozen test.

    All attempted candidates (including duplicates and rejected schemas) consume
    the declared search budget. Invalid output is recorded by error type only;
    provider exception text and malformed model text are never persisted.
    """

    def run(self, *, development_data: pd.DataFrame, config: CampaignConfig,
            backtest_config: BacktestConfig, risk_policy: RiskPolicy,
            candidate_factory: Callable[[pd.DataFrame, int, int], object]) -> dict:
        frame = validate_ohlcv(development_data)
        policy = require_complete_policy(risk_policy)
        windows = []
        train_end = config.train_size
        while train_end + config.validation_size + config.test_size <= len(frame):
            train_start = 0 if config.expanding else train_end-config.train_size
            windows.append((train_start, train_end, train_end+config.validation_size,
                            train_end+config.validation_size+config.test_size))
            train_end += config.step_size
        if len(windows) < config.minimum_folds:
            raise ValueError("Not enough development history for the declared minimum folds.")
        if len(windows)*config.attempts_per_fold > config.maximum_attempts:
            raise ValueError("Planned candidate attempts exceed the global search budget.")
        folds, test_returns, validation_returns = [], [], []
        for number, (start, end, val_end, test_end) in enumerate(windows, 1):
            train = frame.iloc[start:end].copy(deep=True)
            validation = frame.iloc[end:val_end].copy(deep=True)
            test = frame.iloc[val_end:test_end].copy(deep=True)
            attempts, candidates, seen = [], [], set()
            for attempt in range(1, config.attempts_per_fold+1):
                record = {"attempt": attempt, "parent": None, "status": "REJECTED"}
                phase = 'CANDIDATE_FACTORY'
                try:
                    raw = candidate_factory(train.copy(deep=True), number, attempt)
                    phase = 'SCHEMA_VALIDATION'
                    strategy = StrategySchema.model_validate(raw)
                    checked = StrategyValidator().validate(strategy)
                    if not checked.is_valid:
                        raise ValueError("Strategy semantic validation rejected candidate.")
                    identity = _hash(strategy.model_dump(mode="json"))
                    record["strategy_sha256"] = identity
                    if identity in seen:
                        record["status"] = "DUPLICATE"
                    else:
                        seen.add(identity)
                        phase = 'DETERMINISTIC_EVALUATION'
                        effective = self._effective(strategy, backtest_config, policy)
                        result = self._evaluate(strategy, train, validation, effective)
                        score = result.metrics.total_return_pct
                        if score is None or not math.isfinite(score):
                            raise ValueError("Validation score unavailable.")
                        risk = evaluate_risk(result, policy).model_dump(mode="json")
                        record.update(status="EVALUATED", validation_return_pct=score,
                                      validation_metrics=result.metrics.to_dict(), risk=risk)
                        candidates.append((score, identity, strategy, effective))
                        validation_returns.append(score)
                except Exception as exc:
                    # A provider/runtime failure is distinct from a valid research rejection.
                    record["status"] = ("REJECTED" if phase == 'SCHEMA_VALIDATION' and isinstance(exc, (ValueError, TypeError))
                                        else "RUNTIME_FAILURE")
                    record["error_type"] = type(exc).__name__
                    record['failure_phase'] = phase
                attempts.append(record)
            fold = {"fold": number, "train": _window(train), "validation": _window(validation),
                    "test": _window(test), "attempts": attempts, "selected": None,
                    "status": "NO_VALID_CANDIDATE"}
            if candidates:
                # A predeclared ranking; no risk or outcome threshold is tuned on test.
                score, identity, selected, effective = sorted(
                    candidates, key=lambda item: (-item[0], item[1]))[0]
                frozen = {"strategy": selected.model_dump(mode="json"),
                          "strategy_sha256": identity, "config": effective.model_dump(mode="json"),
                          "risk_policy": policy.model_dump(mode="json"),
                          "selection_metric": "validation_total_return_pct", "selection_score": score}
                frozen["freeze_sha256"] = _hash(frozen)
                fold["selected"] = frozen
                # Freeze occurs before evaluation; include only past history for warm-up.
                tested = self._evaluate(selected, frame.iloc[start:val_end], test, effective)
                test_returns.append(tested.metrics.total_return_pct)
                fold.update(status="TESTED", test_metrics=tested.metrics.to_dict(),
                            test_risk=evaluate_risk(tested, policy).model_dump(mode="json"))
                equity = pd.Series([tested.initial_capital] + [p.equity for p in tested.equity_curve])
                if (equity <= 0).any():
                    fold["uncertainty"] = {"status": "INSUFFICIENT_EVIDENCE", "reason": "NONPOSITIVE_EQUITY"}
                else:
                    returns = equity.pct_change().dropna().to_numpy()
                    fold["uncertainty"] = {
                        "psr_dsr": sharpe_diagnostics(returns),
                        "block_bootstrap": block_bootstrap(returns),
                        "trade_permutation": trade_path_distribution(
                            [t.net_pnl for t in tested.trades], tested.initial_capital),
                        "trade_bootstrap": trade_path_distribution(
                            [t.net_pnl for t in tested.trades], tested.initial_capital, bootstrap=True)}
                fold["sensitivity"] = self._sensitivity(
                    selected, frame.iloc[start:val_end], test, effective)
            folds.append(fold)
        return {"schema_version": 1, "status": "DIAGNOSTIC_ONLY", "config": config.model_dump(),
                "selection_rule": "highest validation total return; SHA256 ascending on tie",
                "folds": folds, "attempts_used": len(windows)*config.attempts_per_fold,
                "planned_folds": len(windows), "tested_folds": len(test_returns),
                "validation_distribution": _distribution(validation_returns),
                "test_distribution": _distribution(test_returns),
                "limitations": ["Development walk-forward windows may become training history in later folds.",
                                "Final confirmatory test and holdout are excluded by caller contract.",
                                "Candidate attempts are not an estimate of independent statistical trials."],
                "statistical_certification": False, "paper_eligible": False,
                "real_money_eligible": False}

    @staticmethod
    def _effective(strategy, base, policy):
        return BacktestConfig.model_validate({
            **base.model_dump(), "position_size_pct": min(strategy.position_sizing.value,
                base.position_size_pct, policy.max_position_pct, policy.max_portfolio_exposure_pct),
            "maximum_holding_days": strategy.exit.maximum_holding_days})

    @staticmethod
    def _evaluate(strategy, past, scoring, config):
        history = pd.concat([past.copy(deep=True), scoring.copy(deep=True)])
        compiled = StrategyCompiler().compile(strategy, history)
        return BacktestEngine(config).run(history, compiled, evaluation_start=scoring.index[0])

    def _sensitivity(self, strategy, past, scoring, config):
        costs = []
        for multiplier in (1.0, 1.5, 2.0):
            changed = BacktestConfig.model_validate({**config.model_dump(),
                "commission_pct": config.commission_pct*multiplier,
                "slippage_pct": config.slippage_pct*multiplier})
            result = self._evaluate(strategy, past, scoring, changed)
            costs.append({"cost_multiplier": multiplier, "metrics": result.metrics.to_dict()})
        starts = []
        for offset in (0, 5, 10):
            if len(scoring)-offset < 2:
                continue
            warmup = pd.concat([past, scoring.iloc[:offset]])
            result = self._evaluate(strategy, warmup, scoring.iloc[offset:], config)
            starts.append({"offset_bars": offset, "metrics": result.metrics.to_dict()})
        return {"status": "DIAGNOSTIC_ONLY", "cost_scenarios": costs, "start_date_scenarios": starts,
                "promotion_eligible": False}
