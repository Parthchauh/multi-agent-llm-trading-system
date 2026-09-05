from __future__ import annotations
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from data.research_split import FrozenStrategySelection, ResearchDatasetSplit, ResearchPartition, ResearchPartitionRole
from strategies.schema import Condition, ConditionGroup, EntryRules, ExitRules, PositionSizing, StopLoss, StrategyMetadata, StrategySchema, TakeProfit

def make_simple_strategy(*, entry_indicator="rsi_14", entry_operator=">", entry_value=55.0, exit_indicator="close", exit_operator="<", exit_value="sma_50", stop_loss_pct=5.0, take_profit_pct=10.0, position_size_pct=20.0, max_holding_days=None):
    return StrategySchema(metadata=StrategyMetadata(name="Test RSI Strategy", description="Simple RSI strategy for tests", market_type="equity", timeframe="1d"), entry=EntryRules(long_conditions=ConditionGroup(logic="ALL", conditions=[Condition(indicator=entry_indicator, operator=entry_operator, value=entry_value)])), exit=ExitRules(exit_conditions=ConditionGroup(logic="ANY", conditions=[Condition(indicator=exit_indicator, operator=exit_operator, value=exit_value)]), maximum_holding_days=max_holding_days), stop_loss=StopLoss(type="percentage", value=stop_loss_pct), take_profit=TakeProfit(type="percentage", value=take_profit_pct), position_sizing=PositionSizing(type="fixed_percentage", value=position_size_pct))

def make_crossover_strategy():
    return StrategySchema(metadata=StrategyMetadata(name="EMA Crossover Test", market_type="equity", timeframe="1d"), entry=EntryRules(long_conditions=ConditionGroup(logic="ALL", conditions=[Condition(indicator="ema_20", operator="crosses_above", value="ema_50")])), exit=ExitRules(exit_conditions=ConditionGroup(logic="ANY", conditions=[Condition(indicator="ema_20", operator="crosses_below", value="ema_50")])), stop_loss=StopLoss(type="percentage", value=5.0), take_profit=TakeProfit(type="percentage", value=15.0), position_sizing=PositionSizing(type="fixed_percentage", value=20.0))

def make_ohlcv(n=300, seed=42, start=100.0):
    rng = np.random.default_rng(seed)
    closes = start + np.cumsum(rng.normal(0.05, 0.8, n))
    opens = closes * (1 + rng.normal(0, 0.003, n))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0.001, 0.008, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0.001, 0.008, n))
    volumes = rng.integers(100_000, 600_000, n).astype(float)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes}, index=dates)

def make_dataset_split(n=400, seed=42, *, context_frac=0.60, validation_frac=0.20, test_frac=0.10):
    full = make_ohlcv(n=n, seed=seed)
    c_end = int(n * context_frac)
    v_end = c_end + int(n * validation_frac)
    t_end = v_end + int(n * test_frac)
    def _p(df, role, pid): return ResearchPartition(partition_id=pid, role=role, observations=df)
    return ResearchDatasetSplit(context=_p(full.iloc[:c_end], ResearchPartitionRole.CONTEXT, "ctx-test"), validation=_p(full.iloc[c_end:v_end], ResearchPartitionRole.VALIDATION, "val-test"), test=_p(full.iloc[v_end:t_end], ResearchPartitionRole.TEST, "test-test"), final_holdout=_p(full.iloc[t_end:], ResearchPartitionRole.FINAL_HOLDOUT, "hold-test"))

def make_frozen_selection(strategy_id="strat-001"):
    return FrozenStrategySelection(strategy_id=strategy_id, strategy_hash=hashlib.sha256(strategy_id.encode()).hexdigest(), frozen_at=datetime.now(timezone.utc))
