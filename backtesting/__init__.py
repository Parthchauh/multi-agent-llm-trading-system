"""
backtesting/__init__.py
=======================
Public API for the deterministic backtesting engine package.

Sprint 2 exports the engine, execution engine, models, and metrics calculator.

Example usage::

    from backtesting import (
        BacktestConfig,
        BacktestEngine,
        BacktestResult,
        SignalType,
        ExitReason,
        StrategySignalProvider,
    )
"""

from backtesting.models import (
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    EquityPoint,
    ExitReason,
    Position,
    Signal,
    SignalType,
    Trade,
    TradeSide,
)
from backtesting.execution import TradeExecutor
from backtesting.portfolio import Portfolio
from backtesting.metrics import calculate_metrics
from backtesting.engine import BacktestEngine, StrategySignalProvider

__all__ = [
    # Engine & Protocol
    "BacktestEngine",
    "StrategySignalProvider",
    # Models
    "BacktestConfig",
    "BacktestMetrics",
    "BacktestResult",
    "EquityPoint",
    "ExitReason",
    "Position",
    "Signal",
    "SignalType",
    "Trade",
    "TradeSide",
    # Components
    "TradeExecutor",
    "Portfolio",
    "calculate_metrics",
]
