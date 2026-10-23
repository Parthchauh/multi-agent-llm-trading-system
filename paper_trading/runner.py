"""Paper replay adapter using the same bar kernel and whole-share ledger as backtests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from backtesting.execution import TradeExecutor
from backtesting.kernel import PendingOrder, UnifiedExecutionKernel
from backtesting.models import BacktestConfig, ExitReason
from backtesting.portfolio import Portfolio
from data.schema import validate_ohlcv
from paper_trading.journal import PaperJournal
from paper_trading.models import PaperFill, SignalType
from paper_trading.portfolio import PaperPortfolio
from strategies.compiler import StrategyCompiler
from strategies.schema import StrategySchema
from strategies.validator import StrategyValidator


class _RecordingPortfolio(Portfolio):
    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback

    def open_position(self, **kwargs):
        pos = super().open_position(**kwargs)
        if pos is not None:
            self.callback(
                True,
                pos.entry_price,
                pos.quantity,
                pos.commission_paid_entry,
                kwargs["raw_price"],
                None,
            )
        return pos

    def close_position(self, **kwargs):
        trade = super().close_position(**kwargs)
        if trade is not None:
            reference = trade.exit_price / (1.0 - self.executor.slippage_pct)
            self.callback(
                False,
                trade.exit_price,
                trade.quantity,
                self.executor.calculate_commission(trade.exit_price * trade.quantity),
                reference,
                trade.exit_reason.value,
            )
        return trade


class PaperTradingRunner:
    """One frozen strategy and symbol per runner; all bar fills are in bar_fills."""

    def __init__(
        self,
        strategy: StrategySchema,
        *,
        config: BacktestConfig | None = None,
        journal_path: Path | str | None = None,
    ) -> None:
        if not isinstance(strategy, StrategySchema):
            raise TypeError("strategy must be a canonical StrategySchema instance.")
        validation = StrategyValidator().validate(strategy)
        if not validation.is_valid:
            raise ValueError(
                "PaperTradingRunner requires a semantically valid StrategySchema: "
                + "; ".join(validation.errors)
            )
        self.strategy = strategy.model_copy(deep=True)
        self.strategy_id = hashlib.sha256(strategy.model_dump_json().encode()).hexdigest()
        self.config = config or BacktestConfig(initial_capital=100_000.0)
        self.portfolio = PaperPortfolio(initial_capital=self.config.initial_capital)
        self.journal = PaperJournal(output_path=journal_path)
        self.executor = TradeExecutor(self.config.commission_pct, self.config.slippage_pct)
        self._ledger = _RecordingPortfolio(
            self._record_fill,
            initial_capital=self.config.initial_capital,
            position_size_pct=self.config.position_size_pct,
            executor=self.executor,
        )
        self._previous_history = None
        self._pending_signal = None
        self._symbol = None
        self._current_time = None
        self.bar_fills = []
        self._finished = False
        self._replay_start = None
        self._controls = []

    def _record_fill(self, entry, price, quantity, commission, reference, reason):
        fill = PaperFill(
            fill_id=f"{self.strategy_id[:16]}:{len(self.portfolio.fills)}",
            symbol=self._symbol,
            fill_time=self._current_time,
            signal_type=SignalType.ENTER_LONG if entry else SignalType.EXIT_LONG,
            fill_price=price,
            quantity=float(quantity),
            commission=commission,
            slippage=abs(price - reference) * quantity,
            reference_price=reference,
            exit_reason=reason,
        )
        self.portfolio.process_fill(fill, strategy_id=self.strategy_id)
        self.journal.record_fill(fill)
        self.bar_fills.append(fill)

    def process_bar(self, symbol: str, historical_bar_series: pd.DataFrame, *,
                    entries_enabled: bool = True, force_exit: bool = False):
        if not isinstance(entries_enabled, bool) or not isinstance(force_exit, bool):
            raise TypeError("Replay controls must be boolean.")
        if hashlib.sha256(self.strategy.model_dump_json().encode()).hexdigest() != self.strategy_id:
            raise ValueError("Frozen paper strategy was modified; create a new certified version.")
        if self._finished:
            raise ValueError("Replay was finalized.")
        clean = validate_ohlcv(historical_bar_series)
        if self._replay_start is None:
            self._replay_start = len(clean)
        symbol = symbol.upper()
        if self._symbol is not None and self._symbol != symbol:
            raise ValueError("One paper runner supports exactly one symbol.")
        if self._previous_history is not None and not clean.iloc[:-1].equals(
            self._previous_history
        ):
            raise ValueError(
                "Incoming paper-trading history must append exactly one unrevised bar."
            )
        self._symbol = symbol
        self._current_time = clean.index[-1].to_pydatetime()
        self.bar_fills = []
        compiled = StrategyCompiler().compile(self.strategy, clean)
        frame = clean.rename(columns=str.lower)
        i = len(frame) - 1
        if not entries_enabled and self._pending_signal and self._pending_signal.is_entry:
            self._pending_signal = None
        if force_exit and self._ledger.is_invested():
            self._pending_signal = PendingOrder(False, self._current_time.date())
        UnifiedExecutionKernel.execute(
            config=self.config,
            executor=self.executor,
            portfolio=self._ledger,
            pending_order=self._pending_signal,
            signal_provider=compiled,
            df=frame,
            i=i,
        )
        self._pending_signal = None
        if entries_enabled and not self._ledger.is_invested() and compiled.should_enter(i, frame):
            self._pending_signal = PendingOrder(True, self._current_time.date())
        elif self._ledger.is_invested() and compiled.should_exit(i, frame):
            self._pending_signal = PendingOrder(False, self._current_time.date())
        self._previous_history = clean.copy(deep=True)
        self._controls.append({"entries_enabled": entries_enabled, "force_exit": force_exit})
        state = self.portfolio.snapshot(
            self._current_time, {symbol: float(clean["Close"].iloc[-1])}
        )
        self.journal.record_snapshot(state)
        return (self.bar_fills[-1] if self.bar_fills else None), state

    def finalize(self):
        """Explicit replay-only EOF close, matching BacktestEngine."""
        if self._previous_history is None:
            raise ValueError("Cannot finalize an empty replay.")
        if self._finished:
            return
        close = float(self._previous_history["Close"].iloc[-1])
        self._ledger.close_position(
            exit_date=self._current_time.date(), raw_price=close, exit_reason=ExitReason.END_OF_DATA
        )
        self._pending_signal = None
        self._finished = True

    def save_checkpoint(self, path: Path | str):
        """Atomically persist replay inputs and a ledger checksum; contains no secrets."""
        payload = self.checkpoint_payload()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        os.replace(temporary, target)

    def checkpoint_payload(self):
        """Return replayable state for transactional storage by an operational wrapper."""
        if self._previous_history is None:
            raise ValueError("Cannot checkpoint an empty runner.")
        payload = {
            "version": 1,
            "strategy": self.strategy.model_dump(mode="json"),
            "strategy_sha256": self.strategy_id,
            "config": self.config.model_dump(mode="json"),
            "symbol": self._symbol,
            "replay_start": self._replay_start,
            "history": {
                "index": [stamp.isoformat() for stamp in self._previous_history.index],
                "columns": list(self._previous_history.columns),
                "data": self._previous_history.to_numpy().tolist(),
                "dtypes": [str(dtype) for dtype in self._previous_history.dtypes],
            },
            "cash": self.portfolio.cash,
            "finished": self._finished,
            "fills": [fill.model_dump(mode="json") for fill in self.portfolio.fills],
            "controls": self._controls,
        }
        return payload

    @classmethod
    def restore_checkpoint(cls, path: Path | str, *, journal_path=None):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_checkpoint_payload(payload, journal_path=journal_path)

    @classmethod
    def from_checkpoint_payload(cls, payload, *, journal_path=None):
        if payload["version"] != 1:
            raise ValueError("Unsupported checkpoint schema.")
        runner = cls(
            StrategySchema.model_validate(payload["strategy"]),
            config=BacktestConfig(**payload["config"]),
        )
        if runner.strategy_id != payload["strategy_sha256"]:
            raise ValueError("Frozen strategy checksum mismatch.")
        saved = payload["history"]
        history = pd.DataFrame(
            saved["data"], columns=saved["columns"], index=pd.DatetimeIndex(saved["index"])
        )
        history = history.astype(dict(zip(saved["columns"], saved["dtypes"])))
        count = len(history) - payload["replay_start"] + 1
        controls = payload.get("controls", [{"entries_enabled": True, "force_exit": False}]*count)
        if len(controls) != count:
            raise ValueError("Checkpoint control history is incomplete.")
        for offset, end in enumerate(range(payload["replay_start"], len(history) + 1)):
            runner.process_bar(payload["symbol"], history.iloc[:end], **controls[offset])
        if payload["finished"]:
            runner.finalize()
        if abs(runner.portfolio.cash - payload["cash"]) > 1e-8 or (
            [f.model_dump(mode="json") for f in runner.portfolio.fills] != payload["fills"]
        ):
            raise ValueError("Checkpoint failed deterministic ledger reconciliation.")
        if journal_path is not None:
            runner.journal = PaperJournal(output_path=journal_path)
        return runner
