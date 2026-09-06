"""paper_trading/journal.py
=========================
Append-only journal for paper-trading events, signals, fills, and equity curves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from paper_trading.models import PaperFill, PaperPortfolioState, PaperTradeSignal


@dataclass
class PaperJournalEntry:
    timestamp: datetime
    event_type: str
    data: dict[str, Any]


class PaperJournal:
    """Records chronological, immutable audit logs of paper trading sessions."""

    def __init__(self, output_path: Path | str | None = None) -> None:
        self.output_path = Path(output_path) if output_path else None
        self.entries: list[PaperJournalEntry] = []

    def record_signal(self, signal: PaperTradeSignal) -> None:
        self._append("SIGNAL", signal.model_dump(mode="json"))

    def record_fill(self, fill: PaperFill) -> None:
        self._append("FILL", fill.model_dump(mode="json"))

    def record_snapshot(self, state: PaperPortfolioState) -> None:
        self._append("EQUITY_SNAPSHOT", state.model_dump(mode="json"))

    def record_event(self, event_type: str, details: dict[str, Any]) -> None:
        self._append(event_type, details)

    def _append(self, event_type: str, data: dict[str, Any]) -> None:
        ts = datetime.fromisoformat(data.get("timestamp") or data.get("signal_time") or data.get("fill_time") or datetime.now().isoformat())
        entry = PaperJournalEntry(timestamp=ts, event_type=event_type, data=data)
        self.entries.append(entry)
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp": ts.isoformat(), "event": event_type, "data": data}) + "\n")

    def dump_entries(self) -> list[dict[str, Any]]:
        return [{"timestamp": e.timestamp.isoformat(), "event": e.event_type, "data": e.data} for e in self.entries]
