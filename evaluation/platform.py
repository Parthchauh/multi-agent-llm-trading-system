"""Sprint 8 application services: experiment runs, SQLite audit storage, reports.

This module deliberately knows nothing about legacy Buy/Sell/Hold graphs.  An
architecture executor may return only already-authoritative structured output;
the runner records both successful and failed attempts as research evidence.
"""

from __future__ import annotations

import platform
import sqlite3
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evaluation.experiments import ExperimentArchitecture, ExperimentConfig

_SENSITIVE_KEY_PARTS = ("api_key", "apikey", "authorization", "token", "secret", "password")


class ResearchPlatformError(RuntimeError):
    """Raised for safe application-service failures."""


class FailureType(str, Enum):
    LLM_FORMAT_ERROR = "LLM_FORMAT_ERROR"
    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    SEMANTIC_VALIDATION_ERROR = "SEMANTIC_VALIDATION_ERROR"
    COMPILATION_ERROR = "COMPILATION_ERROR"
    DATA_ERROR = "DATA_ERROR"
    BACKTEST_ERROR = "BACKTEST_ERROR"
    RISK_REJECTION = "RISK_REJECTION"
    VIABILITY_REJECTION = "VIABILITY_REJECTION"
    ROBUSTNESS_REJECTION = "ROBUSTNESS_REJECTION"
    OOS_FAILURE = "OOS_FAILURE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    UNKNOWN = "UNKNOWN"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReproducibilityRecord(_FrozenModel):
    python_version: str
    platform: str
    git_commit: str | None = None
    model_provider: str
    model_name: str
    seed: int | None = None
    prompt_versions: tuple[tuple[str, str], ...] = ()


class ResearchEvent(_FrozenModel):
    timestamp: datetime
    run_id: str = Field(min_length=1, max_length=128)
    node: str = Field(min_length=1, max_length=100)
    status: str = Field(min_length=1, max_length=40)
    event_type: str = Field(min_length=1, max_length=80)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def metadata_must_not_contain_secret_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        if _contains_sensitive_key(value):
            raise ValueError("Research event metadata must not contain secrets.")
        return value


class ResearchRun(_FrozenModel):
    run_id: str = Field(min_length=1, max_length=128)
    experiment_id: str = Field(min_length=1, max_length=160)
    architecture: ExperimentArchitecture
    symbol: str = Field(min_length=1, max_length=32)
    repetition: int = Field(ge=1)
    status: str = Field(min_length=1, max_length=64)
    started_at: datetime
    completed_at: datetime | None = None
    strategy_id: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    failure_type: str | None = None
    failure_code: str | None = None
    reproducibility: ReproducibilityRecord


class ExperimentReport(_FrozenModel):
    experiment_config: ExperimentConfig
    reproducibility: ReproducibilityRecord
    runs: tuple[ResearchRun, ...]
    events: tuple[ResearchEvent, ...]
    failures: tuple[ResearchRun, ...]
    limitations: tuple[str, ...] = ()


class SQLiteResearchRepository:
    """Persistence boundary; SQL is confined to this repository service."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY, config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL,
                    status TEXT NOT NULL, payload_json TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL, payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reports (
                    experiment_id TEXT PRIMARY KEY, report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def save_experiment(self, config: ExperimentConfig) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO experiments VALUES (?, ?, ?)",
                (config.experiment_id, config.model_dump_json(), _utc_now().isoformat()),
            )

    def save_run(self, run: ResearchRun) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?)",
                (run.run_id, run.experiment_id, run.status, run.model_dump_json(), _iso(run.completed_at)),
            )

    def save_event(self, event: ResearchEvent) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO events (run_id, timestamp, payload_json) VALUES (?, ?, ?)",
                (event.run_id, event.timestamp.isoformat(), event.model_dump_json()),
            )

    def list_runs(self, experiment_id: str | None = None) -> tuple[ResearchRun, ...]:
        query = "SELECT payload_json FROM runs"
        values: tuple[str, ...] = ()
        if experiment_id is not None:
            query += " WHERE experiment_id = ?"
            values = (experiment_id,)
        query += " ORDER BY run_id"
        with self._connect() as connection:
            return tuple(ResearchRun.model_validate_json(row[0]) for row in connection.execute(query, values))

    def list_events(self, experiment_id: str) -> tuple[ResearchEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT e.payload_json FROM events e JOIN runs r ON r.run_id=e.run_id "
                "WHERE r.experiment_id=? ORDER BY e.event_id",
                (experiment_id,),
            )
            return tuple(ResearchEvent.model_validate_json(row[0]) for row in rows)

    def save_report(self, report: ExperimentReport) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO reports VALUES (?, ?, ?)",
                (report.experiment_config.experiment_id, report.model_dump_json(), _utc_now().isoformat()),
            )

    def load_report(self, experiment_id: str) -> ExperimentReport:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT report_json FROM reports WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
        if row is None:
            raise ResearchPlatformError("No report exists for this experiment.")
        return ExperimentReport.model_validate_json(row[0])

    def list_reports(self) -> tuple[ExperimentReport, ...]:
        """Return persisted canonical reports newest first for the local dashboard."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT report_json FROM reports ORDER BY created_at DESC, experiment_id"
            )
            return tuple(ExperimentReport.model_validate_json(row[0]) for row in rows)


ArchitectureExecutor = Callable[[ExperimentConfig, str, int], Mapping[str, Any]]


class ExperimentRunner:
    """First-class bounded runner, parameterized by an authoritative executor."""

    def __init__(self, repository: SQLiteResearchRepository, *, git_commit: str | None = None) -> None:
        self.repository = repository
        self.git_commit = git_commit if git_commit is not None else _current_git_commit()

    def run(self, config: ExperimentConfig, executor: ArchitectureExecutor) -> ExperimentReport:
        self.repository.save_experiment(config)
        reproducibility = ReproducibilityRecord(
            python_version=sys.version.split()[0], platform=platform.platform(),
            git_commit=self.git_commit, model_provider=config.provider, model_name=config.model,
            seed=config.seed,
            prompt_versions=tuple((item.role, item.version) for item in config.prompt_versions),
        )
        runs: list[ResearchRun] = []
        events: list[ResearchEvent] = []
        for symbol in config.symbols:
            for repetition in range(1, config.repetitions + 1):
                run_id = f"run-{uuid.uuid4().hex}"
                started = _utc_now()
                start_event = ResearchEvent(
                    timestamp=started,
                    run_id=run_id,
                    node="experiment_runner",
                    status="STARTED",
                    event_type="RUN_STARTED",
                )
                events.append(start_event)
                self.repository.save_event(start_event)
                try:
                    result = _sanitize_mapping(dict(executor(config, symbol, repetition)))
                    run = ResearchRun(run_id=run_id, experiment_id=config.experiment_id,
                        architecture=config.architecture, symbol=symbol, repetition=repetition,
                        status=str(result.pop("status", "COMPLETED")), strategy_id=result.pop("strategy_id", None),
                        started_at=started, completed_at=_utc_now(), result=result,
                        reproducibility=reproducibility)
                except Exception as exc:
                    run = ResearchRun(run_id=run_id, experiment_id=config.experiment_id,
                        architecture=config.architecture, symbol=symbol, repetition=repetition,
                        status="FAILED", started_at=started, completed_at=_utc_now(),
                        failure_type=_classify_failure(exc), failure_code=type(exc).__name__,
                        reproducibility=reproducibility)
                self.repository.save_run(run)
                events.append(ResearchEvent(timestamp=_utc_now(), run_id=run_id, node="experiment_runner", status=run.status, event_type="RUN_FINISHED", metadata={"failure_type": run.failure_type} if run.failure_type else {}))
                self.repository.save_event(events[-1])
                runs.append(run)
        failures = tuple(run for run in runs if run.status != "COMPLETED")
        report = ExperimentReport(experiment_config=config, reproducibility=reproducibility,
            runs=tuple(runs), events=tuple(events), failures=failures,
            limitations=("LLM/provider executor must preserve validation, compiler, and backtest gates.",))
        self.repository.save_report(report)
        return report


def report_markdown(report: ExperimentReport) -> str:
    """Render a readable non-authoritative view from the canonical JSON model."""
    config = report.experiment_config
    completed = sum(run.status == "COMPLETED" for run in report.runs)
    lines = [f"# Experiment {config.experiment_id}", "", "## Executive Summary", "",
        f"- Architecture: `{config.architecture.value}`", f"- Runs: {completed}/{len(report.runs)} completed",
        f"- Dataset: `{config.dataset_id}`", f"- Symbols: {', '.join(config.symbols)}", "", "## Runs", ""]
    for run in report.runs:
        lines.append(f"- `{run.run_id}` — {run.symbol}, repetition {run.repetition}: **{run.status}**")
    if report.failures:
        lines += ["", "## Failure Analysis", ""]
        lines.extend(f"- `{run.run_id}`: `{run.failure_type or run.failure_code}`" for run in report.failures)
    lines += ["", "## Reproducibility", "", f"- Python: `{report.reproducibility.python_version}`", f"- Model: `{config.provider}/{config.model}`"]
    return "\n".join(lines) + "\n"


def _classify_failure(exc: Exception) -> str:
    name = type(exc).__name__.upper()
    if "VALID" in name:
        return FailureType.SCHEMA_VALIDATION_ERROR
    if "COMPIL" in name:
        return FailureType.COMPILATION_ERROR
    if "BACKTEST" in name:
        return FailureType.BACKTEST_ERROR
    if "DATA" in name:
        return FailureType.DATA_ERROR
    return FailureType.UNKNOWN


def _contains_sensitive_key(value: Mapping[str, Any]) -> bool:
    """Reject event metadata that could write a secret at any nesting level."""

    for key, item in value.items():
        if any(part in str(key).lower() for part in _SENSITIVE_KEY_PARTS):
            return True
        if isinstance(item, Mapping) and _contains_sensitive_key(item):
            return True
        if isinstance(item, (list, tuple)) and any(
            isinstance(entry, Mapping) and _contains_sensitive_key(entry) for entry in item
        ):
            return True
    return False


def _sanitize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Redact sensitive executor output before it becomes research evidence."""

    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        key_string = str(key)
        if any(part in key_string.lower() for part in _SENSITIVE_KEY_PARTS):
            sanitized[key_string] = "[REDACTED]"
        elif isinstance(item, Mapping):
            sanitized[key_string] = _sanitize_mapping(item)
        elif isinstance(item, list):
            sanitized[key_string] = [
                _sanitize_mapping(entry) if isinstance(entry, Mapping) else entry
                for entry in item
            ]
        elif isinstance(item, tuple):
            sanitized[key_string] = tuple(
                _sanitize_mapping(entry) if isinstance(entry, Mapping) else entry
                for entry in item
            )
        else:
            sanitized[key_string] = item
    return sanitized


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _current_git_commit() -> str | None:
    """Best-effort code identity; a non-Git installation remains runnable."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            cwd=Path.cwd(),
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None
