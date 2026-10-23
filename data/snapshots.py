"""Content-addressed offline datasets and reproducible deterministic replay.

Hashes detect modification, not the truth of a vendor's adjustment or universe
claims. No credentials, raw model responses, or executable objects belong here.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backtesting.engine import BacktestEngine
from backtesting.models import BacktestConfig
from data.provenance import environment_manifest
from data.schema import CANONICAL_OHLCV_COLUMNS, validate_ohlcv
from strategies.compiler import StrategyCompiler
from strategies.schema import StrategySchema
from strategies.validator import StrategyValidator


def canonical_bytes(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


class DatasetIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    symbol: str = Field(min_length=1, max_length=32)
    provider: str = Field(min_length=1, max_length=128)
    provider_version: str = Field(min_length=1, max_length=128)
    query: dict[str, str | int | float | bool | None]
    adjustment: str
    timezone: str
    retrieved_at: datetime
    calendar_id: str | None = None
    calendar_version: str | None = None
    calendar_source: str | None = None
    universe_basis: str = "single_symbol_survivorship_unknown"
    lifecycle_basis: str = "delisting_history_unverified"

    @field_validator("adjustment")
    @classmethod
    def adjustment_basis(cls, value):
        if value not in {"raw", "split_adjusted", "total_return_adjusted"}:
            raise ValueError("Declare raw, split_adjusted or total_return_adjusted.")
        return value

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value):
        ZoneInfo(value)
        return value

    @field_validator("retrieved_at")
    @classmethod
    def aware(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Retrieval timestamp must have a timezone.")
        return value

    @field_validator("query")
    @classmethod
    def no_secrets_or_endpoints(cls, value):
        # Query identity is deliberately restricted to non-sensitive parameters.
        allowed = {"symbol", "start", "end", "interval", "exchange", "currency", "adjustment"}
        if set(value)-allowed:
            raise ValueError("Query contains unsupported or potentially sensitive fields.")
        canonical_bytes(value)
        return value


def create_snapshot(root: Path, frame: pd.DataFrame, identity: DatasetIdentity,
                    *, expected_sessions: pd.DatetimeIndex | None = None) -> str:
    frame = validate_ohlcv(frame).loc[:, list(CANONICAL_OHLCV_COLUMNS)]
    calendar = None
    if expected_sessions is not None:
        if not all((identity.calendar_id, identity.calendar_version, identity.calendar_source)):
            raise ValueError("Expected sessions require a versioned calendar and its source.")
        sessions = pd.DatetimeIndex(expected_sessions)
        if sessions.empty or sessions.has_duplicates or not sessions.is_monotonic_increasing or sessions.hasnans:
            raise ValueError("Calendar sessions must be unique, ordered and non-empty.")
        if str(sessions.tz) != str(frame.index.tz):
            raise ValueError("Calendar and bar timestamp timezone conventions must match.")
        if not sessions.equals(frame.index):
            raise ValueError("Dataset has missing or unexpected calendar sessions.")
        calendar = [stamp.isoformat() for stamp in sessions]
    if frame.index.tz is not None and str(frame.index.tz) != identity.timezone:
        raise ValueError("Declared timezone differs from the bar index.")
    payload = {"schema_version": 1, "identity": identity.model_dump(mode="json"),
               "timestamps": [stamp.isoformat() for stamp in frame.index],
               "timestamp_timezone": str(frame.index.tz) if frame.index.tz is not None else None,
               "columns": list(CANONICAL_OHLCV_COLUMNS), "values": frame.astype(float).values.tolist(),
               "expected_sessions": calendar,
               "quality": {"zero_volume_bars": int((frame.Volume == 0).sum()),
                           "session_coverage_checked": calendar is not None,
                           "corporate_actions_independently_verified": False,
                           "point_in_time_universe_verified": False,
                           "symbol_lifecycle_verified": False}}
    content = canonical_bytes(payload)
    digest = hashlib.sha256(content).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / (digest+".json")
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise ValueError("Existing snapshot content does not match its identifier.") from None
    return digest


def load_snapshot(root: Path, digest: str) -> tuple[pd.DataFrame, dict]:
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Snapshot identifier must be a SHA256 digest.")
    content = (root / (digest+".json")).read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Dataset snapshot digest mismatch.")
    payload = json.loads(content)
    if payload.get("schema_version") != 1 or payload.get("columns") != list(CANONICAL_OHLCV_COLUMNS):
        raise ValueError("Unsupported snapshot schema.")
    identity = DatasetIdentity.model_validate(payload["identity"], strict=False)
    index = pd.DatetimeIndex(pd.to_datetime(payload["timestamps"], utc=payload["timestamp_timezone"] is not None))
    if payload["timestamp_timezone"]:
        index = index.tz_convert(payload["timestamp_timezone"])
    frame = validate_ohlcv(pd.DataFrame(payload["values"], columns=payload["columns"], index=index))
    if payload["expected_sessions"] is not None:
        if payload["expected_sessions"] != [stamp.isoformat() for stamp in frame.index]:
            raise ValueError("Snapshot calendar coverage mismatch.")
    return frame, {**payload, "identity": identity.model_dump(mode="json"), "sha256": digest}


def record_replay(*, snapshot_root: Path, dataset_sha256: str, strategy: StrategySchema,
                  config: BacktestConfig, source_root: Path, output: Path) -> dict:
    if not StrategyValidator().validate(strategy).is_valid:
        raise ValueError("Strategy failed semantic validation.")
    frame, _ = load_snapshot(snapshot_root, dataset_sha256)
    result = BacktestEngine(config).run(frame, StrategyCompiler().compile(strategy, frame)).to_dict()
    bundle = {"schema_version": 1, "kind": "DETERMINISTIC_REPLAY", "dataset_sha256": dataset_sha256,
              "strategy": strategy.model_dump(mode="json"), "config": config.model_dump(mode="json"),
              "result_sha256": hashlib.sha256(canonical_bytes(result)).hexdigest(), "result": result,
              "environment": environment_manifest(source_root),
              "recorded_at": datetime.now(timezone.utc).isoformat(),
              "statistical_certification": False, "paper_eligible": False}
    bundle["bundle_sha256"] = hashlib.sha256(canonical_bytes(bundle)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical_bytes(bundle))
    return bundle


def verify_replay(bundle_path: Path, snapshot_root: Path, source_root: Path) -> dict:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    expected = bundle.pop("bundle_sha256")
    if hashlib.sha256(canonical_bytes(bundle)).hexdigest() != expected:
        raise ValueError("Replay bundle digest mismatch.")
    if bundle.get("schema_version") != 1 or bundle.get("kind") != "DETERMINISTIC_REPLAY":
        raise ValueError("Unsupported replay bundle.")
    frame, _ = load_snapshot(snapshot_root, bundle["dataset_sha256"])
    strategy = StrategySchema.model_validate(bundle["strategy"])
    if not StrategyValidator().validate(strategy).is_valid:
        raise ValueError("Replay strategy semantic validation failed.")
    result = BacktestEngine(BacktestConfig.model_validate(bundle["config"])).run(
        frame, StrategyCompiler().compile(strategy, frame)).to_dict()
    actual = hashlib.sha256(canonical_bytes(result)).hexdigest()
    if actual != bundle["result_sha256"] or result != bundle["result"]:
        raise ValueError("Replay output diverged from the recorded deterministic result.")
    current = environment_manifest(source_root)
    keys = ("python", "os", "working_tree_code_sha256", "installed_project_code_sha256", "packages", "lockfile_sha256")
    mismatches = [key for key in keys if current.get(key) != bundle["environment"].get(key)]
    return {"status": "REPLAY_MATCH", "result_sha256": actual,
            "environment_match": not mismatches, "environment_mismatches": mismatches,
            "statistical_certification": False, "paper_eligible": False}
