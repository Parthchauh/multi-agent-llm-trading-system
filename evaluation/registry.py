"""Immutable research candidates. Registration never grants paper eligibility."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

from backtesting.models import BacktestConfig
from data.snapshots import canonical_bytes
from risk.models import RiskPolicy
from risk.profiles import require_complete_policy
from strategies.schema import StrategySchema
from strategies.validator import StrategyValidator


class FrozenCandidateRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS frozen_candidates "
                       "(candidate_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def freeze(self, *, strategy: StrategySchema, config: BacktestConfig,
               risk_policy: RiskPolicy, evidence: dict[str, str]) -> str:
        if not StrategyValidator().validate(strategy).is_valid:
            raise ValueError("Candidate semantic validation failed.")
        policy = require_complete_policy(risk_policy)
        if config.position_size_pct > min(policy.max_position_pct, policy.max_portfolio_exposure_pct,
                                           strategy.position_sizing.value):
            raise ValueError("Execution allocation exceeds the frozen strategy/risk limits.")
        if config.maximum_holding_days != strategy.exit.maximum_holding_days:
            raise ValueError("Execution holding period differs from frozen strategy.")
        if not evidence or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in evidence.values()):
            raise ValueError("Candidate needs named SHA256 evidence references.")
        payload = {"schema_version": 1, "strategy": strategy.model_dump(mode="json"),
                   "config": config.model_dump(mode="json"), "risk_policy": policy.model_dump(mode="json"),
                   "evidence": evidence, "status": "FROZEN_RESEARCH_CANDIDATE",
                   "statistical_certification": False, "paper_eligible": False,
                   "real_money_eligible": False,
                   "promotion_blockers": ["STATISTICAL_CERTIFICATION_DISABLED", "RELEASE_APPROVALS_MISSING"]}
        content = canonical_bytes(payload)
        identity = hashlib.sha256(content).hexdigest()
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT OR IGNORE INTO frozen_candidates VALUES (?, ?)", (identity, content.decode()))
        self.get(identity)  # Detect a previously corrupted row rather than silently reusing it.
        return identity

    def get(self, candidate_id: str) -> dict:
        if not re.fullmatch(r"[0-9a-f]{64}", candidate_id):
            raise ValueError("Candidate identifier must be a SHA256 digest.")
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT payload FROM frozen_candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
        if row is None:
            raise KeyError("Frozen candidate not found.")
        payload = json.loads(row[0])
        if hashlib.sha256(canonical_bytes(payload)).hexdigest() != candidate_id:
            raise ValueError("Immutable candidate integrity check failed.")
        return {"candidate_id": candidate_id, **payload}

    def require_paper_eligible(self, candidate_id: str) -> None:
        self.get(candidate_id)
        raise PermissionError("Paper promotion is disabled: reviewed statistical and release evidence is required.")
