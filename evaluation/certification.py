"""Deterministic certification gates and a one-way evaluation lifecycle."""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from evaluation.models import EvaluationScope, ViabilityPolicy


def launch_viability_policy(scope=EvaluationScope.VALIDATION):
    return ViabilityPolicy(
        policy_id="research-viability:v1",
        required_scope=scope,
        minimum_trade_count=20,
        minimum_total_return_pct=0.0,
        minimum_sharpe_ratio=0.5,
        maximum_drawdown_pct=20.0,
        maximum_turnover_pct=2000.0,
    )


class Stage(str, Enum):
    ADAPTIVE = "VALIDATION_ADAPTIVE"
    FROZEN = "FROZEN"
    TEST = "TEST_CONFIRMATORY"
    HOLDOUT = "FINAL_HOLDOUT"
    COMPLETE = "COMPLETE"


@dataclass
class EvaluationLifecycle:
    stage: Stage = Stage.ADAPTIVE
    strategy_hash: str | None = None
    transitions: list[str] = field(default_factory=lambda: [Stage.ADAPTIVE.value])
    state_path: Path | None = None

    def _persist(self):
        if self.state_path is None:
            return
        path = Path(self.state_path)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "stage": self.stage.value,
                    "strategy_hash": self.strategy_hash,
                    "transitions": self.transitions,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def require_adaptive(self):
        if self.stage != Stage.ADAPTIVE:
            raise ValueError("Refinement is forbidden after strategy freeze.")

    def freeze(self, strategy_hash):
        self.require_adaptive()
        if not strategy_hash:
            raise ValueError("A frozen strategy hash is required.")
        self.strategy_hash = strategy_hash
        self.stage = Stage.FROZEN
        self.transitions.append(self.stage.value)
        self._persist()

    def begin_test(self, strategy_hash):
        if self.stage != Stage.FROZEN or strategy_hash != self.strategy_hash:
            raise ValueError("Confirmatory test requires the frozen strategy.")
        self.stage = Stage.TEST
        self.transitions.append(self.stage.value)
        self._persist()

    def begin_holdout(self, *, test_passed):
        if self.stage != Stage.TEST or not test_passed:
            raise ValueError("Final holdout requires a passing confirmatory test.")
        self.stage = Stage.HOLDOUT
        self.transitions.append(self.stage.value)
        self._persist()

    def complete(self):
        if self.stage != Stage.HOLDOUT:
            raise ValueError("Final holdout has not been evaluated.")
        self.stage = Stage.COMPLETE
        self.transitions.append(self.stage.value)
        self._persist()


def evidence_status(
    *,
    technical,
    validation_passed,
    test_passed,
    holdout_passed,
    risk_passed,
    robustness_passed=False,
    release_gates_passed=False,
):
    if robustness_passed or release_gates_passed:
        raise ValueError(
            "Statistical certification and paper promotion are disabled; boolean assertions are not certification evidence."
        )
    screening = bool(technical and validation_passed and test_passed and holdout_passed)
    viable = bool(screening and robustness_passed)
    return {
        "TECHNICALLY_VALID": bool(technical),
        "STATISTICALLY_VIABLE": viable,
        "RISK_ACCEPTED": bool(risk_passed),
        "DEPLOYMENT_ELIGIBLE": bool(viable and risk_passed and release_gates_passed),
        "performance_screening_passed": screening,
        "evidence_state": "SUFFICIENT" if viable else "INSUFFICIENT_EVIDENCE",
        "real_money_execution": False,
    }
