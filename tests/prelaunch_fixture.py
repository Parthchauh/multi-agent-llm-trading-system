"""Development-only deterministic E2E evidence: python -m tests.prelaunch_fixture.

Uses recorded structured responses and synthetic prices. It does not test live
model/data providers and deliberately demonstrates insufficient-evidence rejection.
"""

import argparse
import json
from pathlib import Path

from data.provenance import environment_manifest
from scripts.run_research_e2e import run_e2e_research
from tests.conftest_helpers import make_ohlcv
from tests.test_research_e2e import MockLifecycleClient


def run_fixture(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = make_ohlcv(350, seed=1)
    result = run_e2e_research(
        symbol="AAA",
        df=frame,
        structured_client=MockLifecycleClient(),
        mode="SINGLE_AGENT",
        seed=1,
        output_dir=output,
        exposure_ledger_path=output / "campaign.sqlite",
    )
    if result["status"] != "VIABILITY_REJECTED" or result["certification"]["DEPLOYMENT_ELIGIBLE"]:
        raise AssertionError(
            "Recorded fixture did not produce its expected safe research rejection."
        )
    (output / "fixture_result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": result["status"],
                "outcome_class": "EXPECTED_RESEARCH_REJECTION",
                "runtime_failure": False,
                "certification": result["certification"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    inputs = {
        "kind": "synthetic_prices_and_recorded_llm_responses",
        "seed": 1,
        "index": [stamp.isoformat() for stamp in frame.index],
        "columns": list(frame.columns),
        "data": frame.to_numpy().tolist(),
    }
    (output / "fixture_inputs.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    (output / "environment.json").write_text(
        json.dumps(environment_manifest(Path(__file__).resolve().parents[1]), indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "certification": result["certification"],
                "output": str(output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/prelaunch_fixture")
    run_fixture(parser.parse_args().output)
