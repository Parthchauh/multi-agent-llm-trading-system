# P0 acceptance review and clean environment verification

Review checkpoint: 21 October 2026, Asia/Calcutta. Repository: `D:\TradingAgents`.

The reviewed working-tree candidate passes the P0 regression checks in a separate
locked environment. This is not a release approval: the committed lockfile is
stale, the reviewed implementation remains uncommitted, Docker is unverified,
and no independent human review has occurred. Statistical certification, paper
promotion, and broker execution remain disabled. No later feature package was
started, and nothing was committed, pushed, merged, or deployed.

## Repository and historical evidence

The starting branch was `main`, HEAD
`79f2411b56ee24793abb2a42214674f8d80eeac8`. Existing modifications, untracked user
files, the presentation, and `research.sqlite` were preserved. Starting status is
recorded in [initial-git-status.txt](acceptance-evidence/initial-git-status.txt).
No applicable repository AGENTS.md was found. Ponytail is not available in this
session; the review below is automated testing and direct self-review, not an
independent peer review.

The [historical baseline record](baseline-results.md) remains unchanged. The
original suite produced 376 passes and 15 temporary-directory setup errors. The
391-pass rerun followed initial financial changes. There is no successful
untouched-baseline run or pre-change E2E snapshot. All evidence in this review is
post-change evidence; none repairs or replaces that historical gap.

## Acceptance matrix against actual behavior

| Guide test | Reviewed expectation | Evidence and result |
|---|---|---|
| AT-01 | BUY deducts notional once without costs | PASS: `test_manual_round_trip`, zero-cost case. Starting cash 1000, five shares at 100, remaining cash 500. |
| AT-02 | Slippage affects cash once through fill price | PASS: five shares at 101, commission 2, attribution 5: cash 493, not 488. Attribution is excluded from cash mutation. |
| AT-03 | Round-trip cash, fees and P&L reconcile | PASS: sell the same five shares at 109 with commission 2; proceeds 543, ending cash 1036, realized P&L 36. Four independent fee/slippage cases cover both legs. Oversells and duplicate fills are rejected. |
| AT-04 | Identical backtest/paper ordered fills | PASS: eight scenarios now check independent constants as well as adapter parity. See the golden ledger below. |
| AT-05 | Deterministic stop/target collision | PASS: entry-bar collision exits at the stop, 94.905 after slippage, on 2 January; final equity 9466.389505. |
| AT-06 | Missing launch risk thresholds cause configuration failure | PASS: launch preflight rejects incomplete policies. Named profiles and exact-threshold/breach cases are covered. Low-level exploratory RiskPolicy is not a certification authority. |
| AT-07 | One trade cannot establish viability | PASS: direct check of `launch_viability_policy()` rejects one trade against minimum 20; [one-trade-gate.json](acceptance-evidence/one-trade-gate.json). A valid but insufficient local fixture is also rejected. |
| AT-08 | Refinement after freeze/test is prohibited | PASS: lifecycle guard rejects adaptive work after freeze; instrumented nine-call research/debate/review/refinement run finishes all agent calls before test release. |
| AT-09 | Premature/repeated holdout access is rejected | PASS for canonical API: matching lifecycle stage/hash and prior test release are required; one release per run. Persistent overlap claims survive separate processes, new campaigns, new hashes and provider aliases. Explicit legacy opt-in remains exploratory only. |
| AT-10 | Conventional full-allocation B&H matches arithmetic | PASS: 1000 buys ten shares at 100; exit at 120 gives 1200 with zero costs. |
| AT-11 | Exposure-matched B&H is distinct and labeled | PASS: 20% allocation buys two shares, retains 800 cash, and ends at 1040. Allocation, costs and overlays are recorded in benchmark metadata. |
| AT-12 | Canonical CLI follows authoritative modules | PASS in clean source-suite and isolated installed-package smoke; installed module paths are verified under the clean environment's site-packages. Console help checks pass for both research aliases and the explicitly legacy command. |

The matrix verifies the supported long-only, one-symbol, whole-share local
research/replay contract. It does not certify live fills, partial orders, broker
idempotency, a statistically significant strategy, or a paper beta.

## Independent execution expectations

The earlier parity test compared two adapters sharing the same kernel, with
only the collision reason checked independently. That could miss a shared
execution defect. The strengthened test fixes entry at 100.1, quantity at 99,
entry date at 2 January 2020, and entry cash at 80.1901 after commission.
Expected constants are written directly in the test rather than obtained from
TradeExecutor, Portfolio, the kernel, or the other adapter. Golden comparisons
use absolute tolerance 1e-8 and relative tolerance zero.

| Scenario | Exit date in January 2020 | Executed exit price | Ending equity |
|---|---:|---:|---:|
| Close-derived exit signal | 4 | 99.9 | 9960.4 |
| Intrabar stop | 3 | 94.905 | 9466.389505 |
| Intrabar target | 3 | 109.89 | 10948.42099 |
| Gap through stop | 3 | 89.91 | 8972.37901 |
| Entry-bar stop and target collision | 2 | 94.905 | 9466.389505 |
| Maximum holding period | 3 | 99.9 | 9960.4 |
| Explicit EOF close | 5 | 100.899 | 10059.202099 |
| Repeated entry signals while invested | 5 | 100.899 | 10059.202099 |

## Information boundaries and bounded corrections

`research_graph.py` still requested the old development view, which included a
confirmatory partition even though the graph used only validation. It now uses
`adaptive_view()`, as do the refinement graph and baseline stage. Canonical
ResearchDatasetSplit instances are strict by default: `development_view()` is
rejected, and test/holdout release verifies lifecycle stage, frozen execution
hash and release order. Constructing a FrozenStrategySelection alone no longer
releases canonical data. The old low-level test fixtures explicitly request
`legacy_access=True`, making compatibility behavior visible.

Tests instrument sealed partition reads and place conspicuous price sentinels
in the sealed partitions. Initial generation receives only training features;
debate, risk interpretation, criticism and refinement consume validation
evidence. Sentinel values do not appear in agent messages, invocation audit or
captured logs. A rejected run never reads test or holdout feature frames; a
successful adaptive sequence finishes nine agent calls before test release.
Logs may disclose coverage dates and row counts, and artifacts retain split
identity/fingerprints; these are metadata, not prematurely disclosed outcomes.
Metric artifacts are written after adaptive processing ends.

These APIs are procedural safeguards, not a sandbox against arbitrary Python
code or a person already possessing the input dataset. Explicit legacy access,
direct BacktestEngine use and the backtest CLI can perform exploratory research;
they confer no statistical certification or promotion authority. Low-level
permissive risk policies can produce exploratory assessments, but the launch
entry point requires a complete profile. Boolean assertions cannot enable
statistical/paper certification, and persisted ResearchRun validation rejects
unsupported positive certification from a custom executor.

Holdout exposure uses a SQLite transaction before disclosure. Separate-process
tests retain contamination across strategy/campaign/hash changes and partly
overlapping windows. Provider-adapter renaming is now conservatively grouped by
the symbol family, including existing rows without rewriting their history.
The CLI anchors this ledger to the research database directory, rather than
the installed package or per-run output directory. Direct API users must retain
the same explicit ledger across workspaces. Deleting/replacing the ledger or
renaming the instrument cannot erase knowledge and is outside these local API
guarantees.

The existing recorded-response fixture contained obsolete BullCase field names
and unstructured change strings. The newly exercised full review branch exposed
those invalid fixture responses. They were corrected to the current schemas;
no production validation or boundary assertion was relaxed.

## Clean environment and lock identities

The committed lockfile and current candidate are different:

- Committed `HEAD:uv.lock` SHA256: `f64183494b2bd391712a2e06ae1b2692e384ca1b84c514a13aa4b5280d93e417`.
- Working-tree lockfile SHA256: `6885b73bc318a07bb60e7f1d22a3b52f73a55ddb3a4136a6cd6a2eec3999096e`.

The committed bytes were extracted without text conversion into an isolated
project projection containing the current pyproject and README. `uv lock --check`
fails there with exit 1: the lockfile needs updating. Its project version was
0.2.0 rather than current 0.2.3. Installing the current project from that committed
lock under the supported locked workflow is therefore **blocked**. It was not
silently accepted with `--frozen`, and no commit was created to hide this mismatch.
The current candidate passes the same consistency check with exit 0.

Supplemental clean verification used the matching uncommitted candidate,
uv 0.13.3, CPython 3.13.9 and Windows. The environment is
`.tools/p0-acceptance/clean-env`, with `include-system-site-packages = false`.
uv resolved 134 packages and installed 121 applicable packages. Installation
was non-editable; subsequent offline refreshes reinstalled only tradingagents
after the bounded review fixes. [clean-environment.json](acceptance-evidence/clean-environment.json)
records actual versions. The original `.venv` before/after distributions match
exactly; it was never synchronized or installed into.

Exact commands, run from `D:\TradingAgents` in PowerShell, were:

```powershell
.\.tools\uv\bin\uv.exe lock --check --offline --project .tools\p0-acceptance\committed-lock-project --cache-dir .tools\uv-cache --python .venv\Scripts\python.exe
.\.tools\uv\bin\uv.exe lock --check --offline --cache-dir .tools\uv-cache --python .venv\Scripts\python.exe
$env:UV_PROJECT_ENVIRONMENT='D:\TradingAgents\.tools\p0-acceptance\clean-env'
.\.tools\uv\bin\uv.exe sync --locked --extra dev --no-editable --python .venv\Scripts\python.exe --cache-dir .tools\uv-cache
.\.tools\uv\bin\uv.exe sync --offline --locked --extra dev --no-editable --reinstall-package tradingagents --python .venv\Scripts\python.exe --cache-dir .tools\uv-cache
.\.tools\p0-acceptance\clean-env\Scripts\python.exe -m pytest -m 'not live' -q --basetemp=.test_tmp_p0_acceptance_final -p no:cacheprovider
.\.tools\p0-acceptance\clean-env\Scripts\python.exe -I .tools\p0-acceptance\installed_smoke.py
.\.tools\p0-acceptance\clean-env\Scripts\python.exe -I .tools\p0-acceptance\fixture_installed.py
.\.tools\p0-acceptance\clean-env\Scripts\python.exe -I -m pytest .tools/p0-acceptance/outside/tests/test_research_cli.py::test_canonical_cli_runs_real_local_pipeline -c .tools/p0-acceptance/outside/pytest.ini --rootdir=.tools/p0-acceptance/outside --basetemp=.tools/p0-acceptance/outside/test-tmp-final -q -p no:cacheprovider
```

The isolated helper copies only test Python files into an outside fixture
directory, excludes repository imports with `-I`, and asserts that CLI and E2E
modules resolve under the clean environment's site-packages. Each console
command is launched with that outside directory as its working directory:
`trading-research.exe --help`, `tradingagents.exe --help`, and
`tradingagents-legacy.exe --help`. All three previously returned exit 0;
the final refreshed-wheel repetition also returned three exit codes 0.
The legacy help check does not execute its graph or certify it.

The exact verification helper sources are retained in
[acceptance-tools](acceptance-tools). Prepare their isolated inputs with:

```powershell
New-Item -ItemType Directory -Force .tools\p0-acceptance\outside\tests
Copy-Item tests\*.py .tools\p0-acceptance\outside\tests
Set-Content .tools\p0-acceptance\outside\tests\__init__.py ''
Copy-Item docs\prelaunch\acceptance-tools\pytest.ini .tools\p0-acceptance\outside\pytest.ini
Copy-Item docs\prelaunch\acceptance-tools\installed_smoke.py,docs\prelaunch\acceptance-tools\fixture_installed.py .tools\p0-acceptance
```

These are Windows verification helpers for the recorded `D:\TradingAgents`
workspace, not installed product commands. Raw evidence files and their paired
exit-code files are retained under acceptance-evidence. A first attempt launched
from the nested working directory hit workspace log-write permissions before
its tests ran; it is not counted as a pass. The recorded successful reruns launch
from the workspace and use Python isolation plus explicit outside fixture paths.

Static check command:

```powershell
.\.tools\p0-acceptance\clean-env\Scripts\ruff.exe check backtesting paper_trading risk evaluation data cli\research.py scripts\run_research_e2e.py tradingagents\graph\research_graph.py tradingagents\graph\sprint6_lifecycle.py tradingagents\agents\telemetry.py tests\test_prelaunch.py tests\test_p0_acceptance_review.py tests\prelaunch_fixture.py --per-file-ignores 'backtesting/__init__.py:I001' --output-format concise
git -c safe.directory=D:/TradingAgents diff --check
```

## Actual results and interpretation

| Check | Actual result | Evidence |
|---|---|---|
| Committed lock consistency | FAIL, exit 1; reproducible committed installation blocked | committed-lock-check.txt, committed-lock-exit.txt |
| Candidate lock consistency | PASS, exit 0 | candidate-lock-check.txt |
| Clean candidate install / final offline project refresh | PASS, exit 0 | clean-install.txt, clean-refresh-final.txt |
| Full non-live suite after final ledger fix | PASS: 427 tests, 27 subtests, two warnings, 117.00 seconds | clean-full-tests-final.txt |
| Relevant lint | PASS with one explicit pre-existing import-order exclusion | static-final.txt; unexcluded initial finding in static-initial.txt |
| Diff whitespace | PASS, exit 0 | diff-check-final.txt |
| Installed console commands | PASS, three exit codes 0 | installed-smoke.json and command help logs |
| Installed canonical CLI with local inputs | PASS, one test, exit 0; 20.73 seconds after final wheel refresh | installed-canonical-cli-final.txt |
| Installed recorded-response fixture | PASS, exit 0; expected rejection | installed-fixture-final.txt, installed-fixture-result.json |
| Existing environment preservation | PASS, package distributions unchanged | original-environment-comparison.txt |
| Docker | UNVERIFIED; daemon pipe unavailable; no service changed | docker-check.txt |

The lint exclusion is limited to the existing I001 import-order issue in
`backtesting/__init__.py`, which was not changed. No type-checker is configured;
this review does not claim a type-check pass. The full suite warnings concern
LangGraph's future serializer default and an older UTC fixture call, not failed
assertions. No live provider/model or external-market test ran.

Expected research rejection is now distinct from a runtime failure. The fixture
exits 0 and writes `fixture_result.json` containing
`status: VIABILITY_REJECTED`, `outcome_class: EXPECTED_RESEARCH_REJECTION`,
`runtime_failure: false`, and `evidence_state: INSUFFICIENT_EVIDENCE`.
STATISTICALLY_VIABLE, DEPLOYMENT_ELIGIBLE and real_money_execution are false.
Experiment reports now put research rejections in `rejections`, while caught
runtime exceptions enter `failures` with outcome_class RUNTIME_FAILURE. Older
stored reports are not retroactively rewritten or relabeled as baseline evidence.
Full generated fixture artifacts remain under `.tools/p0-acceptance/outside/fixture`;
compact result/certification records are copied into acceptance-evidence.

## Remaining blockers and bounded next package

Before claiming reproducibility from a release commit, obtain review and commit
the coherent implementation and matching lockfile; then repeat locked verification
from that commit. Other Python versions and operating systems are unverified.
Docker image build/digest verification remains outstanding. No independent peer
review, live-provider verification, paper observation period or deployment review
has been completed. These limitations do not convert the clean candidate test
results into failures, but they prevent a full release-readiness claim.

The next research-robustness package should be bounded to an audit and local
verification of existing rolling/expanding walk-forward and cost-sensitivity
components. Specify fold train/validation/unseen-test boundaries, freeze per fold,
record all candidate/refinement counts and fold distributions, and use hand-sized
chronological fixtures to reject future-data access. Reuse existing evaluators;
do not add a parameter optimizer, new model integrations or speculative frameworks.
Deliver deterministic fold/cost artifacts, leakage tests, reproducibility commands
and an explicit report of limitations. A fixed strategy selected using later
folds must not be presented as independently trained in earlier folds.

PSR/DSR methodology, statistical certification, paper promotion, new production
features and broker execution are outside that next package. Keep all promotion
flags disabled. Preserve the shared exposure ledger and the historical baseline
gap throughout.
