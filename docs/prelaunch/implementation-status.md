# Prelaunch implementation status

Baseline commit: `79f2411b56ee24793abb2a42214674f8d80eeac8`, branch `main`.
Existing user work was preserved. No commit, tag, push, merge, deployment, or
real-money action was performed. This is a P0 implementation checkpoint, not
completion of the entire guide. No independent peer review has occurred.

## Extended implementation checkpoint on 22 October 2026

The subsequent request to complete the guide expanded work beyond the earlier
P0-only package. Local research campaigns, uncertainty diagnostics, adjustment
contracts, immutable datasets/replay, invocation accounting, frozen registry,
durable paper simulation controls/drift, a separate TLS tenant-scoped evidence
API, backup/restore, richer evidence views and locked verification workflows have
been added. This is not a declaration that the entire guide or external beta is
complete. See project-completion-plan.md, project-completion-review.md,
research-methodology.md, operating-runbook.md and release-checklist.md.

The initial dependency audit found 141 advisories in 15 installed packages.
Compatible security floors and the candidate lock were updated; the patched
clean environment's repeated audit reported no known vulnerabilities. A complete
non-live run passed 481 tests and 27 subtests before the final drift/audit
regressions; the final verification record is maintained separately. Static
checks are scoped and do not imply whole-repository type correctness or an
independent security audit. Historical P0 results below remain historical.

Statistical certification, paper promotion and broker execution remain disabled.
Five canonical architecture executors are wired; a genuine five-treatment study,
reviewed certification, licensed PIT/calendar/corporate-action evidence, real
paper observation, independent review and release/terms approvals remain open.
Docker's daemon is unavailable; no host service was changed. Remote workflows
have been written but not run. No push, merge, commit or deployment occurred.

## Acceptance review checkpoint on 21 October 2026

The requested P0 acceptance review and clean-environment verification package is
complete. See [p0-acceptance-review.md](p0-acceptance-review.md) for the acceptance
matrix, bounded fixes, exact commands, evidence and remaining blockers.

- Final clean candidate suite: **427 passed, 27 subtests passed**, two warnings.
- Non-editable installed package verified outside the source tree: three console
  help commands passed, canonical CLI smoke **1 passed**, recorded fixture exited
  0 with `EXPECTED_RESEARCH_REJECTION`, `runtime_failure: false` and all promotion
  flags false. Existing `.venv` distributions are unchanged.
- Relevant Ruff check passed with the documented existing I001 exception in
  backtesting/__init__.py. Diff whitespace check passed. Ponytail is unavailable;
  no independent peer review is claimed.
- **Committed reproducibility remains blocked:** HEAD's lockfile fails the
  current project check. The uncommitted matching candidate lock passes and was
  used for the separate clean installation. No commits or host-service changes
  were made. Docker remains unverified because its daemon is unavailable.
- Canonical split release now verifies lifecycle stage/hash/order. Legacy views
  require explicit opt-in. Eight parity scenarios have independent numerical
  expectations; cross-process exposure includes provider aliases. Persisted
  reports separate research rejection from runtime failure and reject unsupported
  positive certification from custom executors.

Statistical certification, paper promotion and broker execution remain disabled.
This checkpoint does not remove the untouched-baseline evidence gap or certify
completion of the guide's later phases.

| Package | Requirement and actual code paths | Verified status | Dependencies and acceptance | Tests and evidence | Blockers and next action |
|---|---|---|---|---|---|
| Phase 0 | Baseline and environment, docs/prelaunch | IMPLEMENTED with historical evidence gap | Record state and repeatable inputs | Original suite: 376 passed/15 temp setup errors; early regression rerun: 391 passed | No successful untouched baseline or original E2E snapshot; details in baseline-results.md |
| P0-01 | Slippage/cash/P&L, paper_trading/models.py and portfolio.py | VERIFIED_COMPLETE for supported full fills | Independently calculated ledger reconciliation | Four cost combinations, commission once, round trip, oversell rejection, finite values | Partial fills/exits are unsupported and rejected |
| P0-02 | Shared execution, backtesting/kernel.py, engine.py, paper_trading/runner.py | VERIFIED_COMPLETE for local replay | Identical fills and independently expected equity | Eight scenarios with literal expected fills/dates/quantities/equity at absolute 1e-8 tolerance; checkpoint/restart and tamper detection | Local replay recovery is not broker-order idempotency or a live paper-beta claim |
| P0-03 | risk/profiles.py, models.py, evaluator.py; E2E preflight | VERIFIED_COMPLETE for launch-policy validation | All mandatory thresholds, profile/hash, reason codes | Missing policy rejected; three named profiles; exact boundary and breach cases | Thresholds are proposed research screening defaults, not validated investment limits; paper remains disabled |
| P0-04 | evaluation/certification.py, platform.py; runner; CLI/dashboard serializers | VERIFIED_COMPLETE for fail-closed safety behavior | Four separate statuses; incomplete certification cannot promote | Low-evidence and one-trade rejection; boolean assertion and custom-executor bypass rejected; rejection distinguished from runtime failure | Positive statistical certification is unimplemented and disabled; no candidate is paper eligible |
| P0-05 | data/research_split.py, evaluation/certification.py and campaigns.py, runner | VERIFIED_COMPLETE for canonical procedural guards | Stage/hash/order checked; adaptive views omit test; persistent exposure | Separate-process overlap rejection across campaigns/strategies/provider aliases; nine agent calls finish before test; separate artifacts | Preserve the database-directory ledger; direct API callers must retain an explicit shared ledger. Legacy access requires explicit opt-in and is exploratory, not certification |
| P0-06 | evaluation/baselines.py and summaries | VERIFIED_COMPLETE | Conventional 100% B&H and explicit exposure-matched B&H | Independent 1000 -> 1200 vs 1040 golden calculation; costs/overlays metadata | Existing clients must adopt new benchmark key and allocation semantics |
| P0-07 | cli/research.py, main.py, pyproject.toml, README, Dockerfile | VERIFIED_COMPLETE for clean installed CLI | Canonical deterministic research path; explicit legacy demo | Clean non-editable package: all three console help commands pass; isolated installed canonical fixture smoke passes | Docker build unverified; committed code/lock reproducibility requires a reviewed coherent commit |
| Phase 2 | research_validation/walk_forward.py and robustness.py | VERIFIED_EXISTING components; certification NOT_ASSESSED | Rolling/expanding chronology and sensitivities already exist | Existing Sprint 7 suite | Review fold-specific selection, uncertainty methodology, search distribution and certification before promotion |
| Phase 3 | data/schema.py, provider.py, provenance.py, uv.lock | VERIFIED_EXISTING validation; provenance IMPLEMENTED | Data/environment/source fingerprint and dependency consistency | Existing OHLCV contract tests; saved fixture provenance | Working-tree lock for project 0.2.3 passes consistency and clean installation/tests; the committed lock is stale and blocks committed reproducibility. No verified PIT universe, delisting history, session calendar or provider redistribution rights |
| Phase 4 | structured_output.py and agents/telemetry.py | VERIFIED_EXISTING retry/timeout foundation; audit IMPLEMENTED | Record role/model/prompt hash/latency/validity/failures without raw text | E2E invocation audit artifacts | Token/cost fields remain null when metadata/rate basis unavailable; agent-ablation evidence pending |
| Phase 5 | paper_trading/runner.py | IN_PROGRESS | Frozen certified versions, recovery, risk enforcement and observation period | Local kernel/replay/checkpoint checks only | Registry, runtime kill switch, stale-data controls, drift, promotion policy and paper observation evidence remain pending |
| Phase 6 | research_dashboard/server.py | VERIFIED_EXISTING local boundary; production NOT_ASSESSED | Loopback read-only dashboard | Local server/write rejection tests | Public auth/RBAC/isolation/TLS, CI/security/SBOM, backups and production monitoring remain pending; no public deployment |
| Phase 7 | runner summaries, certification artifacts, dashboard | IMPLEMENTED for P0 statuses | Show distinct technical/statistical/risk/deployment evidence | Dashboard API and HTML integration checks; test/holdout CLI mapping regression | Rich strategy/evidence/risk/provenance pages remain later work |
| Phase 8 | LICENSE and README research boundary | VERIFIED_EXISTING research positioning; external review BLOCKED | Retain upstream attribution and exclude execution claims | Apache license preserved; explicitly named legacy paths | Human peer review, dependency notices, model/data terms and jurisdictional review not completed |

## Earlier verification record

Full suite after P0 implementation: **417 passed, 27 subtests passed**, two existing
UTC deprecation warnings. Command:
`python -m pytest -m 'not live' -q --basetemp=.test_tmp_release -p no:cacheprovider`.
Output: final_test_report.txt. Later focused dashboard/artifact review: **38 passed**,
output reviewed_test_report.txt. The final full suite includes the non-finite-value cases. Lint for touched implementation modules:
**All checks passed**, output lint_report.txt. git diff --check found no whitespace
errors. This is automated self-review, not independent approval.

Reproducible local E2E command: `python -m tests.prelaunch_fixture`.
Artifacts: results/prelaunch_fixture, with immutable synthetic inputs, recorded
structured responses, strategy/config, split fingerprints, separate metrics,
risk/viability reports, invocation audit, environment/source fingerprint and summary.
The example correctly returns VIABILITY_REJECTED and INSUFFICIENT_EVIDENCE.
It is post-change fixture evidence, not a live-provider or historical baseline run.

## Immediate next step

Use project-completion-plan.md and release-checklist.md for the expanded scope.
Collect a genuine controlled comparative study before claiming agent utility.
Recorded responses verify executor wiring but do not establish provider quality.
Review research methodology before
implementing any positive certification authority. Obtain real operational and
external approval evidence before enabling paper promotion. Resolve the coherent
source/lock commit through review, then run remote CI and container verification.
Preserve the historical baseline gap and persistent exposure ledger throughout.


