# Remaining guide implementation

The user's request to complete the whole guide supersedes the earlier restriction
to the P0 acceptance package. Work proceeds in bounded, tested packages. Existing
work and the historical baseline gap remain preserved. No push, merge, deployment,
live-provider spending, or real-money execution is part of this implementation.

Default statistical certification and paper promotion remain disabled while
the evidence and operational gates are being implemented. Software delivery must
be distinguished from real paper observation periods, independent review,
licensed point-in-time data, and external legal/provider approvals.

| Package | Remaining implementation | Status | Verification |
|---|---|---|---|
| R | Fold-specific train/validation/test selection; budgets/distributions; uncertainty and sensitivities | LOCAL_DIAGNOSTICS_IMPLEMENTED | Chronology/future-mutation/golden statistics tested; all five architecture executors wired; genuine utility study/certification remain unestablished |
| D | Session/adjustment contracts, snapshots and deterministic replay | LOCAL_IMPLEMENTATION_VERIFIED | Supplied-calendar gaps and factor availability rejected; provider/PIT claims remain unverified |
| O | Invocation identity, fresh usage, explicit price basis, SDK retry cap and matched comparisons | LOCAL_IMPLEMENTATION_VERIFIED | Unknown prices/transport attempt counts remain null; genuine controlled study pending |
| P | Immutable registry, atomic/idempotent simulation, reconciliation, kill and drift | LOCAL_SIMULATION_IMPLEMENTED | Restart/duplicate/corruption/risk tests; real paper activation and elapsed observation blocked |
| S | Separate TLS evidence API, RBAC/tenant boundaries, persistent rate limits/audit, backups | READ_API_IMPLEMENTED | Real TLS and backup/restore tests passed; production workload/security review remains blocked |
| C | Locked CI/static/type/security/SBOM/container verification and disabled release gate | LOCAL_CHECKS_VERIFIED_REMOTE_UNVERIFIED | Vulnerable pins patched; remote CI and Docker build/run still unverified |
| U | Structured strategy/evidence/risk/provenance pages over persisted authoritative reports | LOCAL_IMPLEMENTATION_VERIFIED | Missing facts remain unavailable; local dashboard is read-only and loopback-only |
| L | Attribution, dependency inventory, methodology, release and operations runbooks | DOCUMENTED_EXTERNAL_REVIEW_REQUIRED | Human NOTICE/provider/data/model/jurisdiction approvals not fabricated |

Use the existing `.tools/p0-acceptance/clean-env` for isolated verification and
leave `.venv` unchanged. Record actual commands and results in completion-evidence.
Update this file and implementation-status.md at each package boundary. No task
is complete merely because an implementation file exists.

Checkpoint: 22 October 2026. The full guide is not certified complete. See
release-checklist.md for remaining software/study/operational gates and
project-completion-review.md for commands and actual verification results.
