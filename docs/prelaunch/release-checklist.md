# Release remains blocked

`python -m scripts.release_gate` emits RELEASE_BLOCKED and exits 2.
`--assert-disabled` exits 0 only to verify that promotion remains disabled; it is
not an approval. No push, merge, deploy, broker order or service activation was
performed. A local passing suite is not independent review or a production audit.

| Gate | Current evidence | Required before release |
|---|---|---|
| P0 correctness | Acceptance matrix, independent accounting goldens, full non-live suite | Reviewed coherent source and lock commit |
| Historical baseline | Original 376 passed/15 setup errors and later post-change runs | Preserve gap; never relabel a rerun as untouched |
| Research robustness | Fold-specific development campaigns, distributions, uncertainty and sensitivity diagnostics | Predeclared independent-trial methodology, external review and certification implementation/approval |
| Full architecture study | Five canonical executors; matched validation report comparison; development-only sealing | Collect genuine matched provider evidence and review utility/cost conclusions; recorded fixtures only prove plumbing |
| Data quality | Explicit adjustment/calendar contracts, hashes and offline replay | Licensed PIT universe, delistings, exchange sessions and vendor methodology evidence |
| Paper | Persistent local simulation, reconciliation, kill and drift controls | Certified candidate, approved activation, genuinely elapsed days/trades and reviewed incidents |
| Security | TLS/auth/RBAC/isolation tests, static scanner, dependency audit, SBOM and secret scan records | Independent review, operational monitoring, production capacity/tenant review and deployment approval |
| Reproducibility | Source and matching lock committed together in 7b7a2cb; clean installation and installed commands | Independent review and remote reproduction remain pending |
| Container and CI | Locked workflow and container recipe | Actual remote matrix and container build/run/digest; Docker unavailable locally |
| Backup | Verified local SQLite/artifact restore and session revocation | Operator retention, encryption/off-host storage and scheduled restore drill |
| Terms/legal | Attribution and installed license inventory | Human third-party NOTICE review, provider/model/data agreements and jurisdiction-specific review |

Do not mark any row complete merely because its implementation file or workflow
exists. Keep default statistical certification, paper promotion and real-money
execution disabled until the corresponding gates are independently satisfied.
