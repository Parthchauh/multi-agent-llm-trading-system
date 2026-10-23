# Prelaunch baseline

Baseline commit: `79f2411b56ee24793abb2a42214674f8d80eeac8` on `main`.
Existing uncommitted work was present in README, CLI, experiment contracts,
platform persistence, dashboard, E2E runner, and associated tests. It was retained.
No branch, tag, commit, push, merge, or deployment has been created.
No AGENTS.md was found in the repository file inventory.

Environment: Windows, Python 3.13.9, existing project `.venv`, existing `uv.lock`.
The first unchanged-suite run returned 376 passed and 15 setup errors because
pytest could not use its default temporary directory. Complete output is in
`baseline_test_report.txt`. A workspace temporary directory resolved the setup
errors: 391 passed, 27 subtests passed. That rerun occurred after the first
accounting/kernel edits; it is regression evidence, not an untouched baseline.
No pre-change end-to-end artifact snapshot was captured. The baseline commit
and original source establish the before behavior; later fixtures are labeled
post-change and are not represented as historical baseline evidence.

Confirmed original accounting example: buying five shares at a slipped fill
of 101 with reference price 100, commission 2, and slippage attribution 5
charged 512 instead of 507. The original sale cash also subtracted slippage
again, and realized P&L omitted the entry commission.

Canonical workflow: `python -m cli.research experiment` delegates to
`scripts/run_research_e2e.py`. Historical `main.py` and `cli.main` are legacy.
The dashboard is already local and read-only. Dataset validation, deterministic
compiler, causal warm-up, risk reason codes, strategy lineage, rolling/expanding
walk-forward planning, and sensitivity tools already exist.

Docker build verification is unavailable because the installed Docker client
cannot connect to a running daemon. No live provider call or spending was used
as evidence. No independent peer review has occurred. Ponytail is unavailable.
