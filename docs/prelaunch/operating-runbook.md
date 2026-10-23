# Local operation, incidents and recovery

This is a research-only fork. All commands below remain local. No broker adapter,
paper promotion, statistical certification or automatic deployment is enabled.
Do not expose the unauthenticated local dashboard through a public proxy.

## Install and verify

From the repository root in PowerShell, use a separate environment:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\TradingAgents\.tools\project-completion\final-env'
.\.tools\uv\bin\uv.exe sync --locked --extra dev --extra audit --no-editable --python .venv\Scripts\python.exe --cache-dir .tools\uv-cache
.\.tools\project-completion\final-env\Scripts\python.exe -m scripts.verify_prelaunch --output NEW_VERIFICATION_DIRECTORY --online-audit
```

The verification destination must be new. The existing `.venv` remains untouched.
Online vulnerability queries require network access; their absence is unverified,
not a passing audit. CI uses the same locked extras and verification script.
Remote CI and the Docker image need their own actual successful runs.

## Offline data and research

Consult `trading-research COMMAND --help` for positional arguments and options.
`snapshot-dataset` accepts CSV and a DatasetIdentity JSON. Supply provider/version,
query, retrieval timestamp, adjustment policy and timezone. Optional sessions
must come from a named/versioned calendar. Missing or unexpected sessions fail.
`normalize-prices` requires explicit per-bar vendor factors and availability
timestamps; the adjustment trace must accompany the dataset.

`record-replay` records a validated strategy, full backtest configuration,
dataset ID, environment and deterministic output. `verify-replay` checks dataset,
bundle and output hashes and reports environment differences separately.
These are low-level diagnostics, not a shortcut around canonical research gates.

`fold-research` consumes development CSV, campaign/backtest configuration JSON
and recorded fold/attempt responses. It never calls paid providers. The offline
development fixture generates example inputs:

```powershell
.\.tools\p0-acceptance\clean-env\Scripts\python.exe -m tests.project_completion_fixture --output NEW_FIXTURE_DIRECTORY
```

Use `compare-architectures` only for recorded, matched validation treatments.
Inspect all failures, rejected candidates, unknown costs and search budgets.
Never use final holdout outcomes to select the architecture for another test.

## Frozen registry and simulated paper controls

`freeze-candidate` consumes strategy/config/risk/evidence manifest JSON. The
registry is content-addressed and immutable. Registration always returns
paper_eligible=false. SHA references alone are not scientific approval.

`simulate-paper` replays timezone-aware saved CSV bars against a frozen candidate,
using a SQLite session store. It is LOCAL_SIMULATION_ONLY. It does not submit
orders or establish real paper performance. Duplicate requests return the saved
result without adding fills. Revised, missing, stale or future bars suspend the
session. Cash/position reconciliation failures are runtime failures and preserve
the last accepted checkpoint.

`kill-simulation` permanently latches suspension for that session. It cancels
pending/new entries; existing positions exit at the next valid simulated open
under the shared kernel. A kill never manufactures a fill from stale data.
Re-enable by creating a separately reviewed session, not by editing SQLite.
`monitor-simulation` requires the immutable reference replay bundle. Its policy
cannot change after first use. Insufficient observations remain explicit;
threshold breaches latch suspension. Threshold defaults need human research
review. Historical timestamps never count as elapsed real paper observation.

## Authenticated evidence API

Use `provision-service-user TENANT USERNAME ROLE` locally; passwords are entered
privately. Roles are admin, researcher, reviewer and read_only. Ownership is
assigned once using `claim-evidence EXPERIMENT_ID TENANT`. No role has cross-tenant
resource access. Provisioning and ownership changes are audited. Back up the
access database; protect filesystem access to it and TLS private keys.

`serve-evidence-api CERTIFICATE PRIVATE_KEY` requires a valid TLS certificate/key
and binds only 127.0.0.1. This API is separate from the local dashboard and exposes
saved reports only. Login is POST /v1/session; use the returned opaque Bearer
session for GET /v1/experiments/ID. DELETE /v1/session revokes it. Sessions expire
after 15 minutes. Passwords/tokens/request bodies are never placed in access logs.
Login and authorized reads are rate-limited across process restarts. No paid
research jobs can be started through this API. Public production rollout still
requires independent security review, hosting/TLS operations, retention policy,
capacity controls and approval; none is inferred from local tests.

## Backup, restore and incidents

Quiesce writers before a cross-database backup. A source JSON maps plain filenames
to explicit SQLite/artifact paths. Run `backup-state SOURCES_JSON NEW_DIRECTORY`.
Each database uses SQLite's online backup and integrity check; file hashes are
recorded. Separate database snapshots do not form one global transaction.

Run `restore-state BACKUP_DIRECTORY NEW_DIRECTORY`. Restore validates all inputs
before creating its destination, rejects unsafe names and never replaces existing
work. Saved authenticated sessions are revoked. Check candidate, exposure-ledger,
registry, research and simulation consistency before restarting workers. Keep the
campaign holdout-exposure ledger with the research database and backups.
Encrypt and restrict off-host backups under the operator's approved policy.

For an incident: stop new research/simulation requests, latch affected sessions,
retain the incident ID and immutable inputs, verify ledgers and clocks/calendar,
restore into a new directory if needed, run offline reconciliation and document
the cause before any new reviewed session. Do not reset exposure history to make
a holdout look fresh. Do not start Docker or change host services automatically.
