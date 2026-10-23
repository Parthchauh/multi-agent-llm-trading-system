# Remaining implementation with a zero budget

23 October 2026. User constraint: no funding, no paid model calls.

## Delivered software

- `register-study`: immutable local plan records with exact configurations,
  actual per-symbol dataset SHA256 identities, seeds, chronology, prompt hashes,
  primary metric, complete treatment set and declared invocation/cost budgets.
- `assess-study`: checks every planned treatment and symbol/repetition, run start
  after registration, unchanged configuration and actual data identity. All
  failures remain in the comparison. Unknown costs stay unknown. Budget checks
  are retrospective, not a billing limit or permission to call a provider.
- Architecture comparison additionally controls model sampling settings,
  robustness requests and schema version.
- TLS API request metrics record only numeric status, latency and timestamps.
  `service-health` reports request/error/latency/denial/rate-limit observations and
  persists a local alert inbox. `acknowledge-alert` records local acknowledgement.
  Restart does not lose alerts. No traffic means insufficient observations, not
  healthy uptime. Provider spend/queue metrics remain unavailable for this read API.
- `encrypt-backup` and `restore-encrypted-backup`: AES-256-GCM authenticated
  envelopes with scrypt-derived keys, private passphrase input, tamper rejection,
  safe new-directory restore and session revocation. Plaintext backup inputs are
  retained. Off-host transfer and secure disk erasure are not claimed.
- Explicit SQLite closure fixes a Windows lock found by encrypted restore tests.
- CI now runs on pushes to main, retains verification artifacts, and scans the
  built image for HIGH/CRITICAL vulnerabilities before reporting completion.

## Zero-cost usage

Fresh offline locked installation: `.tools/project-completion/remaining-env`,
Python 3.13.9, 161 distributions. Full suite: 505 passed, 27 subtests, one existing
UTC deprecation warning in 102.08 seconds. Scoped lint/type/Bandit/secret checks
passed; original `.venv` distribution inventory is unchanged. Exact verification
command arrays and results are retained in completion-evidence/remaining-*.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\TradingAgents\.tools\project-completion\remaining-env'
.\.tools\uv\bin\uv.exe sync --offline --locked --extra dev --extra audit --no-editable --python .venv\Scripts\python.exe --cache-dir .tools/uv-cache
.\.tools\project-completion\remaining-env\Scripts\python.exe -m scripts.verify_prelaunch --output .tools/project-completion/remaining-verification
.\.tools\project-completion\remaining-env\Scripts\python.exe -I docs/prelaunch/acceptance-tools/installed_completion_check.py
```

```powershell
python -m tests.remaining_work_fixture --output NEW_FIXTURE_DIRECTORY
trading-research register-study NEW_FIXTURE_DIRECTORY/study-plan.json --database results/studies.sqlite
trading-research service-health --database results/service-metrics.sqlite
trading-research encrypt-backup BACKUP_DIRECTORY OUTPUT.enc
trading-research restore-encrypted-backup OUTPUT.enc NEW_RESTORE_DIRECTORY
```

The fixture uses synthetic data and recorded responses, sets cost budget to zero,
and writes `OFFLINE_RECORDED_FIXTURE_PASSED`. It exercises software behavior and
encrypted recovery; it does not demonstrate LLM investment utility. Its ephemeral
passphrase is deliberately discarded after the test; do not use that sample as
a real recovery backup. Keep real backup passphrases separately with the operator.

The study registry is a local integrity/control tool. It does not independently
attest registration time, verify claimed licenses/reviews, or authorize promotion.
An operator with write access to its database remains trusted. External reviewers
should receive a plan hash before results are collected.

## Evidence still requiring external inputs or elapsed time

No Ollama command or cached Ollama model was found. A Google API credential is
present, but a credential alone does not establish free-tier billing. No provider
request was made with it. A genuine zero-cost model study requires a verified
free-tier setup or locally available model plus approved research data. Do not
silently substitute recorded outputs for that study.

Docker's daemon remains unavailable; no host service was started. Remote CI
results must be recorded after the workflow actually runs. Independent scientific,
security, licensing and jurisdiction review cannot be self-certified. Real elapsed
paper observation cannot be replaced by replay. Off-host storage, external alert
delivery, production capacity and certificate operations still require operator
configuration and drills. Positive certification, paper promotion and broker
execution remain disabled. The historical baseline gap remains unchanged.
