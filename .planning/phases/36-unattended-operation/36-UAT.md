---
status: testing
phase: 36-unattended-operation
source: [36-VERIFICATION.md]
started: 2026-09-18T02:10:00Z
updated: 2026-09-18T02:10:00Z
---

## Current Test

number: 1
name: Re-run UAT Test 3 against a live healthchecks-compatible check configured only from the corrected runbook paragraph
expected: |
  Configure a fresh check using ONLY what the corrected "Heartbeat." paragraph in
  docs/runbooks/telescope_runs_calendar.rst names: expected interval (healthchecks.io
  `Period`) 15 min — or a Cron-type check with `*/15 * * * *` — and grace (`Grace`)
  about 20 min. Point FOMO_HEARTBEAT_URL at it, let one tick ping, then disable the
  crontab line. The check goes late about 15 minutes after the missed tick and alerts
  about 35 minutes after the last successful ping, with FOMO logging nothing and
  mailing nothing.
awaiting: user response

## Tests

### 1. Re-run UAT Test 3 against a live healthchecks-compatible check configured only from the corrected runbook paragraph
expected: Check goes late ~15 min after the missed tick and alerts ~35 min after the last successful ping (expected interval 15 min + grace ~20 min), with FOMO logging nothing and mailing nothing. This is the only proof that closes G-36-3 end-to-end; no automated gate can reach the external service.
result: [pending]

### 2. Decide on WR-22 before shipping (two `logger.debug()` sites format `str(exc)` on the unattended path)
expected: Either (a) fix `solsys_code/management/commands/backfill_lco_observations.py:349` and `solsys_code/unattended.py:191` to log `type(exc).__name__` only, plus an `assertLogs(level='DEBUG')` credential-hygiene case; or (b) record an explicit acceptance that the class-name-only discipline (plan 36-01 truth 10) holds only while `settings.LOGGING` keeps the root logger at `INFO`, documented where an operator raising the log level would see it. SC 4 is not breached as shipped; this is a judgment call on a latent exposure (36-REVIEW.md WR-22, which should be fixed together with WR-18).
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps

_Previous round (2026-09-17): 6 tests, 5 passed, 1 issue → G-36-3 (heartbeat guidance named only the grace time). Closed by gap-closure plan 36-06 (commits 12c51c6, f075a7f, 1f3bbac); the full gap record with root cause is in git history of this file and in `.planning/debug/heartbeat-runbook-period-gap.md`._
