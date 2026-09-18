---
status: complete
phase: 36-unattended-operation
source: [36-VERIFICATION.md]
started: 2026-09-18T02:10:00Z
updated: 2026-09-18T02:32:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Re-run UAT Test 3 against a live healthchecks-compatible check configured only from the corrected runbook paragraph
expected: Check goes late ~15 min after the missed tick and alerts ~35 min after the last successful ping (expected interval 15 min + grace ~20 min), with FOMO logging nothing and mailing nothing. This is the only proof that closes G-36-3 end-to-end; no automated gate can reach the external service.
result: issue
reported: "the \"HeartBeat\" paragraph just starts \"Export `FOMO_HEARTBEAT_URL` in the environment\". There needs to be info or a step before this that says what it is, where to set it up, which website to go to, what to set period and grace time to"
severity: major

### 2. Decide on WR-22 before shipping (two `logger.debug()` sites format `str(exc)` on the unattended path)
expected: Either (a) fix `solsys_code/management/commands/backfill_lco_observations.py:349` and `solsys_code/unattended.py:191` to log `type(exc).__name__` only, plus an `assertLogs(level='DEBUG')` credential-hygiene case; or (b) record an explicit acceptance that the class-name-only discipline (plan 36-01 truth 10) holds only while `settings.LOGGING` keeps the root logger at `INFO`, documented where an operator raising the log level would see it. SC 4 is not breached as shipped; this is a judgment call on a latent exposure (36-REVIEW.md WR-22, which should be fixed together with WR-18).
result: pass
decision: "record the acceptance" -- option (b). Recorded in src/fomo/settings.py (comment above LOGGING), 36-REVIEW.md WR-22 disposition, and 36-VERIFICATION.md § Acknowledged Gaps.

## Summary

total: 2
passed: 1
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

- gap_id: G-36-1
  truth: "An operator working the runbook's 'Setting it up on a fresh host' steps top-down learns, at the point the heartbeat first appears (step 3, 'Export FOMO_HEARTBEAT_URL'), what the heartbeat is, where to create the check (a healthchecks-compatible service such as healthchecks.io, hosted or self-hosted), which URL to copy into FOMO_HEARTBEAT_URL, and what to set the check's expected ping interval (Period: 15 min, or Cron type */15 * * * *) and grace time (Grace: ~20 min) to -- without having to discover the 'Heartbeat.' paragraph ~80 lines later under 'The two failure signals'."
  status: failed
  reason: "User reported: the \"HeartBeat\" paragraph just starts \"Export `FOMO_HEARTBEAT_URL` in the environment\". There needs to be info or a step before this that says what it is, where to set it up, which website to go to, what to set period and grace time to"
  severity: major
  test: 1
  artifacts:
    - path: "docs/runbooks/telescope_runs_calendar.rst"
      issue: "setup step 3 (~1461-1463) exports FOMO_HEARTBEAT_URL with no explanation of the heartbeat, no create-the-check step, and no Period/Grace values; the corrected two-knob guidance from 36-06 lives only in the 'Heartbeat.' paragraph (~1541-1568) under 'The two failure signals', which the setup sequence never points at"
  missing: []        # Filled by diagnosis
  debug_session: ""  # Filled by diagnosis

_Previous round (2026-09-17): 6 tests, 5 passed, 1 issue → G-36-3 (heartbeat guidance named only the grace time). Closed by gap-closure plan 36-06 (commits 12c51c6, f075a7f, 1f3bbac); the full gap record with root cause is in git history of this file and in `.planning/debug/heartbeat-runbook-period-gap.md`._
