---
status: diagnosed
phase: 36-unattended-operation
source: [36-01-SUMMARY.md, 36-02-SUMMARY.md, 36-03-SUMMARY.md, 36-04-SUMMARY.md, 36-05-SUMMARY.md, 36-06-SUMMARY.md, 36-07-SUMMARY.md, 36-08-SUMMARY.md]
started: 2026-09-18T16:38:33Z
updated: 2026-09-18T18:07:54Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

[testing complete]

## Tests

### 1. Cold Start Smoke Test
expected: Stop any running dev server and make sure no unattended tick is in flight. Clear ephemeral state: the lock directory contents (FOMO_LOCK_DIR), the unattended state file and its fallback, any stale temp files. From scratch: `python manage.py migrate` completes with 0022_watchedproposal applied and no errors; `python manage.py check_unattended` runs every check and prints the cron line; `python manage.py run_unattended --dry-run` runs all four steps, prints START/END banners, exits 0 and logs no error.
result: issue
reported: "the no active WatchedProposal prints twice, once uncolored and once in red: ... [WARN] watched_proposals: no active WatchedProposal rows -- discovery will be a quiet no-op; add one in the admin (printed twice, second copy in red); everything else ran fine (migrate, check_unattended all [ok] incl. flock -E, FOMO_LOCK_DIR/LOG_FILE/STATE_DIR, smtp backend, 3 staff recipients, heartbeat, FOMO_BASE_URL, facility_credentials; run_unattended --dry-run fine)"
severity: minor

### 2. SC-5 read-through, heartbeat step: create and configure the check from the fresh-host steps alone
expected: A reader who has NOT been told the heartbeat values out of band works "Setting it up on a fresh host" top-down. Before the FOMO_HEARTBEAT_URL export step, a numbered step tells them what the heartbeat is, which service class to use (healthchecks.io hosted free tier or a self-hosted healthchecks instance), to create one check, to set BOTH knobs (Period 15 min or Cron `*/15 * * * *`; Grace ~20 min) with the one-line alert arithmetic (~35 min) and a pointer to "The two failure signals", and to copy the check's ping URL in the placeholder form `https://hc-ping.com/<uuid>` into FOMO_HEARTBEAT_URL. They reach the export step already knowing all of this without opening source. Run this BEFORE Test 4, which teaches these values.
coverage_id: 36-07 D1
result: pass

### 3. SC-5 read-through, step 2: the API key assignment can be followed verbatim
expected: Same top-down read of "Setting it up on a fresh host". Step 2 names a flat `LCO_API_KEY = '<your key>'` assignment in local_settings.py, says in one clause why a nested FACILITIES[...] form would raise NameError, says what settings.py folds it into and why one key covers both LCO and SOAR, and what omitting it leaves behind. Nowhere in the subsection is there a bracketed settings-dict subscript to copy. Following step 2 verbatim, Django imports cleanly (no NameError) and `check_unattended` reports the facility credentials present.
coverage_id: 36-08 D2
result: pass
note: "Operator reworded step 2 in docs/runbooks/telescope_runs_calendar.rst during the read-through to say where the LCO/SOAR API key comes from (uncommitted working-tree edit at the time of this pass); otherwise passes as written."

### 4. Live heartbeat dead-man re-run against a check configured from the runbook alone
expected: With the check created and configured only from the runbook's guidance (Period 15 min or Cron `*/15 * * * *`; Grace ~20 min), stop the cron schedule. The check goes late ~15 min after the missed tick and alerts ~35 min after the last successful ping, while FOMO itself logs nothing and mails nothing. External-service timer; run AFTER Tests 2 and 3.
coverage_id: 36-06 D1
result: pass

### 5. One end-to-end unattended tick (reconciler step): flock-guarded whole-run lock, per-step lock, heartbeat /start then /<exit-code>, D-11 mail-once-per-newly-failing-set with 24h reminder and one-shot recovery, credential-free logging
expected: as described
result: pass
source: automated
coverage_id: 36-01 D1

### 6. Committed crontab template: */15 flock-guarded schedule, log redirect, skip-visible lock-contention fallback, no secret or host path
expected: as described
result: pass
source: automated
coverage_id: 36-01 D2

### 7. Campaign submission notice rewired onto the shared notifications.notify_staff() helper -- single mail sender in solsys_code/, identical recipient rule and no-PII body preserved
expected: as described
result: pass
source: automated
coverage_id: 36-01 D3

### 8. WatchedProposal model, migration, and admin registration -- a staff user can maintain the watched list in the admin
expected: as described
result: pass
source: automated
coverage_id: 36-02 D1

### 9. sweep_proposal() extracted from Command.handle() as a behavior-preserving refactor
expected: as described
result: pass
source: automated
coverage_id: 36-02 D2

### 10. Bare invocation sweeps every active WatchedProposal row in proposal_code order, with per-proposal failure isolation and last_run_at/last_run_summary bookkeeping
expected: as described
result: pass
source: automated
coverage_id: 36-02 D3

### 11. step_status_refresh(): the FOMO-owned LCO/SOAR status refresh, replacing tomtoolkit's updatestatus command; discards the portal failure message before logging, never touches Gemini/ESO
expected: as described
result: pass
source: automated
coverage_id: 36-03 D1

### 12. step_project_sweep(): the bare projector sweep via project_queryset(), reproducing project_observation_calendar's logic
expected: as described
result: pass
source: automated
coverage_id: 36-03 D2

### 13. step_discovery(): sweeps every active WatchedProposal row through sweep_proposal(), isolating per-row failures (class name only), healthy zero-exit no-op for an empty list
expected: as described
result: pass
source: automated
coverage_id: 36-03 D3

### 14. STEPS registers all four steps in the fixed D-01 order; a failing step never stops the others; D-10 data-shape outcomes never trip the tick or the email
expected: as described
result: pass
source: automated
coverage_id: 36-03 D4

### 15. SCHED-10 credential hygiene across every forced failure path and the failure email's own shape
expected: as described
result: pass
source: automated
coverage_id: 36-03 D5

### 16. check_unattended reports every prerequisite in one run, exiting non-zero only on a hard failure; warnings never trip the exit code
expected: as described
result: pass
source: automated
coverage_id: 36-04 D1

### 17. cron_line() prints the exact cron line to install with real resolved paths, matching every element of deploy/cron/fomo.crontab.example, even when a hard check failed
expected: as described
result: pass
source: automated
coverage_id: 36-04 D2

### 18. --send-test-email sends one message through the configured backend to the staff-with-an-email recipients, reported as one more hard CheckResult
expected: as described
result: pass
source: automated
coverage_id: 36-04 D3

### 19. No check_unattended output surface ever contains a credential or setting value -- only names, paths, and set/unset status/counts
expected: as described
result: pass
source: automated
coverage_id: 36-04 D4

### 20. deploy/logrotate/fomo.example rotates the unattended log daily, keeps 14, documents why copytruncate is used
expected: as described
result: pass
source: automated
coverage_id: 36-04 D5

### 21. The runbook's 'How do I run everything unattended?' section takes an operator from a fresh host to a monitored schedule in one section
expected: as described
result: pass
source: automated
coverage_id: 36-05 D1

### 22. backfill_lco_observations's --proposal-optional/watched-list/failure-isolation contract documented in the runbook with a cheat-sheet row
expected: as described
result: pass
source: automated
coverage_id: 36-05 D2

### 23. backfill_lco_observations_demo.ipynb gains three executed cell pairs (bare sweep, bookkeeping readback, per-proposal failure isolation) and is in docs/notebooks.rst's toctree
expected: as described
result: pass
source: automated
coverage_id: 36-05 D3

### 24. CLAUDE.md's paired-docs map records the runner/check_unattended -> runbook-section pairing; installation.rst cross-references the new section
expected: as described
result: pass
source: automated
coverage_id: 36-05 D4

### 25. The crontab template, both runner docstrings, and check_unattended's [ok] heartbeat line all describe the same two-knob configuration; a named test pins the reminder and asserts no URL leakage
expected: as described
result: pass
source: automated
coverage_id: 36-06 D2

### 26. 36-VERIFICATION.md's heartbeat test script and evidence rows state the corrected two-knob configuration and the ~35-minute time-to-alert
expected: as described
result: pass
source: automated
coverage_id: 36-06 D3

### 27. deploy/cron/fomo.crontab.example's FOMO_HEARTBEAT_URL entry states where the ping URL comes from and points at 'Setting it up on a fresh host'
expected: as described
result: pass
source: automated
coverage_id: 36-07 D2

### 28. 36-VERIFICATION.md lists the SC-5 sufficiency read-through before the live heartbeat dead-man re-run, with the contamination rule recorded
expected: as described
result: pass
source: automated
coverage_id: 36-07 D3

### 29. src/fomo/settings.py's LCO_API_KEY fold also fills FACILITIES['SOAR']['api_key'], proven by a committed test executing the real fold
expected: as described
result: pass
source: automated
coverage_id: 36-08 D1

### 30. 36-VERIFICATION.md's SC-5 hold sites state the release condition naming plan 36-08
expected: as described
result: pass
source: automated
coverage_id: 36-08 D3

## Summary

total: 30
passed: 29
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

- gap_id: G-36-5        # NOT G-36-1: that id was consumed by round 2 (closed by plan 36-07) and would falsely reconcile as resolved
  truth: "check_unattended prints each check result exactly once; the watched_proposals [WARN] line appears a single time, colored consistently with the other lines"
  status: failed
  reason: "User reported: the no active WatchedProposal prints twice, once uncolored and once in red"
  severity: minor
  test: 1
  root_cause: "solsys_code/management/commands/check_unattended.py:614-617 -- Command.handle() builds one line per CheckResult and writes the identical string to BOTH self.stdout and (when status != 'ok') self.stderr. On a terminal or under 2>&1 the two sinks are one destination, so every non-ok line renders twice; Django's BaseCommand sets stderr.style_func = style.ERROR, which is why the second copy is red. Specified by 36-04-PLAN.md:182 (warnings and failures also to stderr) and present since the command's first commit fc2e2bc; NOT a regression from the iteration-5/6 review fixes (WR-35 only added a second line that can double). Tests are structurally blind: _run() passes two separate StringIO sinks and three assertions pin the dual-write as expected."
  artifacts:
    - path: "solsys_code/management/commands/check_unattended.py"
      issue: ":614-617 unconditional dual-write of one line to two sinks (the defect)"
    - path: "solsys_code/tests/test_check_unattended.py"
      issue: ":86-88 _run() uses two separate StringIO sinks so a merge never happens; :170-172, :222-224, :400-401 assert the WARN line in BOTH stdout and stderr, encoding the bug as the expectation"
    - path: "docs/runbooks/telescope_runs_calendar.rst"
      issue: "paired runbook section 'How do I run everything unattended?' -- any change to which stream warnings go to (and the 2>&1 crontab redirect interaction) must be reflected there"
  missing:
    - "Route each result line to exactly one stream (ok -> stdout; WARN/FAIL -> stderr only, matching every other command in solsys_code/management/commands/), or an equivalent single-emission design"
    - "Regression test passing the SAME StringIO as stdout= and stderr= (simulating a tty / 2>&1 merge) asserting the WARN line occurs exactly once, plus a hard-failure case with several non-ok lines"
    - "Revise the three presence assertions that currently pin the dual-write"
    - "Runbook: note which stream warnings/failures go to; keep the crontab-template 2>&1 interaction consistent"
  debug_session: ".planning/debug/check-unattended-duplicate-watched-proposals-warning.md"

_Round 3 (restarted 2026-09-18). Rounds 1-2 -- 6 + 5 tests, gaps G-36-3 (closed by plan 36-06), G-36-1 (closed by plan 36-07) and G-36-4 (closed by plan 36-08), the WR-22 acceptance and the CR-03 decision (closed by fix e2ed553) -- are preserved in this file's git history at 21dc3f8._
