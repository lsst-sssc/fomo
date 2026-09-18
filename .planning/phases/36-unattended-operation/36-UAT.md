---
status: diagnosed
phase: 36-unattended-operation
source: [36-VERIFICATION.md]
started: 2026-09-18T02:10:00Z
updated: 2026-09-18T02:50:00Z
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
  root_cause: "Two documentation-structure halves plus a process cause. (1) Placement: the fresh-host procedure was derived from check_unattended's list of host-inspectable prerequisites (36-CONTEXT.md SC-5 truth :24-27, D-05 :77-85 'watch the heartbeat'), in which the heartbeat is only an environment variable; 36-05-PLAN.md :132-138 / :146-149 accordingly split the heartbeat into an env-var export (setup) and the whole check-side configuration (failure-signals subsection), D-12 named no subsection, and the executor rendered that split faithfully (c9c6fcf) -- so the document's first mention of the heartbeat is the bare variable at :1461, step 3 asks for a URL that only exists after a check is created, no step creates one, and the procedure has zero cross-references to the paragraph at :1541-1568. (2) Content: which service to use and how to obtain the check's ping URL (the https://hc-ping.com/<uuid> form, hosted free tier vs self-hosted) exist in no operator-facing artifact at all -- only in .planning/research/STACK.md :22, :47-48. Process: the prior debug session (heartbeat-runbook-period-gap.md :67-71) observed 'neither step tells the operator to create or configure the check at all' but filed it as corroboration of the one-knob defect, not as its own defect, so 36-06 inherited an artifact list that never named step 3 and rewrote the paragraph in place. Every gate was a whole-file token-presence probe ('FOMO_HEARTBEAT_URL' in src; grep -c Period) that the bare line satisfies, and round-1 UAT Test 6 (SC 5 sufficiency) was administered after Test 3 had taught the operator the Period knob out of band."
  artifacts:
    - path: "docs/runbooks/telescope_runs_calendar.rst"
      issue: ":1448-1514 'Setting it up on a fresh host' has no create-and-configure-the-check step; step 3 (:1461-1463) is the first mention of the heartbeat and carries only export hygiene; no cross-reference to :1541-1568; step 4's sentence (:1484-1487) is the first hint a remote check exists and points at nothing"
    - path: "docs/runbooks/telescope_runs_calendar.rst"
      issue: ":1541-1568 'Heartbeat.' paragraph is complete on Period/Grace/arithmetic/Cron/1-day trap but its only 'where' is 'any healthchecks-compatible endpoint (hosted or self-hosted)' -- no service named, no ping-URL provenance"
    - path: "deploy/cron/fomo.crontab.example"
      issue: ":13-15 lists FOMO_HEARTBEAT_URL by name only; :32-35 defers to the runbook section 'for the full setup', which currently resolves to no step that sets the check up (paired artifact -- keep in agreement)"
    - path: "solsys_code/management/commands/check_unattended.py"
      issue: ":235-252 check_heartbeat() [ok] reminder is paired verbatim with runbook step-4 sentence (:1484-1487) and pinned by test_set_heartbeat_reminds_about_the_check_period -- must stay consistent if the step-4 hint is trimmed to a pointer"
    - path: ".planning/phases/36-unattended-operation/36-VERIFICATION.md"
      issue: ":112 SC 5 marked VERIFIED on structural grounds plus a contaminated Test 6 pass; the SC-5 read-through must be scripted before, not after, the tests that teach the missing knowledge"
  missing:
    - "A new fresh-host setup step BEFORE the export that, in operator order: says what the heartbeat is in one sentence (dead-man's switch for a tick that never ran or hung; optional -- unset disables the layer); says where to get one (a healthchecks-compatible service: healthchecks.io's hosted free tier or a self-hosted healthchecks instance); says to create one check for this schedule and set BOTH knobs -- expected ping interval 15 min (healthchecks.io: Period) or Cron type */15 * * * *, and grace ~20 min (Grace) -- with the one-line arithmetic (alert = last ping + interval + grace, about 35 min) and a pointer to 'The two failure signals' for the reasoning; then says to copy that check's own ping URL (placeholder form https://hc-ping.com/<uuid>; never a real value, D-15) into FOMO_HEARTBEAT_URL"
    - "Split current step 3 into one step per variable (FOMO_HEARTBEAT_URL export hygiene; FOMO_BASE_URL, already self-sufficient -- use it as the template)"
    - "One canonical home for the arithmetic's WHY (:1541-1568); the new step gives values and points there rather than re-copying the paragraph (the 36-06 defect class was one phrasing re-seeded across five artifacts)"
    - "Paired artifacts kept in agreement: deploy/cron/fomo.crontab.example :13-15 and :32-35; check_heartbeat() reminder <-> runbook step-4 sentence; possibly trim step 4's inline hint to a pointer once the new step exists"
    - "A gate with teeth: assert the setup subsection slice (between 'Setting it up on a fresh host' and 'Adding a proposal to watch') itself contains Period, Grace and */15 * * * *, instead of whole-file token presence; and re-script the SC-5 sufficiency test in 36-VERIFICATION.md so the read-through happens before any test that teaches the knowledge"
    - "Optional, planner's call: same-class siblings -- step 2 never names the FACILITIES['LCO']['api_key'] / FACILITIES['SOAR']['api_key'] nesting (settings.py :235-247, documented nowhere); steps 1 and 7 need sudo, unstated (twice filed as optional in round-1 UAT)"
  debug_session: ".planning/debug/heartbeat-setup-step-context-gap.md"

_Previous round (2026-09-17): 6 tests, 5 passed, 1 issue → G-36-3 (heartbeat guidance named only the grace time). Closed by gap-closure plan 36-06 (commits 12c51c6, f075a7f, 1f3bbac); the full gap record with root cause is in git history of this file and in `.planning/debug/heartbeat-runbook-period-gap.md`._
