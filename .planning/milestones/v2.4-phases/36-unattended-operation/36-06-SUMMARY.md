---
phase: 36-unattended-operation
plan: 06
subsystem: docs
tags: [heartbeat, healthchecks, runbook, unattended-operation, gap-closure]

# Dependency graph
requires:
  - phase: 36-unattended-operation
    provides: "36-01/36-03/36-05's runner, preflight and operator runbook, plus 36-UAT.md's Test 3 finding and the heartbeat-runbook-period-gap.md diagnosis of G-36-3"
provides:
  - "Corrected heartbeat alert-window guidance (Period + Grace, both named) in the runbook, the crontab template, the runner docstrings, and check_unattended's preflight output"
  - "A new runbook troubleshooting entry for 'the heartbeat never alerted although the schedule stopped'"
  - "A regression test (test_set_heartbeat_reminds_about_the_check_period) pinning the preflight reminder"
  - "A corrected 36-VERIFICATION.md that no longer scripts the wrong operator instruction for UAT Test 3"
affects: [operator-runbook, unattended-operation, heartbeat-dead-man-switch]

# Actuals (#2632)
actuals:
  tokens: 4549
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Name the general concept first, then the vendor-specific knob name in parentheses -- keeps operator-facing guidance service-agnostic while still giving healthchecks.io's actual field names"

key-files:
  created: []
  modified:
    - docs/runbooks/telescope_runs_calendar.rst
    - deploy/cron/fomo.crontab.example
    - solsys_code/management/commands/check_unattended.py
    - solsys_code/tests/test_check_unattended.py
    - solsys_code/unattended.py
    - .planning/phases/36-unattended-operation/36-VERIFICATION.md

key-decisions:
  - "Named the general concept (expected ping interval) before the vendor-specific knob name (healthchecks.io's `Period`) throughout, per the plan's service-agnostic prohibition -- the runbook still says 'healthchecks-compatible', not 'healthchecks.io-only'."
  - "Left `ping_heartbeat()` and the exit-code ping path in `unattended.py` completely untouched (per the plan's prohibition) -- only two consistency-only docstring sentences that described the backstop with the grace knob alone were reworded to say 'the heartbeat (D-12)' instead."
  - "Extended `check_heartbeat()`'s existing single `CheckResult` call rather than adding a second check -- it stays one soft, warning-level check that still prints set/unset only, never the URL value."

requirements-completed: [SCHED-08, SCHED-09]

coverage:
  - id: D1
    description: "The runbook's heartbeat guidance (the 'Heartbeat.' paragraph, the staleness-triage item, the lock-contention backstop sentence, and a new troubleshooting entry) names both the check's expected ping interval and its grace time, states the alert arithmetic, offers the Cron-type '*/15 * * * *' alternative, and names the 1-day default as the trap"
    requirement: SCHED-09
    verification:
      - kind: other
        ref: "pre-commit run sphinx-build --all-files (docs/runbooks/telescope_runs_calendar.rst, no new warning)"
        status: pass
      - kind: other
        ref: "grep checks: no 'grace period a little above'/'than the configured grace period'/\"The heartbeat's grace period\"; Period present on >=4 lines; 'expected interval', '35 min', 'heartbeat never alerted', '*/15 * * * *' all present"
        status: pass
    human_judgment: true
    rationale: "G-36-3's truth is only fully closed by re-running UAT Test 3 against a live healthchecks.io account, configuring the check from the corrected paragraph alone and confirming it goes late at ~15 min and alerts at ~35 min -- no automated gate can reach the external service (task 1's <human-check>)."
  - id: D2
    description: "The crontab template, both runner docstrings, and check_unattended's [ok] heartbeat line all describe the same two-knob configuration; a named test pins the preflight reminder and asserts no URL leakage"
    requirement: SCHED-08
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_check_unattended.py#TestWarningChecks.test_set_heartbeat_reminds_about_the_check_period"
        status: pass
      - kind: unit
        ref: "python manage.py test solsys_code.tests.test_check_unattended (30 tests)"
        status: pass
      - kind: other
        ref: "pre-commit run ruff --all-files / ruff-format --all-files / sphinx-build --all-files"
        status: pass
    human_judgment: false
  - id: D3
    description: "36-VERIFICATION.md's Test 3 frontmatter entry, its prose human-test script, its why_human line, and the plan 36-05 must-have evidence row all state the corrected two-knob configuration and the ~35-minute time-to-alert, with a traceability note pointing at the gap record; no verdict or VERIFIED marker changed"
    requirement: SCHED-09
    verification:
      - kind: other
        ref: "grep checks against .planning/phases/36-unattended-operation/36-VERIFICATION.md: no stale grace-only phrasing, 'G-36-3' present, '35 min' present, Period on >=3 lines"
        status: pass
    human_judgment: false

# Metrics
duration: 30min
completed: 2026-09-18
status: complete
---

# Phase 36 Plan 06: Heartbeat Guidance Correction (G-36-3) Summary

**Renamed the operator-facing heartbeat alert-window guidance everywhere it appears -- runbook, crontab template, runner docstrings, preflight output, and the verification record -- to name both the check's expected ping interval (healthchecks.io's `Period`) and its grace time (`Grace`), closing the gap where an operator following the runbook literally got a check that first alerted about a day late.**

## Performance

- **Duration:** ~30 min
- **Completed:** 2026-09-18T01:19:57Z
- **Tasks:** 3 (all `type="auto"`, no checkpoints)
- **Files modified:** 6

## Accomplishments
- Rewrote the runbook's "Heartbeat." paragraph, the "When nothing has appeared" staleness-triage item, and the lock-contention backstop sentence to describe the alert window as expected interval + grace, not grace alone; added a new "The heartbeat never alerted although the schedule stopped" troubleshooting entry naming the still-defaulted 1-day interval as the cause
- Propagated the same corrected guidance to `deploy/cron/fomo.crontab.example`'s backstop comment, two consistency-only docstring sentences in `solsys_code/unattended.py`, and `check_unattended`'s `[ok] heartbeat` preflight line -- which now reminds the operator to confirm the check's own expected ping interval is 15 min, still printing set/unset only
- Added `test_set_heartbeat_reminds_about_the_check_period` to `solsys_code/tests/test_check_unattended.py`, asserting the reminder and that the seeded fake heartbeat URL never appears in stdout
- Corrected `.planning/phases/36-unattended-operation/36-VERIFICATION.md`'s Test 3 frontmatter entry, its prose human-test script, and the plan 36-05 must-have evidence row so the verification record no longer scripts the instruction that produced G-36-3, with a traceability note recording the correction

## Task Commits

Each task was committed atomically:

1. **Task 1: Correct the runbook's heartbeat guidance** - `12c51c6` (docs)
2. **Task 2: Propagate the corrected guidance to the crontab template, the runner docstrings, and the preflight's heartbeat line** - `f075a7f` (fix)
3. **Task 3: Correct the verification record** - `1f3bbac` (docs)

**Plan metadata:** committed after this SUMMARY (see `git_commit_metadata` step).

## Files Created/Modified
- `docs/runbooks/telescope_runs_calendar.rst` - Heartbeat paragraph, triage item, lock backstop sentence, new troubleshooting entry, and step-4 preflight prose corrected
- `deploy/cron/fomo.crontab.example` - Backstop comment now names both knobs and points at the runbook
- `solsys_code/management/commands/check_unattended.py` - `check_heartbeat()`'s `[ok]` detail extended with the Period reminder
- `solsys_code/tests/test_check_unattended.py` - New `test_set_heartbeat_reminds_about_the_check_period` test
- `solsys_code/unattended.py` - Two docstring sentences reworded (consistency-only; `ping_heartbeat()` untouched)
- `.planning/phases/36-unattended-operation/36-VERIFICATION.md` - Test 3 script, evidence row, and traceability note corrected

## Decisions Made
- Named the general concept (expected ping interval) before the vendor-specific knob name (`Period`) throughout, keeping the guidance service-agnostic per the plan's prohibition.
- Left `ping_heartbeat()` and the exit-code ping path untouched, per the plan's explicit prohibition -- only two docstring sentences changed.
- Extended `check_heartbeat()`'s existing single `CheckResult` rather than adding a second check.

## Deviations from Plan

None - plan executed exactly as written. `pre-commit run ruff-format --all-files` auto-reformatted one string literal in `check_unattended.py` from a single- to a double-quoted string to avoid an escaped apostrophe (`check's` -> `check's`); this is the project's own formatter applying its standard rule, not a deviation from plan intent.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
All three tasks' automated `<verify>` commands pass: the runbook builds clean under Sphinx, `python manage.py test solsys_code.tests.test_check_unattended` is green (30/30), `pre-commit run ruff --all-files`/`ruff-format --all-files` pass, and the verification-record grep checks all pass. The plan's one `<human-check>` (re-running UAT Test 3 against a live healthchecks.io account with the corrected guidance) remains open for end-of-phase human verification -- this is the same human-only proof G-36-3 itself could only be found by, and no automated gate can reach the external service.

---
*Phase: 36-unattended-operation*
*Completed: 2026-09-18*
