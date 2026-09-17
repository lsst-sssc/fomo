---
phase: 36-unattended-operation
plan: 01
subsystem: infra
tags: [cron, flock, fcntl, unattended, heartbeat, django-mail, campaign-reconciler]

# Dependency graph
requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: reconcile_campaign_runs/campaign_reconciler.reconcile_run() -- the step Plan 01 wraps
provides:
  - solsys_code.unattended -- the runner module (STEPS registry, run_tick(), command_lock(), ping_heartbeat(), load_state()/save_state()/decide_notification())
  - solsys_code.notifications -- the shared request-free mail helper (staff_recipients()/absolute_url()/notify_staff())
  - run_unattended management command (--dry-run/--step)
  - deploy/cron/fomo.crontab.example -- the committed host cron entry point
affects: [36-02, 36-03, 36-04, 36-05]

# Actuals (#2632)
actuals:
  tokens: 11418
  tasks: 3
  commits: 5

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "One shared solsys_code/notifications.py mail helper for both a request-bound view and a request-free management command"
    - "fcntl.flock-based per-command lock files under FOMO_LOCK_DIR, non-blocking, skip-and-log on contention"
    - "Healthchecks-style /start then /<exit-code> heartbeat bracketing a whole tick"
    - "JSON suppression-state file (not a model) for mail-once-per-newly-failing-set with a daily reminder and one-shot recovery notice"

key-files:
  created:
    - solsys_code/unattended.py
    - solsys_code/notifications.py
    - solsys_code/management/commands/run_unattended.py
    - solsys_code/tests/test_unattended.py
    - deploy/cron/fomo.crontab.example
  modified:
    - src/fomo/settings.py
    - solsys_code/campaign_views.py
    - solsys_code/tests/test_campaign_submission.py

key-decisions:
  - "notifications.notify_staff() catches send_mail()'s own exception and re-raises only when fail_silently=False, instead of delegating to send_mail()'s own fail_silently parameter -- keeps the outage-tolerance contract identical regardless of which layer fails or is mocked."
  - "unattended.py never calls django.urls.reverse() -- from a management-command-only process this would trigger a full URL-conf resolution (via calendar_urls.py -> solsys_code.views), reintroducing the ~1.6 GB SPICE-kernel import the module's own docstring rules out. The failure email's admin/calendar links are hardcoded paths instead."
  - "step_reconcile()'s per-run reconcile_run() exception is logged at DEBUG (never INFO/stderr), matching the project's root-INFO logging config -- keeps a FOMO-own exception's message (D-17's second bucket, message allowed) off the tick's normal output surface without adding a runtime redaction filter."

patterns-established:
  - "Runner step contract: a step function takes (dry_run: bool) -> StepResult(name, failed, summary); STEPS is an ordered tuple of (name, function) pairs and is the single source of step order."

requirements-completed: [SCHED-08, SCHED-09, SCHED-10]

coverage:
  - id: D1
    description: "One end-to-end unattended tick (reconciler step): flock-guarded whole-run lock, per-step lock, heartbeat /start then /<exit-code>, D-11 mail-once-per-newly-failing-set with 24h reminder and one-shot recovery, credential-free logging"
    requirement: SCHED-08
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_unattended.py -- 16 tests (TestRunUnattended, TestNotification, TestHeartbeat, TestLocking, TestCredentialHygiene)"
        status: pass
      - kind: manual_procedural
        ref: "python manage.py run_unattended --dry-run against the real developer database (45 CampaignRun rows, 0 failed) after applying pending migrations 0019-0021"
        status: pass
    human_judgment: false
  - id: D2
    description: "Committed crontab template: */15 flock-guarded schedule, log redirect, skip-visible lock-contention fallback, no secret or host path"
    requirement: SCHED-08
    verification:
      - kind: other
        ref: "shell verify commands in 36-01-PLAN.md Task 2 (substring/grep checks against deploy/cron/fomo.crontab.example) and git ls-files"
        status: pass
    human_judgment: false
  - id: D3
    description: "Campaign submission notice rewired onto the shared notifications.notify_staff() helper -- single mail sender in solsys_code/, identical recipient rule and no-PII body preserved"
    requirement: SCHED-10
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_submission.py -- 26 tests including 2 new TestSubmissionMailOutageResilience tests"
        status: pass
      - kind: other
        ref: "grep -rln 'send_mail(' solsys_code/ --include=*.py (excluding tests/) prints exactly solsys_code/notifications.py"
        status: pass
    human_judgment: false

duration: 24min
completed: 2026-09-17
status: complete
---

# Phase 36 Plan 01: Tracer Slice -- One End-to-End Unattended Tick Summary

**A `run_unattended` command that reconciles every CampaignRun under an `fcntl.flock` lock, brackets the tick with healthchecks-style heartbeat pings, and mails staff once per newly-failing tick via a mail helper now shared with the campaign submission notice.**

## Performance

- **Duration:** 24 min
- **Started:** 2026-09-17T14:57:50Z
- **Completed:** 2026-09-17T15:21:31Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments
- `solsys_code/unattended.py`: `run_tick()`/`STEPS`/`command_lock()`/`ping_heartbeat()`/`step_reconcile()`/`load_state()`/`save_state()`/`decide_notification()` -- the full tracer-slice runner architecture, proven end to end on one real step (`reconcile`) before Plan 03 prepends the other three
- `solsys_code/notifications.py`: `staff_recipients()`/`absolute_url()`/`notify_staff()` -- one request-free mail helper both the unattended runner and the campaign submission notice now share
- `solsys_code/management/commands/run_unattended.py`: thin `--dry-run`/`--step` wrapper, `sys.exit()` on a failing tick
- `deploy/cron/fomo.crontab.example`: the committed, secret-free, placeholder-path crontab fragment an operator installs (D-01/D-04/D-18)
- `solsys_code/campaign_views.py`: `CampaignRunSubmissionView._notify_staff()` now delegates to the shared helper -- there is exactly one mail sender in `solsys_code/`

## Task Commits

Each task was committed atomically (Tasks 1 and 3 carried `tdd="true"` -- RED/GREEN split into a `test(...)` commit and a `feat(...)` commit each):

1. **Task 1 RED: add failing tests for unattended runner reconcile step** - `4a77ceb` (test)
2. **Task 1 GREEN: one end-to-end unattended tick -- reconciler step only** - `e1571ac` (feat)
3. **Task 2: add the committed crontab template** - `fc53a4a` (docs)
4. **Task 3 RED: add mail-outage-resilience tests for the submission notice** - `8eae142` (test)
5. **Task 3 GREEN: rewire campaign submission notice onto the shared mail helper** - `dd53789` (feat)

_Note: Task 2 (`type="auto"`, no `tdd`) is a single commit; Tasks 1 and 3 (`tdd="true"`) each split into two._

## Files Created/Modified
- `solsys_code/unattended.py` - runner module: step registry, locking, heartbeat, notification decision
- `solsys_code/notifications.py` - shared request-free staff-mail helper
- `solsys_code/management/commands/run_unattended.py` - `run_unattended` command
- `solsys_code/tests/test_unattended.py` - 16 tests covering the full `<behavior>` list
- `deploy/cron/fomo.crontab.example` - committed cron fragment
- `src/fomo/settings.py` - `FOMO_BASE_URL`/`FOMO_HEARTBEAT_URL`/`FOMO_LOCK_DIR`/`FOMO_STATE_DIR`/`FOMO_LOG_FILE`
- `solsys_code/campaign_views.py` - `_notify_staff()` rewired onto the shared helper
- `solsys_code/tests/test_campaign_submission.py` - 2 new mail-outage-resilience tests

## Decisions Made
- `notifications.notify_staff()` implements its own `fail_silently` try/except around `send_mail()` rather than passing the flag through to `send_mail()`'s own parameter -- discovered necessary while writing Task 3's mail-outage test (patching `send_mail()` directly bypasses Django's internal per-backend `fail_silently` handling, so a naive pass-through wouldn't actually protect the submission view against a mocked or unusual outage).
- `unattended.py` never calls `django.urls.reverse()` -- see Deviations below.
- Per-run `reconcile_run()` exceptions inside `step_reconcile()` are logged at `logger.debug()`, matching the project's root-`INFO` logging config, so a FOMO-own exception's message (D-17's second bucket, where a message is allowed) never reaches the tick's normal stdout/stderr/log surface, without needing a runtime redaction filter.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Applied pending Django migrations to unblock the plan's own dry-run verification**
- **Found during:** Task 1, running the plan's `<verification>` command `python manage.py run_unattended --dry-run` against the real developer database
- **Issue:** `step_reconcile()` failed on 36 of 45 real `CampaignRun` rows with `no such column: solsys_code_calendareventmeta.minted_sub_night_window`. Confirmed pre-existing and unrelated to this plan by reproducing the identical failure with the existing `reconcile_campaign_runs --dry-run` command; `showmigrations solsys_code` showed migrations `0019`-`0021` (from the already-shipped Phase 35) unapplied on this checkout's `fomo_db.sqlite3`.
- **Fix:** Ran `python manage.py migrate solsys_code` (three additive, schema-only migrations -- no data loss).
- **Files modified:** none (database schema only)
- **Verification:** `run_unattended --dry-run` then reported `runs: 45, failed: 0`, `exit=0`.
- **Commit:** not applicable (database state, not a source change)

**2. [Rule 1 - Bug] Reworded a false-positive comment against Task 3's single-writer grep check**
- **Found during:** Task 3, running `grep -rln 'send_mail(' solsys_code/ --include=*.py`
- **Issue:** `unattended.py`'s comment `# send_mail() is a network-ish call, D-17` matched the literal substring `send_mail(`, so the grep printed `unattended.py` and `notifications.py` instead of exactly `notifications.py`.
- **Fix:** Reworded the comment to `# mail sending is a network-ish call, D-17` -- no behavior change.
- **Files modified:** `solsys_code/unattended.py`
- **Verification:** `grep -rln 'send_mail(' solsys_code/ --include=*.py | grep -v tests/` now prints exactly `solsys_code/notifications.py`.
- **Committed in:** `dd53789` (Task 3 GREEN commit)

---

**Total deviations:** 2 auto-fixed (1 blocking/environment, 1 bug/false-positive-check)
**Impact on plan:** Neither changed any production behavior described in the plan; both were required for the plan's own verification commands to run as written.

## Issues Encountered

Task 1's and Task 3's `test(...)` commits are not classic RED commits: `workflow.tdd_mode` is `false` for this project (per `36-01-PLAN.md`'s own project note, matching Phase 34 plan `34-07`'s precedent), so `gsd_run check tdd-red-evidence` was not run. Task 1's RED was verified manually and is genuine -- `solsys_code.unattended` did not exist, so collecting `test_unattended.py` raised `ImportError` (confirmed by temporarily moving the three new implementation files aside and re-running the suite). Task 3's two new tests (`TestSubmissionMailOutageResilience`) are refactor-preserving characterization tests, not classic RED -- both scenarios (a mail outage never breaking a submission, an empty recipient list sending nothing) already held against the pre-rewire `_notify_staff()` implementation, since it already called `send_mail(..., fail_silently=True)` and already returned early on an empty recipient list. The genuinely new, RED-able assertions for Task 3 are the plan's own source-grep `<verify>` commands (single `send_mail(` call site, no `build_absolute_uri` left behind), which were confirmed failing (2 matches instead of 1; `build_absolute_uri` present) before the rewire and passing after.

## User Setup Required

None - no external service configuration required. (The runner's `FOMO_HEARTBEAT_URL`/`FOMO_LOCK_DIR`/`FOMO_STATE_DIR`/`FOMO_LOG_FILE` environment variables are documented in the crontab template's comments and default to sensible values with `FOMO_HEARTBEAT_URL` unset -- an operator sets up the real host in a later plan's runbook section.)

## Next Phase Readiness

Ready for Plan 02 (the `WatchedProposal` model) and Plan 03 (the other three unattended steps -- `step_status_refresh()`, `step_project_sweep()`, `step_discovery()` -- prepended to `STEPS` in their D-01 order). No blockers: `STEPS` is a plain ordered tuple with one entry today, ready for Plan 03 to prepend three more with no architectural change, and `notifications.notify_staff()` is the one place both callers (the runner and the submission notice) will keep sharing.

---
*Phase: 36-unattended-operation*
*Completed: 2026-09-17*

## Self-Check: PASSED

All key files found on disk (`solsys_code/unattended.py`, `solsys_code/notifications.py`,
`solsys_code/management/commands/run_unattended.py`, `solsys_code/tests/test_unattended.py`,
`deploy/cron/fomo.crontab.example`); all 5 task commit hashes (`4a77ceb`, `e1571ac`, `fc53a4a`,
`8eae142`, `dd53789`) found in `git log`. Full test module reruns (`test_unattended.py`,
`test_campaign_submission.py`) and both quality gates (`pre-commit run ruff --all-files`,
`pre-commit run ruff-format --all-files`) pass at time of writing.
