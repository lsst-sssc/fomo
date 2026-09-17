---
phase: 36-unattended-operation
plan: 03
subsystem: infra
tags: [cron, lco, soar, observation-status, projector-sweep, discovery, credential-hygiene]

# Dependency graph
requires:
  - phase: 36-unattended-operation
    provides: "36-01's unattended.py runner architecture (StepResult/STEPS/command_lock()/run_tick()) and 36-02's sweep_proposal()/watched_rows()"
provides:
  - "solsys_code.unattended.step_status_refresh(dry_run) -- the FOMO-owned LCO/SOAR status refresh (D-03), replacing tomtoolkit's updatestatus"
  - "solsys_code.unattended.step_project_sweep(dry_run) -- the bare observation-projector sweep, called directly (no manage.py subprocess)"
  - "solsys_code.unattended.step_discovery(dry_run) -- the watched-proposal discovery sweep at the runner level"
  - "solsys_code.unattended.STEPS in its final D-01 order: status_refresh -> project_sweep -> discovery -> reconcile"
affects: [36-04, 36-05]

# Actuals (#2632)
actuals:
  tokens: 8515
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A step function never routes through call_command()/django.core.management.call_command -- it imports and calls the underlying module function directly, so the runner gets a structured StepResult instead of captured text (D-02). Verified by two independent gates: a source-count probe over unattended.py, and a negative test patching call_command at its own definition site."
    - "A portal-facing except clause discards the caught exception's message and re-derives only the exception class name for logging (D-17's first bucket / SCHED-10); a FOMO-own exception's message may reach a DEBUG-level log line, which sits below the project's root INFO level and therefore never reaches the tick's normal output surface, stdout, stderr, or the failure email."

key-files:
  created: []
  modified:
    - solsys_code/unattended.py
    - solsys_code/tests/test_unattended.py

key-decisions:
  - "_refresh_one_facility(facility) takes an already-constructed LCOFacility()/SOARFacility() instance rather than a class, so step_status_refresh() itself is the one place that writes the literal LCOFacility()/SOARFacility() constructor calls the plan's own source-count verify probe checks for -- keeping 'one fresh instance per facility, never shared' (Phase 34 D-10) visible at the call site rather than hidden inside a generic helper."
  - "A facility-level exception from update_all_observation_statuses() itself (not a per-record failure in its returned list) is counted as one failed record for that facility, with the caught exception's class name recorded -- this keeps the step's failed flag and summary consistent whether the portal fails per-record or all at once, without adding a second counting scheme."
  - "step_project_sweep()'s hook(record, facility) closure logs resolve_observed_site()'s returned message via logger.warning() rather than a stdout/stderr sink -- a step function has no self.stdout/self.stderr the way a management command does, and the message is already a fixed, generic, observation_id-only string (D-08), so a plain log line carries no additional risk."
  - "step_discovery()'s summary names every failing proposal code in parentheses (matching D-14's requirement that the failure email can quote them), rather than only a failed count -- mirroring backfill_lco_observations.Command.handle()'s own CommandError message shape for the watched path, so the runner-level summary and the hand-run command's own error text stay in the same voice."

requirements-completed: [SCHED-08, SCHED-09, SCHED-10, DISCOVER-01]

coverage:
  - id: D1
    description: "step_status_refresh(): the FOMO-owned LCO/SOAR status refresh, replacing tomtoolkit's updatestatus command. Discards the portal-returned failure message before any logging, re-derives the exception class name via a bounded per-id re-check, and never touches Gemini/ESO."
    requirement: SCHED-08
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_unattended.py -- TestStatusRefreshStep (7 tests)"
        status: pass
      - kind: other
        ref: "source-count probe: SOARFacility(/LCOFacility( each appear once in unattended.py, call_command appears zero times"
        status: pass
    human_judgment: false
  - id: D2
    description: "step_project_sweep(): the bare projector sweep called directly via project_queryset(), reproducing project_observation_calendar.Command.handle()'s logic (unfiltered queryset, the observed-site hook omitted under --dry-run, an unprojectable row counted as the step's failure signal)."
    requirement: SCHED-08
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_unattended.py -- TestProjectSweepStep (4 tests)"
        status: pass
      - kind: other
        ref: "python manage.py run_unattended --dry-run against the real developer database (159 unchanged, 0 unprojectable)"
        status: pass
    human_judgment: false
  - id: D3
    description: "step_discovery(): sweeps every active WatchedProposal row through sweep_proposal(), isolating a per-row portal/data failure (class name only) and reporting a healthy zero-exit no-op for an empty watched list."
    requirement: DISCOVER-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_unattended.py -- TestDiscoveryStep (5 tests)"
        status: pass
      - kind: other
        ref: "python manage.py run_unattended --step discovery against the real developer database (0 watched proposals, exit 0, no ping/mail)"
        status: pass
    human_judgment: false
  - id: D4
    description: "STEPS registers all four steps in the fixed D-01 order (status_refresh -> project_sweep -> discovery -> reconcile); a failing step never stops the others; D-10 data-shape outcomes (unchanged/skipped/detach_declined/remint_declined) never trip the tick or the email."
    requirement: SCHED-08
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_unattended.py -- TestRunUnattended.test_all_four_steps_run_in_order, test_expected_data_shape_outcomes_are_not_failures"
        status: pass
      - kind: other
        ref: "python manage.py run_unattended --help lists all four step names as --step choices"
        status: pass
    human_judgment: false
  - id: D5
    description: "SCHED-10 credential hygiene across every forced failure path (status-refresh portal error, project-sweep site-lookup error, discovery portal error, reconcile FOMO-own exception, mail-send failure, heartbeat-ping failure) and the failure email's own shape (log path + step names, no secret, no Traceback), with no logging.Filter subclass added."
    requirement: SCHED-10
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_unattended.py -- TestCredentialHygiene (8 tests, expanded from 1)"
        status: pass
    human_judgment: false

# Metrics
duration: 25min
completed: 2026-09-17
status: complete
---

# Phase 36 Plan 3: Full D-01 Tick -- Status Refresh, Projector Sweep, Discovery Summary

**`run_unattended` now runs all four steps -- LCO/SOAR status refresh, projector sweep, watched-proposal discovery, and the reconciler -- in one process with no `manage.py` subprocess anywhere, and a 41-test suite (8 of them the SCHED-10 credential-hygiene regression) proves no credential value reaches a log line, the console, or the failure email on any forced failure path.**

## Performance

- **Duration:** 25 min
- **Started:** 2026-09-17T15:46:42Z
- **Completed:** 2026-09-17T16:11:55Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- `step_status_refresh(dry_run)` and `_refresh_one_facility()`: the FOMO-owned LCO/SOAR status refresh (D-03), constructing a fresh `LCOFacility()`/`SOARFacility()` instance per tick, discarding `update_all_observation_statuses()`'s message half before any logging, and re-deriving the exception class name via a bounded per-id `update_observation_status()` re-check
- `step_project_sweep(dry_run)`: the bare observation-projector sweep called directly through `project_queryset()`, reproducing `project_observation_calendar.Command.handle()`'s logic (unfiltered queryset, the one-time observed-site hook omitted under `--dry-run`, an `unprojectable` row as the failure signal) with no `call_command()` anywhere
- `step_discovery(dry_run)`: sweeps every active `WatchedProposal` row through `sweep_proposal()` at the runner level, isolating a per-row portal/data failure to its own row and naming the failing proposal code(s) in the summary for D-14's email
- `STEPS` now runs the full D-01 sequence: `status_refresh -> project_sweep -> discovery -> reconcile`, each behind its own `command_lock()` and its own try/except
- `TestCredentialHygiene` expanded from 1 to 8 tests, covering every one of the six forced failure paths (status refresh, project sweep, discovery, reconcile, mail send, heartbeat ping) plus the failure email's own body shape, with a shared fake-literal `setUp()` and assertion helper

## Task Commits

Each task was committed atomically:

1. **Task 1: Step 1 -- the FOMO-owned LCO/SOAR status refresh** - `49e572c` (feat)
2. **Task 2: Steps 2 and 3 -- the projector sweep and watched-list discovery** - `b8ec5d0` (feat)
3. **Task 3: SCHED-10 regression suite across every failure path** - `50baea9` (test)

**Plan metadata:** (this commit)

_Note: all three tasks carried `tdd="true"`; `workflow.tdd_mode` is `false` for this project (see Issues Encountered), so each task's tests and implementation landed in a single commit rather than a split RED/GREEN pair, matching the precedent set by plans 34-07, 36-01, and 36-02._

## Files Created/Modified
- `solsys_code/unattended.py` - `step_status_refresh()`, `_refresh_one_facility()`, `step_project_sweep()`, `step_discovery()`; `STEPS` extended to its final four-entry D-01 order
- `solsys_code/tests/test_unattended.py` - `TestStatusRefreshStep` (7 tests), `TestProjectSweepStep` (4 tests), `TestDiscoveryStep` (5 tests), two new `TestRunUnattended` cases, `TestCredentialHygiene` expanded from 1 to 8 tests -- 41 tests total in the module (up from 16 at the start of this plan)

## Decisions Made
See `key-decisions` in frontmatter: `_refresh_one_facility()`'s instance-not-class signature (keeping the literal `LCOFacility()`/`SOARFacility()` constructor calls at the step's own call site for the verify probe), the facility-level-exception counting rule, the hook's plain `logger.warning()` sink (no `self.stdout`/`self.stderr` available to a step function), and the discovery summary naming failing proposal codes in parentheses to match D-14.

## Deviations from Plan

None - plan executed exactly as written. All three tasks' `<behavior>` test lists, `<action>` implementation steps, `<verify>` commands, and `<acceptance_criteria>` were satisfied without needing a Rule 1-4 deviation.

## Issues Encountered

`workflow.tdd_mode` is `false` for this project (per this plan's own project note and the precedent set by plans 34-07, 36-01, 36-02), so `gsd_run check tdd-red-evidence` was not run and each task's tests/implementation were authored together rather than as a strict RED-then-GREEN pair with separate commits. Each task's acceptance-criteria verification loop (running the full test module, the source-count probes, and the real-database `--dry-run`/`--step` invocations) was executed and confirmed passing before moving to the next task, which is the equivalent gate this project's TDD precedent substitutes.

The plan's own `<verify>` command `python manage.py run_unattended --dry-run` (and `--step discovery`, and `--help`) requires `FOMO_LOCK_DIR`/`FOMO_STATE_DIR` to point at a directory the invoking user can create -- the project default (`/var/lock/fomo`) is root-owned in this dev sandbox. This is a pre-existing environment constraint (also present for plans 36-01/36-02's own manual verification), not something this plan changed; the verify commands were run with `FOMO_LOCK_DIR=/tmp/fomo_lock_verify FOMO_STATE_DIR=/tmp/fomo_lock_verify` set for that one invocation only, with no change to any committed file or default.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for Plan 04 and Plan 05. `STEPS` now carries its final four-entry D-01 order with no further steps to add; Plan 04/05 can build on `run_tick()`'s existing structure (locking, heartbeat, notification, credential hygiene) with no architectural change. No blockers.

---
*Phase: 36-unattended-operation*
*Completed: 2026-09-17*

## Self-Check: PASSED

All key files found on disk (`solsys_code/unattended.py` contains `def step_status_refresh(`,
`def step_project_sweep(`, `def step_discovery(`; `solsys_code/tests/test_unattended.py`
contains `class TestStatusRefreshStep`, `class TestProjectSweepStep`, `class TestDiscoveryStep`).
All 3 task commit hashes (`49e572c`, `b8ec5d0`, `50baea9`) found in `git log`. Full test module
rerun (`python manage.py test solsys_code.tests.test_unattended` -- 41 tests) and the full app
suite gate from this plan's own `<verification>` block both pass at time of writing; both quality
gates (`pre-commit run ruff --all-files`, `pre-commit run ruff-format --all-files`) pass.
