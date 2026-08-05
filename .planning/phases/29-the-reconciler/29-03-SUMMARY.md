---
phase: 29-the-reconciler
plan: 03
subsystem: calendar-projection
tags: [django, management-command, calendar-events, campaign-run, reconciler, idempotent]

# Dependency graph
requires:
  - phase: 29-the-reconciler
    plan: 01
    provides: campaign_reconciler.py's reconcile_run()/ReconcileResult and the
      container/classical-nights stage branches
  - phase: 29-the-reconciler
    plan: 02
    provides: the completed adopt-and-rekey step inside reconcile_run() (three-tier
      per-night resolution order)
provides:
  - solsys_code/management/commands/reconcile_campaign_runs.py -- the one command RECON-01
    promises, looping reconcile_run() over every CampaignRun with --dry-run and D-05
    summary reporting
  - Command-level proof of RECON-01/RECON-06/RECON-07 in
    solsys_code/tests/test_reconcile_campaign_runs.py
affects: [29-04-the-staff-action-rewire]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Batch-loop-level try/except is the only catch point (D-06): no per-run
       transaction.atomic(), no try/except inside reconcile_run() itself"
    - "D-05 summary reporting via stderr-per-row (skip/fail/blocked) plus one final
       stdout f-string, mirroring import_campaign_csv's existing shape exactly"

key-files:
  created:
    - solsys_code/management/commands/reconcile_campaign_runs.py
    - solsys_code/tests/test_reconcile_campaign_runs.py
  modified: []

key-decisions:
  - "Command deliberately unfiltered: CampaignRun.objects.all() -- reconcile_run()'s own
     _skip_reason() guard (including the 'not approved' gate) is the single place that
     decides what does not project, so the command never grows a second, divergent
     copy of that rule"
  - "No --campaign/--run filters and no per-stage (0-4) breakdown line added, per the
     plan's explicit no-scope-creep instruction"

patterns-established: []

requirements-completed: [RECON-01, RECON-06, RECON-07]

# Metrics
duration: ~25min
completed: 2026-08-05
---

# Phase 29 Plan 03: The Batch Command Summary

**`reconcile_campaign_runs` -- the single idempotent sweep that loops `reconcile_run()` over every `CampaignRun`, retiring the backfill-command-per-gap pattern, with `--dry-run` parity and per-run failure isolation proven by 4 command-level tests.**

## Performance

- **Duration:** ~25 min
- **Tasks:** 2 completed
- **Files modified:** 2 (both created)

## Accomplishments

- Created `solsys_code/management/commands/reconcile_campaign_runs.py`: a `BaseCommand`
  subclass that iterates every `CampaignRun` unfiltered, calls `reconcile_run(run,
  dry_run=dry_run)` inside a single batch-loop-level `try/except` (D-06's only catch
  point), itemizes skips and failures by pk to `self.stderr`, and reports a final
  `Done[...]` summary line mirroring `import_campaign_csv`'s existing shape.
- Proved RECON-01 (idempotency), RECON-06 (`--dry-run` parity and per-run failure
  isolation) and RECON-07 (the measured real 8 QUEUE / 11 CLASSICAL / 0 SPACE split of 19
  runs) at the command level in `solsys_code/tests/test_reconcile_campaign_runs.py`
  (`TestIdempotency`, `TestDryRun`, `TestFailureIsolation`, `TestRealDataShapeScenario` --
  4 tests, all passing alongside the existing 25 in `test_campaign_reconciler.py`).
- Verified `python manage.py reconcile_campaign_runs --dry-run` runs clean against the dev
  DB (0 runs, all-zero counters) and `--help` shows the `--dry-run` flag.

## Task Commits

Each task was committed atomically:

1. **Task 1: Create the reconcile_campaign_runs management command** - `306ff15` (feat)
2. **Task 2: Command-level tests -- idempotency, dry-run, failure isolation and the 19-run scenario** - `09d23c6` (test)

## Files Created/Modified

- `solsys_code/management/commands/reconcile_campaign_runs.py` - new `Command` (BaseCommand
  subclass) with `--dry-run`, looping `reconcile_run()` and reporting the D-05 summary
- `solsys_code/tests/test_reconcile_campaign_runs.py` - new command-level test module:
  `TestIdempotency`, `TestDryRun`, `TestFailureIsolation`, `TestRealDataShapeScenario`

## Decisions Made

- Followed the plan's explicit no-scope-creep instructions: no per-stage (0-4) breakdown
  line, no `--campaign`/`--run` filters.
- `TestDryRun` proves preview/write agreement across four call_command invocations (dry,
  real, real again, dry again) rather than hand-computing expected counts, so the test
  itself derives the expected numbers from the command's own first-run output -- avoiding
  a second, potentially-divergent copy of the projection math inside the test.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Missing `src/fomo/_version.py` build artifact blocked all Django
management-command and test invocations in this worktree**
- **Found during:** Task 1 verification (first `python manage.py reconcile_campaign_runs
  --dry-run` invocation)
- **Issue:** Same environment-only gap plans 29-01/29-02 hit: `src/fomo/__init__.py`
  imports the gitignored, `setuptools_scm`-generated `._version` module, absent from this
  fresh worktree checkout.
- **Fix:** Copied the file from the main repo's working tree
  (`/home/tlister/git/fomo_devel/src/fomo/_version.py`) into this worktree -- a harmless,
  gitignored, environment-only file, never staged or committed.
- **Files modified:** none tracked by git (gitignored build artifact)
- **Verification:** `python manage.py reconcile_campaign_runs --dry-run` and `python
  manage.py test` now run in this worktree.
- **Committed in:** n/a (not a git-tracked file)

**2. [Rule 3 - Blocking] Dev-DB schema not migrated in this fresh worktree**
- **Found during:** Task 1 verification (same first invocation, after the `_version.py`
  fix)
- **Issue:** `src/fomo_db.sqlite3` existed but was 0 bytes (a fresh, unmigrated gitignored
  dev-DB file) -- `reconcile_campaign_runs --dry-run` failed with `no such table:
  solsys_code_campaignrun`.
- **Fix:** Ran `python manage.py migrate` once in this worktree -- an environment-only
  action against a gitignored local file, not a code or migration change.
- **Files modified:** none tracked by git (`src/fomo_db.sqlite3` is gitignored)
- **Verification:** `python manage.py reconcile_campaign_runs --dry-run` now exits 0 and
  prints `Done (dry run). runs: 0, ...` against the migrated (empty) dev DB.
- **Committed in:** n/a (not a git-tracked file)

---

**Total deviations:** 2 auto-fixed (both Rule 3 environment-only fixes, neither touching a
git-tracked file)
**Impact on plan:** Both were necessary to complete verification as specified; neither
changed scope or added functionality beyond what the plan already required.

## Issues Encountered

None beyond the auto-fixed items above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `reconcile_campaign_runs` is complete and independently proven at the command level;
  plan 29-04 (the four staff-action call-site rewires, running in this same wave on files
  `campaign_views.py`/`test_campaign_approval.py`/`test_admin.py`/
  `backfill_range_calendar_events.py`/`test_backfill_range_calendar_events.py`) does not
  depend on anything further from this plan.
- No blockers.

## Self-Check: PASSED

- Files verified present: `solsys_code/management/commands/reconcile_campaign_runs.py`,
  `solsys_code/tests/test_reconcile_campaign_runs.py`,
  `.planning/phases/29-the-reconciler/29-03-SUMMARY.md`.
- Commits verified present in `git log --oneline --all`: `306ff15`, `09d23c6`.
- `python manage.py test solsys_code.tests.test_reconcile_campaign_runs
  solsys_code.tests.test_campaign_reconciler` -- **29 tests, OK** (49.9s).
- `ruff check`/`ruff format --check` clean on both new files.
- `python manage.py reconcile_campaign_runs --dry-run` exits 0 against the dev DB, writes
  nothing.

## Self-Check: PASSED

- Files verified present on disk: `solsys_code/management/commands/reconcile_campaign_runs.py`,
  `solsys_code/tests/test_reconcile_campaign_runs.py`,
  `.planning/phases/29-the-reconciler/29-03-SUMMARY.md`.
- Commits verified present in `git log --oneline --all`: `306ff15`, `09d23c6`, `0f948af`
  (final metadata commit, includes this SUMMARY.md and REQUIREMENTS.md's RECON-07
  mark-complete).

---
*Phase: 29-the-reconciler*
*Completed: 2026-08-05*
