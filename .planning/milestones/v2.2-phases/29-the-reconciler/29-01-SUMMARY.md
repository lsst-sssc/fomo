---
phase: 29-the-reconciler
plan: 01
subsystem: calendar-projection
tags: [django, calendar-events, campaign-run, reconciler, no-churn]

# Dependency graph
requires:
  - phase: 27-the-canonical-run-record
    provides: CalendarEventMeta.run FK, CampaignRun.source/telescope_class
  - phase: 26-canonical-record-spike
    provides: the locked RUN:{pk}[:date] two-key-family scheme and queue-run container verdict
provides:
  - solsys_code/campaign_reconciler.py with reconcile_run() and its stage branches
  - Two new public no-churn helpers in calendar_utils.py (update_calendar_event_key_and_fields,
    preview_calendar_event_action) for the D-02 re-key step and D-05/RECON-06 dry-run reporting
  - Unit tests proving RECON-01 (unit level)/RECON-02 (queue half)/RECON-03/RECON-05/RECON-06
affects: [29-02-the-adopt-and-rekey-step, 29-03-the-batch-command, 29-04-the-staff-action-rewire]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pure-logic reconciler module mirroring campaign_gap.py/campaign_utils.py: never imports
       solsys_code.views or ephem_utils"
    - "Ownership-scoped write: _may_write() is the first condition checked in every write path"
    - "Two coexisting CalendarEvent key families: bare RUN:{pk} container vs. date-bearing
       RUN:{pk}:{date} per-night key"
    - "Field-authority split: the container branch owns every field on both create/update; the
       per-night branch only refreshes title/description/target_list after creation"

key-files:
  created:
    - solsys_code/campaign_reconciler.py
    - solsys_code/tests/test_campaign_reconciler.py
  modified:
    - solsys_code/calendar_utils.py
    - solsys_code/tests/test_calendar_utils.py

key-decisions:
  - "update_calendar_event_key_and_fields()/preview_calendar_event_action() added as new public
     calendar_utils.py functions rather than exposing _update_or_unchanged() cross-module,
     per Open Question 2's recommendation"
  - "_reconcile_classical_nights() does not include the D-02 adopt-and-rekey step -- deliberately
     deferred to plan 29-02 per the plan's wave split; this plan's classical branch always mints"
  - "reconcile_run()'s dispatch order is telescope_class non-blank -> container, then satellite
     site -> container, then source in QUEUE_SOURCES -> container, else classical per-night"

patterns-established:
  - "RUN_STATUS_CALENDAR_PREFIX is now public in campaign_reconciler.py (moved from
     campaign_views._RUN_STATUS_CALENDAR_PREFIX); plan 29-04 deletes the campaign_views copy"

requirements-completed: [RECON-01, RECON-02, RECON-03, RECON-05, RECON-06]

# Metrics
duration: ~35min
completed: 2026-08-05
---

# Phase 29 Plan 01: The Reconciler Foundation Summary

**`campaign_reconciler.py`'s `reconcile_run()` -- one idempotent per-run function projecting queue/class-wide/satellite runs to a single bare `RUN:{pk}` container and classical runs to per-night `RUN:{pk}:{date}` events, with a RECON-05 ownership guard and RECON-06 dry-run support.**

## Performance

- **Duration:** ~35 min
- **Tasks:** 3 completed
- **Files modified:** 4 (2 created, 2 modified)

## Accomplishments

- Added `update_calendar_event_key_and_fields()` and `preview_calendar_event_action()` to
  `calendar_utils.py` -- the two new public no-churn helpers the reconciler's D-02 re-key step
  (plan 29-02) and dry-run reporting (RECON-06) need, without any cross-module import of the
  module-private `_update_or_unchanged()`.
- Created `solsys_code/campaign_reconciler.py`: the D-03 shared per-run function
  `reconcile_run()`, its two stage branches (`_reconcile_container`/`_reconcile_classical_nights`),
  the two `RUN:` key-family builders (`run_container_url`/`run_night_url`), the ownership-scoping
  query (`owned_events`), the title/description builders, the RECON-05 ownership guard
  (`_may_write`), and the `CalendarEventMeta.run` writer (`_link_event_to_run`). Verified importable
  with no SPICE kernel download.
- Unit-tested the container branch, all four skip reasons, ownership scoping (including the
  trailing-colon guard and a different-run-owned-event fixture), and container idempotency
  (including dry-run parity) in `test_campaign_reconciler.py`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add the two public no-churn helpers to calendar_utils.py** - `0367b46` (feat)
2. **Task 2: Create campaign_reconciler.py -- the D-03 shared per-run function and its stage branches** - `3ef66f2` (feat)
3. **Task 3: Unit-test the container branch, skip reasons, ownership scoping and idempotency** - `80180c8` (test)

## Files Created/Modified

- `solsys_code/calendar_utils.py` - added `update_calendar_event_key_and_fields()` and `preview_calendar_event_action()`
- `solsys_code/tests/test_calendar_utils.py` - added `TestUpdateCalendarEventKeyAndFields`/`TestPreviewCalendarEventAction`
- `solsys_code/campaign_reconciler.py` - new pure-logic reconciler module (`reconcile_run()` and everything it needs, minus the D-02 adopt step)
- `solsys_code/tests/test_campaign_reconciler.py` - new unit test module: `TestSkipReasons`, `TestQueueStage1`, `TestClassWideStage2`, `TestSatelliteContainer`, `TestOwnershipScoping`, `TestContainerIdempotency`

## Decisions Made

- Followed the plan's explicit deferral: `_reconcile_classical_nights()` in this plan always mints
  a new per-night event (no adopt-and-rekey check against an existing `load_telescope_runs`-created
  row) -- plan 29-02 adds `_adopted_event_for_night()` and wires it in ahead of the mint.
- `night` parameter of `run_night_url()` left without an explicit type annotation, matching the
  plan's prescribed import list (which does not include `datetime.date`) -- callers pass a
  `datetime.date` (from `CampaignRun.window_start`/`window_end` or the classical-branch loop).
- Excluded `zoneinfo.ZoneInfo` from this module's imports since nothing in this plan's scope uses
  it yet (the D-02 adopt step that needs it is plan 29-02's addition) -- importing it unused would
  fail `ruff`'s F401 check.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Rephrased a module docstring sentence that literally contained the
forbidden-import module names, breaking the plan's own grep-based acceptance check**
- **Found during:** Task 2 verification
- **Issue:** `campaign_reconciler.py`'s module docstring explained (as instructed) that the module
  must never import `solsys_code.views`/`solsys_code.ephem_utils`, but writing those exact literal
  strings in prose (not a `#` comment) made `grep -v '^#' campaign_reconciler.py | grep -c
  'solsys_code.views\|ephem_utils'` return 1 instead of the required 0 -- the docstring's own
  explanation was tripping the check meant to catch a real import.
- **Fix:** Reworded the sentence to say "the views module" / "the heavy SPICE-loading ephemeris
  module" instead of the literal dotted paths, preserving the same meaning.
- **Files modified:** `solsys_code/campaign_reconciler.py`
- **Verification:** `grep -v '^#' solsys_code/campaign_reconciler.py | grep -c
  'solsys_code.views\|ephem_utils'` now returns 0; `ruff check`/`ruff format --check` still clean.
- **Committed in:** `3ef66f2` (Task 2 commit)

**2. [Rule 3 - Blocking] Two test fixtures collided on `CampaignRun`'s natural-key unique
constraint**
- **Found during:** Task 3 verification (`test_campaign_reconciler.py` first test run)
- **Issue:** `TestOwnershipScoping`'s "different-run-owned" and "trailing-colon guard" tests each
  created two `CampaignRun`s via the shared `_make_run()` default fixture with identical
  `(campaign, telescope_instrument, window_start, window_end)`, tripping
  `CampaignRun`'s existing natural-key `UniqueConstraint` (`sqlite3.IntegrityError`).
- **Fix:** Gave the second run in each of those two tests a distinct `telescope_instrument`
  (`'Other Telescope/Instrument'`).
- **Files modified:** `solsys_code/tests/test_campaign_reconciler.py`
- **Verification:** `python manage.py test solsys_code.tests.test_campaign_reconciler` passes (14/14).
- **Committed in:** `80180c8` (Task 3 commit)

**3. [Rule 3 - Blocking] Missing `src/fomo/_version.py` build artifact blocked all Django test
runs in this worktree**
- **Found during:** Task 1 verification (first `python manage.py test` invocation)
- **Issue:** `src/fomo/__init__.py` imports `._version`, a `setuptools_scm`-generated file that is
  gitignored and therefore absent from this fresh worktree checkout (it only exists in the main
  repo's working tree, generated once by the editable install there).
- **Fix:** Copied the main repo's `src/fomo/_version.py` into this worktree (a harmless, gitignored,
  environment-only file -- never staged or committed).
- **Files modified:** none tracked by git (gitignored build artifact)
- **Verification:** `python manage.py test` now runs in this worktree.
- **Committed in:** n/a (not a git-tracked file)

---

**Total deviations:** 3 auto-fixed (2 Rule 3 blocking-issue fixes to source, 1 Rule 3 environment-only fix)
**Impact on plan:** All three were necessary to complete verification as specified; none changed scope or added functionality beyond what the plan already required.

## Issues Encountered

None beyond the auto-fixed items above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `reconcile_run()` is ready for plan 29-02 to extend with the D-02 adopt-and-rekey step
  (`_adopted_event_for_night()`) ahead of `_reconcile_classical_nights()`'s mint.
- Plan 29-03 (the batch command) and plan 29-04 (the four staff-action call-site rewires) can
  import `reconcile_run()`/`ReconcileResult` directly; no further scaffolding is needed from this
  plan.
- `RUN_STATUS_CALENDAR_PREFIX` is now public in `campaign_reconciler.py`; plan 29-04 should delete
  the now-superseded `campaign_views._RUN_STATUS_CALENDAR_PREFIX` copy when it rewires the staff
  actions, per D-01.
- No blockers.

## Self-Check: PASSED

- Files verified present: `solsys_code/calendar_utils.py`, `solsys_code/campaign_reconciler.py`,
  `solsys_code/tests/test_calendar_utils.py`, `solsys_code/tests/test_campaign_reconciler.py`,
  `.planning/phases/29-the-reconciler/29-01-SUMMARY.md`.
- Commits verified present in `git log --oneline --all`: `0367b46`, `3ef66f2`, `80180c8`.
- Full regression run: `python manage.py test` across all `solsys_code` test modules except
  `test_views` and `test_ephem_utils` (excluded per project memory -- native ASSIST segfault,
  unrelated to this plan) -- **800 tests, OK** (304s).

---
*Phase: 29-the-reconciler*
*Completed: 2026-08-05*
