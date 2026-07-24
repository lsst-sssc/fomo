---
phase: quick-260723-r5g
plan: 01
subsystem: infra
tags: [django, tom_observations, lco, calendar, sync_lco_observation_calendar]

# Dependency graph
requires:
  - phase: quick-260722-ux0
    provides: "backfill_lco_observation_records fix for the same perpetual-[QUEUED] class of bug (via update_observation_status refresh at record-creation time)"
provides:
  - "sync_lco_observation_calendar._title_for() no longer stamps a permanent [QUEUED] prefix on a COMPLETED record whose scheduled_start was never resolved"
affects: [sync_lco_observation_calendar, calendar_utils, telescope-runs-calendar]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Successful-terminal state set derived as facility.get_terminal_observing_states() minus facility.get_failed_observing_states() (LCOFacility/SOARFacility expose no get_successful_observing_states() method)"

key-files:
  created: []
  modified:
    - solsys_code/management/commands/sync_lco_observation_calendar.py
    - solsys_code/tests/test_sync_lco_observation_calendar.py
    - docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb

key-decisions:
  - "The plan assumed a facility.get_successful_observing_states() method existed; it does not (LCOFacility/SOARFacility only expose get_terminal_observing_states() and get_failed_observing_states()). Fixed by deriving the successful set as terminal - failed (Rule 3 auto-fix, blocking issue: plan's API assumption was wrong)."

patterns-established:
  - "Guard [QUEUED]/banner-stage prefixing against any status already in the successful-terminal set, computed dynamically from the facility rather than hardcoded, matching the existing _FAILURE_PREFIX_BY_STATUS convention in the same file."

requirements-completed: []

coverage:
  - id: D1
    description: "COMPLETED ObservationRecord with scheduled_start=None gets a clean CalendarEvent title (no [QUEUED] prefix)"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_sync_lco_observation_calendar.py#test_d06_completed_with_unresolved_scheduled_start_gets_clean_title"
        status: pass
    human_judgment: false
  - id: D2
    description: "Non-terminal record with scheduled_start=None still gets the [QUEUED] prefix (existing behavior preserved, no regression)"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_sync_lco_observation_calendar.py#test_sync_02_d03_unscheduled_uses_parameters_times_and_queued_title"
        status: pass
    human_judgment: false
  - id: D3
    description: "Paired demo notebook shows a COMPLETED + unresolved-scheduled_start record producing a clean title, with real executed output"
    verification:
      - kind: other
        ref: "jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb (new D-06 cell, executed output confirms 'title : 2m0 2M0-SCICAM-MUSCAT', no [QUEUED])"
        status: pass
    human_judgment: false

duration: 20min
completed: 2026-07-24
status: complete
---

# Quick Task 260723-r5g: Fix sync_lco_observation_calendar COMPLETED-unresolved [QUEUED] title bug Summary

**Fixed `_title_for()` in `sync_lco_observation_calendar.py` so a COMPLETED `ObservationRecord` with an unresolved `scheduled_start` (None) falls through to a clean title instead of being permanently stuck reading `[QUEUED]`.**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-07-24T02:41:57Z
- **Tasks:** 2 completed
- **Files modified:** 3

## Accomplishments
- `_title_for()` now skips the `[QUEUED]` banner-stage prefix when `record.status` is already in the successful-terminal state set, matching the same D-06 reasoning already applied in the failure-prefix branch for COMPLETED.
- Added a regression test (`test_d06_completed_with_unresolved_scheduled_start_gets_clean_title`) proving the fix, alongside the pre-existing `[QUEUED]` test which still passes unchanged (no regression on the non-terminal case).
- Updated the paired demo notebook (`sync_lco_observation_calendar_demo.ipynb`) with a new fixture + `call_command` cell demonstrating the fix, with real executed output confirming the clean title and correct time window (from `parameters['start']`/`['end']`).

## Task Commits

Each task was committed atomically:

1. **Task 1: Guard [QUEUED] prefix against successful-terminal status + regression test** - `1595619` (fix)
2. **Task 2: Add + regenerate demo notebook cell for the COMPLETED-unresolved clean title** - `0917927` (docs)

**Plan metadata:** `bd0683b` (docs: pre-dispatch plan)

## Files Created/Modified
- `solsys_code/management/commands/sync_lco_observation_calendar.py` - `_title_for()` guard: skip `[QUEUED]` when status is already successful-terminal (derived from `get_terminal_observing_states() - get_failed_observing_states()`)
- `solsys_code/tests/test_sync_lco_observation_calendar.py` - new regression test for the COMPLETED-unresolved case
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` - new D-06 demo cell (fixture + `call_command` + assertion), teardown cleanup, and requirements-summary table row, regenerated with executed output

## Decisions Made
- Derived the "successful-terminal" status set dynamically as `set(facility.get_terminal_observing_states()) - set(facility.get_failed_observing_states())` rather than hardcoding `{'COMPLETED'}`, keeping the fix generic across LCOFacility/SOARFacility and consistent with the file's existing `_failure_prefix()` pattern of deriving from the facility rather than a hand-typed constant wherever possible.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Plan's `facility.get_successful_observing_states()` method does not exist**
- **Found during:** Task 1 (running the test suite after the initial edit)
- **Issue:** The plan's context asserted `facility.get_successful_observing_states()` returns `['COMPLETED']`. Neither `LCOFacility` nor `SOARFacility` (from `tom_observations`) expose this method — only `get_terminal_observing_states()` and `get_failed_observing_states()` exist, confirmed via direct instantiation. Using the nonexistent method raised `AttributeError` and broke 20 previously-passing tests (any test invoking the sync command).
- **Fix:** Derived the successful-terminal set as `set(facility.get_terminal_observing_states()) - set(facility.get_failed_observing_states())`, which evaluates to `{'COMPLETED'}` for both facilities (verified interactively) — functionally identical to what the plan intended, just computed from methods that actually exist.
- **Files modified:** solsys_code/management/commands/sync_lco_observation_calendar.py
- **Verification:** `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` — all 44 tests pass (was 24/44 failing before the fix).
- **Committed in:** 1595619 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking — incorrect API assumption in plan)
**Impact on plan:** Necessary correction to make the fix actually work; no scope creep, no behavior change from what the plan intended.

## Issues Encountered
None beyond the deviation above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- No further action needed; this closes the same class of perpetual-`[QUEUED]`/stale-title bug already fixed for `backfill_lco_observation_records` in quick task `260722-ux0`, now also covered for the live `sync_lco_observation_calendar` sync path.
- `ruff check .` / `ruff format --check .` clean for both modified Python files. Pre-existing, unrelated ruff findings in other files (`sync_gemini_observation_calendar_demo.ipynb` D103, several files needing reformatting) were confirmed present before this task's changes and are out of scope (SCOPE BOUNDARY).

---
*Phase: quick-260723-r5g*
*Completed: 2026-07-24*

## Self-Check: PASSED

All created/modified files and commit hashes verified present.
