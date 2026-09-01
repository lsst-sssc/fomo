---
phase: quick-260722-ux0
plan: 01
subsystem: management-commands
tags: [django, lco, tom-toolkit, observation-record, backfill]

# Dependency graph
requires:
  - phase: quick-260722-tkt
    provides: backfill_lco_observation_records command with --create-missing-targets flag
provides:
  - Post-create live status refresh in backfill_lco_observation_records so newly backfilled ObservationRecords get real scheduled_start/scheduled_end instead of staying perpetually [QUEUED] on the calendar
affects: [sync_lco_observation_calendar, backfill_lco_observation_records]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Best-effort post-create side-effect call: create the row first, then attempt a live API refresh in a broad try/except that logs-and-counts on failure but never rolls back the already-persisted row."

key-files:
  created: []
  modified:
    - solsys_code/management/commands/backfill_lco_observation_records.py
    - solsys_code/tests/test_backfill_lco_observation_records.py

key-decisions:
  - "facility.update_observation_status(observation_id) is called exactly once, immediately after ObservationRecord.objects.create(), strictly inside the non-dry-run branch, never in the skipped_existing branch."
  - "A refresh failure is caught with a broad `except Exception`, logged to stderr in the same style as other skip messages, counted in a new status_sync_failed counter, and never rolls back the created record or aborts the run."
  - "All 16 pre-existing tests now run under a class-wide setUp() patch of LCOFacility.update_observation_status (patched at the class level, since the command constructs its own LCOFacility() instance) -- without this, every non-dry-run test would make a real live HTTP call to observe.lco.global using the real LCO_APIKEY configured in settings, which is exactly what happened once during local verification (a real request returned status=WINDOW_EXPIRED for observation_id=10, failing the assertion that expected COMPLETED)."

patterns-established: []

requirements-completed: []

coverage:
  - id: D1
    description: "Newly created (non-dry-run, non-skipped) ObservationRecords trigger exactly one facility.update_observation_status(observation_id) call, refreshing scheduled_start/scheduled_end so sync_lco_observation_calendar._title_for() no longer shows [QUEUED] for terminal records."
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observation_records.py#test_created_record_triggers_status_refresh"
        status: pass
    human_judgment: false
  - id: D2
    description: "--dry-run makes zero update_observation_status calls and creates zero rows."
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observation_records.py#test_dry_run_makes_zero_status_refresh_calls"
        status: pass
    human_judgment: false
  - id: D3
    description: "A skipped_existing (pre-existing) record does not trigger update_observation_status."
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observation_records.py#test_skipped_existing_record_makes_zero_status_refresh_calls"
        status: pass
    human_judgment: false
  - id: D4
    description: "An update_observation_status failure is skip-and-logged to stderr, increments status_sync_failed, does not roll back the created record, and does not abort the run (a later request in the same group still processes)."
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observation_records.py#test_status_refresh_failure_is_non_fatal_and_does_not_roll_back"
        status: pass
    human_judgment: false

duration: 15min
completed: 2026-07-23
status: complete
---

# Quick Task 260722-ux0: Fix backfill_lco_observation_records perpetual [QUEUED] Summary

**Newly created LCO backfill ObservationRecords now get a best-effort post-create `facility.update_observation_status()` refresh, so `scheduled_start`/`scheduled_end` are populated immediately and `sync_lco_observation_calendar._title_for()` stops mislabeling terminal (e.g. COMPLETED) backfilled records as `[QUEUED]`.**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-07-23T05:08:00Z
- **Completed:** 2026-07-23T05:22:53Z
- **Tasks:** 3 (Task 3 verified clean with no additional changes needed)
- **Files modified:** 2

## Accomplishments
- `Command.handle()` now calls `facility.update_observation_status(observation_id)` exactly once, immediately after a real (non-dry-run) `ObservationRecord.objects.create()`, populating `status`/`scheduled_start`/`scheduled_end` from the live LCO API the same way periodic polling does.
- Failures of that refresh call are caught broadly, logged to stderr, counted in a new `status_sync_failed` summary counter, and are non-fatal — the already-created record is never rolled back and the loop continues to the next request.
- `--dry-run` and the `skipped_existing` branch both provably never reach the refresh call (covered by dedicated tests).
- Class docstring updated to document the new best-effort live refresh behavior.
- Extended the test suite from 16 to 20 tests, all green; `ruff check` and `ruff format --check` both pass clean on the two touched files.

## Task Commits

Each task was committed atomically:

1. **Task 1: Refresh scheduled_start/scheduled_end on newly created records** - `8b1b1d2` (feat)
2. **Task 2: Extend tests for status refresh, dry-run, skip, and failure paths** - `6c5b205` (test)
3. **Task 3: Quality gate** - no code changes needed; `ruff check` and `ruff format --check` both passed clean on first run against the already-committed files from Tasks 1-2.

**Plan metadata:** committed separately by the orchestrator (docs commit not made by this executor per constraints).

## Files Created/Modified
- `solsys_code/management/commands/backfill_lco_observation_records.py` - Added `status_sync_failed` counter, the post-create `facility.update_observation_status(observation_id)` call wrapped in try/except, updated summary line, and updated class docstring.
- `solsys_code/tests/test_backfill_lco_observation_records.py` - Added a class-wide `setUp()` patch of `LCOFacility.update_observation_status` (needed so pre-existing tests don't make real network calls now that the create branch calls it) plus 4 new tests covering the refresh-call, dry-run-zero-calls, skipped-existing-zero-calls, and non-fatal-failure-with-no-rollback behaviors.

## Decisions Made
- Chose to add the `LCOFacility.update_observation_status` patch at the `TestCase.setUp()` level (applied to every test in the class via `addCleanup`) rather than adding it individually to each of the 16 pre-existing tests. This satisfies the plan's "keep all existing tests unchanged and passing" requirement — no existing test body or assertion changed — while preventing them from making real live HTTP calls to `observe.lco.global` (confirmed live during verification: without this patch, `test_creates_record_for_matching_group_and_target` failed because the real LCO API returned `WINDOW_EXPIRED` for a real request id=10, not the test's expected `COMPLETED`).
- Kept the summary-line format additive (`, status sync failed: {count}` appended to the existing string) so both the dry-run and normal wording paths share one summary line, per plan instruction; `status_sync_failed` is always `0` in dry-run since the call site is unreachable there.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Pre-existing tests would have made real live HTTP calls to LCO once Task 1 landed**
- **Found during:** Task 1 verification (running the pre-existing test suite against the new code)
- **Issue:** `facility.update_observation_status()` internally calls `OCSFacility.get_observation_status()`, which uses its own module-level `make_request` reference (`tom_observations.facilities.ocs.make_request`) — a different binding from the one the command module imports and the existing tests patch (`solsys_code....backfill_lco_observation_records.make_request`). Patching the command's `make_request` therefore does not intercept the new call, so every non-dry-run pre-existing test would hit the real LCO portal using the real `LCO_APIKEY` configured in `src/fomo/settings.py`. This was directly observed: `test_creates_record_for_matching_group_and_target` failed with `AssertionError: 'WINDOW_EXPIRED' != 'COMPLETED'` after a genuine network round-trip to `observe.lco.global` for observation_id=10.
- **Fix:** Added a class-wide `setUp()` patch of `tom_observations.facilities.lco.LCOFacility.update_observation_status` (patched at the class level, matching the plan's Task 2 guidance for the new tests) so no test — new or pre-existing — makes a real network call via this new code path.
- **Files modified:** `solsys_code/tests/test_backfill_lco_observation_records.py`
- **Verification:** Full test run after the fix: 20/20 tests pass with no network access; re-ran the previously-failing test in isolation to confirm it now passes deterministically.
- **Committed in:** `6c5b205` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix, Rule 1)
**Impact on plan:** Necessary to keep the test suite hermetic and deterministic (no live network calls, no flakiness from real LCO API state). No scope creep — this was the mechanical consequence of Task 1's change interacting with the existing test file's patch target, addressed exactly as the plan's Task 2 instructions anticipated (it directs patching `LCOFacility.update_observation_status` at the class level for the new tests; applying that same patch class-wide via `setUp()` was the minimal way to also keep the 16 pre-existing tests green).

## Issues Encountered
None beyond the deviation above (which was resolved within the deviation).

## Known Stubs
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `backfill_lco_observation_records` now leaves newly created records in a state consistent with normal TOM polling; no further action needed for `sync_lco_observation_calendar` to render them correctly.
- No blockers.

---
*Phase: quick-260722-ux0*
*Completed: 2026-07-23*

## Self-Check: PASSED

- FOUND: solsys_code/management/commands/backfill_lco_observation_records.py
- FOUND: solsys_code/tests/test_backfill_lco_observation_records.py
- FOUND: .planning/quick/260722-ux0-fix-backfill-lco-observation-records-pop/260722-ux0-SUMMARY.md
- FOUND commit: 8b1b1d2
- FOUND commit: 6c5b205
