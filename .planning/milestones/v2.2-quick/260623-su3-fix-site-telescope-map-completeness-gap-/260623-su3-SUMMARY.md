---
phase: quick-260623-su3
plan: 01
subsystem: api
tags: [lco, telescope-label-resolution, sync-lco-observation-calendar, regression-test]

# Dependency graph
requires:
  - phase: 07-live-telescope-label-resolution-with-fallback-failure-report
    provides: SITE_TELESCOPE_MAP, _derive_telescope, _aperture_class_from_telescope_code
provides:
  - "Widened SITE_TELESCOPE_MAP (10 -> 13 entries) closing the coj/ogg aperture-class gap found in Phase 7 UAT Test 1"
  - "Regression test guarding the 3 newly-mapped (site, class) pairs"
affects: [phase-07-uat, sync_lco_observation_calendar]

# Tech tracking
tech-stack:
  added: []
  patterns: ["SITE_TELESCOPE_MAP dict-completeness fix sourced from https://lco.global/observatory/sites/mpccodes/"]

key-files:
  created: []
  modified:
    - solsys_code/management/commands/sync_lco_observation_calendar.py
    - solsys_code/tests/test_sync_lco_observation_calendar.py

key-decisions:
  - "Confirmed the 3 new entries against the authoritative public source https://lco.global/observatory/sites/mpccodes/ rather than re-opening the operator-confirmation channel used for the original Plan 07-01 entries"
  - "tlv stays intentionally excluded per the 07-01 operator decision; the public table listing it does not override confirmed absence from installed LCOSettings/SOARSettings"
  - "No demo notebook regeneration — the command's documented behavior paths (success/fallback/counter) are unchanged; only label coverage widened for sites the notebook's fixtures don't exercise"

patterns-established: []

requirements-completed: [TELESCOPE-01]

# Metrics
duration: 12min
completed: 2026-06-24
status: complete
---

# Quick Task 260623-su3: Fix SITE_TELESCOPE_MAP completeness gap Summary

**Added 3 missing (site, aperture_class) entries to SITE_TELESCOPE_MAP — coj 1m0/0m4 and ogg 0m4 — confirmed against the public LCO sites/mpccodes table, closing the gap that caused a real placed record at coj to fall back to `[UNVERIFIED]` instead of `COJ-1m0`.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-06-24T03:40:00Z
- **Completed:** 2026-06-24T03:52:39Z
- **Tasks:** 2 completed
- **Files modified:** 2

## Accomplishments
- Closed the SITE_TELESCOPE_MAP completeness gap found in Phase 7 UAT Test 1: `_derive_telescope('coj', '1m0a')`, `_derive_telescope('coj', '0m4a')`, and `_derive_telescope('ogg', '0m4b')` now resolve to their verified `SITECODE-CLASS` labels instead of falling to the `[UNVERIFIED]` bucket.
- Added a dedicated regression test (`test_telescope_01_coj_ogg_full_aperture_class_coverage`) that fails against the pre-fix dict and passes once the 3 entries are present — confirmed RED then GREEN during execution.
- Preserved the 07-01 operator decision to keep `tlv` excluded and the dict scoped to exactly the 7 real sites (`ogg`, `elp`, `lsc`, `cpt`, `coj`, `tfn`, `sor`).

## Task Commits

Each task was executed in TDD RED -> GREEN order and committed atomically:

1. **Task 2 (RED): Add regression test for the 3 newly-mapped pairs** - `cd3b17e` (test)
2. **Task 1 (GREEN): Add the 3 missing entries to SITE_TELESCOPE_MAP** - `5583400` (feat)

_Note: executed test-first (RED) then implementation (GREEN) per `tdd="true"` on both tasks, even though the plan listed the implementation task first — this matches the plan's own behavior contract ("must FAIL against the pre-fix dict... PASS once Task 1's entries are present") and is the standard TDD execution order._

## Files Created/Modified
- `solsys_code/management/commands/sync_lco_observation_calendar.py` - Added `('coj','1m0')->'COJ-1m0'`, `('coj','0m4')->'COJ-0m4'`, `('ogg','0m4')->'OGG-0m4'` to `SITE_TELESCOPE_MAP`, grouped next to each site's existing entries, with a comment citing https://lco.global/observatory/sites/mpccodes/ as the confirmation source.
- `solsys_code/tests/test_sync_lco_observation_calendar.py` - Added `test_telescope_01_coj_ogg_full_aperture_class_coverage`, asserting the 3 new resolutions via the real telescope-code form (`'1m0a'`, `'0m4a'`, `'0m4b'`).

## Decisions Made
- Used the public LCO sites/mpccodes table as primary confirmation source for the 3 new entries, documented in-code as stronger evidence than the operator-confirmation basis used for the original Plan 07-01 entries.
- Did not touch the existing `test_telescope_01_verified_dict_covers_all_sites` or `test_telescope_01_aperture_class_from_telescope_code` tests — both remain valid as-is per the plan's explicit instruction.
- Did not add a dict-length assertion anywhere — the plan explicitly noted no existing test asserts exact length and instructed not to invent one.

## Deviations from Plan

None - plan executed exactly as written. Both `must_haves.truths` claims and all `verify` commands in the plan passed without modification.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 7 UAT Test 1's gap is closed. No further action required for this quick task; Phase 7 verification can proceed treating this gap as resolved. The demo notebook (`sync_lco_observation_calendar_demo.ipynb`) was intentionally not touched per the plan's scope guardrail — its fixtures (coj/ogg 2m0, lsc 1m0) don't exercise the newly-added classes and the command's documented behavior paths are unchanged.

---
*Phase: quick-260623-su3*
*Completed: 2026-06-24*

## Self-Check: PASSED
