---
phase: quick-260620-v9x
plan: 01
subsystem: testing
tags: [jupyter, notebook, django, sync_lco_observation_calendar, instrument-extraction]

# Dependency graph
requires:
  - phase: 06-correct-instrument-type-extraction
    provides: "_extract_instrument multi-config scanner and InstrumentExtractionError/extraction_failed counter in sync_lco_observation_calendar.py"
provides:
  - "Phase 6 EXTRACT-01/EXTRACT-02 demo coverage in sync_lco_observation_calendar_demo.ipynb (SOAR multi-config, MUSCAT per-channel, malformed/extraction_failed)"
affects: [phase-07-live-telescope-label-resolution]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "demo-606xxx observation_id prefix convention for Phase-6 notebook fixtures (mirrors demo-602xxx..604xxx Phase-5 convention)"

key-files:
  created: []
  modified:
    - docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb

key-decisions:
  - "Followed the plan's exact fixture shapes from test_extract_02_soar_multi_config_picks_spectrum_not_calibration, test_extract_02_muscat_per_channel_exposure_extracts_instrument, and test_d06_no_extractable_config_logged_and_counted_separately verbatim, translated into get_or_create notebook fixtures"
  - "Used a one-shot Python build script (.planning/quick/260620-v9x.../build_nb.py, untracked) to perform the JSON cell insertion/update rather than hand-editing the .ipynb JSON, to guarantee well-formed cell objects and avoid structural drift"

requirements-completed: [EXTRACT-01, EXTRACT-02]

# Metrics
duration: ~25min
completed: 2026-06-21
status: complete
---

# Quick Task 260620-v9x Summary

**Extended the sync_lco_observation_calendar_demo.ipynb with Phase 6 EXTRACT-01/EXTRACT-02 cells (SOAR SPECTRUM-vs-calibration extraction, LCO MUSCAT per-channel extraction, and a malformed-record extraction_failed counter demo), executed end-to-end with real output committed.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-06-21T05:18:00Z (approx, prior to commit history)
- **Completed:** 2026-06-21T05:42:14Z
- **Tasks:** 2 completed
- **Files modified:** 1 (`docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`)

## Accomplishments

- Added a new "Phase 6: correct instrument-type extraction (EXTRACT-01/EXTRACT-02)" notebook section with three demonstration cells:
  - **SOAR multi-config** (`demo-606001`): a record with `c_1` SPECTRUM/`SOAR_GHTS_REDCAM`, `c_2` ARC/`SOAR_GHTS_REDCAM_ARC`, `c_3` LAMP_FLAT/`SOAR_GHTS_REDCAM_LAMPFLAT`, and a decoy flat `instrument_type`. Printed output confirms `event.instrument == 'SOAR_GHTS_REDCAM'` and neither calibration value leaks through.
  - **LCO MUSCAT per-channel** (`demo-606002`): a record with only the four `c_1_ic_1_exposure_time_{g,r,i,z}` keys (no flat `c_1_exposure_time`). Printed output confirms `event.instrument == '2M0-SCICAM-MUSCAT'`.
  - **D-06 malformed record** (`demo-606003` malformed + `demo-606004` baseline-good sibling): printed stdout shows `extraction_failed: 1` (distinct from `skipped`), stderr logs `demo-606003`, and the good sibling still produces a `CalendarEvent` in the same `call_command` invocation.
- Updated the Summary table with EXTRACT-01/EXTRACT-02 rows in the existing column format, all prior rows (SELECT-01/SYNC-01..04/TERM-01/SELECT-02..05) intact.
- Updated the teardown cell to delete all four new Phase-6 fixtures (CalendarEvent + ObservationRecord) before the shared Target/User cleanup.
- Executed the notebook end-to-end (`jupyter nbconvert --to notebook --execute --inplace`) against the local dev DB; all 15 code cells (minus the Django-setup cell, which legitimately prints nothing) produced output, and the teardown cell's final print confirmed a clean DB state.

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend the demo notebook with EXTRACT-02/D-06 cells, updated Summary table, and teardown** - `0769adc` (docs)
2. **Task 2: Regenerate notebook output and commit per the pre_executed convention** - `d41cdc7` (docs)

_Note: both commits are `docs(260620-v9x): ...` since this plan touches only the `.ipynb` artifact, no application code._

## Files Created/Modified

- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` - Added Phase 6 EXTRACT-01/EXTRACT-02 demo cells (SOAR multi-config, MUSCAT per-channel, malformed/extraction_failed), updated Summary table and teardown cell, regenerated with real executed output.

## Decisions Made

- Built the JSON edit via a small one-shot Python script (`.planning/quick/260620-v9x-update-docs-notebooks-pre-executed-sync-/build_nb.py`, left untracked — not part of the plan's `files_modified` and not committed) rather than hand-constructing JSON via the Edit tool, to guarantee valid nbformat cell structures (fresh unique `id`s, correct `cell_type`/`outputs`/`execution_count` keys) on the first attempt.
- Reused the plan's exact fixture parameter shapes from the three referenced Phase-6 tests (`test_extract_02_soar_multi_config_picks_spectrum_not_calibration`, `test_extract_02_muscat_per_channel_exposure_extracts_instrument`, `test_d06_no_extractable_config_logged_and_counted_separately`) verbatim, translated from `_create_record(**parameter_overrides)` test helper calls into `ObservationRecord.objects.get_or_create(...)` notebook fixtures.

## Deviations from Plan

None - plan executed exactly as written. `sync_lco_observation_calendar.py` and its test file were read but not modified, per the plan's explicit scope boundary.

## Issues Encountered

None. The notebook executed cleanly on the first `nbconvert --execute` run; pre-existing `ruff check`/`ruff format --check` findings on this notebook and others (16 errors, "would reformat") were confirmed identical before and after this plan's edit via a `git stash`/re-check comparison, confirming no new lint regressions were introduced by this change.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 06 (correct-instrument-type-extraction) now has full notebook demo coverage alongside its passing test suite; no further notebook work needed for that phase.
- Project STATE.md/ROADMAP.md are not touched by this quick task per the dispatch's constraints; the orchestrator handles state/docs updates separately.
- Pre-existing uncommitted `src/fomo/settings.py` (LCO_APIKEY edit) was left untouched throughout, as instructed.

---
*Quick task: 260620-v9x*
*Completed: 2026-06-21*

## Self-Check: PASSED

- FOUND: docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb
- FOUND: .planning/quick/260620-v9x-update-docs-notebooks-pre-executed-sync-/260620-v9x-SUMMARY.md
- FOUND: commit 0769adc (Task 1)
- FOUND: commit d41cdc7 (Task 2)
