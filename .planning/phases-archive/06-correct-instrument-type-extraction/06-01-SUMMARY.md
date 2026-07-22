---
phase: 06-correct-instrument-type-extraction
plan: 01
subsystem: infra
tags: [django, management-command, lco, soar, tom_observations]

# Dependency graph
requires:
  - phase: 05-multi-proposal-multi-facility-selection
    provides: per-facility dispatch dict, per-facility counters dict-of-dicts, per-record catch-log-continue convention
provides:
  - "_extract_instrument(parameters) -> str | None helper implementing D-01..D-06 in sync_lco_observation_calendar.py"
  - "_SCIENCE_CONFIGURATION_TYPES whitelist distinguishing science configs from SOAR calibration/NRES configs"
  - "Dedicated 'extraction_failed' per-facility counter, distinct from 'skipped', visible in the run summary"
affects: [07-telescope-label-resolution]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Module-level whitelist set (_SCIENCE_CONFIGURATION_TYPES) checked via `in`, mirroring the existing _FAILURE_PREFIX_BY_STATUS lookup-table convention"
    - "Sentinel-None-on-total-failure + dedicated exception (InstrumentExtractionError) caught in a second except clause, to keep a new failure mode out of the existing generic 'skipped' counter"
    - "c_1..c_5 scan via `for n in range(1, 6)` with `.get(...)` only, never direct indexing on c_N_* keys"

key-files:
  created: []
  modified:
    - solsys_code/management/commands/sync_lco_observation_calendar.py
    - solsys_code/tests/test_sync_lco_observation_calendar.py

key-decisions:
  - "Sentinel None + InstrumentExtractionError (option 1 in PATTERNS.md), not a custom exception raised from the helper itself, since the file had zero existing custom exception classes and _failure_prefix already uses the 'return None to signal non-match' convention"
  - "Added a third fallback tier beyond D-01/D-02: if no c_N_* config exists at all (today's legacy single-config shape), fall back to the flat 'instrument_type' key itself -- required for the 19 pre-existing regression tests to keep passing, since CONTEXT.md's D-02 wording ('first config with a populated exposure signal') doesn't literally cover a parameters dict with zero c_N_* keys"
  - "extra_params: dict | None = None added additively to _parameters(), merged last via params.update(extra_params or {}) -- preserves all 5 existing named params/defaults and all 19 existing call sites unmodified"

requirements-completed: [EXTRACT-01, EXTRACT-02]

# Metrics
duration: 6min
completed: 2026-06-21
status: complete
---

# Phase 06 Plan 01: Correct Instrument-Type Extraction Summary

**Replaced the flat `parameters['instrument_type']` read in `sync_lco_observation_calendar.py` with a `c_1..c_5` multi-config scanner that distinguishes SOAR's SPECTRUM science config from its ARC/LAMP_FLAT calibration configs and detects LCO MUSCAT's per-channel exposure shape, adding a dedicated `extraction_failed` counter for fully-malformed records.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-06-21T01:03:54Z
- **Completed:** 2026-06-21T01:10:02Z
- **Tasks:** 2 completed
- **Files modified:** 2

## Accomplishments

- `_extract_instrument(parameters)` scans `c_1..c_5` for the first config whose `configuration_type` is in a whitelist (`EXPOSE`, `REPEAT_EXPOSE`, `SPECTRUM`, `REPEAT_SPECTRUM`, `STANDARD`), never recognizing SOAR's `ARC`/`LAMP_FLAT` calibration configs or NRES-only types.
- Falls back to the first config with a populated exposure signal — a truthy flat `c_N_exposure_time`, or (D-04) any of the 4 MUSCAT per-channel `c_N_ic_1_exposure_time_{g,r,i,z}` keys — when no config has a recognized `configuration_type`.
- Falls back further to the flat `instrument_type` key for today's legacy single-config shape (no `c_N_*` keys at all), keeping all 19 pre-existing tests passing unmodified as a regression suite.
- A fully-malformed record (no recognized `configuration_type`, no exposure signal, no flat `instrument_type`) is skipped, logged with its `observation_id`, and counted in a new dedicated `extraction_failed` counter — distinct from `skipped` — wired into all four counter locations (eager dict-literal init, `setdefault` defensive default, per-record except clause, summary join).

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend test fixture for c_N_* shapes + add failing SOAR/MUSCAT/malformed tests** - `aaf4d1f` (test)
2. **Task 2: Implement instrument-extraction helper(s) + dedicated extraction-failure counter** - `5e1489c` (feat)

_TDD tasks: RED (`aaf4d1f`) confirmed the 3 new tests fail/error while all 19 pre-existing tests pass; GREEN (`5e1489c`) made all 22 pass. No REFACTOR commit needed — ruff format was applied within the GREEN commit before staging._

## Files Created/Modified

- `solsys_code/management/commands/sync_lco_observation_calendar.py` - Added `_SCIENCE_CONFIGURATION_TYPES`, `_MUSCAT_CHANNEL_SUFFIXES`, `_has_muscat_exposure_signal`, `_find_science_config`, `_find_exposure_signal_config`, `_extract_instrument`, `InstrumentExtractionError`; replaced the flat-key read in `_build_event_fields`; wired the new `extraction_failed` counter into all four locations in `handle()`.
- `solsys_code/tests/test_sync_lco_observation_calendar.py` - Added `extra_params: dict | None = None` to `_parameters()`; added `test_extract_02_soar_multi_config_picks_spectrum_not_calibration`, `test_extract_02_muscat_per_channel_exposure_extracts_instrument`, `test_d06_no_extractable_config_logged_and_counted_separately`.

## Decisions Made

- Chose the sentinel-`None` + dedicated `InstrumentExtractionError` contract (PATTERNS.md option 1) over a bare exception, matching the file's existing "return `None` to signal non-match" style (`_failure_prefix`) and avoiding introducing a new custom-exception convention unnecessarily.
- Added a flat-`instrument_type`-key fallback tier beyond the locked D-01/D-02 algorithm, required to keep the 19 pre-existing tests passing (they exercise today's legacy shape with zero `c_N_*` keys, which D-02's literal wording — "first config with a populated exposure signal" — does not cover on its own). This is a Rule 1 auto-fix: without it, every pre-existing regression test failed/errored after Task 2's helper was wired in.
- Counter key chosen: `extraction_failed` (matches the suggested name in CONTEXT.md/PATTERNS.md).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added flat `instrument_type` fallback for the legacy single-config shape**
- **Found during:** Task 2, after wiring `_extract_instrument` into `_build_event_fields` and running the full test file
- **Issue:** `_extract_instrument`'s D-01/D-02 logic only scans `c_N_*` keys; today's legacy single-config records (the default shape produced by `_parameters()`'s un-overridden defaults) have no `c_N_*` keys at all, so the helper returned `None` for every one of the 19 pre-existing tests, causing 10 failures and 10 errors (including an `IntegrityError` on `CalendarEvent.instrument` NOT NULL)
- **Fix:** Added a third fallback tier in `_extract_instrument`: if no `c_N_*` config is found by either D-01 or D-02, fall back to `parameters.get('instrument_type')` — the original v1.2 behavior for the case where it's the only signal available
- **Files modified:** `solsys_code/management/commands/sync_lco_observation_calendar.py`
- **Verification:** All 22 tests in the file pass after the fix; full `solsys_code` suite (117 tests) green
- **Committed in:** `5e1489c` (part of Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Necessary correctness fix to satisfy the plan's own success criterion ("Single-populated-config record still extracts the same instrument value as before — all 19 existing tests pass unmodified"). No scope creep — this is the D-02 fallback's degenerate case, not a new feature.

## Issues Encountered

None beyond the deviation above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `_extract_instrument` is ready to be the instrument-data source Phase 7's telescope-label resolution can build alongside (Phase 7 covers `_derive_telescope`/`SITE_TELESCOPE_MAP`, explicitly untouched by this phase).
- All 22 tests in `test_sync_lco_observation_calendar.py` pass (19 pre-existing + 3 new); full `solsys_code` suite (117 tests) green; `ruff check`/`ruff format --check` clean for both modified files.
- No blockers identified for Phase 7.

---
*Phase: 06-correct-instrument-type-extraction*
*Completed: 2026-06-21*

## Self-Check: PASSED

- FOUND: solsys_code/management/commands/sync_lco_observation_calendar.py
- FOUND: solsys_code/tests/test_sync_lco_observation_calendar.py
- FOUND: aaf4d1f (test commit)
- FOUND: 5e1489c (feat commit)
