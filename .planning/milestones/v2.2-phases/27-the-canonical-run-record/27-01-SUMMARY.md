---
phase: 27-the-canonical-run-record
plan: 01
subsystem: api
tags: [django, calendar_utils, telescope_class, regex, refactor]

# Dependency graph
requires:
  - phase: 26-canonical-record-spike
    provides: "D-11/D-12/D-16/D-20/D-21 locked verdicts on telescope_class's vocabulary, SPACE's meaning, and the shared-helper placement"
provides:
  - "calendar_utils.derive_telescope_class(site_raw, telescope_instrument) -> '2m0'/'1m0'/'0m4'/'SPACE'/'' -- the single shared derivation Plan 27-04's backfill migration and Plan 27-06's import_campaign_csv will both call"
  - "calendar_utils.py's five cross-module-consumed helpers are now public (aperture_class_from_telescope_code, derive_telescope, resolve_placement_block, extract_instrument, coarse_telescope_label)"
  - "calendar_utils-owned unit tests relocated out of test_sync_lco_observation_calendar.py into test_calendar_utils.py"
affects: [27-04-window-schema-migration-and-telescope-class-backfill, 27-06-import-campaign-csv-source-and-telescope-class]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Primitives-in, never-raise, sentinel-on-no-match helper shape (matches aperture_class_from_telescope_code precedent)"
    - "Linear (non-backtracking) regex over CSV free text, no nested quantifiers, digit-must-precede-'m' anchoring to avoid trailing-digit false positives (T-27-04)"
    - "Function-local import to avoid dragging a live-model-importing module into calendar_utils' import graph (campaign_utils.HORIZONS_OBSERVER_TO_OBSCODE)"

key-files:
  created: []
  modified:
    - solsys_code/calendar_utils.py
    - solsys_code/management/commands/sync_lco_observation_calendar.py
    - solsys_code/management/commands/backfill_lco_observation_records.py
    - solsys_code/tests/test_sync_lco_observation_calendar.py
    - solsys_code/tests/test_calendar_utils.py

key-decisions:
  - "_observations_block_response() stays owned by test_sync_lco_observation_calendar.py (still used by many command-behaviour tests there); test_calendar_utils.py imports it rather than duplicating it, since an import can't silently drift out of sync the way two copies could"
  - "derive_telescope_class's aperture regex uses a single generic \\b(\\d(?:\\.\\d)?)\\s?m\\b pattern for metre-phrases (1m/1.0m/2m/2.0m/4m/4.0m/0.4m) rather than an alternation enumerating every literal phrase -- simpler, still linear, and the digit-precedes-m ordering inherently rejects 'MuSCAT4' (m precedes the digit) without a special-case exclusion"
  - "D-12's subset-assertion test computes calendar_utils' aperture-class set by calling aperture_class_from_telescope_code() on real dome-suffixed codes ('0m4a' etc.) rather than hardcoding the set literal a second time, so the two can't independently drift"

patterns-established:
  - "A CampaignRun-adjacent derivation helper that must be migration-safe takes only primitives and does its third-party-module import (HORIZONS_OBSERVER_TO_OBSCODE) function-locally, not at module scope"

requirements-completed: [CANON-02]

# Metrics
duration: 25min
completed: 2026-07-30
---

# Phase 27 Plan 01: calendar_utils.py Shared-Module Cleanup and telescope_class Derivation Summary

**Renamed calendar_utils.py's five cross-module helpers to public names, added the single `derive_telescope_class()` helper (D-20) with its D-12 subset-assertion test suite, and relocated the six calendar_utils-owned tests out of test_sync_lco_observation_calendar.py.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-07-29T23:48:52Z
- **Completed:** 2026-07-30T00:04:03Z
- **Tasks:** 3 completed
- **Files modified:** 5

## Accomplishments

- Closed both halves of todo `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md`: the five helpers (`aperture_class_from_telescope_code`, `derive_telescope`, `resolve_placement_block`, `extract_instrument`, `coarse_telescope_label`) are now public, and the six tests that exercise them directly now live in `test_calendar_utils.py`.
- Added `derive_telescope_class(site_raw, telescope_instrument)` — the one shared, primitives-only `telescope_class` derivation D-20 requires, ready for Plan 27-04's backfill migration and Plan 27-06's `import_campaign_csv` to both call.
- 13 new tests cover every D-16 dev-DB row shape (LCO 1m/2m/0.4m, JUICE via both blank-site tier-b and Horizons-site tier-a, JWST's Horizons-alias non-SPACE case, HST, Swift, an unrelated site, SOAR's excluded 4m0, MuSCAT4's non-false-positive trailing digit, and the never-raises `(None, None)` case), plus the mandatory D-12 subset assertion naming `4m0` as the known exclusion.

## Task Commits

Each task was committed atomically:

1. **Task 1: Drop the leading underscore on calendar_utils.py's five shared helpers** - `5da1c1e` (refactor)
2. **Task 2: Add derive_telescope_class() and its unit tests, including D-12's subset assertion** - `f1d3ac5` (feat)
3. **Task 3: Move the calendar_utils-owned tests into test_calendar_utils.py** - `a3fcaea` (test)

_No TDD tasks in this plan (autonomous, non-TDD execute plan)._

## Files Created/Modified

- `solsys_code/calendar_utils.py` - Five helpers renamed to public names; new `derive_telescope_class()`, `NO_OBSCODE_SPACE_OBSERVATORIES`, and the two linear aperture regexes added immediately after `aperture_class_from_telescope_code` per D-20's stated placement
- `solsys_code/management/commands/sync_lco_observation_calendar.py` - Import block and 4 call sites updated to the new public names; docstring references updated
- `solsys_code/management/commands/backfill_lco_observation_records.py` - One docstring reference (`calendar_utils._extract_instrument` -> `calendar_utils.extract_instrument`) updated
- `solsys_code/tests/test_sync_lco_observation_calendar.py` - Import block updated then trimmed to only what remains used after the Task 3 move (`SITE_TELESCOPE_MAP`, `aperture_class_from_telescope_code`, `derive_telescope`, `resolve_placement_block`, `re`, `django.forms` all removed as now-unused)
- `solsys_code/tests/test_calendar_utils.py` - New `TestDeriveTelescopeClass` (13 tests), `TestTelescopeLabelResolutionHelpers` and `TestResolvePlacementBlockFailureModes` (the 6 relocated tests, method names byte-identical)

## Decisions Made

- Kept `_observations_block_response()` in `test_sync_lco_observation_calendar.py` and imported it into `test_calendar_utils.py` rather than duplicating it (plan left this as an explicit choice to record) — many command-behaviour tests in the sync module still depend on it, and an import can't silently drift the way two copies could.
- Used one generic metre-phrase regex (`\b(\d(?:\.\d)?)\s?m\b`) instead of enumerating every literal phrase (`1m`, `1.0m`, `2m`, ...) in an alternation — still linear (T-27-04), and the digit-must-precede-`m` ordering is what naturally rejects `MuSCAT4` (the `m` there precedes the trailing digit, not follows it) without a special-cased exclusion list.
- D-12's subset-assertion test derives calendar_utils' aperture-class set by actually calling `aperture_class_from_telescope_code()` on realistic dome-suffixed codes, rather than hardcoding `{'0m4', '1m0', '2m0', '4m0'}` a second time as a literal — so the test can't independently drift from the function it's checking.

## Deviations from Plan

None - plan executed exactly as written. All three tasks' acceptance criteria were met, including the D-12 subset assertion, the `NO_OBSCODE_SPACE_OBSERVATORIES` function-local-import placement, and the six-test relocation with byte-identical method names.

One expected grep-literalism note, not a deviation: the acceptance criterion's zero-hit grep for the five old underscore-prefixed names has one unavoidable false-positive hit — the pre-existing test method name `test_telescope_01_aperture_class_from_telescope_code` contains `_aperture_class_from_telescope_code` as a substring (from `..._01_aperture_class...`). Task 3 explicitly requires keeping every relocated test method name byte-identical, so this method name was not altered. It is a test *name*, not a reference to the old private function (the function itself, and every call site, were renamed and verified via `def` grep and full test-suite pass); no actual `_aperture_class_from_telescope_code` symbol exists anywhere in the codebase.

## Issues Encountered

`./manage.py test solsys_code` (the plan's own end-of-plan full-suite verification command) segfaults inside `sorcha`/`ASSIST`'s native C extension (`assist/extras.py:44` -> `integrate_light_time`) while running `test_views.py:test_K93`, an existing test in `solsys_code/tests/test_views.py` that is entirely unrelated to this plan's files (`ephem_utils`/`views.py` ephemeris generation, not `calendar_utils.py`). This is a pre-existing environment issue (CLAUDE.md documents the ~1.6GB SPICE kernel cost of importing `ephem_utils`/`views`; the plan's own "Quick run" verification command deliberately excludes both SPICE-heavy modules for this reason) and is out of scope per the deviation rules' scope boundary. Verified instead by running all `solsys_code` test modules except `test_ephem_utils` and `test_views` (567 tests, all pass) plus the plan's own quick-run command (307 tests, pass) and the three tasks' targeted module sets (111 and 66 tests respectively, pass).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `calendar_utils.derive_telescope_class()` is ready for Plan 27-04 (data migration backfill, `site__isnull=True` gate) and Plan 27-06 (`import_campaign_csv`, `resolve_site()` returned `None` gate) to both call.
- Plan 27-04's Task 3 must replace the literal `{'2m0', '1m0', '0m4'}` set in `test_calendar_utils.py`'s D-12 subset-assertion test with `CampaignRun.TelescopeClass` once that enum exists (the `# Plan 27-04 wires this to CampaignRun.TelescopeClass` marker is in place and greppable).
- No blockers. `ruff check .`/`ruff format --check .` report the same pre-existing, unrelated issues present before this plan (7 files needing reformat, 1 notebook docstring warning — none touched by this plan).

---
*Phase: 27-the-canonical-run-record*
*Completed: 2026-07-30*

## Self-Check: PASSED

All 6 modified/created files confirmed present on disk; all 3 task commit hashes (`5da1c1e`, `f1d3ac5`, `a3fcaea`) confirmed present in `git log --oneline --all`.
