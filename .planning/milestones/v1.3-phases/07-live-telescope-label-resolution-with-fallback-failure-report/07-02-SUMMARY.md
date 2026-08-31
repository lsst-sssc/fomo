---
phase: 07-live-telescope-label-resolution-with-fallback-failure-report
plan: 02
subsystem: api
tags: [django, tom-toolkit, lco-observation-portal, requests, management-command]

# Dependency graph
requires:
  - phase: 07-live-telescope-label-resolution-with-fallback-failure-report (Plan 01)
    provides: "_resolve_placement_block, _aperture_class_from_telescope_code, 2-arg never-raise _derive_telescope, verified 7-site SITE_TELESCOPE_MAP"
provides:
  - "_coarse_telescope_label(instrument_type) -> str -- coarse aperture-class fallback derived from Phase-6 instrument_type"
  - "_build_event_fields fallback decision tree: placed-record live API resolution -> verified label, or coarse fallback on failure/timeout/unmapped-code (same bucket, Pitfall 4)"
  - "_title_for extended with label_was_fallback -> [UNVERIFIED] prefix, D-09 priority (terminal beats [UNVERIFIED]; [QUEUED]/[UNVERIFIED] mutually exclusive)"
  - "telescope_api_failed per-facility counter, distinct from skipped/extraction_failed, in Command.handle()'s counters dict and summary line"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Never-raise/sentinel-on-failure convention applied to _build_event_fields' label-resolution branch (label_was_fallback bool, never an exception)"
    - "Fixed generic stderr message on API-failure fallback, never interpolating the caught exception (SYNC-09)"

key-files:
  created: []
  modified:
    - solsys_code/management/commands/sync_lco_observation_calendar.py
    - solsys_code/tests/test_sync_lco_observation_calendar.py
    - docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb

key-decisions:
  - "An API call failure/timeout AND a successfully-returned-but-unmapped (site, telescope_code) pair are the SAME fallback bucket (Pitfall 4) -- both set label_was_fallback=True, increment the same telescope_api_failed counter, and get the same [UNVERIFIED] prefix"
  - "D-09 resolved: title-prefix priority is terminal-failure prefix > [UNVERIFIED] > [QUEUED] > clean; [QUEUED] and [UNVERIFIED] are mutually exclusive by construction since [UNVERIFIED] only ever applies to a placed record (D-07)"
  - "_build_event_fields does NOT log the API failure itself -- Command.handle() owns the single stderr log line, keeping caller-logging-discipline (SYNC-09) in one place"
  - "Removed Plan 01's interim single-class shim ({'coj': '2m0', 'ogg': '2m0', 'sor': '4m0'} flat-site lookup) entirely, replacing the call site with the live-API + fallback decision tree"

patterns-established:
  - "telescope_api_failed: bool is returned as an extra key in _build_event_fields' fields dict and popped by Command.handle() before constructing CalendarEvent kwargs, mirroring how 'url' is already popped"

requirements-completed: [TELESCOPE-02, TELESCOPE-03, TELESCOPE-04, SYNC-06, SYNC-07, SYNC-09]

# Metrics
duration: ~55min
completed: 2026-06-23
status: complete
---

# Phase 07 Plan 02: Wire Fallback Decision Tree, [UNVERIFIED] Prefix, telescope_api_failed Counter Summary

**Replaced the flat `parameters['site']` shim with a live-API + coarse-fallback decision tree, an `[UNVERIFIED]` title prefix with D-09-resolved priority, and a per-facility `telescope_api_failed` counter -- completing Phase 7's user-visible behavior.**

## Performance

- **Duration:** ~55 min
- **Started:** 2026-06-23T05:27:53Z (per STATE.md "Phase 07 execution started")
- **Completed:** 2026-06-23 (this session)
- **Tasks:** 3 (Task 1 auto, Task 2 auto/tdd, Task 3 auto)
- **Files modified:** 3

## Accomplishments

- Replaced Plan 01's interim single-class flat-site shim in `_build_event_fields` with the full TELESCOPE-02/03/04 decision tree: a banner-stage record (`scheduled_start is None`) gets the coarse fallback label with no API call (D-01); a placed record attempts a single live `_resolve_placement_block` call, mapping a successful response through `_derive_telescope`, and falling back to the coarse label on any failure/timeout/unmapped-code (Pitfall 4 -- the same bucket for both failure modes)
- Added `_coarse_telescope_label(instrument_type)` deriving the `1m0`/`0m4`/`2m0`/`4m0` fallback token from the Phase-6-extracted instrument type, with a safe never-raise fallback to the raw instrument_type string for unrecognized prefixes
- Extended `_title_for` with a `label_was_fallback` parameter and an `[UNVERIFIED]` branch, resolving D-09's open priority question: terminal-failure prefixes win over `[UNVERIFIED]`; `[QUEUED]` and `[UNVERIFIED]` are mutually exclusive by construction
- Added the `telescope_api_failed` counter to both per-facility counter dicts, the defensive `setdefault` default, and the summary f-string, distinct from `skipped`/`extraction_failed`; `Command.handle()` logs one fixed generic stderr line on fallback (SYNC-09 -- never interpolates the caught exception) and never aborts the run (SYNC-07)
- Added 6 new integration tests covering the fallback path, the visible label-flip (fallback -> verified on a subsequent successful run), the counter's distinctness from `skipped`, no-abort-on-failure, the fixed-generic-message no-leak guarantee, and the banner-record no-API-call/no-`[UNVERIFIED]` guarantee
- Reconciled 4 pre-existing tests that assumed the removed flat-site shim (`test_sync_02`/`test_sync_03`/`test_sync_05`/`test_d06_completed`) and repurposed `test_skip_path_missing_site_logged_and_skipped` into `test_banner_record_missing_site_still_syncs_with_coarse_label` (a missing flat `'site'` key no longer causes a skip)
- Regenerated the paired demo notebook with 3 new cells (TELESCOPE-02 success, TELESCOPE-03/04 fallback, SYNC-06 counter) demonstrating all three new observable behaviors with real executed output

## Task Commits

1. **Task 1: Wire fallback decision tree, [UNVERIFIED] prefix, and telescope_api_failed counter into the command** - `0046e48` (feat)
2. **Task 2: Integration tests for fallback, counter, no-abort, fixed-message log, and [UNVERIFIED]/banner prefix behavior** - `1ae2f5b` (test)
3. **Task 3: Update and regenerate the paired demo notebook** - `7a684e6` (docs), follow-up ruff-import-order fix - `26c88ca` (fix)

## Files Created/Modified

- `solsys_code/management/commands/sync_lco_observation_calendar.py` - `_coarse_telescope_label`, extended `_title_for` (D-09), reworked `_build_event_fields` (decision tree + `telescope_api_failed` key), extended `Command.handle()` counters/log line/summary
- `solsys_code/tests/test_sync_lco_observation_calendar.py` - 6 new integration tests; 4 pre-existing tests updated to mock a successful API response (preserving their original verified-label intent) or updated expectations for the no-API-call banner case; 1 test repurposed for the no-longer-required flat `'site'` key
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` - 3 new cells (TELESCOPE-02 success, TELESCOPE-03/04 fallback, SYNC-06 counter) with real executed output; extended teardown and Summary requirements table

## Decisions Made

- Pitfall 4 applied literally: API failure/timeout and a successfully-returned-but-unmapped code are the same fallback bucket for both the counter and the title prefix -- the `description` field is the only place the unmapped-code sub-case could safely add detail, but the simpler always-generic note was used for both sub-cases per the plan's explicit "simplest compliant approach" guidance.
- D-09 resolved per RESEARCH.md Pattern 3's recommendation: terminal-failure prefix > `[UNVERIFIED]` > `[QUEUED]` > clean, matching Phase 4's existing terminal-prefix-wins precedent.
- `_build_event_fields` does not log anything itself; `Command.handle()` is the single place that writes the SYNC-09 fixed generic stderr line, keeping the existing "caller logs, helper returns sentinels" convention from Plan 01 intact.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - reconciling pre-existing tests broken by Plan 02's own wiring] 4 pre-existing tests assumed the removed flat-site shim**
- **Found during:** Task 2 (after Task 1's commit, running the full pre-existing test file)
- **Issue:** `test_sync_02_d03_unscheduled_uses_parameters_times_and_queued_title`, `test_sync_03_d03_placed_uses_scheduled_times_and_clean_title`, `test_sync_05_telescope_instrument_proposal_populated`, and `test_d06_completed_gets_clean_title_no_prefix` all asserted a verified `SITECODE-CLASS` label derived from the now-removed `parameters['site']` flat-key shim. After Task 1's wiring, a placed record's label now requires a live (mocked, in tests) API call; a banner record's label is now always the coarse fallback (no API call per D-01), never a verified `SITECODE-CLASS` label.
- **Fix:** `test_sync_02` (a banner-stage record) was updated to expect the coarse fallback label (`'2m0'`), matching D-01's "no API call for banner records" rule exactly. `test_sync_03`/`test_sync_05`/`test_d06_completed` (placed records) were updated to mock `make_request` with a successful, site-matching response, preserving their original intent (a verified `SITECODE-CLASS` label, clean title) without relying on the removed shim.
- **Files modified:** `solsys_code/tests/test_sync_lco_observation_calendar.py`
- **Verification:** `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` -- all tests pass; this was explicitly anticipated by the plan's own Task 1 acceptance criteria ("the now-obsolete `test_skip_path_missing_site_logged_and_skipped` behavior may need its expectation updated... adjust it in Task 2").
- **Committed in:** `1ae2f5b` (Task 2's commit)

**2. [Rule 1 - test reconciliation] `test_skip_path_missing_site_logged_and_skipped` no longer matched real behavior**
- **Found during:** Task 2
- **Issue:** The test asserted a record with no flat `parameters['site']` key gets skipped and logged. After Task 1's wiring, `_build_event_fields` no longer reads `parameters['site']` at all -- a missing `'site'` key on a banner record simply means the record still syncs via the coarse fallback path, never causing a skip.
- **Fix:** Repurposed the test as `test_banner_record_missing_site_still_syncs_with_coarse_label`, asserting the record now syncs successfully with a `[QUEUED]` (not `[UNVERIFIED]`) title and no skip.
- **Files modified:** `solsys_code/tests/test_sync_lco_observation_calendar.py`
- **Verification:** `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` passes; full `./manage.py test solsys_code` (128 tests) green.
- **Committed in:** `1ae2f5b` (Task 2's commit)

**3. [Rule 1 - bug] New notebook cell's import block flagged by ruff I001**
- **Found during:** post-Task-3 full-repo `ruff check .` sweep (before writing this summary)
- **Issue:** The new TELESCOPE-02 success cell's import order (`unittest.mock` before `datetime`) was not stdlib-alphabetical, flagged by `ruff check`'s I001 rule.
- **Fix:** Reordered to `datetime`/`datetime.timezone` before `unittest.mock`, then regenerated the notebook via `jupyter nbconvert --to notebook --execute --inplace` to keep executed output in sync with the corrected source.
- **Files modified:** `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`
- **Verification:** `ruff check docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` -- the new cell's I001 finding is gone; only 2 pre-existing baseline findings remain (cell 6 import order, cell 12 line length), both untouched by this plan and confirmed identical via `git stash`/re-check against the pre-Task-3 committed state.
- **Committed in:** `26c88ca` (follow-up fix commit)

**Total deviations:** 3 auto-fixed (all Rule 1). **Impact on plan:** All three were necessary to keep the full existing/new test suite and `ruff check .` green for files this plan touches. No scope creep -- none implement functionality beyond TELESCOPE-02/03/04/SYNC-06/07/09's locked scope.

## Issues Encountered

- `ObservationRecord`'s default model ordering is `('-created',)` (most-recently-created first), which matters for `test_sync_07_api_failure_does_not_abort_run`'s `make_request` `side_effect` list (a list of per-call return values consumed in call order) -- the test's two fixture records had to have their `side_effect` entries ordered to match actual processing order (second-created record processed first), not creation order. Documented inline in the test's docstring and a comment to prevent future confusion.

## User Setup Required

None - no external service configuration required. This plan consumes the existing `LCO_APIKEY` env var already configured for prior phases; no new credentials or settings are introduced.

## Next Phase Readiness

Phase 7 is now complete (both Plan 01 and Plan 02 executed): `TELESCOPE-01` through `TELESCOPE-04` and `SYNC-06` through `SYNC-09` are all implemented and tested. The `sync_lco_observation_calendar` command now resolves telescope labels via a live, timeout-bounded, never-leaking API call for placed records, with a visibly-marked, separately-counted, non-fatal coarse fallback for everything else. Full `./manage.py test solsys_code` (128 tests) and `python -m pytest` are green; `ruff check .`/`ruff format --check .` are clean for every file this plan touched (2 pre-existing, unrelated baseline findings remain in the demo notebook and elsewhere, untouched by this or prior GSD phases).

No blockers carried forward. The `src/fomo/settings.py` uncommitted `LCO_APIKEY`-related change noted in STATE.md's Blockers/Concerns predates this phase and remains untouched by this plan, as instructed.

## Self-Check: PASSED

- FOUND: `solsys_code/management/commands/sync_lco_observation_calendar.py`
- FOUND: `solsys_code/tests/test_sync_lco_observation_calendar.py`
- FOUND: `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`
- FOUND commit: `0046e48`
- FOUND commit: `1ae2f5b`
- FOUND commit: `7a684e6`
- FOUND commit: `26c88ca`

---
*Phase: 07-live-telescope-label-resolution-with-fallback-failure-report*
*Completed: 2026-06-23*
