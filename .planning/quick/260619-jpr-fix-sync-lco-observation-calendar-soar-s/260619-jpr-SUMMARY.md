---
phase: quick-260619-jpr
plan: 01
status: complete
subsystem: testing
tags: [django, soar, lco, calendar-sync, observation-record]

# Dependency graph
requires:
  - phase: quick-260619-f7u
    provides: sync_lco_observation_calendar.py multi-proposal/multi-facility selection (SELECT-02..05) and its demo notebook
provides:
  - "sor" added to SITE_TELESCOPE_MAP so real SOAR ObservationRecords are no longer silently skipped
  - SOAR test fixtures (test_select_03/04/05) now exercise the real SOAR site code instead of the LCO default
  - Demo notebook SOAR fixture cells (demo-603002, demo-604002) produce SOAR-identifiable CalendarEvent titles
affects: [sync_lco_observation_calendar, telescope-runs-calendar-phase-06, telescope-runs-calendar-phase-07]

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - solsys_code/management/commands/sync_lco_observation_calendar.py
    - solsys_code/tests/test_sync_lco_observation_calendar.py
    - docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb

key-decisions:
  - "Used telescope label 'SOAR' (the telescope's common name) for the new SITE_TELESCOPE_MAP['sor'] entry, matching the existing FTS/FTN short-label style"
  - "Used 'SOAR_GHTS_REDCAM' (a real SOAR Goodman spectrograph instrument code) as the SOAR-identifiable instrument_type in fixtures, replacing the incorrectly-reused LCO MuSCAT code"

patterns-established: []

requirements-completed: [QUICK-SOAR-SITE-FIX]

# Metrics
duration: 25min
completed: 2026-06-19
---

# Quick Task 260619-jpr: Fix SOAR Site-Mapping Bug Summary

**Added the real SOAR site code `'sor'` to `SITE_TELESCOPE_MAP` and corrected three test fixtures plus two demo-notebook fixtures that had been silently masking the bug by reusing the LCO site code `'coj'` for SOAR records.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-06-19T20:55:00Z
- **Completed:** 2026-06-19T21:20:30Z
- **Tasks:** 3 (2 code-bearing, 1 verification-only)
- **Files modified:** 3

## Accomplishments
- `SITE_TELESCOPE_MAP` now maps `'sor' -> 'SOAR'`, so `_derive_telescope('sor')` no longer raises `KeyError`, and real SOAR `ObservationRecord`s (`parameters['site'] == 'sor'`) now produce a `CalendarEvent` instead of being silently counted under `skipped`.
- The three SOAR test fixtures (`test_select_03`, `test_select_04`, `test_select_05`) now use `site='sor'` and `instrument_type='SOAR_GHTS_REDCAM'`, so they genuinely exercise the SOAR code path — previously they passed only because they (incorrectly) used the LCO site code `'coj'`, which masked the bug.
- The demo notebook's two Phase-5 SOAR fixture cells (`demo-603002`, `demo-604002`) were corrected the same way, so the next live run produces SOAR-identifiable CalendarEvent titles (`'SOAR SOAR_GHTS_REDCAM'`) instead of the misleading `'FTS 2M0-SCICAM-MUSCAT'`.
- Confirmed `ruff check`/`ruff format --check` are clean for both modified Python files, and the full `test_sync_lco_observation_calendar` suite (19 tests) passes.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add SOAR site code to SITE_TELESCOPE_MAP and fix SOAR test fixtures** - `eec6ba8` (fix)
2. **Task 2: Fix Phase-5 SOAR fixture cells in the demo notebook** - `e25e514` (fix)
3. **Task 3: Confirm quality gates clean** - no code commit (verification-only task; produced `deferred-items.md`, a planning doc handled by the orchestrator's docs commit)

**Plan metadata:** handled by orchestrator (Step 8)

## Files Created/Modified
- `solsys_code/management/commands/sync_lco_observation_calendar.py` - Added `'sor': 'SOAR'` to `SITE_TELESCOPE_MAP`; broadened `_derive_telescope` docstring to "LCO/SOAR site code"; clarified the dict comment to mark the new entry as confirmed (not `[ASSUMED]`).
- `solsys_code/tests/test_sync_lco_observation_calendar.py` - Fixed the three `facility='SOAR'` fixtures (observation_ids `610003`, `620002`, `630001`) to pass `site='sor'`, `instrument_type='SOAR_GHTS_REDCAM'` instead of inheriting the LCO defaults.
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` - Changed the two Phase-5 SOAR fixture cells' `parameters` dicts (`demo-603002`, `demo-604002`) from `'site': 'coj'` / `'instrument_type': '2M0-SCICAM-MUSCAT'` to `'site': 'sor'` / `'instrument_type': 'SOAR_GHTS_REDCAM'`. No other cell touched; notebook was not re-executed (DB-dependent, out of scope per plan).

## Decisions Made
- Telescope label `'SOAR'` chosen for the new map entry (the telescope's common name) — consistent with `'FTS'`/`'FTN'` short-label style; there's no Faulkes-style two-telescope abbreviation for SOAR.
- `'SOAR_GHTS_REDCAM'` chosen as the concrete SOAR instrument code for fixtures (any code containing `'SOAR'` would satisfy the requirement; this is a real Goodman spectrograph code).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Generated missing setuptools_scm `_version.py` for this worktree's editable install**
- **Found during:** Task 1 verification (`./manage.py test ...`)
- **Issue:** The venv's editable install `.pth` file pointed at the main repo checkout (`/home/tlister/git/fomo_devel/src`), not this worktree. When invoked as `python manage.py ...`, Python prepends the worktree's own directory to `sys.path`, so `src.fomo` resolved to this worktree's copy, which lacked the gitignored, setuptools_scm-generated `src/fomo/_version.py` — `ModuleNotFoundError: No module named 'src.fomo._version'`.
- **Fix:** Ran `pip install --no-deps -e .` from the worktree root, which regenerated `_version.py` in this worktree and repointed the editable install here. This is a local dev-environment fix only (the generated file is gitignored); no project files were changed.
- **Files modified:** none (generated file is gitignored, not committed)
- **Verification:** `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` then ran cleanly (19/19 passed).
- **Committed in:** N/A (environment-only fix, nothing to commit)

---

**Total deviations:** 1 auto-fixed (1 blocking, environment-only)
**Impact on plan:** No scope creep — this was a local worktree environment artifact unrelated to the SOAR fix itself; no project source files were touched by the fix.

## Issues Encountered
- `ruff check .`/`ruff format --check .` (run project-wide, as a sanity check beyond the plan's literal Task 3 scope) reported 2 pre-existing findings in `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` (cell 6 unsorted imports, cell 12 a 139-char line) — both confirmed present in the notebook as of the commit immediately prior to this quick task (`fa17e2c`), in cells unrelated to the two SOAR fixture cells this task modified. Logged to `deferred-items.md` per the scope-boundary rule (only auto-fix issues directly caused by the current task's changes) rather than fixed. `ruff check`/`ruff format --check` restricted to the two Python files this task actually modified are clean, satisfying the plan's `<done>` criterion for Task 3.

## Known Stubs
None.

## Threat Flags
None — this task only corrects a site-code mapping and aligns test/demo fixtures with real data; it introduces no new network endpoints, auth paths, file access patterns, or schema changes.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The SOAR site-mapping bug is fixed and regression-guarded: the SOAR test fixtures now genuinely exercise `site='sor'`, so a future regression (removing `'sor'` from `SITE_TELESCOPE_MAP`) would fail `test_select_03/04/05` immediately.
- Phase 06 (correct-instrument-type-extraction) and Phase 07 (telescope-label API+fallback) can build on a `SITE_TELESCOPE_MAP` that now covers all three real production site codes (`'coj'`, `'ogg'`, `'sor'`).
- Deferred (out of scope): pre-existing `ruff check` findings in the demo notebook's cells 6 and 12 — see `.planning/quick/260619-jpr-fix-sync-lco-observation-calendar-soar-s/deferred-items.md`.

---
*Quick task: 260619-jpr*
*Completed: 2026-06-19*

## Self-Check: PASSED

- FOUND: solsys_code/management/commands/sync_lco_observation_calendar.py
- FOUND: solsys_code/tests/test_sync_lco_observation_calendar.py
- FOUND: docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb
- FOUND: .planning/quick/260619-jpr-fix-sync-lco-observation-calendar-soar-s/deferred-items.md
- FOUND commit: eec6ba8
- FOUND commit: e25e514
- Confirmed `'sor': 'SOAR'` present in SITE_TELESCOPE_MAP
