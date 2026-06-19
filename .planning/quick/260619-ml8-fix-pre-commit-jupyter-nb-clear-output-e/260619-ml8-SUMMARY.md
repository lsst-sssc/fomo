---
phase: quick-260619-ml8
plan: "01"
subsystem: infra
tags: [pre-commit, jupyter, ruff, sync-lco-observation-calendar]

requires: []
provides:
  - "Corrected jupyter-nb-clear-output exclude regex matching the real pre_executed notebook path"
  - "Offset-free Window (UTC) rendering in CalendarEvent description"
affects: [docs/notebooks/pre_executed regeneration, future sync_lco_observation_calendar work]

tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - .pre-commit-config.yaml
    - solsys_code/management/commands/sync_lco_observation_calendar.py

key-decisions:
  - "Left test_sync_lco_observation_calendar.py unchanged — no assertion depended on the +00:00 offset form, and existing date-substring checks (2026-09-01/2026-09-02) still pass under the new strftime format."
  - "Ran ruff format on the modified source line; ruff's formatter normalized the f-string's nested quote style (single-quoted f-string with double-quoted strftime format args) rather than the plan's literal single-quote example — functionally identical, still offset-free."

requirements-completed: []

duration: 5min
completed: 2026-06-19
---

# Quick Task 260619-ml8: Fix pre-commit jupyter-nb-clear-output exclude path + offset-free Window (UTC) description Summary

**Fixed pre-commit's mismatched exclude regex (was stripping output from pre-executed notebooks at their real `docs/notebooks/pre_executed/` location) and removed the redundant `+00:00` suffix from the calendar event description's UTC window line.**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-06-19T23:21:21Z
- **Completed:** 2026-06-19T23:26:12Z
- **Tasks:** 3 (2 code tasks + 1 verification-only task)
- **Files modified:** 2

## Accomplishments
- `.pre-commit-config.yaml`'s `jupyter-nb-clear-output` hook now excludes `^docs/notebooks/pre_executed`, matching where pre-executed notebooks actually live — stopping pre-commit from destroying their just-regenerated output (confirmed data loss this session, now prevented going forward).
- `_build_event_fields`'s `Window (UTC):` description line now renders `start_time`/`end_time` via `strftime('%Y-%m-%dT%H:%M:%S')` instead of `.isoformat()`, eliminating the redundant `+00:00` suffix on a line already labelled UTC.
- Confirmed via `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` (19 tests, all pass) and `ruff check`/`ruff format --check` on both modified files (clean) that neither fix broke anything.

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix pre-commit exclude path for pre_executed notebooks** - `3bdf118` (fix)
2. **Task 2: Render Window (UTC) description without +00:00 offset** - `1ad0be8` (fix)
3. **Task 3: Verify tests and lint pass** - verification-only, no code changes, no commit

## Files Created/Modified
- `.pre-commit-config.yaml` - `jupyter-nb-clear-output` hook's `exclude` regex corrected from `^docs/pre_executed` to `^docs/notebooks/pre_executed`
- `solsys_code/management/commands/sync_lco_observation_calendar.py` - `_build_event_fields`'s description `Window (UTC):` line now uses `strftime('%Y-%m-%dT%H:%M:%S')` instead of `.isoformat()` for both `start_time` and `end_time`

## Decisions Made
- No test file changes needed: grepped `test_sync_lco_observation_calendar.py` for `+00:00`/`isoformat` assertions and found none; the existing `test_sync_05_d05_description_contains_proposal_status_and_window` test's substring checks (`'2026-09-01'`, `'2026-09-02'`) remain valid under the new strftime format.
- Allowed `ruff format` to normalize the new f-string's quote style (it chose double quotes for the inner `strftime(...)` format-string argument, since the outer f-string itself is single-quoted) — purely cosmetic, does not change the offset-free behavior the task required.

## Deviations from Plan

None - plan executed exactly as written. The ruff-format quote-style normalization on the new f-string is a formatting-tool side effect of running the gate the plan itself required (Task 3's `ruff format --check .`), not a deviation in approach.

## Issues Encountered
- `./manage.py test` initially failed with `ModuleNotFoundError: No module named 'src.fomo._version'` — the known harmless editable-install pointer quirk documented in this task's constraints. Fixed by re-running `pip install --no-deps -e .` from the worktree root, per the documented workaround; not investigated further.
- `ruff check .` (whole-repo) reports 2 pre-existing errors, both inside `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` — out of scope per this task's constraints (that notebook is explicitly untouched here; the orchestrator regenerates it separately). Confirmed both modified files (`.pre-commit-config.yaml`, `sync_lco_observation_calendar.py`) pass `ruff check`/`ruff format --check` individually.
- `ruff format --check .` (whole-repo) reports 5 pre-existing files needing reformatting (`.planning/quick/.../verify_nb.py`, `.planning/quick/.../verify_project.py`, two `docs/notebooks/pre_executed/*.ipynb` files, `src/fomo/settings.py`) — none touched by this plan, all pre-existing and out of scope.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Both fixes are self-contained and complete; no follow-up work required for this quick task.
- The demo notebook (`docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`) was deliberately left untouched per constraints — the orchestrator regenerates its output separately now that the underlying description-rendering bug is fixed.

---
*Phase: quick-260619-ml8*
*Completed: 2026-06-19*

## Self-Check: PASSED

All claimed files exist and all claimed commit hashes are present in git history:
- FOUND: .pre-commit-config.yaml
- FOUND: solsys_code/management/commands/sync_lco_observation_calendar.py
- FOUND: .planning/quick/260619-ml8-fix-pre-commit-jupyter-nb-clear-output-e/260619-ml8-SUMMARY.md
- FOUND: 3bdf118
- FOUND: 1ad0be8
