---
phase: 10-gemini-calendar-sync-command
plan: "01"
subsystem: calendar-sync
tags:
  - management-command
  - gemini
  - calendar-event
  - no-churn
  - security
dependency_graph:
  requires:
    - tom_calendar.CalendarEvent
    - tom_observations.ObservationRecord
    - settings.FACILITIES.GEM.programs
  provides:
    - sync_gemini_observation_calendar management command
    - test suite covering all 10 GEM-* requirements
    - pre-executed demo notebook (4 scenarios)
  affects:
    - FOMO calendar (CalendarEvent rows keyed GEM:{prog}/{observation_id})
tech_stack:
  added: []
  patterns:
    - get_or_create(url=) + update_fields=changed no-churn idiom
    - safe_params password-strip at loop start (D-04)
    - ToO-type window derivation from settings FACILITIES prefix (Std:/Rap:)
    - "@override_settings(FACILITIES=GEM_SETTINGS) TestCase pattern"
key_files:
  created:
    - solsys_code/management/commands/sync_gemini_observation_calendar.py
    - solsys_code/tests/test_sync_gemini_observation_calendar.py
    - docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb
  modified: []
decisions:
  - "Used update_fields=changed (list of field names) for no-churn save rather than full save — prevents modifying CalendarEvent.modified on unchanged fields while satisfying GEM-NOCHURN-01"
  - "safe_params strips password key as first statement in each loop iteration, before any logging or exception paths (D-04)"
  - "site_key/telescope determination placed BEFORE the try/except block so KeyError from obsid lookup never risks referencing an undefined site_key in the except clause"
  - "Raw-fallback branch (GEM-INSTR-01): explicit window + unknown obs code -> instrument = obs_code; D-01 skip path only applies when no explicit window is present"
  - "Demo notebook (deviation) covers all four D-06 scenarios with executed output; added as CLAUDE.md-mandated deliverable"
metrics:
  duration: "~45 minutes"
  completed: "2026-06-26"
  tasks_completed: 2
  files_created: 3
---

# Phase 10 Plan 01: Sync Gemini Observation Calendar Summary

**One-liner:** `sync_gemini_observation_calendar` management command syncing GEM ObservationRecords to CalendarEvents with per-record password scrubbing, ToO-type window derivation from `FACILITIES['GEM']['programs']`, and no-churn `get_or_create(url=) + save(update_fields=changed)` idiom.

## Tasks Completed

| # | Name | Commit | Files |
|---|------|--------|-------|
| 1 | Write failing test suite | e69dc6b | solsys_code/tests/test_sync_gemini_observation_calendar.py |
| 2 | Implement the command (GREEN) | 55a2e48 | solsys_code/management/commands/sync_gemini_observation_calendar.py |
| D | Demo notebook (CLAUDE.md deviation) | 1e8f5a0 | docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb |

## Verification Results

- `./manage.py test solsys_code.tests.test_sync_gemini_observation_calendar`: **15/15 tests pass**
- `ruff check` on both Python files: **0 errors**
- `ruff format --check` on command file: **clean**
- `grep record.parameters` in command: only the `safe_params` comprehension line + one comment
- `grep update_fields=changed` in command: exactly one occurrence (no unconditional full save)

## Deviations from Plan

### Auto-added: Demo notebook

**[Rule 2 - CLAUDE.md] Added mandatory demo notebook**
- **Found during:** Post-task review
- **Issue:** CLAUDE.md mandates a paired pre-executed demo notebook for every new management command module. The plan's `files_modified` did not include `sync_gemini_observation_calendar_demo.ipynb` and no notebook task was in the plan.
- **Fix:** Created `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb` covering all four D-06 scenarios (explicit window, Rap: derived, Std: derived, ON_HOLD + idempotent re-run) with executed output. Ran `jupyter nbconvert --to notebook --execute --inplace` and committed with output.
- **Commit:** 1e8f5a0

### Implementation refinement: site_key placement

**[Rule 1 - Bug prevention] site_key determination placed before try/except**
- **Found during:** Task 2 implementation
- **Issue:** The PATTERNS.md skeleton placed all loop body code inside `try`, but `site_key` must be defined before the `except` clause references it (to increment `counters[site_key]['skipped']`). A `KeyError` on `safe_params['obsid']` before `site_key` was set would cause a `NameError` in the except handler.
- **Fix:** `safe_params` construction, `prog` extraction, and the `if prog.startswith(...)` branch that sets `site_key` and `telescope` are placed BEFORE the `try` block. Only the `obsid`/instrument/window/get_or_create logic is inside `try`.
- **Files modified:** `solsys_code/management/commands/sync_gemini_observation_calendar.py`

### Environment note: worktree _version.py

- The venv's editable install `.pth` file pointed to a non-existent worktree, causing `ModuleNotFoundError: No module named 'fomo'` when running tests. Fix: copied `_version.py` from the main repo to the worktree's `src/fomo/` and used `PYTHONPATH=$WT_ROOT/src` to run tests. This is a worktree environment issue, not a code issue. The `_version.py` file is gitignored and was not committed.

## Known Stubs

None. All CalendarEvent fields are populated from real data in every code path.

## Threat Flags

No new threat surface introduced. The command reads from trusted DB rows (ObservationRecord) and writes to an internal-only CalendarEvent model. Threat mitigations T-10-01 and T-10-02 from the plan's threat register are fully implemented and verified by GEM-SECURE-01 tests.

## Self-Check: PASSED

- FOUND: `solsys_code/management/commands/sync_gemini_observation_calendar.py`
- FOUND: `solsys_code/tests/test_sync_gemini_observation_calendar.py`
- FOUND: `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb`
- FOUND: commit e69dc6b (test suite)
- FOUND: commit 55a2e48 (command)
- FOUND: commit 1e8f5a0 (notebook)
