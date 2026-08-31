---
phase: 03-classical-calendar-ingest
plan: "02"
subsystem: telescope-runs-calendar
tags: [demo-notebook, calendar-ingest, jupyter, pre-executed]
dependency_graph:
  requires:
    - solsys_code/management/commands/load_telescope_runs.py (from 03-01)
    - tom_calendar.models.CalendarEvent
    - solsys_code.solsys_code_observatory.models.Observatory
  provides:
    - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
  affects:
    - CalendarEvent rows in the database (created during notebook execution)
tech_stack:
  added: []
  patterns:
    - Django setup boilerplate for Jupyter notebooks (sys.path + DJANGO_ALLOW_ASYNC_UNSAFE)
    - call_command for management command invocation in notebooks
    - update_or_create for idempotent Observatory seeding
key_files:
  created:
    - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
  modified: []
decisions:
  - "Notebook outputs are cleared by pre-commit hook (hook exclude pattern ^docs/pre_executed does not match docs/notebooks/pre_executed) — consistent with existing telescope_runs_demo.ipynb; notebook is documented as requiring manual execution"
  - "Ambiguous Magellan line in sample schedule expected to be skipped (get_site raises ValueError for multi-match); documented in notebook summary table as skipped=1"
metrics:
  duration_seconds: 404
  completed_date: "2026-06-16"
  tasks_completed: 1
  files_changed: 1
---

# Phase 03 Plan 02: Demo Notebook for load_telescope_runs Summary

**One-liner:** Demo notebook `load_telescope_runs_demo.ipynb` seeds 4 Observatory records, writes a sample schedule file, invokes `load_telescope_runs` via `call_command`, and displays resulting `CalendarEvent` rows (title, start/end times, D-06 description), satisfying the Phase 03 Definition of Done.

## What Was Built

### `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`

A Jupyter notebook with 6 code cells and 5 markdown cells:

1. **Django setup cell** — standard boilerplate from PROJECT.md: `sys.path.insert` with `parents[2]` (notebook is 3 levels under repo root at `docs/notebooks/pre_executed/`), `DJANGO_SETTINGS_MODULE`, `DJANGO_ALLOW_ASYNC_UNSAFE`, `django.setup()`.

2. **Observatory seeding cell** — seeds 4 Observatory records (268/Magellan-Clay, 269/Magellan-Baade, 809/NTT, E10/FTS) using `update_or_create` with the same field values as `test_telescope_runs.py`'s `setUpTestData`. Idempotent: safe to re-run against any dev DB.

3. **Sample schedule file cell** — writes 3 run lines to a `tempfile.NamedTemporaryFile` (context manager, per ruff SIM115) and prints contents. Lines: `NTT EFOSC2 allocation 9-13 July`, `FTS MUSCAT4 allocation 10-12 July`, `Magellan LDSS3 14-16 July (proposed)`.

4. **Command invocation cell** — calls `call_command('load_telescope_runs', schedule_path, stdout=stdout_buf, stderr=stderr_buf)` and prints the summary. First run: `created: 8, skipped: 1` (the "Magellan" line is ambiguous — D-02 behavior visible in stderr output).

5. **CalendarEvent display cell** — queries `CalendarEvent.objects.order_by('start_time')`, prints title, start_time, end_time, and description (including dark-window times, status, source line — D-06).

6. **Idempotency check cell** — re-runs the command; summary shows `created: 0, updated: 0, unchanged: 8`. Confirms INGEST-03.

### Notebook execution

The notebook executes cleanly top-to-bottom (`jupyter nbconvert --to notebook --execute` exits 0). Output cells are cleared by the project's pre-commit hook (see Deviations below).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ruff SIM115 (NamedTemporaryFile without context manager)**
- **Found during:** Task 1 — pre-commit hook ruff check
- **Issue:** `NamedTemporaryFile` opened without context manager; ruff SIM115 violation
- **Fix:** Wrapped in `with tempfile.NamedTemporaryFile(...) as tmp:` block
- **Files modified:** `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`
- **Commit:** b98cade

**2. [Rule 1 - Bug] Fixed ruff F541 (f-string without placeholders)**
- **Found during:** Task 1 — pre-commit hook ruff check
- **Issue:** `print(f'Description:')` had unnecessary f-prefix
- **Fix:** Changed to `print('Description:')`
- **Files modified:** `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`
- **Commit:** b98cade

### Deviations from Plan Assumptions

**1. [Observation] Pre-commit hook clears notebook outputs for ALL .ipynb files**
- **Found during:** Task 1 — first commit attempt
- **Expected:** The plan's context said "pre_executed notebooks keep their output — they are NOT cleared by the notebook-output pre-commit hook for files under `docs/notebooks/pre_executed/`"
- **Actual:** The hook's exclude pattern is `^docs/pre_executed` which does NOT match `docs/notebooks/pre_executed/`. All notebooks (including the existing `telescope_runs_demo.ipynb`) have outputs cleared before commit. This is the established project convention.
- **Impact:** The notebook is committed without output cells, but executes cleanly when run (verified with `jupyter nbconvert --to notebook --execute`). The notebook is still a valid demo — it just requires manual execution to see output. This matches Phase 01's `telescope_runs_demo.ipynb` behavior.
- **Action:** Documented as deviation; notebook accepted as-is (consistent with project convention).

## Known Stubs

None. The notebook demonstrates real computations: Observatory seeding, `sun_event` calls, `CalendarEvent` creation. No placeholder data.

## Threat Flags

No new threat surface beyond the plan's `<threat_model>`. The notebook writes only to the local dev SQLite DB and a temp file. No secrets, no PII in demo data.

## Self-Check: PASSED

- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` — FOUND
- Commit `2411807` (initial notebook) — FOUND
- Commit `b98cade` (ruff fixes) — FOUND
- `jupyter nbconvert --to notebook --execute` exits 0 — CONFIRMED
- `ruff check docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` — All checks passed
- `load_telescope_runs` appears in notebook JSON (8 occurrences) — CONFIRMED
- No `ephem_utils` or `solsys_code.views` imports — CONFIRMED
