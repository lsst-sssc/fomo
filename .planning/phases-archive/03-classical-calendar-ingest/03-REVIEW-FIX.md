---
phase: 03-classical-calendar-ingest
fixed_at: 2026-06-16T17:00:00Z
fix_scope: critical_warning
findings_in_scope: 5
fixed: 5
skipped: 0
iteration: 1
status: all_fixed
---

# Phase 03: Code Review Fix Report

## Summary

All 5 in-scope findings (1 Critical + 4 Warning) were fixed across 4 atomic commits. Info findings
IN-01 and IN-02 were out of scope for this pass (critical_warning scope).

## Fixes Applied

### WR-02: Remove redundant `return` from `add_arguments`

**Status:** Fixed
**Commit:** 46535b7
**Change:** Removed `return super().add_arguments(parser)` from `add_arguments` in
`load_telescope_runs.py`. Replaced with a clarifying comment. `BaseCommand.add_arguments()` returns
`None`; returning it was misleading.

### CR-01 + WR-03: Handle `OSError` from `open()` and add `encoding='utf-8'`

**Status:** Fixed (combined — both target the same `open()` call)
**Commit:** 5b3a2f5
**Change:** Added `CommandError` import to `load_telescope_runs.py`. Wrapped
`open(filepath, encoding='utf-8')` in `try/except OSError` that raises `CommandError` with a
human-readable message. Restructured to read the full file into `file_lines` first (satisfies ruff
`SIM115` while keeping the `except` around the actual `open()` call).

### WR-01: Replace `mkdtemp()` with `TemporaryDirectory` context manager

**Status:** Fixed
**Commit:** 65c0f3e
**Change:** Replaced `tempfile.mkdtemp()` + `os.unlink()` pattern in `test_load_telescope_runs.py`
with `tempfile.TemporaryDirectory` context manager. `_write_schedule_file` now returns
`tuple[str, tempfile.TemporaryDirectory]`; every test uses `with tmpdir_ctx:` instead of
`try/finally`. Removed the now-unused `import os`. Pre-commit test suite passed.

### WR-04: Add `assert` guard for repo root in notebook Django setup cell

**Status:** Fixed
**Commit:** 5d6d8b2
**Change:** Added `assert (repo_root_path / 'manage.py').exists()` guard in the notebook's Django
setup cell in `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`, with a message directing
users to launch Jupyter from `docs/notebooks/pre_executed/` if the assertion fails.

## Skipped Findings

None — all in-scope findings were fixed.

---

_Fixed: 2026-06-16_
_Fixer: Claude (gsd-code-fixer)_
_Scope: critical_warning_
