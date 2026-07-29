---
phase: 03-classical-calendar-ingest
reviewed: 2026-06-16T16:14:33Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - solsys_code/management/commands/load_telescope_runs.py
  - solsys_code/tests/test_load_telescope_runs.py
  - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
findings:
  critical: 1
  warning: 4
  info: 2
  total: 7
status: issues_found
---

# Phase 03: Code Review Report

**Reviewed:** 2026-06-16T16:14:33Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Three files reviewed: the management command (`load_telescope_runs.py`), its Django test suite
(`test_load_telescope_runs.py`), and a pre-executed demo notebook. The command's core logic is
sound — `get_or_create` + conditional `save()` correctly avoids churning `modified` timestamps on
unchanged re-runs. However, one critical bug (unchecked `open()` failure with no user-facing error),
four warnings (tempdir leak in every test, `add_arguments` redundant `super()` return, no encoding on
`open()`, and a fragile notebook `cwd`-dependent path), and two info items (minor test reliability
and doc accuracy) were found.

## Critical Issues

### CR-01: `open(filepath)` raises an unhandled `OSError`; the user sees a raw Python traceback

**File:** `solsys_code/management/commands/load_telescope_runs.py:58`

**Issue:** `with open(filepath) as f:` is not wrapped in any `try/except`. If `filepath` does not
exist, is not readable, or is a directory, Python raises `OSError` / `FileNotFoundError`. Because
this is a management command, Django will print the raw traceback and exit with status 1 — but with
no human-readable error message. The project convention for management commands is to write a clear
message to `self.stderr` and call `raise CommandError(...)` so that Django exits cleanly with a
formatted error. This is especially important for scripted / cron use where the raw traceback may
not be monitored.

**Fix:**

```python
from django.core.management.base import BaseCommand, CommandError, CommandParser

# in handle():
try:
    f_handle = open(filepath, encoding='utf-8')
except OSError as exc:
    raise CommandError(f'Cannot open schedule file {filepath!r}: {exc}') from exc

with f_handle:
    for line_num, line in enumerate(f_handle, start=1):
        ...
```

## Warnings

### WR-01: Every test leaks a temporary directory — `mkdtemp()` is never cleaned up

**File:** `solsys_code/tests/test_load_telescope_runs.py:54-57`

**Issue:** `_write_schedule_file` creates a temp directory with `tempfile.mkdtemp()`, writes the
schedule file into it, and returns the file path. Every test `finally` block calls `os.unlink(path)`
to delete the file, but the parent directory created by `mkdtemp()` is **never removed**. Six
temporary directories accumulate in `/tmp` per test run. On a CI host that runs tests many times
this is a slow resource leak; on a host where `/tmp` is a `tmpfs`, it occupies memory.

**Fix:** Use `tempfile.TemporaryDirectory` as a context manager so cleanup is guaranteed:

```python
import tempfile

def _write_schedule_file(self, lines: list[str]) -> tuple[str, tempfile.TemporaryDirectory]:
    tmpdir_ctx = tempfile.TemporaryDirectory()
    path = pathlib.Path(tmpdir_ctx.name) / 'schedule.txt'
    path.write_text('\n'.join(lines) + '\n')
    return str(path), tmpdir_ctx
```

Then in each test:

```python
path, tmpdir_ctx = self._write_schedule_file([...])
with tmpdir_ctx:
    call_command(...)
    self.assertEqual(...)
```

Or simpler: replace `mkdtemp()` with `tempfile.mkstemp(suffix='.txt')` so only a single file
(no wrapping directory) is created, and `os.unlink` is sufficient.

### WR-02: `add_arguments` returns the result of `super().add_arguments(parser)`, which is `None` — the `return` is harmless but misleading and triggers a ruff warning

**File:** `solsys_code/management/commands/load_telescope_runs.py:43`

**Issue:** `BaseCommand.add_arguments` is typed `-> None` and returns `None`. The override calls
`return super().add_arguments(parser)`, which propagates `None`. The method's own declared return
type is `None`. Returning a value from a `-> None` function is a ruff `PYI015` / B012-class smell
and can confuse readers into thinking the return value is meaningful. Django's own convention and
every other management command in this codebase omit the `return`.

**Fix:**

```python
def add_arguments(self, parser: CommandParser) -> None:
    parser.add_argument(
        'filepath',
        type=str,
        help='Path to a text file of classical run lines (one per line)',
    )
    # No return statement — BaseCommand.add_arguments() returns None
```

### WR-03: `open(filepath)` uses the platform default encoding instead of `'utf-8'`

**File:** `solsys_code/management/commands/load_telescope_runs.py:58`

**Issue:** `open(filepath)` without an explicit `encoding=` argument uses the system locale encoding
(e.g. `latin-1` on some CI hosts, `utf-8` on others). Schedule files that contain non-ASCII
characters (accented PI names, telescope names like "Gunn-R", or any text editor that emits a BOM)
will either silently misparse or raise a `UnicodeDecodeError` on platforms where the locale is not
`utf-8`. PEP 597 / ruff `UP015` flags this pattern.

**Fix:**

```python
with open(filepath, encoding='utf-8') as f:
```

### WR-04: Notebook `sys.path` injection is `cwd`-dependent and silently breaks when the kernel starts from a non-`pre_executed` directory

**File:** `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` (Django setup cell)

**Issue:** The notebook resolves the repo root as `Path.cwd().resolve().parents[2]`. This is correct
only when the Jupyter kernel's working directory is exactly `docs/notebooks/pre_executed/`. If the
kernel starts from the repo root (the common case when launching `jupyter notebook` from the repo
root), `parents[2]` resolves to `/home/<user>` (three levels above the repo root) and
`src.fomo.settings` will not be importable. The notebook will then fail silently at `django.setup()`
with an `ImportError` that is easy to misdiagnose.

**Fix:** Use `__file__` (the notebook's own absolute path) to anchor the repo root instead of
`cwd`. In a Jupyter notebook cell, `__file__` is not defined, so the canonical approach is:

```python
import os, sys
from pathlib import Path

# __file__ is not available in Jupyter; use the notebook path via IPython:
try:
    _nb_path = Path(get_ipython().run_line_magic('run', '-n __file__'))
except Exception:
    _nb_path = Path.cwd()

# Alternatively, hardcode relative to the known notebook location:
_here = Path(os.path.abspath(''))   # cwd at cell-exec time
# Resolve upward until we find manage.py (repo root marker):
_repo_root = _here
for _ in range(6):
    if (_repo_root / 'manage.py').exists():
        break
    _repo_root = _repo_root.parent

if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
```

The simplest safe fix is to add a comment warning users to start Jupyter from `docs/notebooks/pre_executed/` and add an assertion:

```python
repo_root_path = Path.cwd().resolve().parents[2]
assert (repo_root_path / 'manage.py').exists(), (
    f'Repo root not found at {repo_root_path}. '
    'Run Jupyter from docs/notebooks/pre_executed/ or adjust parents[] index.'
)
```

## Info

### IN-01: Test `test_unchanged_rerun_does_not_update_existing_rows` relies on `CalendarEvent.modified` but the model's `auto_now=True` field is only updated on `save()` — the test is correct but implicitly depends on an external model detail

**File:** `solsys_code/tests/test_load_telescope_runs.py:116-138`

**Issue:** The test reads `event.modified` after the first run and checks it is identical after the
second (unchanged) run. This correctly validates the D-04 requirement. However, `modified` is a
Django `auto_now=True` field defined in `tom_calendar`'s `CalendarEvent` — an external dependency.
If `tom_calendar` ever changes `modified` semantics (e.g. moves to a trigger-based timestamp), the
test will silently stop exercising D-04. A brief inline comment naming the `auto_now=True`
dependency would make the assumption explicit and aid future maintainers.

**Fix:** Add a one-line comment:

```python
# CalendarEvent.modified has auto_now=True; it only updates on .save().
# If unchanged, the command must NOT call .save(), so modified stays constant.
modified_before = {e.pk: e.modified for e in CalendarEvent.objects.all()}
```

### IN-02: Notebook claims "5 NTT + 3 FTS + 3 Magellan events created" in the summary table, but the Magellan line is ambiguous and should be skipped (0 events)

**File:** `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` (summary table cell)

**Issue:** The sample schedule contains `'Magellan LDSS3 14-16 July (proposed)'`. The telescope
token `'Magellan'` is an ambiguous prefix that matches both `Magellan-Clay` and `Magellan-Baade`,
so `_resolve_telescope` raises `ValueError` and the line is skipped. The notebook summary table
states "5 NTT + 3 FTS + 3 Magellan events created" which is factually wrong — the Magellan line
produces 0 events and is logged to stderr. The total should be 8 events (5 NTT + 3 FTS), not 11.
This inconsistency will confuse readers who look at actual output vs. the documentation claim.

**Fix:** Either change the sample schedule to use `'Magellan-Clay LDSS3 14-16 July (proposed)'`
(so 3 Magellan events are actually created and the table is accurate), or correct the summary table
to read "5 NTT + 3 FTS events created; Magellan line skipped (ambiguous telescope name)".

---

_Reviewed: 2026-06-16T16:14:33Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
