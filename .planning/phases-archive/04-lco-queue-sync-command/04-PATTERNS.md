# Phase 4: LCO Queue Sync Command - Pattern Map

**Mapped:** 2026-06-17
**Files analyzed:** 2 (1 new management command, 1 new test file)
**Analogs found:** 2 / 2

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `solsys_code/management/commands/sync_lco_observation_calendar.py` | controller (management command) | CRUD (upsert) | `solsys_code/management/commands/load_telescope_runs.py` | exact |
| `solsys_code/tests/test_sync_lco_observation_calendar.py` | test | CRUD/request-response (call_command) | `solsys_code/tests/test_load_telescope_runs.py` | exact |
| (optional) site/telescope label map constant, e.g. module-level `SITE_TELESCOPE_MAP` in the new command file | config/utility | transform | `solsys_code/telescope_runs.py:SITES` (lines 17-22) | role-match |

No new Django models/migrations are needed for this phase (per CONTEXT.md) — `CalendarEvent` already has the required fields.

## Pattern Assignments

### `solsys_code/management/commands/sync_lco_observation_calendar.py` (controller, CRUD/upsert)

**Analog:** `solsys_code/management/commands/load_telescope_runs.py` (full file, 126 lines — read in one pass)

**Imports pattern** (lines 1-9):
```python
from datetime import date, timedelta
from datetime import timezone as dt_timezone
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser
from tom_calendar.models import CalendarEvent

from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import ParsedRun, get_site, parse_run_line, sun_event
```
For Phase 4, swap the telescope_runs-specific imports for:
```python
from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Any

from django.core.management.base import BaseCommand, CommandParser
from tom_calendar.models import CalendarEvent
from tom_observations.facilities.lco import LCOFacility
from tom_observations.models import ObservationRecord
```

**`add_arguments` pattern** (lines 36-43):
```python
def add_arguments(self, parser: CommandParser) -> None:
    """Parse command line arguments."""
    parser.add_argument(
        'filepath',
        type=str,
        help='Path to a text file of classical run lines (one per line)',
    )
    # No return statement — BaseCommand.add_arguments() returns None
```
Mirror this exactly but with `--proposal` as a required named argument (per CONTEXT.md SELECT-01: `--proposal <code>`), e.g. `parser.add_argument('--proposal', type=str, required=True, help=...)`.

**Class docstring + `help` attribute** (lines 31-34):
```python
class Command(BaseCommand):
    """Load classical telescope run lines from a file and create or update CalendarEvents."""

    help = 'Load classical telescope run lines from a file and create/update CalendarEvents'
```

**Core upsert/CRUD pattern — counters + per-item loop + conditional save** (lines 56-124):
```python
created_count = 0
updated_count = 0
unchanged_count = 0
skipped_count = 0
...
for line_num, line in enumerate(file_lines, start=1):
    ...
    try:
        ...
        event, created = CalendarEvent.objects.get_or_create(
            telescope=parsed.telescope,
            instrument=parsed.instrument,
            start_time=start_time,
            defaults={
                'end_time': end_time,
                'title': title,
                'description': description,
            },
        )
        if created:
            created_count += 1
        else:
            changed = event.end_time != end_time or event.title != title or event.description != description
            if changed:
                event.end_time = end_time
                event.title = title
                event.description = description
                event.save()
                updated_count += 1
            else:
                unchanged_count += 1
    except (ValueError, Observatory.DoesNotExist) as exc:
        self.stderr.write(f'Line {line_num}: {exc} (line text: {line.strip()!r})')
        skipped_count += 1
        continue

self.stdout.write(
    f'Done. lines processed: {lines_processed}, '
    f'created: {created_count}, '
    f'updated: {updated_count}, '
    f'unchanged: {unchanged_count}, '
    f'skipped: {skipped_count}'
)
return
```

**Adaptation for Phase 4 (per RESEARCH.md Pattern 1, exact analog already given there):**
- Replace the `get_or_create` key from `(telescope, instrument, start_time)` to `url=url` (per D-01/SYNC-01) — `url` built once per record via `LCOFacility().get_observation_url(record.observation_id)`.
- Replace the per-item `except (ValueError, Observatory.DoesNotExist)` with whatever exceptions are appropriate for malformed/missing `parameters` keys (e.g. `KeyError`, `ValueError` from `datetime.fromisoformat`) — same per-item try/except/skip/continue shape, same `skipped_count` counter, same `self.stderr.write(...)` reporting style.
- The query loop iterates `ObservationRecord.objects.filter(facility='LCO', parameters__proposal=proposal_code)` instead of file lines — see RESEARCH.md "Pattern 2" for the exact ORM call.
- Compare all 7 changeable fields (`title`, `description`, `start_time`, `end_time`, `telescope`, `instrument`, `proposal`) before `.save()`, per RESEARCH.md Pattern 1's fuller example (this analog only compares 3 fields since fewer vary in Phase 3; Phase 4 needs the longer comparison block).

**Error handling pattern** (lines 62-66, for fatal/setup-level errors — adapt the *style*, not the literal file-open logic):
```python
try:
    with open(filepath, encoding='utf-8') as f:
        file_lines = list(f)
except OSError as exc:
    raise CommandError(f'Cannot open schedule file {filepath!r}: {exc}') from exc
```
Phase 4 has no file to open, but this shows the project convention: raise `CommandError` for command-level setup failures (e.g. if `--proposal` matches zero records — RESEARCH.md's Pitfall/Security note recommends reporting `created: 0` rather than erroring, so prefer the stdout-summary approach over `CommandError` for the zero-match case specifically).

**Stdout summary pattern** (lines 118-124) — same f-string multi-counter style, update labels to match Phase 4's counters (`created`, `updated`, `unchanged`, `terminal-marked` if tracked separately, `skipped`).

---

### `solsys_code/tests/test_sync_lco_observation_calendar.py` (test)

**Analog:** `solsys_code/tests/test_load_telescope_runs.py` (full file, 154 lines — read in one pass)

**Imports + TestCase class pattern** (lines 1-12):
```python
import io
import pathlib
import tempfile

from django.core.management import call_command
from django.test import TestCase
from tom_calendar.models import CalendarEvent

from solsys_code.solsys_code_observatory.models import Observatory


class TestLoadTelescopeRuns(TestCase):
```
For Phase 4, swap `pathlib`/`tempfile`/`Observatory` imports for `ObservationRecord` and target factory imports (no temp files needed — fixtures are `ObservationRecord.objects.create(...)` calls directly, per the official factory shape shown in RESEARCH.md's Code Examples section).

**`setUpTestData` fixture pattern** (lines 13-49) — class-level fixture dict iterated with `update_or_create`:
```python
@classmethod
def setUpTestData(cls) -> None:
    for obscode, fields in {
        '268': dict(name=..., short_name=..., lat=..., lon=..., altitude=..., timezone=...),
        ...
    }.items():
        Observatory.objects.update_or_create(obscode=obscode, defaults=fields)
```
Phase 4 equivalent: create one or more `ObservationRecord` fixtures directly in `setUpTestData` (or per-test, since each test likely needs different `parameters`/`status`/`scheduled_start` combinations — unlike Phase 3 where the Observatory fixtures are identical across all tests). Use the field shape confirmed in RESEARCH.md's "Official factory fixture" section (`facility='LCO'`, `parameters` dict with `proposal`/`start`/`end`/`instrument_type`/`site` keys).

**`call_command` invocation pattern with captured stdout/stderr** (e.g. lines 69, 76, 87, 107, 109, 119, 125, 147):
```python
call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
```
Phase 4: `call_command('sync_lco_observation_calendar', '--proposal', proposal_code, stdout=io.StringIO(), stderr=io.StringIO())`.

**No-churn / idempotency test pattern** (lines 114-136) — directly reusable structure for SYNC-04:
```python
def test_unchanged_rerun_does_not_update_existing_rows(self):
    """D-04: a re-run with unchanged schedule leaves modified timestamps untouched and reports updated: 0."""
    ...
    stdout1 = io.StringIO()
    call_command(..., stdout=stdout1, stderr=io.StringIO())
    modified_before = {e.pk: e.modified for e in CalendarEvent.objects.all()}

    stdout2 = io.StringIO()
    call_command(..., stdout=stdout2, stderr=io.StringIO())
    for event in CalendarEvent.objects.all():
        self.assertEqual(event.modified, modified_before[event.pk], ...)
    summary = stdout2.getvalue()
    self.assertIn('updated: 0', summary)
```

**Field-assertion test pattern** (lines 83-101) — assert `event.telescope`/`event.instrument`/`event.title` and substring-check `event.description` for required pieces (mirrors D-05's "three pieces of information" requirement):
```python
self.assertEqual(event.telescope, 'NTT')
self.assertEqual(event.instrument, 'EFOSC2')
self.assertEqual(event.title, 'NTT EFOSC2')
desc = event.description
self.assertIn('T', desc, 'Expected ISO datetime string in description for dark-window time')
self.assertIn('allocation', desc)
self.assertIn('NTT EFOSC2 allocation 9-13 July', desc)
```
Phase 4 equivalent assertions: proposal code in description, `record.status` in description, time window in description; title prefix logic per state (`[QUEUED]`, `[EXPIRED]`, `[CANCELLED]`, `[FAILED]`, clean for `COMPLETED`/placed).

**Per-item error/skip test pattern** (lines 137-153) — write a malformed/unmappable fixture, assert it's skipped via stderr message and other records still process:
```python
def test_unparseable_line_logged_and_skipped(self):
    ...
    err = stderr_buf.getvalue()
    self.assertIn('2', err, 'Expected line number in stderr error message')
    self.assertIn('Magellan IMACS 13-19 July (proposed)', err)
```
Phase 4 equivalent: an `ObservationRecord` with a missing/unrecognized `parameters['site']` key (or malformed `parameters['start']`) should be skipped with a stderr message identifying the record (e.g. by `observation_id`), while other matching records still sync.

---

### Telescope-label map constant (config/utility, transform)

**Analog:** `solsys_code/telescope_runs.py:SITES` (lines 17-22):
```python
SITES = {
    'Magellan-Clay': '268',
    'Magellan-Baade': '269',
    'NTT': '809',
    'FTS': 'E10',
}
```
Phase 4's site→telescope map is the *inverse* direction (LCO site code → telescope label, not telescope name → MPC obscode) and uses a different vocabulary (LCO 3-letter codes like `'coj'`/`'ogg'`, not MPC obscodes), but should follow the same flat-dict-of-strings convention and produce labels consistent with `SITES`' keys (e.g. `'FTS'`) per D-02. Per RESEARCH.md, this can live as a module-level dict at the top of the new command file (no new module needed), e.g.:
```python
SITE_TELESCOPE_MAP = {
    'coj': 'FTS',  # Siding Spring — [ASSUMED], confirm against real ObservationRecord.parameters['site'] (RESEARCH.md A1/A2)
    'ogg': 'FTN',  # Haleakala — [ASSUMED]
}
```

---

## Shared Patterns

### Per-item try/except/skip/continue (error isolation)
**Source:** `solsys_code/management/commands/load_telescope_runs.py` lines 68-116
**Apply to:** The new command's per-`ObservationRecord` loop body — catch `KeyError`/`ValueError` (missing `parameters` keys, malformed ISO datetime strings, unmapped site code) per record, write to `self.stderr`, increment `skipped_count`, `continue` — never abort the whole command run on one bad record.

### Conditional-save upsert (no `modified` churn)
**Source:** `solsys_code/management/commands/load_telescope_runs.py` lines 91-112; generalized in RESEARCH.md "Pattern 1" (lines 167-205 of 04-RESEARCH.md) with the full 7-field comparison Phase 4 needs.
**Apply to:** The `CalendarEvent.objects.get_or_create(url=url, defaults={...})` block — compare every field that can legitimately change (`title`, `description`, `start_time`, `end_time`, `telescope`, `instrument`, `proposal`) before calling `.save()`.

### Stdout summary line
**Source:** `solsys_code/management/commands/load_telescope_runs.py` lines 118-124
**Apply to:** End of `handle()` — single `self.stdout.write(f'Done. ... created: {n}, updated: {n}, unchanged: {n}, skipped: {n}')` line; tests assert on substrings of this (e.g. `'updated: 0'`).

### Google-style docstrings with `Args`/`Returns`/`Raises`
**Source:** `solsys_code/management/commands/load_telescope_runs.py` lines 12-23, 46-54
**Apply to:** Both the command's `handle()` method and any helper functions (e.g. a `_derive_telescope(parameters)` or `_event_fields_for(record)` helper, mirroring `_iter_run_nights()`'s docstring shape).

### Django TestCase with `call_command` + captured `io.StringIO()` stdout/stderr
**Source:** `solsys_code/tests/test_load_telescope_runs.py` (whole file)
**Apply to:** The new test module — same `call_command(...)`/`io.StringIO()` capture idiom for every test, same per-scenario test method naming (`test_<behavior>_<expected_outcome>`).

## No Analog Found

None — both required new files (command + test) have exact-match analogs already identified by CONTEXT.md/RESEARCH.md, and the supporting label-map constant has a role-match analog (`telescope_runs.py:SITES`).

## Metadata

**Analog search scope:** `solsys_code/management/commands/`, `solsys_code/tests/`, `solsys_code/telescope_runs.py` (all explicitly named in CONTEXT.md canonical_refs)
**Files scanned:** 3 (`load_telescope_runs.py`, `test_load_telescope_runs.py`, `telescope_runs.py`)
**Pattern extraction date:** 2026-06-17
