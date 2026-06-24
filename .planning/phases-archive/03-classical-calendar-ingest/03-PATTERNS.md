# Phase 3: Classical Calendar Ingest - Pattern Map

**Mapped:** 2026-06-13
**Files analyzed:** 2
**Analogs found:** 2 / 2

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|-----------------|----------------|
| `solsys_code/management/commands/load_telescope_runs.py` | management command (controller-like) | file-I/O + CRUD (upsert) | `solsys_code/management/commands/fetch_jplsbdb_objects.py` | role-match |
| `solsys_code/tests/test_load_telescope_runs.py` | test | request-response / DB integration | `solsys_code/tests/test_telescope_runs.py` | exact |

## Pattern Assignments

### `solsys_code/management/commands/load_telescope_runs.py` (management command, file-I/O + CRUD)

**Analog:** `solsys_code/management/commands/fetch_jplsbdb_objects.py`

**Imports pattern** (lines 1-6):
```python
from typing import Any

from django.core.management.base import BaseCommand, CommandParser
from tom_targets.models import TargetList

from solsys_code.views import JPLSBDBQuery
```
For Phase 3, adapt to:
```python
from datetime import date, timedelta, timezone as dt_timezone
from typing import Any

from django.core.management.base import BaseCommand, CommandParser
from tom_calendar.models import CalendarEvent

from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import get_site, parse_run_line, sun_event
```

**Command class / docstring + help pattern** (lines 9-14):
```python
class Command(BaseCommand):
    """
    Fetch new objects matching filter criteria from JPL SBDB and make Targets from the new ones
    """

    help = 'Fetch new objects matching filter criteria from JPL SBDB and make Targets from the new ones'
```
Mirror this docstring/`help` style verbatim for the new command (one-line Google-style class docstring + matching `help` string), per D-01.

**`add_arguments` pattern** (lines 16-39):
```python
def add_arguments(self, parser: CommandParser) -> None:
    """Parse command line arguments"""
    parser.add_argument(
        '--orbit_class',
        action='store',
        type=str,
        default=None,
        help='Orbital constraints as a comma separated string',
    )
    ...
    return super().add_arguments(parser)
```
Phase 3 uses a **positional** arg instead of `--flag` options (per D-01 and RESEARCH.md Pattern 1):
```python
def add_arguments(self, parser: CommandParser) -> None:
    """Parse command line arguments"""
    parser.add_argument(
        'filepath',
        type=str,
        help='Path to a text file of classical run lines (one per line)',
    )
    return super().add_arguments(parser)
```

**`handle()` / stdout/stderr reporting pattern** (lines 41-70):
```python
def handle(self, *args: Any, **options: Any) -> str | None:
    """Make task run when doing `python manage.py fetch_jplsbdb_objects`

    :return: Status value
    :rtype: str | None
    """
    ...
    msg = f'Querying JPL SBDB for new objects with constraints= {new_objects.orbit_class}, ...'
    self.stdout.write(msg)
    results = new_objects.run_query()
    if results is not None:
        ...
        self.stdout.write(f'Created {len(new_targets)} new Targets')
        ...
    else:
        self.stderr.write('Error running query')

    return
```
Carry over: `self.stdout.write(...)` for progress/summary, `self.stderr.write(...)` for errors, return type `str | None`, trailing bare `return`. For Phase 3's per-line error reporting (D-02), use `self.stderr.write(f'Line {line_num}: {exc} (line text: {line.strip()!r})')` and accumulate counters for the end-of-run summary written via `self.stdout.write(...)`.

**get_or_create / counting pattern** — adapt the `TargetList.objects.get_or_create(...)` idiom (lines 64-66) to `CalendarEvent.objects.get_or_create(...)` per RESEARCH.md Pattern 4 (D-03/D-04):
```python
event, created = CalendarEvent.objects.get_or_create(
    telescope=parsed.telescope,
    instrument=parsed.instrument,
    start_time=start_time,
    defaults={'end_time': end_time, 'title': title, 'description': description},
)
if created:
    created_count += 1
else:
    changed = (
        event.end_time != end_time or event.title != title or event.description != description
    )
    if changed:
        event.end_time = end_time
        event.title = title
        event.description = description
        event.save()
        updated_count += 1
    else:
        unchanged_count += 1
```

**Domain helpers to consume directly (no pattern needed, just call):**
- `solsys_code/telescope_runs.py:parse_run_line()` -> raises `ValueError` (catch per D-02)
- `solsys_code/telescope_runs.py:get_site()` -> raises `Observatory.DoesNotExist` (catch alongside `ValueError` per RESEARCH.md A1/Pitfall 3)
- `solsys_code/telescope_runs.py:sun_event(site, d, 'sun'|'dark')` -> returns `(Time, Time)`; convert with `.to_datetime(timezone=dt_timezone.utc)`

**Night iteration helper** (RESEARCH.md Pattern 2 — new code, no direct analog):
```python
def _iter_run_nights(parsed) -> list[date]:
    """Yields one evening date per observing night (E - S + 1 nights, INGEST-01)."""
    if parsed.day2 < parsed.day1:
        raise ValueError(f'Cross-month run ranges not yet supported in Phase 3: {parsed!r}')
    first_night = date(parsed.year, parsed.month, parsed.day1)
    n_nights = parsed.day2 - parsed.day1 + 1
    return [first_night + timedelta(days=i) for i in range(n_nights)]
```

---

### `solsys_code/tests/test_load_telescope_runs.py` (test, DB integration)

**Analog:** `solsys_code/tests/test_telescope_runs.py`

**Imports + class structure pattern** (lines 1-19, 38-40):
```python
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import astropy.units as u
from astropy.time import Time
from django.test import TestCase

from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import (
    SITES,
    ParsedRun,
    get_site,
    parse_run_line,
    sun_event,
)


class TestTelescopeRuns(TestCase):
    @classmethod
    def setUpTestData(cls) -> None:
        ...
```
For Phase 3, replace imports with `tom_calendar.models.CalendarEvent`, `django.core.management.call_command`, and the command's `_iter_run_nights` (if exported). Use the **same** `setUpTestData` Observatory-seeding fixture (lines 40-78ish — `Magellan-Clay`/`Magellan-Baade`/`NTT`/`FTS` `Observatory.objects.create(...)` with `obscode`, `short_name`, `lat`, `lon`, `altitude`, `timezone` fields) so site lookups via `get_site()` succeed in the test DB.

**`setUpTestData` Observatory fixture excerpt** (lines 40-67):
```python
@classmethod
def setUpTestData(cls) -> None:
    for obscode, fields in {
        '268': dict(
            name='Magellan Clay Telescope',
            short_name='Magellan-Clay',
            lat=-29.0146,
            lon=-70.6926,
            altitude=2402,
            timezone='America/Santiago',
        ),
        ...
        'E10': dict(
            name='Siding Spring Observatory',
            short_name='FTS',
            ...
        ),
    }.items():
        Observatory.objects.create(obscode=obscode, **fields)
```
Reuse verbatim (all 4 sites: Magellan-Clay, Magellan-Baade, NTT, FTS) — D-03's idempotency tests and INGEST-01/02 fixtures need `get_site('NTT')`, `get_site('FTS')`, etc. to resolve.

**Test method pattern** — `with self.assertRaises(ValueError) as ctx:` style for `parse_run_line` error cases is the model for asserting D-02 per-line error behavior; for `load_telescope_runs`, instead use `call_command('load_telescope_runs', filepath)` wrapped with `io.StringIO` captured via `stderr=` kwarg, then assert on `CalendarEvent.objects.filter(...)` counts and field values (INGEST-01/02/03, D-04).

---

## Shared Patterns

### Management command scaffolding
**Source:** `solsys_code/management/commands/fetch_jplsbdb_objects.py` (whole file, ~75 lines)
**Apply to:** `load_telescope_runs.py`
- One-line Google-style class docstring + matching `help` string
- `add_arguments(self, parser: CommandParser) -> None` ending with `return super().add_arguments(parser)`
- `handle(self, *args: Any, **options: Any) -> str | None`
- `self.stdout.write(...)` for progress/summary, `self.stderr.write(...)` for errors

### Observatory test fixtures
**Source:** `solsys_code/tests/test_telescope_runs.py:40-78` (`setUpTestData`)
**Apply to:** `test_load_telescope_runs.py`
- Identical `Observatory.objects.create(obscode=..., short_name=..., lat=..., lon=..., altitude=..., timezone=...)` seeding for all 4 `SITES` entries (Magellan-Clay, Magellan-Baade, NTT, FTS)

### Domain helpers (consume, do not reimplement)
**Source:** `solsys_code/telescope_runs.py`
**Apply to:** `load_telescope_runs.py`
- `parse_run_line(line) -> ParsedRun` (raises `ValueError`)
- `get_site(parsed.telescope) -> Observatory` (raises `Observatory.DoesNotExist`)
- `sun_event(site, d, 'sun'|'dark') -> (Time, Time)`, convert via `.to_datetime(timezone=datetime.timezone.utc)`

### Idempotent upsert
**Source:** RESEARCH.md Pattern 4 (no existing analog in codebase for `get_or_create` + conditional save on a third-party model)
**Apply to:** `load_telescope_runs.py`
- `CalendarEvent.objects.get_or_create(telescope=, instrument=, start_time=, defaults={...})`, then compare `end_time`/`title`/`description` on the non-created path and only `.save()` if changed (D-03/D-04)

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (none) | — | — | Both files have strong analogs (`fetch_jplsbdb_objects.py`, `test_telescope_runs.py`); the upsert/night-iteration logic is new but composed entirely from documented stdlib/ORM patterns in RESEARCH.md (no codebase analog needed) |

## Metadata

**Analog search scope:** `solsys_code/management/commands/`, `solsys_code/tests/`, `solsys_code/telescope_runs.py`
**Files scanned:** 3 (`fetch_jplsbdb_objects.py`, `test_telescope_runs.py`, `telescope_runs.py`)
**Pattern extraction date:** 2026-06-13
