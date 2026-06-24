# Phase 3: Classical Calendar Ingest - Research

**Researched:** 2026-06-13
**Domain:** Django management command; idempotent upsert of `tom_calendar.CalendarEvent` rows from parsed run-line data using Phase 1/2 helpers
**Confidence:** HIGH

## Summary

This phase is a pure orchestration layer: a Django management command,
`load_telescope_runs`, that reads a text file of classical-schedule run
lines, calls Phase 2's `parse_run_line()` on each line, expands each
`ParsedRun` into one `tom_calendar.CalendarEvent` per observing night using
Phase 1's `get_site()`/`sun_event()`, and upserts those events keyed on
`(telescope, instrument, start_time)`. No new models, no migrations, no
external libraries beyond what's already installed (`tom_calendar` is already
in `INSTALLED_APPS`, its migrations are already applied as part of
`tom_calendar`'s own migration history). The entire implementation surface is
one new file (`solsys_code/management/commands/load_telescope_runs.py`) plus
one new test file (`solsys_code/tests/test_load_telescope_runs.py`).

The two technical wrinkles are (1) converting `astropy.time.Time` objects
(returned by `sun_event()`) into timezone-aware Python `datetime`s that
Django's `DateTimeField` can store, and (2) correctly iterating a
`date1..date2` range when `ParsedRun` only carries one `(year, month)` pair
for `day1` — `day2` may belong to the next month (and, in the
December-rollover case from Phase 2, `day2`'s year may differ from
`ParsedRun.year` too). The cleanest fix is to construct `date(year, month,
day1)` and then use `timedelta`-based date arithmetic (`date + timedelta(days=n)`)
to step through nights — this handles month/year rollover for free and avoids
any manual "day2 < day1 means next month" branching.

**Primary recommendation:** Build the command around a single per-run helper
function `_iter_run_nights(parsed: ParsedRun) -> Iterator[date]` that yields
each evening date via `timedelta` arithmetic from `date(parsed.year,
parsed.month, parsed.day1)`, run `(n_nights = parsed.day2 - parsed.day1 + 1)`
times. For each night, call `sun_event(site, d, 'sun')` and `sun_event(site,
d, 'dark')`, convert both `Time` pairs to UTC `datetime` via
`.to_datetime(timezone=timezone.utc)`, and use `get_or_create` +
conditional-`save()` (not `update_or_create`, which always writes) to satisfy
D-03/D-04.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Run-line parsing | Business logic (`telescope_runs.py`) | — | Already built in Phase 2; this phase only consumes `parse_run_line()` |
| Site/sun-event lookup | Business logic (`telescope_runs.py`) | Database/ORM (`Observatory`) | Already built in Phase 1; `sun_event()` queries `Observatory` internally via `get_site()` |
| File reading & per-line orchestration | Management command (CLI entry point) | — | New code for this phase; Django `BaseCommand` is the standard place for file-driven batch jobs |
| Night iteration & date arithmetic | Management command | — | New code; pure Python `date`/`timedelta`, no external dependency |
| CalendarEvent upsert | Database/ORM | Management command | `CalendarEvent.objects.get_or_create()`/`save()` is the persistence operation; command orchestrates the compare-and-write logic |
| Error reporting / summary counts | Management command | — | `self.stdout`/`self.stderr` per `fetch_jplsbdb_objects.py` convention |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Django (management commands) | Already installed (Django 5.2 per memory) | `BaseCommand` subclass, `add_arguments`, `handle()` | Existing project convention (`fetch_jplsbdb_objects.py`) |
| `tom_calendar` | Already installed, in `INSTALLED_APPS` | `CalendarEvent` model | Provided by `tom_base`; already migrated |
| `astropy.time.Time` | Already installed (used by Phase 1) | Returned by `sun_event()`; converted to `datetime` via `.to_datetime(timezone=...)` | Already the project's time-handling library for ephemeris |
| Python stdlib `datetime`/`date`/`timedelta` | stdlib | Night-range iteration, `date(year, month, day1) + timedelta(days=n)` | No new dependency; correctly handles month/year rollover |

No new packages are required for this phase. **Package Legitimacy Audit is not
applicable** — nothing new is installed.

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `django.utils.timezone` (optional) | bundled with Django | `timezone.utc` constant / `make_aware` | Alternative to `datetime.timezone.utc` for converting `Time` -> aware `datetime`; either works since `USE_TZ` handling is via aware datetimes either way |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `get_or_create` + conditional `save()` | `update_or_create` with pre-fetch comparison | `update_or_create` always calls `.save()` on match (bumps `modified` even when nothing changed) — violates D-04's "no write if unchanged" requirement. Manual get-then-compare is required either way to decide *whether* to update; `get_or_create` is simplest because the "create" path needs no extra compare. |
| `timedelta`-based date stepping | Manual day1/day2/month/year branching ("if day2 < day1, next month") | `timedelta` arithmetic on `date` objects is correct for all month-length and year-boundary cases automatically; manual branching is exactly the kind of hand-rolled date math this domain is known to get subtly wrong (e.g. doesn't generalize past Dec->Jan to e.g. Jan 30 - Feb 3). |

## Package Legitimacy Audit

Not applicable — this phase installs no new packages. `tom_calendar` is
already an installed, migrated dependency (verified by reading
`tom_calendar/models.py` directly from
`.../site-packages/tom_calendar/models.py` and confirming `'tom_calendar'` is
in `INSTALLED_APPS` in `src/fomo/settings.py:64`).

## Architecture Patterns

### System Architecture Diagram

```
schedule.txt (file path arg)
        |
        v
load_telescope_runs.handle()
        |
        |-- for each line (with line number):
        |     |
        |     v
        |   parse_run_line(line)  --[ValueError]--> log to stderr, count skipped, continue
        |     |
        |     v (ParsedRun)
        |   _iter_run_nights(parsed) -> evening dates d_1..d_N  (N = day2-day1+1)
        |     |
        |     |-- for each evening date d:
        |     |     |
        |     |     v
        |     |   get_site(parsed.telescope) -> Observatory
        |     |     |
        |     |     v
        |     |   sun_event(site, d, 'sun')  -> (sunset_d, sunrise_d+1)   [Time, Time]
        |     |   sun_event(site, d, 'dark') -> (dark_start, dark_end)    [Time, Time]
        |     |     |
        |     |     v
        |     |   build title, description, start_time, end_time (UTC datetimes)
        |     |     |
        |     |     v
        |     |   CalendarEvent.objects.get_or_create(
        |     |       telescope=, instrument=, start_time=,
        |     |       defaults={end_time, title, description})
        |     |     |
        |     |     |-- created=True  -> count "created"
        |     |     |-- created=False -> compare end_time/title/description;
        |     |                          if changed: update + save, count "updated"
        |     |                          else: count "unchanged"
        |     v
        |   (continue to next line)
        |
        v
  print summary: lines processed, events created, updated/unchanged, skipped
```

### Recommended Project Structure
```
solsys_code/
├── management/
│   └── commands/
│       ├── fetch_jplsbdb_objects.py   # existing analog
│       └── load_telescope_runs.py     # NEW: this phase
├── telescope_runs.py                  # existing: ParsedRun, parse_run_line,
│                                       #   get_site, sun_event (consumed, not modified)
└── tests/
    ├── test_telescope_runs.py         # existing
    └── test_load_telescope_runs.py    # NEW: this phase
```

### Pattern 1: Management command skeleton (positional file argument)
**What:** A `BaseCommand` subclass with a required positional `filepath`
argument, modelled on `fetch_jplsbdb_objects.py`'s `add_arguments`/`handle`
structure but with `nargs` omitted (single required positional) instead of
`--flag` options.
**When to use:** Per D-01.
**Example:**
```python
# Source: Django docs (django.core.management.base.BaseCommand) + project
# convention from solsys_code/management/commands/fetch_jplsbdb_objects.py
from typing import Any

from django.core.management.base import BaseCommand, CommandParser


class Command(BaseCommand):
    """Load classical-schedule run lines from a file and upsert CalendarEvents."""

    help = 'Load classical telescope run lines from a file and create/update CalendarEvents'

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            'filepath',
            type=str,
            help='Path to a text file of classical run lines (one per line)',
        )
        return super().add_arguments(parser)

    def handle(self, *args: Any, **options: Any) -> str | None:
        filepath = options['filepath']
        with open(filepath) as f:
            lines = f.readlines()
        # ... per-line processing ...
        return None
```

### Pattern 2: Night-range iteration via `timedelta` (handles month/year rollover)
**What:** Build the first evening date from `ParsedRun.year`/`.month`/`.day1`,
then step forward `day2 - day1 + 1` times with `timedelta(days=1)`.
**When to use:** Always — this is the only iteration approach that correctly
handles a run spanning a month boundary (e.g. `9-13 July` stays in July, but a
hypothetical `28-2` cross-month case — already resolved to a single `month`
by Phase 2's `_CROSS_MONTH_RANGE` for `day1`'s month — still steps correctly
into the next month/year via `timedelta`).
**Example:**
```python
# Source: Python stdlib datetime docs; date + timedelta automatically
# normalizes across month/year boundaries.
from datetime import date, timedelta


def _iter_run_nights(parsed) -> list[date]:
    """Yields one evening date per observing night (E - S + 1 nights, INGEST-01)."""
    first_night = date(parsed.year, parsed.month, parsed.day1)
    n_nights = parsed.day2 - parsed.day1 + 1
    return [first_night + timedelta(days=i) for i in range(n_nights)]
```
Note: `parsed.day2 - parsed.day1 + 1` is correct even for the December/January
rollover case (`ParsedRun(month=12, day1=28, day2=2, year=Y+1)` from Phase 2's
`test_parse_run_line_december_january_rolls_over_year`) **only if** `day2 <
day1` is handled — but per Phase 2's actual `_CROSS_MONTH_RANGE` regex, `day1`
and `day2` belong to *different months* in that case (`28 December-2
January`), and `ParsedRun` stores only `month=12` (December, i.e. `day1`'s
month) with `year` already rolled to `Y+1`. **This means `day2 - day1 + 1`
would be negative/wrong for a genuine cross-month range** — see Open Question
1 below; for the in-scope fixtures (`9-13 July`, `Jul 8-12`, both single-month)
`day2 - day1 + 1` is correct and this is the dominant case.

### Pattern 3: `Time` -> aware UTC `datetime` conversion
**What:** `sun_event()` returns `(astropy.time.Time, astropy.time.Time)`.
Convert each to a timezone-aware `datetime` for `CalendarEvent.start_time`/`end_time`
(Django `DateTimeField` with `USE_TZ=True` requires aware datetimes).
**Example:**
```python
# Source: astropy.time.Time.to_datetime docs; verified interactively
# (Time('2026-06-10T21:59:00').to_datetime(timezone=timezone.utc) ->
#  datetime(2026, 6, 10, 21, 59, tzinfo=timezone.utc))
from datetime import timezone as dt_timezone

sunset, sunrise = sun_event(site, d, 'sun')
start_time = sunset.to_datetime(timezone=dt_timezone.utc)
end_time = sunrise.to_datetime(timezone=dt_timezone.utc)
```

### Pattern 4: Idempotent upsert satisfying D-03/D-04
**What:** `get_or_create` on the upsert key (`telescope`, `instrument`,
`start_time`), with `defaults=` supplying the create-time values; on the
"found existing" path, compare `end_time`/`title`/`description` and only
`.save()` if any differ.
**When to use:** Every event written by this command (INGEST-03).
**Example:**
```python
# Source: Django docs (QuerySet.get_or_create); pattern adapted to satisfy
# D-04's "no write if unchanged" requirement, which update_or_create cannot
# express (it always calls save() on the matched row).
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
    changed = (
        event.end_time != end_time
        or event.title != title
        or event.description != description
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

### Anti-Patterns to Avoid
- **`update_or_create()` for this upsert:** Always calls `.save()` (and thus
  bumps `modified`) on a matched row even when no field actually changed —
  directly violates D-04.
- **Manual `if day2 < day1: month += 1` branching:** Doesn't generalize (what
  if `month == 12`? what about variable month lengths — `30-2` in a
  30-day month vs `31-2` in a 31-day month?). `timedelta` arithmetic on `date`
  objects handles all of this for free.
- **Storing naive `datetime`s:** `sun_event()`'s `Time` objects are UTC-scale;
  always convert with an explicit `timezone=timezone.utc` so Django stores
  aware datetimes consistent with `USE_TZ` (check `USE_TZ` in settings —
  TOM Toolkit projects default to `USE_TZ=True`).
- **Aborting the whole run on first parse error:** D-02 explicitly requires
  catching `ValueError` per line, logging to stderr with line number + text,
  and continuing.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Upsert / dedup logic | Custom "check if exists in a dict, then bulk insert" | `CalendarEvent.objects.get_or_create()` + conditional `save()` | Django ORM handles the DB-level race/uniqueness correctly within the request/command transaction; simpler and matches D-03/D-04 exactly |
| Date range iteration across month boundaries | Manual day/month/year increment logic | `datetime.date` + `datetime.timedelta` arithmetic | stdlib `date.__add__(timedelta)` already normalizes month lengths, leap years, and year rollover |
| UTC datetime conversion from astropy Time | Manual `.jd` -> Unix epoch -> `datetime.utcfromtimestamp()` | `Time.to_datetime(timezone=timezone.utc)` | Built-in astropy method, already used implicitly via `.to_datetime()` in `test_telescope_runs.py`'s `_assert_time_close` helper |

**Key insight:** Every piece of domain logic this phase needs (parsing, site
lookup, sun events) already exists from Phases 1-2. This phase is glue code —
the risk is entirely in date arithmetic correctness and upsert semantics, both
of which have well-established stdlib/ORM solutions.

## Common Pitfalls

### Pitfall 1: `CalendarEvent.description` default vs `None`
**What goes wrong:** `description = models.TextField(blank=True, default="")`
— if code ever does `event.description or default_text`, an existing empty
string `""` is falsy and would be silently replaced, masking a real "no
description was set" bug as "always overwrite".
**Why it happens:** Conflating Django's `blank=True, default=""` (empty
string, never `None`) with a nullable field.
**How to avoid:** Compare directly: `event.description != description`, not
truthiness checks.
**Warning signs:** Tests where `description` always differs even on the
"unchanged" path.

### Pitfall 2: Re-run produces "updated" instead of "unchanged" due to microsecond/precision drift in `Time` -> `datetime`
**What goes wrong:** `sun_event()`'s bisection refines to sub-second
precision (`_find_crossing`'s 10 bisection iterations). If the *same* command
is re-run, `_find_crossing` is deterministic given the same inputs (same
`site`, same `date`, same threshold) so the recomputed `Time` should be
bit-identical — but `Time.to_datetime()` could introduce floating-point
rounding differences at the microsecond level between runs if any
intermediate float computation has non-deterministic ordering (unlikely, but
worth a defensive equality check).
**Why it happens:** Floating-point reproducibility is *usually* exact for
identical inputs, but astropy `Time` internally stores `jd1`/`jd2` as two
floats — repeated identical computation should be exact, but it's worth
verifying empirically in the idempotency test (INGEST-03) rather than
assuming.
**How to avoid:** The idempotency test (run the command twice on the same
file) should assert `updated_count == 0` on the second run, not just "no new
rows" — this catches spurious "updated" classifications from float drift.
**Warning signs:** Second run of `load_telescope_runs` on an unchanged file
reports `updated: N` instead of `updated: 0`.

### Pitfall 3: `get_site()` raising `Observatory.DoesNotExist` mid-run
**What goes wrong:** `get_site(parsed.telescope)` can raise
`Observatory.DoesNotExist` if `parsed.telescope` (a valid `SITES` key) has no
matching `Observatory` row in the DB (e.g. dev DB not yet seeded). This is a
*different* exception class than the `ValueError` that D-02 specifies for
per-line parse errors.
**Why it happens:** `parse_run_line()` only validates against the `SITES`
dict (static); `get_site()` additionally requires a DB row to exist for that
site's obscode.
**How to avoid:** Either (a) catch both `ValueError` and
`Observatory.DoesNotExist` in the per-line try/except (both represent "this
line can't be processed"), or (b) let `Observatory.DoesNotExist` propagate as
a hard error since it indicates a setup/seeding problem distinct from bad
input data. Given D-02's wording ("If a line raises `ValueError`..."), the
safer interpretation consistent with "don't abort the whole run on bad data"
is to catch both, but tag this as Claude's discretion if the planner wants a
harder failure mode for missing `Observatory` seed data (an environment
problem, not a data problem).
**Warning signs:** `./manage.py test solsys_code` runs against a fresh DB
without `Observatory` fixtures for `Magellan-Clay`/`NTT`/`FTS`.

### Pitfall 4: `sun_event()` raising `ValueError` for high-latitude / no-darkness nights
**What goes wrong:** `sun_event(site, d, 'dark')` raises `ValueError` if the
sun never reaches -15° on date `d` (per its docstring, e.g. midnight-sun
conditions). For the in-scope sites (Las Campanas, La Silla, Siding Spring —
all mid-southern-latitude), this should never trigger for realistic schedule
dates, but a malformed/garbage date (e.g. a typo'd year far in the future) in
principle could still hit other edge cases.
**Why it happens:** `_find_crossing` requires exactly 2 crossings; anything
else raises.
**How to avoid:** This `ValueError` will be caught by the same per-line
try/except as `parse_run_line()`'s `ValueError` (D-02's wording covers "If a
line raises `ValueError`" generically) — no special-casing needed, but the
error message logged to stderr should include enough context (line number +
text) to distinguish "bad parse" from "sun-event computation failed for valid
input" during debugging.
**Warning signs:** N/A for the in-scope fixtures — flagged for completeness.

### Pitfall 5: File reading — encoding and trailing whitespace/blank lines
**What goes wrong:** A schedule file with trailing blank lines, Windows
line-endings (`\r\n`), or a trailing newline at EOF could produce an extra
"line" that `parse_run_line('')` rejects with `ValueError: parse_run_line()
received an empty line` — counted as a skipped line in the summary, which is
harmless but could be confusing in the summary output ("1 line skipped" for a
trailing blank line that isn't really a schedule entry).
**Why it happens:** `open(filepath).readlines()` includes the trailing `\n`
in each line; `parse_run_line()` already does `line.strip()` internally so
trailing `\n`/`\r` are handled — but a fully blank line still raises.
**How to avoid:** Either skip blank lines (`if not line.strip(): continue`)
*before* calling `parse_run_line()` so they're not counted in
processed/skipped totals at all, or accept that they're counted as skipped
(both are reasonable; Claude's discretion per D-02's "exact stdout/stderr
message formats... Claude's discretion").
**Warning signs:** Summary reports more "skipped" lines than expected
malformed entries in a test fixture file.

## Code Examples

### Reading lines with line numbers (for D-02's error messages)
```python
# Source: standard Python idiom; enumerate starting at 1 for human-readable
# line numbers in error messages.
with open(filepath) as f:
    for line_num, line in enumerate(f, start=1):
        if not line.strip():
            continue
        try:
            parsed = parse_run_line(line)
        except ValueError as exc:
            self.stderr.write(f'Line {line_num}: {exc} (line text: {line.strip()!r})')
            skipped_count += 1
            continue
        # ... process parsed ...
```

### Building `title` and `description` (D-05/D-06)
```python
# Source: D-05 (title format), D-06 (description content order)
title = f'{parsed.telescope} {parsed.instrument}'

dark_start, dark_end = sun_event(site, d, 'dark')
dark_start_dt = dark_start.to_datetime(timezone=dt_timezone.utc)
dark_end_dt = dark_end.to_datetime(timezone=dt_timezone.utc)

description = (
    f'Dark window (-15 deg, UTC): {dark_start_dt.isoformat()} to {dark_end_dt.isoformat()}\n'
    f'Status: {parsed.status}\n'
    f'Source line: {line.strip()}'
)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| N/A — greenfield within this project | N/A | — | This phase introduces no new tooling; it composes existing Phase 1/2 functions with the existing `tom_calendar` model and the existing management-command convention. |

**Deprecated/outdated:** None applicable.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `Observatory.DoesNotExist` (raised by `get_site()` when no DB row exists for a `SITES` telescope) should be caught alongside `ValueError` in the per-line handler, per D-02's spirit, rather than propagating as a hard command failure | Common Pitfalls / Pitfall 3 | If wrong (planner decides it should hard-fail instead), a missing-Observatory-seed-data condition would abort the whole command rather than being reported as a per-line/per-run issue — low risk either way since both behaviors are defensible and the test DB already seeds all 4 `SITES` observatories (per `test_telescope_runs.py` `setUpTestData`) |
| A2 | A genuine cross-month run (`ParsedRun` with `day2`'s month != `day1`'s month, per Phase 2's `_CROSS_MONTH_RANGE`) is **not** in this phase's required fixtures, so `day2 - day1 + 1` night-count arithmetic (Pattern 2) is sufficient for INGEST-01's stated success criteria; a fully general solution would need `ParsedRun` to carry `month2`/`year2` (a Phase 2 schema question, out of this phase's scope) | Architecture Patterns / Pattern 2, Open Questions | If a real schedule file contains a `_CROSS_MONTH_RANGE` line (e.g. `28 December-2 January`), `_iter_run_nights` as written would compute a wrong/negative night count; D-02's per-line error handling would not catch this (it's not a `ValueError`, it's silently wrong output) — flagged as Open Question 1 for the planner to decide whether to guard against |

## Open Questions

1. **Does `_iter_run_nights` need to handle `ParsedRun`s from `_CROSS_MONTH_RANGE` (e.g. `28 December-2 January`)?**
   - What we know: `ParsedRun` has a single `month`/`year` pair (for `day1`).
     Phase 2's test suite includes `test_parse_run_line_december_january_rolls_over_year`,
     which successfully parses `'NTT EFOSC2 28 December-2 January'` into
     `ParsedRun(month=12, day1=28, day2=2, year=current+1)`. If Phase 3's
     `_iter_run_nights` computes `n_nights = day2 - day1 + 1 = 2 - 28 + 1 = -25`,
     that's clearly wrong.
   - What's unclear: Whether this phase's INGEST-01..03 success criteria
     (and the concrete fixtures: `NTT EFOSC2 allocation 9-13 July`, the two
     ambiguous-Magellan lines) require handling this case at all — none of
     the three documented fixtures are cross-month.
   - Recommendation: The planner should add an explicit guard:
     `if day2 < day1: raise ValueError(f'Cross-month run ranges not yet
     supported in Phase 3: {parsed!r}')`, caught by the same per-line
     try/except as other `ValueError`s (D-02), so a cross-month line is
     *reported and skipped* rather than silently producing wrong/negative
     output. This keeps Phase 3 scoped to its tested fixtures while failing
     loudly (not silently) on the untested case. Document this as a known
     limitation for a future phase if real schedule data needs it.

2. **Catching `Observatory.DoesNotExist` vs propagating it (Pitfall 3 / A1).**
   - What we know: `get_site()` raises `Observatory.DoesNotExist` (a Django
     `Model.DoesNotExist`, not a `ValueError`) if the DB has no `Observatory`
     row for the resolved obscode.
   - What's unclear: D-02 only mentions `ValueError`. Whether
     `Observatory.DoesNotExist` should be (a) caught in the same per-line
     handler (treated like a parse error), or (b) allowed to propagate and
     abort the command (treated like a setup/environment error).
   - Recommendation: Catch both `(ValueError, Observatory.DoesNotExist)` in
     the per-line try/except — simplest, most consistent with "log and
     continue", and low-risk since the test DB seeds all 4 `SITES`
     observatories so this path won't trigger in the test suite regardless.

## Environment Availability

Skipped — this phase has no external dependencies beyond already-installed
Django/`tom_calendar`/`astropy`, which Phase 1/2 already exercise
successfully (`./manage.py test solsys_code` passes 79/79 per STATE.md).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django `TestCase` (via `./manage.py test`) |
| Config file | none — Django test runner configured via `src.fomo.settings` (`DJANGO_SETTINGS_MODULE`) |
| Quick run command | `./manage.py test solsys_code.tests.test_load_telescope_runs` |
| Full suite command | `./manage.py test solsys_code` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INGEST-01 | `NTT EFOSC2 allocation 9-13 July` creates exactly 5 `CalendarEvent`s (E-S+1), one per night, `start_time`/`end_time` from `sun_event(site, d, 'sun')` | integration (Django TestCase, DB) | `./manage.py test solsys_code.tests.test_load_telescope_runs.TestLoadTelescopeRuns.test_creates_one_event_per_night -v 2` | Wave 0 |
| INGEST-01 | `end_time > start_time`, duration 8-15 hours | integration | same file, `test_event_durations_within_range` | Wave 0 |
| INGEST-02 | `telescope`/`instrument`/`title` set correctly (D-05); `description` contains dark-window times, status, and source line text (D-06) | integration | `test_event_fields_set_from_parsed_run` | Wave 0 |
| INGEST-03 | Running the command twice on the same file produces no duplicate events, and reports `updated: 0` on the second run | integration | `test_idempotent_rerun_no_duplicates` | Wave 0 |
| D-02 | A line that raises `ValueError` (ambiguous `'Magellan ...'`) is logged to stderr with line number and skipped; processing continues | integration | `test_unparseable_line_logged_and_skipped` | Wave 0 |
| D-04 | Re-run with an unchanged schedule leaves existing rows untouched (no `modified` bump, `updated: 0`) | integration | `test_unchanged_rerun_does_not_update_existing_rows` | Wave 0 |

### Sampling Rate
- **Per task commit:** `./manage.py test solsys_code.tests.test_load_telescope_runs`
- **Per wave merge:** `./manage.py test solsys_code`
- **Phase gate:** Full suite green (`./manage.py test solsys_code`) before `/gsd-verify-work`; also `ruff check .` and `ruff format --check .` per CLAUDE.md

### Wave 0 Gaps
- [ ] `solsys_code/tests/test_load_telescope_runs.py` — new file, covers INGEST-01/02/03 and D-02/D-04; needs `setUpTestData` seeding the 3 in-scope `Observatory` records (`Magellan-Clay`/`Magellan-Baade`, `NTT`, `FTS` — can mirror `test_telescope_runs.py`'s `setUpTestData` fixture, or factor it into a shared helper/fixture if duplication becomes a concern)
- [ ] `solsys_code/management/commands/load_telescope_runs.py` — new command, no existing scaffold
- [ ] A small test fixture schedule file (e.g. inline string written to a `tempfile`, or a `.txt` fixture under `solsys_code/tests/fixtures/`) containing the documented sample lines for `call_command()`-based integration tests

*(Framework itself — Django `TestCase` / `./manage.py test` — already in place; only new test file and command are gaps.)*

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Management command runs server-side via CLI/cron, not exposed to HTTP users |
| V3 Session Management | no | N/A — no session involved |
| V4 Access Control | no | CLI access already gated by server/OS access control (same as `fetch_jplsbdb_objects`) |
| V5 Input Validation | yes | The schedule file's lines are parsed via `parse_run_line()` (Phase 2), which raises `ValueError` on malformed input — already validated; this phase's only new "input" is the file path itself |
| V6 Cryptography | no | N/A |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via file path argument (e.g. `../../etc/passwd`) | Tampering / Information Disclosure | Low severity here — this is an operator-run CLI command (same trust level as `./manage.py migrate`), not a web-facing endpoint; Django's `open(filepath)` will simply error on a non-existent/unreadable file. No special mitigation beyond normal CLI-operator trust (consistent with `fetch_jplsbdb_objects`, which takes operator-supplied CLI args without sanitization). |
| Resource exhaustion via extremely large schedule file | Denial of Service | Out of scope for Stage 2 — schedule files are small (dozens of lines), operator-controlled. No mitigation needed. |
| Stored XSS via `description`/`title` containing the raw run-line text, later rendered in `tom_calendar` templates | Tampering / Information Disclosure | `tom_calendar`'s templates are Django templates with default auto-escaping (`{{ description }}` auto-escapes HTML by default) — verified by the field being a plain `TextField` rendered via standard Django template tags (per `docs/design/telescope_runs_calendar.rst`'s "Data Model" table, description shows in "Edit modal only"). No `|safe` filter usage expected; this phase does not need to add escaping itself since Django's default auto-escaping covers it. Flagged for awareness only — no action needed in this phase. |

## Sources

### Primary (HIGH confidence)
- `/mnt/wslg/distro/home/tlister/venvs/fomo312_venv/lib/python3.12/site-packages/tom_calendar/models.py` - `CalendarEvent` field definitions (read directly from installed package source) [VERIFIED: read from installed site-packages]
- `/home/tlister/git/fomo/solsys_code/telescope_runs.py` - `ParsedRun`, `parse_run_line()`, `get_site()`, `sun_event()`, `SITES` signatures and docstrings [VERIFIED: read from repo]
- `/home/tlister/git/fomo/solsys_code/tests/test_telescope_runs.py` - existing test conventions, `setUpTestData` Observatory fixtures [VERIFIED: read from repo]
- `/home/tlister/git/fomo/solsys_code/management/commands/fetch_jplsbdb_objects.py` - management command pattern (`add_arguments`, `handle`, `self.stdout`/`self.stderr`) [VERIFIED: read from repo]
- `/home/tlister/git/fomo/docs/design/telescope_runs_calendar.rst` - "The Data Model", "Astronomy: Night Boundaries", "Classical Run Input Format" sections [VERIFIED: read from repo]
- `/home/tlister/git/fomo/src/fomo/settings.py:64` - confirms `'tom_calendar'` in `INSTALLED_APPS` [VERIFIED: read from repo]
- Interactive check: `Time('2026-06-10T21:59:00').to_datetime(timezone=timezone.utc)` returns `datetime(2026, 6, 10, 21, 59, tzinfo=timezone.utc)` [VERIFIED: executed in project venv]

### Secondary (MEDIUM confidence)
None — all findings verified directly against installed package source, repo code, or executed checks.

### Tertiary (LOW confidence)
None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new packages; all components (Django, tom_calendar, astropy) already installed and verified by direct source reading
- Architecture: HIGH - patterns derive directly from reading `CalendarEvent` model fields, `ParsedRun`/`sun_event()` signatures, and the existing `fetch_jplsbdb_objects.py` command structure
- Pitfalls: HIGH - all pitfalls identified from reading actual field defaults, docstrings (`sun_event()`'s documented `ValueError` cases), and Phase 2's actual test suite (cross-month rollover case)

**Research date:** 2026-06-13
**Valid until:** 30 days (stable internal codebase; no fast-moving external dependencies)
