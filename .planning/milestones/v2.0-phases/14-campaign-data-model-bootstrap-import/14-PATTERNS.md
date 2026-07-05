# Phase 14: Campaign Data Model & Bootstrap Import - Pattern Map

**Mapped:** 2026-07-02
**Files analyzed:** 7
**Analogs found:** 7 / 7

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `solsys_code/models.py` (add `CampaignRun`) | model | CRUD | `solsys_code/models.py` (`CalendarEventTelescopeLabel`) | exact (same file, same "flag sidecar field" pattern) |
| `solsys_code/campaign_utils.py` (new) | utility | transform | `solsys_code/calendar_utils.py` (`insert_or_create_calendar_event`, `_derive_telescope`/status-mapping style helpers) | role-match |
| `solsys_code/management/commands/import_campaign_csv.py` (new) | route/config (management command) | file-I/O + CRUD | `solsys_code/management/commands/load_telescope_runs.py` | exact (same skip-and-log, summary-counter shape; only line-loop → `csv.DictReader` differs) |
| `solsys_code/migrations/0002_campaignrun.py` (generated) | migration | CRUD | `solsys_code/migrations/0001_calendareventtelescopelabel.py` | exact (auto-generated; use as a shape reference only) |
| `solsys_code/tests/test_campaign_models.py` (new) | test | CRUD | `solsys_code_observatory/tests/test_models.py` (or nearest model-field test) | role-match |
| `solsys_code/tests/test_import_campaign_csv.py` (new) | test | file-I/O + CRUD | `solsys_code/tests/test_load_telescope_runs.py` | exact |
| `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` (+ `fixtures/campaign_sample.csv`) | test (demo notebook) | file-I/O | `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` | exact |

## Pattern Assignments

### `solsys_code/models.py` — add `CampaignRun` (model, CRUD)

**Analog:** `solsys_code/models.py` (existing `CalendarEventTelescopeLabel`, lines 1-26 — whole file, read in full, 26 lines)

**Imports pattern** (file header, lines 1-2):
```python
from django.db import models
from tom_calendar.models import CalendarEvent
```
For `CampaignRun`, follow the same shape but import `TargetList`/`Target` from `tom_targets.models` and `Observatory` from `solsys_code.solsys_code_observatory.models`:
```python
from django.db import models
from tom_targets.models import Target, TargetList

from solsys_code.solsys_code_observatory.models import Observatory
```

**Sidecar "flag, don't silently guess" field pattern** (lines 5-25, whole class — this is the direct precedent D-09's `site_needs_review` copies):
```python
class CalendarEventTelescopeLabel(models.Model):
    """Sidecar record of whether a CalendarEvent's telescope label was live-verified
    against the LCO API or fallback-guessed (TELESCOPE-03/04). One row per
    CalendarEvent at most; no row at all means "verified" by documented default
    ...
    """

    event = models.OneToOneField(
        CalendarEvent,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name='telescope_label_meta',
        verbose_name='Calendar event',
    )
    is_verified = models.BooleanField(
        default=True, verbose_name='Whether the telescope label was live-verified against the LCO API'
    )

    def __str__(self):
        return f'{"Verified" if self.is_verified else "Fallback"} label for {self.event.title}'
```
Copy the doc-comment style ("One row per X at most; no row at all means Y by documented default") and the boolean-with-explanatory-`verbose_name` idiom for `CampaignRun.site_needs_review`. Use `models.CharField(blank=True, default='')` for `site_raw` (not `TextField`, to match this codebase's existing `CharField`-for-short-strings convention seen on `Observatory.name`/`short_name`) unless the raw sheet strings are expected to be long.

**FK conventions** (from `Observatory` model, `solsys_code/solsys_code_observatory/models.py` lines 1-9 imports + field declarations lines 11-60):
```python
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

obscode = models.CharField(
    max_length=4, null=False, blank=False, default=None, unique=True, verbose_name='MPC observatory code'
)
```
Use this `verbose_name=`-on-every-field convention for all new `CampaignRun` fields (matches both `Observatory` and `CalendarEventTelescopeLabel`).

**`TextChoices` status pattern** — no direct in-repo precedent exists yet (`CalendarEvent`'s own status-like fields, if any, use plain `CharField` without `TextChoices`); use Django's built-in `TextChoices` idiom directly as shown in RESEARCH.md's Pattern 1 (cited from Django docs, Django 5.2.14 confirmed installed via `solsys_code/migrations/0001_calendareventtelescopelabel.py` header). No analog contradicts this; it's a new-but-standard pattern for this codebase.

---

### `solsys_code/campaign_utils.py` (new) (utility, transform)

**Analog:** `solsys_code/calendar_utils.py`

**Create-or-update no-churn idempotency pattern** (`solsys_code/calendar_utils.py:296-332`, `insert_or_create_calendar_event`):
```python
def insert_or_create_calendar_event(
    lookup: dict[str, Any],
    fields: dict[str, Any],
) -> tuple[CalendarEvent, str]:
    """Create or update a CalendarEvent, or leave it unchanged if no fields differ.
    ...
    Returns:
        tuple[CalendarEvent, str]: (event, action) where action is one of
            'created' (new record written), 'updated' (existing record changed
            and saved), or 'unchanged' (existing record matched all fields; no
            save issued). Callers own counter updates and any sidecar writes.
    """
    event, created = CalendarEvent.objects.get_or_create(**lookup, defaults=fields)
    if created:
        return event, 'created'
    changed = [f for f, v in fields.items() if getattr(event, f) != v]
    if changed:
        for f, v in fields.items():
            setattr(event, f, v)
        event.save(update_fields=list(fields.keys()) + ['modified'])
        return event, 'updated'
    return event, 'unchanged'
```
Copy this verbatim, generalized to `CampaignRun` (either genuinely generalize the existing function over a `model` parameter, or fork a `CampaignRun`-specific `insert_or_create_campaign_run(lookup, fields)` with the identical body/shape) — this is the exact "created/updated/unchanged" 3-way return contract `import_campaign_csv.py`'s summary counters need.

**Coarse-label/derivation-with-safe-fallback pattern** (`solsys_code/calendar_utils.py:280-293`, `_derive_telescope`-adjacent helper — read directly):
```python
def _derive_telescope_class(facility_name: str, instrument_type: str) -> str:
    """...
    Returns:
        str: ... this never raises. This only affects the
            coarse label's text -- it never decides whether a record syncs.
    """
    if facility_name.upper() == 'SOAR':
        return '4m0'
    if len(instrument_type) >= 3:
        candidate = instrument_type[:3].lower()
        if candidate in {'0m4', '1m0', '2m0', '4m0'}:
            return candidate
    return instrument_type
```
Copy this "never raises, always returns a safe fallback string" shape for `map_observation_status()` (Pitfall 3) and `resolve_site()` (Pattern 2) — both should be structured as pure functions that always return a usable value plus an explicit flag, never raise for expected messy-data cases.

**MPC obscode lookup / tiered external-resolution pattern** (`solsys_code/solsys_code_observatory/utils.py`, whole file, 98 lines):
```python
import logging
from datetime import datetime, timezone

import requests
from tom_dataservices.dataservices import MissingDataException

from solsys_code.solsys_code_observatory.models import Observatory

logger = logging.getLogger(__name__)


class MPCObscodeFetcher:
    def query(self, obscode: str, dbg: bool = False):
        self.obs_data = None
        response = requests.get('https://data.minorplanetcenter.net/api/obscodes', json={'obscode': obscode})
        if response.ok:
            self.obs_data = response.json()
        else:
            json_resp = response.json()
            errors = self._flatten_error_dict(json_resp)
            logging.error(f'Error: {response.status_code} Message: {". ".join(errors)}')
            return json_resp

    def to_observatory(self):
        if not self.obs_data:
            raise MissingDataException('No observatory data. Did you call query()?')
        else:
            obs = Observatory()
            obs.obscode = self.obs_data['obscode']
            obs.name = self.obs_data['name_utf8']
            obs.short_name = self.obs_data['short_name']
            elong = float(self.obs_data['longitude'])
            obs.lon = elong
            obs.from_parallax_constants(elong, float(self.obs_data['rhocosphi']), float(self.obs_data['rhosinphi']))
            ...
            obs.save()
        return obs
```
Reuse `MPCObscodeFetcher` directly (import, don't reimplement) for D-08 tier 2, following the same call sequence as the live view usage below.

**Live view call sequence for tiers 1-2** (`solsys_code/solsys_code_observatory/views.py:36-57`, `CreateObservatory.form_valid`):
```python
def form_valid(self, form: BaseModelForm) -> HttpResponse:
    """... Performs the query through the MPC API using ``MPCObscodeFetcher()`` and then tries to
    create the ``Observatory`` through ``MPCObscodeFetcher.to_observatory()``. This means
    we shouldn't/don't call the superclass's ``form_valid`` method as this will
    attempt to create a duplicate (which raises an IntegrityError)
    """
    obs = MPCObscodeFetcher()
    errors = obs.query(form.cleaned_data['obscode'])
    try:
        obs = obs.to_observatory()
        self.object = obs
        self.kwargs['pk'] = obs.pk
    except MissingDataException:
        if errors:
            form.add_error('obscode', errors.get('message', 'Invalid MPC site code'))
        return self.form_invalid(form)
    except IntegrityError:
        print('Attempt to create duplicate Observatory')
        messages.error(self.request, 'Attempt to create duplicate Observatory')
```
`resolve_site()` in `campaign_utils.py` should mirror this exact `query()` → `to_observatory()` → `except MissingDataException` / `except IntegrityError` sequence, adapted to return `(Observatory | None, needs_review: bool)` instead of manipulating a Django form (see RESEARCH.md Pattern 2 for the full adapted function — already written there, ready to use near-verbatim).

**`Observatory` field shapes to respect** (`solsys_code/solsys_code_observatory/models.py:11-60`):
```python
obscode = models.CharField(max_length=4, null=False, blank=False, default=None, unique=True, ...)
name = models.CharField(max_length=255, null=False, blank=False, default=None, unique=True, ...)
lon = models.FloatField(null=True, blank=False, validators=[MinValueValidator(-180.0), MaxValueValidator(180.0)], db_index=True)
lat = models.FloatField(null=True, blank=False, validators=[MinValueValidator(-90.0), MaxValueValidator(90.0)], db_index=True)
```
`obscode.max_length == 4` is the hard constraint driving Pitfall 2 (JWST's `500@-170` — 8 chars — must never be passed to `Observatory.objects.create(obscode=...)`). Always length-check the raw Site Code against `Observatory._meta.get_field('obscode').max_length` before any tier-2/tier-3 attempt.

---

### `solsys_code/management/commands/import_campaign_csv.py` (new) (route/config, file-I/O + CRUD)

**Analog:** `solsys_code/management/commands/load_telescope_runs.py` (whole file, 138 lines)

**Imports pattern** (lines 1-9):
```python
from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser

from solsys_code.calendar_utils import insert_or_create_calendar_event
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import ParsedRun, get_site, parse_run_line, sun_event
```
For `import_campaign_csv.py`, mirror this shape: import `TargetList` from `tom_targets.models`, `CampaignRun` from `solsys_code.models`, and the new `campaign_utils` helpers.

**Command class + `add_arguments` pattern** (lines 55-67) — note the required `--campaign` argument (D-06) needs `required=True` added, unlike `load_telescope_runs`'s single positional arg:
```python
class Command(BaseCommand):
    """Load classical telescope run lines from a file and create or update CalendarEvents."""

    help = 'Load classical telescope run lines from a file and create/update CalendarEvents'

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            'filepath',
            type=str,
            help='Path to a text file of classical run lines (one per line)',
        )
```
Add a `--campaign` required option: `parser.add_argument('--campaign', type=str, required=True, help='Campaign TargetList name (found-or-created)')` alongside the positional `filepath`.

**File-open error handling** (lines 86-90):
```python
try:
    with open(filepath, encoding='utf-8') as f:
        file_lines = list(f)
except OSError as exc:
    raise CommandError(f'Cannot open schedule file {filepath!r}: {exc}') from exc
```
Replace with `csv.DictReader(f)` opened the same way (`encoding='utf-8', newline=''` per RESEARCH.md's Code Examples), same `except OSError as exc: raise CommandError(...) from exc` wrapper.

**Per-row skip-and-log + counters** (lines 79-136, full `handle()` body — this is the exact shape to adapt row-by-row):
```python
created_count = 0
updated_count = 0
unchanged_count = 0
skipped_count = 0
lines_processed = 0
...
for line_num, line in enumerate(file_lines, start=1):
    if not line.strip():
        continue
    lines_processed += 1
    try:
        parsed = parse_run_line(line)
        ...
        event, action = insert_or_create_calendar_event({...}, {...})
        if action == 'created':
            created_count += 1
        elif action == 'updated':
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
```
Adapt the `except` clause to catch `ValueError` (raised by `parse_obs_window` on unparseable `Obs. Date`) and add a `site_needs_review_count` accumulator alongside the existing four counters, per CONTEXT.md's discretion note recommending a distinct counter for that case. Use `row_num` (from `enumerate(reader, start=2)`, header = row 1) instead of `line_num`, and include the row dict in the log line the same way `line.strip()!r` is included today.

---

### `solsys_code/tests/test_import_campaign_csv.py` (new) (test, file-I/O + CRUD)

**Analog:** `solsys_code/tests/test_load_telescope_runs.py`

**`setUpTestData` seeding pattern** (lines 1-40+):
```python
import io
import pathlib
import tempfile

from django.core.management import call_command
from django.test import TestCase
from tom_calendar.models import CalendarEvent

from solsys_code.models import CalendarEventTelescopeLabel
from solsys_code.solsys_code_observatory.models import Observatory


class TestLoadTelescopeRuns(TestCase):
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
        }.items():
            Observatory.objects.create(obscode=obscode, **fields)
```
Reuse this exact `for obscode, fields in {...}.items(): Observatory.objects.create(...)` seeding idiom for pre-populating tier-1 `Observatory` matches in `test_import_campaign_csv.py`. Use `NonSiderealTargetFactory` (per CLAUDE.md) — never `SiderealTargetFactory` — for any `Target`/`TargetList` fixtures needed for D-07's auto-target-resolution test.

**`call_command` + temp-CSV-file pattern:** write a `_write_csv(rows)` helper using `tempfile.TemporaryDirectory()` + `csv.DictWriter`, then `call_command('import_campaign_csv', '--campaign', 'Test Campaign', path, stdout=io.StringIO(), stderr=io.StringIO())` — matches this file's existing `call_command(...)` invocation style with captured stdout/stderr for assertion.

---

### `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` + `fixtures/campaign_sample.csv` (test/demo, file-I/O)

**Analog:** `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`

Follow its existing cell structure (intro markdown → setup/imports cell → fixture-file cell or reference → `call_command`/direct-function-call cell showing real executed output → summary/inspection cell). New elements this phase introduces beyond that analog:
1. A `fixtures/` subdirectory colocated with the notebook (first of its kind — confirmed not `.gitignore`d, RESEARCH.md Pitfall 4) holding `campaign_sample.csv`.
2. A cell demonstrating the `pending_review` → `approved`/`rejected` `approval_status` lifecycle on synthetic data (D-03), since the bootstrap import itself always writes `approved` — this has no direct precedent in `load_telescope_runs_demo.ipynb` and should be added as new cells (e.g. manually constructing a `CampaignRun` with `approval_status=PENDING_REVIEW` then updating it), not copied from an analog.
3. Regenerate via `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` and commit with output, per CLAUDE.md's demo-notebook-companion convention.

---

## Shared Patterns

### Create-or-update idempotency ("upsert", called "create-or-update" per CLAUDE.md terminology)
**Source:** `solsys_code/calendar_utils.py:296-332` (`insert_or_create_calendar_event`)
**Apply to:** `campaign_utils.py`'s new `CampaignRun` equivalent, used by `import_campaign_csv.py`
```python
event, created = CalendarEvent.objects.get_or_create(**lookup, defaults=fields)
if created:
    return event, 'created'
changed = [f for f, v in fields.items() if getattr(event, f) != v]
if changed:
    for f, v in fields.items():
        setattr(event, f, v)
    event.save(update_fields=list(fields.keys()) + ['modified'])
    return event, 'updated'
return event, 'unchanged'
```

### Skip-and-log per-row error handling with summary counters
**Source:** `solsys_code/management/commands/load_telescope_runs.py:79-137`
**Apply to:** `import_campaign_csv.py`'s `handle()`
```python
except (ValueError, Observatory.DoesNotExist) as exc:
    self.stderr.write(f'Line {line_num}: {exc} (line text: {line.strip()!r})')
    skipped_count += 1
    continue
...
self.stdout.write(f'Done. ... created: {created_count}, updated: {updated_count}, '
                   f'unchanged: {unchanged_count}, skipped: {skipped_count}')
```

### "Flag, don't silently guess" sidecar/boolean pattern
**Source:** `solsys_code/models.py` (`CalendarEventTelescopeLabel.is_verified`)
**Apply to:** `CampaignRun.site_needs_review` (D-09)
```python
is_verified = models.BooleanField(
    default=True, verbose_name='Whether the telescope label was live-verified against the LCO API'
)
```

### Tiered external-lookup resolution (local DB → external API → placeholder)
**Source:** `solsys_code/solsys_code_observatory/views.py:36-57` (`CreateObservatory.form_valid`) + `solsys_code/solsys_code_observatory/utils.py` (`MPCObscodeFetcher`)
**Apply to:** `campaign_utils.resolve_site()` used by `import_campaign_csv.py` (D-08)
```python
obs = MPCObscodeFetcher()
errors = obs.query(code)
try:
    obs = obs.to_observatory()
except MissingDataException:
    ...  # fall through to tier 3
except IntegrityError:
    ...  # race: re-fetch existing row instead of losing it
```

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `solsys_code/campaign_utils.py` — `parse_obs_window()` (UT-time-range regex parsing) | utility | transform | No existing free-text-datetime regex parser in this codebase to copy from; `load_telescope_runs.py`'s `_resolve_window_time` is the closest structural cousin (converts a token to a UTC datetime) but operates on a controlled vocabulary (`BoN`/`EoN`/`HHMM`), not messy free text — use RESEARCH.md's Code Examples section directly (already grounded against the real sheet's verified format inventory) |
| `solsys_code/campaign_utils.py` — `map_observation_status()` (status-string translation table) | utility | transform | No existing sheet-vocabulary-to-model-vocabulary translation table in this codebase; RESEARCH.md's Code Examples provides a ready-to-use starting table (flagged `[ASSUMED]`, needs discuss/plan confirmation per Pitfall 3) |

## Metadata

**Analog search scope:** `solsys_code/` (models.py, calendar_utils.py, management/commands/, solsys_code_observatory/{models,utils,views}.py, tests/), `docs/notebooks/pre_executed/`
**Files scanned:** `solsys_code/models.py`, `solsys_code/calendar_utils.py`, `solsys_code/management/commands/load_telescope_runs.py`, `solsys_code/solsys_code_observatory/{models,utils,views}.py`, `solsys_code/tests/test_load_telescope_runs.py`, `docs/notebooks/pre_executed/` listing
**Pattern extraction date:** 2026-07-02
