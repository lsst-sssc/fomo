# Phase 10: Gemini Calendar Sync Command - Pattern Map

**Mapped:** 2026-06-26
**Files analyzed:** 3 (new management command, new tests, new demo notebook)
**Analogs found:** 3 / 3

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `solsys_code/management/commands/sync_gemini_observation_calendar.py` | management command | batch transform (read DB, write calendar) | `solsys_code/management/commands/sync_lco_observation_calendar.py` | exact |
| `solsys_code/tests/test_sync_gemini_observation_calendar.py` | test | CRUD / request-response | `solsys_code/tests/test_sync_lco_observation_calendar.py` | exact |
| `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb` | demo notebook | batch transform | `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` | exact |

---

## Pattern Assignments

### `solsys_code/management/commands/sync_gemini_observation_calendar.py`

**Analog:** `solsys_code/management/commands/sync_lco_observation_calendar.py`

**Imports pattern** (lines 1-16 of analog):
```python
from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
```
Note: drop `requests`, `forms`, `LCOFacility`, `SOARFacility`, `make_request`, `urljoin`,
`ImproperCredentialsException`, `CalendarEventTelescopeLabel` — none are needed for the
Gemini command. Add `from django.conf import settings` for `FACILITIES['GEM']['programs']`
lookup (GEM-INSTR-01 and ToO-type detection).

**Password strip at load time** (D-04 decision):
```python
safe_params = {k: v for k, v in record.parameters.items() if k != 'password'}
```
Apply immediately at the top of the per-record loop body, before any logging, field
derivation, or exception paths. Use `safe_params` everywhere downstream; never reference
`record.parameters` directly again within that loop iteration.

**Settings lookup for instrument description and ToO-type** (D-02):
```python
gem_programs = settings.FACILITIES.get('GEM', {}).get('programs', {})
description_str = gem_programs.get(prog, {}).get(obs_code)
# description_str is e.g. 'Rap: GMOS-S MOS' or 'Std: GMOS-S MOS' or None
```
`obs_code` comes from `safe_params['obsid'][0]` (D-03). Two outputs from one read:
instrument label is everything after `'Std: '` / `'Rap: '`; ToO-type is the prefix.

**Multi-obsid warning** (D-03 decision):
```python
obsid_list = safe_params['obsid']
if len(obsid_list) > 1:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        'ObservationRecord pk=%s has multiple obsid entries: %r — using first entry only',
        record.pk,
        obsid_list,
    )
obs_code = obsid_list[0]
```

**Skip + warning when obs_code absent from settings** (D-01 decision):
```python
if description_str is None:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        "GS-2026A-T-999 obs code %r not found in FACILITIES['GEM']['programs'] "
        '— skipping ObservationRecord %s',
        obs_code,
        record.pk,
    )
    counters[site_key]['skipped'] += 1
    continue
```

**Unique URL key** (from CONTEXT.md code context):
```python
url = f"GEM:{safe_params['prog']}/{record.observation_id}"
```

**Window derivation from explicit params (GEM-WINDOW-01)**:
```python
from datetime import datetime
from datetime import timezone as dt_timezone

window_date = safe_params.get('windowDate')
window_time = safe_params.get('windowTime')
window_duration = safe_params.get('windowDuration')
if window_date and window_time and window_duration:
    start_dt = datetime.strptime(window_date, '%Y-%m-%d').replace(tzinfo=dt_timezone.utc)
    time_dt = datetime.strptime(window_time, '%H:%M')
    start_time = start_dt.replace(hour=time_dt.hour, minute=time_dt.minute)
    end_time = start_time + timedelta(hours=int(window_duration))
```

**ToO-type derived window (GEM-WINDOW-02)** — when no explicit window present:
```python
from datetime import timedelta

if description_str.startswith('Rap:'):
    start_time = record.created
    end_time = record.created + timedelta(hours=24)
elif description_str.startswith('Std:'):
    start_time = record.created + timedelta(hours=24)
    end_time = record.created + timedelta(days=7)
```

**ON_HOLD title prefix (GEM-STATUS-01)**:
```python
ready = safe_params.get('ready', 'true')
title_prefix = '[ON_HOLD] ' if ready == 'false' else ''
instrument_label = description_str.split(': ', 1)[1] if ': ' in description_str else description_str
title = f'{title_prefix}{telescope} {instrument_label}'
```

**No-churn get_or_create + update_fields idiom** (lines 617-628 of analog):
```python
event, created = CalendarEvent.objects.get_or_create(url=url, defaults=fields)
if created:
    counters[site_key]['created'] += 1
else:
    changed = any(getattr(event, field_name) != value for field_name, value in fields.items())
    if changed:
        for field_name, value in fields.items():
            setattr(event, field_name, value)
        event.save()
        counters[site_key]['updated'] += 1
    else:
        counters[site_key]['unchanged'] += 1
```
Note: do NOT call `event.save(update_fields=changed_keys)` — the analog uses a full
`event.save()` after setting all attrs. No `CalendarEventTelescopeLabel` sidecar write
for Gemini events (out of scope per CONTEXT.md).

**Counter structure — Gemini sites** (D-08, simpler than LCO's 6-counter shape):
```python
counters = {
    'GS': {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0},
    'GN': {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0},
}
```
No `extraction_failed` or `telescope_api_failed` counters — those are LCO-specific.

**Summary output format** (D-08 — mirror LCO style, per-site lines):
```python
self.stdout.write(f'Gemini South: created: {counters["GS"]["created"]}, '
                  f'updated: {counters["GS"]["updated"]}, '
                  f'unchanged: {counters["GS"]["unchanged"]}, '
                  f'skipped: {counters["GS"]["skipped"]}')
self.stdout.write(f'Gemini North: created: {counters["GN"]["created"]}, '
                  f'updated: {counters["GN"]["updated"]}, '
                  f'unchanged: {counters["GN"]["unchanged"]}, '
                  f'skipped: {counters["GN"]["skipped"]}')
self.stdout.write('Done.')
```

**Command class skeleton** (lines 501-519 of analog):
```python
class Command(BaseCommand):
    """Sync Gemini queue ObservationRecords to the FOMO calendar as CalendarEvents."""

    help = 'Sync Gemini queue ObservationRecords to CalendarEvents'

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        # Optional --proposal flag (Claude's discretion per CONTEXT.md):
        # if added, mirror the LCO analog's _parse_proposal_arg pattern.
        pass

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Sync matching GEM ObservationRecords to CalendarEvents."""
        records = ObservationRecord.objects.filter(facility='GEM')
        ...
```

**per-record try/except skeleton** (lines 592-603 of analog):
```python
for record in records:
    try:
        # password strip, field derivation, get_or_create
        ...
    except (KeyError, ValueError) as exc:
        self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')
        # determine site_key from safe_params.get('prog', '') before this block
        counters[site_key]['skipped'] += 1
        continue
```

**Telescope from program prefix**:
```python
prog = safe_params.get('prog', '')
if prog.startswith('GS-'):
    site_key = 'GS'
    telescope = 'Gemini South'
elif prog.startswith('GN-'):
    site_key = 'GN'
    telescope = 'Gemini North'
else:
    # unknown prefix — skip and log
    self.stderr.write(f'Unknown Gemini program prefix in {prog!r}; skipping record {record.pk}')
    # use a fallback key so the counter increment doesn't KeyError
    counters.setdefault('UNKNOWN', {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0})
    counters['UNKNOWN']['skipped'] += 1
    continue
```

---

### `solsys_code/tests/test_sync_gemini_observation_calendar.py`

**Analog:** `solsys_code/tests/test_sync_lco_observation_calendar.py`

**Imports pattern** (lines 1-26 of analog):
```python
import io
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase, override_settings
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory
```
No `patch`/`MagicMock` needed — Gemini command makes no live API calls. Add
`override_settings` for patching `FACILITIES['GEM']['programs']` in tests.

**`_parameters` helper pattern** (lines 28-60 of analog — adapt for GEM shape):
```python
def _gem_parameters(
    prog: str = 'GS-2026A-T-999',
    obsid: list[str] | None = None,
    ready: str = 'true',
    window_date: str | None = None,
    window_time: str | None = None,
    window_duration: str | None = None,
) -> dict:
    params = {
        'prog': prog,
        'obsid': obsid or ['MM'],
        'ready': ready,
        'password': '[redacted]',
    }
    if window_date:
        params['windowDate'] = window_date
        params['windowTime'] = window_time or '00:00'
        params['windowDuration'] = window_duration or '1'
    return params
```

**TestCase structure** (lines 60-100 area of analog — `setUpTestData`):
```python
class TestSyncGeminiObservationCalendar(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = get_user_model().objects.create_user(username='testuser', password='pass')
        cls.target = NonSiderealTargetFactory.create(name='test-target')

    def _make_record(self, observation_id, params, **kwargs):
        return ObservationRecord.objects.create(
            observation_id=observation_id,
            target=self.target,
            user=self.user,
            facility='GEM',
            status='PENDING',
            parameters=params,
            **kwargs,
        )
```

**`call_command` invocation pattern** (from analog and demo notebook):
```python
stdout_buf = io.StringIO()
stderr_buf = io.StringIO()
call_command('sync_gemini_observation_calendar', stdout=stdout_buf, stderr=stderr_buf)
```

**`override_settings` for GEM programs** (needed in every test that exercises
instrument / ToO-type lookup):
```python
GEM_SETTINGS = {
    'GEM': {
        'programs': {
            'GS-2026A-T-999': {
                'MM': 'Std: GMOS-S MOS',
                'QQ': 'Rap: GMOS-S MOS',
            }
        }
    }
}

@override_settings(FACILITIES=GEM_SETTINGS)
def test_explicit_window_creates_event(self):
    ...
```

---

### `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb`

**Analog:** `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`

**Cell 1 — markdown intro**: describe the command, list the 4 scenarios covered
(explicit window, Rap: derived, Std: derived, ON_HOLD + idempotent re-run per D-06).

**Cell 2 — Django setup** (copy verbatim from analog cells `b2c3d4f2` / `c3d4e5a3`):
```python
import os, sys
from pathlib import Path
import django

repo_root_path = Path.cwd().resolve().parents[2]
assert (repo_root_path / 'manage.py').exists(), ...
repo_root = str(repo_root_path)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'src.fomo.settings')
os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'true')
django.setup()
```

**Cell 3 — settings patch** (D-07 decision — patch FACILITIES['GEM']['programs']):
```python
from django.test.utils import override_settings
from django.conf import settings

# Patch in a minimal GEM programs block so instrument lookup and ToO-type
# detection run without real credentials.
settings.FACILITIES.setdefault('GEM', {})
settings.FACILITIES['GEM']['programs'] = {
    'GS-2026A-T-999': {
        'MM': 'Std: GMOS-S MOS',
        'QQ': 'Rap: GMOS-S MOS',
    }
}
```

**Cell 4 — fixture creation** (mirror analog cell `e5f6a7c5`, adapt for GEM shape;
use `NonSiderealTargetFactory`; `password` placeholder = `'[redacted]'`; D-05 IDs):
```python
from django.contrib.auth import get_user_model
from tom_observations.models import ObservationRecord
from tom_targets.models import Target
from tom_targets.tests.factories import NonSiderealTargetFactory

DEMO_TARGET_NAME = 'sync-gem-demo-target'
demo_user, _ = get_user_model().objects.get_or_create(username='sync-gem-demo-user')
demo_target = Target.objects.filter(name=DEMO_TARGET_NAME).first()
if demo_target is None:
    demo_target = NonSiderealTargetFactory.create(name=DEMO_TARGET_NAME)
```

**Scenario cells** (D-06 decision — four scenarios):

1. Explicit window (GEM-WINDOW-01):
   `parameters` includes `windowDate`, `windowTime`, `windowDuration`; assert
   `event.start_time` and `event.end_time` match the parsed values.

2. Rap: derived window (GEM-WINDOW-02):
   `parameters` has no `windowDate`; obs_code maps to `'Rap: GMOS-S MOS'`; assert
   `event.start_time == record.created` and `event.end_time == record.created + 24h`.

3. Std: derived window (GEM-WINDOW-02):
   obs_code maps to `'Std: GMOS-S MOS'`; assert window is `[created+24h, created+7d]`.

4. ON_HOLD + idempotent re-run (GEM-STATUS-01, GEM-NOCHURN-01):
   `ready='false'`; assert `event.title.startswith('[ON_HOLD] ')`; re-run and assert
   `event.modified` is unchanged + stdout shows `unchanged: 1`.

**call_command pattern** (same as LCO demo, no `--proposal` arg unless added):
```python
import io
from django.core.management import call_command

stdout_buf = io.StringIO()
stderr_buf = io.StringIO()
call_command('sync_gemini_observation_calendar', stdout=stdout_buf, stderr=stderr_buf)
print('stdout:', stdout_buf.getvalue())
```

**Expected stdout format** (D-08):
```
Gemini South: created: 2, updated: 0, unchanged: 1, skipped: 0
Gemini North: created: 0, updated: 0, unchanged: 0, skipped: 0
Done.
```

**Teardown cell** (mirror analog cell `d6e7f8a6`):
Delete all `CalendarEvent`s keyed on `f"GEM:{prog}/{observation_id}"`, all fixture
`ObservationRecord`s, `demo_target`, and `demo_user`.

---

## Shared Patterns

### Module-level logger
**Source:** `solsys_code/management/commands/sync_lco_observation_calendar.py` (convention)
**Apply to:** `sync_gemini_observation_calendar.py`
```python
import logging
logger = logging.getLogger(__name__)
```

### No-churn get_or_create + conditional save
**Source:** `sync_lco_observation_calendar.py` lines 617-628
**Apply to:** `sync_gemini_observation_calendar.py` (every record write path)

Key rule: compare each field value before calling `save()`; only save when `changed`
is non-empty; increment `unchanged` otherwise. Do NOT write a `CalendarEventTelescopeLabel`
sidecar row — that is LCO-only.

### Django test structure (TestCase + setUpTestData + NonSiderealTargetFactory)
**Source:** `solsys_code/tests/test_sync_lco_observation_calendar.py` lines 1-60
**Apply to:** `test_sync_gemini_observation_calendar.py`

Always use `NonSiderealTargetFactory` (never `SiderealTargetFactory`) — FOMO targets are
Solar System objects. Use `setUpTestData` for shared target/user fixture.

### `call_command` with captured stdout/stderr
**Source:** analog test file and demo notebook
**Apply to:** all test methods and notebook demo cells
```python
stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
call_command('sync_gemini_observation_calendar', stdout=stdout_buf, stderr=stderr_buf)
```

### Ruff formatting
**Apply to:** all Python files
- Single quotes throughout
- 120-character line length
- Google-style docstrings with `Args:` / `Returns:` sections on multi-line functions
- `from __future__ import annotations` not used in this codebase; use Python 3.10+
  union syntax (`str | None`) directly

---

## No Analog Found

None. All three new files have direct, exact analogs in the codebase.

---

## Key Differences from LCO Analog

These are intentional simplifications in the Gemini command versus the LCO analog:

| Aspect | LCO Analog | Gemini Command |
|---|---|---|
| Live API calls | Yes (`_resolve_placement_block`) | None |
| Sidecar model | `CalendarEventTelescopeLabel` | Not used |
| Telescope resolution | `SITE_TELESCOPE_MAP` + API | `prog` prefix (`GS-`/`GN-`) only |
| Instrument extraction | `_extract_instrument` scanning `c_N_*` keys | `FACILITIES['GEM']['programs'][prog][obs_code]` description string |
| Window derivation | `parameters['start']`/`['end']` or `scheduled_start`/`scheduled_end` | `windowDate`/`windowTime`/`windowDuration` or ToO-type from settings |
| Counter labels | 6 counters incl. `extraction_failed`, `telescope_api_failed` | 4 counters: `created`, `updated`, `unchanged`, `skipped` |
| Summary format | Single `Done. proposal: X, LCO: ...\|SOAR: ...` line | Two lines (one per Gemini site) + `Done.` |
| Password in params | Not applicable | Strip `password` key at load time (D-04) |

---

## Metadata

**Analog search scope:** `solsys_code/management/commands/`, `solsys_code/tests/`, `docs/notebooks/pre_executed/`, `src/fomo/settings.py`
**Files scanned:** 4 (analog command, analog test, analog notebook, settings)
**Pattern extraction date:** 2026-06-26
