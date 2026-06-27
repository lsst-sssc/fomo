# Phase 8: Telescope Label Verification Sidecar - Pattern Map

**Mapped:** 2026-06-24
**Files analyzed:** 6
**Analogs found:** 5 / 6 (1 partial — first model/migration in this app has no direct in-app analog; closest cross-app analog used)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `solsys_code/models.py` (add `CalendarEventTelescopeLabel`) | model | CRUD | `solsys_code/solsys_code_observatory/models.py` (`Observatory`) | role-match (field/verbose_name conventions; no sidecar/O2O precedent in-repo) |
| `solsys_code/migrations/0001_calendareventtelescopelabel.py` | migration | CRUD | none (first real migration in this app) — generate via `./manage.py makemigrations solsys_code`, do not hand-write | no analog |
| `solsys_code/management/commands/sync_lco_observation_calendar.py` (add sidecar write) | service/command | CRUD (write) | itself, lines 604-624 (existing `get_or_create`/diff/`save()` block) | exact (same file, same loop, same statement style) |
| `src/templates/tom_calendar/partials/calendar.html` (dashed-border + tooltip) | component (template) | request-response | itself, lines 158-167 (existing `[QUEUED]` conditional-class branch) | exact (same file, same precedent pattern, two render branches) |
| `solsys_code/tests/test_sync_lco_observation_calendar.py` (new sidecar-row assertions) | test | CRUD | itself, `test_sync_04_rerun_updates_in_place_no_churn_on_unchanged` (line 320) | exact (same file, same no-churn idiom) |
| new template-rendering test (likely `solsys_code/tests/test_calendar_template.py` or added to an existing test module) | test | request-response | `solsys_code/solsys_code_observatory/tests/test_views.py` (`Client`-based view test, lines 1-17) | role-match (only `Client`-based view test precedent in repo; no `calendar.html`-specific precedent exists yet — confirmed in RESEARCH.md Wave 0 Gaps) |
| `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` | doc/notebook | file-I/O | itself (prior committed version) | exact (CLAUDE.md mandates this companion stays in sync — see Shared Patterns) |

## Pattern Assignments

### `solsys_code/models.py` — add `CalendarEventTelescopeLabel`

**Analog:** `solsys_code/solsys_code_observatory/models.py` (`Observatory` model) for the `verbose_name`-on-every-field convention; structural shape comes directly from RESEARCH.md's validated code example (no in-repo O2O-sidecar precedent exists — this is genuinely this app's first real model).

**Imports pattern** (`solsys_code/solsys_code_observatory/models.py` lines 1-8):
```python
from math import atan2, cos, degrees, radians, sin

import astropy.units as u
import erfa
from astropy.coordinates import EarthLocation
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone as django_timezone
```
For the new model, the minimal equivalent is:
```python
from django.db import models

from tom_calendar.models import CalendarEvent
```

**Field convention — every field carries `verbose_name`** (`Observatory`, lines 27-29):
```python
obscode = models.CharField(
    max_length=4, null=False, blank=False, default=None, unique=True, verbose_name='MPC observatory code'
)
```
Apply the same convention to the new model's two fields (`event`, `is_verified`) — RESEARCH.md's exact code example already does this correctly:
```python
class CalendarEventTelescopeLabel(models.Model):
    """Sidecar record of whether a CalendarEvent's telescope label was live-verified
    against the LCO API or fallback-guessed (TELESCOPE-03/04). One row per
    CalendarEvent at most; no row at all means "verified" by documented default
    (e.g. classically-scheduled events from load_telescope_runs, which never go
    through telescope-label resolution).
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

**No error handling / validation pattern needed** — `BooleanField` with `default=True` requires no custom `clean()`/validator (unlike `Observatory`'s `lon`/`lat` which use `MinValueValidator`/`MaxValueValidator` — not applicable here).

---

### `solsys_code/migrations/0001_calendareventtelescopelabel.py`

**No analog — generate, do not hand-write.** `solsys_code/migrations/` currently contains only `__init__.py` (confirmed). Run:
```bash
./manage.py makemigrations solsys_code
./manage.py migrate
```
then commit the generated file as-is. Do not pattern-match against another app's migration file by hand-editing — Django's `makemigrations` output for a single `OneToOneField(primary_key=True)` + `BooleanField` model is deterministic and simple; no customization is needed.

---

### `solsys_code/management/commands/sync_lco_observation_calendar.py` — sidecar write

**Analog:** same file, lines 604-624 (the existing `get_or_create`/diff/`save()` block this phase's write is colocated with).

**Current code at the integration point** (lines 604-624, confirmed this session):
```python
url = fields.pop('url')
telescope_api_failed = fields.pop('telescope_api_failed')
if telescope_api_failed:
    # SYNC-09/D-11: fixed, generic message -- never interpolates a
    # caught exception (no {exc}/str(exc)/repr(exc) here). SYNC-07: the
    # record still gets a CalendarEvent below; the run continues.
    self.stderr.write(
        f'Telescope API lookup failed or returned an unmapped code for '
        f'observation_id={record.observation_id!r}; using fallback label.'
    )
    counters[record.facility]['telescope_api_failed'] += 1

event, created = CalendarEvent.objects.get_or_create(url=url, defaults=fields)
if created:
    counters[record.facility]['created'] += 1
else:
    changed = any(getattr(event, field_name) != value for field_name, value in fields.items())
    if changed:
        for field_name, value in fields.items():
            setattr(event, field_name, value)
        event.save()
        counters[record.facility]['updated'] += 1
    else:
        counters[record.facility]['unchanged'] += 1
```

**Core pattern to add** (new statement, immediately after the `if created: ... else: ...` block, still inside `for record in records:`):
```python
# Phase 8 / DISPLAY-01: always reconcile the sidecar row to the current
# telescope_api_failed signal, regardless of whether CalendarEvent's own fields
# changed -- kept as a separate statement, never folded into `fields` or `changed`
# (Pitfall 3, RESEARCH.md). is_verified reflects the outcome of the most recent
# sync run that included this record, not real-time state.
CalendarEventTelescopeLabel.objects.update_or_create(
    event=event, defaults={'is_verified': not telescope_api_failed}
)
```

**Import to add** (top of file, near existing `from tom_calendar.models import CalendarEvent` at line 9 — confirmed):
```python
from solsys_code.models import CalendarEventTelescopeLabel
```

**Error handling pattern:** none needed beyond what already exists — `telescope_api_failed` is already computed and validated upstream in `_build_event_fields()`; `update_or_create()` raises nothing new in the success path.

---

### `src/templates/tom_calendar/partials/calendar.html` — dashed-border + tooltip

**Analog:** same file, lines 153-169 (existing all-day-event loop with the `[QUEUED]` conditional-class precedent) and lines 170-183 (timed-event loop, currently has no conditional border logic at all).

**Current all-day branch** (lines 153-169, confirmed this session):
```django
{% for event in day.all_day_events %}
  <div class="cal-event cal-event-all-day-row"
       hx-get="{% url 'calendar:update-event' event.id %}"
  >
    {% include 'tom_calendar/partials/target_list_block.html' with target_list=event.target_list %}
    {% if event.title|slice:":9" == "[QUEUED] " %}
    <div class="cal-event-all-day" style="background-color: rgba(0, 0, 0, 0.45); border: 1px solid rgba(0, 0, 0, 0.55);">
    {% else %}
    <div class="cal-event-all-day" style="background-color: {{ event.color }};">
    {% endif %}
      {{ event.title|truncatechars:18 }}
      {% if event.active_todos.count %}
      ({{ event.active_todos.count }})
      {% endif %}
    </div>
  </div>
{% endfor %}
```

**Pattern to add (DISPLAY-02/03, D-02):** insert an `{% elif %}` branch using the `== False` idiom (RESEARCH.md Pattern 2 — deliberately not bare truthiness, so a missing sidecar row and an explicit `is_verified=True` both fall through to the unstyled/default branch):
```django
{% if event.title|slice:":9" == "[QUEUED] " %}
<div class="cal-event-all-day" style="background-color: rgba(0, 0, 0, 0.45); border: 1px solid rgba(0, 0, 0, 0.55);">
{% elif event.telescope_label_meta.is_verified == False %}
<div class="cal-event-all-day" style="background-color: {{ event.color }}; border: 2px dashed rgba(0, 0, 0, 0.65);"
     title="Telescope label is an estimate — could not be verified against the LCO API; showing a coarse fallback label (1m0/0m4/2m0/4m0).">
{% else %}
<div class="cal-event-all-day" style="background-color: {{ event.color }};">
{% endif %}
```

**Current timed branch** (lines 170-183, confirmed — has no conditional class/style today, just a flat `cal-event-timed` div):
```django
{% for event in day.events %}
  <div class="cal-event cal-event-timed"
       hx-get="{% url 'calendar:update-event' event.id %}"
  >
    {% include 'tom_calendar/partials/target_list_block.html' with target_list=event.target_list %}
    <span class="cal-event-title">
      {{ event.title|truncatechars:16 }}
      {% if event.active_todos.count %}
      ({{ event.active_todos.count }})
      {% endif %}
    </span>
    <span class="cal-event-time">{{ event.start_time|offset_time:utc_offset|time:"H:i" }}</span>
  </div>
{% endfor %}
```
Per D-02, this branch needs the same dashed-border treatment added — there is no `[QUEUED]` precedent here to extend, so introduce the conditional fresh on the outer `<div class="cal-event cal-event-timed" ...>` (the wrapping/hoverable element, per RESEARCH.md Pitfall 4 — title attribute must live on an element that actually receives `:hover`):
```django
{% if event.telescope_label_meta.is_verified == False %}
<div class="cal-event cal-event-timed" style="border: 2px dashed rgba(0, 0, 0, 0.65);"
     hx-get="{% url 'calendar:update-event' event.id %}"
     title="Telescope label is an estimate — could not be verified against the LCO API; showing a coarse fallback label (1m0/0m4/2m0/4m0).">
{% else %}
<div class="cal-event cal-event-timed"
     hx-get="{% url 'calendar:update-event' event.id %}"
>
{% endif %}
```

**Pitfall guard (RESEARCH.md Pitfall 4):** add both the dashed-border style and the `title="..."` attribute in the same task/commit, on the same wrapping `<div>`, for both branches — do not split border and tooltip across separate tasks.

---

### `solsys_code/tests/test_sync_lco_observation_calendar.py` — sidecar-row assertions

**Analog:** same file, `test_sync_04_rerun_updates_in_place_no_churn_on_unchanged` (line 320) — the established no-churn regression idiom this phase's new sidecar-row assertions mirror.

**Imports already present** (lines 1-19, confirmed):
```python
import io
import re
from datetime import datetime
from datetime import timezone as dt_timezone
from unittest.mock import MagicMock, patch

import requests
from django import forms
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_common.exceptions import ImproperCredentialsException
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.soar import SOARFacility
from tom_observations.models import ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory
```
Add `from solsys_code.models import CalendarEventTelescopeLabel` to this block.

**Core pattern to mirror — assert against `CalendarEvent.objects.get()`/`.count()` after `call_command(...)`** (lines 308-318):
```python
with patch(
    'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
    return_value=_observations_block_response(site='coj', telescope='2m0a', state='COMPLETED'),
):
    call_command(
        'sync_lco_observation_calendar',
        '--proposal',
        'MATCHCODE',
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
event = CalendarEvent.objects.get()
self.assertEqual(event.title, 'COJ-2m0 2M0-SCICAM-MUSCAT')
```
New test extends this directly: after the same `call_command`, assert
`CalendarEventTelescopeLabel.objects.get(event=event).is_verified` matches the expected verified/fallback outcome, and (separately) that `load_telescope_runs`-created events have `event.telescope_label_meta.is_verified` raise `DoesNotExist` (no row).

**No-churn pattern to extend** (`test_sync_04_rerun_updates_in_place_no_churn_on_unchanged`, lines 320-359) — same `modified_before`/re-run/`modified` comparison idiom, applied to the sidecar row: re-run the sync twice with an unchanged record and assert `CalendarEventTelescopeLabel.objects.count()` does not grow and the row's own implicit `pk` (it's the same `event` O2O pk) stays stable.

---

### New template-rendering test (Wave 0 gap — no precedent for `calendar.html` specifically)

**Analog:** `solsys_code/solsys_code_observatory/tests/test_views.py` (lines 1-17) — only `Client`-based view test precedent in this repo.

**Pattern to mirror:**
```python
from django.test import Client, TestCase


class CreateObservatoryTest(TestCase):
    def setUp(self) -> None:
        self.client = Client()

    def test_form_view(self):
        response = self.client.post('/observatory/create/', {'obscode': 'Z31'})
        self.assertEqual(response.status_code, 302)
```
Adapt for the calendar: `self.client.get('/calendar/...')` (confirm exact URL name via `calendar:` namespace used in the template, e.g. `reverse('calendar:calendar')` style) with a fixture `CalendarEvent` + `CalendarEventTelescopeLabel(is_verified=False)`, then `self.assertContains(response, 'border: 2px dashed')` and `self.assertContains(response, 'Telescope label is an estimate')`. Use `NonSiderealTargetFactory` per CLAUDE.md if any `Target` fixture is needed (never `SiderealTargetFactory`). Also assert a `load_telescope_runs`-created event (no sidecar row) renders with no exception and without the dashed-border marker.

## Shared Patterns

### No-churn / separate-statement discipline
**Source:** `solsys_code/management/commands/sync_lco_observation_calendar.py` lines 604-624 (existing `fields`/`changed` diff block)
**Apply to:** the new `update_or_create` sidecar write
```python
# Never fold is_verified into `fields` dict or the `changed = any(...)` comparison —
# CalendarEventTelescopeLabel.objects.update_or_create(...) does its own no-churn
# compare internally and must remain a standalone statement.
```

### `verbose_name` on every model field
**Source:** `solsys_code/solsys_code_observatory/models.py` (`Observatory`, lines 27-40)
**Apply to:** `CalendarEventTelescopeLabel`'s `event` and `is_verified` fields (already reflected in the Pattern Assignments section above).

### Defensive `== False` template comparison for reverse-O2O reads
**Source:** RESEARCH.md Pattern 2 (no in-repo precedent yet — this phase establishes it)
**Apply to:** both `calendar.html` render branches
```django
{% if event.telescope_label_meta.is_verified == False %}
```
Do not use bare `{% if not event.telescope_label_meta.is_verified %}` or rely on `|default:True` — `== False` makes "missing row" and "explicit True" both fall through identically, and is easier to verify in a test (RESEARCH.md Assumptions Log A1).

### Demo notebook companion (CLAUDE.md convention — already bitten twice)
**Source:** CLAUDE.md, "Demo notebook companions are part of the deliverable"
**Apply to:** any plan task that modifies `sync_lco_observation_calendar.py`'s behavior
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` MUST be in that task's `files_modified`
- Add/update a cell demonstrating the new sidecar row write with real executed output
- Regenerate via `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` and commit with output (this notebook is exempt from pre-commit's notebook-output-clearing).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `solsys_code/migrations/0001_calendareventtelescopelabel.py` | migration | CRUD | First real migration in this app (`solsys_code/migrations/` is `__init__.py`-only) — generate via `./manage.py makemigrations solsys_code`, do not hand-write or pattern-match against another app's migration. |

## Metadata

**Analog search scope:** `solsys_code/` (models, management commands, tests), `solsys_code/solsys_code_observatory/` (models, tests), `src/templates/tom_calendar/partials/calendar.html`
**Files scanned:** `solsys_code/models.py`, `solsys_code/migrations/`, `solsys_code/solsys_code_observatory/models.py`, `solsys_code/management/commands/sync_lco_observation_calendar.py`, `solsys_code/management/commands/load_telescope_runs.py`, `solsys_code/tests/test_sync_lco_observation_calendar.py`, `solsys_code/solsys_code_observatory/tests/test_views.py`, `src/templates/tom_calendar/partials/calendar.html`
**Pattern extraction date:** 2026-06-24
