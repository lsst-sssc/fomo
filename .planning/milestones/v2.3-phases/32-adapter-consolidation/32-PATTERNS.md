# Phase 32: Adapter Consolidation - Pattern Map

**Mapped:** 2026-09-03
**Files analyzed:** 10 (1 shared helper, 3 adapter rewires, 1 model/migration set, 3 null-guard sites, 2 test files as analogs)
**Analogs found:** 10 / 10 — every file in this phase has a strong, same-codebase analog; this
phase is internal rewiring, so most "new" logic is composition of already-shipped functions
rather than novel patterns.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `solsys_code/campaign_utils.py` (add `write_and_reconcile_campaign_run()`) | service (write-and-project helper) | CRUD + event-driven (create/update then trigger projection) | `solsys_code/campaign_utils.py::insert_or_create_campaign_run` (lines 817-853) | exact — wraps this function directly |
| `solsys_code/models.py` (nullable `campaign`, `source_identifier` field+constraint, `SOAR_QUEUE` Source value) | model | CRUD (schema) | `solsys_code/migrations/0014_alter_campaignrun_source.py` (ESO_QUEUE precedent) | exact |
| `solsys_code/migrations/00XX_*.py` (new migration(s)) | migration | batch/schema | `solsys_code/migrations/0014_alter_campaignrun_source.py` | exact |
| `solsys_code/management/commands/load_telescope_runs.py` (ADAPT-01 cutover) | service (management command, batch ingest) | batch → CRUD | itself (pre-cutover version, lines ~207-216) + `import_campaign_csv.py` for the write-guard pattern | role-match (same file, changed write target) |
| `solsys_code/management/commands/sync_lco_observation_calendar.py` (ADAPT-02/03 cutover) | service (management command, batch ingest) | batch → CRUD | itself (pre-cutover version, lines ~289-341) + `campaign_views.py` confirm-write (lines 1201-1204) for `CampaignRunObservation` linking | role-match |
| `solsys_code/management/commands/sync_gemini_observation_calendar.py` (ADAPT-06 cutover) | service (management command, batch ingest) | batch → CRUD | itself (pre-cutover version, lines ~150-163) | role-match |
| `solsys_code/campaign_reconciler.py::event_title()` (null-guard) | utility (string formatter) | transform | same file's `_skip_reason()` (lines 193-214) for the null-safety idiom already used nearby | exact |
| `solsys_code/models.py::CampaignRun.__str__` (null-guard) | model | transform | n/a — trivial ternary matching `event_title()`'s new guard | role-match |
| `solsys_code/campaign_tables.py` (2x `render_run`, null-guard) | component (django-tables2 column renderer) | transform | `event_title()`'s guard (same fix pattern, different file) | role-match |
| `solsys_code/campaign_attribution.py::_campaign_evidence` (null-guard) | service | transform | `event_title()`'s guard (same fix pattern) | role-match |
| `solsys_code/tests/test_load_telescope_runs.py`, `test_sync_lco_observation_calendar.py`, `test_sync_gemini_observation_calendar.py` (new `CampaignRun`/no-churn/cutover assertions) | test | request-response (Django TestCase) | existing no-churn tests in the same files (`test_sync_04_rerun_updates_in_place_no_churn_on_unchanged` at `test_sync_lco_observation_calendar.py:379`; `test_idempotent_rerun_no_duplicates`/`test_unchanged_rerun_does_not_update_existing_rows` at `test_load_telescope_runs.py:218,229`) | exact |

## Pattern Assignments

### `solsys_code/campaign_utils.py` — new `write_and_reconcile_campaign_run()` (service, CRUD+event-driven)

**Analog:** `solsys_code/campaign_utils.py::insert_or_create_campaign_run` (lines 817-853), composed
with `solsys_code/campaign_reconciler.py::reconcile_run()` and the exact-identity link precedent in
`solsys_code/campaign_views.py:1201-1204`.

**Imports pattern** (`campaign_utils.py` top, lines 15-29):
```python
import difflib
import logging
import re
from datetime import date, datetime
from datetime import timezone as dt_timezone
from typing import Any

import requests
from django.core.cache import cache
from django.db.utils import IntegrityError
from tom_dataservices.dataservices import MissingDataException

from solsys_code.models import CampaignRun
from solsys_code.observer_codes import HORIZONS_OBSERVER_TO_OBSCODE
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.solsys_code_observatory.utils import MPCObscodeFetcher
```
New imports needed for the helper: `CampaignRunObservation` from `solsys_code.models`,
`reconcile_run`/`ReconcileResult` from `solsys_code.campaign_reconciler`, and `django.utils.timezone`.
**Locked constraint:** never import `solsys_code.views` or `solsys_code.ephem_utils` (SPICE kernel
download).

**Core create-or-update pattern to wrap, not reimplement** (`campaign_utils.py:817-853`):
```python
def insert_or_create_campaign_run(lookup: dict[str, Any], fields: dict[str, Any]) -> tuple[CampaignRun, str]:
    """Create or update a CampaignRun, or leave it unchanged if no fields differ."""
    run, created = CampaignRun.objects.get_or_create(**lookup, defaults=fields)
    if created:
        return run, 'created'
    changed = [f for f, v in fields.items() if getattr(run, f) != v]
    if changed:
        for f, v in fields.items():
            setattr(run, f, v)
        run.save(update_fields=list(fields.keys()))
        return run, 'updated'
    return run, 'unchanged'
```

**Exact-identity link precedent** (`solsys_code/campaign_views.py:1201-1204`, human path — mirror the
shape, drop the human actor):
```python
_, created = CampaignRunObservation.objects.get_or_create(
    observation_record_id=orphan_pk,
    defaults={'run_id': run_pk, 'confirmed_by': request.user, 'confirmed_at': timezone.now()},
)
```
Adapter version: `confirmed_by=None` (system link, not staff-confirmed), `confirmed_at=timezone.now()`.

**Approval-gate precondition** the helper's callers must always satisfy
(`solsys_code/campaign_reconciler.py:193-214`, `_skip_reason()`):
```python
def _skip_reason(run: CampaignRun) -> str | None:
    if run.approval_status != CampaignRun.ApprovalStatus.APPROVED:
        return 'not approved'
    if not run.telescope_instrument:
        return 'missing telescope/instrument'
    if run.window_start is None or run.window_end is None:
        return 'TBD window'
    if run.window_end < run.window_start:
        return 'window_end before window_start'
    if run.site is None and not run.telescope_class:
        return 'unresolved site'
    return None
```
Every adapter's `fields` dict must set `approval_status=CampaignRun.ApprovalStatus.APPROVED` and its
own non-`WEB` `source` value, or `reconcile_run()` silently skips the row.

**Natural-key collision guard to copy** (`solsys_code/management/commands/import_campaign_csv.py:356-358`,
WR-01 precedent — apply the same pattern if the helper takes on this responsibility, per Open Question 2):
```python
# WR-01/CANON-01: never relabel a run that came in through the public web form.
if existing is not None and existing.source == CampaignRun.Source.WEB:
    fields.pop('source', None)
    fields.pop('approval_status', None)
```

---

### `solsys_code/models.py` — schema migration (model, CRUD)

**Analog:** `solsys_code/migrations/0014_alter_campaignrun_source.py` (the `ESO_QUEUE` addition —
identical shape for `SOAR_QUEUE`); field-declaration text is already locked verbatim in
`31-DECISION.md` (transcribed in RESEARCH.md "Migration precedent to transcribe").

**Field changes** (transcribe verbatim from RESEARCH.md, sourced from `31-DECISION.md` SCHEMA-01/02):
```python
campaign = models.ForeignKey(
    TargetList,
    on_delete=models.PROTECT,   # unchanged
    null=True,                  # was: null=False
    blank=True,                 # new
    related_name='campaign_runs',
    verbose_name='Campaign target list',
)

source_identifier = models.CharField(max_length=500, null=True, blank=True)

# Meta.constraints addition (additive alongside both existing partial constraints):
models.UniqueConstraint(
    fields=('source_identifier',),
    condition=models.Q(source_identifier__isnull=False),
    name='unique_campaign_run_source_identifier',
),
```

**`Source` TextChoices — current state to extend** (`solsys_code/models.py:108-128`):
```python
class Source(models.TextChoices):
    """Which FOMO ingest path created this row (CANON-01, 26-DECISION.md Criterion 1). ..."""
```
Add `SOAR_QUEUE = 'soar_queue', 'SOAR queue'` following the exact `ESO_QUEUE` precedent already in
this enum (added in plan 29-06) — same docstring-reasoning style: SOAR is a live-read-back facility
distinct from LCO, same as ESO was distinct from LCO/Gemini.

**Migration precedent** (`solsys_code/migrations/0014_alter_campaignrun_source.py`, transcribe the
full ordered `choices` list with `soar_queue` inserted after `lco_queue`):
```python
migrations.AlterField(
    model_name='campaignrun',
    name='source',
    field=models.CharField(
        choices=[
            ('web', 'Web submission'),
            ('classical_file', 'Classical run file'),
            ('lco_queue', 'LCO queue'),
            ('soar_queue', 'SOAR queue'),          # NEW
            ('gemini_queue', 'Gemini queue'),
            ('eso_queue', 'ESO queue'),
            ('csv_import', 'CSV import'),
            ('legacy', 'Legacy (pre-v2.2)'),
        ],
        default='legacy',
        max_length=20,
        verbose_name='Ingest source',
    ),
),
```

---

### `solsys_code/management/commands/load_telescope_runs.py` (ADAPT-01, service/batch→CRUD)

**Analog:** itself, pre-cutover (`load_telescope_runs.py:207-216`).

**Imports pattern** (current, lines 1-10):
```python
from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser
from tom_targets.models import TargetList

from solsys_code.calendar_utils import insert_or_create_calendar_event
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import ESO_NOON_TO_NOON_SITES, ParsedRun, get_site, parse_run_line, sun_event
```
Cutover swaps `from solsys_code.calendar_utils import insert_or_create_calendar_event` for
`from solsys_code.campaign_utils import write_and_reconcile_campaign_run`.

**Code being replaced** (`load_telescope_runs.py:207-216`):
```python
event, action = insert_or_create_calendar_event(
    {'telescope': parsed.telescope, 'instrument': parsed.instrument, 'start_time': start_time},
    {
        'end_time': end_time,
        'title': title,
        'description': description,
        'target_list': campaign,
    },
    start_time_tolerance=_START_TIME_MATCH_TOLERANCE,
)
```
Replace with a `write_and_reconcile_campaign_run(lookup, fields)` call — `lookup`/`fields` built from
the same `parsed`/`campaign` values, `source_identifier=f'CLASSICAL:{telescope}:{instrument}:{bucket}'`
(5-minute-bucketed `start_time`, per 31-DECISION.md), `source=CampaignRun.Source.CLASSICAL_FILE`,
`approval_status=CampaignRun.ApprovalStatus.APPROVED`. `observation_record=None` (classical has no
`ObservationRecord`).

**No-churn test to mirror** (`solsys_code/tests/test_load_telescope_runs.py:218,229`,
`test_idempotent_rerun_no_duplicates` / `test_unchanged_rerun_does_not_update_existing_rows`) — extend
with `CampaignRun.objects.count()` assertions alongside the existing `CalendarEvent` ones.

---

### `solsys_code/management/commands/sync_lco_observation_calendar.py` (ADAPT-02/03, service/batch→CRUD)

**Analog:** itself, pre-cutover (`sync_lco_observation_calendar.py:289-341`).

**Imports pattern** (current, lines 1-18):
```python
from datetime import datetime
from typing import Any

from django.core.management.base import BaseCommand, CommandParser
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.soar import SOARFacility
from tom_observations.models import ObservationRecord

from solsys_code.calendar_utils import (
    ...
)
from solsys_code.models import CalendarEventMeta
```
Cutover adds `from solsys_code.campaign_utils import write_and_reconcile_campaign_run` and
`from solsys_code.models import CampaignRun` (for `CampaignRun.Source.SOAR_QUEUE`/`LCO_QUEUE`).

**Facility-dispatch branch point to extend for Source selection** (`sync_lco_observation_calendar.py:289`):
```python
facilities = {'LCO': LCOFacility(), 'SOAR': SOARFacility()}
```
Add a parallel `{'LCO': CampaignRun.Source.LCO_QUEUE, 'SOAR': CampaignRun.Source.SOAR_QUEUE}` mapping,
read inside the same per-record loop (`for record in records:`) alongside the existing
`facilities.get(record.facility)` lookup.

**Code being replaced** (`sync_lco_observation_calendar.py:329,341`):
```python
url = fields.pop('url')
...
event, action = insert_or_create_calendar_event({'url': url}, fields)
```
Replace with `write_and_reconcile_campaign_run(lookup, fields, observation_record=record)` —
`observation_record=record` triggers the exact-identity `CampaignRunObservation` link (Pattern 2
above); `source_identifier=url` (the portal read-back URL is already the natural identity key).

**Existing SOAR-facility test to extend** (`test_select_05_soar_record_uses_soar_facility_instance`,
`solsys_code/tests/test_sync_lco_observation_calendar.py:608`) — add `Source.SOAR_QUEUE` assertion.

**No-churn test to mirror** (`test_sync_04_rerun_updates_in_place_no_churn_on_unchanged`,
`test_sync_lco_observation_calendar.py:379-421`).

---

### `solsys_code/management/commands/sync_gemini_observation_calendar.py` (ADAPT-06, service/batch→CRUD)

**Analog:** itself, pre-cutover (`sync_gemini_observation_calendar.py:150,163`).

**Imports pattern** (current, lines 1-12):
```python
import logging
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser
from tom_observations.models import ObservationRecord

from solsys_code.calendar_utils import insert_or_create_calendar_event
```
Cutover swaps in `from solsys_code.campaign_utils import write_and_reconcile_campaign_run` and
`from solsys_code.models import CampaignRun`.

**Code being replaced** (`sync_gemini_observation_calendar.py:150,163`):
```python
url = f'GEM:{prog}/{record.observation_id}'
...
_event, action = insert_or_create_calendar_event({'url': url}, fields)
```
Replace with `write_and_reconcile_campaign_run(lookup, fields)` — `source_identifier=url` (the
synthesized `GEM:{program}/{observation-id}` key), `source=CampaignRun.Source.GEMINI_QUEUE`,
`observation_record=None` (Gemini's `ObservationRecord` is the *source* of the write, not a
"realising" record found afterward — no `CampaignRunObservation` link).
**Doc caveat required alongside this code change:** state explicitly (in the command's own
docstring/comment and in `docs/runbooks/telescope_runs_calendar.rst`) that Gemini-sourced runs can
never receive Phase 33's automatic outcome propagation (D-02).

---

### Null-guard sites (utility/model/component, transform)

**Analog for the guard idiom:** `_skip_reason()`'s own None-checking style
(`campaign_reconciler.py:193-214`, shown above) — ternary/early-return guards already used
throughout this module.

**Hot-path site to fix first** (`solsys_code/campaign_reconciler.py:176`, `event_title()`):
```python
base = f'{run.campaign.name}: {run.telescope_instrument}'
```
Guard: `run.campaign.name if run.campaign_id else '<no campaign>'` (exact wording is a planner/UX
call per RESEARCH.md).

**Cold-path sites** (same fix pattern, apply independently):
- `solsys_code/models.py:352` (`CampaignRun.__str__`) — `f'#{self.pk} {self.campaign.name} | ...'`
- `solsys_code/campaign_tables.py:467,538` (two `render_run` methods) —
  `f'{record.run.telescope_instrument} ({record.run.campaign.name})'`
- `solsys_code/campaign_attribution.py:397` (`_campaign_evidence`) —
  `f"run belongs to campaign '{run.campaign.name}' ..."`

---

### Tests (test, request-response)

**Analog:** existing no-churn test methods in each adapter's own test file (already cited above per
file). For the new ADAPT-05 cutover-simulation tests (no existing analog — Wave 0 gap per RESEARCH.md),
model them structurally on the same `setUp`/factory conventions as the no-churn tests:
`tom_targets.tests.factories.NonSiderealTargetFactory` for any `Target` fixture (never
`SiderealTargetFactory` — CLAUDE.md convention, confirmed in use at
`test_sync_lco_observation_calendar.py:16` and `test_campaign_reconciler.py:18`).

## Shared Patterns

### Approval-gate precondition
**Source:** `solsys_code/campaign_reconciler.py:193-214` (`_skip_reason`)
**Apply to:** all three adapter cutovers — every `fields` dict passed into
`write_and_reconcile_campaign_run()` must set `approval_status=CampaignRun.ApprovalStatus.APPROVED`.

### Create-or-update, no-churn contract
**Source:** `solsys_code/campaign_utils.py:817-853` (`insert_or_create_campaign_run`)
**Apply to:** the new shared helper (wraps this directly, does not reimplement).

### Calendar projection (unchanged entry point)
**Source:** `solsys_code/campaign_reconciler.py::reconcile_run()`
**Apply to:** all three adapters, called only through the new shared helper — never a direct
`CalendarEvent` write after cutover (locked anti-pattern in CONTEXT.md/RESEARCH.md).

### Exact-identity `CampaignRunObservation` linking
**Source:** `solsys_code/campaign_views.py:1201-1204`
**Apply to:** LCO/SOAR adapter only (ADAPT-02/03); classical and Gemini pass `observation_record=None`.

### WEB-row protection guard
**Source:** `solsys_code/management/commands/import_campaign_csv.py:356-358`
**Apply to:** the shared helper or each adapter's `fields` construction, if the planner decides the
helper should defend against natural-key collision with a `WEB`-sourced row (Open Question 2 in
RESEARCH.md — planner's call, pattern is ready either way).

## No Analog Found

None — every file/change in this phase has a same-repo, same-role analog (this phase is internal
rewiring against already-shipped patterns per RESEARCH.md's "Don't Hand-Roll" table). The only
genuinely new test category (ADAPT-05 cutover-simulation, before/after `CalendarEvent.objects.count()`
assertions) has no existing analog to copy structurally — it needs the same `TestCase`/factory
conventions as the no-churn tests but its own before/after-cutover fixture shape (see RESEARCH.md
Pitfall 4 and Open Question 1).

## Metadata

**Analog search scope:** `solsys_code/` (models, campaign_utils, campaign_reconciler,
campaign_views, management/commands/, migrations/, tests/)
**Files scanned:** ~15 (all cited above), primarily via targeted grep/read against files already
identified as canonical in RESEARCH.md/CONTEXT.md — no new analog search was needed beyond
confirming exact line numbers, since RESEARCH.md's "Code Examples"/"Migration precedent" sections
already located every analog this phase needs.
**Pattern extraction date:** 2026-09-03
