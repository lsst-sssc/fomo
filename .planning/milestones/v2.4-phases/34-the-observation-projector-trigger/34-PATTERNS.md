# Phase 34: The Observation Projector & Trigger - Pattern Map

**Mapped:** 2026-09-10
**Files analyzed:** 10 (new) + 6 (modified) + 3 (deleted) + paired docs
**Analogs found:** 9 / 9

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `solsys_code/observation_projector.py` (new) | service (pure logic + receiver wrapper) | event-driven / CRUD | `.planning/spikes/002-observation-projector/projector.py` (spike, tracked) | exact — port near-verbatim per RESEARCH.md |
| `solsys_code/apps.py` (modify — add `ready()`) | config (signal wiring) | event-driven | `solsys_code/models.py:423-455` (`_delete_owned_calendar_events_on_campaign_run_delete`, `@receiver(pre_delete, ...)`) + `.planning/spikes/001-b-trigger-django-post-save/spike.py` | role-match (no `ready()` exists yet in this app; pattern comes from the CampaignRun receiver + spike proof) |
| `solsys_code/management/commands/project_observation_calendar.py` (new) | management command | batch / CRUD | `solsys_code/management/commands/reconcile_campaign_runs.py` | exact — sibling sweep, same `--dry-run`/summary/per-record-isolation shape |
| `solsys_code/management/commands/sync_lco_observation_calendar.py` (DELETE, D-18) | management command | batch / CRUD | n/a (retirement target, not an analog) | n/a |
| `solsys_code/models.py` (modify — no new fields; `CalendarEventMeta.observation_record`/`observation_group` already shipped in Phase 33) | model | CRUD | `solsys_code/models.py:12-101` (`CalendarEventMeta`, already present) | exact — fields already exist, no migration needed this phase |
| `solsys_code/templatetags/calendar_display_extras.py` (modify — extend `_TERMINAL_PREFIXES`/`status_border_css`, add legend entries, add a series/"night n of N" tag) | utility (template tags) | request-response | `solsys_code/templatetags/calendar_display_extras.py:433-497` (`campaign_decoration()`) | exact — same file, same read-only-at-display-time pattern |
| `solsys_code/tests/test_observation_projector.py` (new) | test | CRUD | `solsys_code/tests/test_sync_lco_observation_calendar.py` (38 tests, being retired) | role-match — migrate behaviours, don't copy structure verbatim |
| `solsys_code/tests/test_observation_projector_signals.py` (new) | test | event-driven | `.planning/spikes/001-b-trigger-django-post-save/spike.py` (6 scenarios S1-S6) | exact — port scenario shapes |
| `solsys_code/tests/test_project_observation_calendar.py` (new) | test | batch / CRUD | `solsys_code/tests/test_sync_lco_observation_calendar.py` + `reconcile_campaign_runs.py`'s command shape | role-match |
| `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` (new, replaces `sync_lco_observation_calendar_demo.ipynb`) | notebook | file-I/O | `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` (retired) + `.planning/spikes/004-live-narrowing-updatestatus/recheck.py` (SCHED-06 baseline/recheck pattern) | exact |
| `docs/runbooks/telescope_runs_calendar.rst` (modify) | doc | — | existing file, LCO section + Gemini caveat section | exact |
| `CLAUDE.md` (modify — notebook map) | doc | — | existing "Paired docs" bullet list | exact |

## Pattern Assignments

### `solsys_code/observation_projector.py` (new — service, event-driven/CRUD)

**Primary analog:** `.planning/spikes/002-observation-projector/projector.py` (git-tracked, validated against 146 real records)
**Secondary analogs:** `solsys_code/calendar_utils.py` (helpers to call), `solsys_code/management/commands/sync_lco_observation_calendar.py` (`_failure_prefix`, before it is deleted)

**Imports pattern** (adapt from spike `projector.py:16-33`, dropping the `sync_lco_observation_calendar` import since that module is deleted — reimplement `_failure_prefix`/`_FAILURE_PREFIX_BY_STATUS` inside the new module per D-18):
```python
from __future__ import annotations

import logging
from datetime import datetime, timezone as dt_timezone
from typing import Any

from tom_observations.facility import get_service_class
from tom_observations.models import ObservationGroup, ObservationRecord

from solsys_code.calendar_utils import (
    coarse_telescope_label,
    derive_telescope,
    extract_instrument,
    insert_or_create_calendar_event,
    record_time_window,
    resolve_placement_block,
)
from solsys_code.models import CalendarEventMeta
```

**Facility-instance cache pattern** (copy near-verbatim, spike `projector.py:39,51-56`):
```python
_facilities: dict[str, Any] = {}

def facility_for(record: ObservationRecord):
    """One facility instance per facility name, never shared across LCO and SOAR."""
    name = record.facility
    if name not in _facilities:
        _facilities[name] = get_service_class(name)()
    return _facilities[name]
```

**Stage classifier — copy near-verbatim** (spike `projector.py:91-104`, matches RESEARCH.md's D-10 classifier exactly):
```python
def stage_for(record: ObservationRecord, facility) -> str:
    has_start = record.scheduled_start is not None
    has_end = record.scheduled_end is not None
    if has_start != has_end:
        return 'inconsistent'                       # D-13
    has_block = has_start and has_end
    failed = set(facility.get_failed_observing_states())
    successful = set(facility.get_terminal_observing_states()) - failed
    if record.status in failed:
        return 'terminal-negative'                   # D-11
    if record.status in successful:
        return 'observed' if has_block else 'completed-no-block'  # D-12
    return 'placed' if has_block else 'queued'
```

**Never-raise wrapper pattern — copy near-verbatim** (spike `projector.py:158-172`):
```python
def project_record(record: ObservationRecord) -> tuple[str, str]:
    """Create/update/leave-unchanged the record's event. Returns (action, stage).

    Never raises — safe inside a post_save receiver (TRIG-02).
    """
    facility = facility_for(record)
    try:
        fields, stage = event_fields_for(record, facility)
    except Exception as exc:  # noqa: BLE001 -- a projector must never break the triggering save
        logger.warning('unprojectable observation_id=%r: %s', record.observation_id, exc)
        return 'unprojectable', f'{type(exc).__name__}: {exc}'
    event, action = insert_or_create_calendar_event({'url': event_url(record, facility)}, fields)
    _write_calendar_event_meta(event, record)  # PROJ-04: write link fields, not title suffix
    return action, stage
```

**Departure from spike (must implement new, not port):** the spike's `series_for()`/`Series` dataclass wrote series identity into the *title* (`stem = f'{stem} · {series.group_name} {series.index}/{series.size}'`, `projector.py:110-118`). Per D-04, this phase must NOT do that — instead write `CalendarEventMeta.observation_record`/`observation_group` (already-shipped fields, `solsys_code/models.py:54-69`) and let `campaign_decoration()`-style template tag render "night n of N" at display time. Use `campaign_decoration()`'s read-only-at-render-time shape (below) as the pattern for the new series tag, not the spike's title-suffix approach.

**Title-priority ladder** (adapt spike `projector.py:107-118` marker vocabulary to D-01/D-02's single-letter markers; reuse `_failure_prefix`'s status-lookup shape from `sync_lco_observation_calendar.py:28-33,52-65` before that module is deleted):
```python
# Source shape: sync_lco_observation_calendar.py:28-33 (status->prefix table),
# adapted to D-02's marker vocabulary and re-homed here since the source module is deleted.
_FAILURE_MARKER_BY_STATUS = {
    'WINDOW_EXPIRED': '[X]',
    'CANCELED': '[C]',
    'FAILURE_LIMIT_REACHED': '[F]',
    'NOT_ATTEMPTED': '[F]',
}
_STAGE_MARKER = {'queued': '[Q]', 'placed': '[S]', 'observed': '[O]', 'completed-no-block': '[O]'}

def _failure_marker(status: str, facility) -> str | None:
    if status not in set(facility.get_failed_observing_states()):
        return None
    return _FAILURE_MARKER_BY_STATUS.get(status, '[F]')

def title_for(record, stage, token, target_name, facility) -> str:
    marker = _failure_marker(record.status, facility) or _STAGE_MARKER.get(stage, '[?]')  # D-13
    return f'{marker} {token} {target_name}'[:200]
```

**Observed-site lookup fallback rule (Pitfall 4)** — treat a raised exception and a resolved-but-unmapped pair identically, exactly as `sync_lco_observation_calendar.py:176-190` already does:
```python
# Source: solsys_code/management/commands/sync_lco_observation_calendar.py:176-190 (pattern),
# reuse resolve_placement_block()/derive_telescope() unchanged (D-08, sweep-only)
block = resolve_placement_block(record.observation_id, facility)
token = derive_telescope(block.get('site'), block.get('telescope')) if block is not None else None
if token is None:
    counters['site_lookup_failed'] += 1
    token = coarse_telescope_label(instrument, record.facility)  # fallback, unchanged
```

**Credential-free logging (SYNC-09/Pitfall 5)** — never interpolate a caught exception, per `sync_lco_observation_calendar.py:331-338` and `calendar_utils.py:280-304`'s own discipline: log a fixed message string, not `{exc}`.

### `solsys_code/apps.py` (modify — add `ready()`)

**Analog:** `solsys_code/models.py:423-455` (existing `pre_delete` receiver on `CampaignRun`) for the receiver-registration *idea*; `.planning/spikes/001-b-trigger-django-post-save/spike.py` for the proven `post_save` connection contract that must be replicated (S1/S4 scenarios).

**Current file (full contents, no `ready()` yet)** — `solsys_code/apps.py:1-41`:
```python
from django.apps import AppConfig


class SolsysCodeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'solsys_code'

    def target_detail_buttons(self):
        ...
    def nav_items(self):
        ...
    def data_services(self):
        return [{'class': 'tom_fink.fink.FinkDataService'}]
```

**Pattern to add** (mirrors spike `projector.py:198-204`'s `connect()`, adds the `raw`/facility guard from D-16 and the two new receiver types D-15/D-14):
```python
def ready(self):
    from django.db.models.signals import m2m_changed, post_save, pre_delete
    from tom_observations.models import ObservationGroup, ObservationRecord

    from solsys_code import observation_projector as proj

    post_save.connect(
        proj.receiver_on_save, sender=ObservationRecord, weak=False,
        dispatch_uid='solsys_code.observation_projector.post_save',
    )
    m2m_changed.connect(
        proj.receiver_on_group_membership_changed,
        sender=ObservationGroup.observation_records.through, weak=False,
        dispatch_uid='solsys_code.observation_projector.m2m_changed',
    )
    pre_delete.connect(
        proj.receiver_on_record_delete, sender=ObservationRecord, weak=False,
        dispatch_uid='solsys_code.observation_projector.pre_delete',
    )
```
Import lazily inside `ready()` (Django's own convention, and matches `models.py:453`'s lazy `from solsys_code.campaign_reconciler import writable_events` inside the `pre_delete` receiver body, to dodge circular imports at app-loading time).

### `solsys_code/management/commands/project_observation_calendar.py` (new — management command, batch/CRUD)

**Analog:** `solsys_code/management/commands/reconcile_campaign_runs.py` (full file, 128 lines — sibling sweep)
**Secondary analog:** `.planning/spikes/002-observation-projector/sweep.py` and `solsys_code/management/commands/sync_lco_observation_calendar.py` (per-facility counter dict shape, `_new_counters()`/`_COUNTER_KEYS` pattern at lines 35-49)

**Command class + `--dry-run` shape** (copy near-verbatim from `reconcile_campaign_runs.py:1-70`):
```python
import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from solsys_code.observation_projector import project_queryset
from tom_observations.models import ObservationRecord

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        'Sweep every LCO/SOAR ObservationRecord through the projector, creating or updating '
        'its calendar event. --dry-run reports what would change without writing anything.'
    )

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('--proposal', type=str, default=None, help='Comma-separated exact proposal codes.')
        parser.add_argument('--facility', choices=['LCO', 'SOAR'], default=None)
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args: Any, **options: Any) -> str | None:
        records = ObservationRecord.objects.filter(facility__in=['LCO', 'SOAR']).order_by('pk')
        if options['facility']:
            records = records.filter(facility=options['facility'])
        if options['proposal']:
            codes = {c.strip() for c in options['proposal'].split(',') if c.strip()}
            # exact-code filter only (D-17) -- no substring leakage; mirrors
            # sync_lco_observation_calendar.py's _parse_proposal_arg dedup/strip discipline
        ...
```

**Per-facility counter dict pattern** (reuse `_new_counters()`/`_COUNTER_KEYS` shape from `sync_lco_observation_calendar.py:35-49`, with D-17's renamed/added keys `created/updated/unchanged/unprojectable/site_lookups/site_lookup_failed` replacing the retired `skipped/extraction_failed/telescope_api_failed`).

**Per-record failure isolation** (copy near-verbatim from `reconcile_campaign_runs.py:57-70`):
```python
for record in records:
    try:
        action, stage = project_record(record)  # or project_queryset's per-row loop
    except Exception as exc:  # noqa: BLE001 -- the only catch point (D-17)
        logger.debug('project_record() raised for observation_id=%r: %s', record.observation_id, exc)
        self.stderr.write(f'observation_id={record.observation_id}: projection failed -- skipping')
        failed_count += 1
        continue
```

**Summary-line phrasing** (RESEARCH.md Code Examples, adapted from `sync_lco_observation_calendar.py:351-364`):
```python
summary = ' | '.join(
    f'{facility_name}: created: {counts["created"]}, updated: {counts["updated"]}, '
    f'unchanged: {counts["unchanged"]}, unprojectable: {counts["unprojectable"]}, '
    f'site_lookups: {counts["site_lookups"]}, site_lookup_failed: {counts["site_lookup_failed"]}'
    for facility_name, counts in counters.items()
)
```

### `solsys_code/templatetags/calendar_display_extras.py` (modify)

**Analog:** same file, `campaign_decoration()` at lines 433-497 (read-only-at-display-time pattern for the new series tag), and `_TERMINAL_PREFIXES`/`status_border_css()` at lines 109-173 (marker vocabulary extension point).

**Extend `_TERMINAL_PREFIXES`** (lines 109-113) — add the new single-letter markers alongside the existing bracket-word ones (D-02 keeps both vocabularies live simultaneously; legacy rows keep their old prefixes until the takeover sweep runs):
```python
_TERMINAL_PREFIXES = ('[EXPIRED]', '[CANCELLED]', '[FAILED]', '[WEATHERED]', '[X]', '[C]', '[F]')
```
`status_border_css()` (lines 142-173) needs a parallel queued-marker check added (`title.startswith('[QUEUED] ')` currently at line 162) for `'[Q] '`; leave the placed bucket (`[S]`, no prefix) returning `''` unchanged, matching the existing "placed bucket intentionally returns ''" comment at lines 146-151.

**New series/"night n of N" tag** — pattern directly from `campaign_decoration()` (lines 433-497): a `@register.simple_tag` function, never raises, reads only `event.calendar_event_meta.observation_group` (note: different related_name than `campaign_decoration`'s `telescope_label_meta` — same `CalendarEventMeta` row, accessed via `ObservationRecord`'s one-to-one `related_name='calendar_event_meta'`, `solsys_code/models.py:59`, or directly off the event's existing `telescope_label_meta` accessor since it is the same row), returns `None` for no-group/`ObjectDoesNotExist`, orders siblings by `record_time_window()[0]` per D-04, and returns a dict of primitive values only (mirrors `campaign_decoration`'s "never expose PII/provenance fields" discipline).

### Signal receiver test file: `solsys_code/tests/test_observation_projector_signals.py` (new)

**Analog:** `.planning/spikes/001-b-trigger-django-post-save/spike.py` (6 scenarios S1-S6, git-tracked). Port each scenario (schedule-only save, `updatestatus` path, bulk `.update()` non-fire, fixture `raw=True` non-fire, etc.) into a `TestCase` method — this is the proven contract TRIG-01/TRIG-02 must not regress.

### Sweep command test file: `solsys_code/tests/test_project_observation_calendar.py` (new)

**Analog:** `solsys_code/tests/test_sync_lco_observation_calendar.py` (38 tests, being retired per D-18) — migrate behaviours (no-churn, per-facility dispatch, exact-code proposal filter, failure-prefix priority, credential-free logging), not the file structure. Mock `solsys_code.calendar_utils.make_request` exactly as that file already does (proven pattern for the observed-site lookup's network dependency).

### `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` (new)

**Analog:** `.planning/spikes/004-live-narrowing-updatestatus/recheck.py` (git-tracked) for the SCHED-06 baseline/recheck snapshot cell shape — `(status, scheduled_start/end, event start/end/title)` per `KEY2026B-004` record, captured before and after `updatestatus` runs with no sweep in between. Retired notebook `sync_lco_observation_calendar_demo.ipynb`'s general "run the command, show before/after counts" structure is the shape for the takeover-diff section (D-19): snapshot every event's `(url, title, start, end, meta links)` before and after the first sweep, assert the `RUN:`/blank-url/`GEM:` sets are byte-identical.

## Shared Patterns

### Never-raise / caller-safety contract
**Source:** `.planning/spikes/002-observation-projector/projector.py:158-172` (spike), `solsys_code/management/commands/reconcile_campaign_runs.py:59-65` (production sibling)
**Apply to:** `observation_projector.py`'s `project_record()`, all three signal receivers in `apps.py`, and the sweep command's per-record loop. Every failure caught and logged, never re-raised — this is the single most load-bearing rule in the phase (TRIG-02).

### No-churn create/update/unchanged
**Source:** `solsys_code/calendar_utils.py:461-542` (`_update_or_unchanged()`, `insert_or_create_calendar_event()`)
**Apply to:** every place `observation_projector.py` writes a `CalendarEvent` — never hand-roll a `get_or_create` + manual diff (PROJ-05).

### Namespace isolation (`RUN:`/`GEM:`/blank-url never touched)
**Source:** spike `sweep.py`'s isolation check (count before == count after); `solsys_code/campaign_reconciler.py`'s `writable_events()` scoping concept (same idea, different key namespace).
**Apply to:** the sweep command's regression test and the `pre_delete` receiver's `if meta.event.url == facility.get_observation_url(...)` guard (D-14) — never delete/modify an event outside the projector's own `url` key namespace.

### Credential-free logging (SYNC-09)
**Source:** `solsys_code/calendar_utils.py:280-304` (`resolve_placement_block()`'s own discipline), `solsys_code/management/commands/sync_lco_observation_calendar.py:331-338`
**Apply to:** every `except` block in `observation_projector.py` and the sweep command that touches a `requests`/portal exception — log a fixed generic message, never interpolate the exception object.

### Attribution/series-identity is a link, rendered at display time
**Source:** `solsys_code/templatetags/calendar_display_extras.py:433-497` (`campaign_decoration()`)
**Apply to:** the new series template tag — read `CalendarEventMeta.observation_group` at request time; never write group-derived text into `CalendarEvent.title`/`.description` (D-04).

## No Analog Found

None — every file in scope has a git-tracked analog (either a validated spike or an existing production module in the same codebase).

## Metadata

**Analog search scope:** `solsys_code/`, `solsys_code/management/commands/`, `solsys_code/templatetags/`, `solsys_code/tests/`, `.planning/spikes/` (tracked origin of the gitignored `.claude/skills/spike-findings-fomo_devel/sources/` mirror — verified via `git ls-files`), `docs/notebooks/pre_executed/`
**Files scanned:** `solsys_code/apps.py`, `solsys_code/calendar_utils.py`, `solsys_code/models.py`, `solsys_code/management/commands/sync_lco_observation_calendar.py`, `solsys_code/management/commands/reconcile_campaign_runs.py`, `solsys_code/management/commands/backfill_lco_observations.py` (referenced, not re-read in full — cited by RESEARCH.md line 672), `solsys_code/templatetags/calendar_display_extras.py`, `.planning/spikes/002-observation-projector/projector.py`, `.planning/spikes/001-b-trigger-django-post-save/spike.py` (referenced), `.planning/spikes/004-live-narrowing-updatestatus/recheck.py` (referenced)
**Tracked-source note:** `.claude/skills/spike-findings-fomo_devel/` is gitignored (`.gitignore:168`); its `sources/00N-*/` spike code mirrors the git-tracked `.planning/spikes/00N-*/` directory verbatim (confirmed same filenames: `projector.py`, `sweep.py`, `recheck.py`, `spike.py`). All analog paths in this document cite the `.planning/spikes/` tracked origin, never the `.claude/skills/` mirror.
**Pattern extraction date:** 2026-09-10
