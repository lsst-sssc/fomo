# Phase 25: Range-window CalendarEvent Projection - Pattern Map

**Mapped:** 2026-07-17
**Files analyzed:** 3 (2 modified, 1 new)
**Analogs found:** 3 / 3

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `solsys_code/campaign_views.py` (`_project_calendar_event`, `_set_run_status`, new `_calendar_event_title` helper) | controller (view-layer business logic) | CRUD (create/update `CalendarEvent` rows), date-range iteration | itself (in-file precedent: existing single-night branches) + `solsys_code/management/commands/load_telescope_runs.py` (`_iter_run_nights`, per-night loop) | exact (self) / role-match (per-night idiom) |
| `solsys_code/management/commands/backfill_range_calendar_events.py` (NEW) | migration/utility (one-off Django management command) | batch (find qualifying rows, call existing projection logic per row) | `solsys_code/management/commands/load_telescope_runs.py` | role-match (same `BaseCommand` shape, same `insert_or_create_calendar_event` counters/stdout-summary convention) |
| `solsys_code/tests/test_campaign_approval.py` (`TestCalendarProjection`, `TestRunStatusChange`, `TestSitesNeedingReview`, `TestGeminiFtScenario` — revised; possibly new test classes for FIX-04/FIX-08) | test | request-response (Django `TestCase` + `self.client.post`) | itself — `TestCalendarProjection`/`TestRunStatusChange` (existing methods in the same file, e.g. `test_approve_single_night_ground_run_creates_dip_corrected_calendar_event`) | exact |

## Pattern Assignments

### `solsys_code/campaign_views.py` — `_project_calendar_event()` (controller, CRUD)

**Analog:** the function's own existing single-night branches (lines 392-455) plus
`load_telescope_runs.py`'s `_iter_run_nights()` (lines 54-89) and `Command.handle()`'s
per-night loop (lines 129-171) for the *iteration idiom* only — do not copy that file's
title/description format, only its `E - S + 1` inclusive-range shape.

**Current guard** (verbatim, line 412):
```python
if not (run.telescope_instrument and run.site and run.window_start and run.window_start == run.window_end):
    return False
```

**D-01 required guard** (drop equality clause, add `window_end` truthiness):
```python
if not (run.telescope_instrument and run.site and run.window_start and run.window_end):
    return False
```

**Imports already present** (`campaign_views.py` lines 14-52) — no new import needed for
D-01/D-02/D-03 beyond `Q` (see Shared Patterns below):
```python
from datetime import date, datetime
from datetime import time as dt_time
from datetime import timezone as dt_timezone
...
from .calendar_utils import insert_or_create_calendar_event
from .models import CampaignRun
from .telescope_runs import sun_event
```

**Satellite branch — unchanged verbatim** (lines 420-429, D-05, do not touch):
```python
if run.site.observations_type == Observatory.SATELLITE_OBSTYPE:
    event_fields['start_time'] = datetime.combine(run.window_start, dt_time(0, 0), tzinfo=dt_timezone.utc)
    event_fields['end_time'] = datetime.combine(run.window_end, dt_time(23, 59), tzinfo=dt_timezone.utc)
    insert_or_create_calendar_event({'url': f'CAMPAIGN:{run.pk}'}, fields=event_fields)
    return True
```
(Open Question 1 in RESEARCH.md: whether this branch's title should also get the D-06
window suffix when `window_start != window_end` — recommend yes, via the same shared
`_calendar_event_title(run)` helper, so this block's only required edit is swapping the
inline `event_fields['title']` construction for a call to that helper.)

**Ground branch — current single-night version to generalize** (lines 442-455):
```python
try:
    sunset, sunrise = sun_event(run.site, run.window_start, kind='sun')
except ValueError:
    logger.debug(
        'sun_event(sun) raised for site=%s date=%s; re-raising so callers that need the '
        'retry guarantee (resolve_site) see this as a failure, not a by-design skip.',
        run.site,
        run.window_start,
    )
    raise  # CR-01: never silently swallow this
event_fields['start_time'] = sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
event_fields['end_time'] = sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
insert_or_create_calendar_event({'url': f'CAMPAIGN:{run.pk}'}, fields=event_fields)
return True
```

**Per-night inclusive-range idiom to copy** (`load_telescope_runs.py:76-89`, the `E - S +
1` pattern D-02 mirrors — `CampaignRun.window_start`/`window_end` are already `date`
objects, so no `ParsedRun`/month-rollover handling is needed, just the loop shape):
```python
n_nights = parsed.day2 - parsed.day1 + 1
...
first_night = date(parsed.year, parsed.month, parsed.day1)
return [first_night + timedelta(days=i) for i in range(n_nights)]
```
Equivalent for this phase: `n_nights = (run.window_end - run.window_start).days + 1`
(always >= 1; single night -> 1), then `night = run.window_start + timedelta(days=i) for
i in range(n_nights)`.

**`load_telescope_runs.py`'s per-night `insert_or_create_calendar_event()` call shape to
mirror (call-per-night convention only, not its lookup key or title format)**
(lines 137-161):
```python
for d in nights:
    sunset, sunrise = sun_event(site, d, 'sun')
    ...
    event, action = insert_or_create_calendar_event(
        {'telescope': parsed.telescope, 'instrument': parsed.instrument, 'start_time': start_time},
        {'end_time': end_time, 'title': title, 'description': description},
        start_time_tolerance=_START_TIME_MATCH_TOLERANCE,
    )
```
This phase's per-night call must instead use the `url`-keyed lookup (exact-match, no
`start_time_tolerance`) consistent with the rest of `campaign_views.py`:
`{'url': f'CAMPAIGN:{run.pk}'}` for `n_nights == 1` (unchanged, byte-identical to today),
`{'url': f'CAMPAIGN:{run.pk}:{night.isoformat()}'}` for `n_nights > 1` (D-03).

**Error handling to preserve exactly**: the `except ValueError: logger.debug(...); raise`
block (CR-01) must wrap each `sun_event()` call inside the loop, not just the first —
partial projection on a mid-loop raise is the accepted behavior per RESEARCH.md
Assumption A3 (no `transaction.atomic()` wrap, consistent with every other
`insert_or_create_calendar_event()` call site in this codebase).

---

### `solsys_code/campaign_views.py` — `_set_run_status()` (controller, CRUD)

**Analog:** its own current single-event version (lines 708-763).

**Current single-key existence check + update** (verbatim, lines 752-760):
```python
if CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').exists():
    prefix = _RUN_STATUS_CALENDAR_PREFIX[new_run_status]
    insert_or_create_calendar_event(
        {'url': f'CAMPAIGN:{run.pk}'},
        fields={
            'title': f'{prefix} {run.campaign.name}: {run.telescope_instrument}',
            'description': f'{run.observation_details}\nRun status: {run.get_run_status_display()}',
        },
    )
```

**D-04 required rewrite** (find-and-update EVERY matching event; combined-queryset shape
is Claude's Discretion per CONTEXT.md, this is the RESEARCH-recommended default):
```python
matching_events = CalendarEvent.objects.filter(
    Q(url=f'CAMPAIGN:{run.pk}') | Q(url__startswith=f'CAMPAIGN:{run.pk}:')
)
if matching_events.exists():
    prefix = _RUN_STATUS_CALENDAR_PREFIX[new_run_status]
    for event in matching_events:
        insert_or_create_calendar_event(
            {'url': event.url},
            fields={
                'title': f'{prefix} {_calendar_event_title(run)}',
                'description': f'{run.observation_details}\nRun status: {run.get_run_status_display()}',
            },
        )
```
**Pitfall to avoid (see Common Pitfalls in RESEARCH.md, load-bearing)**: this must call
the SAME `_calendar_event_title(run)` helper `_project_calendar_event()` uses for
creation — a re-derived `f'{run.campaign.name}: {run.telescope_instrument}'` string here
would silently strip the D-06 window suffix on every `mark_cancelled`/
`mark_weather_failure`, because `insert_or_create_calendar_event()`'s no-churn diff is a
plain string comparison, not a suffix-aware merge.

**Pitfall to avoid (trailing colon)**: the `url__startswith` prefix MUST be
`f'CAMPAIGN:{run.pk}:'` (trailing colon), never bare `f'CAMPAIGN:{run.pk}'` — without it,
`run.pk == 3` would also match `'CAMPAIGN:34:2026-07-13'` (substring collision between
one- and two-digit pks).

**Guard logic to preserve exactly, unmodified** (lines 727-746, `approval_status` check
+ conditional `.filter().update()` + `updated_count == 0` short-circuit before
`refresh_from_db()`) — D-04 only touches the calendar-sync block below this, nothing
above it.

---

### `solsys_code/management/commands/backfill_range_calendar_events.py` (NEW file)

**Analog:** `solsys_code/management/commands/load_telescope_runs.py` (full file, for
`BaseCommand` structure, `add_arguments`, counters + `self.stdout.write()` summary
convention).

**Class/argument-parsing shape to copy** (lines 92-104):
```python
class Command(BaseCommand):
    """..."""

    help = '...'

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Report what would be backfilled without writing any CalendarEvent rows.',
        )
        # No return statement — BaseCommand.add_arguments() returns None
```
Note: `--dry-run` has **no precedent anywhere in this codebase** (checked via `grep -rn
"dry.run\|dry_run\|store_true" solsys_code/management/` — zero matches); RESEARCH.md
flags this `[ASSUMED]`, not settled. `load_telescope_runs.py` itself takes only a
positional `filepath` arg, no flags at all — confirm with the planner/user whether to
include `--dry-run` before locking the plan's task list.

**Counters + stdout-summary convention to copy** (lines 116-121, 162-171, 173-175):
```python
created_count = 0
updated_count = 0
unchanged_count = 0
skipped_count = 0
...
        if action == 'created':
            created_count += 1
        elif action == 'updated':
            updated_count += 1
        else:
            unchanged_count += 1
...
self.stdout.write(
    f'Done. lines processed: {lines_processed}, '
    f'created: {created_count}, '
    # ...
)
```

**Query shape** (from RESEARCH.md's verified recommendation — `CampaignRun` candidates
with `approval_status=APPROVED`, resolved `site`, non-null `window_start`, and
`window_start != window_end`, with no existing `CAMPAIGN:{pk}*` event):
```python
from django.db.models import F, Q
from tom_calendar.models import CalendarEvent
from solsys_code.campaign_views import _project_calendar_event  # cross-module
    # underscore-prefixed import -- an accepted existing pattern in this codebase
from solsys_code.models import CampaignRun

candidates = CampaignRun.objects.filter(
    approval_status=CampaignRun.ApprovalStatus.APPROVED,
    site__isnull=False,
    window_start__isnull=False,
).exclude(window_start=F('window_end'))

for run in candidates:
    already_has_event = CalendarEvent.objects.filter(
        Q(url=f'CAMPAIGN:{run.pk}') | Q(url__startswith=f'CAMPAIGN:{run.pk}:')
    ).exists()
    if already_has_event:
        continue
    # call _project_calendar_event(run), same try/except ValueError swallow pattern
    # approve() already uses; increment created/skipped counters; final stdout summary.
```

**Error handling to mirror**: `load_telescope_runs.py`'s per-item
`try: ... except (ValueError, Observatory.DoesNotExist) as exc: self.stderr.write(...);
skipped_count += 1; continue` (lines 133-171) — same idea for a `_project_calendar_event()`
`ValueError` raised for a candidate row (e.g. blank `site.timezone`): log to `self.stderr`,
increment a `skipped`/`failed` counter, continue to the next candidate rather than aborting
the whole backfill run.

---

### `solsys_code/tests/test_campaign_approval.py` (test, request-response)

**Analog:** the file's own existing methods in the same classes.

**`_make_pending_run()` fixture helper to reuse unchanged** (lines 129-143):
```python
def _make_pending_run(self, **overrides):
    """Create a PENDING_REVIEW CampaignRun; kwargs override the default field set."""
    kwargs = {
        'campaign': self.campaign,
        'telescope_instrument': 'FTN/MuSCAT3',
        'site_raw': 'F65',
        'window_start': date(2026, 8, 1),
        'window_end': date(2026, 8, 1),
        'observation_details': 'Photometric monitoring',
        'contact_person': CONTACT_PERSON,
        'contact_email': CONTACT_EMAIL,
        'approval_status': CampaignRun.ApprovalStatus.PENDING_REVIEW,
    }
    kwargs.update(overrides)
    return CampaignRun.objects.create(**kwargs)
```
The three "must stay `count == 0`, byte-identical" tests
(`test_approve_tbd_run_creates_no_calendar_event`,
`test_approve_without_telescope_instrument_creates_no_calendar_event`,
`test_sun_event_valueerror_skips_projection_without_reverting_approval`, lines 356-380)
all use this fixture's **default** window (`window_start=window_end=date(2026,8,1)`,
`n_nights==1`) and require literally zero code change — verify explicitly, do not touch.

**Existing single-night assertion pattern to mirror for the FIX-01/FIX-03 revised range
test** (lines 326-334, `test_approve_single_night_ground_run_creates_dip_corrected_calendar_event`):
```python
def test_approve_single_night_ground_run_creates_dip_corrected_calendar_event(self):
    run = self._make_pending_run()
    self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
    event = CalendarEvent.objects.get(url=f'CAMPAIGN:{run.pk}')
    expected_sunset, expected_sunrise = sun_event(self.ground_site, run.window_start, kind='sun')
    self.assertEqual(event.start_time, expected_sunset.to_datetime(timezone=timezone.utc).replace(microsecond=0))
    self.assertEqual(event.end_time, expected_sunrise.to_datetime(timezone=timezone.utc).replace(microsecond=0))
    self.assertEqual(event.target_list_id, self.campaign.pk)
    self.assertEqual(event.telescope, run.telescope_instrument)
```

**Test to REVISE — `test_approve_range_run_creates_no_calendar_event`** (currently
lines 349-354, count-must-flip-to-15 per RESEARCH.md's re-derived arithmetic for the
`ground_site` fixture's 8/1..8/15 default range):
```python
def test_approve_range_run_creates_no_calendar_event(self):
    run = self._make_pending_run(window_start=date(2026, 8, 1), window_end=date(2026, 8, 15))
    self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
    run.refresh_from_db()
    self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
    self.assertEqual(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').count(), 0)
```
Must become (rename to reflect the new behavior, e.g.
`test_approve_range_run_creates_one_event_per_night`): assert
`CalendarEvent.objects.filter(Q(url=f'CAMPAIGN:{run.pk}') | Q(url__startswith=f'CAMPAIGN:{run.pk}:')).count() == 15`,
plus new FIX-03 assertions that the first night's and last night's `start_time`/`end_time`
match `sun_event()` for `window_start`/`window_end` respectively (guards the
first-night-only regression risk RESEARCH.md flags explicitly).

**Class docstrings asserting the OLD "range... skipped by design" contract that must be
rewritten (Pitfall 4, in-scope edits, not just code)**: `TestCalendarProjection`
(lines 300-306), `TestRunStatusChange` (lines 383-391), and — per RESEARCH.md — the
module docstring (lines 1-16) and `TestGeminiFtScenario` (lines 2130-2137). Grep check
after the phase ships: `grep -n "skipped by design\|range.*no.*event\|never project"
solsys_code/campaign_views.py solsys_code/tests/test_campaign_approval.py` should return
no matches.

## Shared Patterns

### No-churn create-or-update — `insert_or_create_calendar_event()`
**Source:** `solsys_code/calendar_utils.py:318-378`
**Apply to:** every `CalendarEvent` write in `_project_calendar_event()` (both branches,
per-night loop), `_set_run_status()`'s per-event update loop, and the new backfill
command — never construct or `.save()` a `CalendarEvent` directly.
```python
def insert_or_create_calendar_event(
    lookup: dict[str, Any],
    fields: dict[str, Any],
    *,
    start_time_tolerance: timedelta | None = None,
) -> tuple[CalendarEvent, str]:
    ...
    event, created = CalendarEvent.objects.get_or_create(**lookup, defaults=fields)
    if created:
        return event, 'created'
    return _update_or_unchanged(event, fields)
```
This phase's calls all use the exact-match `url` lookup (no `start_time_tolerance`,
matching every existing `campaign_views.py` call site — that parameter is a
`load_telescope_runs.py`-only concern for drifting sun-event start times).

### Shared title-building helper (NEW, this phase must extract it)
**Source:** none yet — extract `_calendar_event_title(run) -> str` in
`campaign_views.py`, called from both `_project_calendar_event()` (creation) and
`_set_run_status()` (status-change update, prefixed with `[CANCELLED]`/`[WEATHERED]`).
**Apply to:** every title string this phase writes, ground and satellite branches alike.
```python
def _calendar_event_title(run: CampaignRun) -> str:
    """Base CalendarEvent title (no status prefix), including the D-06 window-context
    suffix for range-window runs (window_start != window_end)."""
    base = f'{run.campaign.name}: {run.telescope_instrument}'
    if run.window_start != run.window_end:
        return f'{base} (window {run.window_start}..{run.window_end})'
    return base
```
Single-night runs (`window_start == window_end`) keep the existing unsuffixed format
byte-identical (D-06 locked wording).

### `Q`/`F` ORM primitives — new import needed
**Source:** `campaign_views.py` line 24 currently imports
`from django.db.models import Case, CharField, Count, EmailField, F, Value, When` — `F`
already present, `Q` is NOT. Add `Q` to this same import line for D-04's combined
queryset filter (`_set_run_status()`) and, if the backfill command lives in a separate
file, import `Q`/`F` fresh there per the Code Examples block above.

### Cross-module underscore-prefixed import (accepted existing pattern)
**Source:** RESEARCH.md notes this is already an established convention in this codebase
(cites the pending 2026-07-02 todo about `calendar_utils.py`'s underscore helpers having
3 consumers) — the new backfill command importing
`from solsys_code.campaign_views import _project_calendar_event` is consistent with this,
not a new anti-pattern.

## No Analog Found

None — every file this phase touches (`campaign_views.py`, the new backfill command,
`test_campaign_approval.py`) has a strong same-codebase analog (itself, or
`load_telescope_runs.py` for the per-night/management-command shape). No file requires
falling back to RESEARCH.md's Code Examples as a first-choice pattern source; those Code
Examples are themselves derived from these same analogs and are reproduced above for
convenience.

## Metadata

**Analog search scope:** `solsys_code/` (`campaign_views.py`, `calendar_utils.py`,
`management/commands/load_telescope_runs.py`, `tests/test_campaign_approval.py`) — no
broader repo search was needed since RESEARCH.md already source-verified every relevant
line number and this phase touches no new architectural layer.
**Files scanned:** 4 (all read directly, targeted non-overlapping ranges)
**Pattern extraction date:** 2026-07-17
