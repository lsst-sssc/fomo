# Phase 29: The Reconciler - Pattern Map

**Mapped:** 2026-08-04
**Files analyzed:** 9 (new + modified, excluding paired docs)
**Analogs found:** 9 / 9

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `solsys_code/campaign_reconciler.py` (new) | service (pure logic module) | CRUD (per-run projection: read `CampaignRun`, write `CalendarEvent`/`CalendarEventMeta`) | `solsys_code/campaign_views.py:445-554` (`_calendar_event_title`/`_project_calendar_event`) | exact — this is a direct port/generalization |
| `solsys_code/management/commands/reconcile_campaign_runs.py` (new) | service (management command) | batch | `solsys_code/management/commands/backfill_range_calendar_events.py` (loop/--dry-run shape) + `import_campaign_csv.py` (summary-line shape) | exact — same command family |
| `solsys_code/campaign_views.py` (modified: delete `_calendar_event_title`/`_project_calendar_event`; rewire `approve()`, `_resolve_site()`, `_set_run_status()`) | controller (Django view) | request-response | itself (pre-change) | exact — in-place rewire |
| `solsys_code/calendar_utils.py` (modified: new small public helper for D-02 re-key) | utility | CRUD | `insert_or_create_calendar_event()` / `_update_or_unchanged()` (`calendar_utils.py:461-542`) | exact — sibling helper in same module |
| `solsys_code/tests/test_campaign_reconciler.py` (new) | test | CRUD/unit | `solsys_code/tests/test_backfill_range_calendar_events.py` (being deleted) + `test_campaign_approval.py`'s `TestCalendarProjection`/`TestCalendarNoChurn` fixture style | role-match |
| `solsys_code/tests/test_reconcile_campaign_runs.py` (new) | test | batch/integration | `solsys_code/tests/test_backfill_range_calendar_events.py` (command test shape, being deleted — read before deletion for its fixture/assert patterns) | exact (structural precedent, source file itself is retired) |
| `solsys_code/tests/test_campaign_approval.py` (modified: rewrite `CAMPAIGN:`→`RUN:` assertions and patch targets, Pitfall 3) | test | request-response | itself (pre-change) | exact — in-place rewrite |
| `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` (new) | file-I/O (paired demo notebook) | batch | `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` / `sync_lco_observation_calendar_demo.ipynb` (sibling command demo notebooks) | role-match |
| `docs/runbooks/telescope_runs_calendar.rst` (modified) | config/docs | n/a | itself (pre-change) | exact — in-place edit |
| `solsys_code/management/commands/backfill_range_calendar_events.py` + `solsys_code/tests/test_backfill_range_calendar_events.py` (both **deleted**, RECON-09) | service/test | batch | n/a (deletion) | n/a |

## Pattern Assignments

### `solsys_code/campaign_reconciler.py` (service, CRUD) — NEW

**Analog:** `solsys_code/campaign_views.py:445-554` (`_calendar_event_title`, `_project_calendar_event`)

**Imports pattern to mirror** — this module must NOT import `solsys_code.views`/`solsys_code.ephem_utils` (SPICE kernel load, locked module-home constraint). Mirror the import shape of `campaign_gap.py`/`campaign_utils.py` (pure-logic sibling modules): plain `datetime`/`timedelta`, `django.db.models.Q`, model imports from `solsys_code.models`, `tom_calendar.models.CalendarEvent`, and:
```python
from solsys_code.calendar_utils import insert_or_create_calendar_event  # + new D-02 helper
from solsys_code.telescope_runs import sun_event
```

**Core CRUD pattern — classical per-night branch** (port from `campaign_views.py:528-553`):
```python
n_nights = (run.window_end - run.window_start).days + 1
is_range = n_nights > 1
for i in range(n_nights):
    night = run.window_start + timedelta(days=i)
    sunset, sunrise = sun_event(run.site, night, kind='sun')  # ValueError NOT caught here (D-06)
    night_fields = dict(event_fields)
    night_fields['start_time'] = sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    night_fields['end_time'] = sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    url = f'RUN:{run.pk}' if not is_range else f'RUN:{run.pk}:{night.isoformat()}'
    insert_or_create_calendar_event({'url': url}, fields=night_fields)
```
**Key differences from the analog to apply during the port:**
- `CAMPAIGN:` → `RUN:` in both key forms.
- The analog's own `try/except ValueError: logger.debug(...); raise` wrapper around `sun_event()` should NOT be reproduced inside `reconcile_run()` — per D-06/Anti-Patterns, let the `ValueError` propagate uncaught out of the per-run function; only the batch-loop (command) and the staff-action call sites catch it, each with their own existing handling.
- Before minting, insert the new D-02 adopt step (see Pattern below) — check for an already-linked `CalendarEventMeta` row for this `(run, night)` and re-key it in place instead of unconditionally calling `insert_or_create_calendar_event()` with a `RUN:` lookup.

**Core CRUD pattern — satellite/space whole-window branch** (port from `campaign_views.py:503-514`, generalized for D-01's "no `CAMPAIGN:` keys" and RECON-02/03's queue and class-wide branches — Pattern 2 in RESEARCH.md):
```python
event_fields['start_time'] = datetime.combine(run.window_start, dt_time(0, 0), tzinfo=dt_timezone.utc)
event_fields['end_time'] = datetime.combine(run.window_end, dt_time(23, 59), tzinfo=dt_timezone.utc)
insert_or_create_calendar_event({'url': f'RUN:{run.pk}'}, fields=event_fields)
```
Reuse this exact shape for: satellite sites (`run.site.observations_type == Observatory.SATELLITE_OBSTYPE`), queue-scheduled runs (`run.source in {CampaignRun.Source.LCO_QUEUE, CampaignRun.Source.GEMINI_QUEUE}` — never touch per-night events, RECON-02/D-02's queue verdict), and class-wide runs (`run.telescope_class` non-blank — RECON-03).

**D-02 adopt-and-rekey step (genuinely new code, no direct precedent — see RESEARCH.md Open Question 2 / Assumption A3):**
```python
# Source: campaign_views.py:1317 for the query shape, generalized per RESEARCH.md Pattern 1's
# sketch. Match by CalendarEventMeta.run_id + event.telescope + site-local-date conversion of
# start_time (never the naive UTC date -- 26-DECISION.md Criterion 3).
from zoneinfo import ZoneInfo
candidates = CalendarEventMeta.objects.filter(
    run_id=run.pk, event__telescope=run.telescope_instrument
).select_related('event')
for meta in candidates:
    local_date = meta.event.start_time.astimezone(ZoneInfo(run.site.timezone)).date()
    if local_date == night:
        # Re-key url + refresh fields via the new calendar_utils.py public helper (see below),
        # not a direct .save() -- must go through the no-churn diff.
        ...
```

**Ownership-scoping query** (copy verbatim, `RUN:` substituted for `CAMPAIGN:`, source: `campaign_views.py:876-878`, also mirrored at `:797`):
```python
matching_events = CalendarEvent.objects.filter(
    Q(url=f'RUN:{run.pk}') | Q(url__startswith=f'RUN:{run.pk}:')
)
```
And for "every event this run already owns" (source: `campaign_views.py:1317`):
```python
CalendarEventMeta.objects.filter(run_id=run.pk).select_related('event')
```

**Error handling pattern:** `reconcile_run(run)` itself does NOT wrap `sun_event()` in try/except and does NOT use `transaction.atomic()` (D-06) — it lets `ValueError` (and any other exception) propagate. The **caller** (command batch loop, or a staff-action view method) is the only catch point, matching each call site's own existing, already-differentiated handling (see below).

---

### `solsys_code/management/commands/reconcile_campaign_runs.py` (service, batch) — NEW

**Analog 1 (loop + `--dry-run` shape):** `solsys_code/management/commands/backfill_range_calendar_events.py` (full file, 104 lines — read above in full)

**Analog 2 (summary-line shape, D-05):** `solsys_code/management/commands/import_campaign_csv.py:380-408`

**Imports pattern** (mirror `backfill_range_calendar_events.py:1-11`, `campaign_reconciler` substituted for the deleted `campaign_views._project_calendar_event`):
```python
import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from solsys_code.campaign_reconciler import reconcile_run
from solsys_code.models import CampaignRun

logger = logging.getLogger(__name__)
```
Note: the retired command imports `_project_calendar_event` directly from `campaign_views` — the RESEARCH.md milestone note calls this an anti-pattern (importing a private view helper); `campaign_reconciler.reconcile_run` is a public function in a dedicated pure module, so the new command's import is now the clean version of this same shape.

**`--dry-run` argument** (copy verbatim, source: `backfill_range_calendar_events.py:29-36`):
```python
def add_arguments(self, parser: CommandParser) -> None:
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report what would be reconciled without writing any CalendarEvent rows.',
    )
```

**Batch loop + per-run try/except (D-06, catch AT the batch-loop level only)** — mirror `backfill_range_calendar_events.py:63-88`'s try/except-continue shape:
```python
for run in CampaignRun.objects.all():
    try:
        action = reconcile_run(run, dry_run=dry_run)  # exact signature at planner's discretion
    except Exception as exc:
        logger.debug('reconcile_run() raised for run pk=%s: %s', run.pk, exc)
        self.stderr.write(f'Run pk={run.pk}: reconcile failed ({exc}) -- skipping')
        skipped_count += 1
        continue
    # tally created/updated/unchanged per D-05
```

**Summary line (D-05, copy shape verbatim, source: `import_campaign_csv.py:399-408`, plain int counters, no dict/dataclass wrapper):**
```python
self.stdout.write(
    f'Done. created: {created_count}, '
    f'updated: {updated_count}, '
    f'unchanged: {unchanged_count}, '
    f'skipped: {skipped_count}'
)
```
Under `--dry-run`, swap in `would_create`/`would_update`/`would_unchanged` counters exactly as `backfill_range_calendar_events.py:90-95` swaps in `would_backfill_count`/`candidates`/`skipped_count` — no writes issued in that branch. Per-row skip reasons go to `self.stderr.write(...)` per row (D-05's "itemized, not a bare count"), matching `import_campaign_csv.py:152-157`'s convention — do not accumulate into an in-memory structured list.

---

### `solsys_code/campaign_views.py` (controller, request-response) — MODIFIED

**Analog:** itself, pre-change (`campaign_views.py:556-900`, `CampaignRunDecisionView.post()`/`_resolve_site()`/`_set_run_status()`, read in full above)

**Auth pattern (unchanged):** `StaffRequiredMixin`, `http_method_names = ['post']` — no change needed, D-04 confirms access control is untouched.

**`approve()` call-site rewire** — replace the existing swallow-ValueError block (source: lines ~640-651) with the identical shape, `campaign_reconciler.reconcile_run` substituted for `_project_calendar_event`:
```python
try:
    reconcile_run(run)
except ValueError:
    logger.debug(
        'Reconcile skipped for CampaignRun %s on approve (sun_event ValueError, e.g. blank site timezone).',
        pk,
    )
```
The outer `except Exception:` revert-to-PENDING_REVIEW block (lines ~630-660, wrapping both site resolution and projection) stays as-is — `reconcile_run()`'s uncaught non-ValueError exceptions still fall through to it.

**`_resolve_site()` call-site rewire** — replace the non-reverting try/except (source: lines ~803-810) verbatim, function name substituted:
```python
try:
    created = reconcile_run(run)  # exact return-value contract at planner's discretion — D-04 requires distinguishing "attempted but failed" from success, matching _project_calendar_event()'s bool-return precedent
except Exception:
    logger.exception('Reconcile failed for CampaignRun %s during resolve_site.', pk)
    messages.warning(
        request,
        "Site resolved, but the calendar entry couldn't be created automatically -- "
        'the run stays in Sites Needing Review; use Resolve to retry.',
    )
    return redirect('campaigns:approval_queue')
```
`site_needs_review = False` is still only cleared after this call returns without raising — ordering is load-bearing per the existing docstring, unchanged by the rewire.

**`_set_run_status()` call-site rewire** — the existing per-event `insert_or_create_calendar_event()` sync loop (source: lines ~886-905) and its `CAMPAIGN:`-keyed `matching_events` query (source: lines ~876-878) both get replaced by a single `reconcile_run(run)` call, wrapped in the same non-reverting try/except this method already uses (D-04: "follow whichever of the two existing patterns they already structurally resemble").

---

### `solsys_code/calendar_utils.py` (utility, CRUD) — MODIFIED (new small public helper for D-02)

**Analog:** `insert_or_create_calendar_event()` / `_update_or_unchanged()` (`calendar_utils.py:461-542`, read in full above)

**Pattern to extend, not reimplement** — per Open Question 2's recommendation, add a small new public function alongside `insert_or_create_calendar_event()` that fetches a specific already-known `CalendarEvent` (via the `CalendarEventMeta` companion row, not a `lookup` dict) and re-keys its `url` + refreshes its other fields through the existing no-churn path:
```python
def update_calendar_event_key_and_fields(event: CalendarEvent, url: str, fields: dict[str, Any]) -> tuple[CalendarEvent, str]:
    """Re-key an existing CalendarEvent's url and refresh its fields, no-churn (D-02)."""
    all_fields = {**fields, 'url': url}
    return _update_or_unchanged(event, all_fields)
```
This reuses `_update_or_unchanged()` (already private/internal to this module — calling it from within `calendar_utils.py` itself, as this new function does, keeps it internal; do NOT call `_update_or_unchanged()` directly from `campaign_reconciler.py`, which is the cross-module private-import anti-pattern the milestone's locked constraints and `backfill_range_calendar_events.py`'s own docstring call out).

---

### `solsys_code/tests/test_campaign_reconciler.py` (test, unit) — NEW

**Analog:** `solsys_code/tests/test_campaign_approval.py`'s `TestCalendarProjection`/`TestCalendarNoChurn` classes (fixture and patch-target style) — read their class names/patch targets via the retirement checklist in RESEARCH.md "Code Examples" #1; also structurally mirrors the now-deleted `test_backfill_range_calendar_events.py` (124 lines) for its dry-run/summary assertion shape.

**Fixture pattern:** plain `Observatory.objects.create(...)` with a resolvable `timezone`, plain `CampaignRun.objects.create(...)` — no `tom_targets.Target` fixture needed unless a `CampaignRunObservation`/`ObservationRecord` test requires one, in which case use `tom_targets.tests.factories.NonSiderealTargetFactory` (CLAUDE.md rule — never `SiderealTargetFactory`).

**Patch-target pattern (post-D-01):** `patch('solsys_code.campaign_reconciler.sun_event', ...)` / `patch('solsys_code.campaign_reconciler.insert_or_create_calendar_event', ...)` — NOT the now-deleted `solsys_code.campaign_views.*` targets these tests currently use.

---

### `solsys_code/tests/test_campaign_approval.py` (test, request-response) — MODIFIED (Pitfall 3, mechanical but non-trivial)

**Analog:** itself, pre-change.

**What must change (RESEARCH.md Pitfall 3, exhaustive):**
- Every `CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}')`-style assertion → `url=f'RUN:{run.pk}'`.
- Every `patch('solsys_code.campaign_views.sun_event', ...)` / `patch('solsys_code.campaign_views._project_calendar_event', ...)` / `patch('solsys_code.campaign_views.insert_or_create_calendar_event', ...)` → the equivalent `solsys_code.campaign_reconciler.*` target.
- Classes affected: `TestCalendarProjection`, `TestSitesNeedingReview` (calendar-projection assertions only), `TestRunStatusChange`, `TestCalendarNoChurn`.

---

## Shared Patterns

### No-churn create-or-update
**Source:** `solsys_code/calendar_utils.py:482-542` (`insert_or_create_calendar_event`), `:461-479` (`_update_or_unchanged`)
**Apply to:** `campaign_reconciler.py`'s every write path (Patterns 1-2 above), the new `calendar_utils.py` D-02 helper.
```python
event, created = CalendarEvent.objects.get_or_create(**lookup, defaults=fields)
if created:
    return event, 'created'
return _update_or_unchanged(event, fields)
```

### Ownership-scoping query (`Q(url=...) | Q(url__startswith=...)`)
**Source:** `solsys_code/campaign_views.py:876-878` (also `:797`, `:1317` for the companion-FK variant)
**Apply to:** `campaign_reconciler.py`'s RECON-05 "never touch an event this run doesn't own" guard, and the new command's/staff-actions' shared lookups. `CAMPAIGN:` literal replaced with `RUN:` everywhere.

### created/updated/unchanged/skipped-with-reason summary shape
**Source:** `solsys_code/management/commands/import_campaign_csv.py:399-408` (line shape), `solsys_code/management/commands/backfill_range_calendar_events.py:90-102` (`--dry-run` counter-substitution pattern)
**Apply to:** `reconcile_campaign_runs.py`'s final `self.stdout.write(...)` line and its `--dry-run` branch. Plain int locals, one final line, per-row skip reasons via `self.stderr.write(...)` — no dict/dataclass wrapper (Don't Hand-Roll table, RESEARCH.md).

### `sun_event()` dip-corrected classical math (unchanged, reused verbatim)
**Source:** `solsys_code/telescope_runs.py:251-299`
**Apply to:** `campaign_reconciler.py`'s classical per-night branch — always `kind='sun'`, never `kind='dark'`; raises `ValueError` on blank `Observatory.timezone` or a non-2-crossing night — this `ValueError` must propagate uncaught out of `reconcile_run()` per D-06 (see "Anti-Patterns" note above).

### Batch loop with per-run try/except-continue (never `transaction.atomic()` per-run)
**Source:** `solsys_code/management/commands/backfill_range_calendar_events.py:63-88`
**Apply to:** `reconcile_campaign_runs.py`'s main loop (D-06: one run's failure is caught only at this level, batch continues to the next run).

## No Analog Found

None — every file in scope has at least a role-match analog already in the codebase; this phase is explicitly a refactor of an already-well-understood, already-tested projection path (RESEARCH.md Summary).

## Metadata

**Analog search scope:** `solsys_code/campaign_views.py`, `solsys_code/calendar_utils.py`,
`solsys_code/management/commands/backfill_range_calendar_events.py`,
`solsys_code/management/commands/import_campaign_csv.py`, `solsys_code/models.py`,
`solsys_code/telescope_runs.py`, `solsys_code/tests/test_campaign_approval.py` (class/patch
inventory), `docs/runbooks/telescope_runs_calendar.rst`.
**Files scanned:** 8 (all read directly, not inferred; excerpts above are copied verbatim from
the live repository, not reconstructed from memory).
**Pattern extraction date:** 2026-08-04
