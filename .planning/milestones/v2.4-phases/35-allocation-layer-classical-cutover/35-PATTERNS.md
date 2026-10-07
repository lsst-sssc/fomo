# Phase 35: Allocation Layer & Classical Cutover - Pattern Map

**Mapped:** 2026-09-12
**Files analyzed:** 13 (create/modify) + 3 test files + 2 notebooks + 1 runbook
**Analogs found:** 13 / 13 (all git-tracked source; verified via `git ls-files`)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|---------------|
| `solsys_code/allocation_projector.py` (new) | service | event-driven / CRUD (calendar rows) | `solsys_code/campaign_reconciler.py` (`_reconcile_classical_nights`, being deleted) | exact — this *is* its replacement |
| `solsys_code/campaign_reconciler.py` (modify: dispatch only) | service | event-driven | itself — `reconcile_run()` (lines 597-628) | exact |
| `solsys_code/campaign_utils.py` (unchanged, reused) | service | CRUD | `write_and_reconcile_campaign_run()` / `adopt_event_into_run()` (lines 949-1057) | exact |
| `solsys_code/telescope_runs.py` (modify: proposal token) | utility | transform | itself — `parse_run_line()` / `ParsedRun` / `KNOWN_STATUSES` | exact |
| `solsys_code/models.py` (modify: `CampaignRun` +2 fields) | model | CRUD | itself — existing `window_start`/`window_end` `DateField` pair (lines ~248-249) | exact |
| `solsys_code/migrations/00XX_*.py` (new) | migration | batch (schema only) | `solsys_code/migrations/0017_calendareventmeta_observation_links.py` | exact |
| `solsys_code/apps.py` (modify: 2 new receivers) | config/provider | event-driven | itself — `SolsysCodeConfig.ready()` (lines 7-45) | exact |
| `solsys_code/observation_projector.py` (modify: re-project linked runs step) | service | event-driven | itself — `receiver_on_record_save()` (lines ~570-608) | exact |
| `solsys_code/management/commands/load_telescope_runs.py` (rewrite) | controller (mgmt command) | batch / file-I/O | itself (current direct-write version) + `write_and_reconcile_campaign_run()` as the new core call | exact |
| `solsys_code/management/commands/cutover_classical_allocations.py` (new) | controller (mgmt command) | batch / transform | `solsys_code/management/commands/repair_stale_campaign_run_sites.py` | exact — same "one-time idempotent data-repair command with `--dry-run`" shape |
| `solsys_code/management/commands/reconcile_campaign_runs.py` (unchanged, first-sweep takeover) | controller (mgmt command) | batch | itself | exact |
| `solsys_code/tests/test_allocation_projector.py` (new) | test | — | `solsys_code/tests/test_campaign_reconciler.py` (`TestClassicalStage1`, `TestObservingNightBoundary`) | exact |
| `solsys_code/tests/test_load_telescope_runs.py` (extend) | test | — | itself | exact |
| `solsys_code/tests/test_campaign_reconciler.py` (extend, dispatch-only) | test | — | itself | exact |
| `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` (update) | doc/notebook | — | itself (prior version) | exact |
| `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` (update) | doc/notebook | — | itself (prior version) | exact |
| `docs/runbooks/telescope_runs_calendar.rst` (update) | doc | — | itself | exact |

All analog paths above are confirmed git-tracked (`git ls-files` returned non-empty for every one).

## Pattern Assignments

### `solsys_code/allocation_projector.py` (new service, event-driven)

**Analog:** `solsys_code/campaign_reconciler.py` (module docstring lines 27-37, `_observing_night()` lines 315-343, `_reconcile_classical_nights()` lines 380-479 — the branch being ported/rewritten, NOT copied verbatim) plus `solsys_code/calendar_utils.py` (`insert_or_create_calendar_event`, `update_calendar_event_key_and_fields`, `preview_calendar_event_action`, lines 585-699) plus spike `sources/003-allocation-night-retirement/spike.py` (lines 75-112, the shape to keep — retire-if-linked / mint-if-not / attribute-via-meta-only).

**Module-level constraint to copy verbatim** (`campaign_reconciler.py:27-30`):
```python
# this module must NEVER import the views module or the heavy SPICE-loading ephemeris module
```
Copy this exact comment/constraint into the new module's docstring — D-09 requires it explicitly.

**Site-local night anchor — promote, don't reimplement** (`campaign_reconciler.py:315-343`):
```python
def _observing_night(start_time: datetime, site_zone: ZoneInfo):
    """The site-local observing night a ``start_time`` belongs to, anchored at local noon."""
    local = start_time.astimezone(site_zone)
    return (local - timedelta(hours=12)).date()
```
D-05 requires promoting this to a shared public helper (e.g. move to `telescope_runs.py` next to `sun_event()`) and importing it from both `campaign_reconciler.py` and the new `allocation_projector.py` — do not paste a second copy.

**sun_event()-only-on-mint discipline (D-13) — the anti-pattern to avoid** (`campaign_reconciler.py:433-477`, current/retiring code, for contrast only):
```python
for i in range(n_nights):
    ...
    active_urls.add(url)
    sunset, sunrise = sun_event(run.site, night, kind='sun')   # WRONG: runs even when existing is not None
    ...
    if existing is None:
        ...
```
The new module must move the `sun_event()` call inside the `if existing is None` (or "sub-night fields changed") branch instead — never call it for an unchanged existing night.

**Retire/restore/attribute shape (proven, spike 003)** — `.claude/skills/spike-findings-fomo_devel/sources/003-allocation-night-retirement/spike.py:75-112`; treat only the *shape* as proven (retire-if-linked, mint-if-not, attribute-via-meta-only) — night derivation and field content are superseded by D-05/D-08/D-12:
```python
def project_allocation(run: CampaignRun) -> dict[str, int]:
    counters = {'created': 0, 'updated': 0, 'unchanged': 0, 'retired': 0}
    taken = linked_nights(run)          # site-local nights of every linked, placed/observed record
    for night in nights_of(run):
        url = alloc_url(run, night)     # f'ALLOC:{run.pk}:{night.isoformat()}'
        if night in taken:
            deleted, _ = CalendarEvent.objects.filter(url=url).delete()
            counters['retired'] += int(deleted > 0)
            continue
        start, end = sunset_sunrise(run.site, night)   # sun_event() — only reached here
        fields = {...}
        _event, action = insert_or_create_calendar_event({'url': url}, fields)
        counters[action] += 1
    for link in run.observation_links.select_related('observation_record'):
        rec = link.observation_record
        base = CalendarEvent.objects.filter(url=projector.event_url(rec, projector.facility_for(rec))).first()
        if base is not None:
            meta, _ = CalendarEventMeta.objects.get_or_create(event=base)
            if meta.run_id != run.pk:
                meta.run = run
                meta.save(update_fields=['run'])
    return counters
```

**No-churn write helpers (reuse, don't reimplement)** — `calendar_utils.py:678-698`:
```python
def preview_calendar_event_action(event: CalendarEvent | None, fields: dict[str, Any]) -> str:
    if event is None:
        return 'created'
    changed = [f for f, v in fields.items() if getattr(event, f) != v]
    return 'updated' if changed else 'unchanged'
```
`insert_or_create_calendar_event()` (`calendar_utils.py:585-645`) is the create-or-update-or-unchanged call for every `ALLOC:{pk}:{night}` write; `update_calendar_event_key_and_fields()` (`calendar_utils.py:648`) is the in-place re-key call the cutover command (and D-16's takeover) must use.

**Attribution writers — never touch `CalendarEvent` fields directly.** Use only:
- `campaign_utils.adopt_event_into_run(event, run)` (lines 949-980) — link, refuses when already attributed elsewhere:
```python
def adopt_event_into_run(event: CalendarEvent, run: CampaignRun) -> bool:
    meta = CalendarEventMeta.objects.filter(event=event).first()
    if meta is not None and meta.run_id is not None and meta.run_id != run.pk:
        return False
    meta, _created = CalendarEventMeta.objects.get_or_create(event=event)
    if meta.run_id != run.pk:
        meta.run = run
        meta.save(update_fields=['run'])
    return True
```
- `campaign_utils.unlink_event_from_run()` / `UNLINK_CLEARED_FIELDS` — the unlink counterpart.

---

### `solsys_code/campaign_reconciler.py` (modify — dispatch only, D-09/D-10)

**Analog:** itself, `reconcile_run()` (lines 597-628, read in full this session):
```python
def reconcile_run(run: CampaignRun, *, dry_run: bool = False) -> ReconcileResult:
    reason = _skip_reason(run)
    if reason is not None:
        return ReconcileResult(skipped_reason=reason)

    if run.telescope_class:
        result = _reconcile_container(run, dry_run=dry_run)
        active_urls = {run_container_url(run)}
    elif run.site is not None and run.site.observations_type == Observatory.SATELLITE_OBSTYPE:
        result = _reconcile_container(run, dry_run=dry_run)
        active_urls = {run_container_url(run)}
    else:
        result, active_urls = _reconcile_classical_nights(run, dry_run=dry_run)
    ...
```
D-10 adds one more `elif run.source in {LCO_QUEUE, SOAR_QUEUE, GEMINI_QUEUE, ESO_QUEUE}: → _reconcile_container(...)` branch (same shape as the two existing container branches) above the final `else`, which now calls `allocation_projector.project_allocation(run, dry_run=dry_run)` instead of `_reconcile_classical_nights()`. `_reconcile_classical_nights()` and `run_night_url()` are deleted entirely (D-09). The convergence step at the bottom of `reconcile_run()` (CR-01 detach logic) is unchanged in shape but D-14 adds an ALLOC-delete branch alongside its existing detach-only `RUN:` logic.

---

### `solsys_code/telescope_runs.py` (modify — proposal token, D-01)

**Analog:** itself — `parse_run_line()` / `ParsedRun` / `KNOWN_STATUSES` (existing "raise `ValueError` on anything unrecognized" discipline, referenced at `telescope_runs.py:361-393` / `477-490`). Extend `ParsedRun` with a `proposal: str | None` field following the same dataclass-field convention already used for `start_window`/`end_window`; extend the parser's token-disambiguation `if/elif` chain (status word / `BoN` / `EoN` / `HHMM` / now proposal) with the same "raise, don't guess" error handling already in place — do not add a silent fallback.

---

### `solsys_code/models.py` (modify — `CampaignRun` D-04 fields)

**Analog:** itself — the existing `window_start` / `window_end` field pair (`models.py:~248-249`):
```python
window_start = models.DateField(null=True, blank=True, verbose_name='Observing window start')
window_end = models.DateField(null=True, blank=True, verbose_name='Observing window end')
```
Copy this null/blank/verbose_name shape for the two new sub-night fields (planner's discretion on name/type — research recommends `TimeField(null=True, blank=True)`, e.g. `night_start_utc` / `night_end_utc`). Follow the same Google-style field-comment convention used throughout this class (see the long comment blocks above `telescope_class`, `source`, `site_needs_review` for the "why this is permanent/nullable" documentation style CLAUDE.md and this file's own convention expect).

---

### `solsys_code/migrations/00XX_campaignrun_night_window_fields.py` (new, additive-only)

**Analog:** `solsys_code/migrations/0017_calendareventmeta_observation_links.py` (full file, read this session) — a small `AddField`-only migration with no `RunPython` step:
```python
class Migration(migrations.Migration):
    dependencies = [
        ('solsys_code', '0017_calendareventmeta_observation_links'),
    ]
    operations = [
        migrations.AddField(
            model_name='campaignrun',
            name='night_start_utc',
            field=models.TimeField(blank=True, null=True, verbose_name='...'),
        ),
        migrations.AddField(
            model_name='campaignrun',
            name='night_end_utc',
            field=models.TimeField(blank=True, null=True, verbose_name='...'),
        ),
    ]
```
D-15 explicitly forbids a `RunPython` data step in this migration — the cutover is a separate management command, not a migration.

---

### `solsys_code/apps.py` (modify — 2 new `CampaignRunObservation` receivers, D-11)

**Analog:** itself — `SolsysCodeConfig.ready()` (lines 7-45, full excerpt read this session):
```python
def ready(self):
    from django.db.models.signals import m2m_changed, post_save, pre_delete
    from tom_observations.models import ObservationGroup, ObservationRecord

    from solsys_code.observation_projector import (
        receiver_on_group_membership_changed,
        receiver_on_record_delete,
        receiver_on_record_save,
    )

    post_save.connect(
        receiver_on_record_save,
        sender=ObservationRecord,
        weak=False,
        dispatch_uid='solsys_code.observation_projector.post_save',
    )
    m2m_changed.connect(...)
    pre_delete.connect(...)
```
Add two new `post_save`/`post_delete` connections on `CampaignRunObservation`, same `weak=False` + unique `dispatch_uid` (e.g. `'solsys_code.allocation_projector.campaign_run_observation.post_save'`) + function-local import shape, calling new receiver functions defined in `allocation_projector.py`.

**Never-raise receiver body pattern** (`observation_projector.py:570-600`, read this session):
```python
def receiver_on_record_save(sender, instance, created, raw, **kwargs):
    if raw:
        return
    if instance.facility not in PROJECTED_FACILITIES:
        return
    try:
        action, stage = project_record(instance)
    except Exception as exc:  # noqa: BLE001 -- TRIG-02: never abort the caller's save
        logger.warning('receiver_on_record_save failed for observation_id=%r: %s',
                        instance.observation_id, type(exc).__name__)
        return
```
Copy this exact shape (raw-guard, try/except Exception, log-and-return, never re-raise) for both new `CampaignRunObservation` receivers.

---

### `solsys_code/observation_projector.py` (modify — re-project linked runs step, D-11)

**Analog:** itself — same `receiver_on_record_save()` function above; append a step after `project_record()` succeeds: iterate `instance.campaign_run_links.all()` and call `allocation_projector.project_allocation(link.run)` inside the same try/except-never-raise body (function-local import of `allocation_projector` to avoid any import-cycle, matching the existing lazy-import convention this module already uses).

---

### `solsys_code/management/commands/load_telescope_runs.py` (rewrite, ALLOC-04)

**Analog:** itself (current version, read in full this session, lines 1-240) — the parts to keep vs. remove:

Keep unchanged: `_resolve_window_time()` (window-token → UTC datetime), the night-iteration logic (`_iter_run_nights()`), `parse_run_line()` call, `get_site()` call, per-line try/except around `(ValueError, Observatory.DoesNotExist)`:
```python
except (ValueError, Observatory.DoesNotExist) as exc:
    self.stderr.write(f'Line {line_num}: {exc} (line text: {line.strip()!r})')
```

Remove: the direct `insert_or_create_calendar_event()` call (lines ~208-232) and its `_START_TIME_MATCH_TOLERANCE` proximity-match usage (that concern moves into the run's own idempotent lookup via `source_identifier`), and `_CLASSICAL_STATUS_PREFIX` (superseded by `RUN_STATUS_CALENDAR_PREFIX` in `campaign_reconciler.py:76-79` once status maps onto `run_status`, D-03).

New core call, following `write_and_reconcile_campaign_run()`'s existing contract (`campaign_utils.py:995-1057`, read this session):
```python
result = write_and_reconcile_campaign_run(
    {'source_identifier': source_identifier},
    {
        'source': CampaignRun.Source.CLASSICAL_FILE,
        'approval_status': CampaignRun.ApprovalStatus.APPROVED,
        'campaign': campaign,  # TargetList or None
        'target': None,
        'site': site,
        'site_raw': parsed.telescope,
        'telescope_instrument': f'{parsed.telescope}/{parsed.instrument}',
        'window_start': window_start,
        'window_end': window_end,
        'night_start_utc': ...,  # from BoN/EoN/HHMM per line
        'night_end_utc': ...,
        'run_status': status_map[parsed.status],
    },
)
action = result.action           # created / updated / unchanged
reconcile = result.reconcile     # ReconcileResult — report alongside action
```

---

### `solsys_code/management/commands/cutover_classical_allocations.py` (new, D-15/D-17/D-18)

**Analog:** `solsys_code/management/commands/repair_stale_campaign_run_sites.py` (full docstring and shape read this session) — the exact "one-time, idempotent, `--dry-run`-able, reports-what-it-cannot-explain" precedent:
```python
"""One-time data repair (D-16): re-resolve stale, site-less approved CampaignRuns.
...
``--dry-run`` performs read-only checks and writes nothing...
"""
import logging
from django.core.management.base import BaseCommand, CommandParser
logger = logging.getLogger(__name__)
```
Copy this module-docstring style (explain *why* the command exists, what it deliberately does NOT touch, the `--dry-run` contract) and its `BaseCommand` + `CommandParser` argument-adding shape (`add_argument('--dry-run', action='store_true', ...)`). D-18's "report and exit non-zero, never delete what it can't explain" maps directly onto this analog's own `skipped_*` counter convention (`skipped_class_wide`, `skipped_no_site_code`) — add a `skipped_unparseable` counter and end `handle()` with `raise CommandError(...)` or `self.stderr.write(...); sys.exit(1)` when that counter is nonzero.

Re-parsing logic reuses `telescope_runs.parse_run_line()` + `get_site()` (same functions `load_telescope_runs` uses) against each blank-`url` event's `Source line:` description line; re-keying reuses `calendar_utils.update_calendar_event_key_and_fields()`.

---

### `solsys_code/tests/test_allocation_projector.py` (new)

**Analog:** `solsys_code/tests/test_campaign_reconciler.py` — `TestClassicalStage1` (lines 1028-1147) for the per-night create/update/unchanged pattern, and `TestObservingNightBoundary` (lines 499-675, already exercises a real `Australia/Sydney` fixture `F65`) for the site-local night boundary pattern. Add a parallel Chilean fixture (`timezone='America/Santiago'`, a fresh test-created `Observatory` row, not a dependency on dev-DB content — per every existing test class's own convention) alongside `CampaignReconcilerTestBase` (lines 35-71).

Mock-based no-recompute regression test (the folded todo, D-13): `unittest.mock.patch('solsys_code.allocation_projector.sun_event')` and assert `not_called()` on the second reconcile of an unchanged multi-night run.

---

### `solsys_code/tests/test_load_telescope_runs.py` (extend)

**Analog:** itself — existing tests `test_idempotent_rerun_no_duplicates` (line 218) and `test_reingest_with_drifted_sun_event_does_not_duplicate` (line 252) must be revised to assert `CampaignRun` create-or-update behavior (via `write_and_reconcile_campaign_run`) instead of direct `CalendarEvent` counts.

## Shared Patterns

### Ownership by key namespace
**Source:** `campaign_reconciler.py:27-37` (module docstring), spike 003
**Apply to:** `allocation_projector.py`, the cutover command, `reconcile_run()`
`ALLOC:` is the allocation projector's alone; `RUN:{pk}` stays the reconciler's; observation URLs are the observation projector's. Never cross-write another namespace's events.

### Never-raise signal receivers, wired in `ready()`
**Source:** `apps.py:7-45`, `observation_projector.py:570-608`
**Apply to:** the two new `CampaignRunObservation` receivers, the observation projector's re-project-linked-runs step
`weak=False`, unique `dispatch_uid`, `raw`-guard, `try/except Exception: log and return` — never let a projector fault break the caller's save/delete.

### Attribution is a link, never a field write
**Source:** `campaign_utils.py:949-1057` (`adopt_event_into_run`, `unlink_event_from_run`), `campaign_reconciler.py:32-37`
**Apply to:** `allocation_projector.py`'s per-run attribution step, D-08's link/unlink handling
A human-confirmed attribution outranks any automated writer; never write `title`/`description`/`start_time`/`end_time` on an observation-record-derived event from the campaign/allocation side.

### No-churn create/update/preview via `calendar_utils`
**Source:** `calendar_utils.py:585-699`
**Apply to:** `allocation_projector.py`, the cutover command's re-key step
`insert_or_create_calendar_event()`, `update_calendar_event_key_and_fields()`, `preview_calendar_event_action()` — never re-implement get_or_create/diff-and-save logic.

### One-time, idempotent, `--dry-run`-able management command
**Source:** `management/commands/repair_stale_campaign_run_sites.py` (full file)
**Apply to:** `cutover_classical_allocations.py`
Explain in the docstring what the command deliberately does NOT touch; never delete what it cannot explain — report and exit non-zero instead (D-18).

## No Analog Found

None — every file in scope has a strong, git-tracked analog in the existing codebase (this phase is explicitly "internal Django app work extending Phases 26-34 patterns," per RESEARCH.md's own summary).

## Metadata

**Analog search scope:** `solsys_code/` (root-level app), `solsys_code/management/commands/`, `solsys_code/tests/`, `solsys_code/migrations/`, `docs/notebooks/pre_executed/`, `docs/runbooks/`
**Files scanned:** `campaign_reconciler.py`, `campaign_utils.py`, `calendar_utils.py`, `telescope_runs.py`, `models.py`, `apps.py`, `observation_projector.py`, `load_telescope_runs.py`, `reconcile_campaign_runs.py`, `repair_stale_campaign_run_sites.py`, `0017_calendareventmeta_observation_links.py`, `test_campaign_reconciler.py`, `test_load_telescope_runs.py`, plus the spike-findings skill's `spike.py`
**Pattern extraction date:** 2026-09-12
