---
phase: 35-allocation-layer-classical-cutover
reviewed: 2026-09-13T00:00:00Z
depth: deep
files_reviewed: 27
files_reviewed_list:
  - CLAUDE.md
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/allocation_projector.py
  - solsys_code/apps.py
  - solsys_code/campaign_reconciler.py
  - solsys_code/campaign_utils.py
  - solsys_code/campaign_views.py
  - solsys_code/management/commands/cutover_classical_allocations.py
  - solsys_code/management/commands/load_telescope_runs.py
  - solsys_code/management/commands/reconcile_campaign_runs.py
  - solsys_code/migrations/0018_campaignrun_night_window_fields.py
  - solsys_code/models.py
  - solsys_code/observation_projector.py
  - solsys_code/telescope_runs.py
  - solsys_code/tests/test_allocation_projector.py
  - solsys_code/tests/test_allocation_projector_signals.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/tests/test_campaign_models.py
  - solsys_code/tests/test_campaign_reconciler.py
  - solsys_code/tests/test_cutover_classical_allocations.py
  - solsys_code/tests/test_load_telescope_runs.py
  - solsys_code/tests/test_observation_projector.py
  - solsys_code/tests/test_observation_projector_signals.py
  - solsys_code/tests/test_project_observation_calendar.py
  - solsys_code/tests/test_reconcile_campaign_runs.py
  - solsys_code/tests/test_telescope_runs.py
  - solsys_code/tests/test_write_and_reconcile.py
findings:
  critical: 6
  warning: 11
  info: 0
  total: 17
status: issues_found
---

# Phase 35: Code Review Report

**Reviewed:** 2026-09-13
**Depth:** deep
**Files Reviewed:** 27
**Status:** issues_found

## Summary

Phase 35 introduces a third calendar-event key family (`ALLOC:{run_pk}:{night}`) alongside the
two `campaign_reconciler` already owned (`RUN:{pk}` container, retired `RUN:{pk}:{date}`
per-night), plus three new write triggers that reach it directly from signal receivers. The
module is carefully documented and `pre-commit run ruff` is clean on every changed file.

The defects cluster in one place: **the `ALLOC:` namespace has neither a dispatch guard on the
way in nor a convergence step on the way out, and its two delete paths skipped the ownership
and human-confirmation guards the peer `RUN:` paths apply.** Six of these are demonstrated
below with executed probe tests run against a real Django test database (`python manage.py
test`, probe module written and removed; no source file was modified by this review).

Concretely, the following are reproduced facts, not inferences:

- A **container-dispatched** run (queue `source`, or `telescope_class` set) acquires a full set
  of `ALLOC:` nights the moment an observation link or a record save fires, on top of its
  `RUN:{pk}` container — and **no later `reconcile_run()` ever removes them** (CR-01, CR-02).
- An **unapproved** (`pending_review`) web submission acquires `ALLOC:` calendar entries the
  same way, bypassing `_skip_reason()`'s approval gate entirely (CR-01).
- The retire path **deletes a `RUN:{pk}:{night}` event that a staff member confirmed**, and one
  attributed to a *different* run, with no `_may_write()` and no `confirmed_by` check —
  directly contrary to D-08 / UAT Option B (CR-03).
- The final convergence step **deletes a stale `ALLOC:` event attributed and confirmed to a
  different run**, because it filters on `allocation_events()` (namespace identity) rather than
  the `writable_allocation_events()` helper written in the same module for exactly this (CR-04).
- An **admin bulk delete** of `CampaignRun` rows (`queryset.delete()`) escapes the new
  `origin`-based cascade guard, leaving orphaned `ALLOC:` events and a **dangling
  `CalendarEventMeta.run_id` foreign key** — SQLite's end-of-test constraint check flagged it;
  PostgreSQL would raise `IntegrityError` and fail the delete (CR-05).
- The sub-night window rule inverts for **Australia/Sydney** (FTS): a `0930-1900` UTC window on
  night 2026-08-01 produces `start_time=2026-08-02T09:30Z`, `end_time=2026-08-01T19:00Z` — a
  calendar event whose start is 14.5 hours after its end (CR-06).

Positives worth recording: the cutover command's `--dry-run` was verified to write nothing (no
`CampaignRun` rows, no `url` changes, no `modified` bumps); the `raw=True` fixture guard, the
never-raise wrapping and the `dispatch_uid` de-duplication in `apps.py` are all correct; the
`_sync_observation_attribution()` unlink half correctly pre-filters on
`confirmed_by__isnull=True`; and no code path infers a run's `source` from a telescope name,
site or event text.

## Structural Findings (fallow)

No `<structural_findings>` block was supplied with this review request.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Signal receivers call `project_allocation()` directly, bypassing `reconcile_run()`'s dispatch and approval gates

**File:** `solsys_code/allocation_projector.py:576-580`, `solsys_code/allocation_projector.py:645-649`, `solsys_code/observation_projector.py:621-625`
**Severity:** BLOCKER

**Issue:** All three new triggers call `project_allocation(run)` directly. `project_allocation()`'s
own docstring states its precondition — *"Must have a resolved `site`, an approved status and a
non-null `window_start`/`window_end` (`reconcile_run()`'s stage-0 guard, `_skip_reason()`,
already enforces this before dispatch)"* — but none of the three triggers go through
`reconcile_run()`, so `_skip_reason()` and the four-way dispatch in
`campaign_reconciler.reconcile_run():608-640` are both skipped.

Two consequences, both reproduced:

1. **Approval-gate bypass.** A `source=web`, `approval_status=pending_review` run with a resolved
   site and window gets `ALLOC:` calendar entries as soon as a `CampaignRunObservation` row is
   created (which `CampaignRunAdmin`'s inline can do). Probe output:

   ```
   SKIP REASON: not approved   alloc= 0        # reconcile_run() correctly refuses
   UNAPPROVED AFTER LINK: alloc= 3 ['ALLOC:1:2026-08-01', 'ALLOC:1:2026-08-02', 'ALLOC:1:2026-08-03']
   UNAPPROVED AFTER 2ND RECONCILE: not approved 3   # and never cleaned up
   ```

   A public, unreviewed submission reaching the shared calendar is precisely what the approval
   gate exists to prevent.

2. **Dispatch bypass.** A `source=lco_queue` run with a resolved ground site is D-10
   container-dispatched (`campaign_reconciler.py:618-629`). Linking its observation record —
   or merely saving that LCO/SOAR record, via
   `observation_projector.receiver_on_record_save()` — mints the whole per-night set anyway:

   ```
   AFTER RECONCILE:   alloc= 0  container= 1
   AFTER LINK:        alloc= 3  ['ALLOC:1:2026-08-01', 'ALLOC:1:2026-08-02', 'ALLOC:1:2026-08-03']
   AFTER RECORD SAVE: alloc= 3
   AFTER 2ND RECONCILE: alloc= 3      # the sweep never removes them
   ```

   The phase's own notes say eight such single-night queue runs exist in the real database. Every
   subsequent portal status poll re-fires this.

**Fix:** Give the allocation projector a single guarded entry point and call it from all three
receivers, so the dispatch decision has exactly one owner:

```python
# campaign_reconciler.py -- extract the existing dispatch decision
def dispatches_per_night(run: CampaignRun) -> bool:
    """True when this run's calendar form is the per-night ALLOC: family."""
    if run.telescope_class:
        return False
    if run.site is None:
        return False
    if run.site.observations_type == Observatory.SATELLITE_OBSTYPE:
        return False
    return run.source not in {
        CampaignRun.Source.LCO_QUEUE, CampaignRun.Source.SOAR_QUEUE,
        CampaignRun.Source.GEMINI_QUEUE, CampaignRun.Source.ESO_QUEUE,
    }

# allocation_projector.py -- the only thing a receiver may call
def reproject_allocation_if_dispatched(run: CampaignRun) -> None:
    """Trigger-side entry point: honours _skip_reason() and the dispatch rule."""
    from solsys_code.campaign_reconciler import _skip_reason, dispatches_per_night
    if _skip_reason(run) is not None or not dispatches_per_night(run):
        return
    project_allocation(run)
```

Then `reconcile_run()`'s `else:` branch becomes `if dispatches_per_night(run)`, and the three
receivers call `reproject_allocation_if_dispatched(run)` instead of `project_allocation(run)`.

---

### CR-02: Re-classifying a run leaves its `ALLOC:` nights on the calendar forever — the runbook documents the opposite

**File:** `solsys_code/campaign_reconciler.py:476-581`, `solsys_code/campaign_reconciler.py:642-661`, `docs/runbooks/telescope_runs_calendar.rst:645-654`
**Severity:** BLOCKER

**Issue:** `_detach_stale_family_events()` converges only over `owned_events(run)`, which is
defined (`campaign_reconciler.py:135-142`) as the `RUN:{pk}` / `RUN:{pk}:` namespace. The
`ALLOC:` namespace is never inspected. Phase 35 added a third key family and did not extend the
convergence step to cover it, so when a run moves from the per-night branch to the container
branch its `ALLOC:` nights are silently orphaned — they duplicate the new whole-window entry and
no code path can ever reach them again (the run no longer dispatches to `project_allocation()`,
whose own `stale_qs` step is the only deleter of `ALLOC:` events).

Reproduced for both re-classification triggers:

```
per-night alloc: 3
after relabel (source -> lco_queue):
  ReconcileResult(created=1, ..., retired=0, rekeyed=0, legacy_deleted=0)
  ALLOC left: ['ALLOC:1:2026-08-01', 'ALLOC:1:2026-08-02', 'ALLOC:1:2026-08-03']
  container: 1
after class relabel (telescope_class='1m0'):
  ALLOC left: ['ALLOC:1:2026-08-01', 'ALLOC:1:2026-08-02', 'ALLOC:1:2026-08-03']
```

This contradicts shipped operator documentation. `docs/runbooks/telescope_runs_calendar.rst`
(new in this phase) tells the operator:

> *"Relabelling a per-night run's `source` to `lco_queue`/... does not change anything on the
> calendar by itself: the next reconcile ... is what converges on it, deleting the run's leftover
> per-night events (counted under `legacy_deleted`) and replacing them with a single whole-window
> entry."*

`legacy_deleted` only covers `RUN:{pk}:{date}`, the *retired* form. Post-cutover, per-night
events are `ALLOC:`-keyed, so the documented convergence does not happen for any run created or
converted by this phase.

**Fix:** Add the `ALLOC:` family to the convergence step, mirroring the existing
confirmed/foreign guards. In `campaign_reconciler.reconcile_run()`, pass the run's allocation
dispatch state down and clear the other family:

```python
# campaign_reconciler.py, inside _detach_stale_family_events()
from solsys_code.allocation_projector import writable_allocation_events

if not dispatches_per_night(run):        # container branch just ran
    stale_alloc = writable_allocation_events(run)
    alloc_ids, alloc_declined = _clearable_and_declined(run, stale_alloc)
    if alloc_ids:
        CalendarEvent.objects.filter(pk__in=alloc_ids).delete()
        legacy_deleted += len(alloc_ids)
    declined += alloc_declined
```

and mirror the same read-only count in the `dry_run` branch of `reconcile_run()` so the preview
and the real sweep still agree.

---

### CR-03: The retire path deletes a legacy `RUN:{pk}:{night}` event with no ownership and no `confirmed_by` guard

**File:** `solsys_code/allocation_projector.py:459-467`
**Severity:** BLOCKER

**Issue:**

```python
if night in retired:
    retired_urls.add(url)
    legacy_urls_claimed.add(legacy_url)
    if not dry_run:
        if existing is not None:
            existing.delete()
        CalendarEvent.objects.filter(url=legacy_url).delete()   # <-- unguarded
```

`_may_write()` was evaluated at line 453 against `existing` (the `ALLOC:`-keyed event), never
against the legacy event. Twenty lines later, the takeover branch *does* check
`_may_write(legacy_event, run)` before touching the same class of row (lines 474-481) — so this
is an inconsistency inside one function, not a deliberate policy.

The result violates both standing rules in this module's own documentation: T-29-19 (*"a
companion row that points at a DIFFERENT run means a staff member attributed that event
elsewhere ... a write path must never touch it"*) and the 2026-09-09 UAT decision B (*"a
human-confirmed attribution is never cleared by an automated sweep"*). Because this is a
`CalendarEvent.delete()`, the cascade also destroys the `CalendarEventMeta` row carrying
`confirmed_by`/`confirmed_at` — the audit trail goes with it.

Reproduced:

```
legacy still exists (human-confirmed):      False
foreign-attributed legacy still exists:     False
```

Both events were deleted by an automated re-projection fired from a `CampaignRunObservation`
save — no operator involved.

**Fix:** Apply the same guard the takeover branch applies, and count a refusal rather than
deleting:

```python
if night in retired:
    retired_urls.add(url)
    legacy_event = CalendarEvent.objects.filter(url=legacy_url).first()
    if legacy_event is not None and not _may_write(legacy_event, run):
        logger.warning(
            'Allocation retire blocked: legacy event pk=%s is not owned by run pk=%s.',
            legacy_event.pk, run.pk,
        )
        totals['blocked'] += 1
    else:
        legacy_urls_claimed.add(legacy_url)
        if not dry_run:
            if existing is not None:
                existing.delete()
            if legacy_event is not None:
                legacy_event.delete()
    totals['retired'] += 1
    continue
```

Additionally gate the delete on `confirmed_by__isnull=True` (reuse
`_clearable_and_declined()`), so a staff-confirmed night is reported under `detach_declined`
rather than destroyed.

---

### CR-04: Final convergence deletes stale `ALLOC:` events attributed to a *different* run

**File:** `solsys_code/allocation_projector.py:534-539`
**Severity:** BLOCKER

**Issue:**

```python
stale_qs = allocation_events(run).exclude(url__in=active_urls | retired_urls)
stale_count = stale_qs.count()
if stale_count:
    if not dry_run:
        stale_qs.delete()
```

`allocation_events()` is namespace identity only — its own docstring at line 89-98 states
*"namespace identity alone is NOT ownership"* and the module supplies
`writable_allocation_events()` (lines 89-103) as the attribution-scoped twin. That helper is used
by the `CampaignRun` `pre_delete` cascade but **not here**, where it matters just as much: this
is the delete path an ordinary reconcile takes.

Reproduced — a staff member re-attributes and confirms night 3 to another run, the run's window
then shrinks:

```
result: ReconcileResult(..., retired=1, ...)
foreign/confirmed ALLOC event still exists: False
```

An event confirmed by a person and attributed to run B is destroyed by run A's automated
convergence step, with no `blocked`/`detach_declined` counter raised.

**Fix:** Use the writable queryset, and report the difference instead of silently absorbing it:

```python
stale_qs = allocation_events(run).exclude(url__in=active_urls | retired_urls)
writable_stale = writable_allocation_events(run).exclude(url__in=active_urls | retired_urls)
stale_ids, declined = _clearable_and_declined(run, writable_stale)
totals['blocked'] += stale_qs.count() - writable_stale.count()
if stale_ids:
    if not dry_run:
        CalendarEvent.objects.filter(pk__in=stale_ids).delete()
    totals['retired'] += len(stale_ids)
```

(`_clearable_and_declined` is already importable from `campaign_reconciler`, alongside
`_may_write`/`_link_event_to_run`, under the module's stated private-import convention.)

---

### CR-05: The cascade guard misses queryset deletes — admin bulk delete orphans events and leaves a dangling FK

**File:** `solsys_code/allocation_projector.py:643-644`
**Severity:** BLOCKER

**Issue:**

```python
if isinstance(kwargs.get('origin'), CampaignRun):
    return
```

Django sets `origin` to the object `.delete()` was called on for `Model.delete()`, but to the
**QuerySet** for `QuerySet.delete()`. The Django admin's "Delete selected" action goes through
`ModelAdmin.delete_queryset()` → `queryset.delete()`, so `origin` is a `QuerySet`, the
`isinstance` check is False, and the receiver proceeds. `models.py:467` explicitly notes the
sibling `pre_delete` receiver was written as a signal *"so this also fires for the admin bulk
delete"* — so the bulk path is a supported, expected one.

What then happens, in Django's `Collector.delete()` order: `pre_delete` fires and
`_delete_owned_calendar_events_on_campaign_run_delete()` clears the `ALLOC:` events; the
`SET_NULL` field updates for `CalendarEventMeta.run` are applied; the `CampaignRunObservation`
rows are deleted and their `post_delete` fires — this receiver re-projects the run (which still
exists in the DB at that moment, as the docstring itself explains), **re-creating both the
`ALLOC:` events and `CalendarEventMeta` rows pointing at the run**; then the `CampaignRun` DELETE
runs.

Reproduced:

```
BEFORE DELETE:   alloc= 3
AFTER QS DELETE: run exists= False   orphan alloc= 3
django.db.utils.IntegrityError: The row in table 'solsys_code_calendareventmeta' with primary
key '5' has an invalid foreign key: solsys_code_calendareventmeta.run_id contains a value '1'
that does not have a corresponding value in solsys_code_campaignrun.id
```

On SQLite (dev) this leaves a corrupt `run_id` pointing at a nonexistent row. On PostgreSQL
(the documented production target) the FK is enforced and the whole admin bulk delete raises
`IntegrityError` and rolls back — the operator cannot delete runs at all.

**Fix:** Test the *model class* of the origin rather than the instance type, which covers both
the instance and queryset forms:

```python
origin = kwargs.get('origin')
origin_model = getattr(origin, 'model', type(origin))
if origin_model is CampaignRun or isinstance(origin, CampaignRun):
    return
```

Add a regression test using `CampaignRun.objects.filter(pk=...).delete()` (the admin path) —
the existing `test_deleting_the_run_cascades_the_link_without_raising_or_re_projecting`
only exercises `self.run.delete()`.

---

### CR-06: Sub-night window fields produce an inverted event span for Australia/Sydney (FTS)

**File:** `solsys_code/allocation_projector.py:161-177`, `solsys_code/models.py:264-283`
**Severity:** BLOCKER

**Issue:** `_time_of_day_to_datetime()` hard-codes *"an hour before 12:00 UTC belongs to the NEXT
morning for that observing night; 12:00 or later belongs to the night's own evening date."* That
rule is correct only for western-hemisphere sites where the night straddles UTC midnight (La
Silla, Las Campanas). For Siding Spring (`FTS`, `Australia/Sydney`, UTC+10/+11) the observing
night's UTC span is entirely inside one UTC date, with the **start in the morning UTC hours and
the end in the evening UTC hours** — exactly inverting the rule's assumption.

Verified against the projector's own computed sun events for the same site and night:

```
FTS FULL NIGHT ALLOC:2:2026-08-01 start= 2026-08-01T07:33:39+00:00 end= 2026-08-01T20:46:06+00:00
FTS EVENT      ALLOC:1:2026-08-01 start= 2026-08-02T09:30:00+00:00 end= 2026-08-01T19:00:00+00:00
                                                                     inverted= True
```

The computed full night is `07:33Z -> 20:46Z` on the *same* UTC date. A `night_start_utc=09:30`
run instead lands on `2026-08-02T09:30Z` — a full day late — producing a `CalendarEvent` whose
`start_time` is 14.5 hours **after** its `end_time`. Nothing validates this; the row is written,
counted as `created=1`, and rendered on the shared calendar.

The rule is carried forward verbatim from the pre-phase
`load_telescope_runs._resolve_window_time()` (confirmed against the diff base), but Phase 35
promotes it from a per-line detail into the documented semantics of two new persisted model
fields (`models.py:264-283` states the rule as the field contract) and into the per-night
re-mint comparison `_span_needs_remint()`, so it is now a durable data-model invariant rather
than a transient parse step. `FTS` is one of the four entries in `telescope_runs.SITES`, and
`Australia/Sydney` is named as a supported timezone in the project constraints.

**Fix:** Anchor the offset on the night's own computed sunset instead of a hard-coded UTC hour,
so the rule works for any longitude:

```python
def _time_of_day_to_datetime(t, night, sunset_utc: datetime) -> datetime:
    """A stored sub-night TimeField -> the UTC datetime nearest the night's own sunset."""
    candidate = datetime.combine(night, t, tzinfo=dt_timezone.utc)
    # Pick whichever of (night, night+1) puts the boundary inside the night's own span.
    if candidate < sunset_utc - timedelta(hours=1):
        candidate += timedelta(days=1)
    return candidate
```

If the sunset is not available cheaply (D-13 forbids a `sun_event()` call on the update path),
derive the anchor from `run.site` longitude or timezone offset instead — e.g. treat a UTC
time-of-day as belonging to the *following* date only when the site's local UTC offset is
negative. Either way, add a guard in `night_bounds()` that refuses to return `start >= end` and
logs/counts it rather than writing an inverted event, plus a `TestSubNightWindow` case using an
`Australia/Sydney` `Observatory`.

---

## Warnings

### WR-01: The linked-run handoff never fires for non-LCO/SOAR records

**File:** `solsys_code/observation_projector.py:604-605`, `solsys_code/observation_projector.py:621-625`
**Severity:** WARNING

**Issue:** The new D-11 linked-run re-projection block was appended *below*
`if instance.facility not in PROJECTED_FACILITIES: return` (line 604). `PROJECTED_FACILITIES` is
`('LCO', 'SOAR')`, so a Gemini or ESO `ObservationRecord` save never re-projects its linked run.
Yet `retired_nights()` (`allocation_projector.py:286-303`) applies **no facility filter at all**
and happily retires a GEM record's night when a sweep eventually runs. The trigger and the
projector therefore disagree about which records matter, and a Gemini record moving from queued
to placed leaves a stale allocation night on the calendar until the next batch sweep.

**Fix:** Move the linked-run block above the facility guard (it does not depend on
`project_record()`), or duplicate the guard's early-return into a dedicated helper:

```python
if raw:
    return
try:
    _reproject_linked_runs(instance)        # facility-independent, D-11
except Exception as exc:  # noqa: BLE001
    logger.warning('linked-run re-project failed for observation_id=%r: %s',
                   instance.observation_id, type(exc).__name__)
if instance.facility not in PROJECTED_FACILITIES:
    return
...
```

### WR-02: One `try` wraps the whole linked-run loop, so one bad run skips every later one

**File:** `solsys_code/observation_projector.py:620-628`
**Severity:** WARNING

**Issue:**

```python
try:
    for link in instance.campaign_run_links.select_related('run'):
        if link.run is not None:
            project_allocation(link.run)
except Exception as exc:  # noqa: BLE001
    ...
```

An exception on the first link aborts the loop; every remaining linked run silently goes
un-projected, and the log names only the record, not which run failed. Today a record has at
most one run (the `unique_campaign_run_observation_record` constraint), but the constraint's own
docstring in `models.py` says it is *"expressed so it can be broadened cheaply"* to many runs per
record — at which point this becomes a live partial-failure bug.

**Fix:** Move the `try` inside the loop and name the run:

```python
for link in instance.campaign_run_links.select_related('run'):
    if link.run is None:
        continue
    try:
        project_allocation(link.run)
    except Exception as exc:  # noqa: BLE001
        logger.warning('linked-run re-project failed for run pk=%s (observation_id=%r): %s',
                       link.run_id, instance.observation_id, type(exc).__name__)
```

### WR-03: `--dry-run` computes and discards two `sun_event()` calls per new night

**File:** `solsys_code/allocation_projector.py:511-523`
**Severity:** WARNING

**Issue:** On the create path, `fields = _mint_fields(run, night)` runs *before* the `if dry_run`
branch. `_mint_fields()` makes two `sun_event()` calls (lines 249-250). The dry-run branch then
calls `preview_calendar_event_action(existing=None, fields)`, which returns `'created'`
**without reading `fields` at all** (`calendar_utils.py:695-696`). The whole computation is
discarded.

Two consequences beyond wasted work: a `--dry-run` preview can now **raise** the `ValueError`
`sun_event()` throws for a blank `Observatory.timezone` (the module docstring positions dry-run
as a read-only preview and explicitly documents that `ValueError` propagating out for the
*staff-action* call sites), and a dry-run sweep over a multi-week window costs the same astropy
time as a real one.

**Fix:**

```python
if existing is None:
    if dry_run:
        totals['created'] += 1
        continue
    fields = _mint_fields(run, night)
else:
    ...
```

### WR-04: `reconcile_campaign_runs --dry-run` reports deletions in the past tense

**File:** `solsys_code/management/commands/reconcile_campaign_runs.py:85-100`
**Severity:** WARNING

**Issue:** The three new per-run messages are emitted unconditionally, with no `dry_run` branch,
in the past tense:

```
Run pk=7: 3 night(s) retired -- now covered by a real observation
Run pk=7: 12 legacy RUN:-keyed night(s) re-keyed into ALLOC: in place
Run pk=7: 8 leftover per-night event(s) deleted -- ...
```

Under `--dry-run` none of that happened, yet the summary line below correctly says
`would_retire` / `would_rekey` / `would_delete_legacy`. An operator reading the per-run stream of
a preview run is told events were deleted. The runbook explicitly instructs the operator to *"run
`--dry-run` first and read the list"*, so these are the lines they are being sent to.

**Fix:** Route the verb through the existing `dry_run` flag, matching the summary-line style:

```python
verb = 'would be retired' if dry_run else 'retired'
if result.retired:
    self.stdout.write(f'Run pk={run.pk}: {result.retired} night(s) {verb} -- ...')
```

### WR-05: The "retired" per-run message conflates three unrelated causes

**File:** `solsys_code/management/commands/reconcile_campaign_runs.py:86-91`, `solsys_code/campaign_reconciler.py:110-113`
**Severity:** WARNING

**Issue:** The message asserts `retired` means *"now covered by a real observation"*, but
`ReconcileResult.retired` is incremented from three different places in `project_allocation()`:
the D-05/D-07 observation handoff (line 466), the D-13 sub-night **re-mint** (line 502, where the
night is immediately re-created and is emphatically *not* covered by an observation), and the
D-14 window-shrink convergence (line 539). An operator reading `3 night(s) retired -- now covered
by a real observation` after editing `night_start_utc` in the admin is being told something
factually untrue.

**Fix:** Either split the counter (`retired` / `reminted` / `converged`) — `ReconcileResult` is a
`NamedTuple` with defaults, so adding fields is cheap — or drop the causal clause:
`f'{result.retired} allocation night(s) removed (observation handoff, sub-night re-mint or window change)'`.

### WR-06: `cutover_classical_allocations` has no transaction boundary and commits before raising

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:196-330`
**Severity:** WARNING

**Issue:** `handle()` creates `CampaignRun` rows and re-keys `CalendarEvent` rows group by group
with no `transaction.atomic()`, then raises `CommandError` at the very end if anything was
unexplained. Every write already committed stays committed. The command is documented as
*"safe to re-run"*, which mitigates but does not remove the hazard: an interruption (operator
Ctrl-C, connection drop, an `IntegrityError` from a path not covered by the per-group
`except Exception`) leaves a partially-converted database in which some runs exist with a subset
of their nights re-keyed and the rest still blank-url. The operator has no transactional
all-or-nothing option for a one-time production migration.

**Fix:** Wrap each group's writes in a savepoint so a failing group rolls back cleanly while
successful groups persist, which matches the command's documented per-group semantics:

```python
from django.db import transaction
...
for source_line, events in groups.items():
    try:
        with transaction.atomic():
            ...          # run write + per-event re-key for this group
    except Exception as exc:  # noqa: BLE001
        _mark_unexplained(events, _OTHER, f'{type(exc).__name__}: {exc}')
```

### WR-07: The cutover creates an empty `CampaignRun` for a group whose events are all foreign-attributed

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:252-286`
**Severity:** WARNING

**Issue:** The `_FOREIGN_ATTRIBUTION` guard marks individual events unexplained and builds
`writable_events`, but the run write at line 268 runs unconditionally afterwards. If *every*
event in the group is attributed elsewhere, `writable_events` is empty and the command still
creates a brand-new `CampaignRun` with zero events. That run is APPROVED, `CLASSICAL_FILE`, with
a resolved site and a window — so the very next `reconcile_campaign_runs` sweep mints a fresh
full set of `ALLOC:` nights for it, duplicating the nights the other run already owns. The
command's docstring promises it *"deliberately does NOT touch any event"* it cannot explain; it
says nothing about creating a run for one.

**Fix:** Skip the run write entirely when nothing is writable:

```python
if not writable_events:
    continue          # every event in this group is already attributed elsewhere
```

### WR-08: The cutover never checks the derived night lies inside the run's own window

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:289-305`
**Severity:** WARNING

**Issue:** The run's window comes from `_iter_run_nights(parsed)` (the schedule line's day range,
adjusted for the ESO noon-to-noon convention), but each event's `ALLOC:` night comes from
`observing_night(event.start_time, site_zone)` — a completely independent derivation from stored
data. Nothing asserts the two agree. When they disagree (an off-by-one ESO boundary, or an event
written under the CR-06 Sydney bug), the event is re-keyed to an `ALLOC:{pk}:{night}` url
*outside* `[window_start, window_end]`. `project_allocation()`'s convergence step
(line 534) then classifies it as stale and **deletes** it on the next sweep.

That silently converts the command's headline guarantee — *"It never removes a `CalendarEvent`
row from the database, on any path"* — into "it hands the next sweep a row to remove".

**Fix:** Validate before re-keying and report a mismatch as a named reason rather than writing it:

```python
night = observing_night(event.start_time, site_zone)
if not (run.window_start <= night <= run.window_end):
    _mark_unexplained(
        [event], _OTHER,
        f"derived night {night} falls outside the run's window "
        f'{run.window_start}..{run.window_end}',
    )
    continue
```

### WR-09: A `KeyError` from `_CLASSICAL_RUN_STATUS` escapes the cutover's per-group handler

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:240`
**Severity:** WARNING

**Issue:** `'run_status': _CLASSICAL_RUN_STATUS[parsed.status]` sits in the `fields` dict
construction, **outside** any `try`. The command is currently safe only because
`telescope_runs.KNOWN_STATUSES` and `load_telescope_runs._CLASSICAL_RUN_STATUS` happen to have
identical key sets — an invariant nothing enforces and which lives in two different modules.
Adding one status word to `KNOWN_STATUSES` turns this into an uncaught `KeyError` that aborts the
whole cutover mid-run, after partial commits (see WR-06), with no reason report. The identical
line in `load_telescope_runs.py:249` *is* inside a `try`, but that handler catches only
`(ValueError, Observatory.DoesNotExist)` — a `KeyError` escapes there too and kills the whole
file import.

**Fix:** Either use `.get()` with an explicit unexplained reason, or (better) make the invariant
structural:

```python
# load_telescope_runs.py, module level
assert set(_CLASSICAL_RUN_STATUS) == KNOWN_STATUSES, (
    'every telescope_runs.KNOWN_STATUSES member needs a CampaignRun.RunStatus mapping'
)
```

plus `except (ValueError, KeyError, Observatory.DoesNotExist)` at both call sites.

### WR-10: A legacy `RUN:{pk}:{date}` event with no companion row is never cleaned up

**File:** `solsys_code/campaign_reconciler.py:394-397`, `solsys_code/campaign_reconciler.py:560-564`
**Severity:** WARNING

**Issue:** `_clearable_and_declined()` starts from
`CalendarEventMeta.objects.filter(run_id=run.pk, event__in=candidates)`, so an event **with no
`CalendarEventMeta` row at all** is in neither the clearable list nor the declined count. For the
bare-container detach path that is correct (there is no attribution to release). For the new
`_stale_dated_events()` **delete** path added by this phase it is not: D-16's stated contract is
that every `RUN:{pk}:{date}` event is *"either re-keyed (elsewhere, by the projector) or removed
(here) -- no third outcome."* A meta-less legacy event is exactly that third outcome — a
container-dispatched run's leftover per-night event that no code path will ever visit again.
Pre-Phase-29 events and events created by the admin FK picker can both lack a companion row.

**Fix:** For the date-bearing group, delete the union of "clearable via an unconfirmed companion
row" and "no companion row at all":

```python
_stale_bare, stale_dated = _split_stale_owned_events(run, active_urls)
if claimed_legacy_urls:
    stale_dated = stale_dated.exclude(url__in=claimed_legacy_urls)
clearable, declined = _clearable_and_declined(run, stale_dated)
orphan_ids = list(stale_dated.filter(telescope_label_meta__isnull=True).values_list('pk', flat=True))
return clearable + orphan_ids, declined
```

### WR-11: Two legacy events for one night collide onto a single `ALLOC:` key, and `CalendarEvent.url` is not unique

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:290-303`, `solsys_code/allocation_projector.py:451`
**Severity:** WARNING

**Issue:** `tom_calendar.CalendarEvent.url` is `URLField(blank=True, default="")` with **no
unique constraint**. The cutover re-keys every writable event in a group by its own derived
night; if two legacy events resolve to the same night (a duplicate row from a pre-cutover
re-ingest whose `start_time` drifted past the `start_time_tolerance` window, or two schedule
lines sharing a night), both receive the *same* `ALLOC:{pk}:{night}` url with no error. From then
on `project_allocation()` reads the night with `CalendarEvent.objects.filter(url=url).first()`
(line 451) — it manages one of them and never sees the other, and because the url *is* in
`active_urls` the convergence step will not remove it either. The duplicate is permanent and
invisible to every counter.

**Fix:** Detect the collision in the cutover and report it rather than writing it:

```python
claimed_nights: set = set()
for event in writable_events:
    night = observing_night(event.start_time, site_zone)
    if night in claimed_nights:
        _mark_unexplained([event], _OTHER, f'a second event already claims night {night}')
        continue
    claimed_nights.add(night)
```

Longer term, a `UniqueConstraint` on `CalendarEvent.url` restricted to non-blank values would
make every `.filter(url=...).first()` in the reconciler and the projector honest; that is a
`tom_calendar` change and out of this phase's scope, but worth recording.

---

_Reviewed: 2026-09-13_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
