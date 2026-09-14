---
phase: 35-allocation-layer-classical-cutover
reviewed: 2026-09-13T00:00:00Z
depth: deep
iteration: 2
prior_review: 35-REVIEW.md (git show eb6a595)
files_reviewed: 15
files_reviewed_list:
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/allocation_projector.py
  - solsys_code/campaign_reconciler.py
  - solsys_code/management/commands/cutover_classical_allocations.py
  - solsys_code/management/commands/load_telescope_runs.py
  - solsys_code/management/commands/reconcile_campaign_runs.py
  - solsys_code/models.py
  - solsys_code/observation_projector.py
  - solsys_code/tests/test_allocation_projector.py
  - solsys_code/tests/test_allocation_projector_signals.py
  - solsys_code/tests/test_campaign_reconciler.py
  - solsys_code/tests/test_cutover_classical_allocations.py
  - solsys_code/tests/test_load_telescope_runs.py
  - solsys_code/tests/test_observation_projector_signals.py
  - solsys_code/tests/test_reconcile_campaign_runs.py
prior_findings:
  total: 17
  closed: 14
  partially_closed: 3
findings:
  critical: 3
  warning: 10
  info: 0
  total: 13
status: issues_found
---

# Phase 35: Code Review Report (iteration 2 — re-review after fixes)

**Reviewed:** 2026-09-13
**Depth:** deep
**Files Reviewed:** 15
**Status:** issues_found

## Summary

This is a re-review of the 22 commits (14 fixer commits plus quick tasks `260913-ng8` and
`260913-npq`) that closed the previous deep review's 17 findings.

**Prior-finding verification: 14 of 17 verified fully closed, 3 partially closed.**

| Prior finding | Verdict |
|---|---|
| CR-01 dispatch/approval bypass | **Closed.** All three receivers now route through `reproject_allocation_if_dispatched()`; `grep` confirms no remaining direct `project_allocation()` call outside `reconcile_run()`. |
| CR-02 re-classification orphans `ALLOC:` | **Closed** (`_stale_allocation_events()` wired into both the real and dry-run convergence). See NF-01 for a residual class it does not reach. |
| CR-03 unguarded legacy delete on retire | **Closed** (ownership + `confirmed_by` guards applied). See NF-01/NF-09. |
| CR-04 convergence deletes foreign/confirmed `ALLOC:` | **Closed** for the foreign/confirmed case — but the fix introduced the opposite defect, NF-01 (BLOCKER). |
| CR-05 admin bulk `QuerySet.delete()` escapes guard | **Closed.** `origin_model = getattr(origin, 'model', type(origin))` covers both forms; regression test exercises `CampaignRun.objects.filter(pk=...).delete()`. |
| CR-06 inverted span for a site east of UTC | **Partially closed.** Sydney is fixed; the rule as implemented is wrong for any site whose UTC offset is in `[0, +6]` — see NF-03 (BLOCKER), reproduced. |
| WR-01 linked-run handoff skips non-LCO/SOAR | **Closed** — but the re-ordering introduced NF-04. |
| WR-02 one `try` around the whole link loop | **Closed** (per-link `try`, run pk named in the log). |
| WR-03 dry-run computes and discards `_mint_fields()` | **Closed** — side effect NF-10. |
| WR-04 past-tense dry-run messages | **Closed** (all three verbs routed through `dry_run`). |
| WR-05 "retired" conflates three causes | **Partially closed.** The command message was fixed; `docs/runbooks/telescope_runs_calendar.rst:995-999` still ships the exact single-cause claim WR-05 called factually untrue — see NF-07. |
| WR-06 no transaction boundary in the cutover | **Closed** (per-group + per-event savepoints) — but the counters are not rolled back with them, NF-05. |
| WR-07 empty `CampaignRun` for an all-foreign group | **Closed** (`if not writable_events: continue`, with the reporting and non-zero exit preserved — verified: every event is marked before the `continue`). |
| WR-08 no window-containment check in the cutover | **Closed on the real path only.** The dry-run path does not apply it, which breaks the command's own documented dry-run contract — NF-02 (BLOCKER), reproduced. |
| WR-09 uncaught `KeyError` from `_CLASSICAL_RUN_STATUS` | **Closed** (import-time assertion + per-group catch) — side effect NF-08. |
| WR-10 meta-less legacy event never cleaned up | **Partially closed.** The `telescope_label_meta__isnull=True` half is covered; the `telescope_label_meta__run IS NULL` half of the same "writable but unattributed" class is not — NF-01. |
| WR-11 `ALLOC:` key collision in the cutover | **Closed** (dedicated `key_collision` category, both checks on both paths, ordering keeps the loser byte-identical). |

**New findings: 13 (3 BLOCKER, 10 WARNING).** Eight of them were introduced by the fixes
themselves; the rest are defects in the fixed code that the first review did not reach.

Four of the findings below are backed by executed probe tests run against a real Django test
database (`python manage.py test solsys_code.tests.<probe module>`, probe modules written and
then deleted; no source file was modified by this review). Reproduced facts, not inferences:

- A stale `ALLOC:` event with no companion row — or with a companion row whose `run` is unset
  — now survives the convergence step **forever and uncounted** (`retired=0, blocked=0,
  detached=0, legacy_deleted=0`). Before the CR-04 fix, `stale_qs.delete()` removed it (NF-01).
- `cutover_classical_allocations --dry-run` reports `unexplained: 0, events re-keyed: 4` and
  **exits zero**; the immediately following real run reports `unexplained: 1, events re-keyed: 3`
  and exits non-zero (NF-02).
- A `CampaignRun` at an `Africa/Johannesburg` (UTC+2) site with `night_end_utc=03:00` raises
  `ValueError: Computed an inverted allocation-night span ...` on **every** reconcile — the run
  can never be projected at all (NF-03).
- After a group-level rollback, the cutover prints `runs created: 1` while
  `CampaignRun.objects.count() == 0` (NF-05).
- On the save that both retires an allocation night and creates the record's own observation
  event, the observation event is left with `CalendarEventMeta.run = None` (NF-04).

Positives worth recording: `pre-commit run ruff` is clean on every changed source file; the
CR-01 guard is applied at all three call sites with no bypass left; the CR-05 origin check
handles the `None`-origin case safely; the WR-11 collision checks are correctly ordered
*before* any write inside the same savepoint, so the losing event really is left
byte-identical; and the WR-07 `continue` provably cannot swallow a group without the
non-zero exit (every event in the group has already been passed to `_mark_unexplained()`).

## Structural Findings (fallow)

No `<structural_findings>` block was supplied with this review request.

## Narrative Findings (AI reviewer)

## Critical Issues

### NF-01: CR-04's fix leaks stale `ALLOC:` events that are writable but unattributed — never deleted, never counted

**File:** `solsys_code/allocation_projector.py:648-665`, `solsys_code/campaign_reconciler.py:551-559`, `solsys_code/campaign_reconciler.py:517-522`, `solsys_code/allocation_projector.py:546-559`
**Severity:** BLOCKER

**Issue:** Three of this iteration's fixes all narrow a delete path through the same two-step
pipeline: `writable_*_events(run)` → `_clearable_and_declined(run, ...)`. Those two steps do
**not** compose the way the fixes assume.

`writable_allocation_events()` (`allocation_projector.py:100-104`) deliberately admits three
shapes: no companion row, a companion row whose `run` is unset, and a companion row pointing at
this run. `_clearable_and_declined()` (`campaign_reconciler.py:429-432`) starts from
`CalendarEventMeta.objects.filter(run_id=run.pk, event__in=candidates)` — so the first two
shapes are in **neither** the clearable list **nor** the declined count. And they are not
counted as foreign either, because `foreign_stale_count = stale_qs.count() -
writable_stale.count()` is zero for them (they *are* in `writable_stale`).

Net effect: the event is not deleted, `retired` is not incremented, `blocked` is not
incremented, `detach_declined` is not incremented, and no warning is logged. It is silently
permanent. Before the CR-04 fix, `stale_qs.delete()` removed it — so this is a regression, not
a pre-existing gap.

Reproduced (probe, executed):

```
# stale ALLOC: night whose CalendarEventMeta row was deleted, then window shrink
PROBE-A result: ReconcileResult(created=0, updated=0, unchanged=2, blocked=0, skipped_nights=0,
                detached=0, detach_declined=0, retired=0, rekeyed=0, legacy_deleted=0)
PROBE-A urls left: ['ALLOC:1:2026-07-09', 'ALLOC:1:2026-07-10', 'ALLOC:1:2026-07-11']
PROBE-A doomed still exists: True

# same night, companion row kept but meta.run set to NULL
PROBE-A2 doomed still exists: True   (all counters zero again)
```

The same hole exists in three more places the fixes touched:

1. `campaign_reconciler._stale_allocation_events()` (CR-02) uses the identical pipeline, so a
   re-classified run's unattributed `ALLOC:` nights are also never cleaned up.
2. `campaign_reconciler._stale_dated_events()` (WR-10) unions in only
   `stale_dated.filter(telescope_label_meta__isnull=True)` — the no-companion-row half. A
   `RUN:{pk}:{date}` event whose companion row exists with `run IS NULL` is in neither half,
   even though `writable_events()` treats the two shapes identically. WR-10 is therefore only
   half-fixed against its own stated D-16 contract ("either re-keyed or removed — no third
   outcome").
3. `allocation_projector.project_allocation()`'s CR-03 retire path: a legacy
   `RUN:{pk}:{night}` event that `_may_write()` accepts but that `_clearable_and_declined()`
   returns empty for is neither deleted, nor claimed into `legacy_urls_claimed`, nor counted.

**Fix:** make "writable but unattributed" an explicit, deletable third case everywhere the
pipeline is used, rather than letting it fall between the two filters:

```python
# campaign_reconciler.py -- one helper, used by all three call sites
def _clearable_declined_and_unattributed(run, candidates) -> tuple[list[int], int]:
    """Split candidates into (deletable_ids, declined). Deletable = an unconfirmed companion
    row attributed to THIS run, OR no attribution at all (no companion row, or run IS NULL) --
    the latter has nothing to preserve and is exactly the 'third outcome' D-16 forbids."""
    clearable, declined = _clearable_and_declined(run, candidates)
    unattributed_ids = list(
        candidates.filter(
            Q(telescope_label_meta__isnull=True) | Q(telescope_label_meta__run__isnull=True)
        ).values_list('pk', flat=True)
    )
    return clearable + unattributed_ids, declined
```

Then use it in `project_allocation()`'s convergence, `_stale_allocation_events()`,
`_stale_dated_events()` (replacing the partial orphan union) and the CR-03 retire branch. Add a
regression test for each of the two unattributed shapes at both `ALLOC:` and `RUN:{pk}:{date}`.

---

### NF-02: `cutover_classical_allocations --dry-run` exits zero on a run the real pass rejects — WR-08's guard was added to only one of the two paths

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:352-383` (dry-run loop), `solsys_code/management/commands/cutover_classical_allocations.py:385-440` (real loop), contract stated at `solsys_code/management/commands/cutover_classical_allocations.py:61-65`
**Severity:** BLOCKER

**Issue:** The WR-11 fix rebuilt the dry-run branch to apply "the identical two collision checks
the real path applies" — but the real path has **three** checks, and the WR-08 window-containment
check (`run.window_start <= night <= run.window_end`, lines 402-406) is not one of the three
copied. The dry-run loop goes straight from `observing_night()` to the collision checks.

This contradicts the module's own docstring, verbatim:

> `--dry-run` ... still exits non-zero when it finds an event it cannot explain, because that
> is exactly the condition the operator must clear before the real run — **a dry run that
> silently exited 0 in the presence of an unexplainable row would hide the one thing the
> operator most needs to see before running for real.**

Reproduced (probe, executed; four blank-url events in one group, one of them deriving a night
outside the schedule line's own window):

```
PROBE-D2 dry-run EXITED ZERO:
  Done (dry run). candidates: 4, groups: 1, runs created: 1, updated: 0, unchanged: 0,
  events re-keyed: 4, unexplained: 0

PROBE-D3 real run:
  Done. candidates: 4, groups: 1, runs created: 1, updated: 0, unchanged: 0,
  events re-keyed: 3, unexplained: 1
    unexplained (other): 1 -- unexpected error
  CommandError: 1 event(s) could not be explained and were left untouched (other=1).
```

Two further consequences of the divergence, both inside the same loop:

- The dry-run adds the out-of-window night to `claimed_nights` and counts it under
  `events_rekeyed`; the real path does neither. A second event on the same (out-of-window) night
  is therefore reported `key_collision` by the dry run and `other` by the real run — the two
  passes disagree on both the count and the reason category.
- In the dry-run branch the window bounds read come from the **existing** `CampaignRun` row
  (`run = existing_run`), not from the previewed `fields`; the real path reads them from the
  row `insert_or_create_campaign_run()` just updated. Even once the check is added, the two
  must be made to read the same window.

The mismatched reason category is itself a smaller defect: WR-08's failure is reported under
`_OTHER` / "unexpected error", while WR-11 created a dedicated `_KEY_COLLISION` category
precisely because "every printed reason tells the operator what to DO". An out-of-window
derived night is a known, named condition, not an unexpected error, and it appears nowhere in
the module docstring's reason vocabulary or in the runbook's reason list.

**Fix:** hoist the per-event checks into one helper used by both branches, and give the window
mismatch its own reason name:

```python
_WINDOW_MISMATCH = 'window_mismatch'
_REASON_LABELS[_WINDOW_MISMATCH] = "derived observing night falls outside the run's own window"

class _WindowMismatchError(Exception): ...

def _check_event_night(event, run, site_zone, claimed_nights):
    """The three per-event preconditions, identical for dry-run and real. Returns the night."""
    night = observing_night(event.start_time, site_zone)
    if run is not None and not (run.window_start <= night <= run.window_end):
        raise _WindowMismatchError(
            f"derived night {night} falls outside the run's window "
            f'{run.window_start}..{run.window_end}'
        )
    if night in claimed_nights:
        raise _KeyCollisionError(f'a second event already claims night {night}')
    if run is not None:
        holder = CalendarEvent.objects.filter(url=allocation_night_url(run, night)).exclude(pk=event.pk).first()
        if holder is not None:
            raise _KeyCollisionError(f'night {night} url is already held by CalendarEvent pk={holder.pk}')
    return night
```

In the dry-run branch, evaluate the window bounds from `fields['window_start']`/
`fields['window_end']` (what the real pass would write) rather than from `existing_run`. Add a
test asserting that a dry run and the immediately following real run agree on
`events_rekeyed`, on `unexplained`, on every reason category count, and on exit status.

---

### NF-03: CR-06's site-direction rule does not generalize — any site whose UTC offset is 0…+6 either can never be projected or is silently written a day early

**File:** `solsys_code/allocation_projector.py:162-213`, `solsys_code/allocation_projector.py:251-266`, `solsys_code/models.py:266-277`
**Severity:** BLOCKER

**Issue:** `_site_runs_behind_utc()` keys the whole date-offset rule on the **sign** of the
site's UTC offset. The property the rule actually needs is *whether the site's observing night
crosses a UTC date boundary*, and those two are not the same predicate.

A local night runs roughly 18:00 → 06:00 next day. In UTC that is `(18 − offset)` →
`(30 − offset)`. It stays inside a single UTC date only when `6 < offset ≤ 18`. For
`0 ≤ offset ≤ 6` the night crosses UTC midnight exactly as it does in Chile, but the code
classifies the site as "not behind UTC" and therefore maps every stored time-of-day onto the
night's own date. That covers, among others, Roque de los Muchachos / La Palma (`Atlantic/Canary`,
UTC+0/+1), SAAO Sutherland (`Africa/Johannesburg`, UTC+2) and IAO Hanle (`Asia/Kolkata`,
UTC+5:30 — the half-hour case). `Observatory.timezone` is a free-form `CharField` populated
from the MPC Observatory Codes API (`MPCObscodeFetcher`), so such a site is one admin action
away, with no code change.

Reproduced (probe, executed, `Africa/Johannesburg` site, one-night window, `night_end_utc=03:00`):

```
PROBE-C  raised: ValueError Computed an inverted allocation-night span for run pk=1
         night=2026-07-09: start=2026-07-09T19:00:00+00:00 >= end=2026-07-09T03:00:00+00:00
PROBE-C2 raised: ValueError ... start=2026-07-09T15:52:40+00:00 >= end=2026-07-09T03:00:00+00:00
```

`PROBE-C2` sets only `night_end_utc`, with the start computed from the site's own sunset — so
the inversion is not an operator input error; it is the rule disagreeing with the projector's
own `sun_event()`. That run raises on **every** reconcile: `reconcile_campaign_runs` counts it
under `failed` forever, `load_telescope_runs` reports its line as skipped forever, and a staff
action on it returns a 500.

`night_bounds()`'s new guard converts the loudest case into an error rather than bad data —
but it does not catch the case where **both** boundaries land on the wrong side together. At
Hanle (UTC+5:30, sunset ≈13:00 UTC on the night's own date, sunrise ≈00:00 UTC the next date),
a `0000`-`0200` schedule window resolves to `00:00 → 02:00` on the *evening* date: start < end,
no error raised, and a `CalendarEvent` written a full day early onto the shared calendar.

The rule is also now stated as the durable field contract in `models.py:266-277`, where the
taxonomy is asserted as complete and is not:

> For a site whose local clock runs AHEAD of UTC (Siding Spring, Australia), the entire local
> night maps into a SINGLE UTC date — the night's own — so that 12:00 threshold does not apply
> at all.

That sentence is false for every site with an offset of +1…+6.

**Fix:** derive the answer from the night itself rather than the offset's sign. The cheap,
astropy-free form D-13 requires is available from the site's own local clock:

```python
def _night_crosses_utc_midnight(run, night) -> bool:
    """True when this site's observing night spans two UTC dates, so an early-UTC-hour
    boundary belongs to the FOLLOWING date. Uses the site's local 18:00 and the next local
    06:00 -- a zoneinfo lookup only, never sun_event()."""
    zone = ZoneInfo(run.site.timezone)
    evening = datetime(night.year, night.month, night.day, 18, tzinfo=zone).astimezone(dt_timezone.utc)
    morning = (datetime(night.year, night.month, night.day, 18, tzinfo=zone)
               + timedelta(hours=12)).astimezone(dt_timezone.utc)
    return evening.date() != morning.date()
```

and replace `west_of_utc` with that predicate in `_time_of_day_to_datetime()`,
`night_bounds()` and `_span_needs_remint()`. Correct the `models.py` field comment and
`_site_runs_behind_utc()`'s docstring, both of which currently state the false two-case
taxonomy. Add `TestSubNightWindowSiteDirection` cases for a UTC+2 site and a UTC+5:30 site —
the existing class covers only the two fixture sites the rule already happens to fit.

---

## Warnings

### NF-04: WR-01's re-ordering leaves the observation event unattributed on the very save that creates it

**File:** `solsys_code/observation_projector.py:610-626`, `solsys_code/allocation_projector.py:404-424`
**Severity:** WARNING

**Issue:** WR-01's fix moved the linked-run re-project block from *after* `project_record()` to
*before* it. `project_allocation()`'s attribution bridge
(`_sync_observation_attribution()`) adopts the record's own event by looking it up:
`CalendarEvent.objects.filter(url=observation_projector.event_url(record, facility)).first()`,
and returns early when it is `None`. Running the bridge before `project_record()` means that on
the save that first creates (or re-keys) the observation event, the event does not exist yet and
the adoption is silently skipped.

Reproduced (probe, executed; a record that is unprojectable on creation — no instrument signal
— is linked to a run, then a portal poll supplies both the placed block and the instrument in
one save):

```
PROBE-F2 event after first save: 0
PROBE-F2 event exists after 2nd save: True
PROBE-F2 attributed run_id: None (expected 1 )
PROBE-F2 alloc nights left: ['ALLOC:1:2026-09-16', 'ALLOC:1:2026-09-17']
PROBE-F2 attributed run_id after a 3rd save: 1
```

The allocation night (`ALLOC:1:2026-09-15`) is correctly retired on that save, but the event
that is supposed to take over the night carries no campaign attribution — so
`calendar_display_extras.campaign_decoration()` renders it without its campaign until some
later save or sweep repairs it. The handoff is momentarily incoherent in exactly the transition
D-11 exists to make immediate.

**Fix:** the two steps have a real ordering dependency (retirement must follow, or at least
accompany, the event that replaces it), so run the base projection first and keep WR-01's
facility-independence by splitting the early return:

```python
if raw:
    return

action = stage = None
if instance.facility in PROJECTED_FACILITIES:
    try:
        action, stage = project_record(instance)
    except Exception as exc:  # noqa: BLE001 -- TRIG-02
        logger.warning('receiver_on_record_save failed for observation_id=%r: %s',
                       instance.observation_id, type(exc).__name__)

# D-11, facility-independent, per-link isolated (WR-01/WR-02), AFTER the event exists
for link in instance.campaign_run_links.select_related('run'):
    ...
```

Add a regression test asserting `CalendarEventMeta.run_id == run.pk` immediately after the
single save that creates the observation event for an already-linked record.

### NF-05: the cutover's group savepoint rolls back the writes but not the counters it already incremented

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:336-341`, `solsys_code/management/commands/cutover_classical_allocations.py:441-445`
**Severity:** WARNING

**Issue:** `runs_created`/`runs_updated`/`runs_unchanged` (lines 336-341) and `events_rekeyed`
(lines 383, 436) are incremented **inside** the `with transaction.atomic()` block WR-06 added.
They are plain Python ints, so the group-level `except Exception` at line 441 rolls back every
database write and leaves the counters at their post-write values. The summary the operator
reads for a one-time production migration then reports work that did not happen.

Reproduced (probe, executed; a failure after the run write but inside the group savepoint — an
invalid IANA name on the resolved `Observatory`, which `ZoneInfo(site.timezone)` at line 349
raises on):

```
PROBE-E stdout: Done. candidates: 3, groups: 1, runs created: 1, updated: 0, unchanged: 0,
                events re-keyed: 0, unexplained: 3 | unexplained (other): 3 -- unexpected error
PROBE-E runs actually in db: 0
```

A second, smaller defect on the same path: the group-level handler calls
`_mark_unexplained(events, ...)` over the **whole** group, including events already marked
`_FOREIGN_ATTRIBUTION` by the loop at lines 298-304. Those events are reported twice on stderr
and counted twice in `reason_counts` and in the `CommandError` breakdown.

**Fix:** accumulate the group's counters locally and fold them into the totals only after the
`with` block exits successfully; and mark only the events not already marked:

```python
group_created = group_updated = group_unchanged = group_rekeyed = 0
try:
    with transaction.atomic():
        ...            # increment the group_* locals only
except Exception as exc:  # noqa: BLE001
    already_marked = {e.pk for e, _c, _r in unexplained}
    _mark_unexplained([e for e in events if e.pk not in already_marked], _OTHER,
                      f'{type(exc).__name__}: {exc}')
    continue
runs_created += group_created
runs_updated += group_updated
runs_unchanged += group_unchanged
events_rekeyed += group_rekeyed
```

### NF-06: `_may_write()` and `writable_allocation_events()` disagree about every unattributed `ALLOC:` event

**File:** `solsys_code/allocation_projector.py:90-104`, `solsys_code/allocation_projector.py:519-523`, `solsys_code/campaign_reconciler.py:248-264`
**Severity:** WARNING

**Issue:** `writable_allocation_events()`'s docstring claims it "mirrors
[`writable_events()`'s] attribution-scoped filter exactly". It does not, because the predicate
the two querysets are supposed to mirror — `_may_write()` — falls back to a **`RUN:`-namespace
url check** when the companion row is absent or its `run` is unset:

```python
container_url = run_container_url(run)
return event.url == container_url or event.url.startswith(f'{container_url}:')
```

An `ALLOC:{pk}:{night}` url can never match that, so `_may_write()` returns `False` for exactly
the two shapes `writable_allocation_events()` admits as writable. The `pre_delete` cascade
(`models.py:504`) will happily delete such an event while `project_allocation()` refuses to
touch it.

Reproduced (probe, executed):

```
PROBE-B _may_write says: False
PROBE-B writable_allocation_events includes it: True
PROBE-B result: ReconcileResult(created=0, updated=2, ..., blocked=1, ...)
PROBE-B description refreshed: False
```

The night is permanently `blocked`, its title/description are never refreshed again (the other
two nights in the same window updated normally), and the operator is told
`N event(s) blocked -- owned by someone else`, which is untrue — nobody owns it. This is a
latent pre-existing defect, but the CR-04 fix made `writable_allocation_events()` load-bearing
on a second write path, so the contradiction now has two consumers that disagree.

**Fix:** teach `_may_write()` about the second namespace it is now asked to police, so the
predicate and its queryset twin state the same rule:

```python
def _may_write(event, run) -> bool:
    if event is None:
        return True
    meta = CalendarEventMeta.objects.filter(event=event).first()
    if meta is not None and meta.run_id is not None:
        return meta.run_id == run.pk
    container_url = run_container_url(run)
    alloc_prefix = f'{ALLOC_URL_NAMESPACE}{run.pk}:'
    return (event.url == container_url
            or event.url.startswith(f'{container_url}:')
            or event.url.startswith(alloc_prefix))
```

(Keeping the rule in `campaign_reconciler` per the module's stated one-owner convention; the
prefix constant can be passed in or imported lazily to avoid the cycle.)

### NF-07: WR-05 and CR-02 fixed the command output but left the runbook's counter definitions stale

**File:** `docs/runbooks/telescope_runs_calendar.rst:995-999`, `docs/runbooks/telescope_runs_calendar.rst:1007-1013`
**Severity:** WARNING

**Issue:** WR-05's finding was that "`retired` means *now covered by a real observation*" is
factually untrue, because `ReconcileResult.retired` is incremented from three unrelated places.
The fix changed the per-run message in `reconcile_campaign_runs.py` but not the runbook, which
still tells the operator, unchanged:

> ``retired`` counts an allocation night handed over to a real observation: a run's linked
> ``ObservationRecord`` placed or observed its block on that night ... Unlinking the record
> restores the night on the next reconcile.

The claim the review called untrue is therefore still shipped, in the operator-facing document
the runbook's own "Always run `--dry-run` first and read the list" instruction sends people to.
Per CLAUDE.md, `docs/runbooks/` pages whose documented behavior a change affects are part of
the deliverable, not optional polish.

The `legacy_deleted` definition two paragraphs down is stale for the same reason:

> ``legacy_deleted`` counts a one-time removal: a run with a queue source ... so its leftover
> per-night ``RUN:{pk}:{date}`` events ... an already-container run reports 0 here on every
> later sweep.

After CR-02 the counter also covers leftover `ALLOC:{pk}:{night}` events from a
`telescope_class`/`site` re-classification (not a queue source at all), and after WR-10 a run
that still dispatches per-night can report a non-zero `legacy_deleted` for a meta-less
`RUN:{pk}:{date}` row — at which point the sibling per-run message, "this run now keeps a
single whole-window entry", is also false.

**Fix:** update both definitions to match the fixed code, e.g. "`retired` counts an allocation
night removed for any of three reasons: a linked record's placed or observed block now covers
it, a sub-night window field changed and the night was re-minted, or the run's window shrank";
and "`legacy_deleted` counts one-time churn from a run's dispatch changing — leftover
`RUN:{pk}:{date}` nights, leftover `ALLOC:{pk}:{night}` nights, and `RUN:{pk}:{date}` rows with
no companion row at all." Gate the "keeps a single whole-window entry" clause on the run
actually being container-dispatched.

### NF-08: WR-09 widened the classical loader's `except` tuple around the whole reconcile call

**File:** `solsys_code/management/commands/load_telescope_runs.py:316-321`
**Severity:** WARNING

**Issue:** `except (ValueError, KeyError, Observatory.DoesNotExist)` now wraps the entire
per-line body — including `write_and_reconcile_campaign_run()` and `reconcile_run()`. The
`KeyError` the fix intended to catch comes from one dictionary lookup at line 264; the handler
it was added to spans ~80 lines of reconciliation. Any `KeyError` raised deep inside the
reconciler or the projector (`totals[preview_calendar_event_action(...)] += 1`,
`ReconcileResult(**totals)`, a `ZoneInfoNotFoundError` — which subclasses `KeyError` — from a
malformed `Observatory.timezone`) is now swallowed and reported as
`Line N: 'some-key' (line text: ...)`, a message that names neither the module nor the stage.

Compounding it: `write_and_reconcile_campaign_run()` has no transaction boundary, so when
`reconcile_run()` raises the `CampaignRun` row it just wrote stays committed while the line is
counted under `run_skipped` — the summary reports a line as skipped when a run row was in fact
created.

**Fix:** narrow the catch to the statement that needs it, and keep the broad reconcile call
under its original tuple:

```python
try:
    run_status = _CLASSICAL_RUN_STATUS[parsed.status]
except KeyError as exc:
    self.stderr.write(f'Line {line_num}: unknown classical status {exc} (line text: {line.strip()!r})')
    run_skipped += 1
    continue
...
except (ValueError, Observatory.DoesNotExist) as exc:
    ...
```

and wrap the write-plus-reconcile pair in `transaction.atomic()` so a reported-skipped line
leaves no row behind.

### NF-09: a declined legacy event on a retired night is counted twice, in two different counters

**File:** `solsys_code/allocation_projector.py:552-559`, `solsys_code/campaign_reconciler.py:660`
**Severity:** WARNING

**Issue:** CR-03's fix reports a human-confirmed legacy `RUN:{pk}:{night}` event on a retired
night under `totals['blocked']` and deliberately does **not** add its url to
`legacy_urls_claimed`. Back in `reconcile_run()`, `_stale_dated_events()` therefore still sees
it, `_clearable_and_declined()` classifies it as declined again, and it is reported a second
time under `detach_declined`. One event, one decision, two counters incremented — the summary
line implies two distinct events were left alone.

The `legacy_urls_claimed` docstring (`allocation_projector.py:487-496`) states that the set is
"every `RUN:{pk}:{date}` legacy url this per-night loop has already decided the fate of (a
takeover re-key **or a retirement delete**)". A declined retirement is also a decided fate; the
set's own contract implies it should be excluded from the downstream step.

**Fix:** add the legacy url to `legacy_urls_claimed` whenever this loop has decided its fate at
all — deleted, blocked or declined — not only when it is deletable:

```python
if legacy_event is not None:
    legacy_urls_claimed.add(legacy_url)     # decided here, whatever the decision
    if not _may_write(legacy_event, run):
        ...
```

### NF-10: WR-03's dry-run short-circuit hides the one failure mode the new `night_bounds()` guard raises

**File:** `solsys_code/allocation_projector.py:610-619`, `solsys_code/allocation_projector.py:251-266`
**Severity:** WARNING

**Issue:** WR-03's fix skips `_mint_fields()` entirely on the dry-run create path — correct for
the wasted astropy work, but `_mint_fields()` is also the only caller of `night_bounds()`, and
`night_bounds()` is where CR-06's new `ValueError` guard lives. So
`reconcile_campaign_runs --dry-run` reports `would_create: N` for a run whose very next real
sweep reports `failed` with an inverted-span `ValueError` (see NF-03, where this is reproduced
for a UTC+2 site). The preview cannot show the operator the one condition that will actually
stop the run from projecting.

**Fix:** keep the mint skipped, but validate the boundaries cheaply — `_span_needs_remint()`
already demonstrates that a set sub-night field's expected boundary is computable with no
astropy call:

```python
if dry_run:
    if run.night_start_utc is not None and run.night_end_utc is not None:
        west = _site_runs_behind_utc(run, night)   # or the NF-03 replacement predicate
        if _time_of_day_to_datetime(run.night_start_utc, night, west) >= _time_of_day_to_datetime(
            run.night_end_utc, night, west
        ):
            raise ValueError(...)   # same message night_bounds() raises
    totals['created'] += 1
    continue
```

### NF-11: a vacuous assertion in the cutover sequence-contract test

**File:** `solsys_code/tests/test_cutover_classical_allocations.py:528-531`
**Severity:** WARNING

**Issue:**

```python
rekeyed_count = 1
legacy_deleted_count = 1
self.assertEqual(rekeyed_count + legacy_deleted_count, 2)
```

Two local literals are added together and compared to their own sum. The assertion is `1 + 1
== 2`; it exercises no production code and cannot fail. It sits under a comment claiming to
verify "three-group reconciliation over the two hand-made legacy artifacts: one re-keyed, one
deleted", which reads as real coverage in a test class named `TestCutoverSequenceContract` —
the most misleading place for a no-op assertion.

**Fix:** assert the real counters from the reconcile the test performs, e.g.

```python
result = reconcile_run(stays_per_night_run)
self.assertEqual(result.rekeyed, 1)
container_result = reconcile_run(now_container_run)
self.assertEqual(container_result.legacy_deleted, 1)
```

or delete the three lines — the preceding assertions in the same test already prove both
outcomes against the database.

### NF-12: the cutover's local `writable_events` collides with the codebase's established ownership-helper name

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:298-315`
**Severity:** WARNING

**Issue:** `writable_events` is the name of `campaign_reconciler.writable_events(run)` — the
canonical queryset-level ownership helper, referenced by name in `models.py`'s cascade
docstring, in `allocation_projector.writable_allocation_events()`'s docstring and throughout
the reconciler. Reusing it here for an unrelated plain `list` of blank-url `CalendarEvent`
objects (built from a different rule: "has no companion row pointing at any run") invites a
reader — or a future edit that adds the import — to conflate the two. The file already imports
several underscore-named helpers across module boundaries, so the import is a plausible next
change.

**Fix:** rename to something that names what it holds, e.g. `unattributed_events` or
`convertible_events`.

### NF-13: the WR-11 runbook edit broke the sentence listing what the cutover converts

**File:** `docs/runbooks/telescope_runs_calendar.rst:912-914`
**Severity:** WARNING

**Issue:**

> ... whose events agree on their campaign, and whose events are not already attributed to a
> different run, **are not already claimed on a colliding night.** What it deliberately leaves
> alone: ...

The new clause was spliced in after the closing `and`, leaving a comma splice with no
conjunction and a dangling subject. In the operator-facing paragraph that defines what the
one-time production migration will and will not convert, the sentence no longer parses
cleanly.

**Fix:** `... whose events agree on their campaign, whose events are not already attributed to
a different run, and whose derived observing nights are not already claimed.`

---

_Reviewed: 2026-09-13_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep (re-review, iteration 2)_
