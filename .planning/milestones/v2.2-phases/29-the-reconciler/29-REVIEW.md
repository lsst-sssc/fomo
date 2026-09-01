---
phase: 29-the-reconciler
reviewed: 2026-08-05T20:17:36Z
depth: deep
files_reviewed: 14
files_reviewed_list:
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/notebooks.rst
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/calendar_utils.py
  - solsys_code/campaign_reconciler.py
  - solsys_code/campaign_views.py
  - solsys_code/management/commands/reconcile_campaign_runs.py
  - solsys_code/migrations/0014_alter_campaignrun_source.py
  - solsys_code/models.py
  - solsys_code/tests/test_admin.py
  - solsys_code/tests/test_calendar_utils.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/tests/test_campaign_reconciler.py
  - solsys_code/tests/test_reconcile_campaign_runs.py
findings:
  critical: 1
  warning: 2
  info: 1
  total: 4
status: issues_found
---

# Phase 29: Code Review Report

**Reviewed:** 2026-08-05T20:17:36Z
**Depth:** deep
**Files Reviewed:** 14
**Status:** issues_found

## Summary

`campaign_reconciler.py` is well-tested for the three dispatch branches it defines
(container / classical-per-night / adopt-and-rekey) taken in isolation, and the new
`ESO_QUEUE` source value is wired correctly through `QUEUE_SOURCES`, the model's
`TextChoices`, and the migration -- no mismatch found there. The unit and command-level
test suites (`test_campaign_reconciler.py`, `test_reconcile_campaign_runs.py`) genuinely
exercise RECON-01/02/03/04/05/06/07 as claimed, and the `campaign_views.py` rewire onto
`reconcile_run()` is a clean, behaviour-preserving extraction of the retired
`_project_calendar_event()`/`_calendar_event_title()` helpers.

The gap this review surfaces is a cross-cutting one the module's own docstring promises
away but the code does not actually enforce: **`reconcile_run()` dispatches a `CampaignRun`
to exactly one of two mutually-exclusive calendar-event key families based on the run's
*current* `source`/`telescope_class`/`site` state, but nothing detects or cleans up a
family the run previously belonged to.** Every test fixture in this phase creates a run
already in its final classification and never exercises a mid-lifecycle re-classification,
so this gap has no test coverage anywhere in the four new/changed test files. It is not a
hypothetical scenario: the runbook and the paired demo notebook both document staff
correcting an already-reconciled row's `source` through the Django admin (`legacy` ->
`lco_queue`/`gemini_queue`/`eso_queue`) as the intended remedy for pre-v2.2 data -- exactly
the sequence that triggers it (see CR-01 below).

## Critical Issues

### CR-01: Re-classifying a CampaignRun's branch (`source`/`telescope_class`/`site` change) leaves stale events behind, and can corrupt an adopted night's timing

**File:** `solsys_code/campaign_reconciler.py:218-255` (`_adopted_event_for_night`), `346-376` (`reconcile_run` dispatch)

**Issue:**

`reconcile_run()` picks exactly one branch per call, based on the run's *current* state
(`telescope_class` non-blank -> container; `site.observations_type == SATELLITE_OBSTYPE` ->
container; `source in QUEUE_SOURCES` -> container; else -> classical per-night). Both
branches are documented as writing into mutually exclusive key families (bare `RUN:{pk}`
vs. date-bearing `RUN:{pk}:{date}`), and the module docstring states "the key form alone
says which family an event belongs to."

That invariant only holds as long as a run's classifying fields never change after its
first successful reconcile. They can, and the codebase explicitly relies on this: `source`
stays editable in the Django admin for every non-`web` row (`test_admin.py`
`SourceProvenanceLockTests.test_submitted_source_value_does_bind_on_non_web_row`,
`docs/runbooks/telescope_runs_calendar.rst` "Can I correct a run's source?"), and both the
runbook ("How do I get every campaign run onto the calendar?") and the paired demo notebook
(`reconcile_campaign_runs_demo.ipynb`, the "Why `source` is set explicitly here" cell)
describe *exactly* this workflow as the fix for real pre-v2.2 data: "a genuine LCO/Gemini
queue run imported before this milestone will render through the classical per-night branch
until a staff member sets its `source` to the correct queue value through the Django admin
-- only then does a real sweep render the queue-versus-classical split correctly."

Concretely, take a run already reconciled once under the classical branch (per-night
`RUN:{pk}:{date}` events created, each linked via `CalendarEventMeta.run = run`, D-02's
adopt step). A staff member then edits `run.source` to `lco_queue`/`gemini_queue`/
`eso_queue` in the admin (a currently-supported, documented action) and re-runs
`reconcile_campaign_runs` (or triggers any of the four staff actions that call
`reconcile_run()`). `reconcile_run()` now dispatches to `_reconcile_container()`, which:

1. Creates a *new* bare `RUN:{pk}` container event alongside the pre-existing per-night
   events -- the run now shows both a whole-window entry and the stale nightly entries on
   the calendar simultaneously, with no code path that ever revisits the per-night rows
   again (`_reconcile_container()` never queries them).
2. If the run's status is later set to cancelled/weathered via `mark_cancelled`/
   `mark_weather_failure`, only the container event gets the `[CANCELLED]`/`[WEATHERED]`
   title prefix -- the orphaned per-night events keep looking like a live commitment
   forever, directly contradicting RECON-08/09's premise that a staff decision "reconciles
   immediately" across the run's whole calendar footprint.

The reverse direction (container -> classical, e.g. clearing a mistakenly-set
`telescope_class`, or a placeholder site being re-resolved to a genuine ground site) is
worse than merely orphaning: `_adopted_event_for_night()` (line 250) selects "adopt"
candidates via `CalendarEventMeta.objects.filter(run_id=run.pk)` with **no filter on
`event.url`**, even though its own docstring says candidates are meant to be
"`load_telescope_runs`-created (blank-`url`) event[s]". That filter is never enforced in
code -- it also matches the run's own *stale container event* (which does have
`CalendarEventMeta.run = run` from its earlier reconcile). For a positive-UTC-offset ground
site -- e.g. `F65`/Faulkes Telescope South, `Australia/Sydney`, the exact fixture site used
throughout `test_campaign_reconciler.py` -- the container event's `start_time`
(`datetime.combine(window_start, time(0,0), tzinfo=utc)`) converted to site-local time still
falls on `window_start`'s calendar date, so on the first loop iteration
(`night == window_start`) `_adopted_event_for_night()` matches the stale container event and
"adopts" it: `update_calendar_event_key_and_fields()` rewrites its `url` to
`RUN:{pk}:{window_start}`, but per the classical branch's own field-authority rule
(`start_time`/`end_time`/`telescope` are only written on **create**, never on update/adopt),
the event's `start_time`/`end_time` are left as the *original whole-window midnight-UTC
span* (e.g. spanning all 15 nights of the old container window) while it now presents as a
single night's entry. The result is a calendar entry keyed and titled as one observing
night but timed as the entire original multi-night window -- silently wrong data, not just
an orphan.

Neither `test_campaign_reconciler.py` nor `test_reconcile_campaign_runs.py` exercises a
mid-lifecycle change to `source`/`telescope_class`/`site` on an already-reconciled run, so
this entire class of bug has zero test coverage despite the otherwise thorough branch-level
unit tests.

**Fix:**

At minimum, close the adopt-corruption vector by scoping `_adopted_event_for_night()`'s
candidate query to the blank-`url` events it was actually designed for, matching its own
docstring:

```python
candidates = (
    CalendarEventMeta.objects.filter(run_id=run.pk, event__url='')
    .select_related('event')
    .order_by('event__start_time')
)
```

That alone stops the corruption case but still leaves the orphaning case (container ->
classical, or classical -> container) unaddressed. `reconcile_run()` (or the callers that
already know a re-classification just happened, e.g. an admin `source` edit) needs an
explicit convergence step: before/after dispatching to the run's current branch, delete or
detach (`CalendarEventMeta.run = None`, restoring the row to the attribution queue) any
`owned_events(run)` row whose key form doesn't match the currently-active family, e.g.:

```python
def _detach_stale_family_events(run: CampaignRun, active_urls: set[str]) -> None:
    stale = owned_events(run).exclude(url__in=active_urls)
    CalendarEventMeta.objects.filter(event__in=stale).update(run=None)
```

called from `reconcile_run()` after computing which family the run now belongs to. This
needs a product decision (delete vs. detach vs. flag-for-review) rather than a purely
mechanical fix, but the current code makes no attempt at all, silently leaving both stale
and (via the adopt bug) corrupted rows on the shared calendar.

## Warnings

### WR-01: Deleting a `CampaignRun` permanently orphans its `RUN:`-namespaced calendar events

**File:** `solsys_code/models.py:34-41` (`CalendarEventMeta.run`, `on_delete=models.SET_NULL`); `solsys_code/campaign_reconciler.py:106-113` (`owned_events`)

**Issue:** `CampaignRunAdmin` (`solsys_code/admin.py`) does not override `has_delete_permission`,
so a staff/superuser can delete a `CampaignRun` from the Django admin change list like any
other model. On delete, `CalendarEventMeta.run` is set to `NULL` (`on_delete=SET_NULL`), but
the `CalendarEvent` rows themselves -- keyed `RUN:{deleted_pk}` / `RUN:{deleted_pk}:{date}`
-- are never deleted or otherwise cleaned up. There is no signal handler, no admin
`delete_queryset` override, and no management command that sweeps for this. The events
persist on the shared calendar indefinitely, referencing a run that no longer exists, with
no reconciler entry point left that can ever touch them again (`reconcile_run()` requires a
live `CampaignRun` instance).

**Fix:** Add a `pre_delete`/`post_delete` signal (or a `CampaignRunAdmin.delete_queryset`
override, and a `CampaignRun.delete()` override for the non-admin path) that deletes
`campaign_reconciler.owned_events(run)` before/when the run itself is deleted. If deletion
of `CampaignRun` rows with existing calendar events is not an intended workflow at all,
consider disabling delete for rows with `owned_events(run).exists()` instead.

### WR-02: No guard against `window_end < window_start`, silently producing zero events or a backward-dated container event

**File:** `solsys_code/campaign_reconciler.py:198-199` (`_reconcile_container`), `295` (`_reconcile_classical_nights`, `n_nights` computation)

**Issue:** `_skip_reason()` only checks that `window_start`/`window_end` are non-`None`; it
never validates `window_start <= window_end`. If a `CampaignRun` somehow carries
`window_end < window_start` (e.g. a hand-edited admin value, or a future upstream bug in
window parsing), `_reconcile_classical_nights()` computes
`n_nights = (run.window_end - run.window_start).days + 1`, which can be `<= 0`; `range(n_nights)`
then silently iterates zero times, producing **no events at all** with no skip reason
reported (the run is not routed through `_skip_reason()`'s vocabulary for this case, so
`reconcile_campaign_runs` would report it as neither `skipped` nor `failed` -- it simply
contributes nothing, indistinguishable from an already-fully-`unchanged` run in the
summary). `_reconcile_container()` has no such implicit floor -- it happily creates one
event whose `end_time` predates its `start_time`, which most calendar renderers will not
handle gracefully.

**Fix:** Add an explicit `_skip_reason()` branch (e.g. `'window_end before window_start'`)
so this data-integrity problem is surfaced the same way every other unprojectable state is,
rather than silently vanishing from the reconciler's counters.

## Info

### IN-01: Operator docs describe the exact workflow that triggers CR-01, without warning about it

**File:** `docs/runbooks/telescope_runs_calendar.rst:287-336` ("Can I correct a run's
source?"), `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` (the "Why
`source` is set explicitly here, and what it means for real data" markdown cell)

**Issue:** Both documents correctly describe that a pre-v2.2 `legacy`-sourced queue run
must have its `source` corrected through the admin before it will render on the calendar
via the correct (container) branch -- but neither mentions that if the run was already
reconciled once under its old classification, the correction leaves the old family's
events on the calendar (CR-01). The runbook's "Can I correct a run's source?" section
covers the provenance/`web`-lock angle thoroughly but is silent on the calendar-event
consequence of a non-`web` source correction.

**Fix:** Once CR-01 is resolved (or as an interim mitigation), add a note to "Can I correct
a run's source?" and to the demo notebook's `source`-correction cell describing what
happens to any calendar events the run already has when its classifying fields change, and
what an operator should do about them (e.g. re-run `reconcile_campaign_runs` and manually
verify/clean up via the Django admin's `CalendarEventMetaInline`).

---

_Reviewed: 2026-08-05T20:17:36Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
