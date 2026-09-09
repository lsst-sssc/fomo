---
phase: 33-series-identity-reconciler-inversion
reviewed: 2026-09-08T00:00:00Z
depth: deep
files_reviewed: 26
files_reviewed_list:
  - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/admin.py
  - solsys_code/campaign_reconciler.py
  - solsys_code/campaign_tables.py
  - solsys_code/campaign_utils.py
  - solsys_code/campaign_views.py
  - solsys_code/management/commands/reconcile_campaign_runs.py
  - solsys_code/migrations/0017_calendareventmeta_observation_links.py
  - solsys_code/models.py
  - solsys_code/templatetags/calendar_display_extras.py
  - solsys_code/tests/test_admin.py
  - solsys_code/tests/test_calendar_event_meta_links.py
  - solsys_code/tests/test_calendar_template.py
  - solsys_code/tests/test_campaign_attribution.py
  - solsys_code/tests/test_campaign_attribution_views.py
  - solsys_code/tests/test_campaign_reconciler.py
  - solsys_code/tests/test_campaign_views.py
  - solsys_code/tests/test_null_campaign_guards.py
  - solsys_code/tests/test_reconcile_campaign_runs.py
  - solsys_code/tests/test_write_and_reconcile.py
  - solsys_code/views.py
  - src/templates/campaigns/campaignrun_table.html
  - src/templates/tom_calendar/partials/calendar.html
  - src/templates/tom_calendar/partials/campaign_chip.html
  - src/templates/tom_calendar/partials/event_form.html
findings:
  critical: 1
  warning: 6
  info: 6
  total: 13
status: issues_found
---

# Phase 33: Code Review Report (post-gap-closure re-review)

**Reviewed:** 2026-09-08
**Depth:** deep
**Files Reviewed:** 26
**Status:** issues_found

## Summary

Re-review of the full Phase 33 diff (`998b45e^..HEAD`) after gap-closure plans 33-06,
33-07 and 33-08. Each of the 16 prior findings was re-verified against the code, not
against the SUMMARY files.

**Verified closed (14 of 16):** CR-01, CR-02, WR-01, WR-02, WR-03, WR-04, WR-05, WR-06,
WR-07, IN-01, IN-02, IN-03, IN-04, IN-05. Spot-checks that actually held up:
`tom_common/base.html` really does declare an `additional_css` block at line 18, so the
`tr:target` rule now renders (three tests assert it); `_observing_night()`'s
`local - 12h` wall-clock arithmetic matches `telescope_runs._local_noon_utc()` exactly on
both sides of the boundary; `UNLINK_CLEARED_FIELDS` is genuinely consumed by both writers
(the `patch.dict` sentinel test distinguishes a loop from three hand-written assignments);
`unlink_event_from_run()` rejects `str`/`bytes` after the `run_pk` guard; the
`CalendarEventMetaInline` docstring and the runbook bullet now describe what Django
actually renders. `pre-commit run ruff` and `ruff-format` are clean, and the 187 tests in
`test_campaign_reconciler`, `test_reconcile_campaign_runs`, `test_admin` and
`test_campaign_attribution_views` pass.

**Still open (1 of 16):** WR-08. The `#run-{pk}` link still lands nowhere for a run past
page 1. Plan 33-06 explicitly deferred the fix and pinned it as a tested, documented
constraint instead — which is the review's own stated minimum — so it is carried forward
below rather than escalated, but the product behaviour is unchanged.

**Partially closed (1 of 16), and the source of this review's blocker:** CR-03. The skip
is now unconditional and the superseded night's event is detached, so the *attribution*
duplicate is gone. But detaching an event this reconciler minted, into a queue that
immediately re-offers it at HIGH band to the very run that just released it, creates a
new confirm/erase loop that destroys human audit stamps on every subsequent sweep. I
reproduced this end to end with a throwaway test module under `solsys_code/tests/`, run
and then deleted; no source file was modified.

The remaining new findings cluster on operator visibility: the `detached` counter is
described to operators as one thing when it counts two, `--dry-run` cannot preview the
only step in the sweep that destroys audit data, and none of the four interactive staff
actions surface either new counter — one of them still reports "run added to the
calendar" when nothing was added.

## Narrative Findings (AI reviewer)

## Critical Issues

*(BLOCKER tier; `CR-` == `BL-` for downstream consumers.)*

### CR-04: the CR-03 detach and Phase 28's queue form a confirm/erase loop that destroys audit stamps on every sweep

**File:** `solsys_code/campaign_reconciler.py:425-427` and `:469-528`;
`solsys_code/campaign_attribution.py:454-477`, `:555-599`
**Severity:** BLOCKER

**Issue:** The CR-03 fix makes the skip unconditional and drops the superseded night's url
out of `active_urls`, so `_detach_stale_family_events()` clears that event's
`run`/`confirmed_by`/`confirmed_at` via `unlink_event_from_run()`. Clearing `run` makes the
event an orphan by `orphan_calendar_events()`'s definition
(`Q(telescope_label_meta__run__isnull=True)`), and the reconciler writes **no**
`CalendarEventDismissal` row — unlike `AttributionDecisionView._undo_confirmation()`, which
writes one precisely so the matcher stops re-suggesting the pair.

So `candidates_for_event()` immediately re-offers the detached event to the *same run that
just released it*, at HIGH band, because the event's telescope/instrument/date were all
copied from that run when the reconciler created it. A staff member draining the queue sees
an obvious match, confirms it, and the next sweep silently detaches it again.

Reproduced (Sydney site, single-night run, reconcile → attribute a facility event →
reconcile → staff re-confirms → reconcile → staff re-confirms → reconcile):

```
R1 ReconcileResult(created=1, ..., skipped_nights=0, detached=0)
R2 ReconcileResult(created=0, ..., skipped_nights=1, detached=1)
detached meta run: None
is orphan: True
candidates offered for the detached event: [(1, 'high', 0.82)]
R3 ReconcileResult(created=0, ..., skipped_nights=1, detached=1)
after R3 -> run: None confirmed_by: None confirmed_at: None
R4 ReconcileResult(created=0, ..., skipped_nights=1, detached=1)
after R4 -> run: None
calendar entries on that night: ['RUN:1:2026-08-01', 'https://observe.lco.global/api/requestgroups/7/']
```

Every unattended `reconcile_campaign_runs` run destroys the "confirmed by X at T" a human
wrote minutes earlier, permanently and with no compensating record — only a
`logger.warning`. This is not the pre-fix behaviour: before the CR-03 change, `active_urls`
always contained every night in the window, so a re-attributed `RUN:` event was never stale
and was never detached. The loop is newly reachable.

The existing test `test_second_reconcile_detaches_the_superseded_run_keyed_event_and_restore_on_third`
(`solsys_code/tests/test_campaign_reconciler.py:686-746`) stops one step short: it clears
the *facility* event's link before the third reconcile, so it never exercises the "staff
re-attributes the detached `RUN:` event" path the queue actually steers them into. The
demo notebook's assertion `"the detached event must be back in Phase 28's attribution
queue"` (`reconcile_campaign_runs_demo.ipynb`, skip-rule cell) celebrates exactly the state
that starts the loop.

**Fix:** the detach must make the pair un-re-offerable, or must not be repeatable. Either:

1. Write the dismissal row the rest of the codebase already uses as the "this pair was
   deliberately released" trace, inside the same write, so the matcher stops re-suggesting
   it (mirrors `_undo_confirmation()`'s established discipline):

```python
def _detach_stale_family_events(run: CampaignRun, active_urls: set[str]) -> int:
    from solsys_code.campaign_utils import unlink_event_from_run
    from solsys_code.models import CalendarEventDismissal

    stale = list(owned_events(run).exclude(url__in=active_urls))
    detached = unlink_event_from_run(stale, run)
    if detached:
        for event in stale:
            CalendarEventDismissal.objects.get_or_create(
                event=event,
                run=run,
                defaults={
                    'dismissed_by': None,
                    'dismissed_at': timezone.now(),
                    'reason': 'Released by the reconciler: this night is covered by another '
                              'attributed entry (Phase 33 CR-03 skip rule).',
                },
            )
        logger.warning(...)
    return detached
```

2. Or narrow the detach so it only fires the first time — e.g. skip an event whose
   `confirmed_by` is set (a human decision outranks an automated release), and count it
   into a separate `blocked`-style counter so the operator is told a human has overridden
   the skip rule for that night.

Either way, add a regression test that reconciles, attributes, reconciles, **re-confirms
the detached `RUN:` event to the same run**, reconciles again, and asserts the stamp
survives (or that the pair is no longer offered).

## Warnings

### WR-08: the decoration's `#run-{pk}` link still lands nowhere for a run past page 1 (carried forward, unfixed)

**File:** `solsys_code/templatetags/calendar_display_extras.py:479-486`;
`solsys_code/campaign_views.py` `table_pagination = {'per_page': 25}`

**Issue:** Unchanged from the prior review. `table_url` is still built as
`reverse('campaigns:table', args=[run.campaign_id]) + f'#run-{run.pk}'` with no page
parameter. Plan 33-06 deferred the positional-page fix (it would add a per-event ordered
query, contradicting plan 33-02's no-per-event-query must-have) and instead pinned the
behaviour with `TestCampaignRunAnchorPagination` and documented it in
`docs/runbooks/telescope_runs_calendar.rst`. That satisfies the prior review's stated
minimum ("at minimum document the limitation ... and add a test pinning the >25-run
behaviour"), so this is recorded as accepted-and-documented rather than escalated — but
the 3I/ATLAS coordination case this feature exists for is exactly a campaign with more
than 25 runs, and the link is still silently dead there.

**Fix:** carry as an explicit backlog item for Phase 34/35 (resolve the run's page in the
view that renders the campaign table, or make the table sort/filter deterministic enough
that a `?run=<pk>` query parameter can jump to it) rather than leaving it as a permanent
documented limitation.

---

### WR-09: the superseded night still shows two calendar entries, and the runbook's stated remedy does not remove either

**File:** `solsys_code/campaign_reconciler.py:469-528`;
`docs/runbooks/telescope_runs_calendar.rst` ("released (never deleted) back into the
attribution queue for a human to re-confirm or discard")

**Issue:** CR-03's complaint was "two calendar entries for one observing night ... a
visibly duplicated night, forever". The fix removes the *attribution* from one of them; the
`CalendarEvent` row itself survives by design (detach, never delete) and keeps rendering in
the month grid. Verified in the reproduction above: after the fix, the night still carries
both `RUN:1:2026-08-01` and the facility event.

Worse than before this phase: D-12 removed the campaign label from `event_title()`, so the
leftover entry now renders as a bare `FTN/MuSCAT3` with no campaign chip (it was detached)
and no campaign name in its title — an unexplained duplicate with nothing linking it back
to the run it came from.

The runbook tells the operator the released entry is there "for a human to re-confirm or
discard". Re-confirming triggers CR-04's loop; "discard" in Phase 28's queue means writing
a `CalendarEventDismissal`, which — as the same runbook states two sections earlier — "is
not an association" and never touches the `CalendarEvent`. Dismissing hides the candidate
from the queue and leaves the duplicate on the calendar. There is no documented operator
action that actually removes it short of the Django admin.

**Fix:** decide and document the real remedy. If the entry is genuinely disposable, the
detach step should delete it (it is inside this module's own `RUN:` namespace, which this
module is the sole writer of, so this does not violate "never delete another writer's
entry"). If it must survive, say so in the runbook and name the Django-admin delete as the
removal path, instead of implying the queue can dispose of it:

```rst
Releasing an entry does not remove it from the calendar. Dismissing the candidate
in the attribution queue only stops it being re-suggested; to remove the duplicate
entry itself, delete it under **Django admin -> Tom calendar -> Calendar events**.
```

---

### WR-10: `detached` is reported to operators as one thing but counts two

**File:** `solsys_code/management/commands/reconcile_campaign_runs.py:83-87`;
`docs/runbooks/telescope_runs_calendar.rst` (the `detached` counter description)

**Issue:** The per-run stderr line reads:

```python
f'Run pk={run.pk}: {result.detached} event(s) detached -- superseded by a later '
'attribution; confirmation stamp(s) cleared'
```

and the runbook says `detached` "counts entries the reconciler released back into the
attribution queue because the night they cover became attributed through another writer
*after* this reconciler had already created its own entry for it."

But `_detach_stale_family_events()` is called for **both** branches and covers two distinct
causes — its own docstring says so (`campaign_reconciler.py:469-487`): the CR-03
supersession case *and* 29-REVIEW.md CR-01's re-classification case (an admin corrects
`telescope_class`/`site`, so the whole old key family goes stale). `ReconcileResult.detached`'s
own docstring is accurate; the two operator-facing surfaces are not.

An operator who corrects a run's `telescope_class` and then sweeps will be told 15 events
were "superseded by a later attribution" when nothing was attributed at all. That sends
them looking for an attribution that does not exist.

**Fix:** either report the two causes separately, or state the counter neutrally:

```python
if result.detached:
    self.stderr.write(
        f'Run pk={run.pk}: {result.detached} event(s) released back into the attribution '
        'queue (superseded by another attributed entry, or left over from a key family '
        'this run no longer belongs to); confirmation stamp(s) cleared'
    )
```

and mirror the same two-cause wording in the runbook's counter description.

---

### WR-11: `--dry-run` cannot preview the one step in the sweep that destroys audit data

**File:** `solsys_code/campaign_reconciler.py:570-574`;
`solsys_code/management/commands/reconcile_campaign_runs.py:89-100`

**Issue:** `reconcile_run()` skips `_detach_stale_family_events()` entirely under
`dry_run`, so `detached` is always 0 there, and the command prints the literal
`would_detach: n/a (dry-run)`. The runbook explains this as "the detach step is itself a
write and does not run in a dry run -- there is nothing to preview".

That reasoning does not hold: computing how many rows *would* be detached is a pure read.
`stale = owned_events(run).exclude(url__in=active_urls)` is already built from data the dry
run has, and the count is
`CalendarEventMeta.objects.filter(run_id=run.pk, event__in=stale).count()`. The runbook
itself instructs operators to "always run this before a real sweep" — and the only
irreversible thing a real sweep does (erasing `confirmed_by`/`confirmed_at`, per WR-03's
whole premise) is precisely the thing the preview refuses to show.

**Fix:** compute the count in dry-run mode without writing:

```python
def _count_detachable(run, active_urls: set[str]) -> int:
    stale = owned_events(run).exclude(url__in=active_urls)
    return CalendarEventMeta.objects.filter(run_id=run.pk, event__in=stale).count()

...
if dry_run:
    detached = _count_detachable(run, active_urls)   # read-only
else:
    detached = _detach_stale_family_events(run, active_urls)
```

and print `would_detach: {detached}` instead of `n/a (dry-run)`, updating
`test_dry_run_reports_skipped_nights_and_would_detach_na_and_writes_nothing` to assert the
number *and* that nothing was written.

---

### WR-12: the four staff-action call sites surface neither new counter, and `_resolve_site()` claims success it did not achieve

**File:** `solsys_code/campaign_views.py:681-699` (and the approve /
`mark_cancelled` / `mark_weather_failure` call sites at `:527`, `:759`)

**Issue:** WR-03's fix threaded `detached` into `ReconcileResult` "so
`reconcile_campaign_runs` and the four staff-action call sites can surface it". The command
now does; none of the four staff actions do. `_set_run_status()` and the approve branch
discard the result entirely; `_resolve_site()` reads only `skipped_reason`.

Two consequences:

1. A staff member clicking **Resolve** can silently destroy a colleague's confirmation
   stamp (CR-04's mechanism, reachable from an interactive surface) and be shown only
   `'Site resolved — run added to the calendar.'`
2. That message is now outright false in the ordinary skip case. With the unconditional
   skip, a run every one of whose nights is already attributed elsewhere returns
   `skipped_reason=None, created=0, skipped_nights=n` — and the branch at line 696 keys on
   `skipped_reason is None` alone, so it reports "run added to the calendar" when nothing
   was added.

**Fix:** key the message on what actually happened, and mention a detach:

```python
if result.skipped_reason is not None:
    messages.success(request, 'Site resolved.')
elif result.created or result.updated:
    messages.success(request, 'Site resolved — run added to the calendar.')
else:
    messages.success(
        request,
        f'Site resolved. {result.skipped_nights} night(s) are already covered by entries '
        'attributed to this run, so no new calendar entries were created.',
    )
if result.detached:
    messages.warning(
        request,
        f'{result.detached} superseded calendar entr(ies) were released back into the '
        'attribution queue; their confirmation record was cleared.',
    )
```

Apply the same `result.detached` warning to the approve and `_set_run_status()` branches,
which currently drop the result on the floor.

---

### WR-13: the unconditional skip silently swallows the `blocked` signal for a contested night

**File:** `solsys_code/campaign_reconciler.py:425-436`

**Issue:** The skip now runs *before* `_may_write()`:

```python
if night in attributed_nights:
    totals['skipped_nights'] += 1
    continue

active_urls.add(url)
sunset, sunrise = sun_event(run.site, night, kind='sun')
existing = CalendarEvent.objects.filter(url=url).first()

if not _may_write(existing, run):
    ...
    totals['blocked'] += 1
```

For a night that is both attributed to this run through a non-`RUN:` event **and** carries a
`RUN:{pk}:{date}` event a staff member has since attributed to a *different* run, the
reconciler now reports `skipped_nights=1, blocked=0` where it previously reported
`blocked=1`. The `blocked` line — the one signal that tells an operator "someone else owns
this night's entry, go look at it" — disappears. The data is safe (the detach step's
`run_id=run.pk` filter still protects the foreign attribution), but the diagnostic is lost,
and no test covers the combination.

**Fix:** evaluate ownership before deciding the night's outcome, so the two signals compose
rather than mask:

```python
existing = CalendarEvent.objects.filter(url=url).first()
if existing is not None and not _may_write(existing, run):
    logger.warning('Reconcile blocked: event pk=%s is not owned by run pk=%s.', existing.pk, run.pk)
    totals['blocked'] += 1
    active_urls.add(url)          # never detach a foreign attribution
    continue
if night in attributed_nights:
    totals['skipped_nights'] += 1
    continue
```

and add a test asserting `blocked == 1` for an attributed-and-contested night.

## Info

### IN-06: `writable_events()`'s docstring names a consumer that does not use it

**File:** `solsys_code/campaign_reconciler.py:136-156` vs. `:520`
**Issue:** The docstring states "Every write path (reconcile's detach step, the
run-deletion cascade) must go through `writable_events()` instead." The run-deletion
cascade does (`models.py:453-455`); `_detach_stale_family_events()` uses `owned_events()`.
Functionally safe — `unlink_event_from_run()`'s `run_id=run_pk` filter gives the same
protection — but the docstring is now the only statement of a rule the code does not
follow, which is exactly the drift WR-02 was raised about elsewhere.
**Fix:** either switch the detach to `writable_events(run)` (a no-op change in behaviour,
one query term more) or amend the docstring to say the detach achieves the same guarantee
through the helper's run filter.

### IN-07: the WR-08 pagination test overrides `setUpTestData` without calling `super()`

**File:** `solsys_code/tests/test_campaign_views.py:696-716`
**Issue:** `TestCampaignRunAnchorPagination` subclasses `CampaignViewTestBase` but replaces
its `setUpTestData` entirely, so `cls.campaign`, `cls.staff_user`, `cls.empty_campaign` and
`cls.most_recent_run` never exist for this class. It happens to be safe today because the
class uses only its own `_pagination_table_url()`, but any inherited helper (`table_url()`,
`list_url()`) would raise `AttributeError`.
**Fix:** call `super().setUpTestData()` first, or subclass `TestCase` directly rather than
`CampaignViewTestBase`.

### IN-08: the reconcile demo notebook leaves a permanently detached event in the shared dev database

**File:** `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` (skip-rule cell)
**Issue:** IN-04's specific complaint is closed — the delete is now scoped to
`blank_url_event_pk`, captured at creation time in the same cell. But the cell now leaves
the superseded `RUN:{pk}:{date}` event behind, detached, in `src/fomo_db.sqlite3`, where it
appears in the real attribution queue as a HIGH-band candidate (this is CR-04's starting
state, seeded into the dev DB by a doc artifact). The next execution's cell-10 sweep
re-links it, so it self-heals across runs, but it persists between them. The 33-08 summary
already records having had to hand-clean `pk=322` from a prior execution of the previous
cell design.
**Fix:** note the residue in the markdown cell above, or clean the detached event by the
`run_keyed_pk` the cell already captured once the assertions have run.

### IN-09: `unlink_event_from_run()`'s `int` branch accepts `bool`, and a `0` pk passes view validation

**File:** `solsys_code/campaign_utils.py:928-930`; `solsys_code/campaign_views.py:786-799`
**Issue:** `isinstance(events, int)` is true for `bool`, so `unlink_event_from_run(True, run)`
filters `event_id=1`. Separately, `_as_pk_or_none('0')` returns `0`, which is not `None`, so
`AttributionDecisionView.post()` accepts it and `unlink_event_from_run(0, run_pk)` falls
into the `int` branch and quietly matches nothing (or, for `run_pk=0`, returns via the
`not run_pk` guard) — the staff member gets no error either way. Neither is reachable
through a real UI today, but both are exactly the "silently mis-filters a scalar" class
WR-04 was raised about.
**Fix:** reject `bool` alongside `str`/`bytes`, and make `_as_pk_or_none()` return `None`
for non-positive values.

### IN-10: the campaign chip's accessible name is inconsistent between its two branches

**File:** `src/templates/tom_calendar/partials/campaign_chip.html:21,23`
**Issue:** The campaign branch renders `title="{name}"` / `aria-label="Campaign: {name}"`
(different strings); the no-campaign branch renders the identical string in both. A screen
reader user hears "Campaign: 3I/ATLAS" in one case and "Attributed run #5 (no campaign)" in
the other, with no shared prefix to signal they are the same control.
**Fix:** give both branches the same `Campaign: ...` / `Attributed run #...` prefix
convention, e.g. `aria-label="Attributed run #{{ deco.run_pk }} — campaign {{ deco.campaign_name }}"`
for both.

### IN-11: the phase's own reproduction fixtures are all `+10` sites, so the noon anchor is under-tested against DST

**File:** `solsys_code/tests/test_campaign_reconciler.py:499-673`
**Issue:** `TestObservingNightBoundary` covers Sydney (+10, August — no DST) and Santiago
(-4, August — no DST). `_observing_night()` relies on Python's wall-clock arithmetic for
`local - timedelta(hours=12)`, which is the behaviour that makes the helper correct across
a DST transition — but no fixture crosses one. Both `America/Santiago` (early September)
and `Australia/Sydney` (early April/October) transition inside the date ranges this feature
operates on.
**Fix:** add one fixture whose observing night spans a DST transition at each site and
assert the derived night, so a future "simplification" to
`(start_time - timedelta(hours=12)).astimezone(zone).date()` (absolute rather than
wall-clock arithmetic — a one-hour-different answer) is caught.

---

_Reviewed: 2026-09-08_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
_Supersedes: 33-REVIEW.md of 2026-09-04 (CR-01..CR-03, WR-01..WR-08, IN-01..IN-05)_
