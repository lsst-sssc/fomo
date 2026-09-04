---
phase: 33-series-identity-reconciler-inversion
reviewed: 2026-09-04T17:59:49Z
depth: deep
files_reviewed: 24
files_reviewed_list:
  - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/admin.py
  - solsys_code/campaign_reconciler.py
  - solsys_code/campaign_tables.py
  - solsys_code/campaign_utils.py
  - solsys_code/campaign_views.py
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
  - solsys_code/tests/test_write_and_reconcile.py
  - solsys_code/views.py
  - src/templates/campaigns/campaignrun_table.html
  - src/templates/tom_calendar/partials/calendar.html
  - src/templates/tom_calendar/partials/event_form.html
findings:
  critical: 3
  warning: 8
  info: 5
  total: 16
status: issues_found
---

# Phase 33: Code Review Report

**Reviewed:** 2026-09-04T17:59:49Z
**Depth:** deep
**Files Reviewed:** 24
**Status:** issues_found

## Summary

Reviewed the Phase 33 diff (`998b45e^..HEAD`, 24 source files) at deep depth: reconciler
inversion (owner -> annotator), display-time campaign decoration, the two new
`CalendarEventMeta` observation-link fields plus migration 0017, and the shared
`unlink_event_from_run()` helper.

`pre-commit run ruff` and `ruff-format` are clean on every changed Python file, and the new
test modules pass (`test_calendar_event_meta_links`, `test_campaign_views.TestCampaignRunRowAnchor`,
11 tests, OK). The migration is genuinely non-destructive and its `TransactionTestCase`
proof is sound. `unlink_event_from_run()`'s `if not run_pk` guard is real and correctly
tested.

Three defects are load-bearing and reproducible:

1. The D-13 row-highlight CSS is inside a `{% extends %}` child template but **outside every
   `{% block %}`**, so Django discards it. The highlight has never rendered and no test
   covers it (CR-01, verified against the Django template engine).
2. `_attributed_nights()` derives the observing night with a plain site-local `.date()`.
   Now that the blank-url restriction has been removed and facility-URL-keyed events match,
   an event starting after local midnight is assigned to the **wrong** night: the reconciler
   mints a duplicate event for the real night *and* spuriously skips the following one
   (CR-02, reproduced).
3. The skip rule only fires when no `RUN:` event exists yet, so the realistic
   reconcile-then-attribute ordering (which is exactly the Phase 34 handoff) leaves two
   calendar entries for the same night forever, and both now carry the campaign chip
   (CR-03, reproduced).

Secondary concerns cluster around the "one shared unlink helper" claim (the admin path does
not actually use it), the new `skipped_nights` counter (never surfaced by the batch
command), a silent audit-destroying detach, and several new tests whose assertions are
satisfied by fixtures other than the one under test.

Reproduction of CR-02/CR-03 was done with a throwaway test module under
`solsys_code/tests/`, run and then deleted; no source file was modified.

## Narrative Findings (AI reviewer)

## Critical Issues

*(BLOCKER tier; `CR-` == `BL-` for downstream consumers.)*

### CR-01: D-13 row-highlight CSS is silently discarded — the highlight never renders

**File:** `src/templates/campaigns/campaignrun_table.html:5-13`
**Severity:** BLOCKER

**Issue:** The file begins with `{% extends 'tom_common/base.html' %}` (line 1). The
`<style>` block sits between `{% block title %}...{% endblock %}` (line 3) and
`{% block content %}` (line 15) — i.e. **outside every block**. Django's `ExtendsNode`
renders the *parent* template with the child's blocks substituted in; any top-level node in
the child that is not inside a `{% block %}` is discarded. So `tr:target { ... }` is never
emitted, and the "highlight the run row the calendar decoration's `#run-{pk}` link lands on"
behaviour that D-13 and the block's own comment promise does not exist.

Verified directly against the installed Django template engine:

```
Template: {% extends "base.html" %}{% block title %}T{% endblock %}<style>ZZZSTYLEZZZ</style>{% block content %}BODY{% endblock %}
Rendered: 'PARENT[BODY]'
STYLE PRESENT: False
```

The contrast case in this same phase is `src/templates/tom_calendar/partials/calendar.html`,
whose `<style>` at line 1 *does* render — because that partial has no `{% extends %}`.

No test asserts the CSS is present. `TestCampaignRunRowAnchor` only asserts the `id="run-N"`
attribute, so the anchor half is covered and the highlight half is not; the feature will stay
broken silently.

**Fix:** move the style into a rendered block. Either put it inside `{% block content %}`,
or (preferred) use the base template's CSS block if one exists:

```django
{% extends 'tom_common/base.html' %}
{% load django_tables2 %}
{% block title %}{{ campaign.name }} — Observing Runs{% endblock %}

{% block content %}
<style>
  /* D-13 (Phase 33 Plan 02): highlight the run row the calendar decoration's
     #run-{pk} link lands on. */
  tr:target {
    background-color: #fff3cd;
    box-shadow: inset 4px 0 0 0 #ffc107;
  }
</style>
<div class="d-flex justify-content-between align-items-center mb-4">
...
```

and add the missing assertion to `TestCampaignRunRowAnchor`:

```python
def test_table_page_ships_the_target_row_highlight_css(self):
    response = self.client.get(self.table_url())
    self.assertContains(response, 'tr:target')
```

---

### CR-02: `_attributed_nights()` assigns a post-local-midnight event to the wrong observing night

**File:** `solsys_code/campaign_reconciler.py:301-328` (specifically line 328)
**Severity:** BLOCKER

**Issue:** The night key is derived as

```python
return {meta.event.start_time.astimezone(site_zone).date() for meta in metas}
```

A plain site-local `.date()` is only correct when the event starts **before** local midnight.
The docstring for `run_night_url()` (line 104-115) explicitly defines `night` as "the
site-local observing night (the same night `sun_event()`'s sunset is computed for)", and
`telescope_runs.py` anchors that definition at **local noon** (`_local_noon_utc`, line 236).
`_attributed_nights()` uses neither convention.

The retired `_adopted_event_for_night()` got away with this because it was restricted to
`event__url=''` — `load_telescope_runs`-created events always start at beginning-of-night,
i.e. before local midnight. This phase deliberately removed that restriction so
"a facility-URL-keyed attributed event (a Phase 34 observation event) must match too". A
facility observation window routinely starts after local midnight, at which point the derived
date is one day too late.

Reproduced (site `Australia/Sydney`, run window 2026-08-01..2026-08-02, one attributed
facility event at `2026-08-01T16:00Z` = `2026-08-02 02:00` local, i.e. observing night
**Aug 1**):

```
PROBE-B result: ReconcileResult(created=1, updated=0, unchanged=0, blocked=0, skipped_nights=1, ...)
PROBE-B urls:  ['RUN:1:2026-08-01', 'https://observe.lco.global/api/requestgroups/2/']
```

Exactly backwards: the reconciler **minted a duplicate** `RUN:1:2026-08-01` alongside the
attributed event for the same real night, and **skipped Aug 2**, which has no coverage at
all. Both failures are the ones D-01/ANNOT-01 exists to prevent, and both are reachable
today via `sync_lco_observation_calendar` events plus Phase 28's attribution queue — this
does not need Phase 34 to ship.

The existing test `test_skip_matches_on_site_local_night_not_naive_utc_date` does not catch
this: its fixture (`14:08Z` -> `00:08` local) is the *only* boundary case where naive-UTC and
local `.date()` differ, and it happens to fall on the correct side.

**Fix:** derive the observing night with the same noon anchor the rest of the codebase uses,
so any local time from noon to noon+24h maps to the starting date:

```python
def _observing_night(start_time, site_zone: ZoneInfo):
    """The site-local observing night a start_time belongs to (noon-anchored, matching
    telescope_runs._local_noon_utc): a 02:00 local start belongs to the PREVIOUS date."""
    local = start_time.astimezone(site_zone)
    return (local - timedelta(hours=12)).date()


def _attributed_nights(run: CampaignRun, site_zone: ZoneInfo) -> set:
    metas = (
        CalendarEventMeta.objects.filter(run_id=run.pk)
        .exclude(event__url__startswith=RUN_URL_NAMESPACE)
        .select_related('event')
    )
    return {_observing_night(meta.event.start_time, site_zone) for meta in metas}
```

Add a regression test with a `16:00Z` / Sydney fixture asserting `skipped_nights == 1` for
the *first* night and `created == 1` for the *second*.

---

### CR-03: skip rule never fires once a `RUN:` event already exists — duplicate entries for the same night, permanently

**File:** `solsys_code/campaign_reconciler.py:370-373`, `456-498`
**Severity:** BLOCKER

**Issue:**

```python
existing = CalendarEvent.objects.filter(url=url).first()
if existing is None and night in attributed_nights:
    totals['skipped_nights'] += 1
    continue
```

The skip is gated on `existing is None`. The module docstring and `_attributed_nights()`
state the contract unconditionally: *"a night with an attributed non-`RUN:` event has no
reconciler event -- the same rule Phase 35's allocation handoff will use."* The
implementation only honours that when the attribution happens **before** the first reconcile.

The opposite ordering is the normal one for the Phase 34 handoff this phase is building
towards: the run is approved, reconciled (mints `RUN:{pk}:{date}`), and only later does the
observation projector / attribution queue attribute a real observation event to the same run
for the same night. Nothing then removes or detaches the reconciler's own event —
`reconcile_run()` builds `active_urls` from *every* night in the window regardless of skip
(line 487-488), so `_detach_stale_family_events()` never sees it as stale.

Reproduced (single-night run, reconcile first, then attribute a facility event):

```
PROBE-A result: ReconcileResult(created=0, updated=0, unchanged=1, blocked=0, skipped_nights=0, ...)
PROBE-A urls:  ['RUN:1:2026-08-01', 'https://observe.lco.global/api/requestgroups/1/']
PROBE-A count: 2
```

Two calendar entries for one observing night, both attributed to the same run, so both now
render the new campaign chip in the month cell — a visibly duplicated night, forever, with
`skipped_nights == 0` giving no signal at all. Every skip test in
`test_campaign_reconciler.py` seeds the attribution *before* the first reconcile, so this
ordering is entirely uncovered.

**Fix:** make the skip unconditional on the attribution, and let the stale-detach step
reclaim the reconciler's now-superseded event by dropping that night's url out of
`active_urls`:

```python
for i in range(n_nights):
    night = run.window_start + timedelta(days=i)
    if night in attributed_nights:
        totals['skipped_nights'] += 1
        continue          # <- no longer gated on `existing is None`
    ...
```

and in `reconcile_run()`, build `active_urls` from the nights the branch actually wrote
rather than re-deriving the full window — e.g. have `_reconcile_classical_nights()` return
the urls it considers current alongside its `ReconcileResult`, so a superseded
`RUN:{pk}:{date}` event is detached back into Phase 28's queue instead of lingering. Add a
test that reconciles, then attributes, then reconciles again, and asserts exactly one event
remains for that night.

## Warnings

### WR-01: `ReconcileResult.skipped_nights` is never surfaced by the batch sweep

**File:** `solsys_code/management/commands/reconcile_campaign_runs.py:49-95`; field added at
`solsys_code/campaign_reconciler.py:93-95`

**Issue:** The command sums `created`/`updated`/`unchanged`/`blocked` and prints only those.
`skipped_nights` is dropped on the floor, so the D-04 operator sweep reports a run with
`created: 0, updated: 0` and no explanation of *why* — indistinguishable from "already
converged". The reconcile demo notebook even documents this gap (cell comment: *"it sums
created/updated/unchanged/blocked and prints no url and no per-run skipped_nights"*) rather
than closing it, and `docs/runbooks/telescope_runs_calendar.rst` describes the skip rule as
operator-visible behaviour with no way to observe it.

**Fix:** accumulate and print it, and log the per-run case like `blocked` already is:

```python
skipped_nights = 0
...
skipped_nights += result.skipped_nights
if result.skipped_nights:
    self.stdout.write(
        f'Run pk={run.pk}: {result.skipped_nights} night(s) already attributed elsewhere -- skipped'
    )
...
f'skipped_nights: {skipped_nights}, '
```

---

### WR-02: the "single writer" invariant is documentation-only — the admin clear path does not use the helper

**File:** `solsys_code/admin.py:391-406`; helper at `solsys_code/campaign_utils.py:860-910`

**Issue:** Commit `789e76b` claims "route all three clear-the-link writers through
`unlink_event_from_run()`", and the helper's docstring calls itself "the single writer that
clears the link again, for every call site that needs to ... the attribution-undo view's
conditional per-pair clear, the reconciler's bulk detach step, and the admin's standalone
clear branch." Only two of the three actually route through it. `CalendarEventMetaAdmin.
save_model()` branch 2 keeps a hand-written in-memory copy:

```python
obj.confirmed_by = None
obj.confirmed_at = None
```

The comment explains *why* the helper can't simply be called before `obj.save()` (it would
be re-persisted), which is correct — but the consequence is that the definition of "what
clearing an attribution means" now lives in two places. If a fourth link/audit field is ever
added to the helper's `.update(...)`, the admin path will silently stop clearing it, and the
admin tests only assert the three fields that exist today.

**Fix:** derive both from one place so they cannot drift, e.g. export the field set from
`campaign_utils` and consume it on both sides:

```python
# campaign_utils.py
UNLINK_CLEARED_FIELDS = {'run': None, 'confirmed_by': None, 'confirmed_at': None}

def unlink_event_from_run(events, run) -> int:
    ...
    return CalendarEventMeta.objects.filter(run_id=run_pk, **event_filter).update(**UNLINK_CLEARED_FIELDS)

# admin.py, branch 2
for field, value in UNLINK_CLEARED_FIELDS.items():
    setattr(obj, field, value)
```

---

### WR-03: the reconciler's detach now destroys audit stamps silently — no count, no log, no compensating trace

**File:** `solsys_code/campaign_reconciler.py:411-453` (line 452-453)

**Issue:**

```python
stale = owned_events(run).exclude(url__in=active_urls)
unlink_event_from_run(stale, run)
```

The helper's return value — the number of rows changed, which the helper's own docstring
calls out as the thing "the caller can use to gate its own follow-on writes" — is discarded,
and the function logs nothing and returns nothing. This phase also newly made this step
destroy `confirmed_by`/`confirmed_at` (a documented, deliberate behaviour change).

Compare the only other path that clears a human confirmation,
`campaign_views.AttributionDecisionView._undo_confirmation()`: it writes a
`CalendarEventDismissal` row precisely so the erased "who/when" leaves a trace. The
reconciler's detach erases the same evidence with *no* replacement record and *no* log line,
during an unattended batch sweep. `_detach_stale_family_events()` also never reports back to
`ReconcileResult`, so neither the command summary nor the staff-action call sites can tell a
human that a confirmation was just discarded.

**Fix:** at minimum log it; ideally count it into the result:

```python
detached = unlink_event_from_run(stale, run)
if detached:
    logger.warning(
        'Reconcile detached %s stale-family event(s) from run pk=%s, clearing their '
        'confirmation stamps.',
        detached,
        run.pk,
    )
return detached
```

and thread the count into `ReconcileResult` (a `detached: int = 0` field) so
`reconcile_campaign_runs` and the four staff-action call sites can surface it.

---

### WR-04: `unlink_event_from_run()` type dispatch silently mis-filters any scalar that is not `CalendarEvent` or `int`

**File:** `solsys_code/campaign_utils.py:901-910`

**Issue:**

```python
if isinstance(events, CalendarEvent):
    event_filter = {'event_id': events.pk}
elif isinstance(events, int):
    event_filter = {'event_id': events}
else:
    event_filter = {'event__in': events}
```

The `else` branch is an unvalidated catch-all. A `str` pk — the shape a POST parameter
naturally arrives in — is iterable, so `unlink_event_from_run('12', run)` produces
`event__in=['1', '2']` and clears the attribution on events 1 and 2 instead of event 12. No
exception is raised. Today's call sites happen to be safe (`campaign_views._as_pk_or_none()`
returns `int`, the reconciler passes a queryset), but the helper is documented as the shared
entry point for "every call site that needs to", and its own signature advertises
`CalendarEvent | int | Any`.

The tests in `TestUnlinkEventFromRun` cover the `CalendarEvent` and `None`-run cases but
neither the bare-`int` branch nor the queryset branch directly.

**Fix:** narrow the dispatch and fail loudly on anything else:

```python
if isinstance(events, CalendarEvent):
    event_filter = {'event_id': events.pk}
elif isinstance(events, int):
    event_filter = {'event_id': events}
elif isinstance(events, (str, bytes)):
    raise TypeError(f'unlink_event_from_run() needs an int pk, not {events!r} -- a str is iterable and would mis-filter.')
else:
    event_filter = {'event__in': events}
```

and add a test for the bare-`int` call shape `_undo_confirmation()` actually uses.

---

### WR-05: several new decoration tests are satisfied by fixtures other than the one under test

**File:** `solsys_code/tests/test_calendar_template.py:768-782` and `:869-905`

**Issue:** Three of the new month-view tests cannot fail for the reason they claim:

1. `test_pending_review_run_shows_no_marker_for_staff_and_anonymous` (line ~889) asserts
   `assertNotIn('Should Stay Hidden Scope', content)` — the run's `telescope_instrument`.
   The month cell never renders `telescope_instrument` at all; the only thing
   `campaign_decoration()` puts in the month grid is `campaign_name` in the chip's `title=`.
   And `pending_run.campaign` is the *same* `Survival Guard Campaign` as `approved_run`, so
   even if the `is_publicly_visible` gate were deleted the assertion would still pass. The
   pending-run gate is therefore **untested in the month view** — the exact leak the gate
   exists to prevent.

2. `test_chip_does_not_consume_title_truncation_budget` (line ~776) asserts the full titles
   `'AllDay Attr'` (11 chars) and `'Timed Attr'` (10 chars) appear. The filters are
   `truncatechars:18` and `truncatechars:16`, so those titles are never truncated whether or
   not the chip is inside the filter expression. The test proves nothing.

3. `test_no_campaign_run_renders_marker_and_no_table_href` (line ~869) asserts
   `assertIn('cal-campaign-chip', content)`, but `linked_event` and `pii_event` in the same
   September grid already emit chips, so the no-campaign case is not isolated. The
   `assertNotIn(self._campaign_table_href())` half is also vacuous — the month cell never
   renders `table_url` for any event.

**Fix:** assert on values only the fixture under test can produce.

```python
def test_pending_review_run_shows_no_marker_for_staff_and_anonymous(self):
    # Give the pending run its OWN campaign, so its name is the discriminator.
    ...
    self.assertNotIn(f'title="{self.pending_campaign.name}"', anon_content)
    self.assertNotIn(f'title="{self.pending_campaign.name}"', staff_content)

def test_chip_does_not_consume_title_truncation_budget(self):
    # A title at exactly the truncation budget, so a chip folded into the filter
    # expression would visibly shorten it.
    ...
    self.assertIn('Eighteen Char Ttl…'[:18], content)

def test_no_campaign_run_renders_marker_and_no_table_href(self):
    self.assertIn(f'title="{NO_CAMPAIGN_LABEL}"', content)
```

---

### WR-06: runbook and admin docstring document an inline operation that does not exist

**File:** `docs/runbooks/telescope_runs_calendar.rst:812-818`; `solsys_code/admin.py:75-78`

**Issue:** Under "Two things to know about that inline", the runbook now says:

> Clearing the **Attributed campaign run** value un-attributes the entry: it removes only
> the pop-up block and the month-cell marker, and clears the "confirmed by"/"confirmed at"
> record along with it

`CalendarEventMetaInline` declares `fk_name = 'run'` (`admin.py:95`). Django's inline
formsets exclude the parent foreign key from the child form, so `run` is **not rendered as
an editable field on that inline at all** — there is no value there to clear. The same
claim sits in the inline's own docstring ("Clearing the `run` value on a row un-attributes
the event...", lines 75-78), which this phase rewrote rather than corrected.

The audit-stamp clearing the new prose describes is only implemented on
`CalendarEventMetaAdmin.save_model()` (the standalone *Calendar event metas* change page),
not on the run's inline. An operator following the runbook step-by-step on the inline will
find no such field.

**Fix:** point the bullet at the surface where the operation actually exists, and correct
the inline docstring:

```rst
* To un-attribute an entry, open it under **Django admin -> Solsys code -> Calendar event
  metas** and clear the **Attributed campaign run** value there. That clears the
  "confirmed by"/"confirmed at" record along with the link. On the run's own inline the
  attribution field is not editable -- delete the row instead (the row IS the link).
```

---

### WR-07: `event_form.html` gates the decoration twice, in two different places

**File:** `src/templates/tom_calendar/partials/event_form.html:118-136`

**Issue:** The template keeps the pre-existing gate

```django
{% with run=event.telescope_label_meta.run %}
{% if run.is_publicly_visible %}
```

and then adds a second, inner gate on the tag's return value:

```django
{% campaign_decoration event as deco %}
{% if deco %}
```

`campaign_decoration()` already applies exactly the same `run is None or not
run.is_publicly_visible` rule (`calendar_display_extras.py:464`). Two independent copies of
one visibility rule is precisely the drift D-10 warns about elsewhere in this codebase: the
template gate silently wins, so a future change to the tag's rule would not take effect in
the modal. The outer `{% with %}` also costs a separate companion-row dereference that the
tag then repeats.

Note the outer branch cannot simply be deleted — the `{% elif not run and
request.user.is_staff %}` arm (27-UAT Test 9) depends on `run`. Restructure rather than
remove.

**Fix:**

```django
{% campaign_decoration event as deco %}
{% if deco %}
  ... the attributed-run block, using deco.* only ...
{% elif not event.telescope_label_meta.run and request.user.is_staff %}
  ... the staff attribution-queue hint ...
{% endif %}
```

---

### WR-08: the decoration's `#run-{pk}` link lands nowhere for a run past page 1 of the campaign table

**File:** `solsys_code/templatetags/calendar_display_extras.py:466-473`;
`solsys_code/campaign_views.py:130` (`table_pagination = {'per_page': 25}`)

**Issue:** `table_url` is built as `reverse('campaigns:table', args=[run.campaign_id]) +
f'#run-{run.pk}'` with no page parameter. `CampaignRunTableView` paginates at 25 rows and
default-sorts by `window_start` descending. A campaign with more than 25 runs — the 3I/ATLAS
coordination case this whole feature exists for — will land the operator on page 1 with no
matching `id="run-{pk}"` anchor anywhere in the document, so the browser scrolls nowhere and
(per CR-01, once fixed) nothing highlights. The failure is completely silent. An active
filter in `CampaignRunFilterSet` can produce the same outcome on any page.

**Fix:** compute the run's page (or at least clear filters) when building the link, e.g.:

```python
# Position within the same default ordering the table uses, so the link lands on the page
# that actually contains this run's row.
if run.campaign_id is not None:
    position = (
        CampaignRun.objects.filter(campaign_id=run.campaign_id)
        .exclude(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
        .order_by(F('window_start').desc(nulls_last=True))
        .values_list('pk', flat=True)
    )
    ...
```

If that is judged too costly for a per-event display tag, at minimum document the limitation
in the runbook and add a test pinning the >25-run behaviour, so it is a known constraint
rather than a silent dead link.

## Info

### IN-01: `campaign_decoration()` returns an unused `run_pk` key

**File:** `solsys_code/templatetags/calendar_display_extras.py:477`
**Issue:** Neither `calendar.html` nor `event_form.html` reads `deco.run_pk` (grepped: no
hits). It is already baked into `table_url`. Dead payload on a documented "exactly these
keys" contract.
**Fix:** drop the key, or add a test that pins a consumer for it.

### IN-02: `n_nights` is derived twice per classical reconcile

**File:** `solsys_code/campaign_reconciler.py:362` and `:487`
**Issue:** `(run.window_end - run.window_start).days + 1` is computed independently inside
`_reconcile_classical_nights()` and again in `reconcile_run()` to build `active_urls`. Two
copies of the window arithmetic that must agree exactly for the detach step to be a no-op.
**Fix:** have `_reconcile_classical_nights()` return the urls it wrote (this also falls out
of the CR-03 fix), so `reconcile_run()` never re-derives the window.

### IN-03: the month-cell campaign marker has no accessible name

**File:** `src/templates/tom_calendar/partials/calendar.html:252-254`, `:281-283`
**Issue:** `<span class="cal-campaign-chip" title="{{ campaign_deco.campaign_name }}">&#9873;</span>`
— a decorative glyph plus a `title` attribute. `title` is not reliably announced by screen
readers and is unreachable on touch. The campaign attribution is therefore visual-only.
**Fix:** `<span class="cal-campaign-chip" title="..." aria-label="Campaign: {{ campaign_deco.campaign_name }}" role="img">&#9873;</span>`

### IN-04: the reconcile demo notebook deletes rows from the live dev database

**File:** `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` (cell 18)
**Issue:** `CalendarEvent.objects.filter(url=existing_run_keyed_url).delete()` runs against
`src/fomo_db.sqlite3`, not a test database. It is scoped to a url the same notebook run just
created, so the blast radius is small, but it is a new unconditional `.delete()` in a doc
artifact that a reader is invited to execute. (`campaign_lifecycle_demo.ipynb` cell 6's
`CampaignRun.objects.filter(campaign=...).delete()` has the same shape and predates this
phase.)
**Fix:** guard the delete on the row having been created by this notebook run (capture the
pk from the earlier sweep) and print a loud banner in the markdown cell above it.

### IN-05: a run with no campaign still gets a "campaign marker" reading `(no campaign)`

**File:** `solsys_code/templatetags/calendar_display_extras.py:476`;
`src/templates/tom_calendar/partials/calendar.html:252`
**Issue:** For a null-campaign run the chip renders with `title="(no campaign)"` — a
campaign marker whose tooltip says there is no campaign. Defensible (the attribution is
real even when the campaign is not), but it is worth an explicit decision rather than a
side effect of `NO_CAMPAIGN_LABEL` reuse; the modal, where more context is visible, is a
better place for it than a bare month-cell glyph.
**Fix:** either suppress the month-cell chip when `campaign_name` is `NO_CAMPAIGN_LABEL`, or
give it distinct tooltip text such as `Attributed run #{{ deco.run_pk }} (no campaign)`.

---

_Reviewed: 2026-09-04T17:59:49Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
