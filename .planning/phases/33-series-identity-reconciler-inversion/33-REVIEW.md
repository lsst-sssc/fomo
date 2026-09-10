---
phase: 33-series-identity-reconciler-inversion
reviewed: 2026-09-09T00:00:00Z
depth: deep
files_reviewed: 30
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
  - solsys_code/tests/test_bootstrap5_rendering.py
  - solsys_code/tests/test_calendar_event_meta_links.py
  - solsys_code/tests/test_calendar_template.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/tests/test_campaign_attribution.py
  - solsys_code/tests/test_campaign_attribution_views.py
  - solsys_code/tests/test_campaign_reconciler.py
  - solsys_code/tests/test_campaign_views.py
  - solsys_code/tests/test_null_campaign_guards.py
  - solsys_code/tests/test_reconcile_campaign_runs.py
  - solsys_code/tests/test_write_and_reconcile.py
  - solsys_code/views.py
  - src/fomo/settings.py
  - src/templates/campaigns/campaignrun_table.html
  - src/templates/tom_calendar/partials/calendar.html
  - src/templates/tom_calendar/partials/campaign_chip.html
  - src/templates/tom_calendar/partials/event_form.html
findings:
  critical: 1
  warning: 8
  info: 6
  total: 15
status: issues_found
---

# Phase 33: Code Review Report (post second gap-closure wave)

**Reviewed:** 2026-09-09
**Depth:** deep
**Files Reviewed:** 30
**Status:** issues_found

## Summary

Re-review of the full Phase 33 diff (`998b45e^..HEAD`, HEAD `fb99f9a`) after the second
gap-closure wave (plans 33-09, 33-10, 33-11). Each of the 13 findings in the 2026-09-08
review was re-verified against the code at HEAD, not against the SUMMARY files.

**Verified closed (8 of 13):** CR-04, WR-10, WR-11, WR-13, IN-08, and the bulk of WR-09 and
WR-12. Spot-checks that actually held up:

- `_stale_attributions()` (`campaign_reconciler.py:482-527`) really is read-only — no
  `.save()`/`.update()`/`.create()`/`.delete()` on that path — and is the *same* predicate
  consumed by both the write path and the dry-run preview, so `would_detach` cannot drift
  from what a real sweep clears (WR-11).
- The `confirmed_by__isnull=True` filter genuinely closes CR-04's confirm/erase loop, and
  `test_staff_reconfirmation_of_the_detached_run_keyed_event_survives_every_later_sweep`
  exercises the exact reconcile→attribute→reconcile→**re-confirm**→reconcile→reconcile
  sequence the prior review said was missing, asserting the stamp survives *and* that no
  `CalendarEventDismissal` is written.
- Ownership is now decided before the night's outcome, and the blocked night's url is added
  to `active_urls`, so an attributed-and-contested night reports `blocked=1` (WR-13).
- `_message_reconcile_side_effects()` exists once and is called from all three
  `CampaignRunDecisionView` reconcile call sites, and `_resolve_site()`'s success message is
  no longer keyed on `skipped_reason is None` alone (WR-12).
- The notebooks really do run against a `tempfile.mkdtemp()` copy and assert the resolved
  `DATABASES['default']['NAME']` in committed output, which retires IN-08 entirely.

443 tests across the nine phase test modules pass, and `pre-commit run ruff` /
`ruff-format` are clean.

**Still open (5 of 13):** WR-08 (carried as accepted-and-documented), WR-09 (reduced), IN-06,
IN-07, IN-09, IN-11. IN-10 is half-closed.

**This review's blocker is new, and the green suite is exactly why it survived.** Plan 33-11
correctly identified that the create-event modal renders `event_form.html` with **no `event`
in context**, and guarded `campaign_decoration()` against the resulting empty-string
placeholder. It did not guard the *other* tag invoked from the same template, in the branch
that is now reachable precisely because `campaign_decoration()` returns `None` there —
`high_band_attribution_candidates`. For any **staff** user, clicking "+ New Event", clicking
an empty day cell, or submitting an invalid create form raises
`ValueError: Field 'id' expected a number but got ''` — a 500 on the primary calendar-write
path. I reproduced this with a throwaway test module under `solsys_code/tests/`, run and then
deleted; no source file was modified.

The phase's own G-33-2 browser test does not catch it for two independent reasons: it runs
unauthenticated, and its only assertions (`#cal-modal.show` is visible, `pageerror` list is
empty) are satisfied by a server 500 — htmx fires `htmx:afterRequest` for non-2xx responses,
so the modal still opens, just with an unswapped body.

The remaining new findings cluster on two themes: staff-facing messages that assert a cause
they have not verified, and a Bootstrap 5 migration that stopped at the modal open call while
the same two files still carry Bootstrap 4-only utility classes.

## Narrative Findings (AI reviewer)

## Critical Issues

*(BLOCKER tier; `CR-` == `BL-` for downstream consumers.)*

### CR-01: the create-event modal 500s for every staff user — 33-11 guarded one tag on this template and left its twin

**File:** `src/templates/tom_calendar/partials/event_form.html:167,178`;
`solsys_code/templatetags/attribution_display_extras.py:39`;
`solsys_code/campaign_attribution.py:575,577`;
`solsys_code/templatetags/calendar_display_extras.py:465-474`
**Severity:** BLOCKER

**Issue:** `campaign_decoration()` now carries an explicit guard whose comment states the
problem exactly:

```python
if not isinstance(event, CalendarEvent):
    # Rule 1 fix (33-11): the create-event form context has no `event` key at all --
    # Django's template engine resolves the missing variable to the empty-string
    # invalid-variable placeholder rather than raising ...
    return None
```

That guard makes `deco` falsy on the create form, which hands control to the very next branch
in the same template:

```django
{% elif not event.telescope_label_meta.run and request.user.is_staff %}
    ...
    {% high_band_attribution_candidates event as attribution_candidates %}
```

`event` is the empty string here, so `event.telescope_label_meta.run` resolves to `None`,
`not None` is `True`, and for a staff user the branch is taken. `high_band_attribution_candidates`
receives `''` and passes it straight into
`CalendarEventDismissal.objects.filter(event='')`, which raises.

Reproduced against a real test database (`python manage.py test` on a throwaway module,
since deleted):

```
ANON STATUS 200
RAISED: ValueError Field 'id' expected a number but got ''.
```

Reachable from three ordinary staff actions on the calendar page, all of which `GET`
`calendar:create-event`:

1. the "+ New Event" button (`calendar.html:209-214`),
2. clicking any empty area of a day cell (`calendar.html:225-229`),
3. re-rendering after an invalid create submission (`tom_calendar/views.py:167` re-renders
   `event_form.html` with `action='create'` and no `event`).

The tag's own docstring claims "Never raises: delegates entirely to `candidates_for_event()`,
which itself never raises" — `candidates_for_event()`'s "never raises" contract is written for
a real `CalendarEvent`, and neither function type-checks its argument.

This branch was equally reachable before Phase 33 (the old `{% with run=event.telescope_label_meta.run %}`
form resolved `run` to `''`, so `{% elif not run and request.user.is_staff %}` was true for the
same reason), so the defect is pre-existing — but 33-11 diagnosed this exact hazard on this
exact template, fixed one of the two tags, and shipped a browser test that reports the
surface as healthy.

**Fix:** guard the tag the same way `campaign_decoration()` is guarded, so the "never raises"
docstring is true:

```python
# solsys_code/templatetags/attribution_display_extras.py
from tom_calendar.models import CalendarEvent

@register.simple_tag
def high_band_attribution_candidates(event) -> list[campaign_attribution.AttributionCandidate]:
    if not isinstance(event, CalendarEvent):
        # The create-event form context has no `event` key; Django resolves it to the
        # empty-string invalid-variable placeholder. Same guard, same reason, as
        # calendar_display_extras.campaign_decoration().
        return []
    return [c for c in campaign_attribution.candidates_for_event(event) if c.band == campaign_attribution.BAND_HIGH]
```

and add a plain Django test (no Playwright needed) that pins both roles, since nothing under
`solsys_code/tests/` currently requests `calendar:create-event` at all:

```python
def test_create_event_form_renders_for_staff(self):
    User.objects.create_user(username='s', password='pw', is_staff=True)
    self.client.login(username='s', password='pw')
    self.assertEqual(self.client.get(reverse('calendar:create-event')).status_code, 200)

def test_create_event_form_renders_for_anonymous(self):
    self.assertEqual(self.client.get(reverse('calendar:create-event')).status_code, 200)
```

## Warnings

### WR-01: the G-33-2 browser tests pass against a 500 and never exercise a staff session

**File:** `solsys_code/tests/test_bootstrap5_rendering.py:108-176`

**Issue:** All four new modal tests assert only that `#cal-modal.show` becomes visible and
that the Playwright `pageerror` list is empty. Neither assertion is sensitive to the server
response status: htmx fires `htmx:afterRequest` (and therefore the inline
`hx-on::after-request` handler) for non-2xx responses too, and simply does not swap — so the
modal opens with an empty body and the test passes. A server-side 500 is not a browser
`pageerror`.

`test_calendar_modal_opens_for_new_event_button_with_no_page_errors` (`:130-143`) is the only
test covering `calendar:create-event`, and it has no content assertion at all — unlike its
sibling at `:124-126`, which does check `#cal-modal-body` text. Combined with the fact that
no test in the suite ever logs in for a calendar page, CR-01 is invisible to the entire
443-test run.

**Fix:** assert on the response, not just on visibility, and add a staff variant:

```python
responses = []
self.page.on('response', lambda r: responses.append(r))
...
assert self.page.locator('#cal-modal-body').inner_text().strip() != ''
assert all(r.status < 400 for r in responses if 'calendar' in r.url)
```

and log the Playwright context in as a staff user for at least one of the four cases (set the
`sessionid` cookie from a `Client().login()`, or POST the login form once in `setUp`).

---

### WR-02: `_resolve_site()`'s fallback message asserts a cause it has not checked, and can print a literal "0 night(s)"

**File:** `solsys_code/campaign_views.py:732-741`

**Issue:** WR-12's fix keyed the success message on what happened, but the `else` branch
treats "not created and not updated" as synonymous with "every night was already covered":

```python
if result.skipped_reason is not None:
    messages.success(request, 'Site resolved.')
elif result.created or result.updated:
    messages.success(request, 'Site resolved — run added to the calendar.')
else:
    messages.success(
        request,
        f'Site resolved — {result.skipped_nights} night(s) are already covered by entries '
        'attributed to this run, so no new calendar entries were created.',
    )
```

Two other outcomes land in that same branch with `skipped_nights == 0`:

- `unchanged > 0` — the documented retry state. `_resolve_site()` is reachable with
  `site_needs_review=True` and a real, non-placeholder site (finding 8c's "projection-failed
  retry" path); a retry that now succeeds against already-existing events reports
  `created=0, updated=0, unchanged=n`.
- `blocked > 0` — every night's `RUN:` key is attributed to a different run.

Both produce **"Site resolved — 0 night(s) are already covered by entries attributed to this
run, so no new calendar entries were created."** The count is nonsense and the stated cause is
false. `test_campaign_approval.py` only covers the `skipped_nights > 0` case
(`:1117-1118`), so neither is caught.

**Fix:** branch on the counter that is actually non-zero, and keep a neutral fallback:

```python
elif result.skipped_nights:
    messages.success(
        request,
        f'Site resolved — {result.skipped_nights} night(s) are already covered by entries '
        'attributed to this run, so no new calendar entries were created.',
    )
else:
    messages.success(request, 'Site resolved — the calendar was already up to date.')
```

---

### WR-03: `blocked` — the one signal that says "someone else owns this night" — reaches the sweep but none of the four staff actions

**File:** `solsys_code/campaign_views.py:435-461`;
`solsys_code/management/commands/reconcile_campaign_runs.py:79-80`

**Issue:** WR-13's premise was that `blocked` is "the one signal that tells an operator
'someone else owns this night's entry, go look at it'". The reconciler now reports it
correctly and `reconcile_campaign_runs` prints a per-run stderr line for it, but
`_message_reconcile_side_effects()` — the shared "what did the reconcile actually do"
messenger introduced by WR-12's fix for exactly these four surfaces — emits messages for
`detached` and `detach_declined` only.

A staff member clicking **Resolve**, **Approve**, **Mark cancelled** or **Mark weather
failure** on a run whose nights are contested therefore sees "Site resolved" / "Run status
updated" and nothing else, while `blocked=n` events were silently left alone. Under WR-02
above they additionally get the false "0 night(s) are already covered" line.

**Fix:** add the third counter to the one place that exists for it:

```python
if result.blocked:
    messages.warning(
        request,
        f'{result.blocked} calendar entr{"y" if result.blocked == 1 else "ies"} in this '
        "run's key namespace are attributed to a different run and were left untouched.",
    )
```

---

### WR-04: the Bootstrap 5 migration stopped at the modal call — both touched templates still carry Bootstrap 4-only utility classes

**File:** `src/templates/tom_calendar/partials/calendar.html:218,220,225,299,305,312`;
`src/templates/campaigns/campaignrun_table.html:24,27,29,30,40,41,51,52`

**Issue:** This site runs Bootstrap 5 (`django_bootstrap5` + `crispy_bootstrap5` in
`settings.py:53-54`, `CRISPY_TEMPLATE_PACK = 'bootstrap5'` at `:118`, tomtoolkit 3.0.1's
`tom_common/base.html:10` emitting `{% bootstrap_css %}`), and plan 33-11's own comment in
`calendar.html:208` states this as the reason for the modal change. But FOMO's *full override*
of the month partial still uses classes Bootstrap 5 deleted:

```django
<div class="cal-grid border-top border-left">                      <!-- BS5: border-start -->
  <div class="... font-weight-bold ... border-right border-bottom p-2">   <!-- BS5: fw-bold / border-end -->
  <div class="cal-day... border-right border-bottom p-1"           <!-- BS5: border-end -->
  <a class="text-secondary mr-2" ...>                              <!-- BS5: me-2 -->
  <span class="cal-legend-swatch mr-3" ...>                        <!-- BS5: me-3 -->
```

The upstream partial this file was forked from has already been migrated
(`border-top border-start`, `fw-bold`, `border-end`), so the divergence is visible side by
side. Result: the month grid renders with no left/right cell borders, unbolded day headers,
and legend chips with no right margin. `campaignrun_table.html` — which this phase edited to
add the D-13 `additional_css` block — has the same problem (`font-weight-bold`, `mr-2`,
`form-group`), so the very page the calendar decoration links to renders unstyled labels.

Pre-existing, not introduced here — but the phase's stated scope for G-33-2 was "the calendar
pop-up Bootstrap 5 migration", and it added a browser test suite whose whole purpose is
proving BS5 rendering works.

**Fix:** sweep both files for BS4-only utilities:
`border-left`→`border-start`, `border-right`→`border-end`, `mr-N`→`me-N`, `ml-N`→`ms-N`,
`font-weight-bold`→`fw-bold`, `form-group`→`mb-3`. Then add a rendering assertion to
`test_bootstrap5_rendering.py` in the shape of the existing `.form-row` check:

```python
def test_calendar_page_has_no_bootstrap4_only_utility_classes(self):
    self.page.goto(self._calendar_url())
    for bs4_class in ('border-left', 'border-right', 'mr-2', 'mr-3', 'font-weight-bold'):
        assert self.page.locator(f'.{bs4_class}').count() == 0, bs4_class
```

---

### WR-05: the calendar create/update endpoints this phase's templates drive have no authentication at all

**File:** `src/templates/tom_calendar/partials/calendar.html:209-233`;
`tom_calendar/views.py:149` (`def create_event(request):`, no decorator/mixin)

**Issue:** `calendar:create-event` is an unauthenticated read *and write* endpoint — an
anonymous `GET` returns 200 (verified) and an anonymous `POST` calls `form.save()` and creates
a `CalendarEvent` row on the shared calendar. FOMO's overridden month partial wires three
separate click targets to it and this phase added four browser tests that exercise it
anonymously without noticing.

The defect is upstream (tomtoolkit 3.0.1), so it is not this phase's regression — but FOMO
overrides this template, ships this calendar as a public page, and the phase's own work put
new eyes on exactly these handlers.

**Fix:** wrap the upstream views in FOMO's own URL layer rather than editing the vendored
package, e.g. in `src/fomo/urls.py`:

```python
from django.contrib.auth.decorators import login_required
from tom_calendar.views import create_event, update_event

path('calendar/create/', login_required(create_event), name='create-event'),
path('calendar/update/<int:pk>/', login_required(update_event), name='update-event'),
```

placed *before* `include('tom_calendar.urls')`, plus a test asserting an anonymous `POST`
does not create a `CalendarEvent`. If the write is genuinely meant to be public, record that
decision in `docs/runbooks/telescope_runs_calendar.rst` — right now it is neither gated nor
documented.

---

### WR-06: a re-classified run's human-confirmed events can now never converge, and the runbook tells the operator there is nothing to do

**File:** `solsys_code/campaign_reconciler.py:482-527,530-595`;
`docs/runbooks/telescope_runs_calendar.rst` (`detach_declined` counter description)

**Issue:** 33-10's `confirmed_by__isnull=True` guard is correct for CR-03's supersession case
(the reconciler minted the event; a human later confirmed it; the sweep must not erase that).
But `_detach_stale_family_events()` covers **two** causes — its own docstring and the
`detached` field docstring both say so — and the guard is applied to both.

For the 29-REVIEW.md CR-01 re-classification case (an admin corrects `telescope_class`/`site`
so the whole old key family goes stale), a human-confirmed leftover event is now **never**
detached. It stays attributed to the run, in a key family the run no longer belongs to, and
keeps rendering on the calendar "looking like a live commitment forever" — the exact outcome
the convergence step was added to prevent. Every subsequent sweep re-reports it as
`detach_declined`, permanently.

The runbook then closes the loop the wrong way:

> There is nothing for an operator to do about a non-zero ``detach_declined``: it is a report
> that a human decision was respected.

That is true for supersession and false for re-classification, where the operator does need to
act (clear the attribution from **Django admin -> Solsys code -> Calendar event metas**, or
delete the stale event) before the calendar is correct again. No test covers a *confirmed*
stale-family row on the re-classification path — `test_campaign_reconciler.py:1534`'s sibling
test uses the shrink-the-window trigger, which is the supersession cause.

**Fix:** split the counter by cause, or — cheaper and adequate — correct the runbook and give
the operator the action:

```rst
``detach_declined`` counts companion rows the sweep deliberately did NOT release because a
person had already confirmed the attribution. For a night superseded by another attributed
entry there is nothing to do: the human decision stands. For an entry left over from a key
family the run no longer belongs to (after a ``telescope_class``/``site`` correction), the
entry is now stale *and* pinned by that confirmation -- clear its **Attributed campaign run**
value under **Django admin -> Solsys code -> Calendar event metas**, or delete the entry,
before the calendar is correct again.
```

and add a regression test asserting a confirmed stale-*family* row is declined and reported.

---

### WR-07: the `#run-{pk}` link still lands nowhere for a run past page 1 (carried forward from WR-08, unfixed by design)

**File:** `solsys_code/templatetags/calendar_display_extras.py:479-486`;
`solsys_code/campaign_views.py` (`table_pagination = {'per_page': 25}`)

**Issue:** Unchanged. `table_url` is still `reverse('campaigns:table', args=[run.campaign_id]) + f'#run-{run.pk}'`
with no page parameter. Plan 33-06 pinned the behaviour with `TestCampaignRunAnchorPagination`
(`test_campaign_views.py:686-727`, one negative and one positive assertion — an adequate pair)
and documented it in the runbook, which satisfies the prior review's stated minimum. Recorded
as accepted-and-documented rather than escalated — but the 3I/ATLAS coordination case this
feature exists for is exactly a campaign with more than 25 runs, and the link is still
silently dead there.

**Fix:** carry as an explicit backlog item for Phase 34/35 (resolve the run's page in the view
that renders the campaign table, or support a `?run=<pk>` query parameter that jumps to it)
rather than leaving it as a permanent documented limitation.

---

### WR-08: the superseded night still shows two calendar entries, and no operator action removes the leftover (reduced from WR-09)

**File:** `solsys_code/campaign_reconciler.py:530-595`;
`docs/runbooks/telescope_runs_calendar.rst` (skip-rule section)

**Issue:** Half closed. The runbook's misleading "re-confirm **or discard**" is gone, the
re-confirm path is now accurate and permanent, and "Clearing the other entry's attribution
instead brings the reconciler's own entry back, in place (same record, same url), on the next
sweep" is verifiably true (`_may_write()` falls through to the namespace check for a
detached row, so the update path re-links it).

Still open: the `CalendarEvent` row itself survives by design and keeps rendering in the month
grid, so the night carries two entries. D-12 removed the campaign label from `event_title()`,
so the released one now renders as a bare `FTN/MuSCAT3` with no campaign chip (it was
detached) and no campaign name in its title — an unexplained duplicate with nothing linking it
back to the run it came from. The runbook still names no way to remove it.

**Fix:** name the removal path explicitly, one sentence in the skip-rule section:

```rst
Releasing an entry does not remove it from the calendar; it stays on the night alongside the
attributed entry until someone re-confirms it or deletes it under **Django admin -> Tom
calendar -> Calendar events**.
```

## Info

### IN-01: `writable_events()`'s docstring still names a consumer that does not use it

**File:** `solsys_code/campaign_reconciler.py:142-162` vs. `:523`
**Issue:** Unchanged from IN-06. The docstring says "Every write path (reconcile's detach
step, the run-deletion cascade) must go through `writable_events()` instead." The cascade does
(`models.py:453-455`); the detach step goes through `_stale_attributions()`, which uses
`owned_events()` plus a `run_id=run.pk` companion-row filter. Functionally equivalent, but the
docstring is the only statement of a rule the code does not follow.
**Fix:** amend the docstring to say the detach achieves the same guarantee through
`_stale_attributions()`'s `run_id` filter, or switch it to `writable_events(run)`.

### IN-02: the WR-08 pagination test still overrides `setUpTestData` without calling `super()`

**File:** `solsys_code/tests/test_campaign_views.py:695-715`
**Issue:** Unchanged from IN-07. `TestCampaignRunAnchorPagination` subclasses
`CampaignViewTestBase` but replaces `setUpTestData` entirely, so `cls.campaign`,
`cls.staff_user`, `cls.empty_campaign` and `cls.most_recent_run` never exist for this class.
Safe today because it uses only its own `_pagination_table_url()`; any inherited helper would
raise `AttributeError`.
**Fix:** call `super().setUpTestData()` first, or subclass `TestCase` directly.

### IN-03: `unlink_event_from_run()`'s `int` branch still accepts `bool`, and `_as_pk_or_none('0')` still returns `0`

**File:** `solsys_code/campaign_utils.py:928-946`; `solsys_code/campaign_views.py:829-842`
**Issue:** Unchanged from IN-09. `isinstance(events, int)` is `True` for `bool`, and the
`str | bytes` rejection added by 33-07 sits *after* the `int` branch, so
`unlink_event_from_run(True, run)` still filters `event_id=1`. Separately `_as_pk_or_none('0')`
returns `0`, which is not `None`, so `AttributionDecisionView.post()` accepts it and the
downstream call quietly matches nothing with no error shown to the staff member.
**Fix:** reject `bool` alongside `str`/`bytes` (`isinstance(events, bool)` checked first), and
make `_as_pk_or_none()` return `None` for non-positive values.

### IN-04: the campaign chip's `title` and `aria-label` still disagree in the campaign branch

**File:** `src/templates/tom_calendar/partials/campaign_chip.html:21,23`
**Issue:** Half closed from IN-10. The no-campaign branch now renders the identical string in
both attributes. The campaign branch still renders `title="{{ name }}"` against
`aria-label="Campaign: {{ name }}"`, so a sighted user hovering and a screen-reader user hear
two different names for the same control.
**Fix:** use `title="Campaign: {{ deco.campaign_name }}"` in the campaign branch so both
branches share the prefix convention.

### IN-05: `_observing_night()`'s wall-clock arithmetic is still untested across a DST transition

**File:** `solsys_code/tests/test_campaign_reconciler.py:499-673`
**Issue:** Unchanged from IN-11. `TestObservingNightBoundary` still covers only Sydney (+10,
August) and Santiago (-4, August) — neither in DST. `_observing_night()` depends on Python's
wall-clock `local - timedelta(hours=12)` being different from absolute arithmetic across a
transition, and both `America/Santiago` (early September) and `Australia/Sydney` (early
April/October) transition inside the date ranges this feature operates on.
**Fix:** add one fixture per site whose observing night spans a transition and assert the
derived night, so a future "simplification" to
`(start_time - timedelta(hours=12)).astimezone(zone).date()` is caught.

### IN-06: the modal-open expression is inlined three times where upstream keeps one helper

**File:** `src/templates/tom_calendar/partials/calendar.html:212,228,233`
**Issue:** `bootstrap.Modal.getOrCreateInstance(document.getElementById('cal-modal')).show();`
is repeated verbatim in three `hx-on::after-request` attributes. Upstream's partial defines a
single `showModal()` helper in a `{% block extra_javascript %}` at the end of the file; FOMO's
override dropped the helper and duplicated the body. Three copies is three places to edit if
the modal id or API ever changes, and the expression throws a `TypeError` on a null element
with no guard.
**Fix:** restore the upstream helper in this override and call `showModal()` from all three
attributes, adding the null guard the duplication currently hides:

```javascript
function showModal() {
  const el = document.getElementById('cal-modal');
  if (el) { bootstrap.Modal.getOrCreateInstance(el).show(); }
}
```

---

_Reviewed: 2026-09-09_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
_Supersedes: 33-REVIEW.md of 2026-09-08 (CR-04, WR-08..WR-13, IN-06..IN-11)_
