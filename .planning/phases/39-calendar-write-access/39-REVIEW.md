---
phase: 39-calendar-write-access
reviewed: 2026-10-08T16:37:40Z
depth: deep
files_reviewed: 8
files_reviewed_list:
  - solsys_code/calendar_access.py
  - solsys_code/calendar_urls.py
  - solsys_code/tests/test_calendar_write_access.py
  - solsys_code/tests/test_calendar_template.py
  - solsys_code/tests/test_bootstrap5_rendering.py
  - src/templates/tom_calendar/partials/event_form.html
  - src/templates/tom_calendar/partials/calendar.html
  - docs/runbooks/telescope_runs_calendar.rst
findings:
  critical: 1
  warning: 2
  info: 4
  total: 7
status: issues_found
---

# Phase 39: Code Review Report

**Reviewed:** 2026-10-08T16:37:40Z
**Depth:** deep
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Reviewed the two calendar guards (`calendar_access.py`), the guarded URL conf, the two template
overrides, the runbook section and the three test modules. I traced them against the installed
tomtoolkit 3.1.0 `tom_calendar.views`/`urls.py`, `tom_common.middleware` (CSRF ->
`Raise403Middleware` -> `HTMXRedirectMiddleware` order), `tom_common.accounts.adapters` and FOMO's
effective settings, including `local_settings.py`.

Mechanically the guards work. Every HTTP method on all five write routes is covered. FOMO's
`calendar/` include completely shadows tom_common's unguarded copy: the same six patterns, matched
first. `tom_calendar` has no other write path, such as a DRF API. The `next` value is a fixed
`reverse()` result, so there is no open redirect. With `require_POST` inside the login guard, a
signed-in user's stray GET can no longer delete an event or blank a todo. The anonymous card
autoescapes everything, including `linebreaksbr`. It sends no CSRF token, form widget or ALLOC:/RUN:
key. Its run and series blocks stay behind their tag-level gates. The signed-in form is unchanged.

There is one blocker. Self-registration is open and accounts need no approval, so the "logged-in"
guard does not stop the public from writing. There are two warnings. A request that fails the CSRF
check takes a different redirect from the one the docs and tests describe. The WARN-01 header still
leaves out one difference, and its guard test cannot catch omissions like that one.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Open self-registration lets anyone through the login guard, so any internet user can still create, edit and delete any calendar event

**File:** `solsys_code/calendar_access.py:48-55, 70-77` (guard condition `request.user.is_authenticated`); `src/fomo/settings.py:280`; `docs/runbooks/telescope_runs_calendar.rst:2576-2590`
**Issue:** The only test both guards apply is `request.user.is_authenticated` (D-01). FOMO's effective
settings, confirmed via `manage.py shell` including `local_settings.py`, are
`TOM_REGISTRATION_STRATEGY = 'open'` and `ACCOUNT_EMAIL_VERIFICATION = 'none'`. Under those settings,
`tom_common.accounts.adapters.TomAccountAdapter.is_open_for_signup()` returns True.
`save_user()` creates the account active, since only `'approval_required'` sets `is_active=False`.
allauth then logs the new user in straight away, with no email verification. So any member of the
public can open `/accounts/signup/`, pick a username and password, and pass both guards within a
minute. They can then delete or rewrite every projector-, reconciler- and allocation-owned
`CalendarEvent` and every todo (D-03 puts no tighter gate on delete). The phase goal is "the public
calendar is read-only to anyone not logged in". On this deployment the public can always log in, so
the read-only state is only cosmetic. None of 39-CONTEXT/RESEARCH/PLAN/SECURITY mentions
self-registration. T-39-* in 39-SECURITY.md is marked closed without considering it. The runbook
paragraph "Any logged-in user, staff or not, can create, edit and delete entries" states the policy
without saying that anyone can get a login.
**Fix:** This needs a decision from the user, because it reopens D-01. Choose one of these:
```python
# Option A (settings, no guard change): self-registered accounts need staff approval
TOM_REGISTRATION_STRATEGY = 'approval_required'   # or None to close signup entirely

# Option B (guard): require an explicit grant instead of bare authentication
from django.contrib.auth.decorators import permission_required
def _may_write(user) -> bool:
    return user.is_authenticated and (user.is_staff or user.has_perm('tom_calendar.change_calendarevent'))
# ...and test `_may_write(request.user)` in both wrappers instead of `is_authenticated`
```
Whichever option you choose, add a test that a freshly self-registered user (posted through
`account_signup`) cannot delete an event. State the dependency on the registration strategy in the
runbook paragraph and in `calendar_access.py`'s module docstring.

## Warnings

### WR-01: A request that fails the CSRF check is redirected to the refused URL, not the calendar page. The docs say the opposite, and no anonymous test covers it

**File:** `solsys_code/calendar_access.py:9-11, 26-35`; `docs/runbooks/telescope_runs_calendar.rst:2586-2589`; `solsys_code/tests/test_calendar_write_access.py:42-255`
**Issue:** `CsrfViewMiddleware.process_view` runs before the guard. A POST with a missing or stale
token never reaches `write_requires_login`. Its 403 is rewritten by
`tom_common.middleware.Raise403Middleware` into
`redirect(reverse('login') + '?next=' + request.path)`, and `HTMXRedirectMiddleware` turns that into
`HX-Redirect`. So for scripts, forged cross-site posts, and stale tabs whose CSRF cookie was rotated
by a later login, `next` is the refused write URL. Three things go wrong as a result:
1. The runbook says "after logging in, the browser returns to the calendar page, never to the
   refused write". The module docstring says "The login `next` is the calendar page, not the refused
   URL". Both claims are false for this path.
2. After logging in, the browser lands on `/calendar/delete/<id>/` (a 405), `/calendar/todo/...` (a
   405), or `/calendar/update/<id>/`. The last one is a bare `event_form.html` fragment with no base
   page, htmx or Bootstrap. The login page also shows the misleading flash "You do not have
   permission to access this page ... contact your PI".
3. Every test in `AnonymousCalendarWriteTest` uses the default test `Client`, which turns CSRF
   checks off. The exact-Location assertions (`?next=/calendar/`) therefore cover only the
   valid-token path. A real anonymous browser or script without a token takes the other path, and no
   test covers it. Only the signed-in CSRF test (line 332) records this behaviour, as a comment.

No data is written on any path: `require_POST` stops a replay on GET. So this is a correctness and
truthfulness defect, not a write hole.
**Fix:** At minimum, correct both claims, for example: "a refused write that passes the CSRF check
returns to the calendar page; one that fails it is sent to login with the refused path as `next`,
where a GET does nothing". Add an anonymous `Client(enforce_csrf_checks=True)` test per route. It
should assert no row changes and pin whichever Location you decide on. To make the behaviour match
the docs, point `CSRF_FAILURE_VIEW` at a small view. For an anonymous request under `/calendar/`,
it would return `redirect_to_login(reverse('calendar:calendar'))`. For everything else it would
defer to `django.views.csrf.csrf_failure`.

### WR-02: The WARN-01 header still leaves out a difference from upstream, and EventFormHeaderMatchesUpstreamTest cannot detect omissions like it

**File:** `src/templates/tom_calendar/partials/event_form.html:6-8, 18-19, 104`; `solsys_code/tests/test_calendar_template.py` (`EventFormHeaderMatchesUpstreamTest.ANCHORS` / `test_every_differing_region_is_listed_and_every_item_differs`)
**Issue:** The header says "Every block that differs from that upstream file is listed below;
everything else is byte-for-byte upstream". Item 3 describes the button change as markup only. The
visible label also changed: upstream renders `"Save and Edit"`, FOMO renders `Save and edit`
(line 104). The header does not mention this, so a maintainer re-diffing on upgrade gets an
unexplained difference — the exact problem WARN-01 exists to prevent. The guard test does not catch
it, for these reasons:
- It accepts any differing region that contains any anchor from any item. `difflib` merges the
  read-only card, the series/campaign/high-band blocks and their comments into one inserted region
  (upstream line 74 -> FOMO lines 113-309). Any future unlisted line inside that region passes,
  because the region already contains `cal-event-card` and `campaign_decoration`.
- Item 3's only anchor is `<button`, so the label text in the same region is never checked.
**Fix:** Either restore upstream's `Save and Edit` label or add it to item 3, for example: "...and
the 'Save and Edit' label is lower-cased to 'Save and edit'". To harden the test, map each differing
region to a single item and require that region's non-comment lines to be fully accounted for. A
simpler option is to pin a normalized snapshot of the diff hunks (opcodes plus changed lines), so any
new difference fails the test until the header and the snapshot are updated together.

## Info

### IN-01: The card prints "UTC" after the active timezone's time instead of converting to UTC

**File:** `src/templates/tom_calendar/partials/event_form.html:119, 121`
**Issue:** `{{ event.start_time|date:'Y-m-d H:i' }} UTC` formats in the current timezone. That is
correct today only because `TIME_ZONE = 'UTC'` and nothing calls `timezone.activate()`. If
`local_settings.py` or a future middleware changes the timezone, the card would show local times
labelled "UTC".
**Fix:** `{% load tz %}` and use `{{ event.start_time|utc|date:'Y-m-d H:i' }} UTC` (same for `end_time`).

### IN-02: A whitespace-only todo returns a 500 for signed-in users through the guarded create-todo route

**File:** `solsys_code/calendar_urls.py:30` (wraps upstream `tom_calendar.views.create_todo`)
**Issue:** Upstream `create_todo` returns `None` when `description.strip()` is empty, and Django
raises `ValueError` ("didn't return an HttpResponse"). The todo input's `required` attribute does
not block a value made of spaces. This bug predates this phase and the upstream view must not be
edited. But FOMO now owns the wrapper layer for this route and could turn the 500 into a no-op.
**Fix:** Add a small FOMO wrapper (or a guard extension) that returns the re-rendered
`todos.html` for an empty, whitespace-only description instead of calling upstream.

### IN-03: Anonymous day cells still highlight on hover, suggesting they can be clicked

**File:** `src/templates/tom_calendar/partials/calendar.html:15-20, 236-242`
**Issue:** D-07 removes the create handler for visitors, but `.cal-day:hover` and
`.cal-day.is-current-month:hover` still change the background. The runbook says "a day cell does
nothing when clicked", yet the cell still looks clickable.
**Fix:** Add a class such as `cal-day-creatable` only inside the `{% if request.user.is_authenticated %}`
branch, and scope the hover rules to `.cal-day-creatable:hover`.

### IN-04: The Bootstrap 5 rename in calendar.html is incomplete, and its test covers only `--white`

**File:** `src/templates/tom_calendar/partials/calendar.html:8-9, 33-34`; `solsys_code/tests/test_calendar_template.py` (`CalendarTemplateBootstrap5ClassTest`)
**Issue:** Line 34 now uses `var(--bs-white)`. Lines 8, 9 and 33 still use `--light`,
`--secondary` and `--primary`. These resolve only because tom_common's legacy `main.css`/`dark.css`
define them; Bootstrap 5 itself defines `--bs-*`. The CSS works today, but D-11's "Bootstrap 5
names" goal is only half applied, and the test only bans `var(--white)`.
**Fix:** Switch to `var(--bs-light)`, `var(--bs-secondary)` and `var(--bs-primary)`, or note in a
comment that these come from tom_common's own stylesheet. Extend the test to cover whichever
convention you choose.

---

_Reviewed: 2026-10-08T16:37:40Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
