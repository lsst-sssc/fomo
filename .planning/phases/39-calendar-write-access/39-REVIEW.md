---
phase: 39-calendar-write-access
reviewed: 2026-10-08T21:01:27Z
depth: deep
scope: incremental (gap-closure plan 39-04; commits bc73bfd, 65ba57c; diff base d0440c3)
files_reviewed: 6
files_reviewed_list:
  - solsys_code/calendar_access.py
  - solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff
  - solsys_code/tests/test_calendar_template.py
  - solsys_code/tests/test_calendar_write_access.py
  - src/templates/tom_calendar/partials/event_form.html
  - docs/runbooks/telescope_runs_calendar.rst
findings:
  critical: 0
  warning: 3
  info: 5
  total: 8
status: issues_found
---

# Phase 39: Code Review Report (incremental, after gap-closure plan 39-04)

**Reviewed:** 2026-10-08T21:01:27Z
**Depth:** deep
**Files Reviewed:** 6
**Status:** issues_found

## Summary

This review covers what gap-closure plan 39-04 changed since the first Phase 39 review at d0440c3:

- bc73bfd: the CSRF-failure refusal path. This adds `AnonymousCsrfFailureWriteTest`, the
  `CalendarRowSnapshotMixin` refactor, the `calendar_access.py` module docstring and the runbook
  paragraph.
- 65ba57c: restores the "Save and Edit" label, adds the pinned body-diff snapshot, and adds the
  `normalized_upstream_diff` helper and its two tests.

The hunks were judged against the whole of each file and cross-checked against the code they depend on:

- the guard in `solsys_code/calendar_access.py`
- the URL conf in `solsys_code/calendar_urls.py`
- tom_common's `Raise403Middleware` / `HTMXRedirectMiddleware` and their order in
  `TOMTOOLKIT_MIDDLEWARE`
- Django's `login()` / `logout()` CSRF rotation
- the installed tomtoolkit 3.1.0 `tom_calendar/partials/event_form.html`

What checks out:

- **Snapshot.** I recomputed the normalized diff statically from the installed upstream file and
  the current template. It matches the committed snapshot byte for byte (10 regions).
- **CSRF-failure tests.** The Locations they assert match what the middleware chain actually
  produces. `CsrfViewMiddleware.process_view` returns a 403; on the way out, `Raise403Middleware`
  turns it into a 302 to `reverse('login') + '?next=' + request.path`, and `HTMXRedirectMiddleware`
  then turns that into a 200 with `HX-Redirect`.
- **Runbook stale-tab claim.** "A tab left open after logging out passes the CSRF check" is
  accurate: Django's `login()` calls `rotate_token`, `logout()` does not, and `CSRF_USE_SESSIONS`
  is unset.
- **Label fix.** The restored label matches upstream's `bootstrap_button "Save and Edit"`.

No blocker was found: no path writes a row, and the guard code is unchanged. The remaining
problems are in the new safety nets and the new documentation:

- The documented post-login landing for the CSRF path is a script-less form fragment. Its Save
  button submits as a GET and puts a valid CSRF token in the URL.
- The snapshot guard's header claims more than the test enforces.
- The snapshot is pinned to 3.1.0 by name only, while CI installs whatever `tomtoolkit>=3.1.0`
  resolves to.

CR-01 (open self-registration) is a recorded accepted risk (39-SECURITY.md AR-39-01 / T-39-22) and
is not re-reported.

## Narrative Findings (AI reviewer)

## Warnings

### WR-01: The CSRF path's post-login landing page is a live, script-less form whose Save submits a GET carrying the CSRF token; the new runbook text calls it "only a form"

**File:** `docs/runbooks/telescope_runs_calendar.rst:2596-2605`; `solsys_code/calendar_access.py:15-19`; `solsys_code/tests/test_calendar_write_access.py:349-370`; `src/templates/tom_calendar/partials/event_form.html:40-45`

**Issue:**

What the new text says:

- The runbook paragraph ends: "after logging in the browser simply opens that address, which
  changes nothing -- ... the create and edit addresses only show a form."
- `test_replaying_the_refused_path_as_a_signed_in_get_changes_nothing` pins those two GETs as
  `200`.

What the browser actually gets:

- `create_event` / `update_event` on GET render `tom_calendar/partials/event_form.html` on its own,
  with no base page. So the full-page navigation after login (plain, or the `HX-Redirect` case)
  shows a bare fragment with no htmx, no Bootstrap and no layout.
- The form in that fragment is `<form hx-post=... hx-target="#calendar-partial">`, with no `method`
  and no `action`, and it contains `{% csrf_token %}`.
- With htmx not loaded, clicking **Save** (or **Save and Edit**) does a native submit. That is a GET
  to the same address, carrying every field plus `csrfmiddlewaretoken=<valid token>` in the query
  string.

The consequences:

- The upstream GET branch ignores those parameters and re-renders an empty or unchanged form, so
  whatever the operator typed is silently thrown away. "Changes nothing" is true, but the operator
  gets no sign that the save didn't happen.
- The session's CSRF token, freshly rotated by the login that just happened, ends up in browser
  history and server access logs. A masked Django token unmasks to the cookie secret, so anyone who
  can read those logs can forge cross-site POSTs for that session until the next login. The
  default `SECURE_REFERRER_POLICY='same-origin'` stops a cross-origin Referer leak, so the exposure
  is limited to history and logs.

The first review (WR-01 item 2) noted the bare fragment. The developer chose to fix that with docs
and tests rather than a `CSRF_FAILURE_VIEW`. But the replacement text presents the landing page as
inert, and neither the docs nor the tests record that its Save button "works" as a token-leaking
GET.

**Fix:** This stays within the developer's no-behaviour-change decision for the redirect itself.

1. Correct the runbook sentence, for example: "the create and edit addresses show a bare, unstyled
   copy of the form; do not use it: go back to the calendar page and make the change there."
2. Optionally harden the FOMO-owned template so a script-less submit can never become a GET. This
   adds one line to the header's item 3 and to the pinned snapshot:

```html
<form method="post"
      {% if action == "create" %}
        action="{% url 'calendar:create-event' %}" hx-post="{% url 'calendar:create-event' %}"
      {% else %}
        action="{% url 'calendar:update-event' event.id %}" hx-post="{% url 'calendar:update-event' event.id %}"
      {% endif %}
        hx-target="#calendar-partial">
```

With `method="post"`, a script-less submit is an authenticated, token-valid POST through the guard.
The token never appears in the URL. A test such as
`self.assertNotIn('csrfmiddlewaretoken=', replay.request['QUERY_STRING'])` is not meaningful on a
GET, so instead assert that the rendered fragment's `<form` tag carries `method="post"`.

### WR-02: The header says the snapshot "fails until this list and that file are updated together", but regenerating the snapshot alone turns the test green; nothing checks the header list

**File:** `src/templates/tom_calendar/partials/event_form.html:9-12`; `solsys_code/tests/test_calendar_template.py:2058-2073`

**Issue:** 39-REVIEW WR-02 was about an unlisted change that sits inside a region an anchor already
covers. The anchor rule cannot see such a change. The snapshot catches it only until someone runs
the regeneration one-liner that the failure message itself prints:
`T.SNAPSHOT.write_text(T.current_diff())`.

That command rewrites the snapshot from the current body. Nothing ties the snapshot to the header's
numbered list:

- `test_every_differing_region_is_listed_and_every_item_differs` still passes, because the new line
  sits inside an anchored region.
- `test_body_diff_matches_pinned_snapshot` passes after the rewrite.

So the WR-02 class of omission (an unlisted line inside the merged card/decoration insert) can come
back with a green suite. The only defence left is a reviewer reading the `.diff` file in the PR.
The header comment, which is the operative documentation for this drift guard, states a guarantee
("fails until this list and that file are updated together") that the code does not provide. The
failure message's "update the header's numbered list, then regenerate" is advice, and nothing
enforces it.

**Fix:** Either reword the header to say what is actually enforced, or bind the header to the
snapshot so one cannot be regenerated without the other.

Reworded header:

```text
   ... and pins the full diff in solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff, so any
   new difference fails until that file is regenerated; review the regenerated file against this
   list before committing it.
```

Or the binding, for example: make the snapshot's first line a SHA-256 of `_header()` text and have
`current_diff()` emit it. Then a regenerated snapshot also changes visibly whenever the header
changes, and a header that did not change shows up as an unchanged hash next to a changed body
diff:

```python
@classmethod
def current_diff(cls) -> str:
    header_hash = hashlib.sha256(cls._header().encode()).hexdigest()
    return f'# header-sha256 {header_hash}\n' + normalized_upstream_diff(cls._upstream_lines(), cls._body_lines())
```

### WR-03: The snapshot is named and described as "vs tomtoolkit 3.1.0", but the test diffs against whatever tomtoolkit is installed, and CI installs `tomtoolkit>=3.1.0` unpinned

**File:** `solsys_code/tests/test_calendar_template.py:1951, 1976-1985, 2058-2073`; `solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff`; `pyproject.toml:20`

**Issue:** `_upstream_lines()` reads the installed `tom_calendar` partial. The only version check
in the class is `test_header_names_the_pinned_upstream`, and it checks that the header text
contains `'tomtoolkit 3.1.0'`, not that 3.1.0 is installed. `pyproject.toml` requires
`tomtoolkit>=3.1.0`, and every CI workflow (`testing-and-coverage.yml`, `pre-commit-ci.yml`, the
daily smoke test) runs `pip install -e .[dev]`.

The first tomtoolkit release that touches this partial, even whitespace, has these effects:

1. It turns CI red with no FOMO change.
2. The failure message blames FOMO ("event_form.html's body now differs from tomtoolkit 3.1.0 in a
   way the pinned snapshot does not record"), which is wrong on both counts.
3. The suggested fix regenerates the snapshot against the new upstream, but leaves it in a file
   named `..._3_1_0.diff` and under a header that still says "compared with tomtoolkit 3.1.0".
   After that, the filename and header describe a comparison the file no longer records.

The T-27-20 tripwire ("re-diff on every tomtoolkit upgrade") is wanted. The defect is that this
test detects an upgrade only indirectly, and then gives the wrong diagnosis and remedy.

**Fix:** Check the installed version first and fail with an upgrade-specific message. Then a drift
in FOMO's body and an upstream upgrade are reported separately, and the regeneration hint only
appears for the former:

```python
from importlib.metadata import version

PINNED_TOMTOOLKIT = '3.1.0'

def test_body_diff_matches_pinned_snapshot(self):
    installed = version('tomtoolkit')
    self.assertEqual(
        installed, PINNED_TOMTOOLKIT,
        f'tomtoolkit {installed} is installed but event_form.html is pinned against {PINNED_TOMTOOLKIT}: '
        're-diff the override against the new upstream partial (T-27-20), update the header, '
        'and add a snapshot named for the new version.',
    )
    ...
```

Alternatively, pin `tomtoolkit==3.1.0` (or `<3.2`) in `pyproject.toml` so an upgrade is a
deliberate change.

## Info

### IN-01: The docstrings credit the HX-Redirect to `Raise403Middleware`; it comes from `HTMXRedirectMiddleware`

**File:** `solsys_code/calendar_access.py:15-17`; `solsys_code/tests/test_calendar_write_access.py:277-278`

**Issue:** Both say `Raise403Middleware` "turns the 403 into a login redirect ... (htmx:
`HX-Redirect`)". In fact:

- `Raise403Middleware` (`tom_common/middleware.py:167-185`) only produces the 302.
- `HTMXRedirectMiddleware` (`tom_common/middleware.py:100-117`) sits outside it in
  `TOMTOOLKIT_MIDDLEWARE`. It is what rewrites the 302 to `200` + `HX-Redirect`, for this path and
  for the guard's own redirect alike.

Anyone debugging a missing `HX-Redirect` would look in the wrong middleware.

**Fix:** "... `Raise403Middleware` turns the 403 into a login redirect whose `next` is the refused
path, and `HTMXRedirectMiddleware` turns that redirect into an `HX-Redirect` for an htmx request."

### IN-02: The new runbook paragraph omits the misleading flash the CSRF path puts on the login page

**File:** `docs/runbooks/telescope_runs_calendar.rst:2596-2605`

**Issue:** `Raise403Middleware` calls `messages.error(request, 'You do not have permission to access
this page. Please login as a user with the correct permissions or contact your PI.')` before it
redirects. So the operator who hits the CSRF path sees "contact your PI" on the login page for what
is really a stale form token. The first review recorded this (WR-01 item 2), but the corrected
paragraph, which is now the operator's description of this exact path, does not mention it.

**Fix:** Add a sentence such as: "The login page may show 'You do not have permission to access this
page ... contact your PI'; that message comes from the TOM Toolkit and here only means the form had
expired. Log in and make the change again from the calendar page."

### IN-03: The runbook's "passes the CSRF check" example (a tab left open after logging out) is not pinned by any CSRF-enforcing test

**File:** `solsys_code/tests/test_calendar_write_access.py:90-97`; `docs/runbooks/telescope_runs_calendar.rst:2597-2599`

**Issue:** `AnonymousCalendarWriteTest` uses the default test client, which skips CSRF entirely. So
the claim that a real stale-after-logout tab still reaches the guard rests on two Django details
that no test exercises: `logout()` does not call `rotate_token`, and `CSRF_USE_SESSIONS` is unset.
If either changes (for example `CSRF_USE_SESSIONS = True` in a `local_settings.py`), that tab moves
to the CSRF path, and the runbook's first branch becomes wrong with the suite still green.

**Fix:** Add one end-to-end case to `AnonymousCsrfFailureWriteTest`'s neighbour:

1. Use `Client(enforce_csrf_checks=True)`, `force_login`, then GET the update-event pop-up and
   extract the token from it.
2. `client.logout()`.
3. POST with that token.
4. Assert the Location is `?next=/calendar/` and that no row changed.

### IN-04: The CSRF-path tests build the expected Location from `settings.LOGIN_URL`, but the code under test uses `reverse('login')`

**File:** `solsys_code/tests/test_calendar_write_access.py:291-293, 457`

**Issue:** `Raise403Middleware` builds `reverse('login') + '?next=' + request.path`, while
`refused_login_url()` uses `settings.LOGIN_URL`. Today both are `/accounts/login/`
(`tom_common/urls.py:48`, `src/fomo/settings.py:123`), but only by coincidence. Change
`LOGIN_URL`, and these tests fail for a reason unrelated to calendar access, while still
describing the guard's own path (which does use `LOGIN_URL` via `redirect_to_login`) correctly.

**Fix:** `return f'{reverse("login")}?next={path}'` in `refused_login_url()`, and the same in
`SignedInCalendarWriteTest.test_post_without_csrf_token_is_refused`.

### IN-05: `calendar_urls.py`'s module docstring still states the single refusal path that 39-04 corrected everywhere else

**File:** `solsys_code/calendar_urls.py:7-10`

**Issue:** "an anonymous caller is redirected to login with next set to the calendar page" is the
same unqualified claim that 39-REVIEW WR-01 flagged and that 39-04 fixed in `calendar_access.py`
and the runbook. Plan 39-04 deliberately did not touch `calendar_urls.py`. That leaves a third copy
of the old wording at the first place a reader of the URL conf looks.

**Fix:** At the next edit to that file, add "(by the guard; a write that fails the CSRF check is
refused earlier, see `calendar_access.py`)".

---

_Reviewed: 2026-10-08T21:01:27Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
