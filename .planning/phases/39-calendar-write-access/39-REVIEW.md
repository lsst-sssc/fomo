---
phase: 39-calendar-write-access
reviewed: 2026-10-08T23:06:45Z
depth: deep
scope: incremental (gap-closure plan 39-05; commits 917a895, c71b7b0; diff base d9132fc)
files_reviewed: 5
files_reviewed_list:
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/templatetags/attribution_display_extras.py
  - solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff
  - solsys_code/tests/test_calendar_template.py
  - src/templates/tom_calendar/partials/event_form.html
findings:
  critical: 0
  warning: 3
  info: 7
  total: 10
status: issues_found
---

# Phase 39: Code Review Report (incremental, after gap-closure plan 39-05)

**Reviewed:** 2026-10-08T23:06:45Z
**Depth:** deep
**Files Reviewed:** 5
**Status:** issues_found

## Summary

This round covers gap-closure plan 39-05 since the previous review at d9132fc:

- 917a895: the `isinstance` guard in `high_band_attribution_candidates`, the action-first `elif` on the
  staff hint in `event_form.html`, header item 4, the regenerated snapshot and six new tests.
- c71b7b0: the runbook's bare-form warning (G-39-3).

How the changes were checked:

- I read each hunk against its whole file. I traced the callers into `campaign_attribution.py`,
  `campaign_views.py`, the installed tomtoolkit 3.1.0 `tom_calendar/views.py` and the sibling tags in
  `calendar_display_extras.py`.
- I ran the new tests on HEAD: 17/17 pass.
- I ran the same tests against a scratch export with `event_form.html` and
  `attribution_display_extras.py` reverted to d9132fc: 15 of 17 fail, including every new
  staff/superuser/invalid-POST/tag/gate test. So the new tests catch G-39-4 and are not tautological.
- I ran a mutation check: swapping `request.user.is_staff` for `request.user.is_authenticated` in the
  new `elif`. Across `test_calendar_template` and `test_calendar_write_access` (133 tests), only the two
  snapshot tests fail. See WR-04.
- Pinned ruff (pre-commit `ruff` and `ruff-format`) is clean on both Python files.

**What checks out:**

- **The guard and the other callers.** The `isinstance` guard is the same guard used by
  `campaign_decoration`, `run_tally`, `unused_night_decoration` and `observation_series_decoration`
  (`calendar_display_extras.py:544, 633, 702, 824`). No other caller hands a non-event to
  `campaign_attribution.candidates_for_event`:
  - `event_attribution_backlog` (`campaign_attribution.py:787`) and `unattributable_orphan_count`
    (`:871`) iterate an `orphan_calendar_events()` queryset.
  - `is_offered_candidate` (`:902`) and `campaign_views._is_sole_high_candidate` (`:1331`) pass the
    result of `CalendarEvent.objects.get(...)` behind a `DoesNotExist` catch. Their pk comes through
    `_as_pk_or_none` (`campaign_views.py:1373, 1469`), so a non-integer can't reach `.get()`.
  - The template tag is the only template caller.
- **The `elif` gate.** Django's smartif `and` short-circuits, so on the create form (`action ==
  "create"`, no `event` in context) the hint branch is never evaluated. Staff and plain users now get
  identical create forms, apart from the CSRF token. The test proves this by comparing the full
  response bodies.
- **No staff-only leak.** Nothing staff-only reaches a non-staff viewer: `is_staff` is still a
  conjunct, and the create form renders no hint for anyone. The anonymous `{% else %}` card is
  unreachable on create, because `write_requires_login(create_event)` redirects first.
- **The third create-form render path.** `create_event`'s `save_and_edit` branch renders with
  `action="update"` and a real, saved event, so the hint path stays safe there too. It is untested
  (IN-06).
- **The snapshot.** It matches the current body (still 10 regions), and header item 4 now says "edit
  form only". The header, the anchors and the snapshot agree.
- **The runbook's new sentence.** It is accurate. The bare fragment loads no htmx, and its `<form>`
  has no `method`/`action`, so Save and "Save and Edit" do a native GET to the same address:
  - `create_event` GET finds no `date` key and renders an empty `EventForm()`.
  - `update_event` GET re-renders the stored instance.

  Either way nothing is saved and what was typed is discarded, as the runbook now says.

**Status of the previous round's findings:**

| ID | Status after 39-05 |
|----|--------------------|
| WR-01 | **Resolved by documentation** (c71b7b0, the developer's UAT decision). The runbook no longer calls the landing page "only a form", and tells the operator not to use it. Residual risk, recorded rather than re-raised because the `method="post"` hardening was offered at UAT and declined: if the operator clicks Save anyway, the CSRF token still goes into the query string (browser history, access logs), and the runbook gives "saves nothing" as its reason not to use that copy, but not the token exposure. |
| WR-02 | **Remains** (header lines 9-12 unchanged). 39-05 did update item 4 and the snapshot together, but only by discipline. The mutation run shows that the snapshot is now the only thing pinning the staff gate, and its own failure message prints the one-liner that turns it green. |
| WR-03 | **Remains** (installed 3.1.0, `pyproject.toml:20` still `>=3.1.0`). |
| IN-01, IN-03, IN-04, IN-05 | **Remain.** Their files (`calendar_access.py`, `test_calendar_write_access.py`, `calendar_urls.py`) were not touched by 39-05. |
| IN-02 | **Remains.** 39-05 rewrote this very paragraph, but it still does not mention the "contact your PI" flash. |
| CR-01 | Accepted risk (AR-39-01 / T-39-22); not re-reported. |

New in this round: WR-04 (no behavioural test keeps the edit-form hint from signed-in non-staff),
IN-06 (staff "Save and Edit" path untested) and IN-07 (the runbook's pop-up troubleshooting paragraph
does not cover the "opens empty" symptom that G-39-4 produced).

## Narrative Findings (AI reviewer)

## Warnings

### WR-02: The header says the snapshot "fails until this list and that file are updated together", but regenerating the snapshot alone turns the test green; nothing checks the header list (carried forward)

**File:** `src/templates/tom_calendar/partials/event_form.html:9-12`; `solsys_code/tests/test_calendar_template.py:2162-2177`

**Issue:** This is unchanged from the previous round.

- `test_body_diff_matches_pinned_snapshot` compares the body diff against a file that
  `T.SNAPSHOT.write_text(T.current_diff())` regenerates from the body alone.
- The failure message prints that exact command.
- `test_every_differing_region_is_listed_and_every_item_differs` cannot see a change inside an
  already-anchored region.

So a header that was not updated survives a snapshot regeneration with a green suite. 39-05 shows the
risk is live, not hypothetical: after its change, the snapshot is the only test that fails when the
staff gate in that anchored region is weakened (see WR-04).

**Fix:** As before, either reword the header to say what is enforced ("any new difference fails until
that file is regenerated; review the regenerated file against this list before committing it"), or
bind the header into the snapshot so the two cannot drift silently:

```python
@classmethod
def current_diff(cls) -> str:
    header_hash = hashlib.sha256(cls._header().encode()).hexdigest()
    return f'# header-sha256 {header_hash}\n' + normalized_upstream_diff(cls._upstream_lines(), cls._body_lines())
```

### WR-03: The snapshot is named and described as "vs tomtoolkit 3.1.0", but the test diffs against whatever tomtoolkit is installed, and CI installs `tomtoolkit>=3.1.0` unpinned (carried forward)

**File:** `solsys_code/tests/test_calendar_template.py:2055, 2080-2085, 2162-2177`; `solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff`; `pyproject.toml:20`

**Issue:** This is unchanged from the previous round. `_upstream_lines()` reads the installed
`tom_calendar` partial. `test_header_names_the_pinned_upstream` only checks that the header text says
`'tomtoolkit 3.1.0'`. Any upstream release that touches this partial has three effects:

- CI turns red with no FOMO change.
- The failure message blames FOMO's body.
- The advice it prints regenerates a file still named `..._3_1_0.diff`, under a header that still
  says 3.1.0.

**Fix:** Assert the installed version first, with an upgrade-specific message, before the snapshot
comparison:

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

Or pin `tomtoolkit<3.2` in `pyproject.toml`.

### WR-04: No behavioural test keeps the edit-form hint from a signed-in non-staff user; 39-05 rewrote that gate, and weakening it is caught only by the regenerable snapshot

**File:** `src/templates/tom_calendar/partials/event_form.html:284`; `solsys_code/tests/test_calendar_template.py:839-842, 889-895, 979-992`; `.planning/phases/39-calendar-write-access/39-05-SUMMARY.md` (coverage D5)

**Issue:** 39-05 rewrote the condition that keeps the "Possible campaign run match" hint staff-only:

```
{% elif action == "update" and request.user.is_staff and not event.telescope_label_meta.run %}
```

The template comment states the security intent (lines 290-293): "an offered candidate run may not
yet be publicly visible, so this hint must never reach an anonymous or non-staff visitor". Since
39-01, every self-registered account (CR-01, accepted) is a signed-in non-staff user who can open the
edit pop-up.

The tests for that intent:

- `EventModalAttributionHintTest` covers anonymous (`test_anonymous_does_not_see_hint`), staff and
  superuser.
- 39-05 added `cls.plain_user` (line 842) but only uses it for the create-form equivalence test, where
  the action gate hides the hint whatever the staff test is.
- `test_hint_is_gated_on_the_edit_form` renders only as `staff_user`.
- No test in the repository GETs the update pop-up as a signed-in non-staff user and asserts that
  the hint is absent.

The 39-05 summary's coverage item D5 claims "non-staff and anonymous still never see it", and cites
only the staff-rendering gate test and the superuser test.

Verified by mutation: replace `request.user.is_staff` with `request.user.is_authenticated` on line 284
and run `test_calendar_template` plus `test_calendar_write_access` (133 tests). The only failures are
`test_body_diff_matches_pinned_snapshot` and
`test_snapshot_detects_an_unlisted_line_inside_an_anchored_region`. Running the regeneration one-liner
that the failure message prints (WR-02) makes the suite fully green. The template would then show a
not-yet-public candidate run's name and score to every signed-in user.

**Fix:** Add a behavioural test for the non-staff case, and a non-staff row to the gate test:

```python
def test_signed_in_non_staff_does_not_see_hint(self):
    """T-27-21: the hint is staff-only; any signed-in non-staff account (open sign-up) must not see it."""
    response = self._signed_in_client(self.plain_user).get(self._modal_url(self.unlinked_event_with_candidate))
    self.assertEqual(response.status_code, 200)
    content = response.content.decode()
    self.assertIn('<form', content)  # the edit form did render, so the absence below is meaningful
    self.assertNotIn('Possible campaign run match', content)
    self.assertNotIn(f'{reverse("campaigns:attribution")}?band=high', content)
```

In `test_hint_is_gated_on_the_edit_form`, iterate over
`(user, action, shown) in ((staff, 'update', True), (staff, 'create', False), (plain, 'update', False))`.

## Info

### IN-01: The docstrings credit the HX-Redirect to `Raise403Middleware`; it comes from `HTMXRedirectMiddleware` (carried forward)

**File:** `solsys_code/calendar_access.py:15-17`; `solsys_code/tests/test_calendar_write_access.py:277-278`

**Issue:** This is unchanged. `Raise403Middleware` only produces the 302.
`HTMXRedirectMiddleware`, which sits outside it, rewrites the 302 to `200` + `HX-Redirect`.

**Fix:** "... `Raise403Middleware` turns the 403 into a login redirect whose `next` is the refused
path, and `HTMXRedirectMiddleware` turns that redirect into an `HX-Redirect` for an htmx request."

### IN-02: The runbook's CSRF paragraph still omits the misleading flash the CSRF path puts on the login page (carried forward)

**File:** `docs/runbooks/telescope_runs_calendar.rst:2596-2607`

**Issue:** `Raise403Middleware` flashes "You do not have permission to access this page. Please login
as a user with the correct permissions or contact your PI." before redirecting. 39-05 rewrote the
ending of this exact paragraph and left the flash undocumented. So an operator who meets a stale form
token is still told to contact their PI.

**Fix:** Add: "The login page may say 'You do not have permission to access this page ... contact your
PI'; that message comes from the TOM Toolkit and here only means the form had expired."

### IN-03: The runbook's "passes the CSRF check" example (a tab left open after logging out) is not pinned by any CSRF-enforcing test (carried forward)

**File:** `solsys_code/tests/test_calendar_write_access.py:90-106`; `docs/runbooks/telescope_runs_calendar.rst:2597-2599`

**Issue:** This is unchanged. `AnonymousCalendarWriteTest` uses the default (CSRF-exempt) client, so
the claim that a stale-after-logout tab reaches the guard rests on two untested Django details:

- `logout()` does not rotate the token.
- `CSRF_USE_SESSIONS` is unset.

**Fix:** Add one end-to-end case:

1. Use `Client(enforce_csrf_checks=True)` and `force_login`.
2. GET the update pop-up and extract its token.
3. `logout()`, then POST with that token.
4. Assert `Location == '...?next=/calendar/'` and that no row changed.

### IN-04: The CSRF-path tests build the expected Location from `settings.LOGIN_URL`, but the code under test uses `reverse('login')` (carried forward)

**File:** `solsys_code/tests/test_calendar_write_access.py:291-293, 457`

**Issue:** This is unchanged. The two values agree only by coincidence (`/accounts/login/`).

**Fix:** `return f'{reverse("login")}?next={path}'` in `refused_login_url()`, and the same at line 457.

### IN-05: `calendar_urls.py`'s module docstring still states the single refusal path (carried forward)

**File:** `solsys_code/calendar_urls.py:7-10`

**Issue:** This is unchanged. The docstring still says "an anonymous caller is redirected to login with
next set to the calendar page", without the CSRF-path qualification.

**Fix:** At the next edit, add "(by the guard; a write that fails the CSRF check is refused earlier,
see `calendar_access.py`)".

### IN-06: The staff "Save and Edit" path -- the third way `create_event` renders `event_form.html` -- is untested

**File:** `solsys_code/tests/test_calendar_template.py:922-998`

**Issue:** Upstream `create_event` renders the partial in three ways:

- GET, with `action="create"`.
- An invalid POST, with `action="create"`.
- A valid POST carrying `save_and_edit`, with `action="update"` and the new event (`tom_calendar/views.py:154-163`).

The new tests cover the first two for staff. The third is the one path out of the New Event pop-up
where the hint branch is still evaluated for staff, and it is never exercised: no test in the
repository POSTs `save_and_edit`. Today it is safe, because the event is a saved `CalendarEvent`.
But G-39-4 reached users through exactly this kind of untested staff-only render path.

**Fix:** Add a staff htmx POST of a valid create form with `save_and_edit=1`. Assert:

- `200`
- `HX-Retarget == '#cal-modal-body'`
- `hx-post` points at the new event's update URL
- exactly one new row

### IN-07: The runbook's "pop-up does not open" troubleshooting does not cover the "opens empty" symptom G-39-4 actually produced

**File:** `docs/runbooks/telescope_runs_calendar.rst:2609-2620`

**Issue:** The paragraph names two faults:

- A pop-up that opens with no attribution block.
- A pop-up that "does not open at all ... on a day cell or the '+ New Event' button", which it calls
  a client-side JavaScript fault.

G-39-4 was a third case. The pop-up opened as an empty box, because the server answered 500 and htmx
did not swap the body. An operator who follows this paragraph would look in the browser console for a
JavaScript fault rather than in the server log for a 500. This is the operator runbook for the very
surface 39-05 fixed (CLAUDE.md paired-docs rule, scoped to `docs/runbooks/`).

**Fix:** Add a sentence: "A pop-up that opens but stays empty (no form, no card) usually means the
server answered with an error; check the server log for a 500 on `/calendar/create/` or
`/calendar/update/<id>/` before looking at the browser console."

---

_Reviewed: 2026-10-08T23:06:45Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
