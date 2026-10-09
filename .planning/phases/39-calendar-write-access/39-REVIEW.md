---
phase: 39-calendar-write-access
reviewed: 2026-10-09T04:02:28Z
depth: deep
scope: incremental (gap-closure plan 39-06; commit cadb56c; diff base bc0e22c)
files_reviewed: 1
files_reviewed_list:
  - solsys_code/tests/test_calendar_template.py
findings:
  critical: 0
  warning: 0
  info: 1
  total: 1
status: issues_found
---

# Phase 39: Code Review Report (incremental, after gap-closure plan 39-06)

**Reviewed:** 2026-10-09T04:02:28Z
**Depth:** deep
**Files Reviewed:** 1
**Status:** issues_found (one Info item; no Critical or Warning)

## Summary

This round covers only gap-closure plan 39-06. That is commit `cadb56c`, which changes one file and
only its tests: `git diff bc0e22c HEAD -- solsys_code/tests/test_calendar_template.py`, 26 insertions
and 4 deletions. No file outside `.planning/` changed between `bc0e22c` and `HEAD` except this test
module. The template under test (`src/templates/tom_calendar/partials/event_form.html`) and the tag
(`solsys_code/templatetags/attribution_display_extras.py`) are byte-identical to the last round.

The change:

- adds a G-39-7 paragraph to the `EventModalAttributionHintTest` class docstring;
- adds a `(self.plain_user, 'update', False)` row to `test_hint_is_gated_on_the_edit_form`, and moves
  `request.user = user` inside the loop;
- adds `test_signed_in_non_staff_does_not_see_hint`.

How the change was checked:

- **Traced the gate end to end.** `update_event` in tomtoolkit 3.1.0 (`tom_calendar/views.py:187-212`)
  renders with `action="update"` on GET, whether or not the request comes from htmx. `calendar_urls.py:28`
  wraps it in `read_open_write_requires_login`, so a signed-in non-staff user gets the form. The hint
  is `{% elif action == "update" and request.user.is_staff and not event.telescope_label_meta.run %}`
  (`event_form.html:284`). It is the `elif` of `{% if deco %}` (line 229), and `campaign_decoration`
  does not depend on the viewer.
- **Confirmed the class docstring's claim.** The `is_staff` conjunct really is the only gate.
  `high_band_attribution_candidates` (`attribution_display_extras.py:24-56`) does not check the user.
- **Checked the fixtures.** `plain_user` comes from `create_user` with defaults (`is_staff=False`,
  `is_superuser=False`, line 847). The plain-user row and the HTTP test use
  `unlinked_event_with_candidate`, the same event on which the staff row and
  `test_staff_sees_high_band_hint_for_unlinked_event` show the hint. With the same event, the same
  `action` and the same `deco`, the only difference between the staff and plain-user renders is the
  viewer's staff flag.
- **Checked the shared `RequestFactory` request.** The loop sets `request.user` before every render.
  The settings use the stock `debug`, `request`, `auth` and `messages` context processors (settings.py
  lines 66-71). None of them caches anything on the request across renders, so the rows cannot leak
  into each other. Each `subTest` names its user and action, so a failing row identifies itself.
- **Checked the anchor in the HTTP test.** `hx-post="{update_url}"` comes only from the authenticated
  `<form>` branch, in its not-`create` arm (`event_form.html:39-43`). So it rules out both the create
  form and the anonymous card, as the comment says. See IN-08 for what it does not prove.
- **Ran the class on HEAD.** `python manage.py test --noinput
  solsys_code.tests.test_calendar_template.EventModalAttributionHintTest` gives 13 tests, OK.
- **Ran two independent mutation checks.** I copied the template to a scratch `DIRS` entry, loaded it
  first through a scratch settings module, and left the tracked file untouched:
  - Replacing `request.user.is_staff` with `request.user.is_authenticated` (the WR-04 mutant) makes
    exactly the two new checks fail in the class: the `attrmodalplain`/`update` subTest and
    `test_signed_in_non_staff_does_not_see_hint`. Across the whole `test_calendar_template` module
    (104 tests) those are still the only 2 failures. The pinned-snapshot tests read the tracked file
    from disk, so the scratch-loader mutant does not reach them.
  - Dropping the conjunct entirely makes the same two checks fail, plus `test_anonymous_does_not_see_hint`.
  So the new checks pin the staff conjunct, and they are not vacuous.
- **Checked line lengths.** No line in the module is longer than 120 characters. The ruff claims in the
  summary match a module with no lint-shaped changes.
- **Checked CLAUDE.md conventions.** The change creates no `Target` fixture. It uses `django.test.TestCase`
  and the Django runner only.

**Previous findings in this round:**

| ID | Status after 39-06 |
|----|--------------------|
| WR-04 | **Resolved** (`cadb56c`). Both checks the previous report proposed are present. The HTTP test's anchor is stricter than the suggested `'<form' in content`, because it pins the update URL. The mutation run above confirms both checks fail when the gate is weakened. |
| WR-02, WR-03 | **Remain**, unchanged. 39-06 did not touch the snapshot test, its header or `pyproject.toml`. They are still recorded `open` in `39-REVIEW-DISPOSITION.md`, and they are not repeated or counted here. |
| IN-01 to IN-05, IN-07 | **Remain**, unchanged. Their files were not touched. They are still `open` in the ledger, and they are not counted here. |
| IN-06 | **Remains.** It lives in this module: the staff "Save and Edit" path of `create_event` is still never POSTed by any test. 39-06 was scoped to G-39-7 and did not claim it. It is still `open` in the ledger, and it is not counted here. |
| CR-01 | Accepted risk (AR-39-01 / T-39-22); not re-reported. |

One note on the planning artifacts, which are out of scope as source: the WR-04 row in
`39-REVIEW-DISPOSITION.md` says the two checks "fail, and only they fail" under the `is_authenticated`
mutant. That is true for the class and for the whole module, as long as the mutant is loaded through
a scratch template directory. With the tracked file edited in place, as in the previous round's
mutation, the two pinned-snapshot tests would also fail. The claim depends on the method, not on the
code.

## Narrative Findings (AI reviewer)

## Info

### IN-08: The new HTTP test has no positive control of its own; whether its "hint absent" result means anything depends on a sibling test sharing the same fixture

**File:** `solsys_code/tests/test_calendar_template.py:1010-1020` (anchor comment at line 1017)

**Issue:** `test_signed_in_non_staff_does_not_see_hint` makes two kinds of assertion:

- That the update form rendered: `hx-post="/calendar/update/<id>/"`.
- That the hint and its `?band=high` link are absent.

The anchor shows that the viewer is signed in and that the form is not the create form. Two things
are left unchecked:

- **That `action == "update"`.** The `hx-post` comes from the `{% else %}` arm of `action == "create"`
  (`event_form.html:40-43`), so any action other than `create` produces the same anchor.
- **That the event still has a HIGH-band candidate in this render.** If a future fixture or scorer
  change dropped `unlinked_event_with_candidate`'s candidate below HIGH, this test would stay green
  for the wrong reason.

Today the HTTP test is protected only because `test_staff_sees_high_band_hint_for_unlinked_event`
(lines 866-872) runs in the same class against the same event and would fail first. The
`RequestFactory` gate test does not have this weakness, because its staff/update row is a positive
control inside the same loop. The comment at line 1017 ("Proves the edit form rendered ... so the
absences below mean something") claims more than the anchor alone proves. This is a test-reliability
nit, not a defect in today's behaviour. The mutation runs show that the test catches the WR-04
regression as written.

**Fix:** Put the positive control in the same test, so the test proves non-vacuity by itself:

```python
def test_signed_in_non_staff_does_not_see_hint(self):
    event = self.unlinked_event_with_candidate
    for user, shown in ((self.staff_user, True), (self.plain_user, False)):
        with self.subTest(user=user.username):
            response = self._signed_in_client(user).get(self._modal_url(event))
            self.assertEqual(response.status_code, 200)
            body = response.content.decode()
            self.assertIn(f'hx-post="{self._modal_url(event)}"', body)
            self.assertEqual('Possible campaign run match' in body, shown)
            self.assertEqual(f'{reverse("campaigns:attribution")}?band=high' in body, shown)
```

Alternatively, keep the test as it is and reword the line 1017 comment to say that non-vacuity comes
from `test_staff_sees_high_band_hint_for_unlinked_event` on the same fixture.

---

_Reviewed: 2026-10-09T04:02:28Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
