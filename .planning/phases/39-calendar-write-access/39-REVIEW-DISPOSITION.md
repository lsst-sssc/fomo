---
phase: 39
review: 39-REVIEW.md
titles: json
findings:
  - id: IN-08
    severity: info
    disposition: open
    title: "The new HTTP test has no positive control of its own; whether its \"hint absent\" result means anything depends on a sibling test sharing the same fixture"
  - id: WR-02
    severity: warning
    disposition: open
    title: "The header says the snapshot \"fails until this list and that file are updated together\", but regenerating the snapshot alone turns the test green; nothing checks the header list (carried forward)"
  - id: WR-03
    severity: warning
    disposition: open
    title: "The snapshot is named and described as \"vs tomtoolkit 3.1.0\", but the test diffs against whatever tomtoolkit is installed, and CI installs `tomtoolkit>=3.1.0` unpinned (carried forward)"
  - id: WR-04
    severity: warning
    disposition: fixed
    title: "No behavioural test keeps the edit-form hint from a signed-in non-staff user; 39-05 rewrote that gate, and weakening it is caught only by the regenerable snapshot"
  - id: IN-01
    severity: info
    disposition: open
    title: "The docstrings credit the HX-Redirect to `Raise403Middleware`; it comes from `HTMXRedirectMiddleware` (carried forward)"
  - id: IN-02
    severity: info
    disposition: open
    title: "The runbook's CSRF paragraph still omits the misleading flash the CSRF path puts on the login page (carried forward)"
  - id: IN-03
    severity: info
    disposition: open
    title: "The runbook's \"passes the CSRF check\" example (a tab left open after logging out) is not pinned by any CSRF-enforcing test (carried forward)"
  - id: IN-04
    severity: info
    disposition: open
    title: "The CSRF-path tests build the expected Location from `settings.LOGIN_URL`, but the code under test uses `reverse('login')` (carried forward)"
  - id: IN-05
    severity: info
    disposition: open
    title: "`calendar_urls.py`'s module docstring still states the single refusal path (carried forward)"
  - id: IN-06
    severity: info
    disposition: open
    title: "The staff \"Save and Edit\" path -- the third way `create_event` renders `event_form.html` -- is untested"
  - id: IN-07
    severity: info
    disposition: open
    title: "The runbook's \"pop-up does not open\" troubleshooting does not cover the \"opens empty\" symptom G-39-4 actually produced"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "The CSRF path's post-login landing page is a live, script-less form whose Save submits a GET carrying the CSRF token; the new runbook text calls it \"only a form\""
  - id: CR-01
    severity: critical
    disposition: skipped
    title: "Open self-registration lets anyone through the login guard, so any internet user can still create, edit and delete any calendar event"
open: 10
total: 13
recorded: 2026-10-09T04:04:49.204Z
---

# Phase 39: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| IN-08 | info | open | - |
| WR-02 | warning | open | - (not in the current review) |
| WR-03 | warning | open | - (not in the current review) |
| WR-04 | warning | fixed | resolved by test in 39-06 Task 1 (cadb56c); test_signed_in_non_staff_does_not_see_hint GETs the edit pop-up as a signed-in non-staff user and asserts the hint and its band=high link are absent; test_hint_is_gated_on_the_edit_form gains a (plain_user, update, False) row; both fail, and only they fail, when the event_form.html gate is weakened to request.user.is_authenticated (scratch-template mutation run); test-only, template and snapshot unchanged; UAT Test 7 / G-39-7. (not in the current review) |
| IN-01 | info | open | - (not in the current review) |
| IN-02 | info | open | - (not in the current review) |
| IN-03 | info | open | - (not in the current review) |
| IN-04 | info | open | - (not in the current review) |
| IN-05 | info | open | - (not in the current review) |
| IN-06 | info | open | - (not in the current review) |
| IN-07 | info | open | - (not in the current review) |
| WR-01 | warning | fixed | resolved by documentation in 39-05 Task 2 (c71b7b0): the runbook now says the create and edit addresses show a bare, unstyled copy of the form, not to use it, and to go back to the calendar page; the residual token-in-URL risk is accepted as AR-39-02 (39-SECURITY.md). Round-3 review (bc0e22c) records it "Resolved by documentation". (not in the current review) |
| CR-01 | critical | skipped | accepted risk (won't fix), Tim Lister 2026-10-08: "Self-signup is wanted; collaborators should be able to join without an operator; calendar edits are visible, attributable and easily reverted." TOM_REGISTRATION_STRATEGY, D-01 and the guard unchanged; see 39-SECURITY.md AR-39-01 and T-39-22 (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
