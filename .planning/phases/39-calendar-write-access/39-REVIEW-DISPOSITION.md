---
phase: 39
review: 39-REVIEW.md
titles: json
findings:
  - id: WR-01
    severity: warning
    disposition: open
    title: "The CSRF path's post-login landing page is a live, script-less form whose Save submits a GET carrying the CSRF token; the new runbook text calls it \"only a form\""
  - id: WR-02
    severity: warning
    disposition: open
    title: "The header says the snapshot \"fails until this list and that file are updated together\", but regenerating the snapshot alone turns the test green; nothing checks the header list"
  - id: WR-03
    severity: warning
    disposition: open
    title: "The snapshot is named and described as \"vs tomtoolkit 3.1.0\", but the test diffs against whatever tomtoolkit is installed, and CI installs `tomtoolkit>=3.1.0` unpinned"
  - id: IN-01
    severity: info
    disposition: open
    title: "The docstrings credit the HX-Redirect to `Raise403Middleware`; it comes from `HTMXRedirectMiddleware`"
  - id: IN-02
    severity: info
    disposition: open
    title: "The new runbook paragraph omits the misleading flash the CSRF path puts on the login page"
  - id: IN-03
    severity: info
    disposition: open
    title: "The runbook's \"passes the CSRF check\" example (a tab left open after logging out) is not pinned by any CSRF-enforcing test"
  - id: IN-04
    severity: info
    disposition: open
    title: "The CSRF-path tests build the expected Location from `settings.LOGIN_URL`, but the code under test uses `reverse('login')`"
  - id: IN-05
    severity: info
    disposition: open
    title: "`calendar_urls.py`'s module docstring still states the single refusal path that 39-04 corrected everywhere else"
  - id: CR-01
    severity: critical
    disposition: skipped
    title: "Open self-registration lets anyone through the login guard, so any internet user can still create, edit and delete any calendar event"
open: 8
total: 9
recorded: 2026-10-08T21:03:16.580Z
---

# Phase 39: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-01 | warning | open | - |
| WR-02 | warning | open | - |
| WR-03 | warning | open | - |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| IN-04 | info | open | - |
| IN-05 | info | open | - |
| CR-01 | critical | skipped | accepted risk (won't fix), Tim Lister 2026-10-08: "Self-signup is wanted; collaborators should be able to join without an operator; calendar edits are visible, attributable and easily reverted." TOM_REGISTRATION_STRATEGY, D-01 and the guard unchanged; see 39-SECURITY.md AR-39-01 and T-39-22 (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
