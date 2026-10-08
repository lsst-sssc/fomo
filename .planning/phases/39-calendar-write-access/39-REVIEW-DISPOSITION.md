---
phase: 39
review: 39-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: skipped
    title: "Open self-registration lets anyone through the login guard, so any internet user can still create, edit and delete any calendar event"
  - id: WR-01
    severity: warning
    disposition: open
    title: "A request that fails the CSRF check is redirected to the refused URL, not the calendar page. The docs say the opposite, and no anonymous test covers it"
  - id: WR-02
    severity: warning
    disposition: open
    title: "The WARN-01 header still leaves out a difference from upstream, and EventFormHeaderMatchesUpstreamTest cannot detect omissions like it"
  - id: IN-01
    severity: info
    disposition: open
    title: "The card prints \"UTC\" after the active timezone's time instead of converting to UTC"
  - id: IN-02
    severity: info
    disposition: open
    title: "A whitespace-only todo returns a 500 for signed-in users through the guarded create-todo route"
  - id: IN-03
    severity: info
    disposition: open
    title: "Anonymous day cells still highlight on hover, suggesting they can be clicked"
  - id: IN-04
    severity: info
    disposition: open
    title: "The Bootstrap 5 rename in calendar.html is incomplete, and its test covers only `--white`"
open: 6
total: 7
recorded: 2026-10-08T16:39:18.859Z
---

# Phase 39: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | skipped | accepted risk (won't fix), Tim Lister 2026-10-08: "Self-signup is wanted; collaborators should be able to join without an operator; calendar edits are visible, attributable and easily reverted." TOM_REGISTRATION_STRATEGY, D-01 and the guard unchanged; see 39-SECURITY.md AR-39-01 and T-39-22 |
| WR-01 | warning | open | - |
| WR-02 | warning | open | - |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| IN-04 | info | open | - |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
