---
phase: 40
review: 40-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: open
    title: "The helper refuses a non-scratch database but leaves Django set up against it, so the next cell writes there"
  - id: WR-01
    severity: warning
    disposition: open
    title: "The static guard does not catch a notebook that re-routes the database after setup"
  - id: WR-02
    severity: warning
    disposition: open
    title: "The resolved-path check covers only the `default` alias, but the teardown says the developer database was never written"
  - id: WR-03
    severity: warning
    disposition: open
    title: "The functional marker test's check that the unmarked row has no bar can pass without testing anything"
  - id: IN-01
    severity: info
    disposition: open
    title: "`FOMO_DATABASE_PATH` (and `DJANGO_ALLOW_ASYNC_UNSAFE`) stay set after teardown or a failed setup"
  - id: IN-02
    severity: info
    disposition: open
    title: "Committed notebook output publishes the developer's absolute home path"
  - id: IN-03
    severity: info
    disposition: open
    title: "Attribution worklists have no pager, and their page keys are outside the return state"
  - id: IN-04
    severity: info
    disposition: open
    title: "Undo posts a page number computed under `per_page`, but the redirect drops `per_page`"
  - id: IN-05
    severity: info
    disposition: open
    title: "`load_telescope_runs_demo` still leaves eight temp files in `/tmp` per regeneration"
  - id: IN-06
    severity: info
    disposition: open
    title: "`teardown_scratch_django()` will delete any directory named `fomo-notebook-db-*`, wherever it is"
  - id: IN-07
    severity: info
    disposition: open
    title: "Small test and code tidy-ups"
open: 11
total: 11
recorded: 2026-10-10T00:37:03.073Z
---

# Phase 40: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | open | - |
| WR-01 | warning | open | - |
| WR-02 | warning | open | - |
| WR-03 | warning | open | - |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| IN-04 | info | open | - |
| IN-05 | info | open | - |
| IN-06 | info | open | - |
| IN-07 | info | open | - |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
