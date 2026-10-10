---
phase: 40
review: 40-REVIEW.md
titles: json
findings:
  - id: WR-01
    severity: warning
    disposition: open
    title: "R12 misses a database re-route through a new `ConnectionHandler` wrapper and through a child process with a cleared environment; the teardown still prints \"never opened for writing\""
  - id: WR-02
    severity: warning
    disposition: open
    title: "The fail-closed step is not exception-safe; with no `default` alias it masks the helper's `RuntimeError` and never switches the aliases to the dummy backend"
  - id: IN-01
    severity: info
    disposition: open
    title: "README and all eight notebooks say a failed setup leaves no reachable database and cannot be re-run, which is not true for failures before the settings are read"
  - id: IN-02
    severity: info
    disposition: open
    title: "Snapshot-notebook prose that does not match the committed output"
  - id: IN-03
    severity: info
    disposition: open
    title: "Two branches of the fail-closed path have no executed test"
  - id: CR-01
    severity: critical
    disposition: fixed
    title: "The helper refuses a non-scratch database but leaves Django set up against it, so the next cell writes there"
  - id: WR-03
    severity: warning
    disposition: open
    title: "The functional marker test's check that the unmarked row has no bar can pass without testing anything"
  - id: IN-04
    severity: info
    disposition: open
    title: "Undo posts a page number computed under `per_page`, but the redirect drops `per_page`"
  - id: IN-05
    severity: info
    disposition: fixed
    title: "`load_telescope_runs_demo` still leaves eight temp files in `/tmp` per regeneration"
  - id: IN-06
    severity: info
    disposition: fixed
    title: "`teardown_scratch_django()` will delete any directory named `fomo-notebook-db-*`, wherever it is"
  - id: IN-07
    severity: info
    disposition: deferred
    title: "Small test and code tidy-ups"
open: 7
total: 11
recorded: 2026-10-10T05:45:34.727Z
---

# Phase 40: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-01 | warning | open | - |
| WR-02 | warning | open | - |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| CR-01 | critical | fixed | fixed in 40-05 Task 1 (222a25f): settings checked before django.setup(); any setup failure switches every database alias to the dummy backend, removes the scratch directory and restores the environment; TestHelperRefusalLeavesNoDatabase proves apps.ready False, a refused raw cursor and a refused second setup; notebooks re-executed in 40-07 (not in the current review) |
| WR-03 | warning | open | - (not in the current review) |
| IN-04 | info | open | - (not in the current review) |
| IN-05 | info | fixed | fixed in 40-07 Task 1 (29490c3): the eight schedule files are created with dir=scratch_db_dir, so teardown removes them (not in the current review) |
| IN-06 | info | fixed | fixed in 40-05 Task 3 (819321e): teardown refuses a directory not directly inside tempfile.gettempdir() or not the one setup created; two refusal tests (not in the current review) |
| IN-07 | info | deferred | deferred. Partly fixed in 40-05 Task 2 (7cfd78e): re.escape(tmp), a snapshot-mode subprocess test and a forced snapshot failure test. The remaining part, using DISMISSED_TABLE_PREFIX and CONFIRMED_TABLE_PREFIX directly instead of the _DISMISSED_TABLE_PREFIX and _CONFIRMED_TABLE_PREFIX aliases in campaign_views.py, is out of scope this round (developer decision 2026-10-09) and left for Phase 41 triage (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
