---
phase: 40
review: 40-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: open
    title: "The automagic rule treats any source-level store of a name as a binding, so an alias cell after a store that binds nothing at runtime is not reported"
  - id: WR-01
    severity: warning
    disposition: open
    title: "R6 follows `shutil` only through `import shutil as X`, so a plain rebinding of the module hides `copytree` and the other copy functions"
  - id: WR-02
    severity: warning
    disposition: open
    title: "R5 and R7 read only `str` constants, so a bytes literal naming `fomo_db.sqlite3` or `FOMO_DATABASE_PATH` is not reported"
  - id: IN-01
    severity: info
    disposition: open
    title: "Carried forward from the third review's IN-01: the logged clean-up warning still overstates, and a non-dict alias stops the switch loop"
  - id: IN-02
    severity: info
    disposition: open
    title: "The guard depends on ipykernel and on IPython internals through the `jupyter` metapackage only"
  - id: CR-02
    severity: critical
    disposition: fixed
    title: "R0 misses IPython aliases, automagic and shell escapes wrapped in a line magic, so a bare `cp` cell can overwrite the developer database"
  - id: IN-03
    severity: info
    disposition: open
    title: "Two branches of the fail-closed path have no executed test"
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
open: 8
total: 12
recorded: 2026-10-10T20:18:35.422Z
---

# Phase 40: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | open | - |
| WR-01 | warning | open | - |
| WR-02 | warning | open | - |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| CR-02 | critical | fixed | fixed in 40-10 Task 3 (c6c6322): R0 now reports any cell that is not Python IPython runs unchanged once `%matplotlib inline` and a first-line `%%time` or `%%capture` are removed, and a one-line cell that starts with an IPython magic or alias the notebook does not define; the review's aliases, automagic, `%time !` and magic cells are reported on a copy of import_campaign_csv_demo (not in the current review) |
| IN-03 | info | open | out of scope for the gap round (developer decision 2026-10-09); for Phase 41 triage (not in the current review) |
| WR-03 | warning | open | - (not in the current review) |
| IN-04 | info | open | - (not in the current review) |
| IN-05 | info | fixed | fixed in 40-07 Task 1 (29490c3): the eight schedule files are created with dir=scratch_db_dir, so teardown removes them (not in the current review) |
| IN-06 | info | fixed | fixed in 40-05 Task 3 (819321e): teardown refuses a directory not directly inside tempfile.gettempdir() or not the one setup created; two refusal tests (not in the current review) |
| IN-07 | info | deferred | deferred. Partly fixed in 40-05 Task 2 (7cfd78e): re.escape(tmp), a snapshot-mode subprocess test and a forced snapshot failure test. The remaining part, using DISMISSED_TABLE_PREFIX and CONFIRMED_TABLE_PREFIX directly instead of the _DISMISSED_TABLE_PREFIX and _CONFIRMED_TABLE_PREFIX aliases in campaign_views.py, is out of scope this round (developer decision 2026-10-09) and left for Phase 41 triage (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
