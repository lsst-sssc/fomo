---
phase: 40
review: 40-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: fixed
    title: "R12 misses plain-source process, connection and ATTACH routes: module names are not matched as names, the `connection` proxy is not listed, and `ATTACH (` passes"
  - id: CR-02
    severity: critical
    disposition: fixed
    title: "R0 misses IPython aliases, automagic and shell escapes wrapped in a line magic, so a bare `cp` cell can overwrite the developer database"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "R6 lists only `copy`, `copy2` and `copyfile`, so `shutil.copytree` or `write_bytes` byte-copies over the developer database without naming it"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "three notebooks still publish the unmeasured \"developer database was never opened\" claim that D-G2 withdrew"
  - id: IN-01
    severity: info
    disposition: fixed
    title: "`_disable_django_databases` docstring overstates what the in-place switch does on its own"
  - id: IN-02
    severity: info
    disposition: fixed
    title: "Snapshot-notebook prose that does not match the committed output"
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
open: 3
total: 12
recorded: 2026-10-10T16:04:32.986Z
---

# Phase 40: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | fixed | fixed in 40-10 Tasks 1-2 (482606b, 3232341) and 40-11 Task 1 (7af64d0): R12 matches every listed name at every identifier, in every part of a string or bytes constant and in string code, lists the lookup functions, IPython's process helpers, the connection proxy and Database and reports ATTACH followed by a parenthesis; every review form is reported on a copy of import_campaign_csv_demo; the guard docstring and README say what it reports and name the residual |
| CR-02 | critical | fixed | fixed in 40-10 Task 3 (c6c6322): R0 now reports any cell that is not Python IPython runs unchanged once `%matplotlib inline` and a first-line `%%time` or `%%capture` are removed, and a one-line cell that starts with an IPython magic or alias the notebook does not define; the review's aliases, automagic, `%time !` and magic cells are reported on a copy of import_campaign_csv_demo |
| WR-01 | warning | fixed | fixed in 40-10 Task 2 (3232341): R6 also reports shutil.copytree, copyfileobj and move, any write_bytes and a copy or write function looked up by name; both review forms are reported on a copy of import_campaign_csv_demo |
| WR-02 | warning | fixed | fixed in 40-11 Task 2 (17168ff): the three teardown cells carry the teardown sentence the other five carry and three fresh-mode setup cells say the helper never opens the developer database in fresh mode; no notebook was re-executed |
| IN-01 | info | fixed | fixed in 40-11 Task 1 (7af64d0): the docstring says the in-place switch only makes new wrappers dummy and that dropping the cached wrappers, whether or not the close worked, is what removes a connection still open; docstring only |
| IN-02 | info | fixed | Fixed in 40-09 Task 3 (0020360): reconcile_campaign_runs_demo's three NF-14 labels describe what the cell counts instead of a hard-coded record, and project_observation_calendar_demo's requirement table cites its real section and tally cell (not in the current review) |
| IN-03 | info | open | out of scope for the gap round (developer decision 2026-10-09); for Phase 41 triage (not in the current review) |
| WR-03 | warning | open | - (not in the current review) |
| IN-04 | info | open | - (not in the current review) |
| IN-05 | info | fixed | fixed in 40-07 Task 1 (29490c3): the eight schedule files are created with dir=scratch_db_dir, so teardown removes them (not in the current review) |
| IN-06 | info | fixed | fixed in 40-05 Task 3 (819321e): teardown refuses a directory not directly inside tempfile.gettempdir() or not the one setup created; two refusal tests (not in the current review) |
| IN-07 | info | deferred | deferred. Partly fixed in 40-05 Task 2 (7cfd78e): re.escape(tmp), a snapshot-mode subprocess test and a forced snapshot failure test. The remaining part, using DISMISSED_TABLE_PREFIX and CONFIRMED_TABLE_PREFIX directly instead of the _DISMISSED_TABLE_PREFIX and _CONFIRMED_TABLE_PREFIX aliases in campaign_views.py, is out of scope this round (developer decision 2026-10-09) and left for Phase 41 triage (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.

## ID reuse across rounds

The 2026-10-10 re-review (commit 269d8c5, after gap plans 40-05..40-07) reused the ids WR-01, WR-02 and IN-01..IN-03 for
new findings, so the rows above under those ids are the NEW findings. The round-1 findings (review commit e129b4f) that
held those ids, and what happened to them:

| Round-1 id | Round-1 finding | Disposition | Source |
|------------|-----------------|-------------|--------|
| WR-01 | The static guard does not catch a notebook that re-routes the database after setup | fixed | 40-06 (50ffb59, 26940ce): rule R12; the re-review confirms both verifier probes are caught, and its new WR-01 reports two further re-route forms R12 misses |
| WR-02 | The resolved-path check covers only the `default` alias | fixed | 40-05 Task 2 (7cfd78e): `check_database_settings` refuses extra aliases and `DATABASE_ROUTERS`; confirmed by the re-review |
| IN-01 | `FOMO_DATABASE_PATH` / `DJANGO_ALLOW_ASYNC_UNSAFE` stay set after teardown or a failed setup | fixed | 40-05 Tasks 1 and 3 (222a25f, 819321e); confirmed by the re-review |
| IN-02 | Committed notebook output publishes the developer's absolute home path | fixed | 40-05 Task 2 (7cfd78e) and 40-07 (29490c3, 8beeb5c, 304022a); the remaining spiceypy stderr line in campaign_lifecycle_demo is third-party output, listed in 40-07-SUMMARY |
| IN-03 | Attribution worklists have no pager, and their page keys are outside the return state | open | out of scope for the gap round (developer decision 2026-10-09); for Phase 41 triage |

## ID reuse across rounds (round 3, 2026-10-10)

The third review (commit 93bb514's 40-REVIEW.md, after gap plans 40-08..40-09) reused four ids. The rows above now
name the new findings; the decisions recorded for the old findings under those ids were:

| Id | Earlier finding (review commit) | Decision |
|----|----------------------------------|----------|
| CR-01 | The helper refuses a non-scratch database but leaves Django set up against it, so the next cell writes there (e129b4f) | fixed in 40-05 Task 1 (222a25f) |
| WR-01 | R12 misses a database re-route through a new `ConnectionHandler` wrapper and through a child process with a cleared environment (269d8c5) | fixed in 40-08 Tasks 1-2 (301bd93, f1b4bb6) |
| WR-02 | The fail-closed step is not exception-safe; with no `default` alias it masks the helper's `RuntimeError` (269d8c5) | fixed in 40-08 Task 3 (c45edbc) |
| IN-01 | README and all eight notebooks say a failed setup leaves no reachable database and cannot be re-run (269d8c5) | fixed in 40-08 Task 3 (c45edbc) and 40-09 (25eae59, c68bf7b, 0020360) |
