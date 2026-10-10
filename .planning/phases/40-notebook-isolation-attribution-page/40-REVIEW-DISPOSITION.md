---
phase: 40
review: 40-REVIEW.md
titles: json
findings:
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "R12 misses a database re-route through a new `ConnectionHandler` wrapper and through a child process with a cleared environment; the teardown still prints \"never opened for writing\""
  - id: WR-02
    severity: warning
    disposition: fixed
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
open: 5
total: 11
recorded: 2026-10-10T05:45:34.727Z
---

# Phase 40: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-01 | warning | fixed | Fixed in 40-08 Task 1 and Task 2 (301bd93, f1b4bb6): rule R12 now reports whole capabilities in every notebook (Django connection objects and database settings, starting a process, the environment, code from a string or another file, raw SQL ATTACH) and R0 reports shell escapes, process magics and manage.py / env -i / env -u text; all four re-review forms are reported on a copy of import_campaign_csv_demo; the docstring names the residual, which 40-09 records as an override; the teardown line was reworded in 40-08 Task 3 (c45edbc) |
| WR-02 | warning | fixed | Fixed in 40-08 Task 3 (c45edbc): every alias is switched to the dummy backend before a best-effort close, a clean-up error is logged and the setup own error is re-raised; TestHelperFailClosedIsExceptionSafe proves RuntimeError with no default alias |
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
