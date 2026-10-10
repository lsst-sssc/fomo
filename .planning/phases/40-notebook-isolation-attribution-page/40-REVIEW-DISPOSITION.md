---
phase: 40
review: 40-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: fixed
    title: "The helper refuses a non-scratch database but leaves Django set up against it, so the next cell writes there"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "The static guard does not catch a notebook that re-routes the database after setup"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "The resolved-path check covers only the `default` alias, but the teardown says the developer database was never written"
  - id: WR-03
    severity: warning
    disposition: open
    title: "The functional marker test's check that the unmarked row has no bar can pass without testing anything"
  - id: IN-01
    severity: info
    disposition: fixed
    title: "`FOMO_DATABASE_PATH` (and `DJANGO_ALLOW_ASYNC_UNSAFE`) stay set after teardown or a failed setup"
  - id: IN-02
    severity: info
    disposition: fixed
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
total: 11
recorded: 2026-10-10T00:37:03.073Z
---

# Phase 40: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | fixed | fixed in 40-05 Task 1 (222a25f): settings checked before django.setup(); any setup failure switches every database alias to the dummy backend, removes the scratch directory and restores the environment; TestHelperRefusalLeavesNoDatabase proves apps.ready False, a refused raw cursor and a refused second setup; notebooks re-executed in 40-07 |
| WR-01 | warning | fixed | fixed in 40-06 (50ffb59, 26940ce): guard rule R12 reports any re-routing name (settings_dict, DATABASES, connections.settings, override_settings, the helper internals, sqlite3 and others) in every notebook; both verifier probes replayed on a copy of import_campaign_csv_demo are reported; the docstring lists what the guard does not see |
| WR-02 | warning | fixed | fixed in 40-05 Task 2 (7cfd78e): check_database_settings refuses any alias other than default and any DATABASE_ROUTERS before django.setup() (developer decision 2026-10-09); the teardown line and rule R10 are unchanged |
| WR-03 | warning | open | - |
| IN-01 | info | fixed | fixed in 40-05 Task 1 and Task 3 (222a25f, 819321e): the failure path and teardown restore or remove FOMO_DATABASE_PATH and DJANGO_ALLOW_ASYNC_UNSAFE; test_teardown_restores_the_routing_environment |
| IN-02 | info | fixed | fixed in 40-05 Task 2 (7cfd78e) and 40-07 (29490c3, 8beeb5c, 304022a): the Django ready line no longer prints the repository path and all eight notebooks were re-executed; the one remaining home-directory path in committed output is a spiceypy library log line in campaign_lifecycle_demo (its ephemeris import cell), not helper output |
| IN-03 | info | open | - |
| IN-04 | info | open | - |
| IN-05 | info | fixed | fixed in 40-07 Task 1 (29490c3): the eight schedule files are created with dir=scratch_db_dir, so teardown removes them |
| IN-06 | info | fixed | fixed in 40-05 Task 3 (819321e): teardown refuses a directory not directly inside tempfile.gettempdir() or not the one setup created; two refusal tests |
| IN-07 | info | deferred | deferred. Partly fixed in 40-05 Task 2 (7cfd78e): re.escape(tmp), a snapshot-mode subprocess test and a forced snapshot failure test. The remaining part, using DISMISSED_TABLE_PREFIX and CONFIRMED_TABLE_PREFIX directly instead of the _DISMISSED_TABLE_PREFIX and _CONFIRMED_TABLE_PREFIX aliases in campaign_views.py, is out of scope this round (developer decision 2026-10-09) and left for Phase 41 triage |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
