---
phase: 40
review: 40-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: open
    title: "IPython's shell is reachable as `InteractiveShell.instance()`, so a notebook can define a shell alias that a later one-line cell runs, with no name R0 or R12 reports"
  - id: CR-02
    severity: critical
    disposition: open
    title: "R3 and R4 do not read code inside strings, so Django set up in a string before the helper call writes the developer database unreported, against the README's explicit claim"
  - id: WR-01
    severity: warning
    disposition: open
    title: "\"a cell has at most one write_bytes problem\" is false"
  - id: IN-01
    severity: info
    disposition: open
    title: "`from tempfile import _shutil as sh` then `sh.move(a, b)` reports `_shutil` but not the `move`, though the `_byte_copy_problems` docstring says a name bound to `_shutil` by an import counts"
  - id: IN-02
    severity: info
    disposition: open
    title: "`exit -1` does not end the kernel, though the docstring and README say the kernel's autocall objects \"end the kernel\" and give `exit -1` as the example"
  - id: IN-03
    severity: info
    disposition: open
    title: "The README and docstring still leave out checks the code makes (round-5 IN-01, unchanged)"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "R0 does not see IPython's autocall of an `IPyAutocall` object, so \"Python that IPython runs unchanged\" overstates the check"
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
recorded: 2026-10-11T03:50:27.714Z
---

# Phase 40: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | open | - |
| CR-02 | critical | open | - |
| WR-01 | warning | open | - |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| WR-02 | warning | fixed | fixed in 40-15 Task 2 (1c830b1) and 40-16 Task 3 (b04f09f): IPyAutocall is an R12 name, and the docstring, README and override reason name IPython's autocall of exit and quit as the rewrite R0 does not see, pinned by a test (D-G13) (not in the current review) |
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

## ID reuse across rounds (round 4, 2026-10-10)

The round-4 review (40-REVIEW.md, reviewed 2026-10-10T19:30:37Z, commit 42dc508) reuses CR-01, WR-01, WR-02, IN-01 and IN-02 for new findings, so the regenerated ledger above no longer carries the third review's decisions for those IDs. They are kept here with the third review's titles (`git show 93aedf79:.planning/phases/40-notebook-isolation-attribution-page/40-REVIEW.md`):

| ID (round 3) | Severity | Title (third review) | Disposition | Decision |
|---|---|---|---|---|
| CR-01 | critical | R12 misses plain-source process, connection and ATTACH routes: module names are not matched as names, the `connection` proxy is not listed, and `ATTACH (` passes | fixed | fixed in 40-10 Tasks 1-2 (482606b, 3232341) and 40-11 Task 1 (7af64d0): R12 matches every listed name at every identifier, in every part of a string or bytes constant and in string code, lists the lookup functions, IPython's process helpers, the connection proxy and Database and reports ATTACH followed by a parenthesis; every review form is reported on a copy of import_campaign_csv_demo; the guard docstring and README say what it reports and name the residual |
| WR-01 | warning | R6 lists only `copy`, `copy2` and `copyfile`, so `shutil.copytree` or `write_bytes` byte-copies over the developer database without naming it | fixed | fixed in 40-10 Task 2 (3232341): R6 also reports shutil.copytree, copyfileobj and move, any write_bytes and a copy or write function looked up by name; both review forms are reported on a copy of import_campaign_csv_demo |
| WR-02 | warning | three notebooks still publish the unmeasured "developer database was never opened" claim that D-G2 withdrew | fixed | fixed in 40-11 Task 2 (17168ff): the three teardown cells carry the teardown sentence the other five carry and three fresh-mode setup cells say the helper never opens the developer database in fresh mode; no notebook was re-executed |
| IN-01 | info | `_disable_django_databases` docstring overstates what the in-place switch does on its own | fixed | fixed in 40-11 Task 1 (7af64d0): the docstring says the in-place switch only makes new wrappers dummy and that dropping the cached wrappers, whether or not the close worked, is what removes a connection still open; docstring only |
| IN-02 | info | (see 93aedf79 REVIEW.md) | fixed | Fixed in 40-09 Task 3 (0020360): reconcile_campaign_runs_demo's three NF-14 labels describe what the cell counts instead of a hard-coded record, and project_observation_calendar_demo's requirement table cites its real section and tally cell (not in the current review) |

## ID reuse across rounds (round 5, 2026-10-10)

The fifth review (commit da88285's 40-REVIEW.md, after gap plans 40-13..40-14) reused the ids CR-01, CR-02, WR-01, WR-02,
IN-01 and IN-02. The rows above now name the new findings; the decisions recorded for the old findings under those ids
(CR-01, WR-01, WR-02, IN-01, IN-02 from the fourth review, commit 42dc508; CR-02 from the third review) were:

| Id | Severity | Earlier finding | Disposition | Source |
|----|----------|-----------------|-------------|--------|
| CR-01 | critical | The automagic rule treats any source-level store of a name as a binding, so an alias cell after a store that binds nothing at runtime is not reported | fixed | fixed in 40-13 Task 1 (905408a) and 40-14 Task 3 (1a315e2): R0's automagic rule reports a one-line cell that starts with an IPython magic or alias name whatever an earlier cell binds (only a builtin, a keyword or a following = or comma is exempt); the four non-binding forms, a plain assignment, an import and an assignment then del are reported on a copy of import_campaign_csv_demo; the guard docstring, README and override reason say so |
| WR-01 | warning | R6 follows `shutil` only through `import shutil as X`, so a plain rebinding of the module hides `copytree` and the other copy functions | fixed | fixed in 40-13 Task 2 (4f02628) and 40-14 Task 3 (1a315e2): R6 reports copy2, copyfile, copytree and copyfileobj at every identifier and string part and follows shutil names across the notebook through imports and assignment-shaped bindings for copy and move; both review rebinding forms are reported on a copy of import_campaign_csv_demo; the docstring names what is not followed |
| WR-02 | warning | R5 and R7 read only `str` constants, so a bytes literal naming `fomo_db.sqlite3` or `FOMO_DATABASE_PATH` is not reported | fixed | fixed in 40-13 Task 2 (4f02628) and 40-14 Task 3 (1a315e2): R5 and R7 read str and bytes constants, so bytes literals naming the developer database file and its variable are reported; the docstring's R5 and R7 sentence says so |
| IN-01 | info | Carried forward from the third review's IN-01: the logged clean-up warning still overstates, and a non-dict alias stops the switch loop | open | out of scope for the gap round (developer decision 2026-10-10); recorded for Phase 41 in 40-11-SUMMARY |
| IN-02 | info | The guard depends on ipykernel and on IPython internals through the `jupyter` metapackage only | fixed | fixed in 40-13 Task 3 (ccfc578): the dev extra lists ipykernel>=7, which the guard imports; every CI workflow installs .[dev], so none changed |
| CR-02 | critical | R0 misses IPython aliases, automagic and shell escapes wrapped in a line magic, so a bare `cp` cell can overwrite the developer database | fixed | fixed in 40-10 Task 3 (c6c6322): R0 now reports any cell that is not Python IPython runs unchanged once `%matplotlib inline` and a first-line `%%time` or `%%capture` are removed, and a one-line cell that starts with an IPython magic or alias the notebook does not define; the review's aliases, automagic, `%time !` and magic cells are reported on a copy of import_campaign_csv_demo |

## ID reuse across rounds (round 6, 2026-10-11)

The sixth review (commit f1369ad's 40-REVIEW.md, after gap plans 40-15..40-16) reused the ids CR-01, CR-02, WR-01, IN-01
and IN-02. The rows above now name the new findings; the decisions recorded for the fifth review's findings under those ids
(commit da88285) were:

| Id | Severity | Earlier finding | Disposition | Source |
|----|----------|-----------------|-------------|--------|
| CR-01 | critical | R6 does not follow shutil into a function parameter, a lambda, a `match` capture or a class-pattern attribute, and the stated boundary does not name these routes | fixed | fixed in 40-15 Task 1 (bc99b0e) and 40-16 Task 3 (b04f09f): R6 reports shutil and tempfile's _shutil at every identifier and string part, so all six review forms (a call argument, a lambda, map, a keyword-only parameter, a match capture, a class pattern) are reported on a copy of import_campaign_csv_demo; the guard docstring, README and override reason state R6's post-fix boundary |
| CR-02 | critical | "any `write_bytes`" is not what R6 checks, and a class-pattern attribute writes the scratch bytes over the developer database unreported | fixed | fixed in 40-15 Task 1 (bc99b0e) and 40-16 Task 3 (b04f09f): R6 matches write_bytes at every identifier and string part; the class-pattern form and a write_bytes parameter and keyword argument are reported on a copy of import_campaign_csv_demo; the docstring, README and reason no longer claim more |
| WR-01 | warning | `is_django_backed` reads the raw source, not the Python view, so Django code under a harmless cell magic can switch off R2-R4 and R8-R11 | fixed | fixed in 40-15 Task 2 (1c830b1: isolation_problems docstring and a pin test) and 40-16 Task 3 (b04f09f): the module docstring says R2-R4 and R8-R11 apply only when a cell that parses as written imports Django (D-13 unchanged, developer decision D-G12) |
| IN-01 | info | The README and the docstring leave out checks the code makes | open | out of scope for the gap round (developer decision 2026-10-10, D-G14); kept for Phase 41 triage |
| IN-02 | info | The automagic name set is pinned against a plain `InteractiveShell`, not the ipykernel shell that nbconvert runs | open | out of scope for the gap round (developer decision 2026-10-10, D-G14); kept for Phase 41 triage |
