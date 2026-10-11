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
| WR-02 | warning | fixed | 40-05 Task 2 (7cfd78e): `check_database_settings` refuses extra aliases and `DATABASE_ROUTERS`; confirmed by the re-review (not in the current review) |
| WR-03 | warning | open | - (not in the current review) |
| IN-04 | info | open | - (not in the current review) |
| IN-05 | info | fixed | fixed in 40-07 Task 1 (29490c3): the eight schedule files are created with dir=scratch_db_dir, so teardown removes them (not in the current review) |
| IN-06 | info | fixed | fixed in 40-05 Task 3 (819321e): teardown refuses a directory not directly inside tempfile.gettempdir() or not the one setup created; two refusal tests (not in the current review) |
| IN-07 | info | deferred | deferred. Partly fixed in 40-05 Task 2 (7cfd78e): re.escape(tmp), a snapshot-mode subprocess test and a forced snapshot failure test. The remaining part, using DISMISSED_TABLE_PREFIX and CONFIRMED_TABLE_PREFIX directly instead of the _DISMISSED_TABLE_PREFIX and _CONFIRMED_TABLE_PREFIX aliases in campaign_views.py, is out of scope this round (developer decision 2026-10-09) and left for Phase 41 triage (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
