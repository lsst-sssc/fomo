---
phase: 40
review: 40-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: open
    title: "R6 does not follow shutil into a function parameter, a lambda, a `match` capture or a class-pattern attribute, and the stated boundary does not name these routes"
  - id: CR-02
    severity: critical
    disposition: open
    title: "\"any `write_bytes`\" is not what R6 checks, and a class-pattern attribute writes the scratch bytes over the developer database unreported"
  - id: WR-01
    severity: warning
    disposition: open
    title: "`is_django_backed` reads the raw source, not the Python view, so Django code under a harmless cell magic can switch off R2-R4 and R8-R11"
  - id: WR-02
    severity: warning
    disposition: open
    title: "R0 does not see IPython's autocall of an `IPyAutocall` object, so \"Python that IPython runs unchanged\" overstates the check"
  - id: IN-01
    severity: info
    disposition: open
    title: "The README and the docstring leave out checks the code makes"
  - id: IN-02
    severity: info
    disposition: open
    title: "The automagic name set is pinned against a plain `InteractiveShell`, not the ipykernel shell that nbconvert runs"
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
open: 9
total: 12
recorded: 2026-10-11T01:44:32.408Z
---

# Phase 40: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | open | - |
| CR-02 | critical | open | - |
| WR-01 | warning | open | - |
| WR-02 | warning | open | - |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | out of scope for the gap round (developer decision 2026-10-09); for Phase 41 triage (not in the current review) |
| WR-03 | warning | open | - (not in the current review) |
| IN-04 | info | open | - (not in the current review) |
| IN-05 | info | fixed | fixed in 40-07 Task 1 (29490c3): the eight schedule files are created with dir=scratch_db_dir, so teardown removes them (not in the current review) |
| IN-06 | info | fixed | fixed in 40-05 Task 3 (819321e): teardown refuses a directory not directly inside tempfile.gettempdir() or not the one setup created; two refusal tests (not in the current review) |
| IN-07 | info | deferred | deferred. Partly fixed in 40-05 Task 2 (7cfd78e): re.escape(tmp), a snapshot-mode subprocess test and a forced snapshot failure test. The remaining part, using DISMISSED_TABLE_PREFIX and CONFIRMED_TABLE_PREFIX directly instead of the _DISMISSED_TABLE_PREFIX and _CONFIRMED_TABLE_PREFIX aliases in campaign_views.py, is out of scope this round (developer decision 2026-10-09) and left for Phase 41 triage (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
