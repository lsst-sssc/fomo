---
phase: 40
review: 40-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: open
    title: "the helper call's own argument runs before the helper and is never read by R3, so Django can be set up on the developer database from string code inside `setup_scratch_django(...)`"
  - id: WR-01
    severity: warning
    disposition: open
    title: "\"no other call of the setup function\" (new in 40-18) and the README's \"once\" are false: an attribute call, a call through another name or a call in string code is not reported"
  - id: WR-02
    severity: warning
    disposition: open
    title: "\"R12 reports the IPython shell names a notebook would write to set those up\" is contradicted by the module's own test and by `exit._ip`"
  - id: WR-03
    severity: warning
    disposition: open
    title: "\"a comment that merely mentions a name cannot ... trip a check\" is false for any cell under a harmless magic, and 40-17's new names widen it"
  - id: WR-04
    severity: warning
    disposition: open
    title: "the README sends you to `/tmp` for a leftover snapshot (a full copy of the developer database), but the helper uses `$TMPDIR`"
  - id: IN-01
    severity: info
    disposition: open
    title: "the README says code inside strings is read, but R5/R7 do not read it, so an escaped `FOMO_DATABASE_PATH` in string code is not reported"
  - id: IN-02
    severity: info
    disposition: open
    title: "\"as the not-vacuous test requires\" no longer holds after the gate learned string code"
  - id: IN-04
    severity: info
    disposition: open
    title: "three spellings of \"the tree plus its string code\""
  - id: CR-02
    severity: critical
    disposition: fixed
    title: "R3 and R4 do not read code inside strings, so Django set up in a string before the helper call writes the developer database unreported, against the README's explicit claim"
  - id: IN-03
    severity: info
    disposition: open
    title: "The README and docstring still leave out checks the code makes (round-5 IN-01, unchanged)"
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
total: 13
unparsed: 1
recorded: 2026-10-11T13:55:46.597Z
---

# Phase 40: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | open | - |
| WR-01 | warning | open | - |
| WR-02 | warning | open | - |
| WR-03 | warning | open | - |
| WR-04 | warning | open | - |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-04 | info | open | - |
| CR-02 | critical | fixed | fixed in 40-17 Task 2 (fdd8b3e) and 40-18 Task 3 (aff2547): R3, R4 and the Django-backed gate read code inside strings to two levels, so the review's timeit probe and the verifier's settings.configure variant are reported on a copy of import_campaign_csv_demo (D-G16) (not in the current review) |
| IN-03 | info | open | out of scope for the gap round (developer decision 2026-10-09); for Phase 41 triage (not in the current review) |
| IN-05 | info | fixed | fixed in 40-07 Task 1 (29490c3): the eight schedule files are created with dir=scratch_db_dir, so teardown removes them (not in the current review) |
| IN-06 | info | fixed | fixed in 40-05 Task 3 (819321e): teardown refuses a directory not directly inside tempfile.gettempdir() or not the one setup created; two refusal tests (not in the current review) |
| IN-07 | info | deferred | deferred. Partly fixed in 40-05 Task 2 (7cfd78e): re.escape(tmp), a snapshot-mode subprocess test and a forced snapshot failure test. The remaining part, using DISMISSED_TABLE_PREFIX and CONFIRMED_TABLE_PREFIX directly instead of the _DISMISSED_TABLE_PREFIX and _CONFIRMED_TABLE_PREFIX aliases in campaign_views.py, is out of scope this round (developer decision 2026-10-09) and left for Phase 41 triage (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
