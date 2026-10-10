---
phase: 40-notebook-isolation-attribution-page
plan: 09
subsystem: docs
tags: [notebooks, paired-docs, scratch-database, snapshot, gap-closure, override]
gap_closure: true
gap_ids: [IN-01, IN-02]

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-08 exception-safe helper (new teardown line, narrowed refusal wording) and extended guard"
provides:
  - "All eight pre-executed notebooks re-executed top to bottom on the 40-08 helper, committed with output"
  - "Setup prose in all eight says a failed setup leaves no database reachable only once the helper has read Django's settings (S1-S3); teardown prose says what the helper checked (T1, T2)"
  - "reconcile_campaign_runs_demo NF-14 labels that describe what the cell counts; project_observation_calendar_demo requirement table citing real headings"
  - "Review ledger rows for IN-01 and IN-02 fixed (open 3 of 11)"
  - "Developer override for the residual WARN-02 risk recorded in 40-VERIFICATION.md"
affects: [phase-41, verify-work]

tech-stack:
  added: []
  patterns:
    - "Data-independent labels: a snapshot notebook's printed label names what the cell counts, never a record the snapshot may no longer hold"

key-files:
  created: []
  modified:
    - docs/notebooks/pre_executed/telescope_runs_demo.ipynb
    - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
    - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
    - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
    - docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb
    - docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
    - .planning/phases/40-notebook-isolation-attribution-page/40-REVIEW-DISPOSITION.md
    - .planning/phases/40-notebook-isolation-attribution-page/40-VERIFICATION.md

key-decisions:
  - "D-G2: teardown prose and printed teardown line now say what the helper checked (T1); the old 'never opened for writing' claim is gone from every cell source and output"
  - "D-G3: refusal wording in all eight notebooks covers only failures once the helper has read Django's settings; an earlier failure routes nothing and allows a retry"
  - "D-G4: reconcile NF-14 labels bound once as NF14_EXPECTED_UNEXPLAINED; projector table cites 'The whole real corpus, not a sample' and its 'Projected events by stage marker' cell"
  - "D-G1: override recorded as the plan's last commit, reason names only the residual the guard docstring names"

requirements-completed: [WARN-05, WARN-02, WARN-03]

actuals:
  tokens: 39500   # chars/4 over the plan's five commits' diff (158158 chars incl. diff headers)
  tasks: 3
  commits: 5
plan_head_before: 391b8adeaa22ac95ccafac3f59c3fd49a0237989
plan_head_after: b6bfbca7fddf794ceb55bf923dbfba010531875b

coverage:
  - id: D1
    description: "All eight notebooks re-executed on the 40-08 helper: counts 1..N, no error output, Django-ready and Resolved lines in /tmp/fomo-notebook-db-*, teardown cell last with both lines and the new second line, directory gone, no /home/ in helper output"
    requirement: WARN-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestPreExecutedNotebooksUseTheHelper.test_every_real_pre_executed_notebook_is_isolated"
        status: pass
      - kind: command
        ref: "per-notebook nbformat check scripts (Task 1-3 verify commands)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Setup and teardown prose (S1-S3, T1, T2, 6a9bd576 clause) matches the helper's refusal and teardown behavior"
    requirement: WARN-03
    verification:
      - kind: command
        ref: "Task 2 and Task 3 sentence checks over all eight notebooks"
        status: pass
    human_judgment: true
    rationale: "Prose is read once at /gsd-verify-work (verification human item 1)"
  - id: D3
    description: "IN-02: NF-14 labels data-independent; requirement table cites real headings only"
    requirement: WARN-02
    verification:
      - kind: command
        ref: "Task 3 snapshot check (three labelled CommandError lines; every quoted table phrase is a heading or an output line)"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_projector_demo_notebook.py"
        status: pass
    human_judgment: true
    rationale: "Cutover and requirement-table claims are read against the committed output for this regeneration at /gsd-verify-work"
  - id: D4
    description: "Ledger records IN-01 and IN-02 fixed; override recorded last"
    requirement: WARN-02
    verification:
      - kind: command
        ref: "Task 3 ledger check and override check"
        status: pass
    human_judgment: true
    rationale: "The developer reads the override reason at /gsd-verify-work and confirms it names only the residual accepted"

duration: 19 min
completed: 2026-10-10
status: complete
---

# Phase 40 Plan 09: Paired notebook docs, IN-02 fixes, ledger and override Summary

**All eight pre-executed notebooks were re-executed on the exception-safe 40-08 helper and now describe its refusal and teardown exactly; the two IN-02 prose defects are fixed with labels and citations that stay true on any snapshot; the ledger records IN-01 and IN-02 fixed, and the developer's override for the residual WARN-02 risk is the round's last commit.**

## Commits

| Task | Commit | What |
|------|--------|------|
| 1 (tracer) | `25eae59` | telescope_runs_demo re-executed; S1-S3 in 23326d63, T1 in 2e2d138e (full hook) |
| 2 | `c68bf7b` | backfill_lco_observations, campaign_lifecycle, import_campaign_csv, load_telescope_runs, sync_gemini_observation_calendar re-executed with the same prose changes (full hook) |
| 3 | `0020360` | project_observation_calendar and reconcile_campaign_runs re-executed (snapshot mode); D-G4 code-cell edits; T1+T2; 6a9bd576 clause (full hook) |
| ledger | `93bb514` | 40-REVIEW-DISPOSITION.md: IN-01 and IN-02 fixed, `open: 3`, `total: 11` (SKIP=django-test, `.planning` only) |
| override | `b6bfbca` | 40-VERIFICATION.md front matter gains `overrides:` (SKIP=django-test, last commit of the plan) |

Nothing was pushed. `plan_head_before` is `391b8ad`; 5 commits since.

## Per notebook

| Notebook | Mode | Resolved line (this run) | Markdown cells whose prose changed | Code cells whose output changed apart from setup/teardown lines |
|----------|------|--------------------------|------------------------------------|----------------------------------------------------------------|
| telescope_runs_demo | fresh | `/tmp/fomo-notebook-db-l7qrueld/fomo_db.sqlite3` (fresh scratch database, migrated from empty) | 23326d63, 2e2d138e | none |
| backfill_lco_observations_demo | fresh | `/tmp/fomo-notebook-db-0ua9h8zk/...` (fresh) | 86dd4f0a | 6a1f0c22 (`last_run_at` timestamps), 02c66802 and e2c7f5a4 (stdout/stderr interleaving only) |
| campaign_lifecycle_demo | fresh | `/tmp/fomo-notebook-db-7wo0zgn_/...` (fresh) | 7dfef438 | 729bd69e (`confirmed_at` timestamp), f13a1c02 (interleaving only) |
| import_campaign_csv_demo | fresh | `/tmp/fomo-notebook-db-9ao_jsn7/...` (fresh) | 43e2c1c3, 9c899021 | none |
| load_telescope_runs_demo | fresh | `/tmp/fomo-notebook-db-qassjyrf/...` (fresh) | 49ff72df | a0c4592f (random temp file name) |
| sync_gemini_observation_calendar_demo | fresh | `/tmp/fomo-notebook-db-ryrc7f1a/...` (fresh) | c674d60c, c8f6eeb3 | d0e1f2a3, b4c5d6e7, d6e7f8a9 (time-relative window values) |
| project_observation_calendar_demo | snapshot | `/tmp/fomo-notebook-db-v763_kpj/...` (scratch copy of the developer database, taken read-only with SQLite's online backup and migrated to head) | 9607e5f3, 317a98bb, 6a9bd576, 7e7bd66e (see Deviations) | b9831b5d (random ids), 055eff76 (stage counts), 250b5d0b (pending count), 35debc54 (code cell, D-G4) |
| reconcile_campaign_runs_demo | snapshot | `/tmp/fomo-notebook-db-i28gr3bk/...` (scratch copy, as above) | dfcc0343, 98bba968 | 2c723427 and 48b9f602 (code cells, D-G4), 717711a4, db8b9701, ed8be386 (random target/observation ids only) |

Confirmed: campaign_lifecycle_demo's 40-04 High-band marker and Undo cells print as before (no output diff against 488a7bb apart from the two timestamp and interleaving cells above); load_telescope_runs_demo's schedule file lies inside the scratch directory (`Schedule file written to: /tmp/fomo-notebook-db-qassjyrf/tmp8_4fa35l.txt`).

## Snapshot headline counts for this regeneration

The developer database file was not touched by me (no `git status` entry; read only by the helper's `mode=ro` backup).

**reconcile_campaign_runs_demo**

| Metric | This regeneration | 40-07 |
|--------|-------------------|-------|
| CalendarEvent total (before cutover) | 511 | 511 |
| facility-url events | 459 | 459 |
| ALLOC: events | 30 | 30 |
| bare RUN: containers | 22 | 22 |
| blank-url / date-bearing RUN: events | 0 / 0 | 0 / 0 |
| CampaignRun rows | 41 | 41 |
| cutover `candidates` (dry and real) | 0, no `CommandError` | 0 |
| first sweep | `runs: 41, created: 0, updated: 0, unchanged: 52, skipped: 4, retired: 5` | same |

**project_observation_calendar_demo**

| Metric | This regeneration | 40-07 |
|--------|-------------------|-------|
| facility-url events | 459 | 459 |
| `RUN:` / `GEM:` / blank-url | 22 / 0 / 0 | 22 / 0 / 0 |
| dry run and both sweeps | `unchanged: 459`, created and updated 0 | same |
| stage markers | `[Q] 95, [S] 52, [O] 176, [X] 60, [C] 75, [F] 1, [?] 0` | `[Q] 96, [S] 52, [O] 176, [X] 59, [C] 75, [F] 1, [?] 0` |
| pending `KEY2026B-004` records in the copy | 147 | 148 |

The only drift is one record moving from `[Q]` to `[X]` and one pending record fewer: the developer's cron run changed the live database between regenerations (RESEARCH A6, accepted). No markdown sentence quotes either number, so no prose changed for it; the reconcile cutover and summary prose (8b703ea8, ce1fd748, e1edbaa7, d2adacf8, 3616c012) still match the output (zero candidates, no CommandError in the real-data cutover cells).

**The three committed NF-14 CommandError lines** (reconcile cells 2c723427 and 48b9f602):

- `dry run CommandError (expected -- the duplicate-identity group plus any unexplained snapshot event): 3 event(s) could not be explained and were left untouched (duplicate_identity=3). Resolve ...`
- `real run CommandError (expected -- the duplicate-identity group plus any unexplained snapshot event): 3 event(s) could not be explained and were left untouched (duplicate_identity=3). Resolve ...`
- `second real-run CommandError (expected -- the duplicate-identity group plus any unexplained snapshot event, unchanged): 3 event(s) could not be explained and were left untouched (duplicate_identity=3). Resolve ...`

## Home-path inventory

The inventory printed one line, and its closing OK line: `OK: all eight notebooks carry the new teardown line and S1, no old claim anywhere; 1 home-path line(s) remain, none from the helper`.

- `HOME-PATH (not helper output): campaign_lifecycle_demo.ipynb d687bfab Using fallback library next to module: /home/tlister/venv/devel_fomo311_venv/lib/python3.11/site-packages/spiceypy/utils/libcspice.so`

Left as is: it is the third-party spiceypy log line that 40-07 already listed; it is not printed by the helper, and removing it would mean editing a code cell or suppressing a library's own log.

## Checks that passed (OK lines)

- Task 1, 2 and 3 test runs: guard + helper + projector evidence, `Ran 98 tests ... OK` (Task 2: `Ran 93 tests ... OK`, guard + helper only). This includes `test_every_real_pre_executed_notebook_is_isolated`, the WARN-06 dip pin and the credential-text test.
- `OK: fresh notebooks checked: 1` (telescope), `OK: fresh notebooks checked: 5`, and the snapshot check `OK: both snapshot notebooks re-executed through the read-only backup; S1-S3, T1, T2 in place; NF-14 labels data-independent; requirement table cites real headings; only the three allowed code cells changed`.
- Ledger: `OK: IN-01 and IN-02 fixed with real commits touching their files, open 5 -> 3, 5 lines since 40-08 and 9 since 488a7bb, other rows untouched`.
- Override: `OK: override recorded last` (the nbformat/YAML check run before and after the commit: verbatim must_have, Tim Lister, 2026-10-10T13:46:17Z, nothing else in the report changed, 23 lines added and none removed, commit holds only 40-VERIFICATION.md and follows the ledger commit).
- Final gate: `OK: snapshot notebooks committed together (0020360...), ledger alone after them (93bb514...), override last at HEAD (b6bfbca...)`; no 40-09 commit touched the helper, README or `solsys_code`; runbooks and 40-REVIEW.md unchanged since 488a7bb.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] project_observation_calendar_demo intro cell 7e7bd66e still carried the old claim**
- **Found during:** Task 3 verify (the snapshot check asserts no cell holds the old "never opened for writing" claim)
- **Issue:** the notebook's title cell 7e7bd66e (not listed in the plan's cell inventory) ended its "Every run works on a scratch copy" paragraph with "the developer database is never opened for writing", the claim D-G2 withdraws.
- **Fix:** replaced that clause with the same wording the plan gives for 6a9bd576 ("the helper opens the developer database only for its read-only (`mode=ro`) online backup"). Markdown only; edited in place after the execution, so no output changed.
- **Files modified:** docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb
- **Commit:** `0020360`

**Total deviations:** 1 auto-fixed (prose cell the plan's inventory missed). **Impact:** none on behavior; the must_have "no cell source carries the old claim" required it.

## Notes

- A directory `/tmp/fomo-notebook-db-i_k4rosi` existed before this plan started (left by an earlier run) and was not touched; every directory this plan's runs created is gone (checked by the per-notebook scripts).
- ruff-format rewrapped the PROJ-03 tuple in cell 35debc54 into the multi-line form; that is the only formatting change.
- IN-03 (two untested fail-closed branches), WR-03, IN-04, IN-07's `campaign_views.py` alias tidy-up and round-1 IN-03's runbook attribution-pass wording remain open for Phase 41 triage. No `docs/runbooks/` page documents the helper, so no runbook changed.

## Threat Flags

None. No new network endpoint, auth path or trust-boundary file access.

## Known Stubs

None.

## Self-Check: PASSED

The eight notebooks, the ledger, 40-VERIFICATION.md and commits `25eae59`, `c68bf7b`, `0020360`, `93bb514` and `b6bfbca` exist and are ancestors of HEAD.
