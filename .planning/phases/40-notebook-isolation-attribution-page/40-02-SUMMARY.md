---
phase: 40-notebook-isolation-attribution-page
plan: 02
subsystem: testing
tags: [jupyter, nbconvert, ast-guard, scratch-database, readme, review-ledger]

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-01's shared helper docs/notebooks/fomo_nb_utils.py and four notebooks already on it"
provides:
  - "AST-based notebook guard (isolation_problems, R0-R11) that checks each notebook's current source, not its committed output"
  - "import_campaign_csv, sync_gemini_observation_calendar, telescope_runs (fresh) and project_observation_calendar (snapshot) moved onto the helper and re-executed: all eight notebooks now call it"
  - "docs/notebooks/README.md states the isolation rule actually followed and lists the snapshot and fresh notebooks, pinned by a test"
  - "ROADMAP Phase 40 success criterion 1 and REQUIREMENTS WARN-05 reworded to the same sentence"
  - "37.1 review ledger: WR-13, WR-14, WR-16, WR-17, IN-13, IN-14, IN-17 marked fixed"
affects: [40-04 campaign_lifecycle_demo re-execution and WR-15/WR-18/IN-12 ledger rows, docs/notebooks/README.md]

requirements-completed: [WARN-02, WARN-03, WARN-05, WARN-06]

coverage:
  - id: D1
    description: "isolation_problems() reports each WR-13 bypass on synthetic notebooks (own FOMO_DATABASE_PATH assignment with stale output, aliased django.setup, setup call hidden in def/if False, nested or earlier Django import, aliased/missing/duplicate helper call, byte copy, developer-database literal, bad execution counts, error output, wrong teardown, assert in setup or wrapping a write, unparseable Django cell)"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestIsolationRulesOnSyntheticNotebooks"
        status: pass
    human_judgment: false
  - id: D2
    description: "The same bypasses applied to a copy of the real import_campaign_csv_demo.ipynb with its committed output kept are reported, and the unmodified notebook is clean"
    requirement: WARN-02
    verification:
      - kind: other
        ref: "plan Task 1 verify 2: manage.py shell bypass replay (3 bypasses caught, real notebook clean)"
        status: pass
    human_judgment: false
  - id: D3
    description: "All eight real notebooks pass the guard; the set of names equals the eight expected, each Django-backed with one valid setup call; guard names match the helper's callables and constants"
    requirement: WARN-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestPreExecutedNotebooksUseTheHelper"
        status: pass
    human_judgment: false
  - id: D4
    description: "WARN-03 audit: 236 demonstration asserts across the eight notebooks, none in a setup cell, none wrapping a write call (guard rule R11 keeps it that way)"
    requirement: WARN-03
    verification:
      - kind: other
        ref: "plan Task 2 verify 2 AST audit: total=236 in_setup_cells=0 wrapping_a_write=0"
        status: pass
    human_judgment: false
  - id: D5
    description: "telescope_runs_demo's committed Las Campanas dip is in degrees and within 1.42-1.46; no committed output reports 'arcmin deg'; no credential-shaped text in any notebook file"
    requirement: WARN-06
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestPreExecutedNotebooksUseTheHelper.test_telescope_runs_demo_horizon_dip_matches_the_stage1_contract"
        status: pass
    human_judgment: false
  - id: D6
    description: "The D-04 rule sentence is verbatim in the README, ROADMAP SC 1 and REQUIREMENTS WARN-05, and the README's snapshot/fresh lists equal each notebook's setup-call mode"
    requirement: WARN-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestReadmeStatesTheIsolationRule"
        status: pass
    human_judgment: false
  - id: D7
    description: "Rewritten notebook prose (setup and teardown cells) and the README read correctly against the helper's behavior and the regenerated output"
    verification: []
    human_judgment: true
    rationale: "Prose accuracy is a reading judgment no test asserts"

commits: 3
plan_head_before: 12d3c49f87c9b562f7f29957b4e0fcca685fdea1
plan_head_after: 9bdc79d4c791f2b97444d5144e00328b599de0d7
actuals:
  tokens: 18000
  tasks: 2
  commits: 3
duration: 20min
completed: 2026-10-09
status: complete
---

# Phase 40 Plan 02: Notebook guard rewrite and the last four notebooks Summary

**One AST guard now checks every pre-executed notebook's current source (helper call first, no own Django setup or `FOMO_DATABASE_PATH`, no byte copy, one fresh run), all eight notebooks run on the shared scratch-database helper, and the README, ROADMAP and REQUIREMENTS state the same isolation rule.**

## Performance

- **Duration:** about 20 min
- **Tasks:** 2 (Task 1 tracer, Task 2 auto)
- **Files modified:** 9 (plus this SUMMARY, STATE.md, ROADMAP.md and REQUIREMENTS.md in the metadata commit)

## Accomplishments

- `solsys_code/tests/test_pre_executed_notebooks.py` rewritten: `is_django_backed`, `setup_call`, `notebook_mode` and `isolation_problems` implement R0-R11 over top-level statements, so a setup call hidden in a `def` or `if False:`, an aliased `django.setup`, an own `FOMO_DATABASE_PATH` assignment (output kept stale) and a Django import before the helper call are all reported. 38 tests: 30 synthetic bypass tests, the real-notebook tests and the two README tests. A no-op guard fails 34 of the synthetic assertions (RED); the real guard passes them (GREEN).
- `import_campaign_csv_demo`, `sync_gemini_observation_calendar_demo`, `telescope_runs_demo` call `setup_scratch_django(repo_root_path, mode='fresh')`; `project_observation_calendar_demo` calls it with `mode='snapshot'` in setup cell `7022f987` (id kept); each ends with `teardown_scratch_django(scratch_db_dir)`. All four were re-executed with `TMPDIR=/tmp`, counts 1..N, no error output; `test_projector_demo_notebook` stays green.
- IN-13: `import_campaign_csv_demo` writes its leading-comment CSV to `scratch_db_dir / 'leading_comment_export.csv'` (cell `3b49c87a`) and reads the fixture in a `with` block (cell `16d14cf8`; the plan named `1093927e`, but the unclosed `fixture_path.open()` is in `16d14cf8`).
- `docs/notebooks/README.md` rewritten: the D-04 rule sentence, the two mode lists (snapshot: `project_observation_calendar_demo`, `reconcile_campaign_runs_demo`; fresh: the other six), helper usage, regeneration, the populated-developer-database requirement for snapshot notebooks, and the leftover `/tmp/fomo-notebook-db-*` cleanup note. ROADMAP Phase 40 success criterion 1 and REQUIREMENTS WARN-05 carry the same sentence.
- 37.1 review ledger: WR-13, WR-14, WR-16, WR-17, IN-13, IN-14, IN-17 `fixed` in the front matter and table, each row citing Phase 40 and commits.

## Task Commits

1. **Task 1 (tracer): guard rewrite and four notebooks, one commit** - `f24b04f` (test)
2. **Task 2: README, ROADMAP SC 1, WARN-05 wording, README test** - `8dfdf56` (docs)
3. **Ledger (.planning-only, `SKIP=django-test`)** - `9bdc79d` (docs)

Both code commits went through the full django-test hook (2271 tests OK in the final whole-suite run).

Tracer gate: auto/default mode with an automated-only `<verify>`; the tracer's three verify commands were run end to end and passed before Task 2 started.

## Audit and snapshot evidence

- **WARN-03 assert audit:** `assert audit: total=236 in_setup_cells=0 wrapping_a_write=0` (per notebook: campaign_lifecycle 93, reconcile_campaign_runs 86, load_telescope_runs 27, sync_gemini 16, project_observation_calendar 8, import_campaign_csv 6, backfill_lco 0, telescope_runs 0). No demonstration assert was edited, so the planning-time count minus the seven setup asserts holds exactly.
- **Projector snapshot headline counts (as committed; vary per regeneration):** facility-url events 459 (was 204), `RUN:` 22, `GEM:` 0, blank-url 0; dry run and both sweeps `unchanged: 459`, created/updated 0 (the developer database is already converged, so the takeover finds nothing); LCO/SOAR records 459 = events 459; stage markers `[Q] 82, [S] 68, [O] 175, [X] 58, [C] 75, [F] 1, [?] 0`; 150 pending `KEY2026B-004` records in the copy (was 0). No prose or code cell quoted any of these numbers, so none needed editing.
- **Fresh notebooks:** output unchanged apart from the setup cell (now the helper's `Django ready:` and `Resolved database:` lines) and the sync notebook's time-relative window values.

## Markdown cells whose prose changed

- `import_campaign_csv_demo`: `43e2c1c3`, `9c899021`
- `sync_gemini_observation_calendar_demo`: `c674d60c`, `c1d2e3f4`, `c8f6eeb3`
- `project_observation_calendar_demo`: `9607e5f3`, `317a98bb`
- `telescope_runs_demo`: `23326d63`, `2e2d138e`

Code cells changed besides each notebook's setup and last cell: `3b49c87a` and `16d14cf8` (import_campaign_csv only).

## Files Created/Modified

- `solsys_code/tests/test_pre_executed_notebooks.py` - helper-based guard, synthetic and real-notebook tests, README tests
- `docs/notebooks/pre_executed/{import_campaign_csv,sync_gemini_observation_calendar,project_observation_calendar,telescope_runs}_demo.ipynb` - converted and re-executed
- `docs/notebooks/README.md` - isolation section rewritten
- `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md` - rule sentence in Phase 40 SC 1 and WARN-05
- `.planning/milestones/v2.4-phases/37.1-.../37.1-REVIEW-DISPOSITION.md` - seven findings fixed

## Decisions Made

- Followed the plan's guard contract (R0-R11), including Open Question 1 (fourth rule banning `fomo_db.sqlite3` in code strings) and the assignment rule as "no code string or keyword naming `FOMO_DATABASE_PATH`".
- Beyond the plan's wording: the byte-copy rule also catches `import shutil as sh; sh.copy(...)` and `from shutil import *`; a second call to the setup function anywhere (not only at top level) is reported.
- The guard module's own docstring and constants are the single source for the rule sentence (`ISOLATION_RULE`); the README test compares the README to it, and the verify script compares ROADMAP and REQUIREMENTS.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] IN-13 fixture read is in cell `16d14cf8`, not `1093927e`**
- **Found during:** Task 1
- **Issue:** the plan said to wrap the fixture read in cell `1093927e`; that cell only passes the path to `call_command`. The unclosed `csv.DictReader(fixture_path.open())` the review meant is in `16d14cf8`.
- **Fix:** applied the `with fixture_path.open(newline='') as fixture_file:` change there.
- **Files modified:** `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`
- **Committed in:** `f24b04f`

**Total deviations:** 1 auto-fixed (Rule 1, wrong cell id in the plan). **Impact:** none on scope.

## Issues Encountered

- The ledger's front-matter counter `open: 27` was already one higher than the real number of `open` findings before this plan (26) and is now stale by eight; it is generator-owned bookkeeping, so it was left alone.
- A pre-existing `/tmp/fomo-notebook-db-i_k4rosi` (dated 2 Oct, noted by 40-01) remains; none of this plan's runs left a scratch directory behind.

## Known Stubs

None.

## Threat Flags

None. T-40-06 (guard bypass), T-40-07 (credential-shaped text in snapshot output) and T-40-08 (a write hidden in an `assert`) are mitigated as planned: AST rules R2/R3/R4/R8 plus per-bypass tests and the real-notebook replay; `test_no_credential_shaped_text_in_any_notebook`; rule R11 and the 236/0/0 audit.

## Next Phase Readiness

All eight notebooks now call the helper, so CLAUDE.md's "all eight" wording is literally true. 40-04 re-executes `campaign_lifecycle_demo` for the attribution page and marks WR-15, WR-18 and IN-12 in the ledger.

## Self-Check: PASSED

- `solsys_code/tests/test_pre_executed_notebooks.py`, `docs/notebooks/README.md` and the four converted notebooks exist and are tracked and committed.
- Commits `f24b04f`, `8dfdf56`, `9bdc79d` are ancestors of HEAD; `git rev-list --count 12d3c49..HEAD` was 3 before this SUMMARY.
- Guard, helper, projector and import-campaign tests: 132 OK; whole unit suite (`--exclude-tag functional --exclude-tag ephemeris_segfault --exclude-tag migration --parallel`): 2271 OK.

---
*Phase: 40-notebook-isolation-attribution-page*
*Completed: 2026-10-09*
