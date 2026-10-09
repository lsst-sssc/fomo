---
phase: 40-notebook-isolation-attribution-page
plan: 01
subsystem: testing
tags: [jupyter, nbconvert, sqlite3-backup, django-setup, scratch-database]

requires:
  - phase: 39.1-test-suite-speed-up
    provides: parallel django-test hook (about 80 s) that gates both commits
provides:
  - docs/notebooks/fomo_nb_utils.py shared scratch-database helper (setup_scratch_django, teardown_scratch_django, snapshot_database, check_resolved_database, prepare_scratch_database, developer_database_path)
  - solsys_code/tests/test_notebook_helper.py (14 tests that run the real helper, including python -O and PYTHONOPTIMIZE refusal)
  - four pre-executed notebooks moved onto the helper and re-executed with committed output
affects: [40-02 guard rewrite and the other four notebooks, 40-04 campaign_lifecycle_demo re-execution, docs/notebooks/README.md]

actuals:
  tokens: 21300
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "One helper owns notebook database setup: scratch dir, FOMO_DATABASE_PATH, fresh migrate or mode=ro online-backup snapshot, django.setup(), resolved-path RuntimeError, evidence line, teardown"
    - "Helper imports Django lazily, so tests load it in-process; anything that really calls django.setup() runs in a subprocess"

key-files:
  created:
    - docs/notebooks/fomo_nb_utils.py
    - solsys_code/tests/test_notebook_helper.py
  modified:
    - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
    - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
    - CLAUDE.md

key-decisions:
  - "Helper API fixed as planned: setup_scratch_django(repo_root, mode='fresh'|'snapshot') -> scratch dir, teardown_scratch_django(scratch_dir); names are called by all eight notebooks and 40-02's guard"
  - "setup refuses (RuntimeError) when Django's app registry is already ready, before creating anything; any failure after the directory exists removes it before re-raising"
  - "Teardown refuses any path not named fomo-notebook-db-* and removes the directory without ignore_errors"
  - "Verdicts as researched: fresh = load_telescope_runs, backfill_lco, campaign_lifecycle; snapshot = reconcile_campaign_runs"

patterns-established:
  - "Notebook setup cell: sys.path gains repo root and docs/notebooks, then `from fomo_nb_utils import ...`, one reason comment, `scratch_db_dir = setup_scratch_django(repo_root_path, mode=...)`; last cell is exactly `teardown_scratch_django(scratch_db_dir)`"

requirements-completed: [WARN-05, WARN-03]

coverage:
  - id: D1
    description: "Shared helper docs/notebooks/fomo_nb_utils.py: special_locations() byte-identical to feature/skymap, no Django import at load, no assert, no shutil copy"
    requirement: WARN-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestNotebookHelperModule"
        status: pass
    human_judgment: false
  - id: D2
    description: "Fresh setup routes to a fomo-notebook-db-* scratch database, migrates it, prints the evidence line, and teardown removes it"
    requirement: WARN-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestHelperRunsForReal.test_fresh_setup_routes_migrates_and_tears_down"
        status: pass
    human_judgment: false
  - id: D3
    description: "A DATABASES override is refused with RuntimeError before migrate, under plain python, python -O and PYTHONOPTIMIZE=1, and the other database file is never created"
    requirement: WARN-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestHelperRunsForReal.test_a_databases_override_is_refused_before_migrate"
        status: pass
    human_judgment: false
  - id: D4
    description: "Snapshot opens the developer database through a mode=ro URI with Connection.backup(), leaves source bytes unchanged, and snapshot mode with no developer database raises before creating anything"
    requirement: WARN-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestSnapshotAndRouting"
        status: pass
    human_judgment: false
  - id: D5
    description: "Four notebooks (load_telescope_runs, backfill_lco_observations, campaign_lifecycle fresh; reconcile_campaign_runs snapshot) call the helper first, contain no django.setup / shutil copy / FOMO_DATABASE_PATH / fomo_db.sqlite3 code, were re-executed once top to bottom with TMPDIR=/tmp, and end with the teardown call"
    requirement: WARN-05
    verification:
      - kind: other
        ref: "plan Task 1 and Task 2 notebook AST verify scripts (all OK); old guard solsys_code.tests.test_pre_executed_notebooks and test_projector_demo_notebook green"
        status: pass
    human_judgment: false
  - id: D6
    description: "Rewritten notebook prose (setup, teardown and data-dependent claims) reads correctly against the new executed output"
    verification: []
    human_judgment: true
    rationale: "Prose accuracy against regenerated output is a reading judgment no test asserts; the reconcile cutover prose in particular now describes a snapshot-dependent no-op"

commits: 2
plan_head_before: 4aa9d7ee83dfb3903e8a60cfac8de3544d1321dd
plan_head_after: c760c61c8685dfaf20b814b5071bc6832ed138bd
duration: 70min
completed: 2026-10-09
status: complete
---

# Phase 40 Plan 01: Notebook scratch-database helper Summary

**One shared helper (`docs/notebooks/fomo_nb_utils.py`) now owns every pre-executed notebook's database setup, with `RuntimeError` routing checks that survive `python -O`, a `mode=ro` online-backup snapshot, and four former hand-rolled notebooks (three of them byte-copiers) re-executed on it.**

## Performance

- **Duration:** about 70 min
- **Started:** 2026-10-09T22:50Z (approx.)
- **Completed:** 2026-10-09T23:15Z (last task commit; metadata commit follows)
- **Tasks:** 2
- **Files modified:** 7 (2 created, 5 modified)

## Accomplishments

- `docs/notebooks/fomo_nb_utils.py`: the skymap `special_locations()` seed kept byte-identical (AST source-segment compare against `origin/feature/skymap`), with the scratch-database functions appended below it. Django is imported only inside function bodies, so importing the module never touches Django.
- `solsys_code/tests/test_notebook_helper.py`: 14 tests that run the real helper. Pure parts run in-process against throwaway SQLite files; fresh setup and the `DATABASES`-override refusal run in subprocesses, the refusal under plain `python`, `python -O` and `PYTHONOPTIMIZE=1`, asserting the other database file is never created.
- `load_telescope_runs_demo`, `backfill_lco_observations_demo`, `campaign_lifecycle_demo` (fresh) and `reconcile_campaign_runs_demo` (snapshot) call `setup_scratch_django` as the first Django-related statement and end with `teardown_scratch_django(scratch_db_dir)`. None byte-copies `src/fomo_db.sqlite3` any more.
- CLAUDE.md paired-docs map gains the `docs/notebooks/fomo_nb_utils.py` line.

## Task Commits

1. **Task 1 (tracer): helper, its tests, load_telescope_runs_demo, CLAUDE.md line** - `55e51e8` (feat)
2. **Task 2: backfill, campaign lifecycle and reconcile demos on the helper** - `c760c61` (docs)

**Plan metadata:** recorded in the following docs commit.

The tracer gate: auto/default end-of-phase mode with an automated-only `<verify>`; the tracer verify (helper tests, old guard, projector guard, notebook AST checks) was run end to end and passed before the second task started. RED was the 14 tests erroring because the helper file did not exist; GREEN was all 14 passing (semantic assessment: the failures were `ModuleNotFoundError` for `fomo_nb_utils` and a missing-file load, the plan's stated RED).

## Reconcile snapshot headline counts (as committed)

`reconcile_campaign_runs_demo` is a snapshot notebook, so these vary per regeneration:

| Metric | Committed value |
|--------|-----------------|
| CalendarEvent total (before cutover) | 511 (was 316) |
| facility-url events | 459 (was 264) |
| ALLOC: events | 30 |
| bare RUN: containers | 22 |
| blank-url / date-bearing RUN: events | 0 / 0 |
| CampaignRun rows | 41 |
| cutover `candidates` (dry and real) | 0, no `CommandError` (the developer database was already cut over) |
| first sweep | `runs: 41, created: 0, updated: 0, unchanged: 52, skipped: 4, retired: 5` |

Fresh-notebook changes: `load_telescope_runs_demo` runs now start from `CampaignRun.objects.count() = 0` and `created` for all four Observatory rows; `campaign_lifecycle_demo` shows `attribution_count=1` (was 8), `CalendarEvent.objects.count()` 7 (was 346).

## Markdown cells whose prose changed

- `load_telescope_runs_demo`: `5febc666`, `49ff72df`, `041476f5`, `78e0bb2f`, `31ff3edb`
- `backfill_lco_observations_demo`: `ae33103d`, `86dd4f0a`, `46602df0`
- `campaign_lifecycle_demo`: `7dfef438`, `061b7205`, `32a7b0c9`, `bd57d926`, `dc2d6ada`, `687e4453`, `234805b6`
- `reconcile_campaign_runs_demo`: `8b703ea8`, `dfcc0343`, `e1edbaa7`, `ce1fd748`, `d2adacf8`, `98bba968`

Code cells other than each notebook's setup and last cell are unchanged (source compared against the plan's starting commit).

## Files Created/Modified

- `docs/notebooks/fomo_nb_utils.py` - seeded `special_locations()` plus the scratch-database helper
- `solsys_code/tests/test_notebook_helper.py` - executed tests for the helper
- `docs/notebooks/pre_executed/{load_telescope_runs,backfill_lco_observations,campaign_lifecycle,reconcile_campaign_runs}_demo.ipynb` - converted and re-executed
- `CLAUDE.md` - paired-docs map line

## Decisions Made

Followed the plan's API and ordering. Small execution choices: the converted setup cells drop the now-unused `import os` (the plan's shape listed it, but no later cell uses `os`); `snapshot_database` resolves the source path before building the `file:` URI so a relative path works.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Stale prose after regenerating reconcile_campaign_runs_demo**
- **Found during:** Task 2
- **Issue:** the snapshot of the current developer database was already cut over, so the committed cutover cells print `candidates: 0` and "No CommandError raised"; the old prose (`ce1fd748`, `e1edbaa7`, `d2adacf8`) said the cutover finds a junk `tmp` row and raises, and quoted a pristine pre-cutover state.
- **Fix:** rewrote those cells (and added a paragraph to `8b703ea8`) to say the outcome depends on the snapshot and describe the committed no-op; the NF-14 fixture still shows a real conversion. This is the plan's RESEARCH A6 / Task 2 step 5 instruction, recorded here because the demonstration of the cutover's conversion on real data is weaker in this regeneration.
- **Files modified:** `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`
- **Committed in:** `c760c61`

**2. [Rule 1 - Bug] Stale "dev DB" wording in campaign_lifecycle_demo**
- **Found during:** Task 2
- **Issue:** `dc2d6ada`, `687e4453` still referred to a shared dev DB and pre-existing orphans, wrong on a fresh database.
- **Fix:** reworded both.
- **Committed in:** `c760c61`

---

**Total deviations:** 2 auto-fixed (both Rule 1, prose accuracy). **Impact on plan:** none on scope; the reconcile cutover section is a no-op on real data for as long as the developer database stays cut over.

## Issues Encountered

- A pre-existing `/tmp/fomo-notebook-db-i_k4rosi` (dated 2 Oct, not created by this plan) remains; left alone.
- `.planning/STATE.md` carried the orchestrator's uncommitted phase-start update; it is committed with the metadata commit.

## Known Stubs

None.

## Threat Flags

None. The plan's register is mitigated as designed: T-40-01 (plain assignment plus `check_resolved_database` before migrate, proven by the override subprocess test), T-40-02 (`mode=ro` + `backup()`, no `shutil.copy*`, AST and unit tests), T-40-03 (failure cleanup in setup, teardown without `ignore_errors`), T-40-04 (prefix-checked teardown, unit test), T-40-05 (`raise`, no `assert`, `-O` and `PYTHONOPTIMIZE=1` subprocess runs).

## Next Phase Readiness

40-02 can build its guard on the helper's names and on the output lines (`Resolved database:`, `Removed scratch database directory:`) and convert the other four notebooks the old guard still names. The old guard and `test_projector_demo_notebook` stay green; the four notebooks they name are byte-identical to the starting commit. CLAUDE.md's map line says "all eight" and becomes literally true after 40-02.

## Self-Check: PASSED

- `docs/notebooks/fomo_nb_utils.py`, `solsys_code/tests/test_notebook_helper.py` exist and are tracked.
- Commits `55e51e8` and `c760c61` are ancestors of HEAD; `git rev-list --count 4aa9d7e..HEAD` was 2.
- Helper tests, `test_pre_executed_notebooks` and `test_projector_demo_notebook`: 40 tests OK; both commits passed the full django-test hook.

---
*Phase: 40-notebook-isolation-attribution-page*
*Completed: 2026-10-09*
