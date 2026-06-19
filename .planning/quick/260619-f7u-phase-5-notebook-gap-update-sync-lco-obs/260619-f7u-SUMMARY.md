---
phase: quick-260619-f7u
plan: 01
subsystem: docs
tags: [jupyter-notebook, sync_lco_observation_calendar, multi-proposal, multi-facility, soar, documentation]

# Dependency graph
requires:
  - phase: 05-multi-proposal-multi-facility-selection
    provides: sync_lco_observation_calendar comma-list/ALL proposal parsing and LCO+SOAR dispatch (SELECT-02/03/04/05)
provides:
  - Phase 5 demo coverage (SELECT-02/03/04/05) appended to the existing Stage 3 sync_lco_observation_calendar_demo.ipynb
  - Stage-vs-Phase numbering clarification note in PROJECT.md
affects: [06-correct-instrument-type-extraction, 07-telescope-label-resolution]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Demo notebooks accumulate phase-by-phase: new requirement coverage is appended as a new markdown+code section rather than a new notebook file, with the Summary table and teardown cell updated in place"

key-files:
  created: []
  modified:
    - docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb
    - .planning/PROJECT.md

key-decisions:
  - "SELECT-05 (per-record facility-instance dispatch) is documented in the notebook as a markdown-only pointer to the existing discriminating spy test rather than attempted via unittest.mock inside the notebook, because LCOFacility/SOARFacility.get_observation_url() return byte-identical strings and cannot be black-box-distinguished"
  - "Pre-existing 'upsert' DB jargon in PROJECT.md's Key Decisions table (predating this task) was also replaced with plain English ('create-or-update'), since the verify_project.py gate checks the whole file and CLAUDE.md's plain-English convention is file-wide, not just net-new content"

patterns-established:
  - "Quick-task notebook extensions insert new sections after the last existing demo section and before the Summary/Teardown cells, reusing already-in-scope fixtures (demo_target/demo_user) rather than re-running Django setup boilerplate"

requirements-completed: [SELECT-02, SELECT-03, SELECT-04, SELECT-05]

# Metrics
duration: 33min
completed: 2026-06-19
---

# Quick Task 260619-f7u: Phase 5 Notebook Gap + Stage/Phase Doc Summary

**Extended the Stage 3 `sync_lco_observation_calendar_demo.ipynb` notebook with Phase 5 multi-proposal/multi-facility demo cells (SELECT-02/03/04/05) and added a Stage-vs-Phase numbering clarification note to PROJECT.md.**

## Performance

- **Duration:** 33 min
- **Started:** 2026-06-19T17:36:00Z (approx, from worktree base commit)
- **Completed:** 2026-06-19T18:09:58Z
- **Tasks:** 2 completed
- **Files modified:** 2

## Accomplishments

- The existing `sync_lco_observation_calendar_demo.ipynb` now demonstrates comma-list `--proposal A,B` selection with a substring-decoy record (`PHASE5-AB`) that correctly gets no event (SELECT-02)
- The notebook demonstrates `--proposal all` (lowercase) syncing every fixture record regardless of proposal or facility, including a SOAR record (SELECT-03)
- The notebook demonstrates a single `call_command` invocation producing `CalendarEvent`s for both an LCO and a SOAR record together, with the per-facility `LCO:`/`SOAR:` summary line printed (SELECT-04, D-08)
- A markdown-only cell documents that SELECT-05 (no shared facility instance between LCO/SOAR) is not black-box-demonstrable in a notebook and points to the discriminating spy test instead
- The notebook's Summary table now lists all ten requirements (SELECT-01/02/03/04/05, SYNC-01..04, TERM-01) in the original column format
- The notebook's teardown cell removes every newly-created Phase-5 `CalendarEvent`/`ObservationRecord` fixture, in addition to the existing Phase-4 cleanup
- `.planning/PROJECT.md` documents that "Stage" (issue #37 feature grouping) and "Phase" (GSD execution count) are intentionally different numbering granularities, with the Stage 2→Phases 2-3 and Stage 3→Phase 4 (extended by Phases 5-7) mappings spelled out

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend the demo notebook with SELECT-02/03/04/05 cells, updated Summary table, and teardown** - `50bb13b` (docs)
2. **Task 2: Document the Stage-vs-Phase numbering mapping in PROJECT.md** - `540c967` (docs)
3. **Fix-up: add explicit `strict=` to `zip()` in the SELECT-02 demo cell** - `fa17e2c` (fix, Rule 1 — ruff B905)

**Plan metadata:** committed separately by the orchestrator after this summary is written.

## Files Created/Modified

- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` - Appended 8 new cells (4 markdown, 4 code/markdown-only mix per spec) covering SELECT-02/03/04/05, updated the Summary requirements table (4 new rows), and updated the teardown cell to remove all new Phase-5 fixtures alongside the existing Phase-4 cleanup
- `.planning/PROJECT.md` - Added a "Stage vs Phase numbering" subsection immediately before "## Demo Notebooks"; updated the "Last updated" footer line; replaced one pre-existing "upsert" occurrence with plain English

## Decisions Made

- Inserted the new Phase 5 section after the existing "No-churn idempotency" cell (`a3b4c5d3`) and before the existing "## Summary" markdown cell (`b4c5d6e4`), per the plan's explicit ordering instruction
- Reused the existing `demo_target`/`demo_user` fixtures (no new `Target`/`User` rows) to keep the notebook's existing Django-setup boilerplate in scope and avoid introducing a second `NonSiderealTargetFactory` fixture unnecessarily
- Chose the "new subsection immediately preceding Demo Notebooks" placement for the Stage-vs-Phase note (rather than appending to the end of Demo Notebooks), since it reads more naturally as project-level context ahead of the notebook listing

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking/lint] `ruff check` B905 flagged the new `zip()` call without `strict=`**
- **Found during:** Task 1 post-commit lint pass (`ruff check .`)
- **Issue:** The new SELECT-02 cell used `zip(demo_phase5_select02_ids, select02_proposals)` without `strict=`, which `ruff`'s `B905` rule flags as missing an explicit length-mismatch guard
- **Fix:** Added `strict=True` (the two lists are always the same length by construction)
- **Files modified:** `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`
- **Verification:** `ruff check docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` no longer reports B905; `verify_nb.py` still passes
- **Committed in:** `fa17e2c`

**2. [Rule 2 - missing critical functionality / CLAUDE.md compliance] Pre-existing "upsert" jargon in PROJECT.md**
- **Found during:** Task 2 verification (`verify_project.py` failed on a whole-file "no upsert" check)
- **Issue:** A pre-existing Key Decisions table row (predating this quick task, from Phase 03) used "upsert" — DB jargon that CLAUDE.md's planning-doc terminology convention asks GSD subagents to avoid in `.planning/` artifacts
- **Fix:** Replaced "CalendarEvent upsert keyed on..." with "CalendarEvent create-or-update keyed on..." — no semantic change, same row otherwise
- **Files modified:** `.planning/PROJECT.md`
- **Verification:** `verify_project.py` passes (no "upsert" in file, case-insensitive)
- **Committed in:** `540c967` (same commit as the rest of Task 2, since both touch PROJECT.md)

---

**Total deviations:** 2 auto-fixed (1 Rule 3 lint fix, 1 Rule 2/CLAUDE.md compliance fix)
**Impact on plan:** Both fixes were small, scoped to the files this task already touches, and necessary to keep `ruff check .` clean and PROJECT.md compliant with the project's plain-English documentation convention. No scope creep.

## Issues Encountered

- The worktree's `src/fomo/_version.py` is missing (setuptools_scm dev-install artifact gap), which blocks `./manage.py migrate`/`./manage.py test` from running directly in this worktree shell. This pre-dates the quick task and is unrelated to the notebook/PROJECT.md content changes, so per the deviation rules' scope boundary it was left unfixed and not investigated further. The pre-commit hook's own "Run unit tests" step passed on both commits, confirming the project's actual test environment (used by pre-commit) is healthy — only this particular worktree shell's ad-hoc `manage.py` invocation was affected.
- Manual end-to-end notebook execution (Run All in Jupyter) was not performed, consistent with the plan's `<verification>` section marking this as "Optional manual verification" with "no automated test suite for notebook content." The structural gate (`verify_nb.py`) and `ruff check`/`ruff format --check` were used as the automated correctness signal instead, supplemented by close comparison against the already-passing `test_select_02_comma_list_matches_any_no_substring_leakage`, `test_select_03_all_token_case_insensitive_syncs_everything`, `test_select_04_single_run_covers_both_facilities`, and `test_select_05_soar_record_uses_soar_facility_instance` test bodies (same fixture shapes, same command invocation pattern) in `solsys_code/tests/test_sync_lco_observation_calendar.py`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- This quick task closes the Phase 5 documentation gap; Phase 6 (correct-instrument-type-extraction) and Phase 7 (telescope-label resolution) are unaffected by these changes and remain ready to start per `.planning/STATE.md`'s existing "Operator Next Steps."
- No blockers introduced. The PROJECT.md Stage-vs-Phase note should make future phase transitions (Phase 6/7, which extend Stage 3) easier to communicate without re-litigating the numbering confusion.

---
*Phase: quick-260619-f7u*
*Completed: 2026-06-19*

## Self-Check: PASSED

- FOUND: docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb
- FOUND: .planning/PROJECT.md
- FOUND commit: 50bb13b
- FOUND commit: 540c967
- FOUND commit: fa17e2c
