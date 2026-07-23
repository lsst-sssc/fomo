---
phase: quick-260722-uyz
plan: 01
subsystem: telescope-runs-calendar
tags: [django, tom_calendar, tom_targets, campaign, sync_lco_observation_calendar]

# Dependency graph
requires:
  - phase: quick-260722-tkt/twe
    provides: backfill_lco_observation_records --create-missing-targets, which adds a Target to a campaign TargetList
provides:
  - CalendarEvent.target_list populated by sync_lco_observation_calendar for LCO/SOAR records whose Target belongs to a campaign TargetList
affects: [calendar-ui, campaign-views]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Deterministic pick-first-by-name for a nullable FK derived from a reverse M2M accessor (record.target.targetlist_set.order_by('name').first())"

key-files:
  created: []
  modified:
    - solsys_code/management/commands/sync_lco_observation_calendar.py
    - solsys_code/tests/test_sync_lco_observation_calendar.py
    - docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb

key-decisions:
  - "target_list is derived once per record in _build_event_fields() via record.target.targetlist_set.order_by('name').first() — deterministic alphabetically-first pick on multi-membership, None (safe, nullable FK) on zero-membership; handle() needed no change since the new field key rides through the existing pass-through path"

patterns-established:
  - "FK-valued fields participate in the existing _update_or_unchanged() no-churn diff (getattr(event, f) != v) without any special-casing, since Django model __eq__ compares by (class, pk)"

requirements-completed: []

coverage:
  - id: D1
    description: "CalendarEvent.target_list is populated from the record's Target's single TargetList membership"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_sync_lco_observation_calendar.py#test_target_list_01_single_membership_sets_target_list"
        status: pass
    human_judgment: false
  - id: D2
    description: "CalendarEvent.target_list is None (no crash) when the Target belongs to zero TargetLists"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_sync_lco_observation_calendar.py#test_target_list_02_zero_membership_sets_none_no_crash"
        status: pass
    human_judgment: false
  - id: D3
    description: "Multi-membership picks the alphabetically-first-by-name TargetList deterministically"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_sync_lco_observation_calendar.py#test_target_list_03_multi_membership_picks_alphabetically_first"
        status: pass
    human_judgment: false
  - id: D4
    description: "Re-syncing an unchanged record with a matching target_list reports 'unchanged', not 'updated' (no-churn preserved for the new FK field)"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_sync_lco_observation_calendar.py#test_target_list_04_no_churn_on_unchanged_fk_field"
        status: pass
    human_judgment: false
  - id: D5
    description: "Paired demo notebook demonstrates a populated target_list with committed executed output"
    verification:
      - kind: manual_procedural
        ref: "jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb"
        status: pass
    human_judgment: false

duration: 20min
completed: 2026-07-22
status: complete
---

# Quick Task 260722-uyz: Populate CalendarEvent.target_list in sync_lco_observation_calendar Summary

**`sync_lco_observation_calendar` now derives `CalendarEvent.target_list` from the synced record's Target's campaign TargetList membership (deterministic alphabetically-first pick, or None), closing a gap present since the command's Phase 04 implementation.**

## Performance

- **Duration:** 20 min
- **Started:** 2026-07-22T22:29:00-07:00 (approx.)
- **Completed:** 2026-07-22T22:37:00-07:00
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- `_build_event_fields()` derives `target_list` via `record.target.targetlist_set.order_by('name').first()` — an existing command function shared by both the LCO and SOAR sync branches, so the fix applies uniformly to both.
- Added four regression tests covering one/zero/two-TargetList membership and no-churn idempotency on the new FK field, alongside all 39 pre-existing tests (43 total, all green).
- Regenerated the mandated paired demo notebook (`sync_lco_observation_calendar_demo.ipynb`) with a re-runnable demo `TargetList` ('Demo Campaign') and executed output showing the populated `target_list` on both the single-event inspection cell and the all-events summary loop.

## Task Commits

Each task was committed atomically:

1. **Task 1: Populate target_list in _build_event_fields()** - `7b1e873` (feat)
2. **Task 2: Add TargetList-membership tests** - `70e5bd3` (test)
3. **Task 3: Update paired demo notebook + quality gate** - `ac5f0ac` (docs)

**Plan metadata:** committed separately by the orchestrator per this run's constraints.

## Files Created/Modified
- `solsys_code/management/commands/sync_lco_observation_calendar.py` - `_build_event_fields()` now returns a `target_list` key derived from the record's Target's TargetList membership; docstring updated
- `solsys_code/tests/test_sync_lco_observation_calendar.py` - 4 new tests (one/zero/two-TargetList membership, no-churn on the FK field)
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` - demo TargetList fixture added; event-inspection cells print `target_list`; regenerated with executed output

## Decisions Made
- `target_list` derivation lives in `_build_event_fields()`, not `handle()` — the existing pass-through (`fields` dict flows unchanged into `insert_or_create_calendar_event`) already carries any new CalendarEvent field key, so `handle()` needed zero changes.
- No UI warning is surfaced for the multi-TargetList case; the deterministic alphabetically-first pick is the accepted behavior per the plan.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Applied 6 pending Django migrations to the local dev DB (solsys_code 0002-0007)**
- **Found during:** Task 3 (notebook re-execution)
- **Issue:** The local dev SQLite DB was missing 6 already-committed `solsys_code` migrations (`0002_campaignrun` through `0007_campaignrun_contact_public_opt_in`), causing `OperationalError: no such table: solsys_code_campaignrun` when the notebook's Django ORM calls ran against the dev DB.
- **Fix:** Ran `python manage.py migrate solsys_code` to bring the dev DB schema in sync with already-committed migration files. No new migration files were created — this only applied existing, already-merged migrations to local dev state.
- **Files modified:** none (DB schema only, not tracked in git)
- **Verification:** `python manage.py showmigrations solsys_code` shows all 7 migrations applied; notebook execution proceeded past the error.
- **Committed in:** N/A (local DB state, not a git change)

**2. [Rule 3 - Blocking] Fixed a pre-existing stale mock-patch target in the demo notebook**
- **Found during:** Task 3 (notebook re-execution)
- **Issue:** Three notebook cells (Phase 07/07.1 fallback-label demos) patched `solsys_code.management.commands.sync_lco_observation_calendar.make_request`, an attribute that no longer exists on that module — `make_request` was relocated into `solsys_code/calendar_utils.py` during a prior extraction refactor, but this notebook's mock-patch targets were never updated to match, so re-execution raised `AttributeError`. This was blocking the required `jupyter nbconvert --execute --inplace` verification step for this task and was unrelated to the target_list feature itself.
- **Fix:** Updated all 3 patch-target strings to `solsys_code.calendar_utils.make_request`, matching the convention already used by `solsys_code/tests/test_sync_lco_observation_calendar.py`.
- **Files modified:** `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`
- **Verification:** Notebook re-executes cleanly end-to-end; `ruff check`/`ruff format --check` clean on the notebook.
- **Committed in:** `ac5f0ac` (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 3 - blocking issues preventing Task 3's mandated verification from completing; neither touched the target_list feature code)
**Impact on plan:** Both fixes were necessary to complete the plan's own verification requirement (successful notebook re-execution). No scope creep beyond what was required to unblock Task 3.

## Issues Encountered
- An initial broad `ruff check . --fix` (run before scoping was tightened to the plan's 3 files) auto-reformatted an import in the unrelated `load_telescope_runs_demo.ipynb`. This was caught before committing and reverted via `git checkout -- docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`, keeping the final diff scoped to this plan's `files_modified`.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CalendarEvents synced from LCO/SOAR queue records now surface their campaign association wherever `CalendarEvent.target_list` is read (e.g. `views.py`'s `target_lists` filter, campaign calendar views).
- No blockers. The fix is narrowly scoped and covered by regression tests; the repo-wide `ruff check .` still reports 3 pre-existing, out-of-scope findings in `load_telescope_runs_demo.ipynb` and `sync_gemini_observation_calendar_demo.ipynb`, both untouched by this task.

---
*Phase: quick-260722-uyz*
*Completed: 2026-07-22*

## Self-Check: PASSED

All 3 modified files found on disk; all 3 task commit hashes (`7b1e873`, `70e5bd3`, `ac5f0ac`) found in git history.
