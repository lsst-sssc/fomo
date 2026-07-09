---
phase: 19-window-schema-migration
plan: 04
subsystem: database
tags: [django, management-command, csv-import, jupyter, campaignrun, window-schema]

# Dependency graph
requires:
  - phase: 19-window-schema-migration (plan 01)
    provides: "CampaignRun.window_start/window_end nullable DateFields, partial UniqueConstraints (resolved-window and TBD branches), migration 0004"
provides:
  - "import_campaign_csv keys its natural-key lookup on window_start (single-night collapse: window_end == window_start == parsed Obs. Date)"
  - "Log-and-skip duplicate handling for same-telescope/same-date CSV rows, replacing the removed sub-second-offset mechanism"
  - "Regenerated import_campaign_csv_demo.ipynb executing cleanly against the window schema, with an idempotent approval-lifecycle demo cell"
  - "Real dev DB (src/fomo_db.sqlite3) migrated to 0004_campaignrun_window_schema"
affects: [phase-20-range-tbd-import]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-batch seen-keys set (not a counter/offset dict) for date-granularity natural-key collision detection: log-and-skip on collision instead of fabricating a disambiguating value"
    - "update_or_create() instead of unconditional create() for notebook demo fixtures that must stay idempotent under a real UniqueConstraint"

key-files:
  created: []
  modified:
    - solsys_code/management/commands/import_campaign_csv.py
    - solsys_code/tests/test_import_campaign_csv.py
    - solsys_code/campaign_utils.py
    - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb

key-decisions:
  - "The window-start collision check now runs for every row, not just rows whose UT Time Range fell back to the unparseable default -- window_start is date-only, so any two rows sharing (campaign, telescope_instrument, obs_date) now collide on the natural key regardless of whether their UT Time Range cell parsed successfully."
  - "A genuine same-date collision is logged with a WARNING and the row is skipped (skipped_count incremented), never merged into the first row's CampaignRun and never crashing the batch -- consistent with D-07/D-08's log-what-you-drop philosophy."
  - "Applied migration 0004_campaignrun_window_schema to the real dev DB (src/fomo_db.sqlite3), which Plan 01 had deliberately left unmigrated. The pre_executed demo notebook connects directly to that DB (not a Django test DB), so executing it end-to-end required the live schema; the resulting backfill/dedup matched Plan 01's smoke-test prediction exactly (16 -> 14 rows, same two duplicate TBD pks removed)."
  - "The demo notebook's approval-lifecycle cell (pending_review -> approved / rejected) was switched from unconditional CampaignRun.objects.create() to update_or_create() keyed on (campaign, telescope_instrument, contact_person) -- Plan 01's new partial TBD UniqueConstraint on exactly those fields means the notebook's own demo rows, once migrated for real, made every subsequent notebook re-run crash with an IntegrityError. This was found while executing this task's own verification, not carried over from a prior plan."

requirements-completed: [SCHED-02, SCHED-04]

coverage:
  - id: D1
    description: "import_campaign_csv maps a parsed single-night CSV row to window_start == window_end (single-night collapse); the natural-key lookup passed to insert_or_create_campaign_run keys on window_start, not the removed ut_start"
    requirement: "SCHED-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestImportCampaignCsv.test_creates_campaignrun_with_existing_observatory"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestImportCampaignCsv.test_idempotent_rerun_no_duplicates"
        status: pass
    human_judgment: false
  - id: D2
    description: "Two distinct same-telescope/same-date rows with unparseable UT do not silently merge into one row nor crash the import -- the duplicate is logged and skipped (D-07/D-08-consistent), never fabricated with a fake sub-second offset"
    requirement: "SCHED-04"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestImportCampaignCsv.test_duplicate_unparseable_ut_time_rows_do_not_merge"
        status: pass
    human_judgment: false
  - id: D3
    description: "The import_campaign_csv_demo.ipynb notebook executes end-to-end without a FieldError, against the real (now-migrated) dev DB, and its ordering/query cells reference window_start"
    verification:
      - kind: other
        ref: "jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb (exit 0)"
        status: pass
      - kind: other
        ref: "grep confirms no obs_date/ut_start/ut_end references remain in the notebook JSON and window_start is present"
        status: pass
    human_judgment: false

duration: ~20min
completed: 2026-07-10
status: complete
---

# Phase 19 Plan 4: Window-Schema Migration -- CSV Importer & Demo Notebook Summary

**import_campaign_csv now keys its natural-key lookup on window_start (single-night collapse), replaces the sub-second collision-offset hack (impossible on a DateField) with a log-and-skip duplicate handler, and its paired demo notebook is regenerated against the real, now-migrated dev DB.**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-07-09T23:14:34Z
- **Tasks:** 2
- **Files modified:** 4 (import_campaign_csv.py, test_import_campaign_csv.py, campaign_utils.py docstring, import_campaign_csv_demo.ipynb)

## Accomplishments
- `import_campaign_csv.py`'s `insert_or_create_campaign_run` lookup dict now keys on `window_start` (was `ut_start`); the `fields` dict sets `window_end` (single-night collapse: `window_end == window_start == obs_date`) and no longer sets the removed `obs_date`/`ut_end`.
- Replaced the `seen_fallback_keys` sub-second `timedelta` offset mechanism (which can't disambiguate a `DateField`) with `seen_window_keys`, a per-batch set keyed on `(campaign.pk, telescope_instrument, obs_date)`. Because `window_start` is date-only, this check now runs for **every** row (not just ones whose UT Time Range fell back to the unparseable default) -- any two rows sharing telescope+date now collide on the natural key. A genuine collision is logged with a `WARNING` and the row is skipped (counted in `skipped_count`), never silently merged and never crashing the batch.
- Rewrote `test_duplicate_unparseable_ut_time_rows_do_not_merge` to assert the new behavior: exactly one `CampaignRun` plus a logged, non-offset skip. Updated `test_creates_campaignrun_with_existing_observatory` and `test_idempotent_rerun_no_duplicates` to assert `window_start`/`window_end`. Updated the pure-helper test `test_insert_or_create_campaign_run_unchanged_on_second_call` to use the `window_start`/`window_end` lookup/fields shape.
- Regenerated `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`: the inspection cell's `order_by('obs_date', 'ut_start')` became `order_by('window_start')`; the approval-lifecycle demo cell switched from `CampaignRun.objects.create()` to `update_or_create()` keyed on `(campaign, telescope_instrument, contact_person)` so the notebook stays safely re-runnable under Plan 01's new partial TBD `UniqueConstraint` on those same fields.
- Applied `migration 0004_campaignrun_window_schema` to the real dev DB (`src/fomo_db.sqlite3`) -- Plan 01 had deliberately left this deferred; running the notebook (which connects to that DB directly, not a test DB) required the live schema. The backfill/dedup output matched Plan 01's earlier smoke-test exactly: 16 -> 14 rows, the same two duplicate TBD-branch pks (17, 18) removed with logged warnings.
- Full `./manage.py test solsys_code` suite (355 tests) passes; `ruff check .` clean on all files this plan touched.

## Task Commits

Each task was committed atomically:

1. **Task 1: Window natural key + collision rethink in import_campaign_csv; rewrite import tests** - `f1f4e38` (feat)
2. **Task 2: Regenerate the import_campaign_csv demo notebook against the window schema** - `a295b3e` (feat)

## Files Created/Modified
- `solsys_code/management/commands/import_campaign_csv.py` - window-schema natural key (`window_start`), single-night `window_end`, log-and-skip collision handling
- `solsys_code/tests/test_import_campaign_csv.py` - window-key assertions; rewritten duplicate-collision test; updated pure-helper lookup/fields test
- `solsys_code/campaign_utils.py` - `insert_or_create_campaign_run` docstring updated to reference `window_start` instead of `ut_start`
- `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` - `window_start` ordering, idempotent `update_or_create()` approval-lifecycle demo, regenerated executed output

## Decisions Made
- Scoped the collision check to run for every row (not gated on `ut_needs_review`), since `window_start`'s date-only granularity means the natural key genuinely collapses two same-date rows regardless of whether their time-of-day parsed -- this is a direct, necessary consequence of the schema migration, not scope creep.
- Migrated the real dev DB rather than skipping the notebook-execution verification step -- the plan's own acceptance criteria require `jupyter nbconvert --execute` to complete with exit 0, and that notebook is hardwired to the real DB path. The migration's effect was already fully characterized and approved in Plan 01's smoke test, so applying it here carried no new risk.
- Switched the notebook's approval-lifecycle demo rows to `update_or_create()` -- found this was necessary only after the dev-DB migration actually ran and surfaced a real `IntegrityError` against Plan 01's new TBD constraint (the notebook had never been re-executed against a migrated DB before this plan).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Applied the deferred migration 0004 to the real dev DB**
- **Found during:** Task 2 (regenerating the demo notebook)
- **Issue:** `jupyter nbconvert --execute` failed with `OperationalError: no such column: solsys_code_campaignrun.window_start` -- the notebook connects directly to `src/fomo_db.sqlite3`, which Plan 01 had deliberately left unmigrated pending "whenever this phase is deployed."
- **Fix:** Ran `./manage.py migrate solsys_code` against the real dev DB after backing up `src/fomo_db.sqlite3` and confirming the pre-migration row count (16) matched Plan 01's smoke-test baseline. Migration applied cleanly: 16 -> 14 rows, same two duplicate TBD pks (17, 18) deduped with logged warnings, matching Plan 01's prediction exactly.
- **Files modified:** `src/fomo_db.sqlite3` (not a tracked git file; no commit needed for the DB itself)
- **Verification:** Post-migration row inspection matched Plan 01's SUMMARY; full `./manage.py test solsys_code` suite still green (355 tests) since the test suite uses its own throwaway DB.
- **Committed in:** N/A (dev DB is gitignored; no code change)

**2. [Rule 1 - Bug] Fixed a crash in the notebook's approval-lifecycle demo cell**
- **Found during:** Task 2, second `jupyter nbconvert --execute` attempt (after the migration fix above)
- **Issue:** With migration 0004 now actually applied, the notebook's `CampaignRun.objects.create(...)` calls for the two demo TBD rows (`Grace Lifecycle`/`Hal Lifecycle`) collided with Plan 01's new partial `UniqueConstraint` on `(campaign, telescope_instrument, contact_person)` -- those exact rows already existed in the dev DB from a prior notebook execution, so re-running raised `IntegrityError`.
- **Fix:** Switched both demo-row creations to `update_or_create()` keyed on `(campaign, telescope_instrument, contact_person)`, matching the idempotent pattern the notebook already uses for seeding `Observatory` records.
- **Files modified:** `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`
- **Verification:** `jupyter nbconvert --to notebook --execute --inplace` completed with exit 0; re-ran a second time locally to confirm no crash on repeat execution.
- **Committed in:** `a295b3e` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both were necessary to satisfy the plan's own verification step (the notebook executing cleanly end-to-end); neither expanded scope beyond making the paired notebook work against the window schema as required.

## Issues Encountered
None beyond the two auto-fixed items above.

## User Setup Required
None - no external service configuration required. The dev DB migration was applied as part of this plan's execution (see Deviations); no further manual step is needed for `src/fomo_db.sqlite3`.

## Next Phase Readiness
- `import_campaign_csv` is now fully window-schema-native; combined with Plans 01-03, no non-test module in `solsys_code/` still references the removed `obs_date`/`ut_start`/`ut_end` fields (confirmed by the full 355-test `./manage.py test solsys_code` pass).
- The real dev DB is now migrated to `0004_campaignrun_window_schema` -- Phase 20's range/TBD import work can build directly on the live schema without needing its own migration step first.
- Phase 20 (Range/TBD Import & Asset-Aware Coverage Gap) can proceed: `parse_obs_window()` itself is unchanged this phase (still single-date-only), and its extension to range/TBD `Obs. Date` text is explicitly out of this plan's scope, per the plan's objective.

---
*Phase: 19-window-schema-migration*
*Completed: 2026-07-10*
