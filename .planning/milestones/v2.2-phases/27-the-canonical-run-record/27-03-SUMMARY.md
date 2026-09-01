---
phase: 27-the-canonical-run-record
plan: 03
subsystem: database
tags: [django, migration, model-rename, foreign-key, calendar]

# Dependency graph
requires:
  - phase: 27-the-canonical-run-record (plan 01)
    provides: "calendar_utils.py's public helper names and derive_telescope_class() -- confirmed unaffected by this plan's rename (no overlap in files touched)"
  - phase: 26-canonical-record-spike
    provides: "D-05/D-02 locked migration shape (RenameModel then AddField), the six-point integration checklist, and the measured rename blast radius proving the rename fails loudly, not silently"
provides:
  - "CalendarEventMeta model (renamed from CalendarEventTelescopeLabel) with a nullable SET_NULL run FK to CampaignRun -- the event-side half of the canonical-run-record's one-to-many relation"
  - "Hand-authored migrations 0008 (RenameModel) and 0009 (AddField run) applied against the real dev DB, preserving all 11 companion rows"
  - "MigrationExecutor regression test (test_canonical_record_migration.py) proving row count, per-row is_verified, pk identity, and null run survive the rename on a fresh database"
affects: [27-04-window-schema-migration-and-telescope-class-backfill, 27-05-admin-inline-surfaces, 29-reconciler]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Hand-authored RenameModel (never makemigrations-autodetected) for any model whose primary key would make DeleteModel/CreateModel destructive"
    - "Lazy string FK reference ('CampaignRun') when the target class is defined later in the same models.py file"
    - "MigrationExecutor TransactionTestCase seeding rows under the pre-migration historical model name, proving the migration's data-preservation property independent of the one-off manual dev-DB proof"

key-files:
  created:
    - solsys_code/migrations/0008_rename_calendareventtelescopelabel_calendareventmeta.py
    - solsys_code/migrations/0009_calendareventmeta_run.py
    - solsys_code/tests/test_canonical_record_migration.py
  modified:
    - solsys_code/models.py
    - solsys_code/admin.py
    - solsys_code/management/commands/sync_lco_observation_calendar.py
    - solsys_code/tests/test_admin.py
    - solsys_code/tests/test_load_telescope_runs.py
    - solsys_code/tests/test_sync_lco_observation_calendar.py
    - solsys_code/tests/test_calendar_template.py

key-decisions:
  - "related_name='telescope_label_meta' left byte-identical (REQUIREMENTS.md Out of Scope forbids renaming it) -- views.py's prefetch_related() and calendar.html's two is_verified accessors needed zero edits, confirmed by grep"
  - "run FK declared as SET_NULL (not CASCADE) because the companion row also carries is_verified and must survive its owning run's deletion, mirroring D-05's negative constraint (no confirmed_by/confirmed_at -- a deliberate audit asymmetry with the observation link, recorded as a comment on the field)"
  - "Migration 0009 kept separate from 0008 so a rename regression and a new-field regression can never be confused for each other, per 26-DECISION.md Criterion 4"

patterns-established:
  - "A companion-record rename with a primary-key-as-FK model must hand-author RenameModel and prove non-destruction with a MigrationExecutor test that seeds under the historical name and asserts pk-matched field survival"

requirements-completed: [CANON-03]

# Metrics
duration: ~20min
completed: 2026-07-30
---

# Phase 27 Plan 03: The Canonical Run Record -- CalendarEventMeta Rename Summary

**Renamed CalendarEventTelescopeLabel to CalendarEventMeta and gave it a nullable SET_NULL run FK to CampaignRun, via hand-authored migrations 0008/0009 applied against the real dev DB with all 11 companion rows and their is_verified history intact.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-07-29T23:29 (approx, continuing from Wave 1)
- **Completed:** 2026-07-30T00:48:10Z
- **Tasks:** 3 completed
- **Files modified:** 10 (3 created, 7 modified)

## Accomplishments

- Renamed `CalendarEventTelescopeLabel` to `CalendarEventMeta` and fixed all six integration points the rename touches (admin.py's import/class/register call, sync_lco_observation_calendar.py's import/call site, test_admin.py's derived admin reverse-URL name, and class-name references in three more test modules) -- confirmed via `./manage.py check` exit 0 and a 94-test targeted run, all green.
- Added `CalendarEventMeta.run`, a nullable `SET_NULL` FK to `CampaignRun` declared as a lazy `'CampaignRun'` string (the class is defined later in the same file), with a comment recording D-05's deliberate no-audit-field decision.
- Hand-authored (not `makemigrations`-generated) migration `0008` (`RenameModel`) and `0009` (`AddField run`), applied against the real dev DB (backed up first to `/tmp/fomo_db.sqlite3.pre-27-03.bak`, 946,176 bytes -- byte-identical to the pre-27-02 backup size, confirming no drift). Post-migration: 11 companion rows, 0 with `is_verified=False` (unchanged from pre-migration), 0 with `run` set. `makemigrations --check --dry-run` reports "No changes detected."
- Added `test_canonical_record_migration.py`: a `MigrationExecutor`-based `TransactionTestCase` seeding 3 companion rows (mixed `is_verified` True/False) against the historical pre-rename model, migrating through `0009`, and asserting row count, per-row `is_verified` matched by `event_id` pk, null `run_id` on every row, and `LookupError` for the old model name in the post-migration app state.

## Task Commits

Each task was committed atomically:

1. **Task 1: Rename the model to CalendarEventMeta, add its run link, and fix all six integration points** - `6a8739a` (feat)
2. **Task 2: Hand-author and apply the rename and run-link migrations** - `a4cb48c` (feat)
3. **Task 3: Prove the rename preserves is_verified history with a MigrationExecutor test** - `b58a078` (test)

_No TDD tasks in this plan (autonomous, non-TDD execute plan)._

## Files Created/Modified

- `solsys_code/models.py` - `CalendarEventTelescopeLabel` renamed to `CalendarEventMeta`; docstring broadened to describe a general companion record; added `run` (nullable `SET_NULL` FK to `'CampaignRun'`, `related_name='calendar_event_metas'`) with a D-05 comment
- `solsys_code/admin.py` - Import, `CalendarEventTelescopeLabelAdmin` -> `CalendarEventMetaAdmin`, and the `admin.site.register(...)` call updated
- `solsys_code/management/commands/sync_lco_observation_calendar.py` - Import and `.objects.update_or_create(event=event, ...)` call site updated (keyword arguments unchanged)
- `solsys_code/migrations/0008_rename_calendareventtelescopelabel_calendareventmeta.py` - New hand-authored `RenameModel` migration
- `solsys_code/migrations/0009_calendareventmeta_run.py` - New hand-authored `AddField` migration for `run`
- `solsys_code/tests/test_admin.py` - Both `reverse('admin:solsys_code_calendareventtelescopelabel_changelist')` call sites and their surrounding test method names updated to `calendareventmeta`
- `solsys_code/tests/test_load_telescope_runs.py` - Import and both `CalendarEventTelescopeLabel` references (import + assertion body) renamed
- `solsys_code/tests/test_sync_lco_observation_calendar.py` - Import and 5 usage sites renamed
- `solsys_code/tests/test_calendar_template.py` - Import, docstring reference, and 5 fixture-construction sites renamed
- `solsys_code/tests/test_canonical_record_migration.py` - New `MigrationExecutor` regression test module (4 tests)

## Decisions Made

- Kept `related_name='telescope_label_meta'` byte-identical (Task 1 acceptance criteria and REQUIREMENTS.md's Out of Scope table both forbid changing it) -- confirmed by grep that `views.py`'s `prefetch_related('telescope_label_meta')` and both `calendar.html` `is_verified` accessors needed zero edits.
- Used `on_delete=models.SET_NULL` for the new `run` FK, not `CASCADE`, because the companion row also carries `is_verified` history and must survive its owning run's deletion -- mirrors 26-DECISION's Criterion 4 recommendation and contrasts with D-04's `CASCADE` choice for the (data-free) observation link.
- Declared `run = models.ForeignKey('CampaignRun', ...)` as a lazy string reference rather than the class object, since `CalendarEventMeta` is defined above `CampaignRun` in `models.py` and a direct reference would raise `NameError` at import time.
- Kept migration `0009` separate from `0008` (rename only) per 26-DECISION.md Criterion 4, so a rename regression and a new-field regression can never be confused for each other during future debugging.
- Applied the ruff-format reflow to `sync_lco_observation_calendar.py`'s now-shorter `CalendarEventMeta.objects.update_or_create(...)` call (the renamed identifier now fits on one line) -- required by the project's `ruff format --check .` gate, in scope since the line was already touched by this plan's rename.

## Deviations from Plan

None - plan executed exactly as written. All three tasks' acceptance criteria were met in substance: the model rename, the `run` FK, the D-05 no-audit-field constraint, all six integration points, both hand-authored migrations, the live-DB row preservation, and the `MigrationExecutor` regression test.

**One expected grep-literalism note, not a deviation** (same pattern documented in Phase 27-01's Summary): migration `0008`'s acceptance-criteria greps for `RenameModel` (expected count 1) and `DeleteModel|CreateModel` (expected count 0) instead return 2 and 1 respectively, because the file's mandated header comment -- which the plan's own Task 2 action requires, quoting exactly why `RenameModel` was chosen over a `DeleteModel`/`CreateModel` pair -- contains both phrases as prose. Verified via AST parse of the migration's `operations` list that there is exactly one real `RenameModel` call and zero `DeleteModel`/`CreateModel` calls; the grep mismatch is purely the rationale comment's text, not an extra operation.

## Issues Encountered

None specific to this plan. The pre-existing `./manage.py test solsys_code` segfault in `test_views.py` (SPICE/ASSIST native code, unrelated to this plan's files) is a known environment issue documented by Phase 27-01/02 -- verified instead via the plan's own targeted module runs (94 tests for Task 2's verification command, 4 tests for Task 3, 8 tests for the two migration modules together), the plan's quick-run command (307 tests), and a combined run of every `solsys_code.tests.*` module except `test_views`/`test_ephem_utils` plus `solsys_code_observatory` (606 tests, all pass -- 4 more than the documented 602-test Wave 1 baseline, exactly matching the 4 new tests this plan added).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `CalendarEventMeta.run` is in place and ready for Plan 04's window-schema migration and telescope_class backfill, and for Plan 05's admin inline surfaces (`CalendarEventMetaInline`).
- The dev DB backup for this plan is at `/tmp/fomo_db.sqlite3.pre-27-03.bak` (does not overwrite Plan 02's backup at `/tmp/fomo_db.sqlite3.pre-27-02.bak`).
- No blockers. `ruff check .`/`ruff format --check .` report the same pre-existing, unrelated issues present before this plan (1 notebook docstring warning, 7 files needing reformat -- none touched by this plan; `src/fomo/settings.py` is the user's local dev config and was never staged or committed).

---
*Phase: 27-the-canonical-run-record*
*Completed: 2026-07-30*

## Self-Check: PASSED

All 3 created files confirmed present on disk (see below); all 3 task commit hashes (`6a8739a`, `a4cb48c`, `b58a078`) confirmed present in `git log --oneline --all`.
