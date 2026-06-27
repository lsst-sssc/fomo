---
phase: 08-telescope-label-verification-sidecar
plan: 01
subsystem: database
tags: [django, orm, one-to-one-field, sidecar-model, migrations, jupyter]

# Dependency graph
requires:
  - phase: 07
    provides: "telescope_api_failed per-record signal computed in _build_event_fields() and popped before the existing CalendarEvent get_or_create write"
  - phase: 07.1
    provides: "facility-aware coarse fallback label (_coarse_telescope_label), confirming telescope_api_failed is the single shared verified/fallback signal for both LCO and SOAR"
provides:
  - "CalendarEventTelescopeLabel sidecar model (OneToOneField(primary_key=True) on tom_calendar.CalendarEvent) with reverse accessor event.telescope_label_meta"
  - "First real migration for the solsys_code app (0001_calendareventtelescopelabel.py)"
  - "Standalone update_or_create write in sync_lco_observation_calendar.py reconciling is_verified = not telescope_api_failed per record"
  - "Test coverage for verified/fallback/no-churn/no-row-for-classical-events"
  - "Regenerated demo notebook cell demonstrating all three sidecar outcomes"
affects: [08-02, phase-09-proposal-color-status]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "OneToOneField(primary_key=True) sidecar model to extend a third-party model (tom_calendar.CalendarEvent) without touching its migrations"
    - "Sidecar write kept as a standalone update_or_create statement, never folded into the existing fields dict / changed comparison (no-churn discipline)"

key-files:
  created:
    - solsys_code/migrations/0001_calendareventtelescopelabel.py
  modified:
    - solsys_code/models.py
    - solsys_code/management/commands/sync_lco_observation_calendar.py
    - solsys_code/tests/test_sync_lco_observation_calendar.py
    - solsys_code/tests/test_load_telescope_runs.py
    - docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb

key-decisions:
  - "Migration generated via `./manage.py makemigrations solsys_code --name calendareventtelescopelabel` (not the Django default `0001_initial.py`) so the file name matches the plan's must_haves artifact exactly, while keeping the generated content itself untouched/deterministic."

patterns-established:
  - "Sidecar write pattern: CalendarEventTelescopeLabel.objects.update_or_create(event=event, defaults={'is_verified': not telescope_api_failed}) placed immediately after the existing get_or_create/diff/save() block, inside the same per-record loop, never merged into fields/changed."

requirements-completed: [DISPLAY-01]

coverage:
  - id: D1
    description: "CalendarEventTelescopeLabel model + first migration for solsys_code app exist and migrate cleanly"
    requirement: "DISPLAY-01"
    verification:
      - kind: unit
        ref: "manage.py makemigrations solsys_code --check --dry-run (no changes detected)"
        status: pass
      - kind: unit
        ref: "manage.py migrate solsys_code (applies cleanly)"
        status: pass
    human_judgment: false
  - id: D2
    description: "sync_lco_observation_calendar writes is_verified=True for a live-verified record"
    requirement: "DISPLAY-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_sync_lco_observation_calendar.py#test_display_01_verified_record_creates_sidecar_row_is_verified_true"
        status: pass
    human_judgment: false
  - id: D3
    description: "sync_lco_observation_calendar writes is_verified=False for a fallback-labeled record"
    requirement: "DISPLAY-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_sync_lco_observation_calendar.py#test_display_01_fallback_record_creates_sidecar_row_is_verified_false"
        status: pass
    human_judgment: false
  - id: D4
    description: "Re-running sync on an unchanged record does not duplicate the sidecar row"
    requirement: "DISPLAY-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_sync_lco_observation_calendar.py#test_display_01_rerun_on_unchanged_record_no_duplicate_sidecar_row"
        status: pass
    human_judgment: false
  - id: D5
    description: "load_telescope_runs-created events have no sidecar row (reverse accessor raises DoesNotExist)"
    requirement: "DISPLAY-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_load_telescope_runs.py#test_display_01_no_sidecar_row_for_classically_scheduled_event"
        status: pass
    human_judgment: false
  - id: D6
    description: "Demo notebook regenerated with an executed cell demonstrating verified/fallback/no-row sidecar outcomes"
    requirement: "DISPLAY-01"
    verification:
      - kind: other
        ref: "jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb (all three new cells print PASS)"
        status: pass
    human_judgment: false

duration: 24min
completed: 2026-06-25
status: complete
---

# Phase 8 Plan 01: Telescope Label Verification Sidecar Summary

**Added `CalendarEventTelescopeLabel` OneToOneField sidecar model (solsys_code's first real model/migration) and a standalone `update_or_create` write in `sync_lco_observation_calendar.py` that persists the live-verified-vs-fallback telescope-label outcome per `CalendarEvent`.**

## Performance

- **Duration:** 24 min
- **Started:** 2026-06-25T05:31:09Z
- **Completed:** 2026-06-25T05:55:21Z
- **Tasks:** 3
- **Files modified:** 6 (1 model file, 1 new migration, 1 management command, 2 test files, 1 notebook)

## Accomplishments
- `CalendarEventTelescopeLabel(models.Model)` added to `solsys_code/models.py` — `event` (`OneToOneField(CalendarEvent, on_delete=CASCADE, primary_key=True, related_name='telescope_label_meta')`) and `is_verified` (`BooleanField(default=True)`), both with `verbose_name`, plus a `__str__`. First real model in this app; first real migration (`0001_calendareventtelescopelabel.py`) confirmed clean via `makemigrations --check --dry-run` and `migrate`.
- `sync_lco_observation_calendar.py`'s per-record loop now reconciles a sidecar row via `CalendarEventTelescopeLabel.objects.update_or_create(event=event, defaults={'is_verified': not telescope_api_failed})` immediately after the existing `get_or_create`/diff/`save()` block — kept as a standalone statement, never folded into the `fields` dict or `changed` comparison.
- Four new behaviors test-covered: verified -> `is_verified=True`, fallback -> `is_verified=False`, re-run on an unchanged record -> no duplicate/no extra row, and `load_telescope_runs`-created events -> no sidecar row at all (`DoesNotExist` on the reverse accessor).
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` regenerated with three new executed cells demonstrating all three outcomes, reusing the existing Phase 07 verified/fallback fixtures plus a new classically-scheduled fixture for the no-row case; teardown updated to clean it up.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add CalendarEventTelescopeLabel model and generate the first migration** - `ef1a9c8` (feat)
2. **Task 2: Write the sidecar row from sync_lco_observation_calendar.py and cover it with tests** - `5eb3168` (feat)
3. **Task 3: Regenerate the sync demo notebook to demonstrate the sidecar write (CLAUDE.md convention)** - `51033ae` (docs)

_Note: pre-commit's `ruff-format` hook auto-reformatted the notebook on the Task 3 commit attempt (cosmetic trailing-comma/multi-line-print fixes only, no content change); the commit was re-run and succeeded on the second attempt with the formatted file staged._

## Files Created/Modified
- `solsys_code/models.py` - defines `CalendarEventTelescopeLabel`, the app's first real model
- `solsys_code/migrations/0001_calendareventtelescopelabel.py` - generated migration, first for this app
- `solsys_code/management/commands/sync_lco_observation_calendar.py` - imports the model, adds the standalone sidecar write
- `solsys_code/tests/test_sync_lco_observation_calendar.py` - 3 new tests (verified, fallback, no-churn) for the sidecar write
- `solsys_code/tests/test_load_telescope_runs.py` - 1 new test asserting no sidecar row for classically-scheduled events
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` - 3 new executed cells (+ updated Summary/Teardown cells) demonstrating the sidecar write

## Decisions Made
- Generated the migration with an explicit `--name calendareventtelescopelabel` flag rather than accepting Django's default `0001_initial.py` filename, so the committed file matches the plan's `must_haves.artifacts` entry (`solsys_code/migrations/0001_*.py`) byte-for-byte in name while the generated *content* itself is untouched/deterministic (not hand-edited).

## Deviations from Plan

None — plan executed exactly as written. The migration filename choice above is a naming detail within the plan's own stated constraint ("Confirm `./manage.py migrate` runs clean", filename pattern `0001_*.py`), not a deviation from scope.

## Issues Encountered
- `ruff check .`/`ruff format --check .` run against explicit file paths (rather than the whole-repo invocation the project's quality gate actually uses) initially flagged the new migration file and a B018 "useless expression" in the new notebook test cell. Confirmed the migration file is correctly excluded by `pyproject.toml`'s `exclude = ["solsys_code/**/migrations/*.py", ...]` when `ruff check .`/`ruff format --check .` run repo-wide (the project's actual gate); fixed the B018 in both the test file and the notebook cell by assigning the reverse-accessor read to `_` before the `assertRaises`/`try` block, since that fix was trivial and consistent with the rest of the codebase's style.
- `ruff check .` / `ruff format --check .` repo-wide still report 2 pre-existing issues in untouched cells of this same notebook (cell 6 import-sort, cell 12 line-length) and pre-existing formatting needs in `src/fomo/settings.py` and two unrelated `.planning/quick/` verify scripts — confirmed these predate this plan's changes (reproduced against the pre-Phase-8 git revision) and are out of scope per the deviation rules' scope boundary (only fix issues directly caused by the current task's changes).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- The sidecar model, write path, and read-side reverse accessor (`event.telescope_label_meta`) are in place and test-proven for both the verified/fallback cases and the "no row at all" classical-event default — ready for Plan 02's template-layer read (`calendar.html` dashed-border + tooltip per DISPLAY-02/03).
- `./manage.py test solsys_code` is green at 135 tests; `ruff check .`/`ruff format --check .` introduce no new issues from this plan's changes.
- No blockers for Plan 02.

---
*Phase: 08-telescope-label-verification-sidecar*
*Completed: 2026-06-25*

## Self-Check: PASSED

All created/modified files confirmed present on disk; all four task/summary commit hashes (`ef1a9c8`, `5eb3168`, `51033ae`, `d50dff7`) confirmed present in `git log --oneline --all`.
