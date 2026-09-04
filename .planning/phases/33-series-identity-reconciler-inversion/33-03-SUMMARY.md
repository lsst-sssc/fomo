---
phase: 33-series-identity-reconciler-inversion
plan: 03
subsystem: database
tags: [django, orm, migration, admin, calendareventmeta]

# Dependency graph
requires: []
provides:
  - "CalendarEventMeta.observation_record (OneToOneField, SET_NULL) and .observation_group (ForeignKey, SET_NULL) — the carrier fields Phase 34's observation projector writes"
  - "Migration 0017: additive, no data-transformation step, cross-app dependency on tom_observations declared explicitly"
  - "Both new fields read-only on CalendarEventMetaAdmin and CalendarEventMetaInline (D-09)"
  - "CalendarEventMeta.run verbose_name and surrounding docstrings/comments restated from ownership to attribution wording (D-17)"
affects: [34-observation-projector-and-trigger, 33-04, 33-05]

# Actuals (#2632)
actuals:
  tokens: 7025
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "MigrationExecutor + TransactionTestCase harness (copied from test_window_schema_migration.py) for proving a schema migration is non-destructive against pre-existing rows"

key-files:
  created:
    - solsys_code/migrations/0017_calendareventmeta_observation_links.py
    - solsys_code/tests/test_calendar_event_meta_links.py
    - .planning/phases/33-series-identity-reconciler-inversion/deferred-items.md
  modified:
    - solsys_code/models.py
    - solsys_code/admin.py
    - solsys_code/tests/test_admin.py

key-decisions:
  - "None beyond the plan — no Rule 4 architectural deviations."

patterns-established:
  - "Pattern 1: a migration test module seeds rows against the historical (pre-migration) model via MigrationExecutor.project_state(...).apps, migrates forward, then asserts field values survive byte-identical — used whenever a plan needs to prove an AddField/AlterField migration preserves existing production rows."

requirements-completed: [PROJ-04]

coverage:
  - id: D1
    description: "CalendarEventMeta gains real observation_record (one-to-one) and observation_group (foreign key) carrier fields, both SET_NULL, matching D-05/D-06/D-07"
    requirement: PROJ-04
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_event_meta_links.py#CalendarEventMetaObservationLinksFieldTests"
        status: pass
    human_judgment: false
  - id: D2
    description: "Migration 0017 is additive-only and non-destructive: existing CalendarEventMeta rows keep run/is_verified/confirmed_by/confirmed_at byte-identical after migrating forward, with both new link columns landing NULL"
    requirement: PROJ-04
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_event_meta_links.py#TestCalendarEventMetaObservationLinksMigration"
        status: pass
      - kind: other
        ref: "grep -c 'RunPython\\|RunSQL' solsys_code/migrations/0017_calendareventmeta_observation_links.py"
        status: pass
    human_judgment: false
  - id: D3
    description: "observation_record/observation_group are read-only on both CalendarEventMetaAdmin (standalone change form) and CalendarEventMetaInline (CampaignRun change page) — no staff surface can bind either value"
    requirement: PROJ-04
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_admin.py#CalendarEventMetaObservationLinksReadOnlyTests"
        status: pass
    human_judgment: false
  - id: D4
    description: "CalendarEventMeta.run presents as an attribution, not an ownership: verbose_name and the surrounding class/inline docstrings and comments no longer describe the run as owning the event (D-17)"
    verification:
      - kind: other
        ref: "grep \"verbose_name='Attributed campaign run'\" solsys_code/models.py; grep -n 'owned\\|owns\\|ownership' solsys_code/admin.py"
        status: pass
    human_judgment: false

# Metrics
duration: ~40min
completed: 2026-09-04
status: complete
---

# Phase 33 Plan 03: Real FK Carrier Fields for CalendarEventMeta Summary

**Added `observation_record`/`observation_group` foreign keys to `CalendarEventMeta` via one additive migration, exposed both read-only in the admin, and restated the model's run-attribution wording from ownership to attribution.**

## Performance

- **Duration:** ~40 min
- **Started:** ~2026-09-04T15:11:00Z (estimated)
- **Completed:** 2026-09-04T15:51:31Z
- **Tasks:** 3
- **Files modified:** 6 (3 created, 3 modified)

## Accomplishments
- `CalendarEventMeta.observation_record` (`OneToOneField`, `SET_NULL`) and `.observation_group`
  (`ForeignKey`, `SET_NULL`) — the real carrier fields Phase 34's observation projector will
  write, replacing spike 002's title-suffix stopgap (PROJ-04)
- Migration `0017_calendareventmeta_observation_links.py`: exactly two `AddField`s and one
  `AlterField`, no data-transformation step, cross-app `tom_observations` dependency declared
  explicitly (Pitfall 5) — applied cleanly to the dev database
- Both new fields are read-only on `CalendarEventMetaAdmin` and `CalendarEventMetaInline` (D-09)
  — only the observation projector may ever write them
- `CalendarEventMeta.run`'s `verbose_name` and surrounding docstrings/comments in both
  `models.py` and `admin.py` restated from ownership ("owns"/"owned by") to attribution
  ("attributed to") wording (D-17)
- New test module `test_calendar_event_meta_links.py`: 6 field-behavior tests plus a
  `MigrationExecutor`-based migration test proving pre-existing rows survive migration 0017
  byte-identical (ROADMAP criterion 1's load-bearing proof)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add the two link fields and generate migration 0017** - `fe4d1ca` (feat)
2. **Task 2: Expose both links read-only in the admin and restate the admin's ownership wording** - `d8b5676` (feat)
3. **Task 3: Prove the fields behave and the migration is non-destructive** - `89ce52c` (test)

**Plan metadata:** (this commit)

## Files Created/Modified
- `solsys_code/models.py` - `CalendarEventMeta.observation_record`/`.observation_group` fields, `run` verbose_name and docstring restated to attribution wording
- `solsys_code/migrations/0017_calendareventmeta_observation_links.py` - additive migration: two `AddField`s + one `AlterField`, no data step
- `solsys_code/admin.py` - both new fields added to `readonly_fields` on `CalendarEventMetaAdmin`/`CalendarEventMetaInline`; ownership wording in `CalendarEventMetaInline`'s docstring and `CalendarEventMetaAdmin.save_model`'s docstring restated to attribution
- `solsys_code/tests/test_admin.py` - `CalendarEventMetaObservationLinksReadOnlyTests` covering `get_readonly_fields()` on both admin surfaces plus POST-level non-binding regressions
- `solsys_code/tests/test_calendar_event_meta_links.py` - new module: field-behavior `TestCase` and `MigrationExecutor`-based `TransactionTestCase`
- `.planning/phases/33-series-identity-reconciler-inversion/deferred-items.md` - logs one out-of-scope, unrelated flaky test discovered while running the full project test command

## Decisions Made
None - plan executed exactly as written. No Rule 4 architectural deviations; `python manage.py check` reported no reverse-accessor clash, so the mechanical confirmation Task 1 called for needed no rename. `CalendarEventMetaAdmin.list_select_related` was deliberately left unchanged, per the plan's own instruction, because neither new field was added to `list_display`.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

Running the project's full-suite `test_command` (`.planning/config.json`) surfaced one failure
unrelated to this plan: `solsys_code.tests.test_bootstrap5_rendering.TestBootstrap5Rendering.
test_observatory_create_form_submits_to_observatory_url` failed with a `requests.exceptions.
ReadTimeout` against `data.minorplanetcenter.net` — a live-network MPC obscode lookup with no
mock/VCR fixture, in a file this plan never touched. Confirmed pre-existing/out-of-scope per the
executor's scope-boundary rule (a re-run in isolation reproduced the same live-network timeout,
not a regression from this plan's changes) and logged to `deferred-items.md` rather than fixed.
All 957 other tests in the full suite passed; both plan-scoped modules
(`test_calendar_event_meta_links`, `test_admin`) pass cleanly in isolation and combined (61
tests), and the plan's own `<verification>` block (migration check, targeted test run, ruff,
ruff-format) all pass.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

PROJ-04's carrier fields now exist on `CalendarEventMeta` in the exact form Phase 34's
observation projector needs to write, with the one-event-per-observation-record constraint
enforced by the database and both fields locked to code-only writes. Plans 33-04 and 33-05 (same
phase) can proceed; Phase 34 has its carrier. The PROJ-04 shared-title-stem clause remains
explicitly deferred to Phase 34, as scoped — this plan added no code reading or writing any
`CalendarEvent.title`, confirmed by source-level checks in Task 3's verify list.

---
*Phase: 33-series-identity-reconciler-inversion*
*Completed: 2026-09-04*

## Self-Check: PASSED

All created files found on disk; all three task commit hashes (`fe4d1ca`, `d8b5676`, `89ce52c`) found in git log.
