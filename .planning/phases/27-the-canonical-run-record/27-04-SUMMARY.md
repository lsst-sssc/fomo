---
phase: 27-the-canonical-run-record
plan: 04
subsystem: database
tags: [django, migration, model, textchoices, unique-constraint, cascade]

# Dependency graph
requires:
  - phase: 27-the-canonical-run-record (plan 01)
    provides: "calendar_utils.derive_telescope_class(site_raw, telescope_instrument) -- the shared, primitives-only derivation this plan's backfill migration calls; the D-12 subset-assertion placeholder this plan wires to the real enum"
  - phase: 27-the-canonical-run-record (plan 02)
    provides: "dev-DB site repair -- the backfill now sees repaired rather than stale site data (pk 8/12/13/21/27/28 resolved; pk 26/29/30 genuinely site-less)"
  - phase: 27-the-canonical-run-record (plan 03)
    provides: "CalendarEventMeta (renamed from CalendarEventTelescopeLabel) with its run FK and migrations 0008/0009 -- this plan's migration 0010 depends on 0009"
provides:
  - "CampaignRun.Source (six-value TextChoices, LEGACY default) and CampaignRun.telescope_class (lowercase 2m0/1m0/0m4 plus SPACE) -- CANON-01/CANON-02 schema half"
  - "CampaignRun.is_publicly_visible property (D-09/D-10) -- Plan 05's calendar-modal template override consumer"
  - "CampaignRunObservation link model: CASCADE run FK, CASCADE observation_record FK, SET_NULL confirmed_by, confirmed_at, named unique_campaign_run_observation_record constraint -- CANON-04 schema half"
  - "Migrations 0010 (fields + link model) and 0011 (derived-rule telescope_class backfill) applied against the real dev DB"
affects: [27-05-admin-inlines-and-calendar-modal-template, 27-06-import-campaign-csv-source-and-telescope-class, 28-attribution-queue, 29-reconciler]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Named single-field UniqueConstraint (not OneToOneField) for a one-to-one-today-but-cheaply-broadenable relation -- broadening later is one RemoveConstraint with no field change and no reader rewrites, since the reverse accessor is already a manager"
    - "Data-migration RunPython importing a primitives-only helper function directly (calendar_utils.derive_telescope_class) while still fetching the model via apps.get_model -- the model needs the historical/frozen state, the helper does not"

key-files:
  created:
    - solsys_code/migrations/0010_campaignrun_source_telescope_class_campaignrunobservation.py
    - solsys_code/migrations/0011_backfill_campaignrun_telescope_class.py
    - solsys_code/tests/test_campaign_run_observation.py
  modified:
    - solsys_code/models.py
    - solsys_code/tests/test_canonical_record_migration.py
    - solsys_code/tests/test_calendar_utils.py

key-decisions:
  - "Migration 0010's header-comment prose was rephrased to avoid the literal tokens 'AddField'/'CreateModel'/'RunPython.noop' appearing outside actual operation calls, so the plan's own acceptance-criteria greps (exact counts, not subset) pass without a grep-literalism footnote"
  - "Test fixture for the confirming-user-deletion test uses a separate record_owner user for the ObservationRecord FK, distinct from the user being deleted -- ObservationRecord.user is on_delete=DO_NOTHING, so deleting a user still referenced by a live ObservationRecord row fails SQLite's deferred foreign-key check"
  - "Migration test seeds all 7 CampaignRun rows with distinct resolved (non-null) windows rather than mixing in TBD rows, since a resolved window is sufficient to avoid every natural-key collision and keeps the fixture simpler"

patterns-established:
  - "A model gaining both a new field and a new link model in the same migration file batches all schema-only operations (AddField x2, CreateModel) into one migration, deferring any RunPython backfill of the new field to its own separate migration"

requirements-completed: [CANON-01, CANON-02, CANON-04]

# Metrics
duration: ~45min
completed: 2026-07-30
---

# Phase 27 Plan 04: Schema Layer -- source, telescope_class and the Observation Link Summary

**`CampaignRun` now records its ingest source and telescope-class allocation (both `TextChoices`, six and four values respectively) and owns a `CampaignRunObservation` link model to `ObservationRecord`, with migrations 0010/0011 applied against the real dev DB and the D-16 telescope_class backfill landing exactly the three predicted rows (JUICE->SPACE, LCO 1m->1m0, LCO 2m->2m0).**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-07-29T~23:56 (local; continuing from Wave 2)
- **Completed:** 2026-07-30T01:22:21Z
- **Tasks:** 3 completed
- **Files modified:** 6 (3 created, 3 modified)

## Accomplishments

- Added `CampaignRun.Source` (six-value `TextChoices`: `WEB`/`CLASSICAL_FILE`/`LCO_QUEUE`/`GEMINI_QUEUE`/`CSV_IMPORT`/`LEGACY`, `LEGACY` default) and `CampaignRun.telescope_class` (`TWO_M0`='2m0'/`ONE_M0`='1m0'/`ZERO_M4`='0m4'/`SPACE`='SPACE', lowercase per D-21) plus the `is_publicly_visible` property (D-09/D-10) -- `CampaignRunTableView.get_queryset()` left untouched, per the plan's negative constraint.
- Added `CampaignRunObservation`: `run` (CASCADE FK to `CampaignRun`), `observation_record` (CASCADE FK to `tom_observations.ObservationRecord`), `confirmed_by` (SET_NULL FK to `AUTH_USER_MODEL`), `confirmed_at`, and the named `unique_campaign_run_observation_record` constraint (D-01..D-04) -- no boolean confirmation flag, by design.
- Neither existing `CampaignRun` partial `UniqueConstraint` was touched (D-14 negative constraint, confirmed by `git diff` grep and `./manage.py check`).
- Hand-authored migrations `0010` (`AddField` x2 + `CreateModel`) and `0011` (`RunPython` backfill via `calendar_utils.derive_telescope_class`, gated on `site__isnull=True`, writing only non-blank derived values). Applied against the real dev DB (backed up first): `pk=26` (JUICE) -> `SPACE`, `pk=29` (LCO 1m) -> `1m0`, `pk=30` (LCO 2m) -> `2m0`; every row's `source` stayed `legacy`; `pk=31` (the rejected `X05` row, D-15) took no write, exactly as its derived value is blank. `makemigrations --check --dry-run` reports "No changes detected".
- 10 new tests: 7 in `test_campaign_run_observation.py` (the named constraint firing, one-run/many-records coexistence, both `CASCADE` directions, `SET_NULL` on confirming-user deletion, blank-by-default audit fields, and run-deletion preserving `CalendarEvent`/`CalendarEventMeta` rows) and 3 in a new `TestSourceAndTelescopeClassBackfill` migration-test class (every D-16 row shape plus a site-resolved control row that must stay blank despite matching instrument text). The D-12 subset-assertion test in `test_calendar_utils.py` now reads `CampaignRun.TelescopeClass` directly instead of Plan 27-01's literal placeholder.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Source, TelescopeClass, is_publicly_visible and the CampaignRunObservation link model** - `9834430` (feat)
2. **Task 2: Hand-author and apply migrations 0010 (fields + link model) and 0011 (telescope_class backfill)** - `35fdbdf` (feat)
3. **Task 3: Test the link model's constraints and cascades, and extend the migration test to cover source and the backfill** - `c3ca809` (test)

_No TDD tasks in this plan (autonomous, non-TDD execute plan)._

## Files Created/Modified

- `solsys_code/models.py` - `CampaignRun.Source`/`CampaignRun.TelescopeClass` `TextChoices`, `source`/`telescope_class` fields, `is_publicly_visible` property, new `CampaignRunObservation` model with its named unique constraint; `ObservationRecord`/`settings` imports added
- `solsys_code/migrations/0010_campaignrun_source_telescope_class_campaignrunobservation.py` - Hand-authored `AddField(source)`/`AddField(telescope_class)` + `CreateModel(CampaignRunObservation)`, depending on `0009_calendareventmeta_run` and `tom_observations`/`AUTH_USER_MODEL`
- `solsys_code/migrations/0011_backfill_campaignrun_telescope_class.py` - Hand-authored one-way `RunPython` backfill calling `calendar_utils.derive_telescope_class` inside the function body, iterating `site__isnull=True` rows
- `solsys_code/tests/test_campaign_run_observation.py` - New module, 7 tests covering the constraint, both cascade directions, `SET_NULL`, blank audit-field defaults, and the calendar-event/companion-record preservation guarantee
- `solsys_code/tests/test_canonical_record_migration.py` - New `TestSourceAndTelescopeClassBackfill` `TransactionTestCase` (3 tests), seeding all 7 D-16 row shapes plus a site-resolved control row
- `solsys_code/tests/test_calendar_utils.py` - D-12 subset assertion now imports `CampaignRun` and compares against `CampaignRun.TelescopeClass`'s real members instead of a literal set; added an explicit assertion that `TelescopeClass.SPACE` is absent from the aperture-class set

## Decisions Made

- Rephrased migration 0010's and 0011's header comments to avoid the literal identifier tokens (`AddField`, `CreateModel`, `RunPython.noop`) appearing in prose outside the real operation calls, so the plan's exact-count acceptance-criteria greps pass cleanly rather than needing a grep-literalism footnote (the pattern Plans 27-01/27-03 both hit and had to explain away).
- Gave the `ObservationRecord` fixture in `test_campaign_run_observation.py` its own `record_owner` user, separate from the `confirmed_by` user under test -- `ObservationRecord.user` is `on_delete=DO_NOTHING`, so deleting a still-referenced user fails SQLite's deferred FK check inside a `TestCase`'s wrapping transaction.
- Seeded the migration test's 7 rows with distinct resolved (non-`None`) windows rather than mixing in TBD (`window_start=None`) rows -- sufficient to dodge every natural-key collision without needing `contact_person` differentiation, and simpler to read.

## Deviations from Plan

None - plan executed exactly as written. All three tasks' acceptance criteria were met, including the D-14 negative-constraint greps, the lowercase-vs-uppercase `TelescopeClass` casing check, the D-13 derived-rule-not-pk-list check on the backfill migration, and the dev-DB verification query's exact expected row set (JUICE/LCO 1m/LCO 2m, zero non-legacy sources).

One expected grep-literalism risk was avoided rather than incurred: the initial migration-file drafts used the words "AddField"/"CreateModel"/"RunPython.noop" in header-comment prose, which would have inflated the acceptance criteria's exact-count greps (2 vs. observed 4, 1 vs. observed 2, 1 vs. observed 2) the same way Plans 27-01 and 27-03 both hit and had to document as a false-positive footnote. Rephrased the prose (see Decisions Made) so the greps return the exact expected counts with no footnote needed.

## Issues Encountered

None specific to this plan's own files. The pre-existing `./manage.py test solsys_code` segfault in `test_views.py`/`test_ephem_utils.py` (SPICE/ASSIST native code, documented since Plan 27-01) is unrelated to this plan -- verified instead via the plan's own quick-run command (307 tests), the plan's targeted module set (37 and 155 tests for Tasks 2/3's own verification commands), and a combined run of every `solsys_code.tests.*` module except `test_views`/`test_ephem_utils` plus `solsys_code_observatory` (616 tests, all pass -- 10 more than the documented 606-test Wave 2 baseline, exactly matching the 10 new tests this plan added: 7 in `test_campaign_run_observation.py` + 3 in `test_canonical_record_migration.py`).

## User Setup Required

None - no external service configuration required. The migration backfill ran entirely against local data with no network dependency (unlike Plan 27-02's HST/Swift tier-2 repair).

## Live Database Note

The dev DB was backed up to `/tmp/fomo_db.sqlite3.pre-27-04.bak` (950,272 bytes, md5 `1622b1f1e535a3929c7338ec1a7b58fe`, verified byte-identical to `src/fomo_db.sqlite3` at backup time) before running `python manage.py migrate`. This does not overwrite the Plan 02 (`pre-27-02.bak`) or Plan 03 (`pre-27-03.bak`) backups, which remain at their original paths.

Post-migration dev-DB state (no `contact_person`/`contact_email` values printed, per T-27-18):

| pk | telescope_instrument | site_raw | telescope_class (after) | source (after) |
|----|----------------------|----------|--------------------------|-----------------|
| 26 | JUICE | (blank) | SPACE | legacy |
| 29 | LCO 1m | (blank) | 1m0 | legacy |
| 30 | LCO 2m | (blank) | 2m0 | legacy |
| 31 | FOO / BAR | X05 | (blank, no write) | legacy |

Every other `CampaignRun` row (site-resolved) also stayed `source=legacy` and `telescope_class=''`. `python manage.py makemigrations --check --dry-run` reports "No changes detected" after both migrations applied.

## Next Phase Readiness

- `CampaignRun.Source`, `CampaignRun.telescope_class`, `CampaignRun.is_publicly_visible`, and `CampaignRunObservation` are all in place and schema-verified, ready for Plan 05's admin inlines (`CalendarEventMetaInline`, `CampaignRunObservationInline`, `CampaignRunAdmin.save_formset` stamping `confirmed_by`/`confirmed_at`) and calendar-modal template override (D-08's `is_publicly_visible` consumer).
- Plan 06's `import_campaign_csv` can now set `source=CSV_IMPORT` and call the same `calendar_utils.derive_telescope_class()` helper this plan's backfill migration used, per D-20's shared-derivation requirement.
- The dev-DB backup for this plan is at `/tmp/fomo_db.sqlite3.pre-27-04.bak` (does not overwrite Plans 02/03's backups at `/tmp/fomo_db.sqlite3.pre-27-02.bak`/`/tmp/fomo_db.sqlite3.pre-27-03.bak`).
- No blockers. `ruff check .`/`ruff format --check .` report the same pre-existing, unrelated issues present before this plan (1 notebook docstring warning, 7 files needing reformat -- none touched by this plan; `src/fomo/settings.py` is the user's local dev config and was never staged or committed).

---
*Phase: 27-the-canonical-run-record*
*Completed: 2026-07-30*

## Self-Check: PASSED

All 6 created/modified source files plus the dev-DB backup (`/tmp/fomo_db.sqlite3.pre-27-04.bak`) confirmed present on disk; all 3 task commit hashes (`9834430`, `35fdbdf`, `c3ca809`) confirmed present in `git log --oneline --all`.
