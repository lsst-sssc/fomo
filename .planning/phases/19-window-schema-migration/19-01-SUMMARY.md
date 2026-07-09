---
phase: 19-window-schema-migration
plan: 01
subsystem: database
tags: [django, migrations, sqlite, postgresql, unique-constraint, campaignrun]

# Dependency graph
requires:
  - phase: 18-uncertain-scheduling-investigation-spike
    provides: locked window schema design (nullable window_start/window_end pair) and TBD natural-key recommendation (campaign, telescope_instrument, contact_person)
provides:
  - "CampaignRun.window_start/window_end nullable DateFields replacing obs_date/ut_start/ut_end"
  - "Two partial UniqueConstraints (resolved-window branch, TBD branch) closing the NULL-uniqueness gap"
  - "Combined migration 0004 with load-bearing operation order (backfill + generic dedup before constraint swap)"
affects: [19-02-window-display-forms, 19-03-coverage-gap-csv-import, phase-20-range-tbd-import]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two partial UniqueConstraint(condition=Q(...)) instances on one model, each scoped to a disjoint row subset (resolved vs. TBD), rather than one unconditional constraint"
    - "Single combined migration with hand-inserted RunPython steps positioned strictly between AddField and RemoveField/AddConstraint operations"
    - "Generic ordered-by-pk RunPython dedup (not hardcoded pks) so the migration is re-runnable/portable"

key-files:
  created:
    - solsys_code/migrations/0004_campaignrun_window_schema.py
  modified:
    - solsys_code/models.py
    - solsys_code/tests/test_campaign_models.py

key-decisions:
  - "Resolved-window UniqueConstraint keys on all four of (campaign, telescope_instrument, window_start, window_end), not window_start alone, so a future range starting the same day as an existing single-night row won't collide (RESEARCH A2)"
  - "TBD UniqueConstraint deliberately excludes window_start/window_end from its fields= tuple (both always NULL under its own condition) and keys on contact_person instead, per Phase 18's locked recommendation"
  - "Migration operations ordered: AddField x2 -> RunPython(backfill) -> RunPython(dedup) -> RemoveConstraint -> RemoveField x3 -> AddConstraint x2, verified programmatically (dedup strictly precedes both AddConstraint and RemoveField)"

patterns-established:
  - "Pattern: partial/conditional UniqueConstraint pairs for 'different uniqueness rules for different row subsets' on the same model"

requirements-completed: [SCHED-02, SCHED-03, SCHED-04, SCHED-05]

coverage:
  - id: D1
    description: "CampaignRun.window_start/window_end nullable DateFields replace obs_date/ut_start/ut_end; a single classically-scheduled night saves with window_start == window_end"
    requirement: "SCHED-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_models.py#TestCampaignRunFieldInventory.test_full_field_inventory_persists_and_reloads"
        status: pass
    human_judgment: false
  - id: D2
    description: "A CampaignRun can save fully TBD (both window fields null), distinct from a resolved window"
    requirement: "SCHED-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_models.py#TestCampaignRunWindowSchema.test_tbd_run_saves_with_both_window_fields_null"
        status: pass
    human_judgment: false
  - id: D3
    description: "Two partial UniqueConstraints enforce the TBD branch (campaign+telescope_instrument+contact_person) and resolved-window branch (campaign+telescope_instrument+window_start+window_end); collisions raise IntegrityError, differing contact_person does not collide"
    requirement: "SCHED-04"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_models.py#TestCampaignRunWindowSchema.test_tbd_same_contact_person_collides"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_models.py#TestCampaignRunWindowSchema.test_tbd_differing_contact_person_both_save"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_models.py#TestCampaignRunWindowSchema.test_resolved_window_same_key_collides"
        status: pass
    human_judgment: false
  - id: D4
    description: "Migration 0004 backfills every pre-existing row (window_start == window_end == former obs_date) and generically dedupes leftover duplicate TBD fixture rows with a logged warning, before the new partial constraints are added"
    requirement: "SCHED-05"
    verification:
      - kind: manual_procedural
        ref: "manage.py migrate solsys_code run against an isolated copy of the real dev DB (src/fomo_db.sqlite3) via a scratch-settings override; confirmed 16 -> 14 rows, all 14 resolved rows window_start==window_end==former obs_date, pk 17 deleted (kept 15) and pk 18 deleted (kept 16) with logged warnings, both partial unique indexes present in the resulting schema"
        status: pass
    human_judgment: false

duration: ~20min
completed: 2026-07-09
status: complete
---

# Phase 19 Plan 1: Window-Schema Migration Foundation Summary

**CampaignRun's obs_date/ut_start/ut_end replaced by a nullable window_start/window_end DateField pair, enforced by two partial UniqueConstraints, via one combined non-reversible migration that backfills and dedupes existing rows before swapping constraints.**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-07-09T22:17:50Z
- **Tasks:** 2
- **Files modified:** 3 (1 created: migration 0004; 2 modified: models.py, test_campaign_models.py)

## Accomplishments
- `CampaignRun.window_start`/`window_end` (nullable `DateField`s) replace `obs_date`/`ut_start`/`ut_end`; `__str__` updated to reference `window_start`.
- Two partial `UniqueConstraint`s replace the single old natural-key constraint: `unique_campaign_run_resolved_window` (keyed on campaign/telescope_instrument/window_start/window_end, condition `window_start IS NOT NULL`) and `unique_campaign_run_tbd_natural_key` (keyed on campaign/telescope_instrument/contact_person, condition `window_start IS NULL`) — closing the NULL-uniqueness gap SCHED-04 targets.
- Combined migration `0004_campaignrun_window_schema.py` in the exact load-bearing operation order: `AddField` x2 -> `RunPython(backfill_window_fields)` -> `RunPython(dedupe_tbd_collisions)` -> `RemoveConstraint` -> `RemoveField` x3 -> `AddConstraint` x2. Order verified programmatically (dedup index strictly precedes both `AddConstraint` and `RemoveField` indices).
- `dedupe_tbd_collisions` is generic (ordered-by-pk Python loop keyed on `(campaign_id, telescope_instrument, contact_person)`, not hardcoded pks) and logs a warning for every row it deletes (D-08, Repudiation mitigation T-19-03).
- Model-level tests prove SCHED-02 (single-night save), SCHED-03 (TBD save), and SCHED-04 (both partial constraints: TBD collision, TBD differing-contact_person no-collision, resolved-window collision).
- Manual smoke-test: ran `manage.py migrate solsys_code` against an isolated copy of the real dev DB — confirmed all 14 surviving rows have `window_start == window_end == former obs_date`, the two known duplicate TBD pairs (pk 17→kept 15, pk 18→kept 16) were removed with logged warnings, and both new partial unique indexes exist in the resulting SQLite schema. The real dev DB (`src/fomo_db.sqlite3`) was left untouched (md5 verified unchanged before/after).

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace CampaignRun window fields + partial constraints and write the combined migration** - `909484c` (feat)
2. **Task 2: Model-level tests for single-night, TBD, and both partial constraints** - `5154bb7` (test)

_Note: Task 2 is TDD-tagged but the schema it tests was implemented in Task 1 (a two-task foundation/proof split, not a strict RED/GREEN pair) — tests were written and run directly against the already-implemented schema, all passing on first run._

## Files Created/Modified
- `solsys_code/models.py` - `window_start`/`window_end` DateFields; two partial `UniqueConstraint`s; `__str__` updated
- `solsys_code/migrations/0004_campaignrun_window_schema.py` - combined migration: AddField, backfill, generic dedup, RemoveConstraint/RemoveField, AddConstraint x2
- `solsys_code/tests/test_campaign_models.py` - rewrote field-inventory test for window fields; added TBD-save, TBD-collision, TBD-differing-contact_person, and resolved-window-collision tests

## Decisions Made
- Resolved-window constraint keys on all four fields (including `window_end`), per RESEARCH.md's A2 recommendation, so a future date-range starting on the same day as an existing single-night row won't false-collide.
- TBD constraint deliberately omits `window_start`/`window_end` from its `fields=` tuple (both always NULL under its own condition — including them would silently do nothing, per RESEARCH's Anti-Pattern warning).
- Performed a manual migration smoke-test against an isolated copy of the dev DB (not the real file) to validate the RunPython backfill/dedup steps end-to-end before considering the plan done, even though this is formally a "MANUAL" verification item per 19-VALIDATION.md — done here as an extra confidence check since it was low-risk and fast.

## Deviations from Plan

None - plan executed exactly as written. `makemigrations` auto-generated a different operation order (RemoveConstraint/RemoveField before AddField/AddConstraint) than Pattern 1 requires; hand-editing the generated file into the exact order specified in the plan/RESEARCH.md was itself part of the planned action, not a deviation.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The new `window_start`/`window_end` schema and both partial constraints are live in `models.py` and migration 0004; `makemigrations solsys_code --check --dry-run` is clean.
- Plan 19-02 (and later 19-03/19-04, and Phase 20's consumers) can now compile against `window_start`/`window_end` — but note per RESEARCH.md's D-01 hard-cutover scope, 7 other non-test modules and 5 other test modules (`campaign_forms.py`, `campaign_gap.py`, `campaign_views.py`, `campaign_utils.py`, `campaign_tables.py`, `management/commands/import_campaign_csv.py`, plus their test files) still reference the removed `obs_date`/`ut_start`/`ut_end` fields and will need updating in subsequent plans of this phase before `./manage.py test solsys_code` (full suite) is green again. This plan's own verification scope (`test_campaign_models`) is green.
- The real dev DB (`src/fomo_db.sqlite3`) has NOT yet had migration 0004 applied — that step (and the deferred quick tasks it may motivate) remains for whenever this phase is deployed/the operator runs `manage.py migrate` for real, per 19-VALIDATION.md's Manual-Only Verification.

---
*Phase: 19-window-schema-migration*
*Completed: 2026-07-09*
