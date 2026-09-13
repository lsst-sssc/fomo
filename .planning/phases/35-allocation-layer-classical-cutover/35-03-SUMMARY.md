---
phase: 35-allocation-layer-classical-cutover
plan: 03
subsystem: calendar-sync
tags: [django, campaign-reconciler, allocation-projector, calendar-events, tdd]

requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: "plan 35-01's ALLOC: namespace, project_allocation()'s per-night loop, the D-13 sun_event()-on-mint-only pattern, ReconcileResult counters"
provides:
  - "CampaignRun.night_start_utc / night_end_utc -- two nullable sub-night window TimeFields, migration 0018"
  - "allocation_projector.night_bounds() -- the per-night, per-end span resolution the classical loader's _resolve_window_time() rule now expresses"
  - "allocation_projector's D-13 re-mint branch: a sub-night field change deletes and re-creates exactly the affected nights, never rewriting start_time/end_time in place"
affects: [35-04, 35-05, 35-06, 35-07]

actuals:
  tokens: 4767
  tasks: 2
  commits: 3
  plan_head_before: 6cde543f0829bb1c724fb14dd513a3c0e0d824a1

tech-stack:
  added: []
  patterns:
    - "Astropy-free re-mint comparison: a null sub-night field means the expected boundary is the sun-event pair, which cannot be known without calling sun_event() -- so a null field never triggers re-mint (D-13's own 'no astropy call for an existing night' rule extended to the new comparison). A SET field's expected boundary is a plain date/time computation, so it is compared directly against the stored value with zero astropy cost."
    - "Delete-and-recreate over rewrite-in-place: an existing allocation night's start_time/end_time are never mutated. A boundary mismatch deletes the row (counted as retired) and inserts a fresh one via sun_event() (counted as created), giving the night a new primary key -- the same vocabulary a legacy RUN: takeover or a D-05 retirement already uses, so the counters stay consistent across every code path that removes-then-re-adds a night."
    - "_mint_fields() factors the entire create-only field set (including both sun_event() calls) into one helper shared by the ordinary create path and the new re-mint path, so both callers pay the same one-time astropy cost and neither can drift from the other."

key-files:
  created: []
  modified:
    - solsys_code/models.py
    - solsys_code/migrations/0018_campaignrun_night_window_fields.py
    - solsys_code/allocation_projector.py
    - solsys_code/tests/test_allocation_projector.py
    - solsys_code/tests/test_campaign_models.py

key-decisions:
  - "night_start_utc/night_end_utc are TimeFields (not integer minutes-after-midnight): they round-trip cleanly from the classical loader's own (hour, minute) integer parse and are directly admin-editable, per the plan's own 'Claude's Discretion' note."
  - "No admin.py change: CampaignRunAdmin declares no explicit fields/fieldsets list, so both new fields are editable by default the moment they exist on the model -- verified via git status --porcelain reporting zero lines for admin.py."
  - "The TestSubNightWindow class treats 'the unaffected nights' (behavior test 4) as a SECOND run's nights, not a coincidental non-match within the same run's window -- a run-level sub-night field applies identically to every night in that run's own window, so isolation across runs is the only construction that gives a genuine unaffected/affected split without relying on an astronomically-improbable coincidence."

requirements-completed: [ALLOC-01, ALLOC-04]

coverage:
  - id: D1
    description: "CampaignRun carries two new nullable sub-night window fields (night_start_utc, night_end_utc), added by one small additive migration (0018) with no data step; both are editable in the Django admin with no admin.py change"
    requirement: ALLOC-04
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_models.py#TestCampaignRunSubNightWindowFields (3 tests)"
        status: pass
      - kind: other
        ref: "python manage.py makemigrations --check --dry-run -> 'No changes detected'"
        status: pass
      - kind: other
        ref: "git status --porcelain -- solsys_code/admin.py -> empty"
        status: pass
    human_judgment: false
  - id: D2
    description: "The allocation projector honours a run's sub-night window per night (null = computed sunset/sunrise; a set field resolves via the before-noon-UTC/at-or-after-noon-UTC date-offset rule), and re-mints (deletes + re-creates) a night whose stored span no longer matches the run's current fields, without ever rewriting start_time/end_time in place; sun_event() is never called for the astropy-free re-mint comparison itself"
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestSubNightWindow (6 tests)"
        status: pass
      - kind: integration
        ref: "python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_campaign_models solsys_code.tests.test_calendar_event_meta_links (64 tests)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Both formatting gates (pre-commit run ruff / ruff-format) clean over this plan's own five files"
    verification:
      - kind: other
        ref: "pre-commit run ruff --files <5 paths> && pre-commit run ruff-format --files <5 paths>"
        status: pass
    human_judgment: false

duration: 21min
completed: 2026-09-13
status: complete
---

# Phase 35 Plan 03: Sub-Night Window Fields & the D-13 Re-Mint Rule Summary

**`CampaignRun` gains two nullable `night_start_utc`/`night_end_utc` `TimeField`s (migration 0018) and the allocation projector's new `night_bounds()`/`_span_needs_remint()` pair applies the classical loader's before-noon/at-or-after-noon date rule per night, deleting and re-creating (never rewriting) a night whose sub-night span changed.**

## Performance

- **Duration:** 21 min
- **Started:** 2026-09-13T04:38:00Z (approx, from STATE.md's session marker)
- **Completed:** 2026-09-13T04:59:01Z
- **Tasks:** 2 (Task 1 auto, Task 2 auto/tdd)
- **Files modified:** 5

## Accomplishments

- `CampaignRun.night_start_utc` / `night_end_utc`: two nullable `TimeField`s with a documentary comment block (null = computed sun event; the before-noon/at-or-after-noon UTC date-offset rule; deliberately no null-together constraint; staff-editable with automatic re-mint on save) immediately after the existing `window_end` field.
- `solsys_code/migrations/0018_campaignrun_night_window_fields.py`: exactly two `AddField` operations, zero `RunPython`, dependent on `0017_calendareventmeta_observation_links` -- `makemigrations` produced the exact target filename with no rename needed.
- No `admin.py` change: `CampaignRunAdmin` declares no explicit `fields`/`fieldsets`, so both new fields are editable by default.
- `allocation_projector.night_bounds(run, night, sunset, sunrise)`: resolves each end of a night independently -- the sun-event value when the matching field is null, otherwise a UTC datetime built from the stored `time` and the correct date (next morning when the hour is before 12:00 UTC, the night's own evening date otherwise) via the new `_time_of_day_to_datetime()` helper.
- `allocation_projector._span_needs_remint(run, night, existing)`: the astropy-free D-13 comparison -- a null field never triggers re-mint (its expected boundary requires a `sun_event()` call, which D-13 forbids for an existing night); a set field's expected boundary is computed directly and compared to the stored value.
- The per-night loop's update branch now checks `_span_needs_remint()` first: a mismatch deletes the existing event and calls the new `_mint_fields()` helper (which itself calls `sun_event()` exactly once for both `'sun'` and `'dark'` kinds) to insert a fresh one, counted as `retired` + `created`, never `updated`. Both halves are skipped under `dry_run` so a preview and a real run report the same pair of counters without any astropy cost in preview mode.
- `_mint_fields()` factors the entire create-only field set (title, description with dark-window line, target_list, telescope, instrument, start_time, end_time via `night_bounds()`) so the ordinary create path and the new re-mint path share one implementation and can't drift apart.
- `TestSubNightWindow` (6 new tests in `test_allocation_projector.py`): null-null byte-identical span; set-end/null-start (sunset start, next-morning end); set-start/null-end (own-evening-date start, sunrise end); changing `night_end_utc` re-mints only the owning run's nights (a sibling run's pks are untouched), counted as `created`+`retired`, never `updated`; a second reconcile with matching sub-night fields makes zero further `sun_event()` calls; and writes nothing at all (`unchanged` for every night, pks stable).
- `TestCampaignRunSubNightWindowFields` (3 new tests in `test_campaign_models.py`): both fields null by default; a partial (end set, start null) round-trip; both set round-trip as `datetime.time` instances.
- Full `test_allocation_projector.py` (40 tests), `test_campaign_models.py` (19 tests) and `test_calendar_event_meta_links.py` all green together (64 tests) -- this plan's own three-module `<verification>` command.

## Task Commits

Each task was committed atomically (Task 2 additionally split into TDD RED/GREEN commits):

1. **Task 1: Two nullable sub-night window fields on CampaignRun, plus the additive migration** - `53132db` (feat)
2. **Task 2 RED: failing `TestSubNightWindow` tests** - `8059eda` (test) -- 3/6 tests failed on real assertion mismatches (span times, re-mint counters) against the unmodified projector; the other 3 passed unchanged since they pin pre-existing behavior
3. **Task 2 GREEN: `night_bounds()`, `_span_needs_remint()`, `_mint_fields()` and the re-mint branch** - `ebf363c` (feat) -- all 6 tests pass; no REFACTOR commit needed (the GREEN implementation needed no follow-up cleanup)

**Plan metadata:** committed alongside this SUMMARY.

## Files Created/Modified

- `solsys_code/models.py` - `CampaignRun.night_start_utc`/`night_end_utc` (D-04)
- `solsys_code/migrations/0018_campaignrun_night_window_fields.py` - new additive-only migration
- `solsys_code/allocation_projector.py` - `night_bounds()`, `_time_of_day_to_datetime()`, `_span_needs_remint()`, `_mint_fields()`, and the per-night loop's new re-mint branch
- `solsys_code/tests/test_allocation_projector.py` - new `TestSubNightWindow` class (6 tests)
- `solsys_code/tests/test_campaign_models.py` - new `TestCampaignRunSubNightWindowFields` class (3 tests)

## Decisions Made

See `key-decisions` in the frontmatter: `TimeField` over an integer minutes-after-midnight representation; no `admin.py` change needed; and the "unaffected nights" in behavior Test 4 constructed as a second run's nights rather than a coincidental non-match, since a run-level field applies identically to every night in its own window.

## Deviations from Plan

None - plan executed exactly as written. The one interpretive judgment call (Test 4's "unaffected nights" construction) is documented above as a decision, not a deviation -- the plan's `<behavior>` block described the required *outcome* (pks change for affected nights, stay stable for unaffected ones, counted as created+retired never updated) without prescribing the exact fixture shape, and Claude's Discretion explicitly covers test file layout.

## Issues Encountered

None. The plan's own acceptance criteria and `<verify>` commands were satisfied on the first implementation pass after RED; the only follow-up needed was a single `pre-commit run ruff` `SIM102` fix (nested `if` -> combined boolean condition in `_span_needs_remint()`), auto-fixed inline before the GREEN commit.

## TDD Gate Compliance

Task 2 carries `tdd="true"`. `workflow.tdd_mode` is `false` in this project's config, so the runtime MVP+TDD halt gate did not apply, but the full RED/GREEN commit-scope contract was followed this time (unlike 35-01's disclosed shortfall):

- **RED** (`8059eda`, `test(35-03): ...`): all six `TestSubNightWindow` tests were written and run against the unmodified projector. 3/6 failed on genuine assertion mismatches for the planned behavior (span boundary values, re-mint counters) -- not import errors, syntax errors, or fixture crashes. The other 3 passed unchanged, correctly pinning behavior that was already correct (the null-null span, the no-recompute guarantee, and the no-write-on-match guarantee) before this task's implementation.
- **GREEN** (`ebf363c`, `feat(35-03): ...`): `night_bounds()`, `_time_of_day_to_datetime()`, `_span_needs_remint()`, `_mint_fields()` and the re-mint branch were implemented; all 6 tests pass.
- **REFACTOR**: not needed -- the GREEN implementation required no follow-up cleanup beyond the ruff `SIM102` auto-fix folded into the same commit before it was made.

`gsd_run check tdd-red-evidence` was not invoked as a formal gate (matching 35-01's precedent under `workflow.tdd_mode: false`); RED was verified manually from the real test-run output shown above, which names the specific failing assertions rather than a bare non-zero exit.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `CampaignRun` can now carry everything a classical schedule line says about its nights (ROADMAP Success Criterion 4's precondition), ready for plan 35-05's classical ingest cutover to populate `night_start_utc`/`night_end_utc` from `BoN`/`EoN`/`HHMM` tokens.
- ROADMAP Success Criterion 1 ("sunset-to-sunrise event per night") now holds for a partial-night allocation, not just a whole-night one.
- Paired docs (notebooks, runbook) for this phase remain owned by plan 35-07 (wave 5), per this plan's own frontmatter scope note -- no paired-doc follow-up is owed from this plan.
- No blockers for 35-04/35-05/35-06/35-07.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-13*

## Self-Check: PASSED

- FOUND: solsys_code/models.py
- FOUND: solsys_code/migrations/0018_campaignrun_night_window_fields.py
- FOUND: solsys_code/allocation_projector.py
- FOUND: solsys_code/tests/test_allocation_projector.py
- FOUND: solsys_code/tests/test_campaign_models.py
- FOUND commit: 53132db
- FOUND commit: 8059eda
- FOUND commit: ebf363c
- All plan-level `<acceptance_criteria>` and `<verify>` commands re-run and passing (see task-by-task output above)
