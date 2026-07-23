---
phase: quick-260722-twe
plan: 01
subsystem: management-command
tags: [django, lco-api, tom-targets, testing]

# Dependency graph
requires:
  - phase: quick-260722-tkt
    provides: "--create-missing-targets flag on backfill_lco_observation_records (RequestTargetInfo, _request_target_info, _resolve_or_build_field_target)"
provides:
  - "Brand-new field Targets created by --create-missing-targets now carry epoch/pm_ra/pm_dec/parallax from the LCO request's target dict when present"
affects: [backfill_lco_observation_records, target-ingestion]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "New-Target-only field pass-through: extend the branch that builds a not-yet-persisted Target, never the reuse-existing branch, to avoid overwriting real metadata on a shared Target"

key-files:
  created: []
  modified:
    - solsys_code/management/commands/backfill_lco_observation_records.py
    - solsys_code/tests/test_backfill_lco_observation_records.py

key-decisions:
  - "No unit conversion needed: LCO wire units (epoch in Julian years, proper_motion_ra/dec in mas/yr, parallax in mas) match Target field units exactly"
  - "Wire-key rename preserved: LCO's proper_motion_ra/proper_motion_dec map to the dataclass's pm_ra/pm_dec to match Target's field names"

requirements-completed: []

coverage:
  - id: D1
    description: "New field Target created via --create-missing-targets carries epoch/pm_ra/pm_dec/parallax mapped from the request's target dict when those keys are present"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observation_records.py#test_flag_on_creates_new_field_target_with_epoch_pm_parallax"
        status: pass
    human_judgment: false
  - id: D2
    description: "New field Target with those keys omitted from the request leaves epoch/pm_ra/pm_dec/parallax as None (regression guard)"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observation_records.py#test_flag_on_creates_new_field_target_without_epoch_pm_parallax"
        status: pass
    human_judgment: false
  - id: D3
    description: "Reusing an existing field Target never overwrites its epoch/pm_ra/pm_dec/parallax even when the incoming request carries values"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observation_records.py#test_flag_on_reuse_never_overwrites_epoch_pm_parallax"
        status: pass
    human_judgment: false
  - id: D4
    description: "All four 260722-tkt scenario tests (flag-off, create-new, reuse-existing, dry-run) still pass unchanged"
    verification:
      - kind: unit
        ref: "./manage.py test solsys_code.tests.test_backfill_lco_observation_records (16 tests, OK)"
        status: pass
    human_judgment: false

duration: 4min
completed: 2026-07-22
status: complete
---

# Quick Task 260722-twe Summary

**Brand-new SIDEREAL field Targets created by `--create-missing-targets` now carry epoch/pm_ra/pm_dec/parallax pulled straight from the LCO request's target dict, with no unit conversion needed and reused Targets left untouched.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-07-23T04:33:29Z
- **Completed:** 2026-07-23T04:36:37Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- `RequestTargetInfo` extended with `epoch`, `pm_ra`, `pm_dec`, `parallax`; `_request_target_info()` reads them from the LCO wire keys `epoch`/`proper_motion_ra`/`proper_motion_dec`/`parallax`
- `_resolve_or_build_field_target()` passes those four values into a newly-built `Target(...)` only — the reuse-existing branch is untouched
- Test helpers (`_configuration`, `_request`, `_field_request`) thread the four wire-key params through so tests can build realistic request payloads
- Three new tests cover: values-present mapping, values-absent -> None, and reuse-never-overwrites; all four 260722-tkt scenario tests still pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Extract epoch/proper-motion/parallax and pass them to newly-built field Targets** - `3735aa6` (feat)
2. **Task 2: Extend test helper and add coverage for the new fields, then run quality gates** - `ba59d0f` (test)

**Plan metadata:** committed separately by the orchestrator (docs commit)

## Files Created/Modified
- `solsys_code/management/commands/backfill_lco_observation_records.py` - `RequestTargetInfo` dataclass gains four fields; `_request_target_info` reads the matching LCO wire keys; `_resolve_or_build_field_target` passes them into newly-built Targets only
- `solsys_code/tests/test_backfill_lco_observation_records.py` - test helpers thread the four wire-key params; three new tests added

## Decisions Made
- No unit conversion needed — LCO wire units (Julian years, mas/yr, mas) match `Target.epoch`/`pm_ra`/`pm_dec`/`parallax` field units exactly, per the plan's pre-verified mapping.
- Kept the wire-key rename (`proper_motion_ra`/`proper_motion_dec` -> `pm_ra`/`pm_dec`) explicit in both the dataclass and the docstring, to avoid a naive-copy bug.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking scope hygiene] Ruff auto-fix/format ran repo-wide and touched unrelated files**
- **Found during:** Task 2 (quality-gate step)
- **Issue:** The plan's literal quality-gate commands (`ruff check . --fix` and `ruff format .`) operate on the whole repo, not just the two touched files. Running them modified unrelated notebook files, two `.planning/quick/260619-f7u-.../verify_*.py` scripts, and `src/fomo/settings.py` — none of which are in this task's scope.
- **Fix:** Reverted the unrelated files with `git checkout --` before staging/committing, keeping only the intended `solsys_code/tests/test_backfill_lco_observation_records.py` change. Verified `ruff check` and `ruff format --check` pass cleanly on the two task-scoped files directly (not repo-wide).
- **Files modified:** none beyond the plan's intended `solsys_code/tests/test_backfill_lco_observation_records.py`
- **Verification:** `git status --short` showed only the intended file after revert; targeted `ruff check`/`ruff format --check` on the two touched files both report clean.
- **Committed in:** not committed (reverted before staging)

---

**Total deviations:** 1 auto-fixed (scope-hygiene revert, no code behavior change)
**Impact on plan:** No scope creep; ensures the commit only contains this quick task's intended changes.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `backfill_lco_observation_records --create-missing-targets` now preserves epoch/proper-motion/parallax metadata for brand-new field Targets when the real LCO API supplies it.
- No further follow-up known for this command at this time.

---
*Phase: quick-260722-twe*
*Completed: 2026-07-22*

## Self-Check: PASSED

All claimed files and commits verified present:
- `solsys_code/management/commands/backfill_lco_observation_records.py` — FOUND
- `solsys_code/tests/test_backfill_lco_observation_records.py` — FOUND
- Commit `3735aa6` — FOUND
- Commit `ba59d0f` — FOUND
