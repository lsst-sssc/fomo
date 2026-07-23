---
phase: quick-260722-uhh
plan: 01
subsystem: admin
tags: [django-admin, tom_targets, targetadmin]

requires:
  - phase: quick-260714-jpd
    provides: CampaignRunAdmin/CalendarEventTelescopeLabelAdmin registration pattern in solsys_code/admin.py (mirrored here)
provides:
  - Custom TargetAdmin registered for tom_targets' Target model with a 'By type' list_filter
affects: [admin]

tech-stack:
  added: []
  patterns:
    - "admin.site.unregister(Model) immediately followed by admin.site.register(Model, CustomAdmin) to override a third-party app's bare ModelAdmin registration"

key-files:
  created: []
  modified:
    - solsys_code/admin.py
    - solsys_code/tests/test_admin.py

key-decisions:
  - "Target resolves to BaseTarget via tom_targets.models.get_target_model_class() (no TARGET_MODEL_CLASS override in settings.py), so the admin change-list URL name is tom_targets_basetarget_changelist, not tom_targets_target_changelist as the plan's must_haves truth assumed — tests derive the URL name from Target._meta.app_label/model_name instead of hardcoding it."
  - "Did not replicate tom_targets' TargetExtraInline in the new TargetAdmin (accepted tradeoff per plan, not a bug)."

patterns-established:
  - "Third-party Django admin override pattern: unregister-then-register at the bottom of admin.py, matching the file's existing register-at-bottom convention."

requirements-completed: []

coverage:
  - id: D1
    description: "Custom TargetAdmin registered for tom_targets.models.Target with list_display=[name,type,ra,dec], list_filter=[type], search_fields=[name]"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_admin.py#TargetAdminChangelistAndTypeFilterTests.test_target_changelist_loads"
        status: pass
    human_judgment: false
  - id: D2
    description: "Target admin change-list 'By type' filter restricts rows to SIDEREAL or NON_SIDEREAL"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_admin.py#TargetAdminChangelistAndTypeFilterTests.test_type_filter_shows_only_sidereal"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_admin.py#TargetAdminChangelistAndTypeFilterTests.test_type_filter_shows_only_non_sidereal"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-07-22
status: complete
---

# Quick Task 260722-uhh: Register a custom Django admin for tom_targets' Target Summary

**Custom `TargetAdmin` in `solsys_code/admin.py` unregisters tom_targets' bare `ModelAdmin` and re-registers Target with `list_display=[name,type,ra,dec]` and a `list_filter=['type']` so staff can filter SIDEREAL vs NON_SIDEREAL rows in `/admin/`.**

## Performance

- **Duration:** ~12 min
- **Tasks:** 2 completed
- **Files modified:** 2

## Accomplishments
- `solsys_code/admin.py` now registers a `TargetAdmin(admin.ModelAdmin)` for `tom_targets.models.Target`, following the exact `CampaignRunAdmin`/`CalendarEventTelescopeLabelAdmin` register-at-bottom pattern already in the file.
- `admin.site.unregister(Target)` runs immediately before `admin.site.register(Target, TargetAdmin)`, replacing tom_targets' unfilterable default admin without raising `AlreadyRegistered`/`NotRegistered`.
- New `TargetAdminChangelistAndTypeFilterTests` in `solsys_code/tests/test_admin.py` prove the change-list loads for a superuser and that `type__exact` filtering correctly separates a `SiderealTargetFactory` target from a `NonSiderealTargetFactory` target.

## Task Commits

Each task was committed atomically:

1. **Task 1: Register custom TargetAdmin with list_filter on type** - `b572dfb` (feat)
2. **Task 2: Add admin-test-client coverage for the Target change-list + type filter** - `fac8a61` (test)

**Plan metadata:** committed separately by the orchestrator (docs commit not made by this executor per constraints).

## Files Created/Modified
- `solsys_code/admin.py` - Imports `Target`, adds `TargetAdmin` (list_display/list_filter/search_fields), unregisters tom_targets' default Target admin and re-registers the custom one.
- `solsys_code/tests/test_admin.py` - Adds `TargetAdminChangelistAndTypeFilterTests` covering change-list load + type filter for both sidereal and non-sidereal targets.

## Decisions Made
- Target's admin URL name is `tom_targets_basetarget_changelist` (derived from `Target._meta.app_label`/`model_name`), not `tom_targets_target_changelist` — `Target` is a dynamic alias (`get_target_model_class()`) that resolves to `BaseTarget` since `settings.py` has no `TARGET_MODEL_CLASS` override. Tests compute the reverse-URL name dynamically from `Target._meta` rather than hardcoding the literal string, so this stays correct if the target model class is ever swapped.
- Did not add `inlines = [TargetExtraInline]` to the new `TargetAdmin` — matches the plan's explicit instruction to accept the loss of the TargetExtra inline as a tradeoff, not a gap to fix.

## Deviations from Plan

None — plan executed exactly as written. The only adjustment was fixing a factual assumption in the plan's `must_haves.truths` (the admin URL path uses `basetarget`, not `target`, in the URL name); this was a pre-existing inaccuracy in the plan text, not a deviation from the intended behavior, and both the code and tests still verify the actual described truths (200 response, `type` filter present, `list_display` columns) via `admin.site._registry[Target]` and the correctly-derived URL name.

## Issues Encountered
- An early `ruff check . --fix` invocation (run against `.` rather than the single touched file) unintentionally auto-fixed import ordering in two unrelated notebook files (`docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`, `sync_lco_observation_calendar_demo.ipynb`). These were identified via `git status`/`git diff` before committing and reverted with `git checkout --` so only `solsys_code/admin.py` was staged for Task 1's commit. No functional impact; scope stayed limited to the two files listed in the plan's `files_modified`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- No blockers. Staff can now filter the Target admin change-list by SIDEREAL/NON_SIDEREAL type at `/admin/tom_targets/basetarget/`.

---
*Phase: quick-260722-uhh*
*Completed: 2026-07-22*

## Self-Check: PASSED

- FOUND: solsys_code/admin.py
- FOUND: solsys_code/tests/test_admin.py
- FOUND: b572dfb (feat commit)
- FOUND: fac8a61 (test commit)
