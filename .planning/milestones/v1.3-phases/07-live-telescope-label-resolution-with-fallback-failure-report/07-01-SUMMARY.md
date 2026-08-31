---
phase: 07-live-telescope-label-resolution-with-fallback-failure-report
plan: 01
subsystem: api
tags: [django, tom-toolkit, lco-observation-portal, requests, management-command]

# Dependency graph
requires:
  - phase: 06-correct-instrument-type-extraction
    provides: "_extract_instrument() correctly scans c_1..c_5 configs for the real multi-config record shape, used by the interim _build_event_fields shim and the coarse fallback Plan 02 will build"
provides:
  - "7-site (site, aperture_class) -> 'SITECODE-CLASS' verified SITE_TELESCOPE_MAP (ogg, elp, lsc, cpt, coj, tfn, sor)"
  - "_aperture_class_from_telescope_code(telescope_code) -> str | None"
  - "_derive_telescope(site, telescope_code) -> str | None (2-arg, never-raise)"
  - "_resolve_placement_block(observation_id, facility) -> dict | None (single timeout-bounded API call, never raises, never logs the caught exception)"
  - "_API_TIMEOUT_SECONDS = 10 module constant"
affects: [07-02-live-telescope-label-resolution-with-fallback-failure-report]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Never-raise / sentinel-on-failure convention extended to _derive_telescope and the new _resolve_placement_block"
    - "Single try/except around make_request() + response.json() catching requests.exceptions.RequestException, ImproperCredentialsException, forms.ValidationError, and ValueError together, never referencing the caught exception"

key-files:
  created: []
  modified:
    - solsys_code/management/commands/sync_lco_observation_calendar.py
    - solsys_code/tests/test_sync_lco_observation_calendar.py
    - docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb

key-decisions:
  - "tlv (Wise Observatory) dropped entirely from SITE_TELESCOPE_MAP rather than shipped as an [ASSUMED] entry -- operator decision at the Task 1 checkpoint, since tlv is confirmed absent from both installed LCOSettings.get_sites() and SOARSettings.get_sites()"
  - "elp/lsc/cpt/tfn confirmed by operator (Tim Lister, LCO staff) as standard 1m-network sites hosting both 1m0 and 0m4 classes -- no [ASSUMED] tag needed, cited as operator-confirmed in the dict comment instead"
  - "Scope corrected from 8 sites to 7 sites at the Task 1 checkpoint (see Deviations below)"

patterns-established:
  - "checkpoint:human-verify gates for unverifiable external-data inventory claims (site/instrument facts the codebase's installed libraries can't corroborate) route to an explicit operator decision rather than shipping a guessed [ASSUMED] default"

requirements-completed: [TELESCOPE-01, TELESCOPE-02, SYNC-08, SYNC-09]

# Metrics
duration: ~50min (including the Task 1 checkpoint pause)
completed: 2026-06-23
status: complete
---

# Phase 07 Plan 01: Verified SITE_TELESCOPE_MAP + Live API Resolution Helpers Summary

**Migrated SITE_TELESCOPE_MAP to a verified 7-site (site, aperture_class) dict and added `_resolve_placement_block`/`_aperture_class_from_telescope_code`/2-arg `_derive_telescope` for single-attempt, timeout-bounded, never-leaking LCO Observation Portal API resolution.**

## Performance

- **Duration:** ~50 min (including the Task 1 human-verify checkpoint pause for operator confirmation)
- **Started:** 2026-06-23T04:46:54Z (per STATE.md "Phase 07 execution started")
- **Completed:** 2026-06-23T05:16:59Z
- **Tasks:** 3 (Task 1 checkpoint, Task 2 auto, Task 3 auto/tdd)
- **Files modified:** 3

## Accomplishments

- Replaced the flat 3-entry, `[ASSUMED]`-tagged `SITE_TELESCOPE_MAP` with a verified `(site, aperture_class) -> 'SITECODE-CLASS'` dict covering 7 real LCO-network sites, migrating the 3 pre-existing entries (`coj`->`COJ-2m0`, `ogg`->`OGG-2m0`, `sor`->`SOR-4m0`) and adding 4 new sites (`elp`, `lsc`, `cpt`, `tfn`) each with both `1m0` and `0m4` entries
- Added `_aperture_class_from_telescope_code` and the new 2-arg `_derive_telescope(site, telescope_code)`, both following the never-raise/sentinel convention (return `None` instead of raising `KeyError`)
- Added `_resolve_placement_block(observation_id, facility)`: a single, 10-second-timeout-bounded `make_request()` call to `/api/requests/{id}/observations/`, reusing the exact COMPLETED-first-else-PENDING block-selection logic from `OCSFacility.get_observation_status()` so telescope resolution and timing data always come from the same block
- Verified `_resolve_placement_block`'s except clause catches all 4 required exception/error types (`requests.exceptions.RequestException`, `ImproperCredentialsException`, `forms.ValidationError`, `ValueError`) and never references, stringifies, or logs the caught exception (SYNC-09/D-11)
- Added 5 new unit tests covering the verified dict's site coverage and label format, aperture-class parsing, the successful API-resolution path, the single-attempt-no-retry guarantee, and the no-credential/no-body-leak guarantee for both library exception types
- Regenerated the paired demo notebook (`sync_lco_observation_calendar_demo.ipynb`) per CLAUDE.md's demo-notebook-companion convention, since the dict migration changed the notebook's pre-executed output (`FTS`/`FTN` -> `COJ-2m0`/`OGG-2m0`)

## Task Commits

1. **Task 1: Confirm per-site aperture-class inventory for the verified dict** — checkpoint (human-verify, no commit; resolved via operator decision, see Deviations)
2. **Task 2: Migrate SITE_TELESCOPE_MAP to verified 7-site dict + add resolution helpers** - `8e89621` (feat)
3. **Task 3: Unit tests for verified dict, aperture-class parsing, single-attempt timeout, and no-leak logging** - `adb6552` (test)

_Note: Task 2's commit also includes the 4 pre-existing test-expectation updates (`'FTS'`/`'FTN'` -> `'COJ-2m0'`/`'OGG-2m0'`) since those literal labels no longer exist after the dict migration. Task 3's commit also includes the paired demo-notebook regeneration, since the same label change made the notebook's pre-executed output stale._

## Files Created/Modified

- `solsys_code/management/commands/sync_lco_observation_calendar.py` - Verified 7-site `SITE_TELESCOPE_MAP`; new `_API_TIMEOUT_SECONDS`, `_aperture_class_from_telescope_code`, `_resolve_placement_block`; 2-arg never-raise `_derive_telescope`; interim single-class shim in `_build_event_fields` (see Deviations)
- `solsys_code/tests/test_sync_lco_observation_calendar.py` - 5 new tests (`test_telescope_01_verified_dict_covers_all_sites`, `test_telescope_01_aperture_class_from_telescope_code`, `test_telescope_02_placed_record_resolves_via_api`, `test_sync_08_single_attempt_no_retry`, `test_sync_09_no_credential_or_body_leak_in_logs`); new `_observations_block_response` mock-response builder; 4 pre-existing test assertions updated for the migrated labels
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` - Regenerated via `jupyter nbconvert --to notebook --execute --inplace` to reflect the migrated `SITE_TELESCOPE_MAP` labels in its pre-executed output

## Decisions Made

- **tlv dropped entirely** (not shipped as `[ASSUMED]`): the operator (Tim Lister, LCO staff) confirmed at the Task 1 checkpoint that `tlv` should be excluded from `SITE_TELESCOPE_MAP` rather than guessed, since it is confirmed absent from both installed `LCOSettings.get_sites()` and `SOARSettings.get_sites()` — shipping a guessed entry for a site this codebase's installed library has never actually talked to would violate TELESCOPE-01's own "verified... real LCO-network sites" intent.
- **elp/lsc/cpt/tfn confirmed by the operator** as standard 1m-network sites hosting both `1m0` and `0m4` classes — cited in the code comment as "confirmed by operator (LCO staff) at the 07-01 Task 1 checkpoint" rather than `[ASSUMED]`.
- **Interim single-class shim in `_build_event_fields`** (not part of the original plan text, but required to keep the file's own existing 19 tests + verification green): Plan 01's objective explicitly scopes `_build_event_fields`/`Command.handle()` wiring out to Plan 02 (Wave 2), but changing `_derive_telescope`'s signature from 1-arg-raising to 2-arg-never-raise broke the one production call site at line 334 that the existing 19 regression tests exercise end-to-end via `call_command`. Added a small, clearly-commented interim lookup (`coj`/`ogg`/`sor` -> their single known aperture class) so `_build_event_fields` keeps producing identical behavior for the only 3 sites any existing test fixture uses, with an explicit code comment flagging that Plan 02 replaces this entire call with the live-API + fallback decision tree. This is a Rule 1/Rule 3 auto-fix (the signature change is itself net-new code Task 2 introduced; the call site break is a direct, in-scope consequence of that same task, not a pre-existing issue).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1/3 — blocking issue + bug] Production call site broken by `_derive_telescope`'s signature change**
- **Found during:** Task 2
- **Issue:** Replacing the old 1-arg `_derive_telescope(site_code) -> str` (raises `KeyError`) with the new 2-arg `_derive_telescope(site, telescope_code) -> str | None` left `_build_event_fields`'s only call site (`_derive_telescope(record.parameters['site'])`) calling the new function with the wrong arity, which would raise `TypeError` for every record and break all 19 pre-existing regression tests that exercise the command end-to-end.
- **Fix:** Added a small, explicitly-commented interim single-class lookup inline in `_build_event_fields` (`{'coj': '2m0', 'ogg': '2m0', 'sor': '4m0'}`) that resolves the legacy flat `parameters['site']` key through `SITE_TELESCOPE_MAP` using each site's single known class, preserving exact prior behavior for the 3 sites any existing fixture uses. The comment explicitly flags that Plan 02 (Wave 2) replaces this entire call with the live-API + fallback decision tree per TELESCOPE-02/03/04 — this shim is not meant to be load-bearing past Plan 02.
- **Files modified:** `solsys_code/management/commands/sync_lco_observation_calendar.py`
- **Verification:** `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` — all 22 then-existing tests pass; full `./manage.py test solsys_code` (122 tests) green.
- **Committed in:** `8e89621` (part of Task 2's commit)

**2. [Rule 2 — missing critical functionality, CLAUDE.md demo-notebook convention] Paired demo notebook went stale**
- **Found during:** Task 3 (post-implementation check, before writing this summary)
- **Issue:** `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`'s pre-executed output (cells 9 and 13) showed the old `'FTS'` telescope label, which is now stale after Task 2's `SITE_TELESCOPE_MAP` migration changed that same fixture's resolved label to `'COJ-2m0'`. CLAUDE.md's demo-notebook-companion convention (added after two prior recurrences in Phases 5 and 6) requires this notebook stay in sync with any behavior change to `sync_lco_observation_calendar.py`, and the plan's own `<files_modified>` frontmatter omitted it for Plan 01.
- **Fix:** Regenerated via `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` (with the required `DJANGO_SETTINGS_MODULE`/`DJANGO_ALLOW_ASYNC_UNSAFE` env vars). Confirmed the regenerated output shows `'COJ-2m0'` instead of `'FTS'` and no remaining `'FTS'`/`'FTN'` strings anywhere in the notebook.
- **Files modified:** `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`
- **Verification:** Parsed the regenerated notebook's JSON and confirmed (a) output cells are non-empty (pre-commit's output-clearing hook correctly excludes `pre_executed/`), (b) no `'FTS'`/`'FTN'` strings remain, (c) `'COJ-2m0'` appears in the expected cells.
- **Committed in:** `adb6552` (part of Task 3's commit)

**Total deviations:** 2 auto-fixed (1 Rule 1/3, 1 Rule 2). **Impact on plan:** Both were necessary to keep the plan's own verification gate (`./manage.py test solsys_code` fully green) and CLAUDE.md's mandatory conventions satisfied. No scope creep — neither deviation implements any of Plan 02's TELESCOPE-02/03/04/SYNC-06/07 wiring; the interim shim is explicitly marked for removal when Plan 02 lands.

### Checkpoint Resolution — Scope Correction (8 sites -> 7 sites)

At the Task 1 `checkpoint:human-verify` gate, the operator (Tim Lister, LCO staff) confirmed the per-site aperture-class inventory and made an explicit scope decision: **drop `tlv` entirely** rather than ship it as a single `[ASSUMED]` entry, since `tlv` is confirmed absent from both installed `LCOSettings.get_sites()` and `SOARSettings.get_sites()` in this codebase — it is not a real, reachable LCO-network site for this FOMO installation. The operator's framing: "all real LCO-network sites" (TELESCOPE-01's own wording) means the 7 sites this codebase's installed library actually confirms (`ogg`, `elp`, `lsc`, `cpt`, `coj`, `tfn`, `sor`), not the 8-site table from PROJECT.md's external MPC-code reference (which is sourced from a website, not this codebase's facility configuration).

This is a legitimate scope correction surfaced by the checkpoint, applied per the operator's explicit instruction:
- Task 2's acceptance criteria and SITE_TELESCOPE_MAP now cover 7 sites, not 8; no `tlv` entry exists.
- Task 3's `test_telescope_01_verified_dict_covers_all_sites` asserts the 7-site set `{ogg, elp, lsc, cpt, coj, tfn, sor}`.
- This plan's `<success_criteria>` TELESCOPE-01 line is satisfied as "verified 7-site... dict (tlv dropped — confirmed absent from installed LCOSettings/SOARSettings, operator-approved at the 07-01 Task 1 checkpoint)" rather than the original plan text's "8-site... dict... tlv [ASSUMED]".

**Flag for the orchestrator / future phase work:** ROADMAP.md and CONTEXT.md's original framing for this phase describes an "8 real LCO-network sites" inventory. This SUMMARY documents the operator-approved correction to 7 sites so Plan 07-02 and any later phase work isn't confused by the discrepancy. This executor did **not** edit ROADMAP.md's phase-level success-criteria wording — `roadmap update-plan-progress` below only updates the plan-progress table row counts, not the phase's narrative success-criteria text. Reconciling ROADMAP.md's "8 sites" language (if it needs updating) is left for the orchestrator/phase-completion step, since ROADMAP.md is a shared artifact spanning both plans of this phase.

## Issues Encountered

None beyond the two auto-fixed deviations above.

## User Setup Required

None - no external service configuration required. This phase consumes the existing `LCO_APIKEY` env var already configured for prior phases; no new credentials or settings are introduced.

## Next Phase Readiness

Plan 02 (Wave 2, `07-02-PLAN.md`) is ready to proceed: `_resolve_placement_block`, `_aperture_class_from_telescope_code`, and the 2-arg `_derive_telescope` all exist with their final signatures and never-raise behavior, exactly as Plan 02's `<read_first>` expects ("FINAL state after Plan 01"). Plan 02 will replace this plan's interim single-class shim in `_build_event_fields` entirely with the live-API + fallback decision tree (TELESCOPE-02/03/04, SYNC-06/07/09), wire the `[UNVERIFIED]` prefix into `_title_for`, add the `telescope_api_failed` counter, and update both the test file and the demo notebook again for the new observable behavior.

No blockers. One non-blocking note carried forward: the 8-vs-7-site scope discrepancy between ROADMAP.md/CONTEXT.md's original framing and this plan's operator-approved correction (see above) should be reconciled by the orchestrator at phase completion, not by this plan-level executor.

## Self-Check: PASSED

- FOUND: `solsys_code/management/commands/sync_lco_observation_calendar.py`
- FOUND: `solsys_code/tests/test_sync_lco_observation_calendar.py`
- FOUND: `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`
- FOUND commit: `8e89621`
- FOUND commit: `adb6552`

---
*Phase: 07-live-telescope-label-resolution-with-fallback-failure-report*
*Completed: 2026-06-23*
