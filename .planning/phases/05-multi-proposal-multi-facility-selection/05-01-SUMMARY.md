---
phase: 05-multi-proposal-multi-facility-selection
plan: 01
subsystem: api
tags: [django, tom-toolkit, management-command, lco, soar, observation-record]

# Dependency graph
requires:
  - phase: 04-lco-queue-sync-command
    provides: sync_lco_observation_calendar baseline (single-proposal, LCO-only sync; CalendarEvent upsert pattern; terminal-state title prefixes)
provides:
  - "_parse_proposal_arg(raw) -> list[str] | None: comma-list/ALL proposal parsing helper"
  - "Eager LCO+SOAR facility dispatch dict in handle(), keyed by record.facility"
  - "facility__in queryset base filter, conditional parameters__proposal__in clause"
  - "Per-facility created/updated/unchanged/skipped counters and summary line"
  - "FACILITIES['SOAR'] settings entry mirroring FACILITIES['LCO']"
  - "_create_record(facility=...) test-fixture parameter for multi-facility fixtures"
affects: [06-instrument-type-extraction, 07-telescope-label-api-fallback]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Eager (not lazy/conditional) per-facility instance dispatch dict, built once before the record loop, looked up by record.facility -- avoids reusing one shared facility instance across heterogeneous facility records"
    - "Discriminating spy test pattern (patch.object(..., autospec=True, side_effect=real_method)) for proving which subclass instance handled a call, when both classes return byte-identical outputs (inherited methods)"

key-files:
  created: []
  modified:
    - solsys_code/management/commands/sync_lco_observation_calendar.py
    - solsys_code/tests/test_sync_lco_observation_calendar.py
    - src/fomo/settings.py

key-decisions:
  - "FACILITIES['SOAR'] mirrors FACILITIES['LCO']'s literal values exactly (portal_url, api_key='') rather than introducing os.getenv('LCO_API_KEY', ...) for either facility -- the narrower RESEARCH.md reading, keeping this phase scoped to query/selection/dispatch only, not credentials plumbing"
  - "Proposal codes are treated as case-sensitive in _parse_proposal_arg; only the 'ALL' sentinel is case-insensitive (D-01/D-02)"
  - "Per-facility summary line keeps the existing 'created: N, updated: N, unchanged: N, skipped: N' phrasing per facility (not a different format) so existing substring-based test assertions ('created: 0', 'updated: 1', 'unchanged: 1') keep passing unchanged"

requirements-completed: [SELECT-02, SELECT-03, SELECT-04, SELECT-05]

# Metrics
duration: 35min
completed: 2026-06-19
---

# Phase 05 Plan 01: Multi-Proposal, Multi-Facility Selection Summary

**Generalized `sync_lco_observation_calendar` to accept a comma-list/ALL `--proposal` argument and dispatch LCO and SOAR `ObservationRecord`s through their own facility instance, fixing the SELECT-05 single-shared-`LCOFacility()` dispatch bug.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-06-19T16:21:00Z (approx)
- **Completed:** 2026-06-19T16:56:14Z
- **Tasks:** 3 completed
- **Files modified:** 3

## Accomplishments

- `_parse_proposal_arg()` parses `--proposal` into either the `ALL` sentinel (`None`, case-insensitive) or a deduped, order-preserving, case-sensitive list of proposal codes, dropping empty comma segments.
- `handle()` now queries `ObservationRecord.objects.filter(facility__in=['LCO', 'SOAR'])` with an optional `parameters__proposal__in=codes` clause (omitted entirely for `ALL`), and dispatches each record through an eagerly-built `{'LCO': LCOFacility(), 'SOAR': SOARFacility()}` dict keyed by `record.facility` — never a single reused instance.
- Per-facility counters (`created`/`updated`/`unchanged`/`skipped`) replace the old flat ints; the end-of-run summary line reports each facility's four counts individually.
- `FACILITIES['SOAR']` added to `src/fomo/settings.py`, mirroring `FACILITIES['LCO']` literal values, so `SOARSettings('SOAR').get_setting('api_key')` resolves a real key.
- Four new tests (`test_select_02`..`test_select_05`) validate substring-safe comma-list matching, case-insensitive `ALL`, single-run dual-facility coverage, and — critically — a discriminating spy proving the SOAR record was processed by `SOARFacility`, not a reused `LCOFacility`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Wave-0 scaffold — FACILITIES['SOAR'] settings entry + _create_record facility param** - `adc5a61` (feat)
2. **Task 2: Generalize the command — proposal parsing, multi-facility queryset, dispatch dict, per-facility summary** - `81c7cad` (feat)
3. **Task 3: Add SELECT-02/03/04/05 tests (with discriminating spy for SELECT-05)** - `61a1c80` (test)

_Note: this plan's `tdd="true"` was not set at the task level; tests were added as the final task per the plan's own task ordering (settings/fixture scaffold -> command logic -> tests), not a RED/GREEN/REFACTOR cycle._

## Files Created/Modified

- `src/fomo/settings.py` - Added `FACILITIES['SOAR']` dict entry (portal_url, api_key='') directly after `FACILITIES['LCO']`; `'LCO'` entry left byte-for-byte unchanged.
- `solsys_code/management/commands/sync_lco_observation_calendar.py` - New `_parse_proposal_arg()` helper; `SOARFacility` import; `handle()` rewritten for multi-facility dispatch dict, `facility__in` + conditional `parameters__proposal__in` queryset, per-facility counters dict, defensive skip-and-log path for an unrecognized `record.facility`, per-facility summary line.
- `solsys_code/tests/test_sync_lco_observation_calendar.py` - `_create_record()` gained a `facility: str = 'LCO'` keyword param; added `SOARFacility` import; added four new test methods (`test_select_02_comma_list_matches_any_no_substring_leakage`, `test_select_03_all_token_case_insensitive_syncs_everything`, `test_select_04_single_run_covers_both_facilities`, `test_select_05_soar_record_uses_soar_facility_instance`).

## Decisions Made

- Mirrored `FACILITIES['LCO']`'s literal values for the new `'SOAR'` entry rather than introducing an env-var-driven `api_key` for either facility — keeps the phase scoped to query/selection/dispatch (per RESEARCH.md's narrower-reading recommendation for Open Question 1).
- Kept the existing `'created: N, updated: N, ...'` phrasing inside the per-facility breakdown (rather than a `'N created'` word order) specifically so the pre-existing substring assertions in `test_sync_04_rerun_updates_in_place_no_churn_on_unchanged` and `test_zero_match_reports_created_zero_no_command_error` continue to pass without modification — confirmed this was the deciding factor after an initial `'{n} created'` draft broke those two existing tests.
- Used `unittest.mock.patch.object(..., autospec=True, side_effect=real_method)` instead of `wraps=Class.method` for the SELECT-05 spy: `wraps` with an unbound method reference does not correctly bind `self` when patching a class method, raising `TypeError: missing 1 required positional argument: 'observation_id'`; `autospec=True` + `side_effect=<unbound real method>` correctly preserves the bound-call signature while still spying.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Missing `src/fomo/_version.py` blocked all Django test runs in this worktree**
- **Found during:** Task 1 verification (first `./manage.py test` invocation)
- **Issue:** `src/fomo/_version.py` is a gitignored, `setuptools_scm`-generated file produced at editable-install time. It exists in the main repo checkout (where `pip install -e .` was originally run) but is absent in this fresh git worktree, causing `ModuleNotFoundError: No module named 'src.fomo._version'` on every Django/manage.py invocation. `setuptools_scm` itself is a build-time-only dependency, not installed in the runtime venv, so it couldn't be regenerated via `python -m setuptools_scm`.
- **Fix:** Copied the already-generated `_version.py` from the main repo checkout into this worktree's `src/fomo/` directory (file is gitignored — not committed, not tracked, has zero effect on the actual commits/diff for this plan).
- **Files modified:** `src/fomo/_version.py` (gitignored, not committed)
- **Verification:** `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` ran successfully afterward.
- **Committed in:** N/A — gitignored file, not part of any task commit.

**2. [Rule 1 - Bug] Pre-commit ruff `--fix` removed the unused `SOARFacility` import added in Task 1 before it was used (intended for Task 3)**
- **Found during:** Task 1 commit (pre-commit hook auto-fix)
- **Issue:** The plan instructs adding the `SOARFacility` import to the test module in Task 1 "for the SELECT-05 spy test in Task 3," but since it's genuinely unused until Task 3's tests exist, `ruff check --fix` (run automatically by the pre-commit hook) removed it on the first commit attempt, causing the commit to abort (pre-commit hooks that modify files fail the first attempt and require restaging).
- **Fix:** Re-staged the ruff-fixed file (import removed) and committed Task 1 without the premature import; re-added `from tom_observations.facilities.soar import SOARFacility` in Task 3 when it became actually used by the new spy test.
- **Files modified:** `solsys_code/tests/test_sync_lco_observation_calendar.py`
- **Verification:** `ruff check` clean at both Task 1 and Task 3 commit points; full Django suite green at both points.
- **Committed in:** `adc5a61` (Task 1, import absent), `61a1c80` (Task 3, import added and used)

---

**Total deviations:** 2 auto-fixed (1 blocking environment issue, 1 bug/lint-driven sequencing adjustment)
**Impact on plan:** Neither affected the shipped command/settings logic. The `_version.py` workaround is a local dev-environment artifact with no repo footprint. The import-sequencing adjustment only changed *which* task commit the `SOARFacility` import landed in (Task 3 instead of Task 1) — the final diff and behavior match the plan's intent exactly.

## Issues Encountered

- Initial SELECT-05 spy implementation used `patch.object(SOARFacility, 'get_observation_url', wraps=SOARFacility.get_observation_url)`, which raised `TypeError: OCSFacility.get_observation_url() missing 1 required positional argument: 'observation_id'` because `wraps` with an unbound class-level method reference doesn't rebind `self` correctly when the patched attribute is called on an instance. Resolved by switching to `autospec=True, side_effect=real_get_observation_url` (captured once via `LCOFacility.get_observation_url` before patching either class, since both inherit the same implementation).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 06 (instrument-type extraction) and Phase 07 (telescope-label API + fallback) both build on this phase's now-correct multi-facility queryset/dispatch; `_build_event_fields()`'s site/instrument-type extraction itself was explicitly NOT touched this phase (still assumes flat `parameters['site']`/`parameters['instrument_type']` keys) — that remains Phase 06/07 scope, unchanged here.
- All four target requirements (SELECT-02/03/04/05) have passing, requirement-ID-traceable tests.
- `ruff check .` / `ruff format --check .` are clean for all three files this plan touched; two pre-existing, untouched files (`docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`, `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`) and `src/fomo/settings.py`'s unrelated module-docstring blank-line formatting remain flagged by `ruff format --check .` repo-wide — already tracked in `.planning/phases/04-lco-queue-sync-command/deferred-items.md`, not newly introduced.

---
*Phase: 05-multi-proposal-multi-facility-selection*
*Completed: 2026-06-19*
