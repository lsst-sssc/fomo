---
phase: 04-lco-queue-sync-command
plan: 01
subsystem: infra
tags: [django-management-command, tom-toolkit, tom_observations, tom_calendar, lco-facility, jsonfield, tdd]

# Dependency graph
requires:
  - phase: 03-classical-run-ingest
    provides: load_telescope_runs.py upsert pattern (get_or_create + conditional save), stdout summary convention, per-item try/except skip pattern
provides:
  - sync_lco_observation_calendar management command that creates/updates one CalendarEvent per matching LCO ObservationRecord, keyed on the real LCO portal URL
  - SITE_TELESCOPE_MAP (LCO site code -> telescope label) and helper functions (_derive_telescope, _title_for, _time_window, _build_event_fields, _failure_prefix)
affects: [Stage 4 future work (full LCO facility sync), any future calendar-sync command for other facilities]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "create-or-update keyed on a single field (url) with conditional save comparing all changeable fields before .save() — avoids modified-timestamp churn on unchanged records"
    - "use library helper methods (LCOFacility().get_failed_observing_states(), get_observation_url()) instead of hand-rolling status lists or URL strings"

key-files:
  created:
    - solsys_code/management/commands/sync_lco_observation_calendar.py
    - solsys_code/tests/test_sync_lco_observation_calendar.py
  modified: []

key-decisions:
  - "D-06 research correction implemented: prefix trigger uses LCOFacility().get_failed_observing_states() (4 states), not get_terminal_observing_states() (5 states, includes COMPLETED) — COMPLETED records get a clean title, same as any other placed record"
  - "COMPLETED test fixture corrected to set scheduled_start/scheduled_end — a completed observation has necessarily been placed by the LCO scheduler, so scheduled_start=None would be an unrealistic fixture, not a real test of D-06"

patterns-established:
  - "Pattern: terminal-failure prefix lookup built from facility.get_failed_observing_states() membership check + a small status->prefix dict, rather than hardcoding the failure-state list itself"

requirements-completed: [SELECT-01, SYNC-01, SYNC-02, SYNC-03, SYNC-04, SYNC-05, TERM-01]

# Metrics
duration: 35min
completed: 2026-06-17
---

# Phase 4 Plan 01: LCO Queue Sync Command Summary

**`sync_lco_observation_calendar` management command syncs LCO ObservationRecords to CalendarEvents via TDD, keyed on the real `LCOFacility().get_observation_url()` portal URL, with no-churn create-or-update and a terminal-failure title prefix system that correctly excludes COMPLETED (D-06 research correction).**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-06-17T21:11:52Z (per init context)
- **Completed:** 2026-06-17T21:36:46Z
- **Tasks:** 3/3 completed
- **Files modified:** 2 (1 command file, 1 test file) + 1 deviation-tracking file (deferred-items.md)

## Accomplishments

- `sync_lco_observation_calendar --proposal <code>` filters `ObservationRecord(facility='LCO', parameters__proposal=code)` via ORM-level JSONField lookup and creates/updates one `CalendarEvent` per matching record
- Idempotency key is the real LCO portal URL (`LCOFacility().get_observation_url(observation_id)`), never the stale `requestgroups/<id>/` format
- Two time-source branches correctly implemented: `parameters['start']`/`['end']` (parsed to aware UTC) when `scheduled_start is None`, `scheduled_start`/`scheduled_end` once placed
- Terminal-failure title prefixes (`[EXPIRED]`/`[CANCELLED]`/`[FAILED]`) derived from `LCOFacility().get_failed_observing_states()` membership, with `COMPLETED` deliberately excluded (D-06) so it gets a clean title like any other placed record
- No-churn create-or-update: all 7 changeable fields compared before `.save()`, verified by a test asserting `modified` is untouched on an unchanged record across two runs
- 14 new Django tests, full `solsys_code` suite (109 tests) green, `ruff check .` / `ruff format --check .` clean for all files this plan touched

## Task Commits

Each task was committed atomically:

1. **Task 1: Site map, test fixtures, and failing tests for selection + sync + terminal states** - `c3b74cb` (test) — RED: 14 tests written, stub command added, all fail
2. **Task 2: Implement the sync_lco_observation_calendar command (GREEN)** - `1b49957` (feat) — GREEN: full implementation, all 14 tests pass
3. **Task 3: Full suite + lint quality gate** - `2af4792` (chore) — verification only; logged 2 pre-existing unrelated ruff-format findings to deferred-items.md

_TDD tasks 1 and 2 form the RED/GREEN pair; no REFACTOR commit was needed since GREEN already satisfied all acceptance criteria and lint cleanly._

## Files Created/Modified

- `solsys_code/management/commands/sync_lco_observation_calendar.py` - the `sync_lco_observation_calendar` command: `SITE_TELESCOPE_MAP`, `Command(BaseCommand)` with `--proposal`, `handle()`, and helpers `_derive_telescope`, `_failure_prefix`, `_title_for`, `_time_window`, `_build_event_fields`
- `solsys_code/tests/test_sync_lco_observation_calendar.py` - `TestSyncLcoObservationCalendar(TestCase)`, 14 test methods covering SELECT-01, SYNC-01..05, TERM-01 (all 4 failure states), D-06, no-churn idempotency, skip path, and zero-match reporting
- `.planning/phases/04-lco-queue-sync-command/deferred-items.md` - records two pre-existing, plan-unrelated `ruff format` findings (out of scope per scope boundary rule)

## Decisions Made

- Used `LCOFacility().get_failed_observing_states()` (not `get_terminal_observing_states()`) as the failure-prefix trigger, per the plan's locked D-06 decision — confirmed by an explicit grep-based acceptance check (`get_terminal_observing_states` count is 0 outside comments) and a passing COMPLETED-record test
- Corrected the COMPLETED test fixture (added in Task 1, fixed during Task 2's GREEN work) to set `scheduled_start`/`scheduled_end` — a record that reached `COMPLETED` status must have been observed, i.e. placed by the scheduler; the original RED-phase fixture left `scheduled_start=None`, which produced a `[QUEUED]`-prefixed title instead of testing the intended clean-title branch. This is a test-fixture bug fix (Rule 1), not a change to locked plan decisions or source behavior.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed unrealistic COMPLETED test fixture**
- **Found during:** Task 2 (GREEN implementation) — running the Task-1 RED tests against the real implementation surfaced that the D-06 test was asserting against an inconsistent fixture (COMPLETED status but `scheduled_start=None`, which the implementation correctly classified as `[QUEUED]` per its own scheduling-state branch, contradicting the test's clean-title expectation)
- **Issue:** Test fixture for the D-06 COMPLETED case did not set `scheduled_start`/`scheduled_end`, producing a `[QUEUED]` title instead of testing the intended COMPLETED-gets-clean-title behavior
- **Fix:** Added `scheduled_start`/`scheduled_end` to the fixture in `test_d06_completed_gets_clean_title_no_prefix`
- **Files modified:** `solsys_code/tests/test_sync_lco_observation_calendar.py`
- **Verification:** Test now passes; full 14-test suite green
- **Committed in:** `1b49957` (part of Task 2 commit)

**2. [Rule 1 - Bug] Switched hand-typed failure-state dict to a library-backed membership check**
- **Found during:** Task 2 (GREEN implementation) — Task 3's acceptance criteria require the source to actually call `get_failed_observing_states()` (RESEARCH.md's "Don't Hand-Roll" guidance), not just hardcode the same 4 strings
- **Issue:** Initial implementation used a flat `_FAILURE_PREFIXES` dict keyed directly on status strings, satisfying the test suite but not calling `LCOFacility().get_failed_observing_states()` as the plan's Task 2 action explicitly required
- **Fix:** Added `_failure_prefix(status, facility)` which checks membership against `set(facility.get_failed_observing_states())` before looking up the prefix, so a future library change to the failure-state set is picked up automatically
- **Files modified:** `solsys_code/management/commands/sync_lco_observation_calendar.py`
- **Verification:** `grep -c 'get_failed_observing_states'` returns 3 (definition + 2 call sites); all 14 tests still pass
- **Committed in:** `1b49957` (part of Task 2 commit)

**3. [Rule 1 - Bug] Wrapped overlong description f-string to satisfy 120-col ruff line length**
- **Found during:** Task 2 (GREEN implementation), pre-commit ruff check
- **Issue:** Single-line f-string for `description` was 132 characters, failing `E501`
- **Fix:** Split into a parenthesized multi-line f-string concatenation
- **Files modified:** `solsys_code/management/commands/sync_lco_observation_calendar.py`
- **Verification:** `ruff check .` and `ruff format --check .` both pass
- **Committed in:** `1b49957` (part of Task 2 commit)

---

**Total deviations:** 3 auto-fixed (2 Rule 1 bug fixes in test correctness/library-usage, 1 Rule 1 lint-line-length fix)
**Impact on plan:** All three were necessary to make the implementation match the plan's explicit Task 2 requirements (D-06 correctness, `get_failed_observing_states()` usage) and the project's ruff line-length convention. No scope creep — no new files, no architectural changes.

## Issues Encountered

- `ruff format --check .` (repo-wide, run for Task 3's phase-level verification) flagged two pre-existing files (`src/fomo/settings.py`, `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`) unrelated to this plan's changes. Confirmed via `git status --short` that neither file was touched by this plan; logged to `.planning/phases/04-lco-queue-sync-command/deferred-items.md` per the scope-boundary rule rather than fixed. `ruff check .` (lint) is fully clean repo-wide; only `ruff format` flags these two pre-existing files.

## User Setup Required

None - no external service configuration required. No new packages installed (confirmed: all imports are from already-installed `tomtoolkit`/Django, per RESEARCH.md's Package Legitimacy Audit).

## Known Stubs

None. `SITE_TELESCOPE_MAP` values (`'coj': 'FTS'`, `'ogg': 'FTN'`) are flagged `[ASSUMED]` per RESEARCH.md Assumptions Log A1/A2 (web-search only, not yet confirmed against a real `ObservationRecord.parameters['site']` value from this project's actual LCO proposal data) — this is an explicit, documented assumption inline in the source code (not a stub or placeholder), and does not block the command's correctness for any site code that IS in the map; an unmapped site code correctly routes to the skip path (tested by `test_skip_path_missing_site_logged_and_skipped`) rather than crashing or silently mislabeling.

## Threat Flags

None. All three threat-register items disposed `mitigate` in the plan's `<threat_model>` (T-04-02, T-04-03, T-04-04) are implemented as specified: per-record `KeyError`/`ValueError` skip-and-log (T-04-02), `url` built exclusively via `LCOFacility().get_observation_url()` rather than from raw parameters (T-04-03), and an always-emitted stdout summary including explicit `created: 0` on a no-op run (T-04-04, verified by `test_zero_match_reports_created_zero_no_command_error`). No new network endpoints, auth paths, or schema changes were introduced — this phase adds no models/migrations.

## Next Phase Readiness

- Phase 4 success criteria 1, 2, and 4 (full criteria list in PLAN.md `<success_criteria>`) are satisfied: `./manage.py test solsys_code` passes (109/109), `ruff check .` clean, `ruff format --check .` clean for all plan-touched files
- Criterion 3 (manual spot-check with a real fixture + command run) is optional/end-of-phase human verify per the plan — not exercised in this automated run but the automated test suite already covers the equivalent behavior end-to-end
- `SITE_TELESCOPE_MAP`'s two site-code values remain `[ASSUMED]` (A1/A2 in RESEARCH.md) — if a future real `ObservationRecord` from this project's LCO proposal data has a `parameters['site']` value other than `'coj'`/`'ogg'`, it will hit the documented skip path (logged to stderr, not silently mismapped) until the map is extended
- No blockers for closing Phase 4 / milestone v1.2

---
*Phase: 04-lco-queue-sync-command*
*Completed: 2026-06-17*

## Self-Check: PASSED

- FOUND: solsys_code/management/commands/sync_lco_observation_calendar.py
- FOUND: solsys_code/tests/test_sync_lco_observation_calendar.py
- FOUND: .planning/phases/04-lco-queue-sync-command/04-01-SUMMARY.md
- FOUND commit: c3b74cb (test)
- FOUND commit: 1b49957 (feat)
- FOUND commit: 2af4792 (chore)
- FOUND commit: 85b46ad (docs)
