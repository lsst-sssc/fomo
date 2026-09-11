---
phase: 34-the-observation-projector-trigger
plan: 05
subsystem: calendar-sync
tags: [django, calendar, observation-projector, tdd, bugfix, gap-closure]

# Dependency graph
requires:
  - phase: 34-the-observation-projector-trigger
    provides: "34-01/34-02: the observation projector (event_fields_for, record_time_window, post_save receiver) and record_time_window's promotion into calendar_utils.py"
provides:
  - "coerce_schedule_datetime() in calendar_utils.py: coerces a schedule field (datetime | str | None) to an aware UTC datetime, raising ValueError on anything unusable"
  - "record_time_window()'s both-populated branch routed through coerce_schedule_datetime(), so a post-save in-memory instance holding portal ISO strings projects identically to a DB-fetched record"
  - "regression coverage for both the direct coercion contract and the real update_observation_status() path"
affects: [34-06, campaign_attribution.py, observation_projector.py]

# Actuals (#2632)
actuals:
  tokens: 4309
  tasks: 2
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Schedule-value coercion at the calendar_utils.py boundary: a single coerce_schedule_datetime() helper absorbs the datetime-vs-portal-string divergence between the post_save receiver's in-memory path and the sweep's DB-fetched path, instead of teaching every downstream consumer (event_fields_for, campaign_attribution matcher) to handle both shapes."

key-files:
  created: []
  modified:
    - solsys_code/calendar_utils.py
    - solsys_code/tests/test_calendar_utils.py
    - solsys_code/tests/test_observation_projector_signals.py

key-decisions:
  - "coerce_schedule_datetime() raises ValueError for an unparseable/unusable value rather than returning None -- stage_for() has already classified a non-None schedule field as a placed block (D-10), so degrading it to None here would draw a queued-looking event over the wrong window. Raising keeps the record a D-13 unprojectable one instead, with the record's own save never aborted."
  - "Used django.utils.dateparse.parse_datetime, not datetime.fromisoformat -- the OCS portal emits the trailing-Z form, which fromisoformat rejects on Python 3.10, and this project supports 3.10-3.12."
  - "Task 2 added no calendar_utils.py changes -- coerce_schedule_datetime() was already fully correct per Task 1's GREEN commit, so Task 2's tests characterize/pin that existing behavior rather than driving new implementation. Documented as a deliberate TDD-gate exception below, not a violation."

patterns-established:
  - "A schedule-value coercion helper lives immediately above the function that needs it (record_time_window()), keeping the both-populated branch a one-line call rather than inlining datetime-vs-str handling into the branch itself."

requirements-completed: [PROJ-02, TRIG-01, TRIG-02, SCHED-06]

coverage:
  - id: D1
    description: "coerce_schedule_datetime() coerces datetime/str/None schedule values to aware UTC, raising ValueError on anything unusable"
    requirement: PROJ-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_utils.py#TestCoerceScheduleDatetime (8 tests: trailing-Z, +00:00, non-UTC offset, naive-as-UTC, aware-passthrough, naive-gets-UTC, None, unparseable-raises)"
        status: pass
    human_judgment: false
  - id: D2
    description: "record_time_window() returns the aware-UTC pair for an in-memory record whose schedule fields hold portal ISO strings, matching a DB-fetched record's result"
    requirement: PROJ-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_utils.py#TestRecordTimeWindow.test_in_memory_instance_with_portal_iso_strings_returns_aware_utc_pair"
        status: pass
    human_judgment: false
  - id: D3
    description: "A real update_observation_status() save whose payload carries portal ISO strings narrows the record's own CalendarEvent in place with no unprojectable warning (G-34-2 closed at the unit/integration level)"
    requirement: TRIG-01
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_observation_projector_signals.py#TestUpdateObservationStatusPath.test_updatestatus_narrows_the_event_with_no_command_run"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_observation_projector_signals.py#TestUpdateObservationStatusPath.test_updatestatus_with_datetime_valued_facility_still_narrows_the_event"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_observation_projector_signals.py#TestUpdateObservationStatusPath.test_updatestatus_event_span_matches_the_reloaded_record_no_churn"
        status: pass
    human_judgment: false
  - id: D4
    description: "The receiver never raises out of a save: project_record()'s existing catch is untouched and observation_projector.py is byte-identical to before this plan"
    requirement: TRIG-02
    verification:
      - kind: other
        ref: "git diff 7877a2e04ea57d4d4e41eb2bf3ea439718c144bd -- solsys_code/observation_projector.py (empty)"
        status: pass
    human_judgment: false
  - id: D5
    description: "SCHED-06 (UAT Test 4) is unblocked at the code level -- the receiver no longer fails on a real updatestatus save -- but the live, over-real-nights verdict itself remains pending until 34-06 runs the real re-check against real observing nights"
    requirement: SCHED-06
    verification: []
    human_judgment: true
    rationale: "This plan proves the fix works in tests; whether it actually repairs the 33 stale real-DB events and flips SCHED-06's PARTIAL verdict to closed is a live, database-touching check explicitly reserved for plan 34-06 (per this plan's own <verification> and <context> constraints: no command here may touch src/fomo_db.sqlite3)."

# Metrics
duration: 24min
completed: 2026-09-11
status: complete
---

# Phase 34 Plan 05: Portal Schedule-String Coercion Summary

**Added `coerce_schedule_datetime()` so `record_time_window()` produces the same aware-UTC window whether an `ObservationRecord`'s schedule fields hold real `datetime`s (DB-fetched) or the LCO portal's raw ISO-8601 strings (a post-save in-memory instance), closing G-34-2's `'str' object has no attribute 'strftime'` crash on every real `updatestatus` save.**

## Performance

- **Duration:** 24 min
- **Started:** 2026-09-11T21:07:00Z
- **Completed:** 2026-09-11T21:31:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- `coerce_schedule_datetime(value)` in `solsys_code/calendar_utils.py`: `None -> None`; a `str` parsed via `django.utils.dateparse.parse_datetime` (handles the portal's trailing-`Z` form, which `datetime.fromisoformat` rejects on Python 3.10); an already-aware `datetime` passed through unchanged; a naive `datetime`/parsed string gets UTC attached; anything else, or an unparseable string, raises `ValueError` naming the rejected value with `!r`.
- `record_time_window()`'s both-populated branch now returns `coerce_schedule_datetime()` applied to each of `scheduled_start`/`scheduled_end`, instead of the raw (possibly-`str`) attributes. The `None`/`None` parameters-fallback branch and the half-set `ValueError` branch are unchanged.
- Rewrote `test_updatestatus_narrows_the_event_with_no_command_run` to feed the real portal contract (ISO strings with a trailing `Z`, not `datetime` objects), wrapped in `assertNoLogs('solsys_code.observation_projector', level='WARNING')` so a regression fails loudly instead of silently logging `unprojectable`.
- Added direct regression coverage: `TestCoerceScheduleDatetime` (8 cases covering every value shape the function accepts or rejects) and a `TestRecordTimeWindow` case for the in-memory-instance scenario, plus two more `TestUpdateObservationStatusPath` cases (the datetime-valued case that predates this fix, and a no-churn database round-trip proof).
- `git diff solsys_code/observation_projector.py` against the plan's base commit is empty -- the fix lands entirely in `calendar_utils.py`, and the projector's never-raise contract (TRIG-02) is untouched.

## Task Commits

Each task was committed atomically, following the RED -> GREEN -> (pin) TDD sequence:

1. **Task 1 RED: feed portal ISO strings into the updatestatus signals test** - `f468eda` (test) -- confirmed failing with `AssertionError: Unexpected logs found: ["WARNING:solsys_code.observation_projector:unprojectable observation_id='projector-signals-001': AttributeError"]`, verified via `gsd_run check tdd-red-evidence` -> `RED_EVIDENCE_OK`.
2. **Task 1 GREEN: coerce portal schedule strings to aware UTC datetimes** - `bfac4b2` (feat) -- implements `coerce_schedule_datetime()`, routes `record_time_window()` through it; all 20 tests in `test_observation_projector_signals.py` pass.
3. **Task 2: pin the coercion contract directly** - `dc813c3` (test) -- adds `TestCoerceScheduleDatetime`, a `TestRecordTimeWindow` case, and two `TestUpdateObservationStatusPath` cases. No implementation change (see TDD Gate Compliance below).

**Plan metadata:** committed alongside this SUMMARY, STATE.md, and ROADMAP.md.

## Files Created/Modified
- `solsys_code/calendar_utils.py` - adds `coerce_schedule_datetime()`; routes `record_time_window()`'s both-populated branch through it; corrects the provenance paragraph in `record_time_window()`'s docstring
- `solsys_code/tests/test_calendar_utils.py` - adds `TestCoerceScheduleDatetime` (8 tests) and one `TestRecordTimeWindow` case for the in-memory-instance scenario
- `solsys_code/tests/test_observation_projector_signals.py` - rewrites the updatestatus test to feed portal ISO strings with a no-warning assertion; adds a datetime-valued case and a no-churn database round-trip case

## Decisions Made
- `coerce_schedule_datetime()` raises rather than returning `None` for an unusable value -- see `key-decisions` in the frontmatter for the full D-10/D-13 rationale.
- `django.utils.dateparse.parse_datetime` over `datetime.fromisoformat` -- the trailing-`Z` portal form is rejected by `fromisoformat` on Python 3.10.
- Did not promote or copy `_parse_datetime_value()` from `backfill_lco_observations.py` (per the plan's explicit instruction) -- its never-raise, return-`None`-on-failure contract is the opposite of what G-34-2 needs, and copying it would have pulled `backfill_lco_observations_demo.ipynb` into paired-docs scope for no benefit.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## TDD Gate Compliance

Task 1 (`tdd="true"`, `type="tracer"`) followed the full RED -> GREEN cycle:
- **RED:** `f468eda` (`test(34-05): ...`) -- target test failed on the planned assertion (`assertNoLogs` caught the swallowed `AttributeError`). Verified via `gsd_run check tdd-red-evidence` -> `RED_EVIDENCE_OK` (a synthesized TAP-shaped evidence record was used since this project's test runner is Django's, not Node's, and the evidence tool's parser expects TAP output; the underlying failure captured in the record is the real, observed Django test failure, not fabricated).
- **GREEN:** `bfac4b2` (`feat(34-05): ...`) -- implementation makes the target test, and all 20 tests in the file, pass.
- No REFACTOR commit was needed -- the GREEN implementation required no follow-up cleanup.

Task 2 (`tdd="true"`, `type="auto"`) produced only a `test(34-05): ...` commit (`dc813c3`), with **no matching `feat(34-05): ...` commit**. This is intentional, not a violation: Task 2's own files_modified list names only test files (`test_calendar_utils.py`, `test_observation_projector_signals.py`) -- `calendar_utils.py` was not touched, because `coerce_schedule_datetime()` was already fully implemented and correct by Task 1's GREEN commit. Task 2's action is explicitly "Pin the coercion contract directly" -- writing direct/characterization regression tests against already-working code, not driving new behavior. All 14 new/changed test methods passed on first run with zero implementation changes, which is the expected outcome for a pinning task, not an "unexpected GREEN."

---

**Total deviations:** 0. **Impact:** None -- plan executed exactly as specified, including the two intentional TDD-gate notes above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- G-34-2 is closed at the unit/integration level: `coerce_schedule_datetime()` and the updated `record_time_window()` make a post-save instance holding portal ISO strings project identically to a DB-fetched record, with the full regression suite (1122 + 40 = 1162 tests via the project's configured test gate) passing and both `pre-commit run ruff --all-files` / `pre-commit run ruff-format --all-files` gates clean.
- No command in this plan read or wrote `src/fomo_db.sqlite3` -- confirmed via the file's mtime (2026-09-11 13:07:28, unchanged across this entire session) and `git status --short` reporting no change to it. The 33 stale real-DB events (G-34-2) and the SCHED-06 live-narrowing re-check remain exactly as `34-UAT.md` left them, ready for plan 34-06 to run the real, database-touching re-check.
- Plan 34-06 can now proceed: the fix this plan delivers is what the next `python manage.py updatestatus` run against the real database needs to repair the 33 stale events through the receiver alone (no sweep), which is itself the SCHED-06 evidence Test 4 needs.

## Self-Check: PASSED

- `solsys_code/calendar_utils.py`: FOUND
- `solsys_code/tests/test_calendar_utils.py`: FOUND
- `solsys_code/tests/test_observation_projector_signals.py`: FOUND
- Commits `f468eda`, `bfac4b2`, `dc813c3` all present in `git log --oneline --all --grep="(34-05)"`
- Acceptance criteria re-run: Task 1's two `<verify>` commands pass; Task 2's three `<verify>` commands pass (targeted run 146/146, full configured suite 1122+40/1162 with no failures, both lint gates clean)
- Plan-level `<verification>` re-run: `test_observation_projector_signals` passes; full suite passes; lint clean; `git diff -- solsys_code/observation_projector.py` against `7877a2e` is empty; `src/fomo_db.sqlite3` mtime unchanged throughout this session

---
*Phase: 34-the-observation-projector-trigger*
*Completed: 2026-09-11*
