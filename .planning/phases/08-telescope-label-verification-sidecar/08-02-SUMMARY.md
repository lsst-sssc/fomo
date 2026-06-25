---
phase: 08-telescope-label-verification-sidecar
plan: 02
subsystem: frontend
tags: [django-templates, tom_calendar, calendar-html, view-test, dashed-border, tooltip]

# Dependency graph
requires:
  - phase: 08-01
    provides: "CalendarEventTelescopeLabel sidecar model with reverse accessor event.telescope_label_meta (is_verified bool); missing row = verified by documented default"
provides:
  - "calendar.html dashed-border + tooltip rendering for fallback-labeled events on both all-day and timed branches (DISPLAY-02, DISPLAY-03)"
  - "First calendar.html view-level rendering test in this codebase (solsys_code/tests/test_calendar_template.py)"
affects: [phase-09-proposal-color-status]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Defensive `== False` template comparison on a reverse-O2O accessor so a missing sidecar row and an explicit True both fall through to the unstyled/verified branch"
    - "First calendar.html Client+TestCase rendering test, using reverse('calendar:calendar') with year/month query params to target fixture events into the rendered month grid"

key-files:
  created:
    - solsys_code/tests/test_calendar_template.py
  modified:
    - src/templates/tom_calendar/partials/calendar.html

key-decisions:
  - "Test expectation for the dashed-border marker count is expressed as day-cell occurrences, not event count: the all-day-loop bucketing in tom_calendar's render_calendar() renders a multi-day all-day event once per day cell it spans (offset_date(start) <= d <= offset_date(end)), so a 2-day fallback event contributes 2 marker occurrences, not 1. This is upstream view behavior, not a defect in the new template branches — confirmed by reading tom_calendar/views.py's day-cell construction before adjusting the test's expected count."

requirements-completed: [DISPLAY-02, DISPLAY-03]

coverage:
  - id: D7
    description: "All-day loop's [QUEUED] branch remains first; a new elif branch renders the dashed border + tooltip only when telescope_label_meta.is_verified == False"
    requirement: "DISPLAY-02, DISPLAY-03"
    verification:
      - kind: unit
        ref: "python -c \"...is_verified == False... count==2...\" verification script from PLAN.md Task 1"
        status: pass
    human_judgment: false
  - id: D8
    description: "Timed loop's outer cal-event-timed div gets the same dashed-border + tooltip treatment, preserving hx-get, on an if/else (not elif, no [QUEUED] precedent in this branch)"
    requirement: "DISPLAY-02, DISPLAY-03"
    verification:
      - kind: unit
        ref: "same Task 1 verification script (is_verified == False count==2 spans both branches)"
        status: pass
    human_judgment: false
  - id: D9
    description: "A fallback event (all-day or timed) renders the dashed-border marker and tooltip substring; verified and no-sidecar-row events do not"
    requirement: "DISPLAY-02, DISPLAY-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#test_fallback_events_get_dashed_border_and_tooltip, #test_dashed_border_count_matches_fallback_event_count_only"
        status: pass
    human_judgment: false
  - id: D10
    description: "A CalendarEvent with no sidecar row renders the calendar page with status 200 and no exception (silenced DoesNotExist path)"
    requirement: "DISPLAY-01 (read-side)"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#test_calendar_renders_200_including_no_sidecar_row_events"
        status: pass
    human_judgment: false

duration: 11min
completed: 2026-06-25
status: complete
---

# Phase 8 Plan 02: Telescope Label Verification Sidecar — Calendar UI Render Summary

**Added a dashed-border + native-tooltip render branch to both the all-day and timed event loops in `calendar.html`, plus the first `calendar.html` view-level rendering test in this codebase, proving fallback-labeled events are visually distinguishable and verified/no-row events are unaffected.**

## Performance

- **Duration:** 11 min
- **Started:** 2026-06-25T06:19:10Z
- **Completed:** 2026-06-25T06:30:20Z
- **Tasks:** 2
- **Files modified:** 2 (1 template, 1 new test file)

## Accomplishments
- `src/templates/tom_calendar/partials/calendar.html`'s all-day loop gained a new `{% elif event.telescope_label_meta.is_verified == False %}` branch, inserted between the existing `[QUEUED]` branch (which still takes precedence) and the verified `{% else %}` branch — renders the existing `background-color: {{ event.color }}` plus a `2px dashed rgba(0, 0, 0, 0.65)` border and a `title=` tooltip.
- The timed loop's previously-flat `cal-event-timed` div is now wrapped in an `{% if event.telescope_label_meta.is_verified == False %}` / `{% else %}` chain on the outer (hoverable) div, adding the identical dashed-border style and tooltip in the fallback branch while preserving the existing `hx-get` attribute in both branches.
- Both branches use the `== False` comparison (not bare truthiness, not `|default:True`), so a missing sidecar row and an explicit `is_verified=True` both fall through identically to the unstyled/verified branch — matching the Phase 8 Plan 01 documented default.
- New `solsys_code/tests/test_calendar_template.py` establishes the first `calendar.html` view-level rendering test in this codebase (`Client` + `TestCase`, mirroring the only existing `Client`-based precedent in `solsys_code_observatory/tests/test_views.py`). It builds 6 `CalendarEvent` fixtures (fallback/verified/no-row x all-day/timed), hits `reverse('calendar:calendar')` with `year`/`month` query params targeting the fixture month, and asserts: status 200 including the no-row events (proves the silenced `DoesNotExist` path doesn't 500); the dashed-border marker and tooltip substring are present; and the dashed-border marker count equals the exact number of fallback-event day-cell occurrences (not just event count — see Decisions).

## Task Commits

Each task was committed atomically:

1. **Task 1: Add dashed-border + tooltip branches to calendar.html (all-day and timed)** - `68dd6f4` (feat)
2. **Task 2: Add a calendar-template rendering test (verified / fallback / no-row)** - `57949b5` (test)

## Files Created/Modified
- `src/templates/tom_calendar/partials/calendar.html` - new `elif`/`if`-`else` dashed-border + tooltip branches on both render loops
- `solsys_code/tests/test_calendar_template.py` - new file, 3 tests covering fallback/verified/no-row across both branches

## Decisions Made
- Test's expected dashed-border marker count is expressed in terms of day-cell occurrences (2 for the 2-day all-day fallback event + 1 for the timed fallback event = 3), not raw event count (2), because `tom_calendar`'s `render_calendar()` view buckets a multi-day all-day event into every day cell it spans (`offset_date(start) <= d <= offset_date(end)`). Confirmed by reading `tom_calendar/views.py` directly rather than assuming a 1-event-to-1-occurrence mapping; this is correct upstream view behavior the new template branches inherit, not a defect.

## Deviations from Plan

None — plan executed exactly as written. The test-count adjustment above is test-fixture-design detail discovered while writing Task 2 (the plan's `<action>` did not specify exact fixture day spans), not a deviation from the plan's stated tasks, files, or acceptance criteria.

## Issues Encountered
- `./manage.py` is not executable in this checkout (`Permission denied` on direct invocation); used `python manage.py test ...` instead, which works identically. Confirmed this is an environment/permissions detail, not a code issue — no fix applied since `python manage.py` is the documented CLAUDE.md command form already.
- `ruff check .` / `ruff format --check .` repo-wide still report the same pre-existing issues noted in Plan 01's Summary (notebook cells, `src/fomo/settings.py`, two unrelated `.planning/quick/` scripts) — confirmed none touch this plan's two changed/created files; out of scope per the deviation rules' scope boundary.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- DISPLAY-01 (Plan 01), DISPLAY-02, and DISPLAY-03 (this plan) are fully delivered: the sidecar model, write path, and now the read-side dashed-border + tooltip render are all in place and test-proven for verified, fallback, and no-sidecar-row cases on both calendar render branches.
- `./manage.py test solsys_code` is green at 138 tests (135 from before this plan + 3 new); `ruff check .` / `ruff format --check .` introduce no new issues from this plan's changes.
- This is the last plan in Phase 8 — Phase 8 (telescope-label-verification-sidecar) is complete. Phase 9 (proposal-color-status visual treatment, DISPLAY-04/05/06/07) can proceed; per the locked D-03 UI-SPEC constraint, Phase 9 must reserve dash-style borders exclusively for this phase's verification signal and use a different border property (color/thickness/double-border) for its own status treatment.
- No blockers for Phase 9.

---
*Phase: 08-telescope-label-verification-sidecar*
*Completed: 2026-06-25*
