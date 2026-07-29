---
phase: 25-range-window-calendarevent-projection-allow-approved-site-re
plan: 01
subsystem: api
tags: [django, calendar, campaign-coordination, sun-event]

# Dependency graph
requires:
  - phase: 23-weather-storm-cancellation-handling
    provides: "_set_run_status() single-event sync mechanism this plan generalizes to multi-event"
provides:
  - "_project_calendar_event() projects one dip-corrected CalendarEvent per night for a resolved-site, ground, range-window CampaignRun"
  - "_calendar_event_title(run) shared title helper (base title + D-06 window-context suffix)"
  - "_set_run_status() updates every CalendarEvent belonging to a run (bare key + per-night keys) via a combined Q(...) queryset"
  - "revised test_campaign_approval.py assertions covering per-night counts (15/15/4/15), first/last-night span, satellite single-event, partial projection, and genuine TBD resolve"
affects: [campaign-coordination, calendar-sync]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-night event loop (n_nights = (window_end - window_start).days + 1) mirroring load_telescope_runs' E - S + 1 inclusive-range idiom"
    - "Single shared title-building helper reused by both the create path and the status-update path to prevent title-format drift"

key-files:
  created: []
  modified:
    - solsys_code/campaign_views.py
    - solsys_code/tests/test_campaign_approval.py

key-decisions:
  - "Guard drops the window_start == window_end equality clause, keeps window_end truthiness (D-01) -- a range-window run with a resolved site and telescope_instrument now reaches the projection date-math"
  - "Ground branch projects one CalendarEvent per night (D-02) using a per-night url key CAMPAIGN:{pk}:{date.isoformat()} for a range, keeping the bare CAMPAIGN:{pk} key for a single night (D-03)"
  - "Satellite branch date-math and single-event, bare-key behavior are unchanged (D-05); only its title now flows through the shared _calendar_event_title() helper"
  - "_set_run_status() looks up CalendarEvent via Q(url=CAMPAIGN:{pk}) | Q(url__startswith=CAMPAIGN:{pk}:) with a trailing colon, so a status change updates every night's event without a pk-digit-prefix collision"
  - "A mid-window sun_event() ValueError leaves already-created earlier nights' events in place -- partial projection accepted, no transaction.atomic() wrap (RESEARCH Assumption A3, locked by a new test)"

patterns-established:
  - "Pattern: extract one shared title-building function (no prefix logic) that both the creation path and the status-update path call, so a prefix/suffix composition never drifts out of sync between the two call sites"

requirements-completed: [FIX-01, FIX-02, FIX-03, FIX-04, FIX-05, FIX-06, FIX-07]

coverage:
  - id: D1
    description: "A ground range-window run projects one dip-corrected CalendarEvent per night, keyed CAMPAIGN:{pk}:{date}, with the first/last night's start/end times matching sun_event() and every title carrying the D-06 window suffix"
    requirement: "FIX-01/FIX-02/FIX-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestCalendarProjection.test_approve_range_run_creates_one_event_per_night"
        status: pass
    human_judgment: false
  - id: D2
    description: "A satellite range-window run still projects exactly one whole-day-span event under the bare key, with the window suffix applied to its title"
    requirement: "FIX-04"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestCalendarProjection.test_approve_range_run_space_site_creates_single_whole_day_span_event"
        status: pass
    human_judgment: false
  - id: D3
    description: "_set_run_status() updates every night's event with the [CANCELLED]/[WEATHERED] prefix, preserving the window suffix, for a range run and for the real Gemini FT-115 scenario"
    requirement: "FIX-05"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestRunStatusChange.test_mark_range_window_run_updates_every_night_event"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestGeminiFtScenario.test_gemini_ft115_range_window_projects_per_night_events"
        status: pass
    human_judgment: false
  - id: D4
    description: "resolve_site's retroactive projection creates per-night events for a range run with the 'added to the calendar' message, and a genuine TBD-resolve still projects zero events with the plain 'Site resolved.' message"
    requirement: "FIX-06"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_resolve_range_run_projects_per_night_calendar_events"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_resolve_tbd_run_clears_flag_with_no_calendar_event"
        status: pass
    human_judgment: false
  - id: D5
    description: "TBD/unresolved-site/missing-telescope_instrument/sun_event-ValueError guard-exclusion tests still pass with their assertion bodies unchanged, and a mid-window sun_event() ValueError leaves earlier nights' events in place (partial projection)"
    requirement: "FIX-07"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestCalendarProjection (test_approve_tbd_run_creates_no_calendar_event, test_approve_without_telescope_instrument_creates_no_calendar_event, test_sun_event_valueerror_skips_projection_without_reverting_approval)"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestCalendarProjection.test_approve_range_run_partial_projection_on_mid_window_sun_event_error"
        status: pass
      - kind: unit
        ref: "python manage.py test solsys_code (539 tests)"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-07-17
status: complete
---

# Phase 25 Plan 01: Range-window CalendarEvent Projection Summary

**Approved, site-resolved range-window CampaignRuns now project one dip-corrected CalendarEvent per night (ground) or one whole-day-span event (satellite), replacing the silent zero-event guard; a shared title helper keeps the window-context suffix intact through status changes.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-07-17T22:53:00Z (approx, first task commit)
- **Completed:** 2026-07-17T23:05:18+01:00
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- `_project_calendar_event()`'s guard now admits any resolved-site run with both `window_start` and `window_end` set (dropped the `window_start == window_end` equality clause), so a range-window ground run reaches the projection date-math instead of returning `False` unconditionally
- The ground branch projects one dip-corrected `sun_event()`-derived `CalendarEvent` per night for a range window (`CAMPAIGN:{pk}:{date.isoformat()}` keys), while a single-night run keeps the existing bare `CAMPAIGN:{pk}` key and identical dip-corrected math
- A new `_calendar_event_title(run)` helper is the single source of truth for the title (base title + D-06 window-context suffix for a range), used by both `_project_calendar_event()` (creation) and `_set_run_status()` (status update) so the suffix can never drift between the two call sites
- `_set_run_status()` now finds and updates every `CalendarEvent` belonging to a run via `Q(url=f'CAMPAIGN:{pk}') | Q(url__startswith=f'CAMPAIGN:{pk}:')` (trailing colon required to avoid a pk-digit-prefix collision), so `[CANCELLED]`/`[WEATHERED]` prefixes land on every night's event
- The satellite branch's whole-day-span date-math (00:00 → 23:59 UTC, bare key) is completely unchanged; only its title now flows through the shared helper
- Test suite revised: 4 tests renamed/rewritten to assert per-night projection (15/15/4/15 counts), 1 new satellite-range single-event test, 1 new partial-projection lock test, 1 new genuine-TBD-resolve test; the 3 guard-exclusion tests kept byte-identical

## Task Commits

Each task was committed atomically:

1. **Task 1: Extract shared title helper + rewrite `_project_calendar_event()` for per-night ground projection** - `efbd366` (feat)
2. **Task 2: Rewrite `_set_run_status()` to update every night's event with the shared title** - `df676f1` (feat)
3. **Task 3: Revise `test_campaign_approval.py` to the per-night contract** - `5187e4b` (test)
4. **Docs fix: stale comment in `_set_run_status()`** - `e2f703a` (docs)

**Plan metadata:** committed after this SUMMARY.

## Files Created/Modified
- `solsys_code/campaign_views.py` - `_calendar_event_title()` helper added; `_project_calendar_event()` guard/ground-branch rewritten for per-night projection; `_set_run_status()` rewritten to update every matching event
- `solsys_code/tests/test_campaign_approval.py` - 4 tests renamed/rewritten to assert per-night counts, 3 new tests added (satellite range, partial projection, genuine TBD resolve), stale docstrings updated

## Decisions Made
- Guard fix (D-01): drop `window_start == window_end`, keep `window_end` truthiness — TBD/unresolved-site/missing-telescope_instrument exclusions unaffected
- Ground branch per-night expansion (D-02) mirrors `load_telescope_runs`' `E - S + 1` idiom exactly; url key scheme is `CAMPAIGN:{pk}:{date.isoformat()}` for a range, bare `CAMPAIGN:{pk}` for a single night (D-03)
- `_set_run_status()` uses one combined `Q(...)` queryset with a trailing-colon `startswith` prefix (D-04) rather than two separate filter calls, and reuses `_calendar_event_title()` rather than re-deriving the title inline
- Satellite branch stays untouched except its title now goes through the shared helper (D-05); the title suffix is applied uniformly regardless of branch (RESEARCH Open Question 1 / Assumption A2, resolved: apply uniformly)
- Partial projection (no `transaction.atomic()` wrap) is the accepted behavior for a mid-window `sun_event()` ValueError (RESEARCH Open Question 2 / Assumption A3), now locked by a dedicated test

## Deviations from Plan

None - plan executed exactly as written. One follow-up docs-only commit (`e2f703a`) fixed a stale inline comment in `_set_run_status()` that still claimed range runs never reach the projection date-math branch — this is Task 3's own Pitfall-4 requirement (stale docstrings/comments), caught during the final grep sanity check rather than during the initial edit.

## Issues Encountered
- `ruff`'s pre-commit hook auto-stripped the `Q` import added in Task 1 (unused until Task 2 actually calls it), requiring it to be re-added before Task 2's tests would pass — expected consequence of splitting the import addition from its first use across two atomic task commits, not a real issue.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Plan 02 (the D-07 one-off backfill management command for already-`APPROVED` range-window runs, e.g. the real GS-2026A-FT-115 pk=34 row) can now call the rewritten `_project_calendar_event()` directly — no further changes to this plan's files are needed for it to work
- Full `python manage.py test solsys_code` (539 tests) and `ruff check`/`ruff format --check` on both plan files are green

---
*Phase: 25-range-window-calendarevent-projection-allow-approved-site-re*
*Completed: 2026-07-17*

## Self-Check: PASSED
