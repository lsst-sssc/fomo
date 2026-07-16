---
phase: 23-weather-storm-cancellation-handling-give-staff-a-way-to-mark
plan: 02
subsystem: api
tags: [django, django-tables2, calendar-sync, campaign-coordination]

# Dependency graph
requires:
  - phase: 22-site-matching-at-submission-and-unmatched-site-resolution-workflow
    provides: ApprovalQueueTable's show_actions/mode/candidate_pool conventions and the CampaignRunDecisionView.post() action-dispatch shape this plan extends
provides:
  - "CampaignRunDecisionView._set_run_status(): guarded, staleness-safe endpoint that sets run_status to CANCELLED or WEATHER_TECH_FAILURE on an APPROVED CampaignRun"
  - "Decided-table Mark Cancelled / Mark Weathered action, gated by an independent ApprovalQueueTable.status_actions flag"
  - "'[WEATHERED]' terminal box-shadow ring in calendar_display_extras._TERMINAL_PREFIXES"
affects: [24-weather-storm-cancellation-handling-if-more-plans-follow, any-future-run-status-ui-work]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Conditional queryset .update() with an explicit updated_count == 0 short-circuit BEFORE refresh_from_db()/side-effect branches -- mirrors _resolve_site()'s claimed == 0 guard, now used a third time in this view (approve/reject, resolve_site, set_run_status)"
    - "Independent boolean flags for independently-gated table features (status_actions alongside show_actions) rather than overloading one flag for two different render decisions"

key-files:
  created: []
  modified:
    - solsys_code/campaign_views.py
    - solsys_code/campaign_tables.py
    - solsys_code/templatetags/calendar_display_extras.py
    - solsys_code/tests/test_campaign_approval.py
    - solsys_code/tests/test_calendar_display_extras.py

key-decisions:
  - "Reused _resolve_site()'s exact guard -> conditional-update -> updated_count-check -> refresh_from_db() shape for _set_run_status(), rather than inventing a new pattern, so the lost-update protection (REVIEW finding #1) is structurally identical to the codebase's established discipline."
  - "Calendar sync is existence-guarded (CalendarEvent.objects.filter(url=...).exists()) rather than unconditional, so a range/TBD/unresolved-site run's run_status can be set without ever reaching insert_or_create_calendar_event()'s non-nullable start_time/end_time create-path."
  - "status_actions is a new, independent ApprovalQueueTable flag (not a repurposed show_actions) so the Decided table can render the new action while its Site column keeps the plain-text fallback (never the live-search widget)."

patterns-established:
  - "Business-logic-guard + conditional-update + updated_count-checked-before-any-side-effect is now this view's standard shape for any staff action that mutates a CampaignRun's status fields."

requirements-completed: [D-03, D-04, D-05]

coverage:
  - id: D1
    description: "Staff mark an APPROVED CampaignRun cancelled or weathered from the Decided table's new action (first action the Decided table has ever had)"
    requirement: "D-04"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestDecidedTableStatusActions.test_decided_table_renders_status_actions_for_approved_run"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestDecidedTableStatusActions.test_decided_table_no_status_actions_for_rejected_run"
        status: pass
    human_judgment: false
  - id: D2
    description: "CANCELLED and WEATHER_TECH_FAILURE produce two distinct calendar title prefixes ('[CANCELLED]' vs '[WEATHERED]'), never a shared label"
    requirement: "D-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestRunStatusChange.test_mark_cancelled_single_night_updates_existing_event_in_place"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestRunStatusChange.test_mark_weather_failure_uses_distinct_weathered_prefix"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_calendar_display_extras.py#StatusBorderCssTest.test_weathered_returns_terminal_box_shadow"
        status: pass
    human_judgment: false
  - id: D3
    description: "Setting run_status on a single-night, resolved-site run updates its existing CAMPAIGN:{pk} CalendarEvent title/description in place -- never deletes it, never duplicates it"
    requirement: "D-05"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestRunStatusChange.test_mark_cancelled_single_night_updates_existing_event_in_place"
        status: pass
    human_judgment: false
  - id: D4
    description: "Marking a range/TBD/unresolved-site run (never had a projected CalendarEvent) does not crash and does not fabricate a CalendarEvent -- run_status still gets set (RESEARCH Pitfall 1 backstop)"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestRunStatusChange.test_mark_range_window_run_does_not_crash_and_creates_no_event"
        status: pass
    human_judgment: false
  - id: D5
    description: "A mark-status POST for a non-APPROVED run, or from an anonymous/non-staff session, is rejected server-side and makes no run_status change (business-logic bypass guard)"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestRunStatusChange.test_mark_status_on_non_approved_run_rejected"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestRunStatusChange.test_mark_status_anonymous_or_non_staff_makes_no_change"
        status: pass
    human_judgment: false
  - id: D6
    description: "A mark-status POST whose conditional .update() matches 0 rows (lost-update race) short-circuits with a warning + redirect and does not call refresh_from_db() or the calendar-sync branch -- no 500, no CampaignRun.DoesNotExist, no calendar mutation (REVIEW finding #1 backstop)"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestRunStatusChange.test_mark_status_lost_update_race_warns_no_calendar_mutation"
        status: pass
    human_judgment: false
  - id: D7
    description: "The Decided table's site column stays plain-text (never the live-search widget) -- gated by the independent status_actions flag, not by flipping show_actions"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestDecidedTableStatusActions.test_decided_table_site_column_stays_plain_text"
        status: pass
    human_judgment: false

duration: 10min
completed: 2026-07-16
status: complete
---

# Phase 23 Plan 02: Weather/Storm Cancellation for CampaignRun (Approval-Queue Action) Summary

**Staff can now mark an APPROVED CampaignRun cancelled or weathered from a new Decided-table action, which updates the linked CAMPAIGN:{pk} calendar event in place with a distinct `[CANCELLED]`/`[WEATHERED]` title prefix and terminal box-shadow ring, without ever fabricating an event for a range/TBD/unresolved-site run.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-07-16T22:12:24+01:00 (immediately after Plan 01 completed)
- **Completed:** 2026-07-16T22:21:44+01:00
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- `CampaignRunDecisionView._set_run_status()` — a new guarded, staleness-safe endpoint that sets `run_status` to `CANCELLED` or `WEATHER_TECH_FAILURE` on an already-`APPROVED` `CampaignRun`, dispatched from `post()`'s extended action whitelist (`mark_cancelled`/`mark_weather_failure`)
- The linked `CAMPAIGN:{pk}` `CalendarEvent`, if one already exists, is updated in place (title prefix + a new `Run status:` description line) via the existing no-churn `insert_or_create_calendar_event()` helper — never created for a run that never had a projected event
- REVIEW finding #1 backstop: the conditional `.update()`'s row count is checked and short-circuits with a warning *before* `refresh_from_db()` or the calendar-sync branch, closing a lost-update race window that would otherwise raise `CampaignRun.DoesNotExist` (500) or silently report false success
- `ApprovalQueueTable` gains an independent `status_actions` flag (default off); the Decided table's `render_actions()` now shows Mark Cancelled/Mark Weathered buttons for any `APPROVED` row, while its Site column keeps the existing plain-text fallback untouched (RESEARCH Pitfall 3)
- `[WEATHERED]` added to `calendar_display_extras._TERMINAL_PREFIXES` alongside the pre-existing `[CANCELLED]`, so weathered runs get the same terminal box-shadow ring

## Task Commits

Each task was committed atomically:

1. **Task 1: CampaignRunDecisionView._set_run_status() — guarded run_status write + in-place-only calendar sync** - `11cd4c6` (feat)
2. **Task 2: Decided-table Mark Cancelled / Mark Weathered action gated by an independent status_actions flag** - `dea7e76` (feat)
3. **Task 3: Add [WEATHERED] to _TERMINAL_PREFIXES so the weathered prefix gets the box-shadow ring** - `e65ba86` (feat)

_TDD was not required for this plan (`tdd` not set on any task); tests were added alongside each implementation task and verified green before commit._

## Files Created/Modified
- `solsys_code/campaign_views.py` - `_RUN_STATUS_CALENDAR_PREFIX`/`_ACTION_TO_RUN_STATUS` module dicts, extended `post()` whitelist/dispatch, new `CampaignRunDecisionView._set_run_status()`, `decided_table` construction now passes `status_actions=True, request=self.request`, `CalendarEvent` import added
- `solsys_code/campaign_tables.py` - `ApprovalQueueTable.__init__` gains `status_actions=False`; `render_actions()` gets a new `status_actions`-gated branch inside the `if not self.show_actions:` early return
- `solsys_code/templatetags/calendar_display_extras.py` - `'[WEATHERED]'` appended to `_TERMINAL_PREFIXES`
- `solsys_code/tests/test_campaign_approval.py` - new `TestRunStatusChange` (7 tests) and `TestDecidedTableStatusActions` (3 tests) classes
- `solsys_code/tests/test_calendar_display_extras.py` - `test_weathered_returns_terminal_box_shadow`, extended `test_no_dashed_in_terminal_result`

## Decisions Made
- `_set_run_status()` deliberately mirrors `_resolve_site()`'s exact shape (business-logic guard → conditional `.update()` → `updated_count`-checked short-circuit → `refresh_from_db()` → guarded side effect) rather than inventing a new pattern, matching the codebase's established discipline for staff-mutation endpoints.
- The updated `CalendarEvent.description` appends a `Run status: {get_run_status_display()}` line (REVIEW finding #3) so the no-churn helper's field-diff actually detects and writes a change, and the reason for the terminal state is captured beyond the title prefix alone.
- `status_actions` is a new, independent `ApprovalQueueTable` flag rather than a repurposed `show_actions`, per RESEARCH Pitfall 3 — this keeps the Decided table's Site column on its existing plain-text fallback path with zero risk of leaking the live-search widget.

## Deviations from Plan

None - plan executed exactly as written. All `<behavior>`/`<action>`/`<acceptance_criteria>` items for all three tasks were implemented as specified; no Rule 1-4 fixes were needed beyond what the plan itself already called for (the plan's own text already specified the REVIEW-finding-#1 guard and the description-line addition, so those aren't deviations).

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Plan 23-01 (classical-run title-prefix cancellation) and Plan 23-02 (this plan, CampaignRun weather/cancellation) both complete for Wave 1 of Phase 23.
- `python manage.py test solsys_code.tests.test_campaign_approval` (104 tests) and `solsys_code.tests.test_calendar_display_extras` (28 tests) both green; `ruff check .`/`ruff format --check .` clean on all five files this plan touched.
- Full `python manage.py test solsys_code` run confirmed at wave merge: **534 tests, all pass** (no cross-module regressions from either Plan 01 or Plan 02).

## Self-Check: PASSED

All 5 modified source/test files and the SUMMARY.md itself confirmed present on disk; all 4 commit hashes (`11cd4c6`, `dea7e76`, `e65ba86`, `8a2dcec`) confirmed present in `git log`.

---
*Phase: 23-weather-storm-cancellation-handling-give-staff-a-way-to-mark*
*Completed: 2026-07-16*
