---
phase: 25-range-window-calendarevent-projection-allow-approved-site-re
plan: 02
subsystem: api
tags: [django, management-command, calendar, campaign-coordination, backfill]

# Dependency graph
requires:
  - phase: 25-01
    provides: "_project_calendar_event() ground/satellite per-night projection logic this command delegates to"
provides:
  - "backfill_range_calendar_events management command projecting CalendarEvents for already-APPROVED, site-resolved range-window CampaignRuns"
affects: [campaign-coordination, calendar-sync]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "One-off backfill command mirroring load_telescope_runs.py's BaseCommand/counters/stdout-summary shape, delegating all projection math to an existing cross-module underscore-prefixed helper rather than reimplementing it"

key-files:
  created:
    - solsys_code/management/commands/backfill_range_calendar_events.py
    - solsys_code/tests/test_backfill_range_calendar_events.py
  modified: []

key-decisions:
  - "Command name backfill_range_calendar_events; --dry-run flag included (D-07, Claude's Discretion) since it writes real production CalendarEvent rows against the live dev DB"
  - "Candidate query deliberately not filtered by site.observations_type -- _project_calendar_event() already routes ground vs satellite correctly, so a hypothetical satellite range candidate is handled safely too"
  - "candidates materialized to a list (not left as a lazy queryset) before the loop so the final summary's count reflects the original candidate set, not a re-issued query after CalendarEvents have been written"

patterns-established:
  - "A one-off data-backfill command delegates 100% of its domain logic to the existing view-layer helper it backfills for, keeping projection math defined in exactly one place"

requirements-completed: [FIX-08]

coverage:
  - id: D1
    description: "A qualifying APPROVED, site-resolved ground range-window run with no existing CAMPAIGN:{pk}* event gets one dip-corrected CalendarEvent per night; the real dev-DB CampaignRun pk=34 (GS-2026A-FT-115) is confirmed as a --dry-run candidate"
    requirement: FIX-08
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_range_calendar_events.py#TestBackfillRangeCalendarEvents.test_backfill_projects_per_night_events_for_qualifying_range_run"
        status: pass
      - kind: manual
        ref: "./manage.py backfill_range_calendar_events --dry-run against src/fomo_db.sqlite3"
        status: pass
    human_judgment: false
  - id: D2
    description: "Re-running the command is idempotent (no duplicate events); non-qualifying runs (single-night, TBD, unresolved-site, PENDING_REVIEW) get no events"
    requirement: FIX-08
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_range_calendar_events.py#TestBackfillRangeCalendarEvents (test_backfill_is_idempotent_on_second_run, test_backfill_skips_non_qualifying_runs)"
        status: pass
    human_judgment: false
  - id: D3
    description: "--dry-run reports candidates without writing any CalendarEvent rows; a per-candidate sun_event() ValueError is reported and skipped, never aborting the whole backfill run"
    requirement: FIX-08
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_range_calendar_events.py#TestBackfillRangeCalendarEvents (test_backfill_dry_run_writes_nothing, test_backfill_skips_and_continues_on_sun_event_valueerror)"
        status: pass
    human_judgment: false

duration: 20min
completed: 2026-07-18
status: complete
---

# Phase 25 Plan 02: Backfill Management Command for Already-Approved Range-Window Runs Summary

**A one-off `backfill_range_calendar_events` management command finds already-APPROVED, site-resolved range-window `CampaignRun`s with no existing calendar event and projects them by delegating entirely to Plan 01's rewritten `_project_calendar_event()`, closing the gap left by projection only firing on the approve/resolve_site POST actions.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-07-18T02:10:00Z (approx, first task commit)
- **Completed:** 2026-07-18T02:30:39Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- `backfill_range_calendar_events.py`: a `BaseCommand` with a `--dry-run` flag that queries `CampaignRun.objects.filter(approval_status=APPROVED, site__isnull=False, window_start__isnull=False).exclude(window_start=F('window_end'))`, pre-checks for an existing `CAMPAIGN:{pk}*` `CalendarEvent` via the same trailing-colon combined `Q(...)` queryset `_set_run_status()` uses, and delegates all projection date-math to `campaign_views._project_calendar_event(run)` — never reimplementing it
- A per-candidate `ValueError` from `_project_calendar_event()` (e.g. a blank-timezone Tier-2-resolved site) is caught, logged to `self.stderr`, and counted as `failed` without aborting the rest of the run
- `--dry-run` reports each candidate's pk/campaign/telescope_instrument/window and a final summary, writing zero `CalendarEvent` rows
- Test suite (`test_backfill_range_calendar_events.py`, 5 tests): qualifying 4-night ground range run gets 4 per-night events; re-running is idempotent; single-night/TBD/unresolved-site/`PENDING_REVIEW` runs all get zero events; `--dry-run` writes nothing and names the candidate; a `sun_event()` `ValueError` (mocked via `unittest.mock.patch`) is skipped without aborting the command
- Manual `--dry-run` smoke test against the real dev DB (`src/fomo_db.sqlite3`) confirmed `CampaignRun` pk=34 (`GS-2026A-FT-115`, `Didymos 2026: Gemini South/GMOS-s`, window 2026-07-13..2026-07-16) is correctly identified as a qualifying candidate — matching the phase's motivating case exactly

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement the backfill_range_calendar_events management command** - `1fb56ae` (feat)
2. **Task 2: Test the backfill command (qualifying / non-qualifying / idempotent / dry-run / error-skip)** - `0ff4a70` (test)

**Plan metadata:** committed after this SUMMARY.

## Files Created/Modified
- `solsys_code/management/commands/backfill_range_calendar_events.py` - NEW: `class Command(BaseCommand)` with `--dry-run`, candidate query, existence pre-check, delegated projection, per-candidate error handling, and a counters/stdout summary mirroring `load_telescope_runs.py`'s convention
- `solsys_code/tests/test_backfill_range_calendar_events.py` - NEW: `class TestBackfillRangeCalendarEvents(TestCase)`, 5 tests covering the qualifying/non-qualifying/idempotent/dry-run/error-skip contract

## Decisions Made
- Command name `backfill_range_calendar_events`; `--dry-run` flag included (D-07 was Claude's Discretion) — a preview mode is standard safety for a one-off command writing real production rows
- Candidate query is deliberately not scoped to ground-only sites — `_project_calendar_event()` already branches correctly on `Observatory.observations_type`, so including a hypothetical satellite range candidate is safe and avoids an unnecessary extra filter
- `candidates` is materialized as a `list(...)` before iterating so the closing summary's total reflects the original candidate set rather than a queryset re-evaluated after events have been written mid-loop

## Deviations from Plan

None - plan executed exactly as written.

## Manual Smoke Test Finding (not acted on)

Running `./manage.py backfill_range_calendar_events --dry-run` against the real dev DB
(`src/fomo_db.sqlite3`) surfaced 3 candidates, not just the motivating pk=34 row:

```
Would backfill run pk=34 (Didymos 2026: Gemini South/GMOS-s) window 2026-07-13..2026-07-16
Would backfill run pk=27 (3I/ATLAS (demo): FTN/FLOYDS) window 2025-08-01..2025-08-15
Would backfill run pk=29 (Crash Test Campaign: FTN/MuSCAT3) window 2027-04-20..2027-05-11
```

pk=34 is exactly the confirmed motivating case (GS-2026A-FT-115). pk=27 and pk=29 read as
demo/test data left in the dev DB from earlier phases' notebook/UAT runs, not real campaign
data the operator necessarily wants backfilled. Per the plan's own verification section, the
real (non-dry-run) invocation against `src/fomo_db.sqlite3` is an **operator-facing manual
step, not an automated gate** — this executor deliberately did not run the real backfill
against the live dev DB, since deciding whether pk=27/29 should also get calendar events is a
data-ownership judgment call for the operator, not something to apply unilaterally during
plan execution. The command itself is proven correct and idempotent by the test suite and the
dry-run smoke test; running it for real against `src/fomo_db.sqlite3` is left for the operator.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. Optional operator follow-up: run
`./manage.py backfill_range_calendar_events` (without `--dry-run`) against
`src/fomo_db.sqlite3` once the pk=27/29 question above is resolved, to give pk=34 (and any
other genuinely-desired candidates) their per-night `CalendarEvent`s.

## Next Phase Readiness

- Phase 25 (FIX-01..08) is now fully implemented and tested: Plan 01 rewrote
  `_project_calendar_event()`/`_set_run_status()` for per-night ground projection, and this
  plan closes the retroactive-backfill gap for runs approved before that rewrite existed
- Full `python manage.py test solsys_code` (544 tests) and `ruff check`/`ruff format --check`
  on both new plan files are green

---
*Phase: 25-range-window-calendarevent-projection-allow-approved-site-re*
*Completed: 2026-07-18*

## Self-Check: PASSED
