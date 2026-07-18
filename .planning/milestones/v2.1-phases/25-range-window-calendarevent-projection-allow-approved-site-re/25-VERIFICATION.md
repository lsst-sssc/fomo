---
phase: 25-range-window-calendarevent-projection-allow-approved-site-re
verified: 2026-07-18T02:58:36Z
status: passed
score: 15/16 must-haves verified
behavior_unverified: 0
overrides_applied: 0
human_verification:

  - test: "Run `./manage.py backfill_range_calendar_events` (without `--dry-run`) against the live dev DB (`src/fomo_db.sqlite3`), after deciding whether to also backfill pk=27/29 (demo/test data) alongside the real motivating pk=34 (GS-2026A-FT-115) row."
    expected: "CampaignRun pk=34 gets its 4 per-night CalendarEvents (CAMPAIGN:34:2026-07-13 .. CAMPAIGN:34:2026-07-16); `CalendarEvent.objects.filter(Q(url='CAMPAIGN:34') | Q(url__startswith='CAMPAIGN:34:')).count()` becomes 4, resolving the exact symptom the debug report opened with."
    why_human: "This is a one-off, real-production-data-writing operator decision (per Plan 02's own SUMMARY and `<verification>` section, deliberately left un-executed because pk=27/29 also qualify and whether they should be backfilled too is a data-ownership judgment call, not something the executor should decide unilaterally). Verified directly against the live DB during this verification pass: `CalendarEvent.objects.filter(Q(url='CAMPAIGN:34') | Q(url__startswith='CAMPAIGN:34:')).count()` still returns 0 as of this report."
---

# Phase 25: Range-window CalendarEvent Projection Verification Report

**Phase Goal:** Approved, site-resolved range-window CampaignRuns (window_start != window_end) become visible on the campaign calendar: a ground run projects one dip-corrected CalendarEvent per night, a satellite run keeps its single whole-day span, and each range event's title carries a window-context suffix. Marking such a run cancelled/weathered updates every night's event, and a one-off backfill command gives already-approved runs (the real GS-2026A-FT-115 pk=34) their events. The four Phase 19/23 tests that encoded the zero-event behavior are deliberately revised.

**Verified:** 2026-07-18T02:58:36Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Guard drops the `window_start == window_end` clause and requires `window_end` truthiness, admitting range-window runs into projection (FIX-01/D-01) | ✓ VERIFIED | `campaign_views.py:429`: `if not (run.telescope_instrument and run.site and run.window_start and run.window_end): return False` — confirmed via `git diff aeda66e df676f1`, equality clause removed, `window_end` truthiness added. |
| 2 | Ground branch projects one dip-corrected `CalendarEvent` per night; count == `(window_end - window_start).days + 1` (FIX-02/D-02) | ✓ VERIFIED | `campaign_views.py:468-486` per-night loop mirroring `load_telescope_runs`' E-S+1 idiom. `test_approve_range_run_creates_one_event_per_night` asserts count==15 for an 8/1..8/15 fixture — ran directly, passes. `test_gemini_ft115_range_window_projects_per_night_events` asserts count==4 for the real 7/13..7/16 window — ran directly, passes. |
| 3 | First night's `start_time` and last night's `end_time` match `sun_event()` for `window_start`/`window_end` respectively, not a first-night-only mislabel (FIX-03) | ✓ VERIFIED | `test_approve_range_run_creates_one_event_per_night` (lines 368-379) asserts `first_event.start_time == sun_event(..., 2026-08-01).sunset` and `last_event.end_time == sun_event(..., 2026-08-15).sunrise` — ran directly, passes. |
| 4 | Range events keyed `CAMPAIGN:{pk}:{date.isoformat()}`; single-night run keeps bare `CAMPAIGN:{pk}` key unchanged (D-03) | ✓ VERIFIED | `campaign_views.py:485`: `url = f'CAMPAIGN:{run.pk}' if not is_range else f'CAMPAIGN:{run.pk}:{night.isoformat()}'`. Single-night regression test `test_approve_single_night_ground_run_creates_dip_corrected_calendar_event` still passes against the bare key. |
| 5 | Every range-window event title carries the ` (window {start}..{end})` suffix; single-night titles stay byte-identical (D-06) | ✓ VERIFIED | `_calendar_event_title()` (`campaign_views.py:392-401`) is the single source of truth, reused by both creation and status-update paths. Assertions at lines 373, 400, 532, 2297, 2310 confirmed passing. |
| 6 | Satellite range-window run still produces exactly one whole-day-span event under the bare key, date-math unchanged, title suffixed (FIX-04/D-05) | ✓ VERIFIED | `campaign_views.py:437-448` (satellite branch) confirmed byte-unchanged except the title call. `test_approve_range_run_space_site_creates_single_whole_day_span_event` asserts count==1, 00:00→23:59 UTC span, and the suffix — ran directly, passes. |
| 7 | A mid-window `sun_event()` ValueError leaves earlier nights' events in place (partial projection, no `transaction.atomic` rollback); approval still succeeds | ✓ VERIFIED | `test_approve_range_run_partial_projection_on_mid_window_sun_event_error` patches `sun_event` to raise on night 3 of a 4-night window, asserts `APPROVED` + count==2 — ran directly, passes. Source confirms no `transaction.atomic()` wrap around the loop. |
| 8 | `_set_run_status()` updates EVERY matching night's event (bare key + per-night keys) with the `[CANCELLED]`/`[WEATHERED]` prefix, and the window suffix survives (FIX-05/D-04) | ✓ VERIFIED | `campaign_views.py:788-804`: combined `Q(url=...) \| Q(url__startswith=...)` queryset, title built via `f'{prefix} {_calendar_event_title(run)}'`. `test_mark_range_window_run_updates_every_night_event` (15 events) and `test_gemini_ft115_range_window_projects_per_night_events` (4 events, weathered→cancelled) both ran directly and pass. |
| 9 | `resolve_site()`'s retroactive projection creates per-night events + "Site resolved — run added to the calendar." for a range run; a genuine TBD-resolve still yields zero events + "Site resolved." (FIX-06) | ✓ VERIFIED | `campaign_views.py:720-738`. `test_resolve_range_run_projects_per_night_calendar_events` (count==15, correct message) and `test_resolve_tbd_run_clears_flag_with_no_calendar_event` (count==0, plain message) both ran directly and pass. |
| 10 | TBD, unresolved-site, and missing-telescope_instrument runs still project zero events (guard-exclusion prohibition) | ✓ VERIFIED | `test_approve_tbd_run_creates_no_calendar_event`, `test_approve_without_telescope_instrument_creates_no_calendar_event` ran directly and pass; guard truthiness checks confirmed in source. |
| 11 | The three original guard-exclusion tests keep byte-identical assertion bodies (FIX-07 prohibition) | ✓ VERIFIED | `git diff aeda66e 5187e4b -- solsys_code/tests/test_campaign_approval.py` shows all three test bodies (`test_approve_tbd_run_creates_no_calendar_event`, `test_approve_without_telescope_instrument_creates_no_calendar_event`, `test_sun_event_valueerror_skips_projection_without_reverting_approval`) appear only as unmodified context lines, never as `+`/`-` diff lines. |
| 12 | `StaffRequiredMixin` + the `approval_status` business-logic guards in `_set_run_status()`/`_resolve_site()` preserved exactly, not weakened (D-04 prohibition) | ✓ VERIFIED | `git diff aeda66e df676f1` shows the guard block (`approval_status != APPROVED` check, conditional `.update()`, `updated_count == 0` short-circuit, `refresh_from_db()`) is untouched — only the calendar-sync block below it changed. |
| 13 | Backfill command finds qualifying APPROVED range-window resolved-site runs with no existing `CAMPAIGN:{pk}*` event, delegates to `_project_calendar_event()`, never reimplements date-math (FIX-08) | ✓ VERIFIED | `backfill_range_calendar_events.py:50-82`: candidate query, trailing-colon existence pre-check, `_project_calendar_event(run)` call — no inline sun/date-math. `./manage.py backfill_range_calendar_events --help` exits 0 and lists `--dry-run`. |
| 14 | Backfill is idempotent; `--dry-run` writes nothing; a per-candidate `ValueError` is logged/reported and skipped, never aborting the run (FIX-08) | ✓ VERIFIED | 5 tests in `test_backfill_range_calendar_events.py` ran directly and pass: qualifying/idempotent/skips-non-qualifying/dry-run-writes-nothing/skips-and-continues-on-ValueError. Manual `--dry-run` run against the live dev DB by this verifier reproduced the SUMMARY's exact 3-candidate output (pk=34, 27, 29). |
| 15 | Full `solsys_code` test suite passes green; `ruff check`/`ruff format --check` clean | ✓ VERIFIED | `python manage.py test solsys_code -v 0` → "Ran 544 tests ... OK" (run directly by this verifier, not taken from SUMMARY). `ruff check` and `ruff format --check` on all 4 phase-touched files → clean. |
| 16 | The real, disputed motivating case (CampaignRun pk=34, GS-2026A-FT-115) is now actually visible on the calendar (the literal symptom the debug report opened with) | ⚠️ NOT YET DONE (see Human Verification) | This verifier ran `CalendarEvent.objects.filter(Q(url='CAMPAIGN:34') \| Q(url__startswith='CAMPAIGN:34:')).count()` against the live `src/fomo_db.sqlite3` dev DB: **still returns 0**. The code capability is fully proven (truths 1-15), and a `--dry-run` correctly identifies pk=34 as a qualifying candidate, but the real (non-`--dry-run`) backfill has never been executed — SUMMARY 25-02 explicitly documents this as a deliberate, un-acted-on deferral to the operator (pk=27/29 are also candidates and whether to backfill them too is a data-ownership call). |

**Score:** 15/16 truths verified (1 routed to human verification — an operator execution step, not a code defect)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/campaign_views.py` | `_calendar_event_title()` helper, rewritten `_project_calendar_event()` guard/ground-branch, rewritten `_set_run_status()` sync block | ✓ VERIFIED | All three present, substantive, wired (called from `CampaignRunDecisionView`), confirmed via direct read + `git diff` against the pre-phase commit. |
| `solsys_code/tests/test_campaign_approval.py` | 4 renamed range tests + 3 new tests (satellite range, partial projection, genuine TBD resolve), guard-exclusion tests unchanged | ✓ VERIFIED | All 7 test methods present at expected names; old zero-event-encoding names fully removed (no duplication); 114 tests in this file + the backfill file ran directly, all pass. |
| `solsys_code/management/commands/backfill_range_calendar_events.py` | `BaseCommand` with `--dry-run`, delegates to `_project_calendar_event()` | ✓ VERIFIED | Matches plan spec exactly; `--help` exits 0; ruff clean; manual dry-run against live DB matches SUMMARY's documented output. |
| `solsys_code/tests/test_backfill_range_calendar_events.py` | 5 tests: qualifying / idempotent / non-qualifying / dry-run / error-skip | ✓ VERIFIED | All 5 present and pass when run directly. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `_project_calendar_event()` (creation) | `_calendar_event_title()` | direct call, `event_fields['title']` | ✓ WIRED | `campaign_views.py:432`. |
| `_set_run_status()` (status update) | `_calendar_event_title()` | direct call, `f'{prefix} {_calendar_event_title(run)}'` | ✓ WIRED | `campaign_views.py:801` — confirmed no re-derived inline title string exists anywhere in the function (the Pitfall-1 drift this was designed to prevent). |
| `_set_run_status()` | `CalendarEvent` (per-night + bare key) | `Q(url=f'CAMPAIGN:{pk}') \| Q(url__startswith=f'CAMPAIGN:{pk}:')` | ✓ WIRED | `campaign_views.py:788-790`, trailing colon present (Pitfall 2 addressed). |
| `backfill_range_calendar_events.Command` | `campaign_views._project_calendar_event()` | cross-module import + direct call | ✓ WIRED | `backfill_range_calendar_events.py:8,82` — no reimplemented date-math in the command. |

### Requirements Coverage

Phase requirement IDs FIX-01..FIX-08 are phase-local (gap-closure phase from `/gsd-debug`) with no `REQUIREMENTS.md` mapping — confirmed via `grep -n "FIX-0" .planning/REQUIREMENTS.md` returning no matches. This absence is expected, not a gap, per the phase's own roadmap note. All eight FIX-IDs are covered by the observable truths above (FIX-01→#1, FIX-02→#2, FIX-03→#3, FIX-04→#6, FIX-05→#8, FIX-06→#9, FIX-07→#11, FIX-08→#13/#14).

### Anti-Patterns Found

None. Scanned all 4 phase-touched files for `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER`/empty-implementation patterns — all matches were either pre-existing, unrelated vocabulary (`is_placeholder_observatory()`, `NEEDS REVIEW:` tier-3 placeholder Observatory concept, both pre-dating this phase) or the intentional `TBD run` domain terminology carried over from the diagnosed root cause, not debt markers. `ruff check` on all 4 files: clean.

### Behavioral Spot-Checks / Test Execution

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Targeted test_campaign_approval.py + test_backfill_range_calendar_events.py suites | `python manage.py test solsys_code.tests.test_campaign_approval solsys_code.tests.test_backfill_range_calendar_events -v 1` | "Ran 114 tests ... OK" | ✓ PASS |
| Full solsys_code app suite | `python manage.py test solsys_code -v 0` | "Ran 544 tests ... OK" | ✓ PASS |
| ruff lint | `ruff check` on all 4 phase-touched files | "All checks passed!" | ✓ PASS |
| ruff format | `ruff format --check` on all 4 phase-touched files | "4 files already formatted" | ✓ PASS |
| Backfill `--dry-run` against live dev DB | `python manage.py backfill_range_calendar_events --dry-run` | Identified pk=34/27/29 as candidates, matching SUMMARY 25-02's documented output exactly | ✓ PASS |
| Real dev-DB state check for the motivating case | `CalendarEvent.objects.filter(Q(url='CAMPAIGN:34') \| Q(url__startswith='CAMPAIGN:34:')).count()` | `0` | ⚠️ Confirms truth #16 is not yet resolved in the live DB |

All test execution above was run directly by this verifier (not taken from SUMMARY.md claims).

### Human Verification Required

### 1. Execute the real backfill for CampaignRun pk=34 (and decide on pk=27/29)

**Test:** Run `./manage.py backfill_range_calendar_events` (without `--dry-run`) against `src/fomo_db.sqlite3`, after deciding whether pk=27 (`3I/ATLAS (demo): FTN/FLOYDS`) and pk=29 (`Crash Test Campaign: FTN/MuSCAT3`) — both also surfaced as qualifying candidates by the dry-run — should be included, or whether the run should be scoped to pk=34 only (e.g. by approving/rejecting those rows first, or running the command as-is and accepting all three).
**Expected:** `CampaignRun` pk=34 (GS-2026A-FT-115) gets its 4 per-night `CalendarEvent`s; `CalendarEvent.objects.filter(Q(url='CAMPAIGN:34') | Q(url__startswith='CAMPAIGN:34:')).count()` becomes 4, closing the exact symptom (count() returns 0) that the debug report (.planning/debug/range-window-calendar-event.md) opened with.
**Why human:** This writes real rows against production dev data and involves a genuine data-ownership judgment call (whether pk=27/29 demo/test data should also be backfilled) that Plan 02's SUMMARY explicitly declined to make unilaterally. The code capability itself is fully proven by 5 passing unit tests plus a live `--dry-run` smoke test — this item is purely "should this be executed now, and for which rows," not a code defect.

### Gaps Summary

No code, test, or wiring gaps found. All 8 FIX-IDs are implemented, tested (114 targeted + 544 full-suite tests all passing when run directly by this verifier, not merely claimed), and ruff-clean. The guard fix, per-night ground date-math, shared title helper, multi-event status sync, and backfill command all match the diagnosed root cause's before/after spec in `.planning/debug/range-window-calendar-event.md` line-for-line.

The one open item is operational, not a code gap: the real disputed CampaignRun pk=34 row that motivated the entire debug investigation still shows zero `CalendarEvent`s in the live dev DB, because the non-dry-run backfill has deliberately not been executed yet (an explicit, documented decision in Plan 02's SUMMARY, not an oversight). Since the phase's own goal statement and Plan 02's must-haves reference "gives CampaignRun pk=34 ... its 4 per-night events" as an outcome, this is surfaced as a human-verification item rather than silently treated as complete.

---

*Verified: 2026-07-18T02:58:36Z*
*Verifier: Claude (gsd-verifier)*
