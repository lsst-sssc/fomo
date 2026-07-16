---
phase: 23-weather-storm-cancellation-handling-give-staff-a-way-to-mark
plan: 03
subsystem: testing
tags: [django-test, campaign-coordination, mpc-obscode, calendar-event, gemini]

# Dependency graph
requires:
  - phase: 23-weather-storm-cancellation-handling-give-staff-a-way-to-mark
    provides: "Plan 02's CampaignRunDecisionView._set_run_status() action (mark_cancelled/mark_weather_failure), and its existence-guarded CalendarEvent sync"
provides:
  - "Proof that resolve_site('I11') resolves the real Gemini South Observatory as a ground-based, non-placeholder site with a real America/Santiago timezone via the resolver's actual Tier-2 single-code path (D-06)"
  - "Proof that the real GS-2026A-FT-115 Gemini range-window run flows through the same approve -> mark_weather_failure -> mark_cancelled mechanism as any Magellan run, with no special-casing, projecting zero CalendarEvents throughout (D-07)"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Tier-2 MPC resolver test pattern: patch MPCObscodeFetcher.query (no-op MagicMock) AND MPCObscodeFetcher.to_observatory (side_effect creating a real Observatory row) together -- BULK_MPC_FIXTURE only feeds the unrelated bulk query_all()/build_site_candidates() path and must never be used to fake resolve_site()'s single-code Tier-2 path"

key-files:
  created: []
  modified:
    - solsys_code/tests/test_campaign_approval.py

key-decisions:
  - "Corrected the plan's original 'add I11 to BULK_MPC_FIXTURE' approach (REVIEW finding #2, HIGH): that fixture feeds query_all()/build_site_candidates() (the bulk fuzzy-match widget path), not resolve_site()'s single-code Tier-2 query()/to_observatory() path -- used the Phase 21 P04 dual-patch precedent instead"
  - "TestGeminiFtScenario creates its own 'Didymos 2026' TargetList (not the shared CampaignApprovalTestBase '3I/ATLAS' campaign) to match the real D-06 seed values exactly, passed as a campaign= override to _make_pending_run()"
  - "Live GS-2026A-FT-115 dev-DB row is deliberately operator data entry (user_setup), not a committed fixture -- the two tasks in this plan prove the mechanism handles that exact row shape without needing the live row to exist"

requirements-completed: [D-06, D-07]

coverage:
  - id: D1
    description: "resolve_site('I11') resolves Gemini South as a ground-based, non-placeholder Observatory with a real IANA timezone via the actual Tier-2 single-code resolver path, no manual admin edit needed"
    requirement: D-06
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestResolveSiteI11GeminiSouth.test_resolve_site_i11_resolves_gemini_south_ground_based"
        status: pass
    human_judgment: false
  - id: D2
    description: "The Gemini FT-115 range-window run flows through the same approve -> mark-status mechanism as any Magellan run (no special-casing), projecting zero CalendarEvents through approve, mark_weather_failure, and mark_cancelled"
    requirement: D-07
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestGeminiFtScenario.test_gemini_ft115_range_window_flows_through_same_mechanism_no_event_fabricated"
        status: pass
      - kind: integration
        ref: "./manage.py test solsys_code (full suite, 536/536 pass)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Operator creates the live GS-2026A-FT-115 informational CampaignRun row in the dev DB via the existing admin/approval paths (user_setup runbook)"
    verification: []
    human_judgment: true
    rationale: "Deliberately operator data entry per user_setup -- not something this plan's automated tasks create; requires a human to act through Django admin or the submission form against the live dev DB"

# Metrics
duration: 35min
completed: 2026-07-16
status: complete
---

# Phase 23 Plan 3: Gemini FT-115 Test Encoding Summary

**Proved resolve_site('I11') resolves Gemini South (ground, real timezone) and that the real GS-2026A-FT-115 range-window Gemini run flows through the exact same approve/mark-status mechanism as any Magellan run, with zero CalendarEvents fabricated at any step (D-06/D-07) -- no production code added.**

## Performance

- **Duration:** ~35 min (includes two full 536-test suite runs for regression verification)
- **Started:** 2026-07-16T22:00:00Z (approx, after Plan 02 close)
- **Completed:** 2026-07-16T22:18:30Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- `resolve_site('I11')` proven to resolve the real Gemini South Observatory (Cerro Pachón) as a ground-based, non-placeholder site with a real `America/Santiago` timezone via the resolver's ACTUAL Tier-2 single-code path (`MPCObscodeFetcher.query()` → `to_observatory()`), with `needs_review=False` -- no manual admin edit needed before a Decided-table status change (D-06).
- End-to-end proof that the real Gemini Fast-Turnaround GS-2026A-FT-115 informational run (Didymos 2026 TargetList, `Gemini-South GMOS-S`, `I11` site, 2026-07-13..2026-07-16 range window, `Thomas-Osip` contact, `target=None`) flows through the exact same approve → `mark_weather_failure` → `mark_cancelled` mechanism as any Magellan run, with no special-casing: approving the 4-day range window projects zero `CAMPAIGN:{pk}` `CalendarEvent`s (range-window projection skipped by design), marking it weathered returns a normal redirect and sets `run_status=WEATHER_TECH_FAILURE` with still zero events, and a follow-up `mark_cancelled` is proven a REAL `WEATHER_TECH_FAILURE → CANCELLED` transition (two distinct `RunStatus` values, not a no-op) with still zero events fabricated (D-07, RESEARCH Pitfall 1, T-23-07).
- Full `./manage.py test solsys_code` suite (536/536) green -- no regression from Plans 01/02/03 merged.
- Correctly identified and avoided the plan's originally-proposed `BULK_MPC_FIXTURE` approach for testing `resolve_site()`'s Tier-2 path (that fixture feeds the unrelated bulk `query_all()`/`build_site_candidates()` widget path), instead using the established Phase 21 P04 dual-patch precedent (`MPCObscodeFetcher.query` no-op + `MPCObscodeFetcher.to_observatory` `side_effect`).

## Task Commits

Each task was committed atomically:

1. **Task 1: Verify resolve_site('I11') resolves Gemini South as a ground-based site with a real timezone** - `e61ef89` (test)
2. **Task 2: End-to-end Gemini FT-115 scenario — same mechanism, no special-casing, no event fabricated** - `aeda66e` (test)

**Plan metadata:** (this commit)

_Note: Both tasks are test-only additions -- no production code was created or modified, per the plan's objective ("no new code path, just new data")._

## Files Created/Modified
- `solsys_code/tests/test_campaign_approval.py` - Added `_stub_i11_to_observatory()` helper, `TestResolveSiteI11GeminiSouth` (D-06 Tier-2 resolver proof), and `TestGeminiFtScenario` (D-07 end-to-end mechanism proof); also added `is_placeholder_observatory` to the existing `campaign_utils` import.

## Decisions Made
- Did NOT add `'I11'` to `BULK_MPC_FIXTURE` as the original plan draft suggested (REVIEW finding #2, HIGH, Codex) -- confirmed by reading `resolve_site()` that it never calls `query_all()`/`build_site_candidates()`, so that fixture cannot influence the single-code Tier-2 path at all. Instead patched `MPCObscodeFetcher.query` (no-op) and `MPCObscodeFetcher.to_observatory` (`side_effect` creating a real Observatory row) directly, mirroring the Phase 21 P04 precedent recorded in STATE.md.
- `TestGeminiFtScenario` creates its own `Didymos 2026` `TargetList` (distinct from `CampaignApprovalTestBase`'s shared `3I/ATLAS` campaign) via a `campaign=` override to `_make_pending_run()`, to match the real D-06 seed values exactly rather than reusing an unrelated campaign name.
- Kept the I11 `Observatory` fixture in `TestGeminiFtScenario.setUpTestData` as a Tier-1-resolvable local row (mirroring `TestCalendarProjection`'s `F65` `ground_site` convention) so the end-to-end scenario resolves the site deterministically offline via Tier 1 -- Task 1 already separately proves the Tier-2 resolver path for I11, so Task 2 doesn't need to re-mock the MPC fetch.

## Deviations from Plan

None - plan executed exactly as written, including the plan's own REVIEW-finding-#2 correction (the `BULK_MPC_FIXTURE` avoidance was already specified in the plan text itself, not discovered during execution).

## Issues Encountered

Committing the two tasks atomically required splitting a single-file diff (both tasks touch `solsys_code/tests/test_campaign_approval.py`) into two sequential commits: staged Task 1's addition (import line + `TestResolveSiteI11GeminiSouth`) first via a temporary truncated copy of the file, verified `ruff format --check` and the test suite passed on that intermediate state, committed, then restored the full file (adding `TestGeminiFtScenario`) and committed Task 2 separately. No content was lost; the reconstructed full file was diffed byte-for-byte identical against the original edit before committing Task 1.

One full-suite background test run's completion status was ambiguous due to apparent PID reuse (a `ps -p <pid>` check reported "still running" then "finished" for what may have been a different process reusing the same PID) -- resolved by re-running the full suite cleanly with output redirected to a dedicated log file and an explicit `EXIT_CODE:$?` marker, confirming 536/536 tests passed with exit code 0 before proceeding to commit Task 2.

## User Setup Required

**Operator data entry required for the live production row (not automatable, not part of this plan's tasks).** Per the plan's `user_setup` block:

1. Create the FT-115 `CampaignRun` via Django admin (or the existing CSV/submission path) under the Didymos 2026 `TargetList` (pk=1) with: `telescope_instrument='Gemini-South GMOS-S'`, `site_raw='I11'`, `window_start=2026-07-13`, `window_end=2026-07-16`, `contact_person='Thomas-Osip'`, `observation_details` noting "GS-2026A-FT-115, 6.50 awarded hours" (informational only — NOT a real Gemini ODB submission), `target` left unset (None), `run_status=REQUESTED`.
   - Location: Django admin → Solsys_code → Campaign runs → Add, or the existing campaign submission form.
2. Let the approval queue resolve the site: `'I11'` Tier-2-resolves to `'Gemini South Observatory, Cerro Pachon'` (ground-based) via the live MPC ObsCodes API, with a real `America/Santiago` timezone backfilled automatically. Approve the run through the approval queue. If the storm hits FT-115, mark it Weathered from the Decided table exactly like the Magellan runs (D-07, no special-casing) — verified safe by this plan's `TestGeminiFtScenario` test (no crash, no fabricated event, since the window is a 4-day range).
   - Location: Approval queue → Decided table.

This step was deliberately NOT performed by this executor run — it is real production data entry against the live dev DB, out of scope for this plan's automated tasks per the plan's own instructions.

## Next Phase Readiness
- Phase 23 (weather-storm-cancellation-handling) is now fully implemented and test-verified across all 3 plans (01/02/03). All D-01..D-07 decisions from the phase's CONTEXT/RESEARCH are proven in code and tests.
- The only remaining action is the operator runbook step above (creating the live FT-115 row) — no code blockers.
- Full `./manage.py test solsys_code` suite: 536/536 passing. `ruff check .` / `ruff format --check .` clean on all Phase 23 files.

---
*Phase: 23-weather-storm-cancellation-handling-give-staff-a-way-to-mark*
*Completed: 2026-07-16*

## Self-Check: PASSED

- FOUND: solsys_code/tests/test_campaign_approval.py
- FOUND commit: e61ef89
- FOUND commit: aeda66e
