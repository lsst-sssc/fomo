---
phase: 17-coverage-gap-analysis-deferrable-to-v2-1
plan: 01
subsystem: api
tags: [django, cache, campaign, ephemeris, coverage-gap]

# Dependency graph
requires:
  - phase: 14-campaign-data-model-bootstrap-import
    provides: CampaignRun model (approval_status/run_status, obs_date, ut_start, site FK)
  - phase: 01-site-ephemeris-helper (v1.0)
    provides: telescope_runs.sun_event()/get_site() dark-window ephemeris helpers
provides:
  - "campaign_gap.py: pure-logic coverage-gap computation (clamp_date_range, build_gap_cache_key, observable_dates, claimed_dates, get_or_compute_gap)"
  - "17-GAP-01-DECISION.md: citable dark-window-only decision artifact satisfying GAP-01"
  - "Unit test coverage for all D-03 through D-11 behaviors, ready for Plan 02's view wiring"
affects: [17-02-campaign-gap-analysis-view, 17-03-campaign-gap-analysis-template]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pure-logic computation module (campaign_gap.py) mirrors campaign_utils.py's role -- no view/request concerns, never raises for expected messy data"
    - "Cache-or-compute wrapper with computed_at stamped only on miss, preserved on hit (D-10)"
    - "Per-date log+skip on ValueError, never abort the observable-dates loop (D-03)"

key-files:
  created:
    - .planning/phases/17-coverage-gap-analysis-deferrable-to-v2-1/17-GAP-01-DECISION.md
    - solsys_code/campaign_gap.py
    - solsys_code/tests/test_campaign_gap.py
  modified: []

key-decisions:
  - "GAP-01 satisfied via a short written decision doc (17-GAP-01-DECISION.md) rather than a fresh spike, per D-02 -- pre-milestone research already unanimously recommended dark-window-only"
  - "Multi-target campaigns: target=None CampaignRuns are collected into a separate unattributed_runs list, never counted as claiming any specific target's dates (Pitfall 4, RESEARCH.md recommendation A1)"
  - "Single-target campaigns: claimed_dates() does not filter by target at all -- the single target is implied and target=None is the common real-data case, matching import_campaign_csv's D-07 precedent"

patterns-established:
  - "campaign_gap.py's module docstring explicitly states its only ephemeris dependency (telescope_runs.sun_event), mirroring campaign_views.py's existing 'never import ephem_utils/solsys_code.views' discipline"

requirements-completed: [GAP-01, GAP-02]

coverage:
  - id: D1
    description: "17-GAP-01-DECISION.md documents the dark-window-only decision with rationale citing pre-milestone research"
    requirement: "GAP-01"
    verification:
      - kind: manual_procedural
        ref: "document review: .planning/phases/17-coverage-gap-analysis-deferrable-to-v2-1/17-GAP-01-DECISION.md"
        status: pass
    human_judgment: false
  - id: D2
    description: "observable_dates() returns dates with a non-zero dark window, skipping ValueError dates without aborting the loop"
    requirement: "GAP-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestObservableDates.test_returns_dates_with_nonzero_dark_window"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestObservableDates.test_valueerror_date_is_skipped_loop_completes"
        status: pass
    human_judgment: false
  - id: D3
    description: "claimed_dates() excludes cancelled/not_awarded/weather_tech_failure runs, derives night from obs_date else ut_start via site-local observing-night convention, and flags undated/unattributed runs separately"
    requirement: "GAP-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClaimedDates.test_approved_completed_run_claimed_via_obs_date"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClaimedDates.test_cancelled_run_not_claimed"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClaimedDates.test_pending_review_run_not_claimed"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClaimedDates.test_ut_start_only_keys_to_site_local_observing_night"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClaimedDates.test_undated_runs_flagged"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClaimedDatesMultiTarget.test_target_none_run_is_unattributed_not_claimed_for_either_target"
        status: pass
    human_judgment: false
  - id: D4
    description: "clamp_date_range enforces the 90-day default and 180-day hard cap regardless of client input"
    requirement: "GAP-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClampDateRange.test_default_window_is_90_days"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClampDateRange.test_far_future_end_clamps_to_180_days"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClampDateRange.test_request_inside_cap_is_honoured"
        status: pass
    human_judgment: false
  - id: D5
    description: "build_gap_cache_key includes all four dimensions (campaign, target, site, date range); a null target is encoded explicitly, never omitted"
    requirement: "GAP-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestBuildGapCacheKey.test_key_contains_all_four_dimensions"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestBuildGapCacheKey.test_null_vs_real_target_do_not_collide"
        status: pass
    human_judgment: false
  - id: D6
    description: "campaign_gap.py never imports the heavy SPICE-loading ephemeris module or solsys_code.views at module scope"
    requirement: "GAP-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestNoHeavyEphemerisImport.test_campaign_gap_source_has_no_forbidden_imports"
        status: pass
      - kind: other
        ref: "grep -rnE forbidden-import-pattern solsys_code/campaign_gap.py solsys_code/tests/test_campaign_gap.py (zero matches)"
        status: pass
    human_judgment: false

duration: 21min
completed: 2026-07-04
status: complete
---

# Phase 17 Plan 01: Coverage-Gap Computation Core Summary

**Pure-logic `campaign_gap.py` module composing `telescope_runs.sun_event()` (observable side) with a `CampaignRun` query (claimed side) into a cached set-difference, plus the GAP-01 dark-window-only decision artifact.**

## Performance

- **Duration:** 21 min
- **Started:** 2026-07-04T21:24:27Z (session start; decision doc committed 22:26:43)
- **Completed:** 2026-07-04T22:33:14Z
- **Tasks:** 3 completed
- **Files modified:** 3 (all new)

## Accomplishments
- Wrote `17-GAP-01-DECISION.md`, satisfying GAP-01's success criterion with a citable
  decision artifact documenting the dark-window-only choice and its rationale.
- Built `solsys_code/campaign_gap.py`: `clamp_date_range`, `build_gap_cache_key`,
  `observable_dates`, `claimed_dates`, `_observing_night_date`, `_compute_gap`,
  `get_or_compute_gap` -- the full pure-computation core for GAP-02, depending only on
  `telescope_runs.sun_event` for ephemerides.
- Added 16 passing unit tests in `solsys_code/tests/test_campaign_gap.py` covering every
  D-03 through D-11 decision, plus a static import-guard test mirroring the plan's own
  grep verification.

## Task Commits

Each task was committed atomically:

1. **Task 1: Write the GAP-01 dark-window-only decision artifact** - `589e949` (docs)
2. **Task 2: Build the campaign_gap.py pure-computation module** - `cd4061c` (feat)
3. **Task 3: Unit tests for campaign_gap.py + module-scope import guard** - `bcfba8a` (test)

**Plan metadata:** _(this commit, docs: complete plan)_

## Files Created/Modified
- `.planning/phases/17-coverage-gap-analysis-deferrable-to-v2-1/17-GAP-01-DECISION.md` - GAP-01 decision doc: dark-window-only, rationale, consequences
- `solsys_code/campaign_gap.py` - pure-logic coverage-gap computation core (constants, clamp/key-builder, observable/claimed-date functions, cache wrapper)
- `solsys_code/tests/test_campaign_gap.py` - 16 unit tests across 6 test classes covering all plan behaviors and the import guard

## Decisions Made
- Multi-target campaign `target=None` runs are collected into a separate `unattributed_runs`
  list and never counted as claiming either target's date (Pitfall 4's recommendation,
  since CONTEXT.md's locked decisions don't address this case directly).
- `unknown_date_count` is computed as `range_len - len(observable_dates)`, since under D-04
  (any non-zero dark window counts as observable) every date that didn't raise `ValueError`
  ends up in the observable set -- so a date missing from that set is exactly a date whose
  `sun_event()` call raised and was skipped.
- Test fixtures build `Observatory` rows directly with a `timezone` set (rather than relying
  on dev-DB seed data), per RESEARCH.md's Environment Availability guidance, so tests never
  depend on live MPC calls or pre-existing seed data.

## Deviations from Plan

None - plan executed exactly as written. All exported symbols, constants, and test classes
match the plan's `<artifacts_produced>` and `<tasks>` specifications exactly.

## Issues Encountered

None. `ruff check`/`ruff format --check` clean on both new source files; the 5 pre-existing
`ruff check .` findings repo-wide are all in unrelated Jupyter notebook files, not touched by
this plan. Full `./manage.py test solsys_code` suite (319 tests, up from 303 pre-phase) passes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

`campaign_gap.py`'s full public API (`get_or_compute_gap`, `clamp_date_range`,
`build_gap_cache_key`) is ready for Plan 02 to wire into a `CampaignGapAnalysisView` in
`campaign_views.py`, following the IDOR-revalidation and D-12/D-13 target/site-selection
patterns already scoped in 17-RESEARCH.md/17-PATTERNS.md. No blockers. Plan 02 should be aware
of the measured ~520ms/call `sun_event()` cost (17-RESEARCH.md Pitfall 1) when designing the
view's UX copy around the up-to-~94s worst-case synchronous wait for a 180-day window.

---
*Phase: 17-coverage-gap-analysis-deferrable-to-v2-1*
*Completed: 2026-07-04*

## Self-Check: PASSED

All 4 created files verified present on disk; all 4 task/plan commits (`589e949`, `cd4061c`,
`bcfba8a`, `2c98dbc`) verified present in git log.
