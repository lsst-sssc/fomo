---
phase: 20-range-tbd-import-asset-aware-coverage-gap
plan: 01
subsystem: api
tags: [django, coverage-gap, observatory, asset-aware, campaign]

# Dependency graph
requires:
  - phase: 19-window-schema-migration
    provides: nullable window_start/window_end DateField pair on CampaignRun, Observatory.observations_type == SATELLITE_OBSTYPE ground-vs-space precedent (campaign_views.py:339)
provides:
  - "claimed_dates() 4-tuple return ending in pending_narrowing_runs, distinguishing ground vs. space-mission CampaignRuns"
  - "_compute_gap() result dict pending_narrowing_runs key"
  - "campaignrun_gap_analysis.html alert block surfacing pending_narrowing_runs count"
affects: [21-site-disambiguation-submitter-contact-optin]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Bucketed non-claiming run lists (undated_runs, unattributed_runs, pending_narrowing_runs) computed once per site classification, never per-row site reads"]

key-files:
  created: []
  modified:
    - solsys_code/campaign_gap.py
    - solsys_code/tests/test_campaign_gap.py
    - src/templates/campaigns/campaignrun_gap_analysis.html

key-decisions:
  - "Ground-vs-space classification computed once from the site parameter (is_space_mission = site.observations_type == Observatory.SATELLITE_OBSTYPE) before the per-run loop, never per-row, preserving the PII-minimizing .only('pk','window_start','window_end') queryset (Pitfall 3)."
  - "TBD runs (both window fields null) always land in undated_runs regardless of site type; only a space-mission run with an un-narrowed range (window_start != window_end) lands in pending_narrowing_runs (D-09 explicit distinction)."
  - "No change to campaign_views.py — context.update({'result': result}) already passes the whole result dict opaquely, so the template reaches the new key via normal dict-dotted lookup (Pitfall 4)."

patterns-established:
  - "Third bucketed non-claiming run list (pending_narrowing_runs) follows the exact declare/append/continue shape as the existing undated_runs list."

requirements-completed: [ASSET-01, ASSET-02]

coverage:
  - id: D1
    description: "claimed_dates() distinguishes ground vs. space-mission CampaignRuns: ground runs keep claiming every date in their window; un-narrowed space-mission runs claim nothing and land in a new pending_narrowing_runs bucket; TBD runs stay in undated_runs regardless of site type"
    requirement: "ASSET-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClaimedDatesSpaceMission (test_narrowed_space_run_claims_its_single_night, test_unnarrowed_space_run_claims_nothing_and_lands_in_pending_narrowing, test_tbd_space_run_lands_in_undated_not_pending_narrowing)"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClaimedDates.test_range_run_claims_every_date_in_window"
        status: pass
    human_judgment: false
  - id: D2
    description: "A single-night space-mission run (window_start == window_end) claims exactly that one date, exercising ASSET-02's narrowed-window path"
    requirement: "ASSET-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClaimedDatesSpaceMission.test_narrowed_space_run_claims_its_single_night"
        status: pass
    human_judgment: false
  - id: D3
    description: "campaign_gap_analysis page renders a distinct alert-info block reporting the count of pending_narrowing_runs when the bucket is non-empty, separate from the existing undated_runs/unattributed_runs alert"
    requirement: "ASSET-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_gap.py#TestGapAnalysisView.test_pending_narrowing_alert_shown_for_unnarrowed_space_run"
        status: pass
    human_judgment: false

duration: ~20min
completed: 2026-07-10
status: complete
---

# Phase 20 Plan 01: Asset-Aware Coverage Gap Summary

**claimed_dates() now distinguishes ground vs. space-mission CampaignRuns — space-mission runs with an un-narrowed window claim nothing and surface in a new pending_narrowing_runs gap-page alert, computed once from the site parameter without widening the PII-minimizing queryset.**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-07-10T19:04Z (approx, based on commit timestamps)
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- `claimed_dates()` returns a 4-tuple ending in `pending_narrowing_runs`; ground-vs-space classification is computed once from the `site` parameter before the per-run loop (`is_space_mission = site.observations_type == Observatory.SATELLITE_OBSTYPE`), never per-row, so the existing `.only('pk', 'window_start', 'window_end')` PII-minimizing queryset is unchanged (Pitfall 3).
- Space-mission runs with `window_start != window_end` claim zero dates and are collected into `pending_narrowing_runs`, distinct from `undated_runs` (TBD, both window fields null) — matching D-09's "no info at all" vs. "a real space-mission run, just not scheduled tight enough yet" distinction.
- Ground-based runs are unaffected: every date in `[window_start, window_end]` is still claimed regardless of range width (regression-verified by the existing `test_range_run_claims_every_date_in_window`).
- `_compute_gap()` unpacks the new 4-tuple and adds `'pending_narrowing_runs'` to its result dict; `get_or_compute_gap()` required no change (opaque cache/passthrough).
- `campaignrun_gap_analysis.html` gains a new `alert-info` block (`{% if result.pending_narrowing_runs %}`) reporting the bucket's count, alongside the existing `undated_runs`/`unattributed_runs` `alert-warning` block. `campaign_views.py` was not touched — the whole result dict already flows into `context['result']`.
- All existing `claimed_dates()` call sites in `test_campaign_gap.py` updated to the new 4-tuple arity; new `TestClaimedDatesSpaceMission` class covers narrowed, un-narrowed, and TBD space-mission runs; new view-level test proves the alert renders with the correct count.

## Task Commits

Each task was committed atomically:

1. **Task 1: Ground-vs-space branch and pending_narrowing_runs bucket in claimed_dates()** - `eec0ec0` (feat)
2. **Task 2: Gap-analysis alert block for pending_narrowing_runs** - `3013ec1` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified
- `solsys_code/campaign_gap.py` - `claimed_dates()` 4-tuple return with `pending_narrowing_runs`; `_compute_gap()` dict key added
- `solsys_code/tests/test_campaign_gap.py` - updated 4-tuple unpackings at every existing call site; added `TestClaimedDatesSpaceMission` (3 tests); added `TestGapAnalysisView.test_pending_narrowing_alert_shown_for_unnarrowed_space_run`
- `src/templates/campaigns/campaignrun_gap_analysis.html` - new alert-info block for `result.pending_narrowing_runs`

## Decisions Made
- Classification computed once before the loop from the `site` parameter, not per-row from `run.site` — avoids an N+1 read and keeps the PII-minimizing `.only()` queryset intact (Pitfall 3, threat T-20-01/T-20-02 mitigations).
- No change to `campaign_views.py` — the dict-passthrough was already generic enough (Pitfall 4).
- Alert placement: a sibling `alert-info` block (not folded into the existing `alert-warning` div) to visually distinguish "pending narrowing" (informational, expected workflow state) from "needs review" (data-quality flag).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Initial `assertContains` string in the view-level test spanned a template line-wrap and failed to match; split into two shorter substring assertions that don't cross the wrap boundary. Fixed within Task 2 before committing (not a deviation from plan scope, just test-string tuning).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `pending_narrowing_runs` is now available wherever `_compute_gap()`'s result dict is consumed; Plan 20-02/20-03/20-04 (import + notebook work) can proceed independently.
- Full `solsys_code` test suite (364 tests) and `ruff check`/`ruff format --check` both pass with no regressions.
- No blockers for the remaining Phase 20 plans.

---
*Phase: 20-range-tbd-import-asset-aware-coverage-gap*
*Completed: 2026-07-10*
