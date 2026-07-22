---
phase: 19-window-schema-migration
plan: 02
subsystem: database
tags: [django, coverage-gap, campaignrun, window-schema]

# Dependency graph
requires:
  - phase: 19-window-schema-migration (plan 01)
    provides: "CampaignRun.window_start/window_end nullable DateFields replacing obs_date/ut_start/ut_end"
provides:
  - "campaign_gap.claimed_dates() reads window_start/window_end directly, claiming every date in an inclusive range"
  - "_observing_night_date() helper and its ZoneInfo/time-based night-derivation logic removed entirely"
  - "test_campaign_gap.py fixtures fully migrated off obs_date/ut_start/ut_end"
affects: [phase-20-range-tbd-import, phase-20-asset-aware-coverage-gap]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Inclusive date-range claiming via a timedelta loop (n_days = (window_end - window_start).days + 1), mirroring the existing observable_dates() loop shape"

key-files:
  created: []
  modified:
    - solsys_code/campaign_gap.py
    - solsys_code/tests/test_campaign_gap.py

key-decisions:
  - "Deleted _observing_night_date() and its ZoneInfo/time imports outright rather than keeping it as dead code — window_start/window_end are already plain dates, no time-of-day-to-night-boundary conversion is needed anymore (Pitfall 4)"
  - "Deleted (not renamed) test_ut_start_only_keys_to_site_local_observing_night since it tested the now-deleted _observing_night_date() fallback path directly and is untestable under the new schema (Pitfall 5)"
  - "Updated every CampaignRun fixture across the whole test file (not just TestClaimedDates/TestClaimedDatesMultiTarget) to window_start/window_end, since obs_date is no longer a valid model field after 19-01's migration and any fixture still passing obs_date= raises TypeError"

patterns-established:
  - "Pattern: window-range claiming for a pure date-pair field mirrors observable_dates()'s existing per-day timedelta loop, kept intentionally simple (no asset-awareness) per this phase's explicit ASSET-02 phase-boundary"

requirements-completed: [SCHED-02, SCHED-03]

coverage:
  - id: D1
    description: "claimed_dates() reads window_start/window_end directly and claims every date in [window_start, window_end] inclusive for a counted approved run (single night and multi-date range both proven)"
    requirement: "SCHED-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClaimedDates.test_approved_run_claims_its_single_night_window"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClaimedDates.test_range_run_claims_every_date_in_window"
        status: pass
    human_judgment: false
  - id: D2
    description: "A TBD run (window_start is None) is collected into undated_runs and never added to the claimed set; multi-target unattributed_runs bucketing is unchanged"
    requirement: "SCHED-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClaimedDates.test_undated_runs_flagged"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestClaimedDatesMultiTarget.test_target_none_run_is_unattributed_not_claimed_for_either_target"
        status: pass
    human_judgment: false
  - id: D3
    description: "_observing_night_date() and its now-dead ZoneInfo/time imports are removed; ruff check is clean on campaign_gap.py"
    verification:
      - kind: unit
        ref: "ruff check solsys_code/campaign_gap.py -- All checks passed!"
        status: pass
    human_judgment: false

duration: ~10min
completed: 2026-07-09
status: complete
---

# Phase 19 Plan 2: Coverage-Gap claimed_dates() Window Rewrite Summary

**campaign_gap.claimed_dates() rewritten to claim every date in an inclusive window_start/window_end range directly, replacing the obs_date/ut_start-derived night-boundary logic and deleting the now-dead `_observing_night_date()` helper.**

## Performance

- **Duration:** ~10 min
- **Completed:** 2026-07-09T22:34:41Z
- **Tasks:** 2
- **Files modified:** 2 (campaign_gap.py, test_campaign_gap.py)

## Accomplishments
- `claimed_dates()` now queries `.only('pk', 'window_start', 'window_end')` (PII-free, unchanged discipline) and claims every date in `[window_start, window_end]` inclusive via a `timedelta` loop identical in shape to the existing `observable_dates()` loop — a single-night run (`window_start == window_end`) claims exactly one date; a range claims every date in between.
- A run with `window_start is None` (TBD) is bucketed into `undated_runs` and never contributes to the claimed set — the single-target vs. multi-target `unattributed_runs` bucketing rule (Pitfall 4) is preserved exactly.
- `_observing_night_date()` (the local-noon-anchored `ut_start`-to-observing-night helper) and its `ZoneInfo`/`time` imports are deleted entirely — `ruff check solsys_code/campaign_gap.py` is clean (no `F401`).
- `test_campaign_gap.py` fully migrated off `obs_date`/`ut_start`/`ut_end`: `TestClaimedDates.test_approved_completed_run_claimed_via_obs_date` renamed to `test_approved_run_claims_its_single_night_window`; a new `test_range_run_claims_every_date_in_window` proves the inclusive multi-date range-claim loop; `test_ut_start_only_keys_to_site_local_observing_night` deleted (tested a now-deleted code path, per Pitfall 5); every other `CampaignRun` fixture in the file (`TestClaimedDatesMultiTarget`, `TestGapAnalysisView`, `TestGapAnalysisButton`) updated to seed `window_start`/`window_end`.
- No asset-awareness (ground-vs-space) logic was added to `claimed_dates()` — every counted run claims its full window regardless of site type, exactly as scoped (that distinction is explicitly Phase 20's ASSET-02).

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite claimed_dates() to iterate the window range; delete _observing_night_date()** - `f0479f7` (feat)
2. **Task 2: Rewrite TestClaimedDates for the window schema** - `23057a4` (test)

## Files Created/Modified
- `solsys_code/campaign_gap.py` - `claimed_dates()` rewritten to iterate `[window_start, window_end]`; `_observing_night_date()` and its `ZoneInfo`/`time` imports deleted
- `solsys_code/tests/test_campaign_gap.py` - `TestClaimedDates` rewritten (single-night + range claiming, TBD bucketing); night-derivation test deleted; every `CampaignRun` fixture across the file migrated to `window_start`/`window_end`

## Decisions Made
- Deleted `_observing_night_date()` outright (not deprecated/kept) since `window_start`/`window_end` are already plain dates with no time-of-day component to convert — matches the plan's Pitfall 4 guidance and keeps `ruff check` clean.
- Deleted (not renamed) `test_ut_start_only_keys_to_site_local_observing_night` since the code path it tested no longer exists and the test is untestable under the new schema, per Pitfall 5.
- Updated fixtures across the entire test file, not just the two classes Task 2's prose names explicitly (`TestClaimedDates`/`TestClaimedDatesMultiTarget`) — `TestGapAnalysisView` and `TestGapAnalysisButton` also seed `CampaignRun` rows and would otherwise raise `TypeError` on the now-removed `obs_date` kwarg. This matches Task 2's own `read_first` note ("31 total" `obs_date`/`ut_start` references across the file) and its action text ("Update every fixture that seeds CampaignRuns... across the file").

## Deviations from Plan

None - plan executed exactly as written for this plan's own file scope (`campaign_gap.py`, `test_campaign_gap.py`).

## Issues Encountered

`TestGapAnalysisView.test_table_view_does_not_trigger_computation` and all three `TestGapAnalysisButton` test cases still fail with `FieldError: Cannot resolve keyword 'obs_date' into field` — they exercise `CampaignRunTableView`, whose `ALLOWED_FIELDS_FOR_NON_STAFF` list in `campaign_views.py` still references the removed `obs_date` field. `campaign_views.py` is explicitly scoped to sibling plan `19-03-PLAN.md`'s `files_modified`, not this plan's — this is the exact cross-plan gap `19-01-SUMMARY.md`'s "Next Phase Readiness" section flagged ("7 other non-test modules... still reference the removed obs_date/ut_start/ut_end fields"). `19/23` tests in `solsys_code.tests.test_campaign_gap` pass; the remaining 4 failures are pre-existing (caused by 19-01's migration, not by this plan's changes) and resolve once `19-03` updates `campaign_views.py` in the same wave. Verified this is not caused by my changes: `campaign_views.py` was not touched by either of this plan's two commits.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `campaign_gap.py` compiles and works cleanly against `window_start`/`window_end`; its own consumer-side tests (`TestClaimedDates`, `TestClaimedDatesMultiTarget`, `TestObservableDates`, `TestClampDateRange`, `TestBuildGapCacheKey`, `TestNoHeavyEphemerisImport`, and the two `TestGapAnalysisView` cases that don't touch the table view) all pass.
- `python manage.py test solsys_code` (full app suite) will not be fully green until sibling plan `19-03` (same wave) updates `campaign_views.py`'s `ALLOWED_FIELDS_FOR_NON_STAFF` and `campaign_tables.py`/`campaign_forms.py` — per this phase's own verification note, full-suite green is a wave-merge gate, not a per-plan one.
- Phase 20's asset-aware `claimed_dates()` rewrite (ASSET-02) can build directly on this plan's window-range claiming loop — the ground-vs-space distinction is the only thing still missing from `claimed_dates()`, deliberately deferred per this plan's own scope boundary.

---
*Phase: 19-window-schema-migration*
*Completed: 2026-07-09*

## Self-Check: PASSED

All modified files found on disk (`solsys_code/campaign_gap.py`, `solsys_code/tests/test_campaign_gap.py`); both task commit hashes (`f0479f7`, `23057a4`) found in git log.
