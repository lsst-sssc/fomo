---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
plan: 03
subsystem: coverage-gap-analysis
tags: [django, coverage-gap, observation-record, provenance-blind, status-vocabulary]

requires:
  - phase: 37-01
    provides: "solsys_code/status_vocabulary.py -- classify_record()/DisplayState this plan's claim source classifies observation events through"
provides:
  - "campaign_gap.observation_claimed_dates()/observation_site_obscode() -- the second GAPB-01 claim source: an observed/scheduled observation block claims its site-local night alongside approved run windows"
  - "claimed_dates()'s widened return tuple (observation_claimed, site_unknown_count) and _compute_gap()'s two new result-dict keys (observation_claimed_dates, claimed_site_unknown_count)"
  - "campaignrun_gap_analysis.html surfaces which kind of claim covered a night and how many observations could not be placed on a site"
affects: [37-04, 37-07]

actuals:
  tokens: 10600
  tasks: 2
  commits: 2
  plan_head_before: e7a36d2bc963d659070b895d60a8b850ff946c6c

tech-stack:
  added: []
  patterns:
    - "Second claim source added alongside the first inside claimed_dates() (add-alongside, not a pluggable-source registry) -- matches the plan's assumption_delta_decision"
    - "D-17 site-resolution ladder implemented as a pure function (observation_site_obscode()) reusing campaign_attribution's existing label/site-code-to-obscode bridge tables and its private _extract_lco_site_code() helper, mirroring the codebase's existing cross-module private-helper-reuse precedent (allocation_projector -> campaign_reconciler._skip_reason)"

key-files:
  created: []
  modified:
    - solsys_code/campaign_gap.py
    - solsys_code/tests/test_campaign_gap.py
    - src/templates/campaigns/campaignrun_gap_analysis.html
    - solsys_code/tests/test_campaign_views.py

key-decisions:
  - "observation_site_obscode()'s rung 2 (attributed run's site) is read from a single annotated scalar column (F('calendar_event_meta__run__site__obscode')) rather than select_related, so the new ObservationRecord query can never be widened into CampaignRun.contact_person/.contact_email (T-37-09)."
  - "A selected site with no usable IANA timezone (ZoneInfoNotFoundError) makes observation_claimed_dates() return (set(), 0) rather than raising -- the per-record D-17 site-unknown counter is reserved for a record's own site being unresolvable, not for the query-level site parameter's own timezone being unset."
  - "Added a 'Claimed nights' list to the gap page (not explicitly named in must_haves.truths but required by Task 2's <action> block and by the 'reader can tell a forward-looking awarded night from a night that was actually used' acceptance bar) -- marks each date in result.claimed_dates as covered by an observation vs. an approved run window."

requirements-completed: [GAPB-01]

coverage:
  - id: D1
    description: "An observed or scheduled observation block claims its site-local night alongside today's approved run-window claims, unioned (not substituted) into claimed_dates()"
    requirement: GAPB-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestObservationClaimedDates.test_observed_block_claims_its_site_local_night"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestObservationClaimedDates.test_scheduled_block_claims_its_site_local_night"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestObservationClaimedDates.test_union_with_approved_run_window_produces_one_claimed_date"
        status: pass
    human_judgment: false
  - id: D2
    description: "A queued request's window claims nothing; an expired, cancelled, failed or inconsistent record claims nothing"
    requirement: GAPB-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestObservationClaimedDates.test_queued_record_with_a_request_window_claims_nothing"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestObservationClaimedDates.test_terminal_and_inconsistent_records_claim_nothing"
        status: pass
    human_judgment: false
  - id: D3
    description: "An observation whose site cannot be resolved is reported as a claimed-but-site-unknown count and closes no per-site gap -- never silently dropped"
    requirement: GAPB-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestObservationClaimedDates.test_record_with_unresolvable_site_increments_unknown_count_not_claimed"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestGapAnalysisSiteUnknownCount"
        status: pass
    human_judgment: false
  - id: D4
    description: "A record resolving to a different observatory than the one selected claims nothing for that site; a record whose record_time_window() raises is skipped as unknown, never aborting the computation"
    requirement: GAPB-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestObservationClaimedDates.test_record_resolving_to_a_different_observatory_claims_nothing_for_this_site"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_gap.py#TestObservationClaimedDates.test_record_time_window_raising_is_skipped_never_aborts_the_loop"
        status: pass
    human_judgment: false
  - id: D5
    description: "The gap page names both claim kinds in its empty-state copy, marks which nights an observation covered, and shows the site-unknown count only when non-zero"
    requirement: GAPB-01
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestGapAnalysisSiteUnknownCount.test_site_unknown_count_line_renders_when_nonzero"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestGapAnalysisSiteUnknownCount.test_site_unknown_count_line_absent_when_zero"
        status: pass
    human_judgment: false

duration: 46min
completed: 2026-09-19
status: complete
---

# Phase 37 Plan 03: Provenance-Blind Coverage-Gap Analysis Summary

**`campaign_gap.claimed_dates()` now unions approved-run-window claims with a second claim source -- observed/scheduled observation blocks on the campaign calendar -- so classical and queue observations of a campaign's own target read as covered even before anyone works the attribution queue, with an unresolvable-site count surfaced rather than silently dropped.**

## Performance

- **Duration:** ~46 min
- **Started:** 2026-09-18T20:20:00-07:00 (approx., first read of required context)
- **Completed:** 2026-09-18T21:06:00-07:00
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- `solsys_code/campaign_gap.py` gained `_CLAIMING_DISPLAY_STATES`, `observation_site_obscode()` (the D-17 site-resolution ladder) and `observation_claimed_dates()` (the D-16/D-18 second claim source), all reading `status_vocabulary.classify_record()` and `observation_projector.facility_for()` rather than re-deriving a classifier.
- `claimed_dates()` unions the new observation claims into its existing run-window claims (D-19) and returns a widened 6-tuple; `_compute_gap()` adds `observation_claimed_dates` and `claimed_site_unknown_count` to its result dict. `get_or_compute_gap()`'s existing 1-hour cache wrapper is untouched.
- `campaignrun_gap_analysis.html`'s empty-state copy no longer attributes every claimed night to an approved run alone; a new "Claimed nights" list marks which dates an observation covered vs. an approved run window; a new count line (styled like the existing needs-review block) states how many observations could not be assigned to a site, rendered only when non-zero.
- `test_campaign_gap.py` gained a 15-test class (`TestObservationClaimedDates`) covering all eight required behaviours plus the empty-campaign, no-runs, no-observations and claim-order-independence edges; `test_campaign_views.py` gained a 2-test integration class proving the site-unknown line's real end-to-end appearance/disappearance (not mocked).

## Task Commits

Each task was committed atomically:

1. **Task 1: Observation blocks claim nights alongside run windows** - `dec12a8` (feat)
2. **Task 2: Say on the gap page which kind of thing covered a night, and how many could not be placed** - `b18b345` (feat)

**Plan metadata:** committed alongside this SUMMARY.

## Files Created/Modified

- `solsys_code/campaign_gap.py` - `observation_site_obscode()`, `observation_claimed_dates()`, widened `claimed_dates()`/`_compute_gap()`
- `solsys_code/tests/test_campaign_gap.py` - widened all 13 pre-existing `claimed_dates()` unpacking call sites to the 6-tuple; new `TestObservationClaimedDates` class (15 tests)
- `src/templates/campaigns/campaignrun_gap_analysis.html` - corrected empty-state copy, new "Claimed nights" list, new site-unknown count line
- `solsys_code/tests/test_campaign_views.py` - new `TestGapAnalysisSiteUnknownCount` class (2 tests)

## Decisions Made

- **T-37-09 mitigation implemented via a single annotated scalar column, never `select_related`.** `observation_claimed_dates()`'s `ObservationRecord` queryset reads the attributed run's site obscode as `F('calendar_event_meta__run__site__obscode')` and restricts fields with `.only('pk', 'status', 'facility', 'scheduled_start', 'scheduled_end', 'parameters')` -- confirmed by a source-level check that `select_related` does not appear anywhere in the function (including its docstring, which was reworded once to avoid a false-positive on the literal string during the plan's own verify check).
- **A query-level site with no usable timezone short-circuits to `(set(), 0)`.** `ZoneInfo(site.timezone)` is constructed once per call; `ZoneInfoNotFoundError`/`TypeError`/`ValueError` there means the selected site itself can't anchor a site-local night for any observation, which is a different failure mode from a per-record D-17 site-unknown result and must not be conflated with it.
- **A "Claimed nights" section was added to the gap page.** The plan's `must_haves.truths` list only describes the computation contract, but Task 2's `<action>` block explicitly asks to "mark a date that appears in `result.observation_claimed_dates` as covered by an observation" and the acceptance criteria requires the template to reference that key -- since the template previously never rendered `result.claimed_dates` at all, a new list section was the natural way to satisfy both.
- **Reused `campaign_attribution`'s private `_extract_lco_site_code()` helper** rather than re-deriving the SITECODE-CLASS-label-to-site-code split. This mirrors an existing precedent in this codebase (`allocation_projector.py` imports `campaign_reconciler._skip_reason`), and re-deriving it would risk the exact drift RESEARCH.md's "Don't Hand-Roll" section warns against.
- **GAPB-01 stays "blocked" in REQUIREMENTS.md's traceability table.** It is declared by both this plan and plan 37-07 (the legacy-title re-title sweep / paired-docs plan). Per the shared-ID gate (`requirements.ready-ids`), it cannot flip to Complete until 37-07 also finishes -- confirmed via the tool (`0/1 requirement(s) ready to mark complete`). Mirrors plan 37-01's identical STATUS-01 deferral.

## Deviations from Plan

None - plan executed exactly as written. (One self-correction during development: the docstring for `observation_claimed_dates()` originally used the literal phrase `select_related('calendar_event_meta__run')` to describe what it deliberately avoids, which tripped the plan's own `select_related` grep check since `inspect.getsource()` includes docstrings; reworded before committing, not treated as a deviation since no behavior changed.)

## Issues Encountered

None. Full `solsys_code` test suite (1533 tests, excluding `test_views.TestEphemeris`) passed `OK (skipped=1)`; the two `test_views` tests this project's own test command runs separately also passed (40 tests, `OK`).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**GAPB-01 stays "blocked" in REQUIREMENTS.md's traceability table for now** -- it is also declared by plan 37-07 (the paired-docs/re-title sweep plan), which has not run yet. Per the shared-ID gate (`requirements.ready-ids`), confirmed via the tool ("0/1 requirement(s) ready to mark complete"), it cannot flip to Complete until 37-07 also finishes. This plan's own implementation and verification are complete regardless -- the gate only defers the traceability-table checkbox, mirroring plan 37-01's identical STATUS-01 situation.

`campaign_gap.observation_claimed_dates()`/`observation_site_obscode()` are the only new public symbols this plan produces; plan 37-07's paired-docs work should be aware that `campaignrun_gap_analysis.html`'s wording changed (empty-state copy, new "Claimed nights"/site-unknown sections) when it writes the coverage-gap runbook section RESEARCH.md confirmed does not exist yet.

No blockers.

---
*Phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps*
*Completed: 2026-09-19*

## Self-Check: PASSED
