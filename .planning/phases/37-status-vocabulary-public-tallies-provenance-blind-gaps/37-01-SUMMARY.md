---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
plan: 01
subsystem: calendar-display
tags: [django, status-vocabulary, calendar-events, tdd]

requires:
  - phase: 34-the-observation-projector-trigger
    provides: "Provisional [Q]/[S]/[O]/[X]/[C]/[F]/[?] markers and stage_for() classifier shape this plan promotes to canonical"
  - phase: 35-allocation-layer-classical-cutover
    provides: "Run-level [CANCELLED]/[WEATHERED] prefixes and allocation_night_title() this plan migrates to [C]/[W]"
provides:
  - "solsys_code/status_vocabulary.py -- the single definition of every calendar status marker (OCSState, DisplayState, MARKER, LABEL, LEGEND, STAGE_MARKER, FAILURE_MARKER_BY_STATUS, RUN_STATUS_MARKER, RING_QUEUED_STATES, RING_TERMINAL_STATES, RETIRED_TITLE_PREFIXES, state_for_title(), observed_states_for(), failed_states_for(), classify_record(), OBSERVED_STATES_BY_FACILITY)"
  - "One facility-aware terminal classifier (observed_states_for()/classify_record()) replacing the hardcoded 'COMPLETED' string comparison"
affects: [37-02, 37-03, 37-04, 37-05, 37-06, 37-07]

actuals:
  tokens: 14500
  tasks: 3
  commits: 5
  plan_head_before: 6efc6e134de8f2df99981ec93f4a4822a00d76e9

tech-stack:
  added: []
  patterns:
    - "Single shared vocabulary module consumed by every calendar-title producer and display-time consumer, replacing three independently-maintained marker tables"
    - "FOMO-side per-facility observed-state override table (OBSERVED_STATES_BY_FACILITY) rather than trusting a TOM facility's own terminal-state vocabulary"

key-files:
  created:
    - solsys_code/status_vocabulary.py
    - solsys_code/tests/test_status_vocabulary.py
  modified:
    - solsys_code/observation_projector.py
    - solsys_code/templatetags/calendar_display_extras.py
    - solsys_code/campaign_reconciler.py
    - solsys_code/allocation_projector.py
    - solsys_code/calendar_utils.py
    - solsys_code/tests/test_calendar_display_extras.py
    - solsys_code/tests/test_campaign_approval.py
    - solsys_code/tests/test_allocation_projector.py
    - solsys_code/tests/test_load_telescope_runs.py
    - solsys_code/tests/test_write_and_reconcile.py
    - solsys_code/tests/test_calendar_utils.py

key-decisions:
  - "STATUS-01 marked complete only in the traceability sense possible today -- it is shared with plan 37-07 (the legacy-title re-title sweep) and stays 'blocked' in REQUIREMENTS.md until that plan also finishes; STATUS-02 (this plan's sole classifier requirement) is marked complete now."
  - "status_border_css() keeps its literal '[QUEUED] ' word-form check as a small, deliberate duplication alongside the new state_for_title()-driven queued/terminal ring buckets, rather than adding '[QUEUED]' to RETIRED_TITLE_PREFIXES -- it was never one of the vocabularies this phase's D-01/D-02 decisions name for migration, and the existing display-extras test suite pins the exact word-form behavior."

patterns-established:
  - "Peer module under solsys_code/ that imports only solsys_code.models at module scope, matching campaign_gap.py's heavy-import discipline documented in its own docstring."

requirements-completed: [STATUS-01, STATUS-02]

coverage:
  - id: D1
    description: "One module, solsys_code/status_vocabulary.py, defines every calendar status marker; observation_projector, campaign_reconciler, allocation_projector and calendar_display_extras read from it and hold no marker table of their own"
    requirement: STATUS-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_status_vocabulary.py#TestOneMarkerOneModule"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_status_vocabulary.py#TestVocabularyStructure"
        status: pass
    human_judgment: false
  - id: D2
    description: "The calendar legend lists every visible state in one fixed order [Q][S][O][X][C][F][W][?][U], read from status_vocabulary.LEGEND, never from a database query; [S] is named Scheduled"
    requirement: STATUS-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_display_extras.py#TestObservationStatusLegend.test_returns_nine_entries_covering_every_marker"
        status: pass
    human_judgment: false
  - id: D3
    description: "Run-level and record-level cancellation share the [C] marker; weather/technical failure has [W]; RUN_STATUS_MARKER has exactly two entries so no other RunStatus value can acquire a marker by accident"
    requirement: STATUS-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_status_vocabulary.py#TestRunStatusMarker"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py, solsys_code/tests/test_allocation_projector.py, solsys_code/tests/test_load_telescope_runs.py, solsys_code/tests/test_write_and_reconcile.py"
        status: pass
    human_judgment: false
  - id: D4
    description: "One facility-aware terminal classifier (observed_states_for()/classify_record()) replaces the hardcoded status == 'COMPLETED' check; no facility's own terminal-state vocabulary alone makes a record read as observed -- GEM/ESO records classify as submitted, never observed"
    requirement: STATUS-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_status_vocabulary.py#TestFacilityAwareClassifier"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_calendar_utils.py#TestResolvePlacementBlockFailureModes.test_completed_first_else_pending_selection"
        status: pass
    human_judgment: false
  - id: D5
    description: "The legacy bracket-word titles ([CANCELLED]/[WEATHERED]/[EXPIRED]/[FAILED]) are still recognised by the status ring via RETIRED_TITLE_PREFIXES until plan 37-07 proves the developer database holds none, so no stored event silently loses its ring during the migration"
    requirement: STATUS-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_status_vocabulary.py#TestRunStatusMarker.test_cancelled_and_legacy_cancelled_share_the_same_terminal_ring"
        status: pass
    human_judgment: false

duration: 70min
completed: 2026-09-19
status: complete
---

# Phase 37 Plan 01: Status Vocabulary Consolidation Summary

**Collapsed three independently-maintained calendar-status-marker vocabularies (the observation projector's stage/failure marker dicts, the reconciler's run-status prefix dict, and the display-extras legend/ring tuples) into one module, `solsys_code/status_vocabulary.py`, and routed the last hardcoded portal-state literal (`'COMPLETED'`) through a shared facility-aware classifier.**

## Performance

- **Duration:** ~70 min (includes several long full-test-suite runs, one ~9.5 min and one ~17.5 min, triggered by the codebase's real astropy/sun-event-heavy test modules)
- **Started:** 2026-09-19T02:06Z
- **Completed:** 2026-09-19T03:16Z
- **Tasks:** 3 (Task 1 tracer + Tasks 2-3 TDD)
- **Files modified:** 13 (2 created, 11 modified)

## Accomplishments

- `solsys_code/status_vocabulary.py` is now the single definition of every calendar status marker: `OCSState`, `DisplayState`, `MARKER`, `LABEL`, `LEGEND` (9 entries, fixed order `[Q] [S] [O] [X] [C] [F] [W] [?] [U]`), `STAGE_MARKER`, `FAILURE_MARKER_BY_STATUS`, `RUN_STATUS_MARKER`, `RING_QUEUED_STATES`, `RING_TERMINAL_STATES`, `RETIRED_TITLE_PREFIXES`, `state_for_title()`.
- `observation_projector.py`, `campaign_reconciler.py`, `allocation_projector.py` and `calendar_display_extras.py` no longer carry a local copy of any marker/prefix/legend table -- all four import from `status_vocabulary` and the "must stay byte-identical" comments describing the old drift risk are gone.
- Run-level cancellation (`CampaignRun.RunStatus.CANCELLED`) and weather/technical failure (`WEATHER_TECH_FAILURE`) now write the short markers `[C]`/`[W]` instead of the legacy bracket-word `[CANCELLED]`/`[WEATHERED]`, sharing `[C]` with a portal-cancelled `ObservationRecord` per D-02.
- `status_vocabulary.classify_record()`/`observed_states_for()`/`failed_states_for()` give every facility an explicit, FOMO-side mapping onto the canonical LCO/SOAR OCS vocabulary; `GEM` and `ESO` are excluded from ever classifying as observed (`OBSERVED_STATES_BY_FACILITY` maps both to an empty set), closing the "a submitted-but-unobserved ToO reads as observed" risk named in the plan's prohibitions.
- `calendar_utils.resolve_placement_block()`'s two quoted state literals (`'COMPLETED'`, `'PENDING'`) now read from `status_vocabulary.OCSState`; the completed-first-else-pending selection logic is unchanged and pinned by a new test.

## Task Commits

Each task was committed atomically (Tasks 2 and 3 followed the TDD RED->GREEN commit contract per this dispatch's explicit TDD-applicable instruction, even though `workflow.tdd_mode` is `false` for this project):

1. **Task 1: End-to-end "one marker, one module" -- the queued/cancelled record path only** - `2adcd8c` (feat, tracer)
2. **Task 2: Run-level prefixes join the same vocabulary** - RED `df8c449` (test) -> GREEN `f68cf65` (feat)
3. **Task 3: One facility-aware terminal classifier replaces the hardcoded COMPLETED check** - RED `70017e3` (test) -> GREEN `47f8c6f` (feat)

**Plan metadata:** committed alongside this SUMMARY.

## TDD Gate Compliance

`workflow.tdd_mode` is `false` for this project, so the tool-mediated `gsd_run check tdd-red-evidence` gate was not invoked. RED evidence for both TDD tasks was verified manually against the named import errors (the exact new symbol each RED test imports does not yet exist), matching this repo's Phase 34 precedent (34-07 SUMMARY) for the same situation.

- Task 2 RED (`df8c449`): `TestRunStatusMarker.test_run_status_marker_has_exactly_two_entries` and `test_no_other_run_status_value_has_a_marker` failed with `ImportError: cannot import name 'RUN_STATUS_MARKER'` -- confirmed intentional (target symbol did not exist), not an INVALID_RED pattern (no zero-test discovery, no fixture crash, no unrelated failure).
- Task 2 GREEN (`f68cf65`): same two tests pass after `RUN_STATUS_MARKER` was added.
- Task 3 RED (`70017e3`): all 9 tests in `TestFacilityAwareClassifier` failed with `ImportError` on `observed_states_for`/`classify_record`/`OBSERVED_STATES_BY_FACILITY`.
- Task 3 GREEN (`47f8c6f`): all 9 pass after the classifier functions were added.

## Files Created/Modified

- `solsys_code/status_vocabulary.py` - the new single vocabulary module (created across Tasks 1-3)
- `solsys_code/tests/test_status_vocabulary.py` - end-to-end tracer test, run-status-marker tests, facility-classifier tests (created across Tasks 1-3)
- `solsys_code/observation_projector.py` - imports `STAGE_MARKER`/`FAILURE_MARKER_BY_STATUS`/`observed_states_for`/`failed_states_for` instead of local dicts and facility calls
- `solsys_code/templatetags/calendar_display_extras.py` - `status_border_css()`/`observation_status_legend()` read from `status_vocabulary`
- `solsys_code/campaign_reconciler.py` - `event_title()`/`event_description()` read `RUN_STATUS_MARKER`
- `solsys_code/allocation_projector.py` - `allocation_night_title()` reads `RUN_STATUS_MARKER`
- `solsys_code/calendar_utils.py` - `resolve_placement_block()` reads `OCSState.COMPLETED`/`.PENDING`
- `solsys_code/tests/test_calendar_display_extras.py` - legend test updated for the 9-entry order
- `solsys_code/tests/test_campaign_approval.py` - imports `RUN_STATUS_MARKER` from `status_vocabulary` instead of the reconciler
- `solsys_code/tests/test_allocation_projector.py`, `test_load_telescope_runs.py`, `test_write_and_reconcile.py` - pre-existing tests asserting the literal `'[CANCELLED]'` title on newly-produced (not legacy-fixture) events updated to `'[C]'`
- `solsys_code/tests/test_calendar_utils.py` - new test pinning `resolve_placement_block()`'s selection behaviour through the literal-to-constant swap

## Decisions Made

- **STATUS-01 stays "blocked" in REQUIREMENTS.md's traceability table for now.** It is declared by both this plan and plan 37-07 (the legacy-title re-title sweep and `RETIRED_TITLE_PREFIXES` deletion). Per the shared-ID gate (`requirements.ready-ids`), it cannot flip to Complete until 37-07 also finishes -- confirmed via the tool (`ready: [STATUS-02], blocked: [STATUS-01]`). STATUS-02 (this plan's sole classifier requirement, not shared with any other plan) was marked complete.
- **`status_border_css()` keeps a small, deliberate literal check for the legacy `'[QUEUED] '` word-form** rather than folding it into `RETIRED_TITLE_PREFIXES`. The plan's D-01 names only the four cancellation-family bracket words (`[EXPIRED]`, `[CANCELLED]`, `[FAILED]`, `[WEATHERED]`) as the retirement list; `[QUEUED]` was never part of the byte-identical-vocabulary drift this phase cures (it predates the marker system entirely, alongside `[UNVERIFIED]`, which the existing test suite also exercises as a "no known prefix" case). Preserving the literal check kept every pre-existing `calendar_display_extras` test green with no behaviour change.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug/Regression] Pre-existing tests outside this plan's declared `files_modified` asserted the literal legacy `[CANCELLED]` bracket-word title on newly-produced events**
- **Found during:** Task 2's plan-mandated verify command (`python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_campaign_reconciler`)
- **Issue:** `test_allocation_projector.py` (3 sites), `test_load_telescope_runs.py` (2 sites) and `test_write_and_reconcile.py` (1 site) asserted a freshly-written `CampaignRun.RunStatus.CANCELLED` event's title equals or starts with `'[CANCELLED]'` -- the literal string Task 2 intentionally replaces with `'[C]'` for every newly-produced title. These are direct, in-scope regressions of Task 2's own action, not pre-existing unrelated failures (Rule 1's scope boundary is satisfied: only tests asserting on freshly-produced titles were touched; the many other `'[CANCELLED]'`/`'[WEATHERED]'` occurrences left unchanged are legitimate legacy-title *fixtures* or tests of the retained `RETIRED_TITLE_PREFIXES` recognition, verified by reading each site's context).
- **Fix:** Updated the 6 assertion sites (and one docstring) across the three files to expect `'[C]'` instead of `'[CANCELLED]'`.
- **Files modified:** `solsys_code/tests/test_allocation_projector.py`, `solsys_code/tests/test_load_telescope_runs.py`, `solsys_code/tests/test_write_and_reconcile.py`
- **Verification:** Full re-run of all four affected test modules (`test_allocation_projector`, `test_campaign_reconciler`, `test_load_telescope_runs`, `test_write_and_reconcile`) plus the complete `solsys_code` suite (1519 tests) all green.
- **Committed in:** `f68cf65` (part of Task 2's GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 Rule 1). **Impact:** No scope creep -- the fix was a mechanical follow-through of Task 2's own intended behaviour change onto tests the plan's `files_modified` list happened not to enumerate. All other legacy-title occurrences in the test suite were verified, by reading each one's context, to be either unrelated fixture data or intentional tests of the retained backward-compatible `RETIRED_TITLE_PREFIXES` recognition, and were left untouched.

## Issues Encountered

None. The full `solsys_code` test suite (1519 tests across every module except `test_views.TestEphemeris`, plus the two `test_views` tests the project's own `test_command` runs separately) passed with `OK (skipped=1)`, and the two-test `test_views` subset also passed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

`status_vocabulary.py`'s classifier (`classify_record()`) and marker tables are the shared foundation the rest of Phase 37 consumes:

- Plan 37-02 (TALLY-01) reads `classify_record()`/`observed_states_for()` for per-record night classification.
- Plan 37-03 (GAPB-01) reads the same classifier for its "only observed/scheduled blocks claim a night" rule.
- Plan 37-04 (TALLY-01/02/03/UNUSED-01) reads `MARKER`/`DisplayState`/`RING_*` for the tally cell and unused-night decoration.
- Plan 37-07 owns deleting `RETIRED_TITLE_PREFIXES` once it proves (via a re-title sweep against the real developer database) that no stored `CalendarEvent.title` still starts with a legacy bracket-word prefix -- at that point STATUS-01 can flip from blocked to complete.

No blockers. The full regression suite (1519 tests) is green, confirming this consolidation introduced no behavioural regression anywhere in the codebase.

---
*Phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps*
*Completed: 2026-09-19*

## Self-Check: PASSED

- `solsys_code/status_vocabulary.py` -- FOUND
- `solsys_code/tests/test_status_vocabulary.py` -- FOUND
- Commit `2adcd8c` -- FOUND
- Commit `df8c449` -- FOUND
- Commit `f68cf65` -- FOUND
- Commit `70017e3` -- FOUND
- Commit `47f8c6f` -- FOUND
- All plan-level `<verification>` commands re-confirmed: `test_status_vocabulary` (11 tests, OK), full `solsys_code` suite (1519 tests, `OK (skipped=1)`), `pre-commit run ruff --all-files` / `ruff-format --all-files` (both Passed)
