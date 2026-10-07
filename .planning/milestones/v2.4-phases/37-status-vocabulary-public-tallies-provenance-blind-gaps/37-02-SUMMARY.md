---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
plan: 02
subsystem: campaign-coordination
tags: [django, lco-portal, proposal-allocation, unattended-runner, tdd]

requires:
  - phase: 36-unattended-operation
    provides: "unattended.py's STEPS registry, StepResult, command_lock() -- the fifth step this plan adds joins"
  - phase: 35-allocation-layer-classical-cutover
    provides: "telescope_runs.ParsedRun.proposal / _resolve_proposal() -- the bracketed [proposal] token this plan's classical-loader edit finally stores structurally"
provides:
  - "CampaignRun.proposal_code -- the structured carrier of a run's proposal code, populated by the classical loader"
  - "ProposalTimeAllocation model -- one row per (proposal_code, semester, instrument_type, allocation_type), written only by the unattended runner"
  - "solsys_code/proposal_allocation.py -- proposal_codes_to_fetch()/fetch_proposal_allocations()/store_proposal_allocations()/unused_hours_for()/estimated_unused_nights()/refresh_all(), the read-only-at-request-time source for a proposal's estimated unused nights"
  - "unattended.step_proposal_allocation() -- the fifth and final STEPS entry, refreshing every watched/run-carried proposal's time allocation once per tick"
affects: [37-04, 37-05, 37-06, 37-07]

actuals:
  tokens: 11400
  tasks: 3
  commits: 4
  plan_head_before: dbed6757646b04581b4ef28c78f3200b83ccd231

tech-stack:
  added: []
  patterns:
    - "Portal fetch confined to an unattended runner step; every public/request-time reader consumes only the stored ProposalTimeAllocation rows, never the portal directly (D-07's trust-boundary rule)"
    - "Store every reported allocation type, sum only a configured subset into the derived estimate -- ESTIMATE_ALLOCATION_TYPES is the single place the summation rule lives, so it can change without a re-fetch"

key-files:
  created:
    - solsys_code/proposal_allocation.py
    - solsys_code/tests/test_proposal_allocation.py
    - solsys_code/migrations/0023_proposal_time_allocation_and_campaignrun_proposal_code.py
  modified:
    - solsys_code/models.py
    - solsys_code/admin.py
    - solsys_code/management/commands/load_telescope_runs.py
    - solsys_code/unattended.py
    - solsys_code/tests/test_load_telescope_runs.py
    - solsys_code/tests/test_admin.py
    - solsys_code/tests/test_unattended.py

key-decisions:
  - "Task 1 checkpoint (human decision, recorded verbatim): proposal-code carrier is `a-store-all-types` -- add CampaignRun.proposal_code; store EVERY allocation type the portal returns (std, rr, tc, AND an undocumented realtime pair a prior live portal check found) as separate ProposalTimeAllocation rows; sum only ESTIMATE_ALLOCATION_TYPES=('std',) into the unused-so-far estimate. Rationale the human accepted: a later change to the summation rule then needs only a code change, never a re-fetch. tc/rr/realtime rows are deliberately stored but not summed -- a real proposal (UTX2026A-002, per the prior agent's live check) holds nonzero Time-Critical hours the human chose to preserve for a later policy decision rather than count now."
  - "estimated_unused_nights() rounds ties away from zero via math.floor(x + 0.5), not Python's round() (which uses banker's rounding to even and would report 2, not 3, for 25 unused standard hours) -- the plan's own behavior spec required 25/10=2.5 to report 3."
  - "unused_hours_for() returns None (never 0.0) when a proposal has no stored rows at all, distinguishing 'not yet fetched' from 'zero unused' -- callers (37-04/37-05/37-06) must render None as unknown."
  - "fetch_proposal_allocations()'s except clause and PortalUnavailable payload copy calendar_utils.resolve_placement_block()'s SYNC-09/D-11 discipline verbatim: the caught exception is never referenced, stringified, or logged -- only type(exc).__name__ crosses the boundary, because ImproperCredentialsException/forms.ValidationError embed the response body and the request carried the API key."

requirements-completed: []

coverage:
  - id: D1
    description: "CampaignRun.proposal_code is a structured, blank-default field populated by the classical loader from the parsed bracketed [proposal] token, leaving the existing free-text observation_details line unchanged"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_load_telescope_runs.py#TestLoadTelescopeRuns.test_proposal_token_populates_campaignrun_proposal_code"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_load_telescope_runs.py#TestLoadTelescopeRuns.test_no_proposal_token_leaves_campaignrun_proposal_code_blank"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_proposal_allocation.py#CampaignRunProposalCodeFieldTests"
        status: pass
    human_judgment: false
  - id: D2
    description: "ProposalTimeAllocation stores one row per (proposal_code, semester, instrument_type, allocation_type), find-or-update on that key (never a duplicate), read-only everywhere except the unattended runner's own fetch step"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_proposal_allocation.py#ProposalTimeAllocationModelTests"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_proposal_allocation.py#StoreProposalAllocationsTests"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_admin.py#ProposalTimeAllocationAdminTests.test_every_field_is_readonly"
        status: pass
    human_judgment: false
  - id: D3
    description: "The portal fetch runs only inside the unattended tick (step_proposal_allocation, fifth/final STEPS entry) with dry-run/lock/failure isolation matching step_status_refresh's shape; a public page never triggers a credentialed call"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_unattended.py#TestProposalAllocationStep"
        status: pass
      - kind: other
        ref: "python manage.py run_unattended --dry-run (proposal_allocation reports 'skipped (dry run)')"
        status: pass
    human_judgment: false
  - id: D4
    description: "estimated_unused_nights()/unused_hours_for() answer from stored rows only (no network call), distinguish None ('never fetched') from a genuine zero, and round the D-06 estimate with ties away from zero"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_proposal_allocation.py#UnusedHoursForTests"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_proposal_allocation.py#EstimatedUnusedNightsTests"
        status: pass
    human_judgment: false
  - id: D5
    description: "No log line, StepResult.summary, or stored row ever contains the portal API key or a raw response body -- verified by a credential-hygiene test forcing an ImproperCredentialsException through the real code path"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_unattended.py#TestProposalAllocationStep.test_summary_leaks_no_api_key_or_response_body"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_proposal_allocation.py#FetchProposalAllocationsTests.test_credential_error_leaks_nothing"
        status: pass
    human_judgment: false

duration: 95min
completed: 2026-09-19
status: complete
---

# Phase 37 Plan 02: Proposal Time Allocation Summary

**A `CampaignRun.proposal_code` field, a `ProposalTimeAllocation` model, and a `solsys_code/proposal_allocation.py` fetch/estimate module give Phase 37 the one figure it didn't have — a proposal's unused awarded time from the LCO Observation Portal — fetched once per unattended tick and served to public pages with zero credentialed calls at request time.**

## Performance

- **Duration:** ~95 min (continuation agent resuming after a human checkpoint decision; Task 1 was resolved by the human before this agent started — see Decisions Made)
- **Started:** 2026-09-19T05:04Z (this agent's start; the original Task 1 checkpoint was raised and answered in a prior session)
- **Completed:** 2026-09-19T05:39Z
- **Tasks:** 3 (Task 1: checkpoint resolved by human, recorded here; Tasks 2-3: TDD RED->GREEN)
- **Files modified:** 10 (3 created, 7 modified)

## Accomplishments

- `CampaignRun.proposal_code` is a blank-default `CharField` beside `source_identifier`; `load_telescope_runs` now writes the classical loader's parsed bracketed `[proposal]` token into it while leaving the existing free-text `observation_details` proposal line unchanged.
- `ProposalTimeAllocation` (new model, migration `0023`) stores one row per (proposal code, semester, instrument type, allocation type), create-or-update on that key, strip-on-save mirroring `WatchedProposal`, and is registered in the admin with every field read-only.
- `solsys_code/proposal_allocation.py` is the only writer of `ProposalTimeAllocation`: `fetch_proposal_allocations()` copies `calendar_utils.resolve_placement_block()`'s call shape and credential-safe except-clause verbatim; `store_proposal_allocations()` stores every allocation type the portal returns (`std`/`rr`/`tc`/`realtime`) but `unused_hours_for()`/`estimated_unused_nights()` sum only `std` into the D-06 estimate, per the Task 1 human decision.
- `unattended.step_proposal_allocation()` joins `STEPS` as the fifth and final entry, mirroring `step_status_refresh()`'s dry-run/lock/try-except/`StepResult` shape; `python manage.py run_unattended --dry-run` lists it and exits 0.

## Task Commits

Each task was committed atomically (Tasks 2 and 3 followed the TDD RED->GREEN commit contract, matching this dispatch's TDD-applicable instruction even though `workflow.tdd_mode` is `false` for this project):

1. **Task 1: Decide the proposal-code carrier and the unused-hours formula** - resolved by human checkpoint decision in a prior session (no commit in this plan's range; the decision is recorded in this SUMMARY's Decisions Made).
2. **Task 2: The proposal-allocation model, the run's proposal code, and one migration** - RED `87f5238` (test) -> GREEN `094e0f9` (feat)
3. **Task 3: Fetch the portal's time allocations in the unattended tick, and expose the estimate** - RED `f4bcf01` (test) -> GREEN `aeef307` (feat)

**Plan metadata:** committed alongside this SUMMARY.

## TDD Gate Compliance

`workflow.tdd_mode` is `false` for this project, so the tool-mediated `gsd_run check tdd-red-evidence` gate was not invoked. RED evidence for both TDD tasks was verified manually against the named import/attribute errors (the exact new symbol each RED test imports or calls does not yet exist), matching this repo's Phase 34 (34-07) and Phase 37 (37-01) precedent for the same situation.

- Task 2 RED (`87f5238`): `test_proposal_allocation.py`/`test_admin.py` failed with `ImportError: cannot import name 'ProposalTimeAllocation'`/`'ProposalTimeAllocationAdmin'`; `test_load_telescope_runs.py`'s new test failed with `AttributeError: 'CampaignRun' object has no attribute 'proposal_code'`. All three confirmed intentional (target symbols did not exist), no INVALID_RED pattern (no zero-test discovery, no fixture crash, no unrelated failure).
- Task 2 GREEN (`094e0f9`): all 99 tests across the three affected modules pass after the model/admin/loader edits and migration `0023` land.
- Task 3 RED (`f4bcf01`): `test_proposal_allocation.py` failed with `ImportError: cannot import name 'proposal_allocation'` (the module was temporarily moved aside to confirm genuine RED after having already been drafted); `test_unattended.py`'s new `TestProposalAllocationStep` class failed with `AttributeError: module 'solsys_code.unattended' has no attribute 'step_proposal_allocation'` (6 of 7 tests) and one assertion failure (`STEPS` still ending in `'reconcile'`).
- Task 3 GREEN (`aeef307`): the module was restored, `step_proposal_allocation()` added and registered fifth in `STEPS`; all 99 tests across `test_proposal_allocation`/`test_unattended`/`test_load_telescope_runs`/`test_admin` pass (192 combined with Task 2's modules).

## Files Created/Modified

- `solsys_code/models.py` - `ProposalTimeAllocation` model; `CampaignRun.proposal_code` field
- `solsys_code/migrations/0023_proposal_time_allocation_and_campaignrun_proposal_code.py` - `AddField` + `CreateModel`, generated then renamed per the plan's mandated filename, operations unedited
- `solsys_code/admin.py` - `ProposalTimeAllocationAdmin` (every field read-only), registered; `proposal_code` added to `CampaignRunAdmin.list_display`
- `solsys_code/management/commands/load_telescope_runs.py` - `'proposal_code': parsed.proposal or ''` added to the per-run `fields` dict
- `solsys_code/proposal_allocation.py` - the new fetch/store/estimate module (created across Tasks 2-3's tests, then Task 3's implementation)
- `solsys_code/unattended.py` - `step_proposal_allocation()`; `STEPS` gains its fifth entry
- `solsys_code/tests/test_proposal_allocation.py` - new test module (model tests + module tests, created across Tasks 2-3)
- `solsys_code/tests/test_load_telescope_runs.py` - two new tests for `proposal_code` population
- `solsys_code/tests/test_admin.py` - `ProposalTimeAllocationAdminTests`
- `solsys_code/tests/test_unattended.py` - `TestProposalAllocationStep`

## Decisions Made

- **Task 1 human checkpoint decision (recorded verbatim, resolved in a prior session before this agent started):** proposal-code carrier is `a-store-all-types` — add `CampaignRun.proposal_code`; store EVERY allocation type the portal returns (`std`, `rr`, `tc`, AND an undocumented `realtime_allocation`/`realtime_time_used` pair a prior live portal check found on the real `timeallocation_set` payload) as separate `ProposalTimeAllocation` rows; sum only `ESTIMATE_ALLOCATION_TYPES = ('std',)` into the "unused so far" estimate. The human's accepted rationale: a later change to the summation rule then needs only a code change, never a re-fetch. `tc`/`rr`/`realtime` rows are deliberately stored but not summed — real proposal `UTX2026A-002` was found (via a prior agent's live portal check) to hold nonzero Time-Critical time (10.0/4.11 and 10.0/6.06 hours), which the human chose to preserve in the database for a later policy decision rather than count now.
- **`estimated_unused_nights()` rounds ties away from zero via `math.floor(x + 0.5)`**, not Python's built-in `round()` (banker's-rounding-to-even would report 2, not 3, for 25 unused standard hours) — the plan's own behavior spec required 2.5 to round to 3. Safe because `unused_hours_for()` always floors its input at zero first.
- **`unused_hours_for()` returns `None` (never `0.0`) for a proposal with zero stored rows**, distinguishing "not yet fetched" from "genuinely zero unused" — every downstream reader (37-04's tally, 37-05's Progress column, 37-06's unused-night decoration) must render `None` as unknown, never as zero.
- **`fetch_proposal_allocations()`'s except clause and `PortalUnavailable` payload copy `calendar_utils.resolve_placement_block()`'s SYNC-09/D-11 discipline verbatim**: only `type(exc).__name__` ever crosses the function boundary (via `raise ... from None`, suppressing exception chaining) — the caught exception itself is never referenced, stringified, or logged, because `ImproperCredentialsException`/`forms.ValidationError` embed response content and the request carried the API key (T-37-04).
- **`store_proposal_allocations()` treats a missing `<type>_allocation`/`<type>_time_used` PAIR as absent, not zero** — no row is written for a time type the portal response doesn't carry at all, per the plan's explicit behavior spec.
- **`PortalUnavailable` keeps its plan-mandated exact name** despite ruff's `N818` (Exception-suffix) naming rule, suppressed with a `# noqa: N818` comment mirroring `unattended.LockContended`'s identical precedent (`36-01-PLAN.md`).

## Deviations from Plan

None - plan executed exactly as written, with Task 1's outcome supplied by the continuation prompt's authoritative human decision (not re-asked, per the resume instructions).

## Issues Encountered

None. All 192 tests across the four affected test modules (`test_proposal_allocation`, `test_unattended`, `test_load_telescope_runs`, `test_admin`) pass; `python manage.py makemigrations --check --dry-run` exits 0 both before and after the migration; `python manage.py run_unattended --dry-run` exits 0 and lists `proposal_allocation` as a fifth, skipped-dry-run step; `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` both pass repo-wide.

## User Setup Required

None - no external service configuration required. The portal fetch reuses the existing `FACILITIES['LCO']['api_key']` credential every other portal call in this codebase already uses; no new environment variable or secret name.

## Next Phase Readiness

`solsys_code/proposal_allocation.py`'s `estimated_unused_nights()`/`unused_hours_for()` and `CampaignRun.proposal_code` are the shared foundation the rest of Phase 37's tally work consumes:

- Plan 37-04 (TALLY-01/02/03/UNUSED-01) reads `estimated_unused_nights(run.proposal_code)` for the container-run "unused so far" figure and must render `None` as "not yet fetched", never as zero.
- Plan 37-05 (TALLY-01/02) reads the same for the campaign-table Progress column and roll-up.
- Plan 37-06 (TALLY-01/UNUSED-01) reads it for the unused-night calendar decoration.
- Plan 37-07 owns the paired-docs update (the runbook's "How do I run everything unattended?" section gains the fifth step) — not scoped to this plan's `files_modified`.

**TALLY-01 stays "Pending" in `REQUIREMENTS.md`'s traceability table.** It is declared by this plan and four siblings (37-04 through 37-07), none of which have run yet. Per the shared-ID gate (`requirements.ready-ids`), it cannot flip to Complete until every declaring plan finishes — this plan's own implementation and verification are complete regardless.

No blockers. The full targeted regression the plan's own `<verification>` names (192 tests across `test_proposal_allocation`/`test_unattended`/`test_load_telescope_runs`/`test_admin`, `makemigrations --check --dry-run`, `run_unattended --dry-run`, repo-wide `ruff`/`ruff-format`) is green. A broader `solsys_code`-wide regression run (the project's full `test_command`) was also launched during this plan's close-out as an extra cross-module safety net beyond the plan's own scope; it was still running when this SUMMARY was written and its result is not claimed here either way.

---
*Phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps*
*Completed: 2026-09-19*

## Self-Check: PASSED

- `solsys_code/proposal_allocation.py` -- FOUND
- `solsys_code/tests/test_proposal_allocation.py` -- FOUND
- `solsys_code/migrations/0023_proposal_time_allocation_and_campaignrun_proposal_code.py` -- FOUND
- Commit `87f5238` -- FOUND
- Commit `094e0f9` -- FOUND
- Commit `f4bcf01` -- FOUND
- Commit `aeef307` -- FOUND
- Plan-level `<verification>` re-confirmed: `makemigrations --check --dry-run` exit 0; 192 tests across `test_proposal_allocation`/`test_unattended`/`test_load_telescope_runs`/`test_admin` pass; `run_unattended --dry-run` exit 0 listing `proposal_allocation`; `pre-commit run ruff --all-files` / `ruff-format --all-files` both Passed
