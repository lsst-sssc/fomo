---
phase: 30-v2-2-tech-debt-cleanup
plan: 01
subsystem: campaign-attribution
tags: [django, attribution, campaign-run, approval-status, ruff, sphinx, jupyter]

# Dependency graph
requires:
  - phase: 28
    provides: "campaign_attribution.py's eligibility gates, candidate scoring, and is_offered_candidate() server-side re-derivation (operator-assisted attribution)"
provides:
  - "A REJECTED CampaignRun is excluded from attribution eligibility at both gates (_eligible_runs_for_event, _eligible_runs_for_record), closing 27-REVIEW IN-02"
  - "TestApprovalStatusGate: 8 tests pinning the exclusion, the APPROVED/PENDING_REVIEW non-vacuous controls, the is_offered_candidate() server-side refusal, and confirmed-link survival"
  - "Runbook paragraph and a fifth, rejected demo run in campaign_lifecycle_demo.ipynb documenting/demonstrating the exclusion"
affects: [30-02, 30-03, 30-04]

# Actuals (#2632)
actuals:
  tokens: 11775
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "One module-level frozenset constant (_ATTRIBUTION_INELIGIBLE_APPROVAL_STATUSES), referenced through CampaignRun.ApprovalStatus, applied via .exclude(approval_status__in=...) at both eligibility gates -- never inlined per-gate, never a bare string"
    - "A fifth, deliberately-off-narrative CampaignRun in a pre-executed demo notebook, submitted through the same public form and never added to the notebook's shared submitted_runs dict, so pre-existing assertions over the four-run lifecycle stay untouched"

key-files:
  created: []
  modified:
    - solsys_code/campaign_attribution.py
    - solsys_code/tests/test_campaign_attribution.py
    - docs/runbooks/telescope_runs_calendar.rst
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb

key-decisions:
  - "Reused a single frozenset constant rather than two separate .exclude() literals, per D-03's anti-drift rationale -- one place to add a future disqualifying status"
  - "Gave rejected_run/approved_run/pending_review_run distinct window_end values (not distinct telescope_instrument) to satisfy CampaignRun's natural-key UniqueConstraint while keeping all three score identically well against the same orphan"
  - "Set target=run_target on the event-path fixture runs but a separate field_target on the record orphan, deliberately unequal, so the record-path tests cannot silently reintroduce the target-FK-equality behavior _eligible_runs_for_record's docstring prohibits"
  - "Notebook's fifth run reuses classical_run's exact telescope_instrument string but a different window_end, so it scores highly against orphan_event while its natural-key lookup still needs window_start/window_end to disambiguate from classical_run"

requirements-completed: [D-01, D-02, D-03, D-12]

coverage:
  - id: D1
    description: "A REJECTED CampaignRun is never returned by _eligible_runs_for_event or _eligible_runs_for_record, while APPROVED and PENDING_REVIEW runs in the identical position both still are"
    requirement: "D-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution.py#TestApprovalStatusGate.test_rejected_run_never_offered_for_event"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution.py#TestApprovalStatusGate.test_approved_run_still_offered_for_event"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution.py#TestApprovalStatusGate.test_rejected_run_never_offered_for_record"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution.py#TestApprovalStatusGate.test_approved_run_still_offered_for_record"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution.py#TestApprovalStatusGate.test_pending_review_run_still_offered_for_event"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution.py#TestApprovalStatusGate.test_pending_review_run_still_offered_for_record"
        status: pass
    human_judgment: false
  - id: D2
    description: "The filter is applied once, at the two eligibility gates, through one named constant, so candidates_for_event/candidates_for_record/the backlog builders/unattributable_orphan_count/is_offered_candidate/the high-band template hint all move together"
    requirement: "D-02, D-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution.py#TestApprovalStatusGate.test_is_offered_candidate_refuses_a_rejected_run"
        status: pass
      - kind: other
        ref: "grep -c approval_status__in=_ATTRIBUTION_INELIGIBLE_APPROVAL_STATUSES solsys_code/campaign_attribution.py"
        status: pass
    human_judgment: false
  - id: D3
    description: "An association already confirmed before its run was rejected stays intact"
    requirement: "D-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution.py#TestApprovalStatusGate.test_confirmed_attribution_survives_its_run_being_rejected"
        status: pass
    human_judgment: false
  - id: D4
    description: "The runbook's attribution section documents the exclusion, and the pre-executed demo notebook demonstrates it end-to-end with real executed output"
    requirement: "D-12"
    verification:
      - kind: other
        ref: "grep -c 'A rejected run is never offered as a match' docs/runbooks/telescope_runs_calendar.rst"
        status: pass
      - kind: manual_procedural
        ref: "jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb"
        status: pass
    human_judgment: false

duration: 8min
completed: 2026-08-31
status: complete
---

# Phase 30 Plan 01: Attribution REJECTED-run exclusion (tracer slice) Summary

**A REJECTED CampaignRun is excluded from `_eligible_runs_for_event`/`_eligible_runs_for_record` via one named `frozenset` constant, pinned by 8 new tests and demonstrated in the campaign-lifecycle demo notebook's real executed output.**

## Performance

- **Duration:** ~8 min (commit-to-commit; longer in wall-clock including read_first/discovery)
- **Started:** 2026-08-31T20:43:21-07:00 (first commit)
- **Completed:** 2026-08-31T20:49:35-07:00 (last commit)
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- Closed 27-REVIEW IN-02: a REJECTED `CampaignRun` is never offered as a suggested attribution match for an orphan `CalendarEvent` or `ObservationRecord`, at either eligibility gate, while `APPROVED` and `PENDING_REVIEW` runs in the identical position both stay offered (D-01)
- Fixed the D-02 gap the original audit missed: `_eligible_runs_for_record` had the identical missing filter as the event gate and now carries it too
- Proved the D-03 anti-drift property live: `is_offered_candidate()`'s server-side re-derivation refuses a POST naming a rejected run's pk, and `orphan_calendar_events()` never re-examines an already-confirmed link, so rejecting a run after confirmation cannot unlink it
- Extended both paired docs (CLAUDE.md rule): the runbook's attribution section now states the exclusion in plain English, and the pre-executed demo notebook shows a fifth, deliberately-off-narrative rejected submission scored as a candidate before rejection and absent after, with real executed output committed

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end "a rejected run is never offered" — the event path only** - `7c0a99a` (feat)
2. **Task 2: Expand to the record path, the PENDING_REVIEW control, and the confirmed-link survival** - `4c7668f` (test)
3. **Task 3: Paired docs — runbook attribution paragraph and the notebook's fifth, rejected submission** - `d56fefc` (docs)

**Plan metadata:** committed alongside this SUMMARY

## Files Created/Modified
- `solsys_code/campaign_attribution.py` - new `_ATTRIBUTION_INELIGIBLE_APPROVAL_STATUSES` constant; `.exclude(approval_status__in=...)` applied to both `_eligible_runs_for_event` and `_eligible_runs_for_record`; both docstrings extended citing 27-REVIEW IN-02 and D-01/D-02/D-03
- `solsys_code/tests/test_campaign_attribution.py` - new `TestApprovalStatusGate` class (8 tests): event/record exclusion + APPROVED control, PENDING_REVIEW control for both paths, `is_offered_candidate()` refusal, confirmed-link survival
- `docs/runbooks/telescope_runs_calendar.rst` - new bold-lead-in paragraph in the existing attribution section (`**A rejected run is never offered as a match.**`), extending it in place, no new heading
- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` - new markdown+code cell pair between the orphan-event cell and "Confirm the attribution": a fifth `rejected_demo_run`, scored candidate before rejection, absent after; public-table `visible_pks` assertion widened; Summary cell extended; regenerated in place with real executed output

## Decisions Made
- Reused one shared `frozenset` constant across both gates (D-03) rather than inlining the status literal twice, so a future disqualifying status only needs adding in one place
- Fixture runs (rejected/approved/pending_review) differ only on `window_end`, not `telescope_instrument`, to satisfy `CampaignRun`'s natural-key `UniqueConstraint` while all three score identically well against the same orphan — isolates the variable under test to `approval_status` alone
- Event-path fixture runs carry `target=run_target`; the record-orphan fixture uses a deliberately separate `field_target` in the same campaign, so the record-path tests cannot pass by accident via a reintroduced target-FK-equality check (the standing prohibition in `_eligible_runs_for_record`'s docstring)
- Notebook's fifth run reuses `classical_run`'s exact `telescope_instrument` text (so it scores highly against `orphan_event`) but a different `window_end`, requiring the lookup after creation to include `window_start`/`window_end` to disambiguate from `classical_run` (a bare `telescope_instrument` lookup would raise `MultipleObjectsReturned`)

## Deviations from Plan

None - plan executed exactly as written. One minor formatting adjustment: a single fixture-construction line in the new test class was collapsed to one line to match the pinned-ruff (v0.2.1, 120-column) formatting the project's `ruff format --check` expects — not a behavior change, caught and fixed before the Task 1 commit.

## Issues Encountered
- The dev environment's unpinned `ruff` (0.15.20, per 30-CONTEXT.md's documented drift) flagged one new line in the test file as reformattable even though the pinned pre-commit hook (v0.2.1) accepted it as written elsewhere; resolved by collapsing that one line to match the 120-column limit, which both versions agree on. No repo-wide reformat was run (D-05 held).
- The pre-commit `ruff-format` hook (running under the pinned `.pre-commit-config.yaml` toolchain, which also formats `.ipynb` files) auto-reformatted three lines in the notebook after the first commit attempt on Task 3 — pure line-joining, no content change. Re-staged and re-committed per the standard auto-fix-hook handling.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- The attribution eligibility fix (D-01/D-02/D-03) and its paired docs (D-12) are complete and independently verified (39 attribution-module tests, 230-test regression across the five downstream-consumer modules, sphinx build clean, notebook executed with zero errors)
- Plan 30-01 was the phase's tracer slice; plans 30-02/30-03/30-04 (WR-01 CSV re-import guard, ruff toolchain pin, Nyquist validation reconciliation, bookkeeping) can proceed independently — none of them touch `campaign_attribution.py` or its tests
- No blockers

## Self-Check: PASSED

- All 4 `files_modified` paths verified present on disk
- All 3 task commit hashes (`7c0a99a`, `4c7668f`, `d56fefc`) verified in `git log`
- `python manage.py test solsys_code.tests.test_campaign_attribution -v2` re-run: OK (39 tests)
- `grep -c approval_status__in=_ATTRIBUTION_INELIGIBLE_APPROVAL_STATUSES solsys_code/campaign_attribution.py` = 2
- `grep -c 'A rejected run is never offered as a match' docs/runbooks/telescope_runs_calendar.rst` = 1

---
*Phase: 30-v2-2-tech-debt-cleanup*
*Completed: 2026-08-31*
