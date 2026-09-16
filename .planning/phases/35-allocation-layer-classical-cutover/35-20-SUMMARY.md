---
phase: 35-allocation-layer-classical-cutover
plan: 20
subsystem: allocation-calendar
tags: [django, calendar, human-confirmation, transaction, cr-01, cr-03]

# Dependency graph
requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: allocation_projector.py's ALLOC: per-night projection, the sub-night window fields, the mint-provenance recording (plan 35-19), and the classical cutover
provides:
  - "_remint_decline_reason(run, existing): a reads-only predicate deciding whether an automated re-mint may destroy and re-create an ALLOC: night, closing CR-01 (35-REVIEW.md iteration 8)"
  - "The re-mint branch's compute-before-destroy reorder and its single transaction.atomic() wrap, closing CR-03 (35-REVIEW.md iteration 8)"
affects: [35-allocation-layer-classical-cutover verification/UAT, plan 35-21 (CR-02, depends on this plan), plan 35-22 (paired docs for the whole gap-closure round)]

# Actuals (#2632)
actuals:
  tokens: 6511
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Human-confirmation decline guard: a reads-only predicate (_remint_decline_reason) consulted BEFORE any destructive branch moves a counter, reusing the same shared partition helper (_clearable_declined_and_unattributed) every sibling delete/detach path in this module and campaign_reconciler.py already uses -- plus one branch-local addition for the destroy-and-immediately-recreate shape this branch alone has."
    - "Compute-before-destroy + scoped transaction.atomic(): the one movable failure point (_mint_fields()) is computed before the delete; the three unmovable write steps (delete, create, link, record-provenance) are wrapped in a transaction scoped to exactly that one night's pair, never wider -- preserving the existing per-run sweep-isolation contract (test_campaign_reconciler.py:1102)."

key-files:
  created: []
  modified:
    - solsys_code/allocation_projector.py
    - solsys_code/tests/test_allocation_projector.py

key-decisions:
  - "Followed <design_rationale> exactly rather than the review's literal fix snippet: the foreign-ownership arm is deliberately absent (the per-night loop's own _may_write() gate already routes a foreign-owned night to blocked before this branch is reached, so a third silent outcome would re-open D-16/NF-01); the guard adds a re-mint-local staff-state check (observation_record, observation_group, is_verified=False) on top of the shared _clearable_declined_and_unattributed() rule, since that shared rule alone declines only on confirmed_by and would leave probe 9's other two states open; and the branch DECLINES a protected night rather than preserving the companion row across the delete/create, matching the sibling delete/detach paths' rule and CalendarEventMeta's own 'written only by the observation projector' docstring contract."
  - "CR-03's two mechanisms (compute-before-destroy, transaction.atomic()) are implemented and tested SEPARATELY, per the plan's own instruction that a single test cannot distinguish which one saved the night -- TestRemintAtomicity has one test per mechanism."
  - "transaction.atomic() is scoped to exactly the delete/create/link/record-provenance group for one night, never wider (prohibition 7) -- test_campaign_reconciler's 63 tests (including the per-run sweep-isolation test at test_campaign_reconciler.py:1102) confirm a sweep over many runs still commits the runs it has already finished when a later one raises."
  - "The over-decline guard test required by Task 3 (an ordinary unconfirmed night still re-mints) is satisfied by test_unconfirmed_night_still_remints_normally, added in Task 1 -- not duplicated in Task 3, since it already asserts the boundary against a live sun_event() result for the same fixture shape Task 3 asks for."

requirements-completed: [ALLOC-01, ALLOC-03]

coverage:
  - id: D1
    description: "CR-01 closed: the re-mint branch is no longer the only delete path in allocation_projector.py with no human-confirmation guard. A confirmed night, and a night carrying observation_record/observation_group/is_verified=False staff-set state, all decline an automated re-mint instead of being destroyed -- reported under detach_declined with a warning naming the run, the event and the reason, and surviving the WHOLE reconcile_run() call (asserted on a fresh query after the call returns)."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestRemintHumanConfirmationGuard (9 tests: confirmed-night survival, unconfirmed control, 3 staff-state siblings, dry-run parity, unrecorded-provenance interaction)"
        status: pass
    human_judgment: false
  - id: D2
    description: "CR-03 closed on both halves: an inverted sub-night window edit raises before anything is destroyed (compute-before-destroy reorder), and a failure landing strictly between the delete and the create rolls the delete back (transaction.atomic() wrap scoped to one night's pair)."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestRemintAtomicity (2 tests: compute-before-destroy on an inverted span, rollback on a patched create failure)"
        status: pass
    human_judgment: false
  - id: D3
    description: "ALLOC-03 holds through a re-mint: a night carrying a real observation link or a human stamp keeps that state, and the ordinary (unconfirmed, unlinked) re-mint case is unchanged from plans 35-03/35-19."
    requirement: ALLOC-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestRemintHumanConfirmationGuard.test_unconfirmed_night_still_remints_normally, and the four sibling classes TestRetirePathLegacyEventGuard/TestFinalConvergenceGuard/TestObservationHandoff/TestClearedSubNightFieldRemints/TestUnrecordedProvenanceNight left unedited"
        status: pass
    human_judgment: false
  - id: D4
    description: "The full phase surface (five modules) is green at 221/221 tests (baseline 212 + 9 new), both ruff hooks pass, makemigrations reports no drift, and every forbidden path (docs/, management/, migrations/, models.py, admin.py, campaign_reconciler.py) is untouched."
    requirement: ALLOC-01
    verification:
      - kind: integration
        ref: "python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_cutover_classical_allocations solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_campaign_reconciler -- 221 tests, OK"
        status: pass
    human_judgment: false

duration: 65min
completed: 2026-09-16
status: complete
---

# Phase 35 Plan 20: Guard the Re-mint Branch Against Destroying Human-Confirmed State Summary

**A reads-only `_remint_decline_reason()` predicate, plus a compute-before-destroy reorder and a scoped `transaction.atomic()`, close CR-01 and CR-03 (35-REVIEW.md iteration 8) in `project_allocation()`'s re-mint branch without reopening any of the four prior gap-closure rounds' fixes.**

## Performance

- **Duration:** ~65 min
- **Started:** 2026-09-16T13:25:05Z (approx.)
- **Completed:** 2026-09-16T14:30Z (approx.)
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- **CR-01 closed**: `_remint_decline_reason(run, existing)` runs before any counter moves or the `dry_run` short-circuit. It reuses `campaign_reconciler._clearable_declined_and_unattributed()` unmodified (the same UAT-2026-09-09 "human outranks machine" rule the four sibling delete/detach paths already apply), plus a re-mint-local check for `observation_record`, `observation_group`, or `is_verified=False` -- the staff-set facts probe 9 showed the shared rule alone would miss. A declined night is counted under `detach_declined`, logs a warning naming the run/event/reason, and its url joins `active_urls` so the D-14 convergence step thirty lines later does not delete it moments after the guard refused to.
- **CR-03 closed on both halves**: `_mint_fields(run, night)` is now computed into `remint_fields` BEFORE `existing.delete()` -- the one movable failure point (an inverted-span `ValueError` from `night_bounds()`) now raises with nothing destroyed. The three unmovable write steps (`existing.delete()`, the create, `_link_event_to_run()`, `_record_sub_night_provenance()`) are wrapped in a single `transaction.atomic()` scoped to exactly this one night's pair, so a failure anywhere in that group rolls the delete back too.
- **9 new tests** across `TestRemintHumanConfirmationGuard` (7) and `TestRemintAtomicity` (2), covering every companion-row state the reviewer's probes 8 and 9 reproduced as destroyed, dry-run/real-run parity for a declined night, the 35-19 unrecorded-provenance branch's interaction with this guard, and both CR-03 mechanisms separately (a single test cannot tell which one saved the night).
- **No regressions**: the full five-module surface (`test_allocation_projector`, `test_allocation_projector_signals`, `test_cutover_classical_allocations`, `test_load_telescope_runs`, `test_campaign_reconciler`) is green at 221/221 (baseline 212 + 9 new). `TestRetirePathLegacyEventGuard`, `TestFinalConvergenceGuard`, `TestObservationHandoff`, `TestNoSunEventRecompute`, `TestSubNightWindow`, `TestClearedSubNightFieldRemints` and `TestUnrecordedProvenanceNight` all pass **unedited** (confirmed by diff: only new `def test_...` lines were added anywhere in the test file, no existing test body changed).

## RED Failure Output (confirmed-night test, before the fix)

The plan's Task 1 tracer required capturing the RED failure before the guard existed. Since this task is `type="tracer"` (not `tdd="true"`), the guard was written first; RED evidence was then captured by temporarily reverting `allocation_projector.py` to its pre-fix state (via the sanctioned `git checkout -- <file>` single-file revert, never a blanket reset) and re-running the new test, then reapplying the implementation.

First revert (test as originally written, with `assertLogs`):
```
AssertionError: no logs of level WARNING or higher triggered on solsys_code.allocation_projector
```
(No guard existed yet, so no warning was ever logged -- confirming the pre-fix branch had no decline path at all.)

Second pass (with the `assertLogs` wrapper temporarily removed to see the destruction itself):
```
File "solsys_code/tests/test_allocation_projector.py", line 1700, in test_confirmed_night_survives_a_would_be_remint
    self.assertEqual(result.detach_declined, 1)
AssertionError: 0 != 1
```
This is the same destruction the reviewer's probe 8 recorded: the confirmed night's `reconcile_run()` reported ordinary `retired=1/created=1` work (no `detach_declined`), with the old primary key gone and a fresh companion row (`confirmed_by=None`) replacing it. After reapplying the fix, both forms of the test pass (GREEN).

## Counter Triples Per Declined Companion-Row State

Every declined re-mint below reports the identical triple, on both a real run and its immediately preceding `dry_run=True` preview:

| Companion-row state | `detach_declined` | `retired` | `created` |
|---|---|---|---|
| `confirmed_by` set (probe 8) | 1 | 0 | 0 |
| `is_verified=False` (probe 9) | 1 | 0 | 0 |
| `observation_record` linked (probe 9) | 1 | 0 | 0 |
| `observation_group` linked (probe 9) | 1 | 0 | 0 |
| Confirmed + unrecorded provenance (35-19 interaction) | 1 | 0 | 0 |
| Dry-run preview of the `confirmed_by` case | 1 | 0 | 0 (matches the following real run) |
| **Control: unconfirmed, unlinked** | 0 | 1 | 1 |

## Probe 6 Before/After (CR-03, `TestRemintAtomicity`)

- **Before this fix** (35-REVIEW.md's own reproduction): `PROBE6 exists after = False` -- `existing.delete()` ran, then `_mint_fields()` -> `night_bounds()` -> `_raise_if_inverted()` raised, leaving the night permanently gone.
- **After this fix**: `test_compute_before_destroy_leaves_the_event_in_place_on_an_inverted_span` reproduces the exact fixture (mint valid, then set `night_start_utc=09:00`/`night_end_utc=23:00` -- inverted for this site) inside `assertRaises(ValueError)`, then asserts on a fresh query that the event's primary key, `start_time` and `end_time` are all unchanged from before the call. Passes.
- `test_a_failure_between_the_delete_and_the_create_rolls_back` patches `insert_or_create_calendar_event` to raise `RuntimeError` strictly between the delete and the create, and asserts the same untouched-event outcome via the `transaction.atomic()` rollback (a savepoint rollback inside Django `TestCase`'s own outer transaction). Passes.

## Full Five-Module Test Count

```
python manage.py test solsys_code.tests.test_allocation_projector \
  solsys_code.tests.test_allocation_projector_signals \
  solsys_code.tests.test_cutover_classical_allocations \
  solsys_code.tests.test_load_telescope_runs \
  solsys_code.tests.test_campaign_reconciler -v 1
```

**Ran 221 tests ... OK** (prior baseline: 212; this plan added 9: 2 in Task 1, 2 in Task 2, 5 in Task 3).

## Ruff Results

- `pre-commit run ruff --files solsys_code/allocation_projector.py solsys_code/tests/test_allocation_projector.py` -- **Passed**
- `pre-commit run ruff-format --files solsys_code/allocation_projector.py solsys_code/tests/test_allocation_projector.py` -- **Passed** (one auto-reformat applied after Task 2's edit, re-verified clean afterward)

## Migrations and Forbidden-Path Checks

- `python manage.py makemigrations --check --dry-run` -- **No changes detected.**
- `git status --short -- docs/ solsys_code/management/ solsys_code/migrations/ solsys_code/models.py solsys_code/admin.py solsys_code/campaign_reconciler.py` -- **empty** (nothing outside the plan's two `files_modified` paths was touched).

## Unedited Test Classes (Confirmed by Diff)

`git diff` from before this plan to `HEAD` over `test_allocation_projector.py` shows only new `def test_...` method additions (7 in `TestRemintHumanConfirmationGuard`, 2 in a new `TestRemintAtomicity` class) plus one new import (`ObservationGroup`). No line inside `TestRetirePathLegacyEventGuard`, `TestFinalConvergenceGuard`, `TestObservationHandoff`, `TestNoSunEventRecompute`, `TestSubNightWindow`, `TestClearedSubNightFieldRemints` or `TestUnrecordedProvenanceNight` was changed.

## Task Commits

Each task was committed atomically:

1. **Task 1: Make the re-mint branch ask whether it may destroy the night, before it destroys it** -- `5572181` (fix): `_remint_decline_reason()` added; re-mint branch guarded; `TestRemintHumanConfirmationGuard` with the confirmed-night test and the unconfirmed control test.
2. **Task 2: Compute before destroying, and make the destroy/re-create pair one unit** -- `f4dc2aa` (fix): `from django.db import transaction` added; `_mint_fields()` moved ahead of `existing.delete()`; the delete/create/link/record-provenance group wrapped in `transaction.atomic()`; `TestRemintAtomicity` with both CR-03 mechanism tests.
3. **Task 3: Pin the remaining declined companion-row states, the dry-run parity, and the whole phase surface** -- `e000ae6` (test): the three remaining probe-9 sibling tests, the dry-run parity test, the unrecorded-provenance interaction test; full five-module regression, both ruff hooks, migrations check and forbidden-path check all recorded above.

**Plan metadata:** this commit (docs: complete plan) -- see final commit below.

## Files Created/Modified

- `solsys_code/allocation_projector.py` -- `_remint_decline_reason()` (new, ~45-line docstring + body), the guarded re-mint branch preamble, `from django.db import transaction`, the `remint_fields` compute-before-destroy local, and the single `with transaction.atomic():` block wrapping the delete/create/link/record-provenance group.
- `solsys_code/tests/test_allocation_projector.py` -- `TestRemintHumanConfirmationGuard` (9 tests) and `TestRemintAtomicity` (2 tests); one new import (`ObservationGroup`).

## Decisions Made

See `key-decisions` in frontmatter. Summary: implemented `<design_rationale>`'s three amendments to the review's literal fix (no foreign arm, a re-mint-local staff-state addition on top of the shared rule, decline-not-preserve); implemented CR-03's two mechanisms separately with separate tests per the plan's own instruction; and satisfied Task 3's over-decline requirement with Task 1's existing control test rather than duplicating it.

## Deviations from Plan

None -- plan executed exactly as written. One minor side effect worth recording (not a deviation, since it did not require any fix): `test_staff_state_observation_record_link_declines_the_remint`'s fixture sets `observation_record` directly on the companion row without also creating a `CampaignRunObservation` link, so `_sync_observation_attribution()`'s post-loop cleanup step (which runs after the guard) treats the record as "no longer linked" and clears the row's `run`/`confirmed_by`/`confirmed_at` via `unlink_event_from_run()` -- but per `UNLINK_CLEARED_FIELDS`, `observation_record` itself is not one of the cleared fields, so the test's actual assertion (the link survives) still holds and passes. This is a fixture-realism note, not a defect: a real `observation_record` link is normally accompanied by a `CampaignRunObservation` row, which this synthetic fixture omits for simplicity.

## Issues Encountered

None.

## User Setup Required

None -- no schema change, no external service configuration required.

## Threat Flags

None -- this plan's threat model (in `35-20-PLAN.md`) is the authoritative register for the surface it touches; no new surface outside that register was introduced.

## Next Phase Readiness

- CR-01 and CR-03 (35-REVIEW.md iteration 8) are both closed. The re-mint branch now applies the same UAT-2026-09-09 Option B rule as the retired branch, the final convergence step and the legacy-takeover branch.
- ALLOC-03 holds through a re-mint: a night carrying a real link or a human stamp keeps that state. ALLOC-01's SC-1 is unchanged for the ordinary case.
- **Plan 35-21 (CR-02) can now proceed** -- its `depends_on: ["35-20"]` is satisfied: the re-mint branch this plan guards is the same branch 35-21's widened provenance-unrecorded funnel will point a larger volume of nights at.
- **The round is not complete.** Per this plan's own `<objective>` and `<artifacts>` sections, plan 35-22 (wave 3) still owes the paired-docs update: `docs/runbooks/telescope_runs_calendar.rst`'s `retired`/`detach_declined` documentation and `reconcile_campaign_runs_demo.ipynb`'s re-execution, against the combined state of this plan and 35-21. Until 35-22 runs, the runbook and the committed notebook describe a sweep that no longer exists exactly as written.
- No blockers for plan 35-21.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-16*

## Self-Check: PASSED
