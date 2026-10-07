---
phase: 35-allocation-layer-classical-cutover
plan: 04
subsystem: calendar-sync
tags: [django, allocation-projector, observation-projector, django-signals, calendar-events]

requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: "plan 35-01's ALLOC: namespace and project_allocation()'s retire/restore/attribution-bridge mechanics; plan 35-03's sub-night window fields"
provides:
  - "allocation_projector.receiver_on_run_observation_save() / receiver_on_run_observation_delete() -- post_save/post_delete receivers on CampaignRunObservation that re-project the linked run immediately"
  - "observation_projector.receiver_on_record_save()'s new linked-run re-project step, appended after the record's own event is projected"
  - "SolsysCodeConfig.ready()'s five-receiver wiring (three from Phase 34, two new from this plan), each with its own dispatch_uid"
affects: [35-05, 35-06, 35-07]

actuals:
  tokens: 11055
  tasks: 3
  commits: 3
  plan_head_before: 08996d24d2f9b12c21bb388c3e2732aed70ef861

tech-stack:
  added: []
  patterns:
    - "origin-based cascade guard: Django's post_delete signal carries an `origin` kwarg naming the model instance the enclosing `.delete()` call was originally made on, which stays the same across every signal a single Collector run fires -- including a CASCADE child's own post_delete. Checking `isinstance(kwargs.get('origin'), CampaignRun)` reliably distinguishes 'this CampaignRunObservation is being deleted as a side effect of its own run's deletion' from a standalone link delete, which a plain `CampaignRun.objects.filter(pk=...).exists()` check cannot do (see Deviations)."
    - "Second, separate try/except for an appended step: the record-side linked-run re-project step in receiver_on_record_save() gets its own try/except, deliberately not folded into the base projection's existing one, so a linked-allocation fault can never mask or discard the already-successful base projection."

key-files:
  created:
    - solsys_code/tests/test_allocation_projector_signals.py
  modified:
    - solsys_code/allocation_projector.py
    - solsys_code/apps.py
    - solsys_code/observation_projector.py
    - solsys_code/tests/test_observation_projector_signals.py
    - solsys_code/tests/test_allocation_projector.py
    - solsys_code/tests/test_campaign_reconciler.py
    - .planning/phases/35-allocation-layer-classical-cutover/35-VALIDATION.md

key-decisions:
  - "receiver_on_run_observation_delete() uses Django's post_delete `origin` kwarg, not a plain CampaignRun.objects.filter(pk=...).exists() check, to detect a CampaignRun delete cascade. Empirically verified (a throwaway probe script against this project's Django 5.2.17) that a CASCADE child's post_delete fires BEFORE the parent row's own DELETE statement runs in the same transaction, so the naive existence check would still find the run and re-mint fresh ALLOC: nights moments before the run itself disappears -- exactly the leak the plan's own behavior test 5 exists to catch. The plan's own draft assumed the run row was 'already gone by then'; that assumption does not hold in this Django version. See Deviations."
  - "Task 2's record-side reproject step tests moved to test_observation_projector_signals.py (a new TestLinkedRunReproject class), matching the plan's own <files> tag for that task, rather than staying in test_allocation_projector_signals.py where they were first drafted."
  - "Two 'no network call' tests use different patch targets on purpose: Task 3's own CampaignRunObservation-receiver test patches observation_projector.facility_for with a GEM-facility (non-LCO/SOAR) linked record, since facility_for is legitimately (and harmlessly -- no I/O) called by project_allocation()'s D-08 attribution bridge for any LCO/SOAR link; Task 2's record-side reproject test instead patches calendar_utils.make_request (the function that actually performs HTTP I/O), because that test's own linked record must be LCO/SOAR to reach the new step at all, which would make a facility_for-based probe always 'reached' regardless of whether a real network call occurred. See Deviations."

requirements-completed: [ALLOC-03]

coverage:
  - id: D1
    description: "Confirming an attribution (creating a CampaignRunObservation) retires the linked record's allocation night immediately, with no operator command; undoing it (deleting the link) restores the night and clears the record's own event attribution, with the event's title/description/start/end byte-identical"
    requirement: ALLOC-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector_signals.py#TestCampaignRunObservationSaveReceiver (3 tests)"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector_signals.py#TestCampaignRunObservationDeleteReceiver (3 tests)"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector_signals.py#TestCampaignRunObservationReceiverWiring (1 test)"
        status: pass
    human_judgment: false
  - id: D2
    description: "A record moving from queued to placed retires its allocation night on its own save (no sweep), through a second re-project step appended to receiver_on_record_save() after the base observation projection succeeds; a record with no campaign_run_links costs nothing; the step makes no network call and reaches sun_event() only when a night is minted or re-minted"
    requirement: ALLOC-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_observation_projector_signals.py#TestLinkedRunReproject (5 tests)"
        status: pass
    human_judgment: false
  - id: D3
    description: "The two CampaignRunObservation receivers never raise (logging only the exception type name, never its message), never reach a facility, and invoke project_allocation() a bounded, exact number of times per transition"
    requirement: ALLOC-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector_signals.py#TestAllocationTriggerContract (6 tests)"
        status: pass
    human_judgment: false

duration: 77min
completed: 2026-09-13
status: complete
---

# Phase 35 Plan 04: Immediate Allocation Handoff Triggers Summary

**Two new never-raise `CampaignRunObservation` receivers plus a second re-project step on the observation projector's own `post_save` receiver make an allocation night retire or restore the instant staff confirm or undo an attribution, or the scheduler places a queued request — no sweep needed.**

## Performance

- **Duration:** ~77 min
- **Started:** 2026-09-13T05:02:00Z (approx, from STATE.md's prior session marker)
- **Completed:** 2026-09-13T06:18:40Z
- **Tasks:** 3
- **Files modified:** 8 (1 created, 7 modified)

## Accomplishments

- `solsys_code/allocation_projector.py`: `receiver_on_run_observation_save()` and `receiver_on_run_observation_delete()`, both never-raise, no-network, `raw`-guarded (save only; `post_delete` carries no `raw`), reaching `sun_event()` only through `project_allocation()`'s own mint/re-mint branch.
- `receiver_on_run_observation_delete()` correctly no-ops during a `CampaignRun` delete cascade using Django's `post_delete` `origin` kwarg — empirically verified this Django version fires a CASCADE child's `post_delete` *before* the parent row's own DELETE, so a plain existence check would have re-minted fresh nights moments before the run itself vanished.
- `solsys_code/apps.py`: `SolsysCodeConfig.ready()` now wires five receivers total; the two new ones each carry their own `dispatch_uid` (`solsys_code.allocation_projector.campaign_run_observation.post_save`/`.post_delete`); no receiver added on `CampaignRun` itself.
- `solsys_code/observation_projector.py`: `receiver_on_record_save()` gains a second, separately-guarded step — after the record's own event projects successfully, every linked `CampaignRun` is re-projected via `allocation_projector.project_allocation()` (one function-local import, no module-level cycle).
- `solsys_code/tests/test_allocation_projector_signals.py` (new, 13 tests): the six Task 1 link/unlink behaviors plus the six-test `TestAllocationTriggerContract` class (never-raise ×4, never-call-out ×1, never-recurse ×1).
- `solsys_code/tests/test_observation_projector_signals.py`: new `TestLinkedRunReproject` class (5 tests) covering the record-side step's five behaviors.
- Two pre-existing tests (`test_allocation_projector.py`, `test_campaign_reconciler.py`) corrected for the new immediacy behavior — see Deviations.
- `.planning/phases/35-allocation-layer-classical-cutover/35-VALIDATION.md`: 35-04 row flipped to green; Wave 0 checkbox ticked for the new test file.
- Full label-list regression suite (`workflow.test_command`) re-run twice during this plan (once after each deviation fix): 1177 tests, green (1 pre-existing skip).

## Task Commits

Each task was committed atomically:

1. **Task 1: Link and unlink receivers on CampaignRunObservation** - `96d3c94` (feat)
2. **Task 2: A record that moves from queued to placed retires its night without a sweep** - `aad61e4` (feat)
3. **Task 3: Trigger contract regression — never raise, never call out, never recurse** - `bbd4267` (test)

**Plan metadata:** committed alongside this SUMMARY.

## Files Created/Modified

- `solsys_code/allocation_projector.py` - two new receivers (Task 1)
- `solsys_code/apps.py` - five-receiver wiring (Task 1)
- `solsys_code/observation_projector.py` - linked-run re-project step (Task 2)
- `solsys_code/tests/test_allocation_projector_signals.py` - new module, 13 tests (Tasks 1 & 3)
- `solsys_code/tests/test_observation_projector_signals.py` - new `TestLinkedRunReproject` class, 5 tests (Task 2)
- `solsys_code/tests/test_allocation_projector.py` - one assertion corrected for the new immediacy behavior (deviation, found during Task 3's full-suite verification)
- `solsys_code/tests/test_campaign_reconciler.py` - one assertion corrected for the same reason (deviation, found during Task 3's full-suite verification)
- `.planning/phases/35-allocation-layer-classical-cutover/35-VALIDATION.md` - 35-04 row and Wave 0 checkbox updated (Task 3)

## Decisions Made

See `key-decisions` in the frontmatter: the `origin`-based cascade guard for `receiver_on_run_observation_delete()`; Task 2's tests placed in `test_observation_projector_signals.py` per the plan's own file assignment; and the two different "no network call" patch targets (`facility_for` with a GEM record for Task 3's receiver-level test, `calendar_utils.make_request` for Task 2's record-side test).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `receiver_on_run_observation_delete()`'s CASCADE-safety mechanism corrected — the plan's own assumption about Django delete ordering was wrong**
- **Found during:** Task 1, before writing behavior test 5 ("deleting a CampaignRun cascades its CampaignRunObservation rows without re-projecting")
- **Issue:** The plan's own action text asserted "the run lookup is what makes this safe during a CampaignRun delete cascade: the run row is already gone by then, the lookup returns None." A throwaway probe script (Django `TestCase`-style setup, a `post_delete` receiver printing `CampaignRun.objects.filter(pk=run_pk).exists()`) run against this project's actual Django version (5.2.17) showed this is false: `django.db.models.deletion.Collector.delete()` deletes a CASCADE child's row and fires its `post_delete` signal *before* the parent row's own DELETE statement runs, in the same transaction. A plain existence check would therefore still find the run and call `project_allocation(run)` — re-minting fresh `ALLOC:` nights (the run's own `pre_delete` receiver in `models.py` had already cleared them) moments before the run itself disappeared, leaking orphaned events.
- **Fix:** `receiver_on_run_observation_delete()` first checks `isinstance(kwargs.get('origin'), CampaignRun)` — Django's `post_delete` signal carries an `origin` kwarg naming the model instance the enclosing `.delete()` call was originally made on, the same value for every signal a single `Collector` run fires, including a cascaded child's. When `origin` is the `CampaignRun` itself (not this link), the delete is a cascade side effect and the receiver returns without projecting anything. The plain `CampaignRun.objects.filter(pk=...).exists()` lookup is kept as a secondary, defensive check for a standalone link delete.
- **Files modified:** `solsys_code/allocation_projector.py`
- **Verification:** `test_deleting_the_run_cascades_the_link_without_raising_or_re_projecting` passes with a mocked `project_allocation` asserted never called during `self.run.delete()`.
- **Committed in:** `96d3c94` (Task 1 commit)

**2. [Rule 1 - Bug] Two pre-existing tests corrected for the new immediate-retirement behavior**
- **Found during:** Task 3, running the plan's own full affected-module and full-label-list regression commands
- **Issue:** `test_allocation_projector.TestObservationHandoff.test_unlinking_restores_the_retired_night_with_a_fresh_event` and `test_campaign_reconciler.TestReconcileThenAttributeOrdering.test_second_reconcile_deletes_the_superseded_allocation_night_and_restore_on_third` each called an explicit `reconcile_run()` immediately after linking/unlinking a `CampaignRunObservation`, and asserted `created`/`retired` counts that assumed no automatic re-projection had happened yet — true before this plan's receivers existed. With Task 1's new `post_save`/`post_delete` receivers wired, that automatic re-project already runs (inside `link.delete()`/`CampaignRunObservation.objects.create()` itself) before the test's own explicit `reconcile_run()` call, so the explicit call now converges on already-current state and correctly reports `unchanged` where the tests expected `created`.
- **Fix:** Updated both tests' counter assertions to `unchanged` (with an inline comment explaining why), leaving every substantive assertion (the event exists / is retired, with the right url) unchanged.
- **Files modified:** `solsys_code/tests/test_allocation_projector.py`, `solsys_code/tests/test_campaign_reconciler.py`
- **Verification:** Both fixed tests pass individually and as part of their full modules; the complete `workflow.test_command` label-list suite (1177 tests) re-run green after the fix.
- **Committed in:** `bbd4267` (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs — one a genuine correctness gap the plan's own draft would have shipped, one a mechanical consequence of shipping the new immediate triggers). **Impact:** Both fixes are necessary for correctness; no scope creep — deviation 1 closes exactly the leak Task 1's own behavior test 5 was written to catch, and deviation 2 is test-only, correcting counter expectations to match the intended new behavior.

## Issues Encountered

None blocking. The plan's own acceptance criteria and `<verify>` commands all pass as committed; both deviations above were caught and fixed by the plan's own verification gates (Task 1's behavior test 5, Task 3's full-suite regression run) before this SUMMARY was written.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- ALLOC-03's "linking removes the night, unlinking restores it" now holds without an operator command, at every trigger D-11 names: staff confirmation/undo (`CampaignRunObservation` create/delete) and the scheduler placing a queued request (the record's own save).
- `SolsysCodeConfig.ready()` now wires five receivers, all with unique `dispatch_uid`s, none on `CampaignRun` itself.
- Full label-list regression suite green (1177 tests, 1 pre-existing skip) — no blockers for 35-05/35-06/35-07.
- Paired docs for this phase remain owned by plan 35-07 (wave 5), per this plan's own scope note — no paired-doc follow-up is owed from this plan.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-13*

## Self-Check: PASSED

- FOUND: solsys_code/allocation_projector.py
- FOUND: solsys_code/apps.py
- FOUND: solsys_code/observation_projector.py
- FOUND: solsys_code/tests/test_allocation_projector_signals.py
- FOUND: solsys_code/tests/test_observation_projector_signals.py
- FOUND commit: 96d3c94
- FOUND commit: aad61e4
- FOUND commit: bbd4267
- All plan-level `<acceptance_criteria>` and `<verify>` commands re-run and passing (see task-by-task output above)
- Full label-list regression suite (`workflow.test_command`): 1177 tests, green
