---
phase: 35-allocation-layer-classical-cutover
plan: 01
subsystem: calendar-sync
tags: [django, campaign-reconciler, allocation-projector, calendar-events, tdd]

requires:
  - phase: 33-series-identity-reconciler-inversion
    provides: "the human-outranks-machine attribution guard (_stale_attributions()), campaign_utils.adopt_event_into_run()/unlink_event_from_run(), the noon-anchored _observing_night() this plan promotes"
  - phase: 34-the-observation-projector-trigger
    provides: "the observation projector's own event namespace (observation_projector.PROJECTED_FACILITIES/facility_for()/event_url()) this plan's attribution bridge reads, and CalendarEventMeta.observation_record"
provides:
  - "solsys_code/allocation_projector.py -- the ALLOC:{run_pk}:{night} namespace, dispatched to from reconcile_run() for every resolved-site, non-queue-sourced CampaignRun"
  - "telescope_runs.observing_night() -- the shared, promoted site-local night anchor"
  - "campaign_reconciler.split_telescope_instrument() (public) and two new ReconcileResult counters (retired, rekeyed)"
  - "The full D-08 observation-handoff bridge: a linked placed/observed record retires its night and self-attributes; unlinking restores the night and clears the attribution unless a staff member confirmed it"
affects: [35-02, 35-03, 35-04, 35-05, 35-06, 35-07]

actuals:
  tokens: 20560
  tasks: 3
  commits: 3
  plan_head_before: af884f1df3813fc90c1e366f8d38585681dfe2aa

tech-stack:
  added: []
  patterns:
    - "Peer-module dispatch seam: reconcile_run() stays the single entry point; a new namespace gets a new module with a function-local import back into the dispatcher, avoiding an import cycle (mirrors _detach_stale_family_events()'s existing campaign_utils idiom)."
    - "sun_event()-on-mint-only: compute expensive solar-crossing math only when a night is created or re-minted, never on an idempotent update -- closes a named tech-debt todo at the point the retired code was replaced, rather than patching the old branch."
    - "Convergence-by-exclusion: a per-run 'active set' built during the per-night loop, then a single trailing delete of anything not in it -- used for both the D-14 re-classification cleanup and (via a second local set) protecting an already-explicitly-handled retired night from being double-counted by that same convergence step in a dry-run preview."

key-files:
  created:
    - solsys_code/allocation_projector.py
    - solsys_code/tests/test_allocation_projector.py
  modified:
    - solsys_code/telescope_runs.py
    - solsys_code/campaign_reconciler.py
    - solsys_code/management/commands/reconcile_campaign_runs.py
    - solsys_code/models.py
    - solsys_code/tests/test_campaign_reconciler.py
    - .planning/phases/35-allocation-layer-classical-cutover/35-VALIDATION.md

key-decisions:
  - "retired_nights() unconditionally counts a night in the retired set as +1 in ReconcileResult.retired, even on the very first reconcile of a run when nothing existed yet to delete -- the plan's own Task 2 Test 1 requires retired == 1 for exactly that case, so 'retired' means 'this night is retired', not 'a delete had a nonzero rowcount'."
  - "Convergence excludes a locally-tracked retired_urls set (in addition to the returned active_urls) so a dry-run preview never double-counts a night the per-night loop already reported as retired -- real (non-dry) mode never hits this because the event is already gone from the DB by the time convergence runs, so the guard is dry-run-only insurance."
  - "The test suite's record-derived events call observation_projector.write_event_meta() (not a bare CalendarEvent.objects.create()) so CalendarEventMeta.observation_record is actually populated -- without it the D-08 unlink half's observation_record__isnull=False filter silently never matches, which is exactly the bug the first attribution-bridge test run caught (see Deviations)."

requirements-completed: [ALLOC-01, ALLOC-02, ALLOC-03]

coverage:
  - id: D1
    description: "Tracer slice: a campaign-less, resolved-site CampaignRun projects one ALLOC:{pk}:{night} event per window night through reconcile_run(); a queue-sourced run keeps its single RUN:{pk} container; the retired per-night RUN: branch is gone"
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestEndToEndAllocationNight (6 tests)"
        status: pass
    human_judgment: false
  - id: D2
    description: "The observation handoff: a linked placed/observed record retires its night and self-attributes the record's own event via adopt_event_into_run(); unlinking restores the night and clears the attribution via unlink_event_from_run() unless a staff member confirmed it; a legacy RUN:{pk}:{night} event is re-keyed in place"
    requirement: ALLOC-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestObservationHandoff (6 tests)"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestAttributionBridge (3 tests)"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestAllocationDeletionCascade (2 tests)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Nights are keyed by the site-local, noon-anchored observing night for both a Chilean and an Australian site, including the UTC-date-differs edge and the exact-local-noon boundary; sun_event() is never recomputed on an idempotent re-reconcile"
    requirement: ALLOC-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestAllocationNightBoundary (8 tests)"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestNoSunEventRecompute (2 tests)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Degenerate CampaignRun states (null window, inverted window, no observation links) are handled without error; 35-VALIDATION.md's evidence columns reflect the shipped module"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestEmptyAndDegenerateWindows (3 tests)"
        status: pass
      - kind: other
        ref: "python -c ... 35-VALIDATION.md row check -> '3 0'"
        status: pass
    human_judgment: false
  - id: D5
    description: "The full label-list regression suite stays green after the RUN:{pk}:{date} per-night branch is removed (plan's own backstop truth)"
    verification:
      - kind: integration
        ref: "workflow.test_command label-list run"
        status: fail
    human_judgment: true
    rationale: "Explicitly deferred by the plan itself: Task 3's own action text says test modules other than test_allocation_projector.py are owned by plan 35-02, and any failure there is expected pre-migration fallout to record, not fix, here. All 29 failures + 18 errors are confined to test_campaign_reconciler.py, test_reconcile_campaign_runs.py, and test_campaign_approval.py's run_night_url import (plus test_campaign_site_search.py's cascading import of it) -- every one traces directly to the retired RUN:{pk}:{date} per-night branch this plan intentionally removed. A human/plan-35-02 must confirm this is the full and only fallout before the phase can call the backstop truth satisfied."

duration: 53min
completed: 2026-09-13
status: complete
---

# Phase 35 Plan 01: Tracer Slice, Observation Handoff & Site-Local Boundaries Summary

**New `solsys_code/allocation_projector.py` owns a fresh `ALLOC:{run_pk}:{night}` calendar namespace that `campaign_reconciler.reconcile_run()` now dispatches every resolved-site, non-queue `CampaignRun` to, complete with the full observation-handoff (retire/restore/attribute) mechanics and noon-anchored site-local night keying for both hemispheres.**

## Performance

- **Duration:** ~53 min
- **Started:** 2026-09-13T02:46:00Z (approx, from STATE.md's last recorded activity)
- **Completed:** 2026-09-13T03:39:03Z
- **Tasks:** 3 (Task 1 tracer, Task 2 auto/tdd, Task 3 auto)
- **Files modified:** 8 (2 created, 6 modified)

## Accomplishments

- `solsys_code/allocation_projector.py`: a new peer module owning the `ALLOC:` key namespace, with `allocation_night_url()`, `allocation_events()`, `writable_allocation_events()`, `allocation_night_title()`, `allocation_night_description()`, `preserved_dark_window_line()`, `retired_nights()`, and `project_allocation()` as its public surface.
- `telescope_runs.observing_night()`: the site-local, noon-anchored night anchor promoted from `campaign_reconciler`'s private copy to a shared public helper, used by both the old reconciler's `_attributed_nights()` and the new module's `retired_nights()`.
- `campaign_reconciler.reconcile_run()`: a new fourth dispatch branch routes `LCO_QUEUE`/`SOAR_QUEUE`/`GEMINI_QUEUE`/`ESO_QUEUE` runs to the existing whole-window container regardless of site (D-10); every other resolved-site, windowed run now dispatches to `project_allocation()` (D-09). `run_night_url()`, `_observing_night()` and the entire `_reconcile_classical_nights()` per-night branch are deleted. `_split_telescope_instrument()` is renamed to the public `split_telescope_instrument()`.
- The full D-08 observation handoff: a `CampaignRunObservation` link whose record has both `scheduled_start`/`scheduled_end` set retires exactly the site-local night that block starts in (deleting the `ALLOC:` event and any legacy `RUN:{pk}:{night}` twin); deleting the link and re-reconciling re-mints the night; the record's own event is attributed to the run via `adopt_event_into_run()` and un-attributed via `unlink_event_from_run()` when its link disappears, unless a staff member's `confirmed_by` outranks the automated clear.
- A legacy `RUN:{pk}:{night}` event for a night the run still owns is re-keyed in place into `ALLOC:{pk}:{night}` (same primary key, `start_time`/`end_time` untouched, no `sun_event()` recompute) rather than duplicated.
- D-14 convergence deletes any `ALLOC:` event left over from a re-classified (shrunk) window.
- `writable_allocation_events()` closes the run-deletion leak: `CampaignRun`'s `pre_delete` cascade now also clears a deleted run's own allocation nights, never one attributed to a different run.
- Site-local night boundaries pinned for both a Chilean (`America/Santiago`, UTC-4) and Australian (`Australia/Sydney`, UTC+10) fixture site: a UTC-date-differs case, the exact-local-noon boundary, one second before it, and a post-local-midnight start -- 8 tests, each asserting the exact surviving/retired url.
- The folded todo (`2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md`) is closed with a non-vacuous regression: an idempotent re-reconcile of an unchanged 5-night run makes zero `sun_event()` calls, and a night whose event was deleted out from under the run calls it exactly twice (sun + dark) on the next reconcile.
- Degenerate `CampaignRun` states (null window, inverted window, zero observation links) are covered without error.
- `34` tests total in `test_allocation_projector.py`, all green.

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end ALLOC: night — one allocation run, one path through every layer** - `8f732d4` (feat)
2. **Task 2: The handoff — a linked observation retires its night, unlinking restores it** - `9a6cb32` (feat)
3. **Task 3: Site-local night boundaries, the no-recompute regression, and a green suite** - `f5c83b1` (test)

_Tasks 1 and 2 were marked `tdd="true"`; see **TDD Gate Compliance** below for the actual commit shape versus the plan's RED/GREEN contract._

## Files Created/Modified

- `solsys_code/allocation_projector.py` - new module: the `ALLOC:` namespace, retirement, legacy takeover, attribution bridge, D-14 convergence
- `solsys_code/tests/test_allocation_projector.py` - new test module: 34 tests across 9 classes
- `solsys_code/telescope_runs.py` - new public `observing_night()`
- `solsys_code/campaign_reconciler.py` - dispatch rewrite (D-09/D-10), `run_night_url()`/`_observing_night()`/`_reconcile_classical_nights()` deleted, `split_telescope_instrument()` made public, `ReconcileResult` gains `retired`/`rekeyed`
- `solsys_code/management/commands/reconcile_campaign_runs.py` - sweep surfaces the two new counters in both dry-run and real summary lines, plus a per-run retired/rekeyed stdout line
- `solsys_code/models.py` - `CampaignRun`'s `pre_delete` receiver also cascades `writable_allocation_events()`
- `solsys_code/tests/test_campaign_reconciler.py` - one-line import update (`split_telescope_instrument as _split_telescope_instrument`) so its own private-name test class keeps working unchanged; body untouched (plan 35-02's scope)
- `.planning/phases/35-allocation-layer-classical-cutover/35-VALIDATION.md` - flipped the `35-01-0{1,2,3}` rows and the two Wave 0 requirements this plan closes to their green evidence state

## Decisions Made

- `retired_nights()` counts every night in the retired set as +1 in `ReconcileResult.retired`, even when nothing existed yet to delete (first-ever reconcile of a run whose linked record was already placed before the run was ever reconciled) — this is what the plan's own Task 2 Test 1 requires, and "retired" reads as "this night is retired" rather than "a delete had a nonzero rowcount".
- Convergence tracks a second, un-returned `retired_urls` set purely to protect a dry-run preview from double-counting a night the per-night loop already reported as retired (real mode never needs this guard, since the event is already gone from the DB by the time convergence runs).
- Test fixtures for the D-08 attribution bridge call `observation_projector.write_event_meta()` (not a bare `CalendarEvent.objects.create()`) so `CalendarEventMeta.observation_record` is actually populated — see Deviations for the bug this caught.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug, caught by the plan's own test-first discipline] Test fixture omitted `write_event_meta()`, making the unlink half's `observation_record__isnull=False` filter unreachable**
- **Found during:** Task 2, first run of `test_d08_round_trip_link_and_unlink_attribution`
- **Issue:** The test built the record's own observation-projector-style event with a bare `CalendarEvent.objects.create()`. In production, `observation_projector.project_record()` always follows a create with `write_event_meta(event, record)`, which sets `CalendarEventMeta.observation_record`. Skipping that step left `observation_record` unset on the companion row, so the real `_sync_observation_attribution()` unlink-half query (`observation_record__isnull=False`) never matched it — the test failed with the attribution still present after unlink, exposing that the fixture, not the production code, was wrong.
- **Fix:** `_make_record_event()` test helper now calls `op.write_event_meta(event, record)` immediately after creating the event, matching what `project_record()` does in production. Two other tests in the same class that used to `.create()` a second `CalendarEventMeta` row for the same one-to-one `event` primary key (which would have raised an `IntegrityError` once the helper started pre-creating that row) were updated to fetch-and-update the existing row instead.
- **Files modified:** `solsys_code/tests/test_allocation_projector.py`
- **Verification:** All 3 `TestAttributionBridge` tests pass; full module re-run green.
- **Committed in:** `9a6cb32` (Task 2 commit)

**2. [Rule 1 - Bug] Own module docstring tripped the plan's own "no direct attribution write" grep**
- **Found during:** Task 2, running the plan's own acceptance-criteria verify command
- **Issue:** The module docstring's prose contained the literal substring `` `meta.run = ...` `` as an example of what NOT to do — the acceptance check `grep`s the whole file (not just code) for `meta.run =`, so the docstring itself tripped its own anti-pattern gate.
- **Fix:** Reworded the sentence to describe the same constraint without the literal assignment substring.
- **Files modified:** `solsys_code/allocation_projector.py`
- **Verification:** `python -c "... print(body.count('meta.run =') + ...)"` now prints `0`.
- **Committed in:** `9a6cb32` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs, both caught before commit by the plan's own verification gates). **Impact:** Both fixes are test/documentation-only; no production behavior changed as a result of either.

## TDD Gate Compliance

Tasks 1 and 2 carry `tdd="true"`. `workflow.tdd_mode` is `false` in this project's config, so the runtime MVP+TDD halt gate did not apply, but the tdd.md commit-scope contract (`test({phase}-{plan})` RED commit, failing for the named reason, followed by a `feat({phase}-{plan})` GREEN commit) still describes the intended shape.

**Actual shape, both tasks:** implementation and its tests were written and iterated together, verified fully green (all target tests passing, verify commands matching expected output) before a single `feat({phase}-{plan})` commit per task — no separate `test({phase}-{plan})` RED commit exists for either task, and no RED-evidence record was captured via `gsd_run check tdd-red-evidence`.

**Why this is disclosed rather than silently accepted:** the RED phase's value is proving the test fails for the *intended* reason before the code exists — for Task 1 this was structurally true in practice (the target module and its imports genuinely did not exist until the implementation was written, so any test run against it would have failed on import), but that failure was never captured as a persisted, verified RED-evidence record, and the two are not committed separately. Task 2's real bug (see Deviations #1) was in fact caught by running the new test against the new code and observing a genuine failure — an informal RED/GREEN cycle occurred, just not one bounded by a commit boundary.

**Disposition:** no code or test defect results from this — every acceptance criterion and `<verify>` command for both tasks passes as committed, and Task 2's real defect was caught and fixed before commit regardless of the missing commit-boundary discipline. Flagged here per the tdd.md gate-enforcement contract rather than omitted.

## Issues Encountered

None blocking. See "Deviations from Plan" for the two auto-fixed issues, both caught by the plan's own gates before commit.

**Expected pre-migration fallout (not fixed here, per Task 3's own action text):** the full label-list regression suite (`workflow.test_command`) reports 29 failures + 18 errors after this plan's `RUN:{pk}:{date}` removal, entirely confined to `solsys_code/tests/test_campaign_reconciler.py` (asserting on the retired per-night `RUN:` branch and the pre-D-10 queue-source-with-resolved-site behavior), `solsys_code/tests/test_reconcile_campaign_runs.py` (same), and `solsys_code/tests/test_campaign_approval.py`'s module-level `from solsys_code.campaign_reconciler import run_night_url` (now deleted), which also cascades into `test_campaign_site_search.py`'s import of that module. This is exactly the fallout the plan's own frontmatter backstop truth and Task 3's action text name as owned by plan 35-02, not this one. `test_allocation_projector.py` (34 tests), the two `test_views` label-list tests, and both `pre-commit run ruff`/`ruff-format` gates are all green.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The `ALLOC:` namespace, its dispatch seam, and the full observation-handoff mechanics are in place and tested for both hemispheres — plan 35-02 can now rewrite `test_campaign_reconciler.py`'s body (and `test_reconcile_campaign_runs.py`, `test_campaign_approval.py`) onto the new dispatch without inheriting stale `RUN:{pk}:{date}` assertions.
- `campaign_reconciler.run_container_url()`, `_may_write()`, `_link_event_to_run()` and `RUN_STATUS_CALENDAR_PREFIX` are the stable seam the new module and future 35-0x plans should keep importing from, per this plan's own import-discipline paragraph in `allocation_projector.py`'s module docstring.
- Blocker for phase close: the full label-list suite must go green again once plan 35-02 lands — tracked as this plan's own backstop truth, deliberately left `fail`/human-judgment in this SUMMARY's coverage block rather than silently marked passing.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-13*

## Self-Check: PASSED

- FOUND: solsys_code/allocation_projector.py
- FOUND: solsys_code/tests/test_allocation_projector.py
- FOUND commit: 8f732d4
- FOUND commit: 9a6cb32
- FOUND commit: f5c83b1
- All plan-level `<acceptance_criteria>` and `<verify>` commands re-run and passing (see task-by-task output above)
