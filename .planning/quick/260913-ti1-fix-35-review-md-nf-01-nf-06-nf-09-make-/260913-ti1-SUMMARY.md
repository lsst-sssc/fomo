---
phase: quick-260913-ti1
plan: 01
subsystem: campaign-calendar-reconciler
tags: [django, calendar-reconciliation, ownership-predicate, test-coverage, docs]

requires:
  - phase: 35 (Allocation Layer & Classical Cutover)
    provides: the ALLOC:/RUN: two-namespace reconciler and allocation projector this fix corrects
provides:
  - "_clearable_declined_and_unattributed() -- one shared total-partition helper used at all four delete/detach call sites"
  - "_may_write() widened to police both the RUN: and ALLOC: namespaces"
  - "regression coverage for every unattributed companion-row shape at both namespaces, plus a predicate-agreement test"
  - "runbook/notebook counter definitions corrected to match the fixed behavior"
affects: [35-REVIEW.md follow-up work, any future reconciler/allocation-projector delete-path change]

actuals:
  tokens: 11107
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Total-partition helper: every stale-event candidate must land in exactly one of deletable/declined/left-alone -- a candidate landing in none is the bug class this fix closes (D-16's 'no third outcome' contract)."
    - "Row-level predicate and its queryset twins must be kept in lockstep by construction (or tested for agreement directly), not just have each engineered to look right independently."

key-files:
  created: []
  modified:
    - solsys_code/campaign_reconciler.py
    - solsys_code/allocation_projector.py
    - solsys_code/tests/test_campaign_reconciler.py
    - solsys_code/tests/test_allocation_projector.py
    - docs/runbooks/telescope_runs_calendar.rst
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb

key-decisions:
  - "One shared helper (_clearable_declined_and_unattributed) at all four call sites, not four local patches -- matches the plan's stated root-cause fix and prevents a fifth iteration of the same class of gap."
  - "_may_write()'s fallback now checks both the RUN: container/per-night prefix and the ALLOC:{run.pk}: prefix in one return, via a function-local import of ALLOC_URL_NAMESPACE (avoids the campaign_reconciler <-> allocation_projector import cycle, same idiom already used elsewhere in the file)."
  - "NF-09: legacy_urls_claimed.add() now fires the moment the CR-03 retire branch has decided a legacy event's fate AT ALL (deleted, blocked, or declined), not only on the deletable path -- closes the blocked-AND-detach_declined double count for one event, one decision."

patterns-established:
  - "Total-partition delete/detach helper: split candidates into (this-run-unconfirmed OR unattributed-unconfirmed) = deletable, (this-run-confirmed OR unattributed-confirmed) = declined, everything else = left alone. Reusable shape for any future ownership-gated cleanup path in this codebase."

requirements-completed: [NF-01, NF-06, NF-09, NF-07]

coverage:
  - id: D1
    description: "A stale ALLOC:{pk}:{night} event with no companion row (shape a) or a companion row with run IS NULL (shape b) is deleted by project_allocation()'s convergence step and reported under exactly one counter (retired)"
    requirement: "NF-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestFinalConvergenceGuard.test_window_shrink_deletes_an_unattributed_night_shape_b"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestFinalConvergenceGuard.test_window_shrink_deletes_an_unattributed_night_shape_a"
        status: pass
    human_judgment: false
  - id: D2
    description: "A leftover RUN:{pk}:{date} event of shape (a) or (b) under a container-dispatched run is deleted and reported under legacy_deleted"
    requirement: "NF-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestLegacyPerNightFamilyDeletion.test_orphan_legacy_event_with_no_companion_row_is_still_deleted"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestLegacyPerNightFamilyDeletion.test_orphan_legacy_event_with_unset_run_companion_row_is_still_deleted"
        status: pass
    human_judgment: false
  - id: D3
    description: "A leftover ALLOC:{pk}:{night} event of shape (b) belonging to a run re-classified INTO container dispatch is deleted by _stale_allocation_events() (CR-02's call site)"
    requirement: "NF-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestReclassificationConvergence.test_reclassifying_allocation_dispatch_to_class_wide_deletes_unattributed_nights_shape_b"
        status: pass
    human_judgment: false
  - id: D4
    description: "An unattributed legacy RUN:{pk}:{night} event on a retired night is deleted by project_allocation()'s CR-03 retire branch instead of being left untouched and uncounted"
    requirement: "NF-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestRetirePathLegacyEventGuard.test_retiring_a_night_deletes_an_unattributed_legacy_event_shape_a"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestRetirePathLegacyEventGuard.test_retiring_a_night_deletes_an_unattributed_legacy_event_shape_b"
        status: pass
    human_judgment: false
  - id: D5
    description: "_may_write() and its queryset twins (writable_events(), writable_allocation_events()) agree for every companion-row shape at both namespaces"
    requirement: "NF-06"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestOwnershipScoping.test_may_write_agrees_with_both_queryset_twins_for_every_shape"
        status: pass
    human_judgment: false
  - id: D6
    description: "A human-confirmed legacy event on a retired night increments exactly one counter (blocked) across the whole reconcile_run() call, not blocked AND detach_declined"
    requirement: "NF-09"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestRetirePathLegacyEventGuard.test_retiring_a_night_never_deletes_a_human_confirmed_legacy_event"
        status: pass
    human_judgment: false
  - id: D7
    description: "Shape (c)-different-run is never deleted and a confirmed_by row is never cleared at any of the four call sites -- the four pre-existing protected-case guard tests stay green with original assertions unedited"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestRetirePathLegacyEventGuard and TestFinalConvergenceGuard (pre-existing tests)"
        status: pass
    human_judgment: false
  - id: D8
    description: "Runbook retired/legacy_deleted definitions and the paired notebook's counter-narrative cell state what the code now does (NF-07)"
    requirement: "NF-07"
    verification:
      - kind: manual_procedural
        ref: "docs/runbooks/telescope_runs_calendar.rst and docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb, reviewed by this executor against the code changes"
        status: pass
    human_judgment: true
    rationale: "Prose-quality documentation correctness is a judgment call about whether the wording faithfully and clearly describes the new behavior, not something a unit test asserts."

duration: ~55min
completed: 2026-09-14
status: complete
---

# Quick Task 260913-ti1 Summary

**Closed the silent third-outcome gap in the campaign reconciler's delete/detach pipeline (35-REVIEW.md NF-01 BLOCKER): an unattributed ALLOC:/RUN: calendar event now converges to deletion at all four call sites, `_may_write()` polices both key namespaces (NF-06), the CR-03 retire branch no longer double-counts one decision under two counters (NF-09), and the runbook/notebook counter prose was corrected to match (NF-07).**

## Performance

- **Duration:** ~55 min
- **Completed:** 2026-09-14T04:55:39Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Added `_clearable_declined_and_unattributed()` in `campaign_reconciler.py` -- a total-partition helper that splits every stale-event candidate into deletable/declined/left-alone, covering shape (c)-this-run (via the existing `_clearable_and_declined()`) AND the previously-invisible shapes (a) no companion row at all and (b) a companion row with `run IS NULL`.
- Routed all four call sites named in 35-REVIEW.md's ground truth through the new helper: `project_allocation()`'s CR-04 convergence step, its CR-03 retire branch, `campaign_reconciler._stale_dated_events()` (WR-10), and `_stale_allocation_events()` (CR-02).
- Widened `_may_write()` to check both the `RUN:` and `ALLOC:{run.pk}:` prefixes in its fallback branch, closing the divergence from `writable_allocation_events()`'s queryset filter that left an unattributed `ALLOC:` night permanently `blocked`.
- Fixed the NF-09 double-count: `legacy_urls_claimed` is now claimed the moment the CR-03 retire branch decides a legacy event's fate at all (deleted, blocked, or declined), not only on the deletable path -- so a blocked/declined legacy event is no longer also reported under `detach_declined` by the downstream `_stale_dated_events()` step.
- Added 12 new regression tests plus 4 assertion-only extensions to existing tests across `test_allocation_projector.py` and `test_campaign_reconciler.py`, pinning every unattributed shape at every call site, plus a dedicated `_may_write()`/queryset-twin predicate-agreement test.
- Corrected the runbook's `retired` and `legacy_deleted` counter definitions (NF-07, folded in) and the paired notebook's matching narrative markdown cell.

## Task Commits

Each task was committed atomically:

1. **Task 1: One total-partition helper at all four call sites, one predicate that polices both namespaces** - `3c00e44` (fix)
2. **Task 2: Pin the remaining unattributed cases, predicate agreement, and the one-counter invariant** - `7c9afcd` (test)
3. **Task 3: Operator-facing counter definitions, paired-docs audit, quality gates** - `a0834b3` (docs)

**Plan metadata:** committed separately by the orchestrator (this executor does not commit `.planning/` docs artifacts per its constraints).

_Note: Task 1's commit contains only the source fix plus its own tracer test (the shape-(b) PROBE-A2 case); Task 2's shape-(a) twin and the remaining regression tests were split into a separate commit via a hand-constructed patch (`git apply --cached`) since both tests were written into the same file region before either was committed._

## Files Created/Modified

- `solsys_code/campaign_reconciler.py` - Added `_clearable_declined_and_unattributed()`; widened `_may_write()`; routed `_stale_dated_events()` and `_stale_allocation_events()` through the new helper
- `solsys_code/allocation_projector.py` - Routed the CR-04 convergence step and CR-03 retire branch through the new helper; moved `legacy_urls_claimed.add()` to fire on every retire-branch decision (NF-09); corrected `writable_allocation_events()`'s docstring claim
- `solsys_code/tests/test_campaign_reconciler.py` - Added shape-(b) coverage for `TestLegacyPerNightFamilyDeletion` and `TestReclassificationConvergence`; added the `TestOwnershipScoping` predicate-agreement test; extended one existing test with counter-invariant assertions
- `solsys_code/tests/test_allocation_projector.py` - Added shape-(a)/(b) tracer and twin tests to `TestFinalConvergenceGuard`; added two new NF-01-item-4 tests plus `detach_declined == 0` assertions to `TestRetirePathLegacyEventGuard`
- `docs/runbooks/telescope_runs_calendar.rst` - Rewrote the `retired` and `legacy_deleted` counter definitions (NF-07)
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` - Corrected the "Six counters on ReconcileResult" markdown cell's stale `retired` claim (markdown-only edit, not re-executed)

## Decisions Made

- One shared helper at all four call sites rather than four local patches, per the plan's stated root-cause analysis and to avoid a fifth iteration of the same gap class.
- `_may_write()`'s namespace check uses a single combined `return` with three OR'd conditions (container exact-match, `RUN:` prefix, `ALLOC:` prefix) rather than an early-return per namespace, keeping the early-return/exact-match branch byte-identical as instructed.
- Test coverage follows each class's existing fixture idioms exactly (`self._make_run()`, `self._link_record()`, `_make_container_run_with_legacy_nights()`) rather than introducing a new fixture style.

## Deviations from Plan

None - plan executed exactly as written. No Rule 1-4 deviations were needed; the fix, its call-site routing, and the regression coverage all matched the plan's `<action>` blocks directly.

## Issues Encountered

- **Test-commit atomicity for Task 1 vs. Task 2:** Both the Task 1 tracer test (shape-b) and Task 2's shape-a twin were written into the same file region (`TestFinalConvergenceGuard`) before either was committed, since Task 2's test writing proceeded in parallel with a long-running background verification test run for Task 1. Resolved by hand-constructing a unified-diff patch containing only the shape-(b) hunk and applying it to the git index with `git apply --cached`, so Task 1's commit contains exactly its own tracer test and Task 2's commit contains everything else. Verified via `git diff --cached` before each commit that the staged content matched the intended task boundary.
- **RED verification for the TDD tracer task:** Since the source fix was written before the RED/GREEN cycle was formally demonstrated, RED was independently confirmed by `git stash push` on just the two source files, re-running the new test (confirmed failure matching the reviewer's PROBE-A2 output: event survives, all counters zero), then `git stash pop` to restore the fix. GREEN was then re-confirmed with the fix restored.

## Paired-Docs Audit (CLAUDE.md)

`campaign_reconciler.py`'s paired notebook is `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` (per CLAUDE.md's module map). Audit performed and recorded per the plan's Task 3 instructions:

- **Markdown cell correction:** the "Six counters on ReconcileResult" cell repeated the same stale `retired` claim ("an allocation night handed over to a real observation") the runbook carried. Corrected to summarize all four `retired` reasons, matching the runbook's new wording. Markdown-only; no code-cell output affected; not re-executed.
- **Code-cell output risk check:** inspected every committed code-cell output in the notebook that reports `blocked`, `detach_declined`, `legacy_deleted`, or `retired`. Every value is `0` across every reconcile/sweep call in the notebook, except: (a) the pre-existing WR-10 `legacy_deleted` values from the shape-(a) no-companion-row union, which were already covered by the pre-fix code and are therefore unaffected by this change, and (b) the intentional `retired: 1` from the D-05/D-07 observation-handoff demo cell. No committed output changed under this fix because the scratch-copied developer-database dataset captured at the notebook's last execution contained no shape-(b) (companion row with `run` unset) unattributed row among its `RUN:`/`ALLOC:` events. **Not re-executed** (per the plan's own decision rule: re-execute only if a committed output would change).
- **`load_telescope_runs_demo.ipynb`:** confirmed every `ALLOC_URL_NAMESPACE` usage in that notebook is a plain `CalendarEvent.objects.filter(url__startswith=...)` read, unaffected by the `_may_write()`/helper changes. No edit needed.
- **Runbook "blocked means owned by someone else" grep (NF-06):** searched `docs/runbooks/telescope_runs_calendar.rst` for any claim framing a `blocked` event as necessarily "owned by someone else." No such phrase exists in the page, and no dedicated `blocked` counter-definition paragraph exists at all (unlike `retired`/`rekeyed`/`legacy_deleted`, which each have one). Nothing to correct.

## Known Stubs

None.

## Threat Flags

None -- this change closes an existing security-relevant gap (silent, uncounted, unlogged permanence of an unattributed calendar event) rather than introducing new surface. The threat model in the plan (T-ti1-01 through T-ti1-04) covers the widened-deletion risk; all four mitigations described there are implemented as specified: `foreign_stale_count`/`run_id=run.pk` scoping stays untouched (shape (c)-different-run still excluded everywhere), the `confirmed_by` split routes a stray confirmed-but-unattributed row to `declined` rather than deleting it, the one-counter invariant is asserted in every new test, and `_may_write()`'s new function-local import sits after the exact-match early return so the common attributed path never reaches it.

## Next Phase Readiness

- 35-REVIEW.md's NF-01 (BLOCKER), NF-06, NF-07, and NF-09 findings are closed. Remaining 35-REVIEW.md findings (if any beyond NF-02/NF-03/WR-07/WR-11, already closed by sibling quick tasks 260913-ti3/rmd/ng8/npq) should be checked against the current state of that file before Phase 35 is marked fully reviewed.
- No blockers introduced. The reconciler and allocation projector's delete/detach pipeline now has a single, total-partition ownership rule enforced at all four call sites and pinned by regression tests at every companion-row shape and namespace combination.

## Self-Check: PASSED

All claimed files exist on disk (`solsys_code/campaign_reconciler.py`,
`solsys_code/allocation_projector.py`, `solsys_code/tests/test_campaign_reconciler.py`,
`solsys_code/tests/test_allocation_projector.py`, `docs/runbooks/telescope_runs_calendar.rst`,
`docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`, and this SUMMARY.md itself),
and all three task commit hashes (`3c00e44`, `7c9afcd`, `a0834b3`) are present in `git log`.

---
*Quick task: 260913-ti1*
*Completed: 2026-09-14*
