---
phase: quick-260805-qdc
plan: 01
subsystem: database
tags: [django, orm, security, data-integrity, campaign-reconciler]

# Dependency graph
requires:
  - phase: 29
    provides: "campaign_reconciler.py's reconcile_run()/owned_events()/_may_write() and models.py's CalendarEventMeta ownership model"
provides:
  - "writable_events(run) -- the queryset-level ownership rule shared by both write paths"
  - "Cross-run ownership guarantee for the run-deletion cascade and the reclassification detach step"
affects: [campaign-reconciler, campaign-attribution, phase-30-if-any]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Queryset-level ownership rule (writable_events) mirroring an existing single-event rule (_may_write) -- keep the read-only identity query (owned_events) separate from the write-scoped query so read-only consumers are never affected by a write-path security fix"

key-files:
  created: []
  modified:
    - solsys_code/campaign_reconciler.py
    - solsys_code/models.py
    - solsys_code/tests/test_campaign_reconciler.py

key-decisions:
  - "writable_events() added as a new function beside owned_events() rather than changing owned_events() itself, since owned_events() is a read-only identity query consumed by test_campaign_approval.py and the demo notebook and its semantics must not change"
  - "_detach_stale_family_events()'s bulk update gained a run=run filter term rather than switching to writable_events() directly -- the existing owned_events().exclude(...) shape is preserved, with ownership added as a second, narrower filter on the CalendarEventMeta side"
  - "Paired-docs check re-run at execution time (not trusted from planning time): confirmed docs/runbooks/telescope_runs_calendar.rst and the reconcile_campaign_runs demo notebook only describe same-run detach behavior and use owned_events() read-only, so neither needed editing"

requirements-completed: [T-29-19]

# Metrics
duration: 9min
completed: 2026-08-05
---

# Quick Task 260805-qdc Summary

**Closed security finding T-29-19 by adding `writable_events()`, a queryset-level ownership guard, and routing the run-deletion cascade and reconcile's stale-family detach step through it so neither can touch another run's staff-confirmed calendar attribution.**

## Performance

- **Duration:** 9 min
- **Started:** 2026-08-05T19:05:05-07:00
- **Completed:** 2026-08-05T19:14:30-07:00
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added `writable_events(run)` to `campaign_reconciler.py`: the queryset-level twin of `_may_write()`, narrowing `owned_events(run)` to events this run may actually write (no companion row, companion row with `run` unset, or companion row already pointing at this run).
- Routed the `pre_delete` cascade (`models.py`) through `writable_events()` instead of `owned_events()`, so deleting a `CampaignRun` can no longer hard-delete a `CalendarEvent` whose companion row attributes it to a different run.
- Narrowed `_detach_stale_family_events()`'s bulk update with an extra `run=run` filter term, so a reconcile of run A can no longer clear a `CalendarEventMeta.run` that a staff member has since re-pointed at run B.
- Added three regression tests (`TestCrossRunOwnershipGuards`) mirroring the auditor's three probes, proven RED (2 of 3 failing) before the fix and GREEN (all passing) after.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add the three cross-run ownership regression tests and prove them RED** - `7070fae` (test)
2. **Task 2: Route both write paths through the ownership rule and turn the tests GREEN** - `009195f` (feat)

**Plan metadata:** commit handled by orchestrator (not made by this executor per constraints)

## Files Created/Modified
- `solsys_code/campaign_reconciler.py` - Added `writable_events(run)`; narrowed `_detach_stale_family_events()`'s bulk update with a `run=run` filter term
- `solsys_code/models.py` - `_delete_owned_calendar_events_on_campaign_run_delete` now imports and calls `writable_events()` instead of `owned_events()`
- `solsys_code/tests/test_campaign_reconciler.py` - Added `TestCrossRunOwnershipGuards` with three regression tests

## Decisions Made
- `writable_events()` is a new function beside `owned_events()`, not a change to it -- `owned_events()` stays byte-identical (verified via `git diff`) since it is a read-only identity query used by `test_campaign_approval.py` (lines 253, 407, 437, 459, 466) and the `reconcile_campaign_runs_demo.ipynb` notebook.
- The OR-of-`Q` ownership pattern in `writable_events()` follows the existing precedent in `campaign_attribution.orphan_calendar_events()` (`Q(telescope_label_meta__isnull=True) | Q(telescope_label_meta__run__isnull=True)`), extended with a third term (`Q(telescope_label_meta__run=run)`) for the "already owned by this run" case that a write path (unlike an orphan-finder) must also include.
- Re-ran the paired-docs grep from `<paired_docs_assessment>` at execution time rather than trusting the planning-time conclusion: `docs/runbooks/telescope_runs_calendar.rst:310-328` describes detaching strictly within one run (a `source`/`telescope_class`/`site` correction on that same run), and `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` uses `owned_events()` only as a read-only inspection helper (lines 367, 418, 430) which is deliberately unchanged. Neither document asserts the old, buggy cross-run behavior as correct, so neither was edited and the notebook was not regenerated.

## Deviations from Plan

None - plan executed exactly as written. The one environment-only step taken (regenerating a gitignored `src/fomo/_version.py` stub inside this worktree so `manage.py` could import `src.fomo` at all -- the editable install points at the main repo checkout, not this worktree) is not a code deviation: the file is gitignored, untracked, and not part of any commit.

## Issues Encountered
- Running `manage.py test` initially failed inside this git worktree with `ModuleNotFoundError: No module named 'src.fomo._version'`, because the `fomo` package's editable install (`pip show fomo`) resolves to `/home/tlister/git/fomo_devel` (the main checkout), and `setuptools_scm`'s generated, gitignored `_version.py` only exists there, not in this worktree's own `src/fomo/`. Copied that gitignored stub into the worktree (a version string only, no behavior) so the test runner's `import src.fomo` succeeded. Resolved cleanly; no code changes involved and nothing to commit (git confirmed the file stays untracked/gitignored).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- T-29-01's ownership guarantee ("the ownership rule is the first condition checked in every write path") now holds for every write path in `campaign_reconciler.py`/`models.py`, including the two added post-review that this finding targeted.
- No further follow-up scoped by this task; `owned_events()`'s read-only consumers are confirmed unaffected.

---
*Phase: quick-260805-qdc*
*Completed: 2026-08-05*

## Self-Check: PASSED

- FOUND: solsys_code/campaign_reconciler.py
- FOUND: solsys_code/models.py
- FOUND: solsys_code/tests/test_campaign_reconciler.py
- FOUND: .planning/quick/260805-qdc-fix-t-29-19-phase-29-security-audit-rout/260805-qdc-SUMMARY.md
- FOUND commit: 7070fae
- FOUND commit: 009195f
