---
phase: 28-operator-assisted-attribution
plan: 06
subsystem: attribution
tags: [django, transactions, data-integrity, gap-closure]

# Dependency graph
requires:
  - phase: 28-04
    provides: attribution_queue.html, campaign_tables.py, campaign-list banner, operator runbook section (all four plans 28-01..28-04 complete)
provides:
  - "_undo_confirmation() reordered and gated: the D-13 dismissal write only happens after the link-clearing write actually matched a row, closing 28-VERIFICATION.md's WR-01 anti-pattern"
  - "event_attribution_backlog()/record_attribution_backlog() compute sole_high_candidate_pk from the orphan's full uncapped candidate list, matching campaign_views._is_sole_high_candidate()'s server-side gate, closing WR-02"
  - "candidates_for_event()/candidates_for_record() docstrings corrected to state the drop threshold applies to the rounded display score, closing IN-01's docstring/implementation mismatch (no behavior change)"
affects: [phase-29-reconcile-sweep]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Gate a side-effect write on the changed_count of the write that actually proves the precondition, inside the same transaction.atomic() block, rather than writing the side effect unconditionally before the precondition check"
    - "When a helper's docstring documents a 'full list' contract, bind a full_candidates local before any filtering and pass that explicitly, so a future filter added at the call site cannot silently narrow what an unrelated downstream consumer receives"

key-files:
  created: []
  modified:
    - solsys_code/campaign_views.py
    - solsys_code/campaign_attribution.py
    - solsys_code/tests/test_campaign_attribution_views.py
    - solsys_code/tests/test_campaign_attribution.py

key-decisions:
  - "IN-01 (candidates_for_event()/candidates_for_record() comparing the ROUNDED display score against 0.0 rather than the raw weighted sum) was deliberately NOT fixed behaviorally in this plan -- only the docstring was corrected to describe the actual (rounded-score) comparison. At the current weights (0.25/0.35/0.40) no real signal combination produces a nonzero raw total below 0.005, so the behavior change is a no-op today, and reaching the raw score would require widening _build_candidate()'s return or AttributionCandidate itself -- touching the scoring path TestCriterion5RealCase (the phase's acceptance test) depends on, for zero present-day effect. Re-tuning trigger recorded inline at both drop sites: if _build_candidate()'s weights are ever changed such that a meaningful signal can produce a raw total below 0.005, filter on the raw score there instead of the rounded one."

patterns-established: []

requirements-completed: [ATTRIB-02, ATTRIB-04]

# Metrics
duration: 70min
completed: 2026-08-01
---

# Phase 28: Operator-Assisted Attribution Summary — Plan 06

**Gated `_undo_confirmation()`'s dismissal write on the link-clearing write's `changed_count`, and rebound `sole_high_candidate_pk` to the full uncapped candidate list in both backlog builders — closing 28-VERIFICATION.md's WR-01/WR-02 anti-patterns and correcting the IN-01 docstring/implementation mismatch, with no behavior change on IN-01.**

## Performance

- **Duration:** ~70 min
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- `_undo_confirmation()` (`campaign_views.py`) reordered: the event-side `.update()` / record-side `.delete()` now runs first inside the outer `transaction.atomic()` block, and the D-13 dismissal `get_or_create` (with its nested savepoint / `IntegrityError` handling, unchanged) runs only when `changed_count` is truthy. A stale resubmit after a re-point, or a tampered POST naming a pair that was never actually confirmed, now writes nothing at all instead of permanently dismissing an un-confirmed pair.
- `event_attribution_backlog()`/`record_attribution_backlog()` (`campaign_attribution.py`) now bind a `full_candidates` local before applying the band filter, and pass `full_candidates` (not the filtered list) to `_sole_high_candidate_pk()`, matching its documented "full uncapped list" contract and `campaign_views._is_sole_high_candidate()`'s own re-derivation from the unfiltered list. `candidates[:MAX_CANDIDATES_PER_ORPHAN]` and `total_candidate_count` remain bound to the band-filtered list, so the "showing top N of M candidates" line is unchanged.
- `candidates_for_event()`/`candidates_for_record()` docstrings corrected (IN-01): the `<= 0.0` drop compares the *rounded display score*, not the raw weighted sum; inline comments at both drop sites name the deferral reasoning and the re-tuning trigger.
- 5 new tests (`TestUndoConfirmationOrdering`) pinning WR-01, including a mutation check (Task 1d) confirming tests 1-4 fail and test 5 stays green under the original (pre-fix) ordering.
- 4 new tests (`TestSoleHighCandidateUnderBandFilter`) pinning WR-02, including a precondition test asserting the fixture produces exactly one High and one Medium candidate by band constant.

## Task Commits

Each task was committed atomically:

1. **Task 1: WR-01 — gate the undo's dismissal write on the link-clearing write's changed_count** - `c0e5d2c` (fix)
2. **Task 2: WR-02 — compute sole_high_candidate_pk from the full candidate list; IN-01 docstring fix** - `05db925` (fix)

## Files Created/Modified
- `solsys_code/campaign_views.py` - `_undo_confirmation()` reordered and gated; docstring rewritten to state the new order and the WR-01 rationale
- `solsys_code/campaign_attribution.py` - `event_attribution_backlog()`/`record_attribution_backlog()` bind `full_candidates` and pass it to `_sole_high_candidate_pk()`; `candidates_for_event()`/`candidates_for_record()` docstrings and inline comments corrected for IN-01
- `solsys_code/tests/test_campaign_attribution_views.py` - `TestUndoConfirmationOrdering` (5 tests), inserted immediately after `TestConfirmUndo`; `AttributionViewTestBase` untouched (verified via `git diff ed0a497 HEAD` — only additive)
- `solsys_code/tests/test_campaign_attribution.py` - `TestSoleHighCandidateUnderBandFilter` (4 tests), appended at end of file

## Decisions Made
- **IN-01 deferral** — see `key-decisions` in frontmatter above for the full reasoning and re-tuning trigger. Summary: docstring corrected to describe the actual (rounded-score) comparison; behavior deliberately left unchanged because it is a no-op at current weights and touching it would widen the scoring path's return shape for zero present-day effect.
- **WR-02 fixture design** (`TestSoleHighCandidateUnderBandFilter`) — per the plan's instruction to vary the *run* rather than the orphan (so as not to duplicate `_medium_band_event()`'s existing event-side recipe), the Medium-band candidate was produced by giving `medium_run` a partial-overlap window (missing the orphan's first day, `date_overlap_score` = 0.5) and an unresolvable telescope string (`site=None`, `telescope_instrument` carrying no aperture-derivable token and no `500@`/metre pattern, so `telescope_match_score()` falls to `TELESCOPE_MATCH_INDETERMINATE` = 0.3) combined with a moderate instrument-similarity string (`'SCICAM-MUSCAT (site unresolved)'`, sharing tokens with the orphan's `'2M0-SCICAM-MUSCAT'`) landing the weighted sum at 0.62 — safely inside the Medium band (0.50–0.74). Verified empirically via a throwaway scratch test before finalizing, then codified as the precondition test (`test_precondition_the_fixture_really_produces_one_high_and_one_medium_candidate`) so any future re-tuning of weights/cut-points that breaks this fixture fails loudly rather than silently passing for the wrong reason.

## Deviations from Plan

None - plan executed exactly as written. Both tasks' acceptance criteria were met without any Rule 1/2/3 auto-fixes; the only judgment call (the exact WR-02 fixture shape) was explicitly left to planner/executor discretion by the plan's own wording ("Claude's Discretion" pattern from 28-CONTEXT.md carried into this plan's Task 2c instructions) and is documented above under Decisions Made.

## Issues Encountered
- The worktree's `src/fomo/_version.py` (setuptools_scm-generated, gitignored) did not exist in this freshly-created worktree, causing `python manage.py test` to fail with `ModuleNotFoundError: No module named 'src.fomo._version'` before any test could run. Recreated the file locally (not committed — it's gitignored, matching the main checkout's already-generated copy) so the Django test runner could import `src.fomo.settings`. This is a worktree-environment artifact, not a code change, and required no deviation-rule handling.
- Confirmed via `git diff --stat . 2>&1 | grep -v _version` equivalent (i.e. `git status --short`) that `_version.py` never appeared as a tracked/staged change in either task commit.

## Task 1d Mutation-Check Result

Recorded per the plan's `<output>` requirement. With `_undo_confirmation()`'s original ordering temporarily restored (dismissal write unconditional, before the link-clearing write) via `git apply -R` on the task's own diff:

```
FAIL: test_undo_naming_a_run_that_never_confirmed_this_event_writes_no_dismissal
  AssertionError: True is not false   (a bogus dismissal row WAS written)
FAIL: test_undo_naming_a_run_that_never_confirmed_this_record_writes_no_dismissal
  AssertionError: True is not false
FAIL: test_undo_naming_a_wrong_run_leaves_that_pair_still_offerable
  AssertionError: 3 not found in {1}   (stale_run permanently excluded from candidates_for_event())
FAIL: test_undo_of_a_never_confirmed_pair_writes_nothing_at_all
  AssertionError: 1 != 0               (a dismissal row was written despite no prior confirmation)

Ran 5 tests in 2.712s
FAILED (failures=4)
```

`test_a_genuine_undo_still_clears_the_link_and_writes_its_dismissal` (test 5) stayed green under the original ordering, confirming it pins the D-13 guarantee itself rather than the WR-01 fix, and that the reorder does not over-gate and silently disable a genuine undo's attributability. The fix was then reapplied via `git apply` (restoring the reordered, gated code) and the full 5-test class re-verified green before proceeding.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Both WARNING-level findings (WR-01, WR-02) and the INFO finding (IN-01) recorded in `28-VERIFICATION.md`'s Anti-Patterns table are now closed. Combined with 28-05 (executed in parallel, closing the two BLOCKER findings CR-01/CR-02 in `attribution_queue.html`/`admin.py` — no file overlap with this plan), all findings from the initial `28-VERIFICATION.md` pass (score 6/8) are addressed pending re-verification.
- `TestCriterion5RealCase` — the phase's acceptance test — passes unmodified, confirming neither fix touched the scoring path's real-case behavior.
- No file belonging to plans 28-01 through 28-04 (`*-PLAN.md`, `*-SUMMARY.md`) was modified, and no file overlaps with `28-05-PLAN.md`'s changes (`src/templates/campaigns/attribution_queue.html`, `solsys_code/admin.py`).

---
*Phase: 28-operator-assisted-attribution*
*Completed: 2026-08-01*

## Self-Check: PASSED

- FOUND: solsys_code/campaign_views.py
- FOUND: solsys_code/campaign_attribution.py
- FOUND: solsys_code/tests/test_campaign_attribution_views.py
- FOUND: solsys_code/tests/test_campaign_attribution.py
- FOUND: .planning/phases/28-operator-assisted-attribution/28-06-SUMMARY.md
- FOUND commit: c0e5d2c
- FOUND commit: 05db925
