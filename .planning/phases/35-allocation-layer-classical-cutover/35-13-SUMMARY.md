---
phase: 35-allocation-layer-classical-cutover
plan: 13
subsystem: api
tags: [django, astropy, zoneinfo, allocation, dry-run-parity, docstrings]

# Dependency graph
requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: "35-09's shared _raise_if_set_window_inverted() guard and both _mint_fields() caller-branch wiring; NF-17's two corrected claimed_legacy_urls docstring copies"
provides:
  - "_raise_if_set_window_inverted(run, night, existing=None) -- reaches the half-null sub-night shape the re-mint branch routes into, closing WR-01"
  - "A half-null dry-run/real-run parity regression test, twinning the existing set/set test"
  - "The third, now-current copy of the claimed_legacy_urls contract on _stale_dated_events(), closing WR-03"
affects: [allocation_projector, campaign_reconciler, cutover_classical_allocations, load_telescope_runs]

# Actuals (#2632)
actuals:
  tokens: 3148
  tasks: 3
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dry-run inversion guard falls back to a stored CalendarEvent's own start_time/end_time for a null sub-night field on the re-mint branch, rather than calling sun_event() -- astropy-free, D-13-safe boundary resolution for a preview."

key-files:
  created: []
  modified:
    - solsys_code/allocation_projector.py
    - solsys_code/tests/test_allocation_projector.py
    - solsys_code/campaign_reconciler.py

key-decisions:
  - "Committed Task 1 (guard fix + regression test) and Task 2 (docstring/comment narrowing) as one commit rather than two: the docstring and call-site comment changes are physically interleaved with Task 1's logic change in the same function and the same few call-site lines, so git produces no clean hunk boundary between them -- splitting would require hand-editing a single coherent diff with no semantic benefit."

requirements-completed: [ALLOC-01, ALLOC-02]

coverage:
  - id: D1
    description: "reconcile_run(run, dry_run=True) over a half-null sub-night run whose one set boundary is inverted against its stored counterpart raises the same ValueError the immediately following real run raises (WR-01, PROBE-D fixture)."
    requirement: ALLOC-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#test_dry_run_of_a_half_null_remint_inverted_window_also_raises"
        status: pass
    human_judgment: false
  - id: D2
    description: "The guard's both-null short-circuit matches _span_needs_remint()'s convention; both parity claims (helper docstring, create-branch comment) now name the one shape (create-path half-null) that remains unpreviewable rather than claiming full parity."
    requirement: ALLOC-02
    verification:
      - kind: unit
        ref: "grep -c 'night_start_utc is None and run.night_end_utc is None' solsys_code/allocation_projector.py == 2"
        status: pass
      - kind: unit
        ref: "grep -c 'half-null' solsys_code/allocation_projector.py -ge 2"
        status: pass
    human_judgment: false
  - id: D3
    description: "The third copy of the claimed_legacy_urls contract, on _stale_dated_events() (the function that performs the exclusion), now states the same four outcomes and load-bearing-in-real-mode claim as the two already-corrected copies (WR-03)."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "grep -v '^ *#' solsys_code/campaign_reconciler.py | grep -c 'double-counts the SAME url' == 0"
        status: pass
      - kind: unit
        ref: "solsys_code.tests.test_campaign_reconciler (63 tests)"
        status: pass
    human_judgment: false

duration: ~20min
completed: 2026-09-15
status: complete
---

# Phase 35 Plan 13: Half-Null Dry-Run Parity Guard + Stale Docstring Fix Summary

**Closed the third consecutive iteration of the dry-run/real-run inversion-guard parity bug (NF-10 -> NF-20 -> WR-01) by matching the guard's null-field short-circuit to `_span_needs_remint()`'s convention, and corrected the last stale copy of the `claimed_legacy_urls` contract (WR-03).**

## Performance

- **Duration:** ~20 min (estimated; start time not captured at kickoff)
- **Completed:** 2026-09-15T16:44:14Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- `_raise_if_set_window_inverted()` now short-circuits only when **both** `night_start_utc`
  and `night_end_utc` are `None` (matching `_span_needs_remint()`'s `and`, not the prior
  `or`). A half-null run (one field set, one null -- the shape a `1130-EoN`/`BoN-0230`
  half-night classical line produces) now has its missing boundary resolved from the
  re-mint branch's own stored `existing.start_time`/`existing.end_time`, so the dry run
  raises the same `ValueError` the real run raises, with no new `sun_event()` call and no
  D-13 breach. The create-path half-null case (no stored counterpart) remains the one shape
  only the real run can detect -- now explicitly documented rather than silently claimed
  covered.
- Added `test_dry_run_of_a_half_null_remint_inverted_window_also_raises`, twinning the
  existing set/set regression, built from 35-REVIEW.md's PROBE-D fixture (La Silla,
  `night_start_utc=23:00`/`night_end_utc=None` minted first, then the start edited to
  `11:30` -- after that night's 11:29:46 sunrise). Both the dry run and the real run now
  raise, with identical exception text.
- Narrowed both parity claims in `allocation_projector.py` (the helper's docstring and the
  create-branch comment) to state exactly what the guard checks and what it does not, rather
  than claiming full parity -- closing the same code/docstring disagreement that caused
  NF-20.
- Replaced the third, stale copy of the `claimed_legacy_urls` `Args:` description on
  `_stale_dated_events()` -- the function that actually performs the
  `.exclude(url__in=claimed_legacy_urls)` -- with the same wording the two already-corrected
  copies (`campaign_reconciler.py:746-756`, `allocation_projector.py:576-595`) use: all four
  outcomes (takeover re-key, retirement delete, block on either branch, human-confirmed
  decline), and the exclusion stated as load-bearing in real mode for a blocked or declined
  url. No executable code changed in this task.

## Task Commits

Each task was committed atomically (Task 1 and Task 2 combined -- see Decisions Made):

1. **Task 1 + Task 2: Make the shared inversion guard reach the half-null shape / narrow both parity claims** - `ffd5174` (fix)
2. **Task 3: Replace the third copy of the claimed_legacy_urls contract** - `2448264` (docs)

## Files Created/Modified

- `solsys_code/allocation_projector.py` - `_raise_if_set_window_inverted()` gained an `existing` parameter and a both-null short-circuit; both call sites and both parity-claim docstrings/comments updated
- `solsys_code/tests/test_allocation_projector.py` - added the half-null dry-run/real-run parity regression test
- `solsys_code/campaign_reconciler.py` - `_stale_dated_events()`'s `claimed_legacy_urls` Args description replaced with the current four-outcome contract

## Decisions Made

- **Committed Task 1 and Task 2 together.** Task 2's docstring/comment narrowing work
  turned out to be inseparable at the git-hunk level from Task 1's implementation: writing
  the correct behavior naturally produces the correct docstring in the same edit, and the
  create-branch comment sits a handful of lines from Task 1's call-site change, so `git diff`
  merges them into the same two hunks with no clean split point. Splitting them into two
  commits would have meant temporarily committing a docstring that still claimed full parity
  it now demonstrably didn't have, or hand-splitting a single coherent diff for no semantic
  benefit. All of Task 1's and Task 2's individual verify commands and acceptance criteria
  pass against the combined commit.
- No other deviations. Both tasks matched the plan's suggested helper body and wording
  almost verbatim (from 35-REVIEW.md's WR-01/WR-03 sections).

## Deviations from Plan

None beyond the commit-grouping decision documented above (not a code deviation -- no
auto-fix rule applies; it is a task-boundary/commit-granularity judgment call).

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Verification

All plan-level verification commands were run and passed:

- `python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_cutover_classical_allocations solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_observation_projector_signals` -> **217 tests, OK** (216 before this plan after 35-12's own additions, +1 for the new half-null regression test).
- `pre-commit run ruff --all-files` -> Passed.
- `pre-commit run ruff-format --all-files` -> Passed.
- `git status --short` -> only `.planning/` and pre-existing untracked scratch files beyond the three `files_modified` paths.
- `grep -c 'night_start_utc is None and run.night_end_utc is None' solsys_code/allocation_projector.py` -> `2`.
- `grep -c 'half-null' solsys_code/allocation_projector.py` -> `4` (>= 2 required).
- `grep -v '^ *#' solsys_code/campaign_reconciler.py | grep -c 'double-counts the SAME url'` -> `0`.
- `grep -ci 'load-bearing' solsys_code/campaign_reconciler.py` -> `2` (>= 2 required).
- `git diff --stat solsys_code/campaign_reconciler.py` for Task 3's commit shows changes confined to the docstring line range (8 insertions, 5 deletions, one paragraph).

## Next Phase Readiness

WR-01 and WR-03 from `35-VERIFICATION.md`'s second gap-closure round are closed. Remaining
open items from that verification pass (CR-01 already closed by plan 35-12; WR-02, WR-04,
IN-01/02/03) belong to plans 35-14/35-15 or later, per the dispatch scope of this plan.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-15*

## Self-Check: PASSED

All modified files (`solsys_code/allocation_projector.py`, `solsys_code/tests/test_allocation_projector.py`, `solsys_code/campaign_reconciler.py`) and this SUMMARY.md confirmed present on disk. Both task commits (`ffd5174`, `2448264`) confirmed present in `git log`.
