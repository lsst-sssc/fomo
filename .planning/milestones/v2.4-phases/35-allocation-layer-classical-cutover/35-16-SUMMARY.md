---
phase: 35-allocation-layer-classical-cutover
plan: 16
subsystem: api
tags: [django, astropy-free, dry-run-parity, revert, docstrings]

# Dependency graph
requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: "35-13's shared _raise_if_set_window_inverted(run, night, existing) guard and both _mint_fields() caller-branch wiring -- the exact code this plan reverts"
provides:
  - "_raise_if_set_window_inverted(run, night) -- two-parameter guard with no stored-boundary fallback, resolving both boundaries only from the run's own sub-night fields"
  - "PROBE-P1 pinned as agreement (dry run and real run both clean) and PROBE-P6 pinned as a documented, deliberate limitation (preview silent, real run raises)"
  - "Corrected guard docstring and both call-site comments stating what the guard proves instead of a provenance property the run row does not record"
affects: [allocation_projector, campaign_reconciler, cutover_classical_allocations, load_telescope_runs]

# Actuals (#2632)
actuals:
  tokens: 4227
  tasks: 2
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dry-run inversion guard restored to silence-on-unknown: when either sub-night boundary cannot be resolved from the run's own fields alone, the guard returns without raising rather than substituting a stored value of unproven provenance."

key-files:
  created: []
  modified:
    - solsys_code/allocation_projector.py
    - solsys_code/tests/test_allocation_projector.py

key-decisions:
  - "Also corrected the create-branch's `if dry_run:` comment (lines ~751-772), even though Task 2's action text named only the re-mint branch's comment. That comment still asserted 'a half-null pair on the re-mint branch, which falls back to `existing`'s stored boundary (WR-01)' -- a claim the Task 1 revert falsified. Leaving it would have reintroduced exactly the code/prose mismatch this plan exists to close (Rule 1 auto-fix: a stale, now-false comment adjacent to the code being corrected)."

requirements-completed: [ALLOC-01, ALLOC-02]

coverage:
  - id: D1
    description: "_raise_if_set_window_inverted() no longer reads a boundary off an existing CalendarEvent on any path; two-parameter signature at the one definition and both call sites."
    requirement: ALLOC-02
    verification:
      - kind: unit
        ref: "grep -c '_raise_if_set_window_inverted(run, night)' solsys_code/allocation_projector.py == 2"
        status: pass
      - kind: unit
        ref: "grep -c 'if existing is not None else None' solsys_code/allocation_projector.py == 0"
        status: pass
    human_judgment: false
  - id: D2
    description: "PROBE-P1 (dry run and real run agree cleanly on a half-null re-mint whose previously-set start was nulled) is pinned as a regression test."
    requirement: ALLOC-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#test_dry_run_of_a_half_null_remint_after_nulling_a_set_start_agrees_with_the_real_run"
        status: pass
    human_judgment: false
  - id: D3
    description: "PROBE-P6 (dry run stays silent, real run raises, on a half-null re-mint whose previously-set start was nulled to invert against the site's sunset) is pinned as the accepted, documented limitation."
    requirement: ALLOC-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#test_dry_run_cannot_see_a_half_null_remint_inversion_and_the_real_run_still_raises"
        status: pass
    human_judgment: false
  - id: D4
    description: "No test and no docstring asserts a half-null re-mint's dry run raises; the guard's docstring and both call-site comments state the limit and cite PROBE-P1/PROBE-P6 rather than a provenance property the CampaignRun row does not record."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "grep -c 'test_dry_run_of_a_half_null_remint_inverted_window_also_raises' solsys_code/tests/test_allocation_projector.py == 0"
        status: pass
      - kind: unit
        ref: "grep -c 'the one shape that remains unchecked' solsys_code/allocation_projector.py == 0 and grep -c 'minted from the same deterministic' solsys_code/allocation_projector.py == 0 and grep -c 'PROBE-P1' solsys_code/allocation_projector.py -ge 1"
        status: pass
    human_judgment: false
  - id: D5
    description: "Six-module regression surface, ruff and ruff-format all pass; working tree clean beyond the two files_modified paths and .planning/."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "python manage.py test solsys_code.tests.{test_allocation_projector,test_allocation_projector_signals,test_campaign_reconciler,test_cutover_classical_allocations,test_load_telescope_runs,test_observation_projector_signals} (235 tests)"
        status: pass
      - kind: unit
        ref: "pre-commit run ruff --all-files && pre-commit run ruff-format --all-files"
        status: pass
    human_judgment: false

duration: ~10min
completed: 2026-09-15
status: complete
---

# Phase 35 Plan 16: Revert the Half-Null Dry-Run Fallback (Defect A) Summary

**Deleted the round-2 stored-boundary fallback in `_raise_if_set_window_inverted()` -- the fourth iteration of the same dry-run/real-run inversion-guard parity bug (NF-10 -> NF-20 -> WR-01(round2) -> this) -- ending it by subtraction: the guard resolves both sub-night boundaries only from the run's own fields and returns silently, on either caller branch, whenever either is unknown.**

## Performance

- **Duration:** ~10 min (estimated; start time not captured at kickoff)
- **Completed:** 2026-09-15T18:30:06Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `_raise_if_set_window_inverted()` reverted to a two-parameter signature `(run, night)`.
  The round-2 fallback (`existing.start_time`/`existing.end_time` for a null sub-night
  field) is deleted outright -- no `sun_event()` call, no astropy import, no new query
  added; the function now only deletes. Both call sites (the re-mint branch's `if
  dry_run:` short-circuit and the create branch) call the guard identically.
- Added `test_dry_run_of_a_half_null_remint_after_nulling_a_set_start_agrees_with_the_real_run`,
  pinning PROBE-P1 (35-VERIFICATION.md) as a regression: mint `night_start_utc=23:00`/
  `night_end_utc=None`, edit to `night_start_utc=None`/`night_end_utc=22:30` (a
  `2300-EoN` -> `BoN-2230` operator edit), and assert BOTH `reconcile_run(run,
  dry_run=True)` and the real `reconcile_run(run)` complete with no error, with the real
  run's event ending at `22:30` UTC and `start_time < end_time`.
- Rewrote the stale third sibling inversion test in place --
  `test_dry_run_of_a_half_null_remint_inverted_window_also_raises` (which asserted a
  half-null re-mint's dry run raises the same error as the real run, and only ever
  passed for the one sub-shape whose null field was ALSO null at mint time) is now
  `test_dry_run_of_a_half_null_remint_inverted_window_stays_silent_while_the_real_run_raises`,
  asserting the narrowed, true contract: the preview stays silent, the real run still
  raises. The old name and assertions no longer exist anywhere in the module.
- Added `test_dry_run_cannot_see_a_half_null_remint_inversion_and_the_real_run_still_raises`,
  pinning PROBE-P6 as a KNOWN, DELIBERATE limitation rather than a passing contract: mint
  `night_start_utc=21:00`/`night_end_utc=None`, edit to `night_start_utc=None`/
  `night_end_utc=21:30`, and assert the preview stays silent while the real run raises.
  Its docstring states that a future fix must solve provenance (whether a stored
  boundary really is sun-derived) rather than re-infer it, and names both PROBE-P1 and
  PROBE-P6 as the two directions -- keeping a previously-set field's old value on either
  side of the real sunset/sunrise -- that falsified round 2's substitution.
- Rewrote the guard's docstring and both call-site comments (the re-mint branch's `if
  dry_run:` comment, plus the create branch's parallel comment) to state what the guard
  actually proves -- it checks a span only when both sub-night fields are set, and a
  half-null run is not previewed for inversion on either branch -- deleting the
  falsified claims that a stored boundary was "minted from the same deterministic
  `sun_event()`" and that the create path was "the one shape that remains unchecked".
  Both PROBE-P1 and PROBE-P6 are now cited by name in the module's own prose.
- Both existing set/set inversion regressions
  (`test_dry_run_of_a_brand_new_inverted_window_also_raises`,
  `test_dry_run_of_a_remint_inverted_window_also_raises`) are unchanged and still pass --
  the revert narrows only the half-null shape.

## Task Commits

Each task was committed atomically:

1. **Task 1: One half-null night, previewed and really run, agreeing end to end** - `5eeaa48` (fix)
2. **Task 2: Pin the limit the guard actually has, in a test and in its own prose** - `3635573` (docs)

## Files Created/Modified

- `solsys_code/allocation_projector.py` - `_raise_if_set_window_inverted()` reverted to a
  two-parameter signature with no stored-boundary fallback; both call sites and the
  guard's docstring/comments corrected to state the true, narrower contract
- `solsys_code/tests/test_allocation_projector.py` - added the PROBE-P1 regression and
  the PROBE-P6 limitation-pin; rewrote the stale half-null test in place

## Decisions Made

- **Also corrected the create-branch's `if dry_run:` comment**, beyond what Task 2's
  action text literally named (only the re-mint branch's comment). The create-branch
  comment still asserted, post-Task-1, that "a half-null pair on the re-mint branch...
  falls back to `existing`'s stored boundary (WR-01)" -- a claim Task 1's revert had
  already falsified. Leaving a stale, now-false comment two branches away from the code
  it describes would have reintroduced the exact code/docstring mismatch this plan
  exists to close. Treated as a Rule 1 auto-fix (bug: a false statement in the code),
  not scope creep -- no grep gate required it, but the plan's own closing instruction
  ("Write no hedge. State the predicate: which shapes are checked, which are not, and
  why.") applies to both comments equally.

## Deviations from Plan

None beyond the above create-branch-comment correction, which is documented above as a
Rule 1 auto-fix rather than a plan deviation requiring a decision.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Verification

All plan-level verification commands were run and passed:

- `python manage.py test solsys_code.tests.test_allocation_projector` -> `OK` (64 tests),
  and `TestSubNightWindowSiteDirection` on its own -> `OK` (13 tests).
- `python manage.py test solsys_code.tests.test_allocation_projector
  solsys_code.tests.test_allocation_projector_signals
  solsys_code.tests.test_campaign_reconciler
  solsys_code.tests.test_cutover_classical_allocations
  solsys_code.tests.test_load_telescope_runs
  solsys_code.tests.test_observation_projector_signals` -> **235 tests, OK**.
- `pre-commit run ruff --all-files` -> Passed.
- `pre-commit run ruff-format --all-files` -> Passed.
- `git status --short` -> only `.planning/` and pre-existing untracked scratch files
  beyond the two `files_modified` paths.
- `grep -c '_raise_if_set_window_inverted(run, night)' solsys_code/allocation_projector.py`
  -> `2`.
- `grep -c 'if existing is not None else None' solsys_code/allocation_projector.py` -> `0`.
- `grep -c 'night_start_utc is None and run.night_end_utc is None'
  solsys_code/allocation_projector.py` -> `2`.
- `grep -c 'def _raise_if_set_window_inverted' solsys_code/allocation_projector.py` -> `1`.
- `grep -c 'test_dry_run_of_a_half_null_remint_inverted_window_also_raises'
  solsys_code/tests/test_allocation_projector.py` -> `0`.
- `grep -c 'the one shape that remains unchecked' solsys_code/allocation_projector.py`
  -> `0`.
- `grep -c 'minted from the same deterministic' solsys_code/allocation_projector.py`
  -> `0`.
- `grep -c 'PROBE-P1' solsys_code/allocation_projector.py` -> `3` (>= 1 required).

## Next Phase Readiness

Defect A (35-VERIFICATION.md gap 1, WR-01's fourth iteration) is closed by subtraction,
per the verification report's recommendation. Plans 35-17 and 35-18 (Defect B -- the
create-arm loader preview, and the `duplicate_identity` vocabulary fix) are separate
dispatches, not touched by this plan.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-15*

## Self-Check: PASSED

Both modified files (`solsys_code/allocation_projector.py`,
`solsys_code/tests/test_allocation_projector.py`) and this SUMMARY.md confirmed present
on disk. Both task commits (`5eeaa48`, `3635573`) confirmed present in `git log`.
