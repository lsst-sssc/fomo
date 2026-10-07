---
phase: 35-allocation-layer-classical-cutover
plan: 19
subsystem: allocation-calendar
tags: [django, calendar, provenance, sun-event, astropy, tdd]

# Dependency graph
requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: allocation_projector.py's ALLOC: per-night projection (plans 35-01..35-18), the sub-night window fields (plan 35-03), and the classical cutover (plans 35-08..35-18)
provides:
  - "CalendarEventMeta.minted_sub_night_window: a nullable, staff-readonly provenance column recording the sub-night window pair an ALLOC: night's boundaries were minted from"
  - "_span_needs_remint() correctly re-mints a night whose previously-SET sub-night field was cleared to null, closing CR-01 (35-REVIEW.md iteration 7)"
  - "A bounded, one-time sun_event() resolution for legacy nights whose mint provenance was never recorded"
affects: [35-allocation-layer-classical-cutover verification/UAT, any future plan touching allocation_projector.py's re-mint decision]

# Actuals (#2632)
actuals:
  tokens: 8838
  tasks: 3
  commits: 5

tech-stack:
  added: []
  patterns:
    - "Mint-provenance recording: a pure token function (_sub_night_provenance_token) plus a scoped writer (_record_sub_night_provenance) that records ONLY the provenance field, called exclusively at the two sites that actually minted the boundaries -- never on plain-update or legacy-re-key paths."
    - "Bounded legacy resolution: an unrecorded-provenance night pays exactly one sun_event() call, once ever, compared against the stored boundary with a named tolerance constant separating astropy session drift from a genuinely stale operator value; the result is then recorded so the same night never pays the call again."

key-files:
  created:
    - solsys_code/migrations/0019_calendareventmeta_minted_sub_night_window.py
  modified:
    - solsys_code/models.py
    - solsys_code/admin.py
    - solsys_code/allocation_projector.py
    - solsys_code/tests/test_allocation_projector.py
    - solsys_code/tests/test_cutover_classical_allocations.py

key-decisions:
  - "Rejected the review's naive fix (call sun_event() from _span_needs_remint() whenever a field is null) because it would make every idempotent sweep of a null/null classical run pay an astropy solar scan forever, breaking the D-13 budget test TestNoSunEventRecompute pins. Implemented the plan's provenance-recording design instead: the discriminating fact (what a night was minted from) is persisted at mint time, so the null-case decision becomes a zero-astropy string comparison once recorded."
  - "A cleared sub-night field's replacement boundary is proven only by deleting and re-minting the night through the existing _mint_fields()/night_bounds() path -- never by writing a computed value in place -- preserving D-13's delete-and-recreate contract."
  - "The unrecorded-provenance (legacy) branch never infers provenance from the stored value -- it computes and compares against a fresh sun_event() call, exactly once, and records only what that computation proves. This mirrors round 3's reversion of round 2's stored-boundary substitution mistake."
  - "Rule 1 deviation: rebuilt test_cutover_classical_allocations.py's TestCutoverSequenceContract fixture with real sun_event()-derived boundaries instead of the file's shared round-hour (23:00/09:00 UTC) convention, since that convention sits outside CR-01's one-minute tolerance and the fix correctly re-minted it as a genuinely-stale legacy night -- the fix working as designed, not a regression. Scoped to the one test that runs the reconciler sweep right after cutover; the shared _make_three_night_group() helper (used by ~30 other tests that never run the sweep) was left untouched."

requirements-completed: [ALLOC-01]

coverage:
  - id: D1
    description: "A staff admin edit clearing a previously-SET night_start_utc/night_end_utc field re-mints exactly that night on the next reconcile, instead of leaving the calendar permanently stale (CR-01 closed) -- all three probe shapes (A: both fields cleared, B: half-null's remaining field cleared, C: one of two set fields cleared) proven against a live sun_event() result."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestClearedSubNightFieldRemints (4 tests: shape C, shape A, shape B, dry-run parity)"
        status: pass
    human_judgment: false
  - id: D2
    description: "A legacy night whose mint provenance was never recorded (minted before this column existed, or taken over by the cutover's re-key path) is resolved exactly once against a real sun event: correct boundaries report unchanged and record provenance; stale boundaries re-mint and log a warning; a dry run agrees with the real run and writes nothing."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestUnrecordedProvenanceNight (3 tests)"
        status: pass
    human_judgment: false
  - id: D3
    description: "D-13's astropy budget is intact: an idempotent re-reconcile of a run whose nights carry recorded provenance makes zero sun_event() calls; astropy session drift alone never re-mints a night."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestNoSunEventRecompute (unedited, 3 tests) and TestSubNightWindow (unedited, 6 tests)"
        status: pass
    human_judgment: false
  - id: D4
    description: "minted_sub_night_window is not staff-writable on either CalendarEventMeta admin surface."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "grep -n 'minted_sub_night_window' solsys_code/admin.py -- present only inside both readonly_fields lists"
        status: pass
    human_judgment: false

duration: 18min
completed: 2026-09-15
status: complete
---

# Phase 35 Plan 19: Close CR-01 -- Mint-Provenance Re-mint Decision Summary

**A `minted_sub_night_window` provenance column on `CalendarEventMeta` turns the sub-night re-mint decision's null case into a zero-astropy string comparison once recorded, and a bounded one-time `sun_event()` call resolves any legacy night whose provenance was never recorded -- closing CR-01 without reopening D-13's astropy-budget test.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-09-15T19:39:00Z (approx.)
- **Completed:** 2026-09-15T19:57:05Z
- **Tasks:** 3
- **Files modified:** 6 (5 in the plan's `files_modified`, plus 1 test file via a documented Rule 1 deviation)

## Accomplishments

- Closed CR-01 (35-REVIEW.md iteration 7, 35-VERIFICATION.md fourth pass): `_span_needs_remint()` now correctly re-mints a night whose previously-SET sub-night field was cleared to null, for all three probe shapes the verifier reproduced (A, B, C), each asserted against a live `sun_event()` result rather than a hardcoded timestamp or a bare counter.
- Added `CalendarEventMeta.minted_sub_night_window` (migration 0019, schema-only, no `RunPython`) -- a nullable provenance column recording the sub-night window pair a night's boundaries were minted from, read-only on both `CalendarEventMetaAdmin` and `CalendarEventMetaInline`.
- Added the bounded, one-time `sun_event()` resolution for a legacy night whose provenance was never recorded: correct boundaries record provenance and never pay astropy again; stale boundaries re-mint and log a warning naming the run, the night, the stored boundary and the resolved sun event.
- Preserved D-13's astropy budget exactly: `TestNoSunEventRecompute` and `TestSubNightWindow` pass unedited (zero lines changed inside either class), and the new unrecorded-provenance branch is the only place a new `sun_event()` call was added, gated to run at most once per legacy night.
- Full five-module regression surface (`test_allocation_projector`, `test_allocation_projector_signals`, `test_campaign_reconciler`, `test_cutover_classical_allocations`, `test_load_telescope_runs`) green: 212/212 tests (baseline 205 + this plan's 7 new). Both ruff hooks pass. `git status --short -- docs/ solsys_code/management/` is clean. `makemigrations --check --dry-run` reports no drift.

## RED Failure Output (Shape C, before the fix)

```
FAIL: test_clearing_only_one_of_two_set_fields_remints_the_night
AssertionError: 0 != 1
```
(`result.retired` was `0` instead of `1` -- the night's stale `start_time` from before the field was cleared silently survived, and `reconcile_run()` reported `unchanged` instead of `retired=1/created=1`, exactly the staleness 35-VERIFICATION.md's PROBE C recorded.)

## Probe Shape Counters (before -> after the fix)

| Shape | Setup | Before fix | After fix |
|---|---|---|---|
| C (Task 1 tracer) | set/set -> only `night_start_utc` cleared | `unchanged=1`, stale `start_time` survives | `retired=1, created=1, unchanged=0`; new pk; `start_time` == real sunset; `end_time` unchanged (still-set boundary) |
| A (Task 3) | set/set -> both fields cleared | `unchanged=1` | `retired=1, created=1, unchanged=0`; new pk; both boundaries == real sunset/sunrise |
| B (Task 3) | half-null -> remaining set field cleared | `unchanged=1` | `retired=1, created=1, unchanged=0`; new pk; both boundaries == real sunset/sunrise |
| Dry-run parity (shape A) | same as A | n/a (not previously tested) | `dry_run=True` reports `retired=1, created=1` identically to the following real run; preview leaves pk and both boundaries untouched |

## Full Five-Module Test Count

`python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_cutover_classical_allocations solsys_code.tests.test_load_telescope_runs -v 1`

**Ran 212 tests ... OK** (prior baseline: 205; this plan added 7: 1 shape-C tracer test, 3 unrecorded-provenance tests, 2 additional probe-shape tests (A, B), 1 dry-run-parity test).

## Ruff Results

- `pre-commit run ruff --all-files` -- **Passed**
- `pre-commit run ruff-format --all-files` -- **Passed** (one auto-reformat applied to `allocation_projector.py`'s multi-line `if` conditions during Task 2, re-verified clean afterward)

## D-13 Astropy Budget -- Explicitly Confirmed

`TestNoSunEventRecompute` (`solsys_code/tests/test_allocation_projector.py`) was left **unedited** -- `git diff` from before this plan to `HEAD` over that file shows **zero deletions**, purely additive content. It still asserts, unchanged:
- `test_second_reconcile_of_unchanged_run_never_calls_sun_event`: a second reconcile of an unchanged multi-night run makes zero `sun_event()` calls.
- `test_a_deleted_night_calls_sun_event_exactly_twice_on_next_reconcile`: a deleted night calls `sun_event()` exactly twice (sun + dark) on the next reconcile.
- `test_dry_run_of_a_brand_new_run_never_calls_sun_event`: a dry run of a brand-new run never calls `sun_event()`.

`TestSubNightWindow` (6 tests) is likewise unedited and green, including `test_reconcile_with_sub_night_fields_set_makes_no_further_sun_event_calls` and `test_second_reconcile_with_matching_sub_night_fields_writes_nothing`.

This is the constraint that decided the fix's design (see `<design_rationale>` in the plan): the review's suggested naive fix (calling `sun_event()` from `_span_needs_remint()` whenever a field is null) would have broken these tests by making every idempotent sweep of a null/null run pay an astropy solar scan forever. The provenance-recording design avoids this entirely -- once a night's provenance is recorded, the null-case decision is a zero-astropy string comparison, forever.

## Task Commits

Each task was committed atomically, following RED/GREEN discipline per the plan's `tdd="true"` tasks:

1. **Task 1: Record the sub-night window a night was minted from, and decide the cleared-field case from it**
   - `20795ce` (test, RED): probe shape C fails before the fix -- `retired: 0 != 1`.
   - `d02e856` (feat, GREEN): `minted_sub_night_window` column + migration 0019, admin readonly-field hardening on both surfaces, `_sub_night_provenance_token()`/`_record_sub_night_provenance()`, and `_span_needs_remint()` rewritten to decide the null case from recorded provenance. 65/65 `test_allocation_projector` tests green; `TestNoSunEventRecompute`/`TestSubNightWindow` pass unedited.
2. **Task 2: Resolve a night whose provenance was never recorded, once, and record what it proves**
   - `7310615` (test, RED): three unrecorded-provenance probes fail against Task 1's placeholder.
   - `43cf93c` (feat, GREEN): the bounded, one-time `sun_event()` resolution replacing the placeholder, `_UNRECORDED_PROVENANCE_TOLERANCE` (one minute) declared at module level. 68/68 `test_allocation_projector` tests green; `test_campaign_reconciler`/`test_load_telescope_runs` (93 tests) unaffected.
3. **Task 3: Pin all three probe shapes and prove nothing else in the phase moved**
   - `857c3b7` (test): probe shapes A and B, the dry-run parity test, and a documented Rule 1 fix to a pre-existing `test_cutover_classical_allocations.py` fixture the CR-01 fix correctly exposed as stale (see Deviations below). 212/212 tests across the full five-module surface; both ruff hooks pass; `git status --short -- docs/ solsys_code/management/` clean; `makemigrations --check` clean.

**Plan metadata:** this commit (docs: complete plan) -- see final commit below.

_Note: both Task 1 and Task 2 carry `tdd="true"` per the plan frontmatter and produced separate RED/GREEN commits._

## Files Created/Modified

- `solsys_code/models.py` -- `CalendarEventMeta.minted_sub_night_window`, a nullable/blank `CharField(max_length=32)`; class docstring extended in the existing voice.
- `solsys_code/migrations/0019_calendareventmeta_minted_sub_night_window.py` -- schema-only `AddField` migration, no `RunPython`.
- `solsys_code/admin.py` -- `minted_sub_night_window` added to `readonly_fields` on both `CalendarEventMetaAdmin` and `CalendarEventMetaInline`.
- `solsys_code/allocation_projector.py` -- `_sub_night_provenance_token()`, `_record_sub_night_provenance()`, `_UNRECORDED_PROVENANCE_TOLERANCE`, and `_span_needs_remint()` rewritten (keyword-only `dry_run`, no both-null early return, decides the null case from recorded provenance or a bounded one-time `sun_event()` resolution). Provenance recorded at exactly the two `_mint_fields()` call sites (re-mint branch, create branch) plus the Task 2 unrecorded-provenance branch.
- `solsys_code/tests/test_allocation_projector.py` -- `TestClearedSubNightFieldRemints` (4 tests: shapes A/B/C + dry-run parity) and `TestUnrecordedProvenanceNight` (3 tests).
- `solsys_code/tests/test_cutover_classical_allocations.py` -- Rule 1 deviation (see below): `test_cutover_then_sweep_reaches_the_pinned_end_state`'s convertible-group fixture rebuilt with real `sun_event()`-derived boundaries.

## Decisions Made

See `key-decisions` in frontmatter. Summary: implemented the plan's provenance-recording design exactly as specified in `<design_rationale>`, rejecting the review's naive per-sweep `sun_event()` call; never inferred provenance from a stored value (round 2's reverted mistake); and fixed one pre-existing test fixture that the correctness fix legitimately exposed as unrealistic, scoped to the single test that exercises it.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug, in a pre-existing test the fix legitimately exposed] `test_cutover_then_sweep_reaches_the_pinned_end_state` used unrealistic legacy boundaries**

- **Found during:** Task 3's full five-module regression run.
- **Issue:** `TestCutoverSequenceContract.test_cutover_then_sweep_reaches_the_pinned_end_state` runs the cutover command then a single `reconcile_campaign_runs` sweep and asserts the convertible three-night group keeps its primary keys (no re-mint). Its fixture used `_make_three_night_group()`'s shared round-hour convention (23:00/09:00 UTC) for La Silla, which sits ~53 minutes outside CR-01's one-minute tolerance from the real `sun_event()` result (22:06:35/11:29:46 UTC for 2026-07-09). With CR-01 fixed, the reconciler sweep immediately following the cutover now correctly audits this re-keyed legacy night's boundary against the true sun event (Task 2's unrecorded-provenance branch) and re-mints it as genuinely stale -- exactly the fix working as designed, not a regression. The real pre-cutover `load_telescope_runs` writer always computed `sun_event()`-derived boundaries, so the round-hour convention never represented a genuine legacy night; it was a test-authoring convenience that predates this fix.
- **Fix:** Rebuilt this one test's own three-night fixture inline with real `sun_event()`-derived boundaries (matching `night_bounds()`'s own expression) instead of calling the shared `_make_three_night_group()` helper. No other test in this file runs the reconciler sweep after cutover, so the shared helper (used by ~30 other tests) was left untouched, and this fix is scoped to the one test it affects.
- **Files modified:** `solsys_code/tests/test_cutover_classical_allocations.py` (added a `sun_event` import; rebuilt one test's fixture construction).
- **Verification:** `test_cutover_then_sweep_reaches_the_pinned_end_state` passes; full five-module suite green (212/212).
- **Committed in:** `857c3b7` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 -- a pre-existing test fixture the correctness fix legitimately exposed as unrealistic)
**Impact on plan:** No scope creep. The fix is confined to one test method in a file the plan's `files_modified` frontmatter did not list but the plan's `<prohibitions>` did not forbid touching (prohibition 1 names only the runbook, the two notebooks, `load_telescope_runs.py` and `cutover_classical_allocations.py` itself -- not its test file). `TestNoSunEventRecompute`/`TestSubNightWindow` remain unedited as required, and every file under `docs/` and `solsys_code/management/` is untouched (`git status --short -- docs/ solsys_code/management/` is clean).

## Issues Encountered

None beyond the deviation documented above.

## User Setup Required

None -- no external service configuration required. This is a schema-only migration (`AddField`, no `RunPython`) that applies cleanly on the next `python manage.py migrate`.

## Next Phase Readiness

- CR-01 is closed: `_span_needs_remint()` decides the null case correctly for all three probe shapes, restoring 35-03 truth 5 (D-13: a stored night whose boundaries no longer match the run's current sub-night fields is deleted and re-created) and 35-03 truth 2 (D-04: a null sub-night field means the computed sunset/sunrise) for the SET -> null transition, matching what already held for SET -> SET.
- 35-03 key_link 2 now reads WIRED for every transition: a staff admin edit of either sub-night field causes the next `reconcile_run()` to re-mint exactly the affected nights, including clear-to-null.
- ROADMAP SC-1 is no longer falsified by the documented operator action this round reproduced, and ALLOC-01 is no longer BLOCKED per 35-VERIFICATION.md.
- This was the fourth and, per the plan's own framing, intended-final gap-closure round scoped to CR-01 alone for Phase 35. WR-01 (dry-run half-null visibility), WR-02 (cutover re-run caveat's field list) and WR-03 (the discarded refusal return) remain recorded as `user_deferred:` in `35-VERIFICATION.md` by explicit prior user decision and were not touched by this round.
- No blockers for whatever GSD step runs next (re-verification or `/gsd-ship`).

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-15*

## Self-Check: PASSED

All claimed files exist on disk; all claimed commit hashes (`20795ce`, `d02e856`, `7310615`, `43cf93c`, `857c3b7`) resolve in `git log --oneline --all`.
