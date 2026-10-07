---
phase: 35-allocation-layer-classical-cutover
plan: 12
subsystem: allocation-layer
tags: [django-management-command, cutover, identity-guard, campaign-run, tdd, gap-closure, blocker-fix]

# Dependency graph
requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: "cutover_classical_allocations's database-scoped duplicate_identity guard (35-08), the guard's no-marker permissive predicate this plan inverts"
provides:
  - "cutover_classical_allocations's duplicate_identity guard refuses (never find-and-updates) a database claimant whose observation_details carries no recoverable Source line: marker, closing CR-01 (BLOCKER, 35-REVIEW.md)"
  - "a branching duplicate_identity reason string naming the exact Django admin remedy for the no-marker case, distinct from the differing-marker case"
  - "TestDatabaseScopedIdentityGuard's fourth case inverted to assert refusal, plus a fifth destructive-case regression (PROBE-A) asserting run_status/observation_details/target are byte-identical after a refused pass"
  - "module docstring and CommandError text both name the marker precondition instead of asserting the no-rewrite guarantee unconditionally"
affects: [35-15]

# Actuals (#2632)
actuals:
  tokens: 3908
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Absence is not agreement: when a guard's authority derives from re-parsing an admin-editable free-text field, a parse failure (None) must route to the REFUSING branch, never the permissive one -- the field's absence is not evidence the underlying fact holds, it is evidence the guard cannot prove anything."
    - "Branching operator remedy text: a duplicate_identity-style reason string branches on which sub-condition produced it (marker differs vs. marker absent), each naming the specific admin action that resolves it, rather than folding both into one generic message."

key-files:
  created: []
  modified:
    - solsys_code/management/commands/cutover_classical_allocations.py
    - solsys_code/tests/test_cutover_classical_allocations.py

key-decisions:
  - "Fixed a pre-existing test fixture regression the CR-01 inversion exposed (out of Task 3's literal scope but required for the plan's own full-module verify gate): TestDryRunAndRealRunAgree's _make_all_three_preconditions_fixture() built its pre-existing CampaignRun with no observation_details Source line: marker at all, so under the inverted guard the whole group was now refused under duplicate_identity before ever reaching the key_collision/window_mismatch preconditions the fixture exists to exercise. Gave the fixture's CampaignRun a matching observation_details marker (Rule 1 auto-fix -- a bug the correct behavior change exposed in an unrelated test, not a design question)."
  - "The no-marker CommandError/reason wording follows 35-REVIEW.md's CR-01 suggested text closely but keeps the existing _REASON_LABELS[_DUPLICATE_IDENTITY] prefix shape both duplicate_identity sites already share (seen_keys guard and the database-scoped guard), per the plan's explicit instruction, rather than adopting 35-REVIEW.md's example verbatim (which drops that shared prefix)."

requirements-completed: [ALLOC-04, ALLOC-05]

coverage:
  - id: D1
    description: "A database claimant of the derived identity key whose observation_details yields NO recoverable Source line: marker is reported under duplicate_identity and the command exits non-zero -- never a find-and-update (CR-01, re-resolving the ALLOC-01 empty edge probe)."
    requirement: ALLOC-05
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_cutover_classical_allocations.TestDatabaseScopedIdentityGuard#test_pre_existing_claimant_with_no_recoverable_source_line_is_refused"
        status: pass
    human_judgment: false
  - id: D2
    description: "After a refused pass over PROBE-A's exact fixture, the claimant's run_status, observation_details and target are byte-identical to their pre-command values, asserted field by field."
    requirement: ALLOC-05
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_cutover_classical_allocations.TestDatabaseScopedIdentityGuard#test_no_marker_claimant_keeps_run_status_details_and_target_byte_identical"
        status: pass
    human_judgment: false
  - id: D3
    description: "The duplicate_identity reason text branches on whether a marker was recovered, naming the Django admin Source line: restore action for the no-marker case."
    requirement: ALLOC-05
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_cutover_classical_allocations.TestDatabaseScopedIdentityGuard#test_pre_existing_claimant_with_no_recoverable_source_line_is_refused (asserts 'Django admin' and 'CampaignRun pk=' in stderr)"
        status: pass
    human_judgment: false
  - id: D4
    description: "A CampaignRun whose stored Source line: MATCHES the group's still converts normally (WR-11's cutover-after-import ordering is not over-rejected)."
    requirement: ALLOC-04
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_cutover_classical_allocations.TestDatabaseScopedIdentityGuard#test_pre_existing_claimant_with_same_source_line_still_converts (unmodified)"
        status: pass
    human_judgment: false
  - id: D5
    description: "No guarantee statement in cutover_classical_allocations.py asserts the no-rewrite property without naming the marker precondition (module docstring and CommandError text both corrected)."
    requirement: ALLOC-05
    verification:
      - kind: other
        ref: "grep -c 'holds on every invocation' and grep -c 'rewrites no existing CampaignRun' both return 0 against the executable module"
        status: pass
    human_judgment: false
  - id: D6
    description: "The redundant existing_run re-query inside the dry-run arm of the group transaction (IN-01) is removed; exactly one CampaignRun.objects.filter(source_identifier=key).first() call remains in the module."
    requirement: ALLOC-05
    verification:
      - kind: other
        ref: "grep -c 'CampaignRun.objects.filter(source_identifier=key).first()' solsys_code/management/commands/cutover_classical_allocations.py == 1"
        status: pass
    human_judgment: false
  - id: D7
    description: "Full phase regression suite (215 pre-existing tests + Task 2's addition) and both pinned ruff/ruff-format gates stay green."
    requirement: ALLOC-05
    verification:
      - kind: integration
        ref: "python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_cutover_classical_allocations solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_observation_projector_signals -> Ran 216 tests, OK; pre-commit run ruff --all-files; pre-commit run ruff-format --all-files"
        status: pass
    human_judgment: false

duration: ~40min
completed: 2026-09-15
status: complete
---

# Phase 35 Plan 12: Invert the Cutover Identity Guard's No-Marker Default Summary

**`cutover_classical_allocations`'s database-scoped identity guard now refuses (never silently find-and-updates) a database claimant whose `observation_details` carries no recoverable `Source line:` marker, closing the Phase 35 BLOCKER (CR-01, 35-REVIEW.md) where an ordinary Django-admin edit to a classical run's staff note silently disarmed the guard and let a one-time destructive migration overwrite every dispatch-deciding field on an APPROVED run.**

## Performance

- **Duration:** ~40 min
- **Completed:** 2026-09-15T16:34:24Z
- **Tasks:** 3 (each independently committed)
- **Files modified:** 2

## Accomplishments

- Closed the Phase 35 BLOCKER (CR-01): the guard's predicate at `cutover_classical_allocations.py` is now a direct inequality (`existing_source_line != source_line`) instead of a permissive membership test against `(None, source_line)`. A database claimant whose stored `Source line:` marker is absent -- reachable via an ordinary Django-admin edit to `observation_details`, an `import_campaign_csv` CSV column, or the campaign submission form's free-text Textarea -- is now reported under `duplicate_identity` and refused, never find-and-updated.
- The `duplicate_identity` reason string now branches on whether a marker was recovered: a claimant with a differing marker keeps the existing "for a different Source line" wording; a claimant with no recoverable marker gets its own wording naming the claimant's pk and directing the operator to restore or correct the run's `observation_details` `Source line:` text in the Django admin, or disambiguate the two lines.
- Replaced (not added beside) `test_pre_existing_claimant_with_no_recoverable_source_line_still_converts` -- the test that pinned the wrong outcome as correct -- with its inverse, `test_pre_existing_claimant_with_no_recoverable_source_line_is_refused`. `TestDatabaseScopedIdentityGuard` still reports `Ran 4 tests` after Task 1 (a clean replacement, not a fifth case).
- Added the destructive-case regression the pre-CR-01 suite never had: `test_no_marker_claimant_keeps_run_status_details_and_target_byte_identical` reproduces 35-REVIEW.md's PROBE-A exactly (a realistic staff note, not the empty-string fixture that made every overwritten field invisible to the old assertions; a real `NonSiderealTargetFactory` target; a colliding group whose line is the `cancelled` counterpart of the claimant's own line) and asserts `run_status`, `observation_details` and `target_id` are byte-identical after a refused pass, with `events re-keyed: 0`. `TestDatabaseScopedIdentityGuard` reports `Ran 5 tests` after Task 2.
- Corrected the module docstring's `duplicate_identity` paragraph and the `CommandError`'s closing clause, both of which previously asserted the no-rewrite guarantee unconditionally ("holds on every invocation" / "rewrites no existing CampaignRun"). Both now name the marker precondition the guarantee actually depends on, and cite CR-01 alongside the existing NF-14/NF-19 citations.
- Folded in IN-01 (advisory): the `if dry_run:` arm of the group transaction no longer re-queries `CampaignRun.objects.filter(source_identifier=key).first()` -- it reuses the `existing_run` already bound by the guard above, removing a redundant query and a name a future reader could mistake for a fresh read.

## Task Commits

1. **Task 1: Invert the identity guard's default and replace the test that pinned the blocker** - `e0f1f66` (fix)
2. **Task 2: Add the destructive-case regression the suite never had** - `478852d` (test)
3. **Task 3: Correct the two in-module guarantee statements and drop the shadowed re-query** - `479c529` (docs)

**Plan metadata:** pending (this commit)

## Files Created/Modified

- `solsys_code/management/commands/cutover_classical_allocations.py` - inverted database-scoped `duplicate_identity` predicate (`!=` instead of `not in (None, ...)`); branching no-marker reason string; rewritten rationale comment naming `admin.py:165`/`import_campaign_csv.py:321`/`campaign_forms.py:65` as the writers that make the field untrusted; corrected module docstring paragraph and `CommandError` closing clause; removed the shadowed `existing_run` re-query in the dry-run arm (IN-01).
- `solsys_code/tests/test_cutover_classical_allocations.py` - replaced the wrong-outcome fourth case in `TestDatabaseScopedIdentityGuard` with its inverse; added the fifth destructive-case regression test; updated the class docstring; fixed `TestDryRunAndRealRunAgree._make_all_three_preconditions_fixture()` to give its pre-existing `CampaignRun` a matching `Source line:` marker (a pre-existing fixture the CR-01 inversion exposed, see Deviations).

## Decisions Made

- The no-marker `CommandError`/reason wording follows 35-REVIEW.md's CR-01 suggested text closely but keeps the existing `_REASON_LABELS[_DUPLICATE_IDENTITY]` prefix shape both `duplicate_identity` sites already share (the `seen_keys` in-process guard and the database-scoped guard), per the plan's explicit instruction, rather than adopting 35-REVIEW.md's illustrative snippet verbatim (which drops that shared prefix).
- IN-01's fix was folded into Task 3 as the plan directed, rather than deferred, since it sits in the same predicate's immediate surroundings and the verifier explicitly recommended folding it in here.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `TestDryRunAndRealRunAgree`'s `_make_all_three_preconditions_fixture()` broke under the CR-01 inversion**
- **Found during:** Task 3 (running the full test module verify gate)
- **Issue:** This fixture (used by `test_dry_run_and_real_run_report_identical_counts_and_reasons` and `test_out_of_window_event_is_byte_identical_after_both_passes`, both pre-existing and out of this plan's stated scope) builds a pre-existing `CampaignRun` directly via `CampaignRun.objects.create()` with no `observation_details` set at all -- so under the CR-01 inversion, `_extract_source_line('')` returns `None`, and the whole group was now refused under `duplicate_identity` before the per-event `key_collision`/`window_mismatch` preconditions the fixture exists to exercise were ever reached. `test_dry_run_and_real_run_report_identical_counts_and_reasons` failed: `AssertionError: 0 != 2` on `events_rekeyed`.
- **Fix:** Gave the fixture's `CampaignRun` an `observation_details` value with a `Source line:` marker matching `_THREE_NIGHT_LINE`, so the guard's own precondition is satisfied and the fixture exercises what it was built to exercise.
- **Files modified:** `solsys_code/tests/test_cutover_classical_allocations.py`
- **Commit:** `479c529`
- **Impact on plan:** None on correctness -- this is exactly the class of regression CR-01's fix is supposed to surface in any fixture that (like real-world admin-edited data) carries no marker. The plan's own `<verification>` step 1 (the five-module regression suite) requires this fixed for the plan to close; without it, `python manage.py test solsys_code.tests.test_cutover_classical_allocations` would not print `OK`.

**Total deviations:** 1 (Rule 1 auto-fix, required for the plan's own verify gate to pass)
**Impact on plan:** No scope creep beyond fixing an exposed pre-existing fixture bug. All plan `<verify>` and `<acceptance_criteria>` gates pass against the final committed tree.

## Issues Encountered

None beyond the fixture regression documented above.

## User Setup Required

None - no external service configuration required.

## Verification (plan-level, all 4 steps run from repo root)

1. `python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_cutover_classical_allocations solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_observation_projector_signals` -> **Ran 216 tests, OK** (215 pre-existing + Task 2's addition; Task 1 replaced rather than added).
2. `pre-commit run ruff --all-files` -> **Passed**.
3. `pre-commit run ruff-format --all-files` -> **Passed**.
4. `git status --short` -> only `solsys_code/management/commands/cutover_classical_allocations.py`, `solsys_code/tests/test_cutover_classical_allocations.py` (this plan's `files_modified`) and pre-existing `.planning/`/untracked scratch artifacts modified; no unexpected file.

## Next Phase Readiness

- CR-01 (BLOCKER) is closed with tests that failed against the pre-fix tree (the replaced test previously pinned the permissive outcome as correct; the new destructive-case regression reproduces PROBE-A and now fails loudly if the no-marker merge ever returns).
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` still prints the OLD `CommandError` text verbatim in its committed output (the unconditional "rewrites no existing CampaignRun" clause this plan corrected) -- this plan deliberately does NOT regenerate it. That regeneration, plus the three runbook passages (`docs/runbooks/telescope_runs_calendar.rst` L938-949, L1497, L1503) that still state the guarantee unconditionally, are explicitly owed to **plan 35-15** (`depends_on` this plan per its objective).
- Plans 35-13, 35-14, 35-15 are untouched by this dispatch, as instructed.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-15*

## Self-Check: PASSED
- FOUND: solsys_code/management/commands/cutover_classical_allocations.py
- FOUND: solsys_code/tests/test_cutover_classical_allocations.py
- FOUND: .planning/phases/35-allocation-layer-classical-cutover/35-12-SUMMARY.md
- FOUND: commit e0f1f66
- FOUND: commit 478852d
- FOUND: commit 479c529
