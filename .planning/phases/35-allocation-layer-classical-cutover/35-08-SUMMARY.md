---
phase: 35-allocation-layer-classical-cutover
plan: 08
subsystem: allocation-layer
tags: [django-management-command, cutover, identity-guard, campaign-run, tdd]

# Dependency graph
requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: cutover_classical_allocations command (plans 35-01..35-07), source_identifier field, _extract_source_line()/_source_identifier() helpers
provides:
  - "cutover_classical_allocations's duplicate_identity guard reads the database (CampaignRun.objects.filter(source_identifier=key)), not just the current process's seen_keys dict"
  - "an actionable duplicate_identity remedy naming the Django admin Source line: edit, replacing the un-actionable schedule-file remedy"
  - "seen_keys[key] claimed only after a group is known convertible, closing the premature-claim gap that mis-blamed a convertible sibling group"
  - "TestDatabaseScopedIdentityGuard: 4 new tests pinning the re-run case, the pre-existing-claimant-on-first-pass case, and the two benign paths (same Source line, no recoverable marker) that must still convert"
affects: [35-09, 35-10, 35-11]

# Actuals (#2632)
actuals:
  tokens: 4836
  tasks: 3
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Database-scoped identity guard: an in-process dict (seen_keys) is not sufficient to protect a find-or-update against a database -- the guard must also query the table the write path matches against, recovering the pre-existing claimant's own identity marker with the same parser used elsewhere, rather than inventing new state."
    - "Claim only when convertible: an in-process 'this key is taken' marker should be set after the last point a group can still be rejected for unrelated reasons (campaign mismatch, unknown status, all-foreign-attributed), not at the moment the key first resolves -- otherwise an unconvertible group poisons a convertible sibling's error message."

key-files:
  created: []
  modified:
    - solsys_code/management/commands/cutover_classical_allocations.py
    - solsys_code/tests/test_cutover_classical_allocations.py

key-decisions:
  - "Split the NF-19 database-scoped guard (Task 1) and the NF-25/IN-02 remedy-text-and-reorder fix (Task 2) into two separate commits per the plan's task boundaries, even though both touch overlapping lines in the same function -- Task 1's database-claimant branch initially reused the pre-existing (soon-to-be-replaced) 'add a bracketed proposal token' remedy text so Task 1's own diff and verify gates stood independently of Task 2's changes."
  - "IN-01's comment fix (test_cutover_classical_allocations.py) was implemented before Task 1's commit boundary and landed inside Task 1's commit rather than a separate Task 3 commit -- Task 3's remaining scope (full-suite + lint verification) was then executed with no further file changes to commit."

requirements-completed: [ALLOC-04, ALLOC-05]

coverage:
  - id: D1
    description: "A second invocation of cutover_classical_allocations over a two-group status-only-collision fixture no longer silently merges the second group into the first group's CampaignRun -- the first group's run_status/observation_details stay byte-identical and the reason stays duplicate_identity, never key_collision (NF-19 case 1, BLOCKER)."
    requirement: ALLOC-05
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_cutover_classical_allocations.TestDatabaseScopedIdentityGuard#test_second_invocation_over_two_group_fixture_does_not_silently_merge"
        status: pass
    human_judgment: false
  - id: D2
    description: "A pre-existing database claimant (from an earlier cutover invocation or from load_telescope_runs) whose stored Source line: differs from the group's is rejected under duplicate_identity on the FIRST pass, with a non-zero exit -- never a silent find-and-update (NF-19 case 2, BLOCKER)."
    requirement: ALLOC-05
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_cutover_classical_allocations.TestDatabaseScopedIdentityGuard#test_pre_existing_claimant_with_different_source_line_rejects_on_first_pass"
        status: pass
    human_judgment: false
  - id: D3
    description: "A pre-existing database claimant whose stored Source line: is the SAME as the group's still converts normally (the cutover-after-import ordering, WR-11), and a claimant with no recoverable Source line: marker at all is treated permissively and also converts (ALLOC-01 empty edge probe)."
    requirement: ALLOC-04
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_cutover_classical_allocations.TestDatabaseScopedIdentityGuard#test_pre_existing_claimant_with_same_source_line_still_converts"
        status: pass
      - kind: unit
        ref: "solsys_code.tests.test_cutover_classical_allocations.TestDatabaseScopedIdentityGuard#test_pre_existing_claimant_with_no_recoverable_source_line_still_converts"
        status: pass
    human_judgment: false
  - id: D4
    description: "The duplicate_identity reason (both branches), the module docstring, and the CommandError text name an action this command can actually carry out (editing the events' description Source line: in the Django admin), never a schedule file this command does not read (NF-25)."
    requirement: ALLOC-05
    verification:
      - kind: other
        ref: "! grep -q 'proposal token to one of the two lines' solsys_code/management/commands/cutover_classical_allocations.py && test $(grep -cF 'disambiguate the two groups in the Django admin' solsys_code/management/commands/cutover_classical_allocations.py) -ge 2"
        status: pass
    human_judgment: false
  - id: D5
    description: "seen_keys[key] is claimed only after the group is known to have something to write (after the if not unattributed_events: gate), so an unconvertible group no longer steals the identity key from a convertible sibling (IN-02)."
    requirement: ALLOC-05
    verification:
      - kind: other
        ref: "python -c source-line-order-check (seen_keys[key] = source_line line index > 'if not unattributed_events:' line index)"
        status: pass
    human_judgment: false
  - id: D6
    description: "No comment in test_cutover_classical_allocations.py cites an absolute line number as the location of an assertion (IN-01)."
    requirement: ALLOC-05
    verification:
      - kind: other
        ref: "grep -nE 'asserted above at line [0-9]+' solsys_code/tests/test_cutover_classical_allocations.py (no match, exit 1)"
        status: pass
    human_judgment: false
  - id: D7
    description: "--dry-run and the real pass still report identical summaries, reason breakdowns and exit status on every existing fixture (NF-02 parity contract preserved)."
    requirement: ALLOC-04
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_cutover_classical_allocations.TestDryRunAndRealRunAgree, TestDuplicateIdentityKeyAcrossGroups#test_dry_run_and_real_run_agree_on_duplicate_identity"
        status: pass
    human_judgment: false
  - id: D8
    description: "Full solsys_code test suite (43 modules, test_views excluded) and both pinned ruff/ruff-format gates stay green after the fix."
    requirement: ALLOC-05
    verification:
      - kind: integration
        ref: "python manage.py test (43 module labels, excludes test_views); pre-commit run ruff --all-files; pre-commit run ruff-format --all-files"
        status: pass
    human_judgment: false

duration: ~30min
completed: 2026-09-15
status: complete
---

# Phase 35 Plan 08: Database-Scoped Identity Guard for the Cutover Command Summary

**`cutover_classical_allocations`'s duplicate-identity guard now reads the database (not just its own in-process dict), closes the NF-19 BLOCKER where the command's own prescribed re-run silently flipped a pre-existing `CampaignRun`'s status, and gives operators an actionable remedy instead of one that names a file the command never reads.**

## Performance

- **Duration:** ~30 min
- **Completed:** 2026-09-15T13:42:00Z
- **Tasks:** 3 (Tasks 1-2 each independently committed; Task 3's file change landed inside Task 1's commit, see Deviations)
- **Files modified:** 2

## Accomplishments
- Closed the Phase 35 BLOCKER (NF-19): a second cutover invocation over a colliding two-group fixture -- the exact re-run the command's own `CommandError` prescribes -- no longer silently merges the second group into the first group's `CampaignRun`. A pre-existing database claimant (from an earlier cutover run or from `load_telescope_runs`) is now also rejected on the FIRST pass when its stored `Source line:` differs, instead of exiting zero.
- Closed NF-25: the `duplicate_identity` reason, the module docstring, and the `CommandError` text now tell the operator to edit the affected events' description `Source line:` text in the Django admin -- an action this command can actually verify, since it reads no schedule file.
- Closed IN-02: `seen_keys[key]` is now claimed only after a group is confirmed to have something writable, so an unconvertible group (campaign mismatch, unknown status, all-events-foreign) no longer poisons a convertible sibling's error message with the wrong "already claimed" line.
- Closed IN-01: the test comment documenting the removed vacuous `1 + 1 == 2` assertion now cites assertions by identifier (`rekeyed_event.url`, `delete_legacy_pk`) instead of absolute line numbers.
- Preserved the two benign paths the fix must not break: a database claimant whose stored `Source line:` matches the group's still converts (WR-11's cutover-after-import ordering), and a claimant with no recoverable `Source line:` marker at all is treated permissively (ALLOC-01 `empty` edge probe) and also converts.

## Task Commits

1. **Task 1: Make the cutover's identity-key guard read the database, end to end** - `868caa6` (fix)
2. **Task 2: Make the duplicate_identity remedy actionable and claim the key only when the group is convertible** - `9e555eb` (fix)
3. **Task 3: Drop the line-number-citing test comment and re-green the full suite** - no separate commit (see Deviations); file change already present in `868caa6`, remaining scope was verification-only and passed clean.

**Plan metadata:** pending (this commit)

## Files Created/Modified
- `solsys_code/management/commands/cutover_classical_allocations.py` - database-scoped `duplicate_identity` guard (queries `CampaignRun.objects.filter(source_identifier=key)`, recovers the claimant's own stored `Source line:` via `_extract_source_line()`); actionable remedy text in both branches, the module docstring, and the `CommandError`; `seen_keys[key]` claim moved to after the group's convertibility gate.
- `solsys_code/tests/test_cutover_classical_allocations.py` - new `TestDatabaseScopedIdentityGuard` class (4 tests); `_ALLOCATION_LINE`/`_CANCELLED_LINE`/`_make_two_groups_same_identity_key_fixture()` lifted onto the shared `CutoverClassicalAllocationsTestBase` (no duplicated fixture body); NF-11 replacement comment rewritten to cite assertions by identifier, not line number.

## Decisions Made
- Task 1's database-claimant branch initially reused the pre-existing "add a bracketed proposal token to one of the two lines" remedy text (mirroring the in-run branch's original wording at the time), so Task 1's commit and its own verify gates (the 4 new tests, the full test module, the `_extract_source_line` grep) stood independently of Task 2's remedy-text rewrite. Task 2 then replaced both branches' remedy text, the docstring, and the `CommandError` in one coherent commit, matching the plan's stated task boundary ("Task 2 introduces" the shared remedy phrase, "Apply the same replacement to the database-claimant message Task 1 added").
- IN-01's test-comment fix was made before Task 1's commit boundary (while reading and understanding the fixture to write the new tests) and ended up committed as part of Task 1 rather than a standalone Task 3 commit. Task 3's remaining, distinct scope -- running the full Django suite and both pinned ruff gates to prove Tasks 1-2 broke nothing outside the cutover module -- was executed afterward with no further file changes, so no new commit was needed for Task 3.

## Deviations from Plan

**1. [Process, non-functional] Task 3's file-level change (IN-01 comment fix) landed inside Task 1's commit instead of its own commit**
- **Found during:** Task 3
- **Issue:** The plan structures Task 1 (tests + guard), Task 2 (remedy text + reorder), and Task 3 (comment fix + full-suite verification) as three separately-committed units. The IN-01 comment rewrite was made while building the new `TestDatabaseScopedIdentityGuard` test class (before Task 1's commit), so it was staged and committed together with Task 1's other test-file changes.
- **Fix:** No code fix needed -- the comment content is correct and matches the plan's `<action>` for Task 3 verbatim. Task 3's remaining, genuinely separate scope (full Django suite + both pinned ruff/ruff-format gates) was executed as its own verification pass after Task 2's commit, and passed clean with no further diff to commit.
- **Files modified:** `solsys_code/tests/test_cutover_classical_allocations.py` (already included in commit `868caa6`).
- **Impact on plan:** None on correctness or on the must-haves; all of Task 3's `<verify>` gates pass against the final tree. Purely a commit-grouping deviation, noted for traceability.

---

**Total deviations:** 1 (process-only, no functional impact)
**Impact on plan:** No scope creep, no correctness impact. All plan `<verify>` and `<acceptance_criteria>` gates pass against the final committed tree.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The Phase 35 BLOCKER (NF-19) is closed with tests that failed against the pre-fix tree; the phase's own review findings NF-25, IN-01, and IN-02 are also closed.
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` and `docs/runbooks/telescope_runs_calendar.rst` still owe the regeneration/correction this plan's objective explicitly deferred to **35-10-PLAN.md Tasks 1 and 2** (`depends_on` this plan) -- they are regenerated once, after 35-09's projector fixes also land, not twice. Not part of this plan's `files_modified`.
- Plans 35-09, 35-10, 35-11 are untouched by this dispatch, as instructed.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-15*

## Self-Check: PASSED
- FOUND: solsys_code/management/commands/cutover_classical_allocations.py
- FOUND: solsys_code/tests/test_cutover_classical_allocations.py
- FOUND: .planning/phases/35-allocation-layer-classical-cutover/35-08-SUMMARY.md
- FOUND: commit 868caa6
- FOUND: commit 9e555eb
