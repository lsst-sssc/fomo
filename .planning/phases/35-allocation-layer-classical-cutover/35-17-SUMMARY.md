---
phase: 35-allocation-layer-classical-cutover
plan: 17
subsystem: allocation-layer
tags: [django-management-command, cutover, loader, documentation, gap-closure, dry-run-parity]

# Dependency graph
requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: "35-12's inverted database-scoped duplicate_identity guard (existing_source_line != source_line predicate), which this plan documents but does not modify"
  - phase: 35-allocation-layer-classical-cutover
    provides: "35-14's counter-fold fix for the existing-run dry-run/real-run parity, which this plan scopes narrower rather than extends"
provides:
  - "load_telescope_runs's create-arm (existing is None) dry-run limitation stated in code (comment, --dry-run help text) and pinned by a test, citing PROBE-P5's exact five-tuples"
  - "cutover_classical_allocations's _REASON_LABELS[_DUPLICATE_IDENTITY] naming both causes (group-vs-group collision, and a database claimant already holding the identity key) behind a shared 'already holds the derived identity key' anchor"
  - "cutover_classical_allocations's module docstring re-run gotcha paragraph documenting the matching-marker overwrite, and the no-marker remedy's added consequence clause"
  - "a pinning test for PROBE-P4's intended matching-marker-overwrite outcome, distinct from the sibling refusal-case tests"
affects: [35-18]

# Actuals (#2632)
actuals:
  tokens: 3327
  tasks: 2
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Narrow the claim, don't extend the fix: when a parity/guarantee statement in code is true for one code path and false for a sibling path, state the predicate (which arm/branch has the property) in the code itself, rather than either softening the claim with a hedge word or adding a new mechanism to make the false path true."
    - "A reason label prefixed onto multiple distinct causes must name every cause it can be printed for, behind a shared greppable anchor phrase, so downstream documentation (runbook, notebooks) can be gated on the same phrase."

key-files:
  created: []
  modified:
    - solsys_code/management/commands/load_telescope_runs.py
    - solsys_code/tests/test_load_telescope_runs.py
    - solsys_code/management/commands/cutover_classical_allocations.py
    - solsys_code/tests/test_cutover_classical_allocations.py

key-decisions:
  - "Kept the docstring's 'does not survive the next cutover run' sentence on a single physical line (splitting the paragraph earlier) rather than letting it wrap across two lines -- the plan's own verify gate greps for that exact phrase with a single-line grep, and Python triple-quoted docstrings are not rewrapped by ruff-format."
  - "Did not use the literal string 'transaction.set_rollback' anywhere in the create-arm comment (referred to the rejected alternative as 'a transient, rolled-back CampaignRun row' instead) because the plan's own verify gate greps for that literal string and requires a zero count."

requirements-completed: [ALLOC-04, ALLOC-05]

coverage:
  - id: T1
    description: "The loader's create-arm limitation is stated in the code (comment at the fold, --dry-run help text) with PROBE-P5 named, and pinned by a test asserting the preview/real divergence over the no-existing-run fixture, with CampaignRun.objects.count() == 0 afterwards."
    requirement: ALLOC-04
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_load_telescope_runs.TestMalformedTimezoneSkipsOneLine#test_dry_run_of_a_brand_new_line_cannot_predict_a_reconcile_failure"
        status: pass
    human_judgment: false
  - id: T2
    description: "No new preview write path, transient row or rollback probe exists in load_telescope_runs; no hedge word was introduced; the phrase 'create arm' is greppable."
    requirement: ALLOC-04
    verification:
      - kind: other
        ref: "grep -c 'transaction.set_rollback' == 0; grep -ciE 'usually|in normal operation|in most cases' == 0; grep -c 'create arm' >= 1; grep -c PROBE-P5 >= 1"
        status: pass
    human_judgment: false
  - id: T3
    description: "_REASON_LABELS[_DUPLICATE_IDENTITY] names both causes behind the shared 'already holds the derived identity key' anchor."
    requirement: ALLOC-05
    verification:
      - kind: other
        ref: "grep -c 'already holds the derived identity key' solsys_code/management/commands/cutover_classical_allocations.py == 1"
        status: pass
    human_judgment: false
  - id: T4
    description: "The cutover module docstring carries a re-run gotcha naming every re-applied field and the phrase 'does not survive the next cutover run'; the no-marker remedy names the consequence of restoring a marker via 're-applied from the schedule line'."
    requirement: ALLOC-05
    verification:
      - kind: other
        ref: "grep -c 'does not survive the next cutover run' == 1; grep -c 're-applied from the schedule line' == 2"
        status: pass
    human_judgment: false
  - id: T5
    description: "PROBE-P4's outcome (a matching-marker claimant's staff-edited run_status reverted to the schedule line's value) is pinned as intended by a sixth test in TestDatabaseScopedIdentityGuard; CR-01's guard predicate (existing_source_line != source_line) is unchanged, count 1."
    requirement: ALLOC-05
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_cutover_classical_allocations.TestDatabaseScopedIdentityGuard#test_matching_marker_claimant_has_its_staff_edited_fields_reapplied_from_the_line"
        status: pass
    human_judgment: false
  - id: T6
    description: "The six-module test surface (237 tests), ruff and ruff-format all pass; git status shows no modified file beyond the four in files_modified plus pre-existing .planning/ state."
    requirement: ALLOC-05
    verification:
      - kind: integration
        ref: "python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_cutover_classical_allocations solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_observation_projector_signals -> Ran 237 tests, OK; pre-commit run ruff --all-files; pre-commit run ruff-format --all-files"
        status: pass
    human_judgment: false

duration: ~35min
completed: 2026-09-15
status: complete
---

# Phase 35 Plan 17: Third Gap-Closure Round -- Narrow the Loader Claim, Document the Cutover's Re-run Gotcha Summary

**Two independent, subtractive corrections: `load_telescope_runs --dry-run`'s create arm now states (and a test pins) that it cannot predict a reconcile failure, and `cutover_classical_allocations` now names both causes of `duplicate_identity` and documents the matching-marker overwrite as an intended, already-accepted re-run gotcha rather than a silent surprise.**

## Performance

- **Duration:** ~35 min
- **Completed:** 2026-09-15
- **Tasks:** 2 (each independently committed)
- **Files modified:** 4

## Accomplishments

- **Task 1 (loader create-arm claim, PROBE-P5):** Extended the create-arm comment (`existing is None` branch, no preview `reconcile_run()` call) to state, in place, that nothing on that arm can fail because no preview reconcile runs there, while the real branch's `write_and_reconcile_campaign_run()` call for the same line can raise and report the line under `skipped`. Named PROBE-P5 and quoted its two summary lines. Scoped the counter-fold comment's ordering claim to the `existing is not None` arm only -- the arm that has a preview reconcile to order against. Extended the `--dry-run` help text with the run-level consequence. Added `test_dry_run_of_a_brand_new_line_cannot_predict_a_reconcile_failure` to `TestMalformedTimezoneSkipsOneLine`, using the class fixture as seeded (NTT's typo'd timezone, no pre-existing `CampaignRun`) without repairing it first, and asserting PROBE-P5's exact five-tuples: dry `(1, 1, 0, 0, 0)` against real `(1, 0, 0, 0, 1)`, with `CampaignRun.objects.count() == 0` afterward.
- **Task 2 (cutover reason vocabulary + re-run gotcha, PROBE-P4):** Rewrote `_REASON_LABELS[_DUPLICATE_IDENTITY]` to name both causes -- the original group-vs-group collision, and a database claimant that already holds the derived identity key and cannot be proved to have come from the line in hand -- behind the shared, greppable phrase `already holds the derived identity key`. Added a "Re-run gotcha" paragraph to the module docstring immediately after the reason-enumeration paragraph, naming every field a matching-marker claimant has re-applied on every invocation (`source`, `approval_status`, `run_status`, `target`, `campaign`, `site`/`site_raw`, `window_start`/`window_end`, the two sub-night fields, `observation_details`), stating this is the project's already-accepted file-authoritative semantics shared with `load_telescope_runs` and `import_campaign_csv`'s documented "Re-import gotcha," and closing the loop the no-marker remedy opens. Extended the no-marker branch's `reason` string with a clause (verbatim phrase `re-applied from the schedule line`) naming that consequence in the same sentence that sends the operator to the Django admin. Added `test_matching_marker_claimant_has_its_staff_edited_fields_reapplied_from_the_line` to `TestDatabaseScopedIdentityGuard`, reproducing PROBE-P4 (matching-marker claimant, staff sets `run_status` to `CANCELLED`, re-run reverts it to `PLANNED` with exit 0, `updated: 1`, `unexplained: 0`).
- Neither task changed any predicate, branch or write path. CR-01's guard predicate (`existing_source_line != source_line`) remains at exactly one occurrence, verified by the plan's own grep gate.

## Task Commits

1. **Task 1: Say what the loader's preview can and cannot predict, and pin it** - `1686a63` (docs)
2. **Task 2: Two causes in the reason label, and the cutover's missing re-run gotcha** - `73a2149` (docs)

**Plan metadata:** pending (this commit)

## Files Created/Modified

- `solsys_code/management/commands/load_telescope_runs.py` -- extended create-arm comment naming PROBE-P5 and the rejected transient-row alternative; scoped the counter-fold comment to the `existing is not None` arm; extended `--dry-run` help text with the run-level-decision limitation.
- `solsys_code/tests/test_load_telescope_runs.py` -- added `test_dry_run_of_a_brand_new_line_cannot_predict_a_reconcile_failure` to `TestMalformedTimezoneSkipsOneLine`, pinning PROBE-P5's exact five-tuples.
- `solsys_code/management/commands/cutover_classical_allocations.py` -- two-cause `_REASON_LABELS[_DUPLICATE_IDENTITY]`; new module-docstring re-run gotcha paragraph; extended no-marker `reason` string clause.
- `solsys_code/tests/test_cutover_classical_allocations.py` -- added `test_matching_marker_claimant_has_its_staff_edited_fields_reapplied_from_the_line` to `TestDatabaseScopedIdentityGuard` (now 6 tests), pinning PROBE-P4's outcome as intended.

## Decisions Made

- Split the re-run gotcha paragraph's sentence boundary so "does not survive the next cutover run" lands entirely on one physical source line -- the plan's own verify gate is a single-line `grep`, and a Python triple-quoted docstring is not rewrapped by `ruff-format`, so a phrase split across a line wrap would silently fail the gate without any tool flagging it.
- Described the rejected preview-write-path alternative in prose ("a transient, rolled-back CampaignRun row") rather than naming the literal API (`transaction.set_rollback`), since the plan's verify gate greps for that literal string and requires zero occurrences -- using it even in a "we rejected this" comment would have tripped the same gate meant to catch it being *implemented*.

## Deviations from Plan

None - plan executed exactly as written. Both tasks' code changes, test additions, and verify-gate greps matched the plan's `<action>` and `<verify>` sections on first attempt, aside from the one wrapping fix noted above (caught by the plan's own gate before commit, not a post-hoc correction).

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Verification (plan-level, all 4 steps run from repo root)

1. `python manage.py test solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_cutover_classical_allocations` -> **Ran 64 tests, OK**.
2. `python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_cutover_classical_allocations solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_observation_projector_signals` -> **Ran 237 tests, OK**.
3. `pre-commit run ruff --all-files` -> **Passed**. `pre-commit run ruff-format --all-files` -> **Passed**.
4. `git status --short` -> only `.planning/config.json`, `.planning/state.json` (pre-existing at session start) and pre-existing untracked scratch artifacts modified beyond this plan's four `files_modified`; `docs/`, `src/fomo_db.sqlite3` and `solsys_code/allocation_projector.py` (plan 35-16's file) are clean.

## Next Phase Readiness

- This plan's two code-side corrections (loader claim narrowing, cutover reason label + re-run gotcha) are landed and pinned by tests. Plan 35-18 mirrors both into `docs/runbooks/telescope_runs_calendar.rst` (the reason vocabulary's two other definition sites at L933-937/L1498-1500, and a caveat matching this plan's docstring paragraph) and regenerates the two notebooks that quote the loader's parity claim and the cutover's reason vocabulary.
- Plans 35-16 (already complete on this branch) and 35-18 (separate dispatch) are untouched by this plan, as instructed. `solsys_code/allocation_projector.py` remains clean.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-15*

## Self-Check: PASSED
- FOUND: solsys_code/management/commands/load_telescope_runs.py
- FOUND: solsys_code/tests/test_load_telescope_runs.py
- FOUND: solsys_code/management/commands/cutover_classical_allocations.py
- FOUND: solsys_code/tests/test_cutover_classical_allocations.py
- FOUND: .planning/phases/35-allocation-layer-classical-cutover/35-17-SUMMARY.md
- FOUND: commit 1686a63
- FOUND: commit 73a2149
