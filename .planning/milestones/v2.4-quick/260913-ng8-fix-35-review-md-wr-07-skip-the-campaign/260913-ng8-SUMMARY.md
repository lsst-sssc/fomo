---
phase: quick-260913-ng8
plan: 01
subsystem: campaign-runs
tags: [django, cutover, campaign-run, regression-test, wr-07]

requires:
  - phase: 35 (Allocation Layer & Classical Cutover)
    provides: cutover_classical_allocations management command, D-17/D-18 conversion contract
provides:
  - WR-07 guard skipping the CampaignRun write for a group with zero writable events
  - Regression test class TestAllForeignAttributedGroupWritesNothing (real path + --dry-run)
  - Module docstring and operator runbook sentence documenting the new rule
affects: [cutover_classical_allocations, reconcile_campaign_runs, 35-REVIEW.md WR-07]

actuals:
  tokens: 1785
  tasks: 3
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Guard-then-continue before a group's transaction.atomic() block, placed after
      per-event _mark_unexplained() calls so reporting and non-zero exit are preserved
      while the write itself is skipped"

key-files:
  created: []
  modified:
    - solsys_code/management/commands/cutover_classical_allocations.py
    - solsys_code/tests/test_cutover_classical_allocations.py
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "Guard placed immediately after the writable_events-building loop and before the
    WR-06 transaction.atomic() block, so it governs the real and --dry-run paths
    identically without duplicating the check inside the dry_run branch"
  - "Paired-notebook audit: reconcile_campaign_runs_demo.ipynb's three cutover cells only
    ever exercise the no_source_line reason (pk=334 junk row) -- no foreign_attribution
    case appears in its committed output, so no re-execution was required"

patterns-established:
  - "WR-07 guard: 'no writable_events -> continue before the group write' -- future
    per-group guards in this command should follow the same ordering relative to
    _mark_unexplained() and the transaction.atomic() block"

requirements-completed: [WR-07]

coverage:
  - id: D1
    description: "A Source line: group whose every event is foreign-attributed produces no CampaignRun write on the real path (count unchanged, per-event reporting preserved, non-zero exit)"
    requirement: "WR-07"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_cutover_classical_allocations.py#TestAllForeignAttributedGroupWritesNothing.test_all_foreign_attributed_group_creates_no_run"
        status: pass
    human_judgment: false
  - id: D2
    description: "The same all-foreign group produces no CampaignRun write on --dry-run, matching the real path's runs created: 0"
    requirement: "WR-07"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_cutover_classical_allocations.py#TestAllForeignAttributedGroupWritesNothing.test_all_foreign_attributed_group_dry_run_predicts_no_run"
        status: pass
    human_judgment: false
  - id: D3
    description: "Groups with at least one writable event are unaffected -- TestForeignAttributionLeftUntouched and the full pre-existing test module stay green"
    requirement: "WR-07"
    verification:
      - kind: unit
        ref: "python manage.py test solsys_code.tests.test_cutover_classical_allocations"
        status: pass
    human_judgment: false
  - id: D4
    description: "Module docstring and operator runbook both state the no-run-for-an-all-foreign-group rule; paired-notebook audit recorded"
    verification:
      - kind: other
        ref: "grep -c 'runs created: 0' docs/runbooks/telescope_runs_calendar.rst; python -c docstring substring check"
        status: pass
    human_judgment: false

duration: ~20min
completed: 2026-09-13
status: complete
---

# Quick Task 260913-ng8 Summary

**Closed 35-REVIEW.md WR-07: `cutover_classical_allocations` now skips the `CampaignRun` write entirely for a `Source line:` group whose every event is already attributed to a different run, so no ownerless run reaches the next `reconcile_campaign_runs` sweep.**

## Performance

- **Duration:** ~20 min
- **Tasks:** 3
- **Files modified:** 3 (`cutover_classical_allocations.py`, `test_cutover_classical_allocations.py`, `telescope_runs_calendar.rst`)

## Accomplishments

- Added a WR-07 guard in `Command.handle` that `continue`s past a group's run write when `writable_events` is empty — placed after the per-event `_mark_unexplained()` loop and before the WR-06 `transaction.atomic()` block, so it governs the real and `--dry-run` paths identically with no duplicated check.
- Added regression test class `TestAllForeignAttributedGroupWritesNothing` (two tests: real path, `--dry-run`), confirmed RED against pre-fix code (`CampaignRun.objects.count()` went from an expected 1 to an observed 2; `runs created: 0` assertion failed against an observed `runs created: 1`) before writing the fix.
- Documented the rule once in the module docstring (`cutover_classical_allocations.py`'s unexplained-reasons paragraph) and once in the operator runbook (`docs/runbooks/telescope_runs_calendar.rst`'s cutover section), in each file's existing prose style.
- Audited the paired notebook (`docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`) per CLAUDE.md's paired-docs rule and confirmed no re-execution is required (see below).

## Task Commits

1. **Task 1: Skip the run write when a group has nothing writable, pinned by a regression test** — `9ce07bd` (fix)
2. **Task 2: State the no-run-for-an-all-foreign-group rule in the docstring and the runbook, and audit the paired notebook** — `bb20c2e` (docs)
3. **Task 3: Run the project's quality gates on the changed files** — no new commit; `pre-commit run ruff` and `pre-commit run ruff-format` were already clean on both changed Python files (no reformatting needed), and the targeted test module passed on the final tree, so there was nothing further to commit.

**Plan metadata:** commit handled by the orchestrator after this SUMMARY.

## Files Created/Modified

- `solsys_code/management/commands/cutover_classical_allocations.py` — added the WR-07 guard (`if not writable_events: continue`) ahead of the group's `transaction.atomic()` block, with a comment citing WR-07/35-REVIEW.md; extended the module docstring's unexplained-reasons paragraph with the same rule.
- `solsys_code/tests/test_cutover_classical_allocations.py` — added `TestAllForeignAttributedGroupWritesNothing` with two tests (real path, `--dry-run`) over a fixture of two blank-url events sharing `_THREE_NIGHT_LINE`, both pre-attributed to `other_run`.
- `docs/runbooks/telescope_runs_calendar.rst` — added one sentence to the cutover section's unexplained-list paragraph stating that an all-foreign group gets no run created or updated, so `runs created: 0` next to a `foreign_attribution` count is the designed outcome.

## Decisions Made

- Guard placed strictly after the `writable_events`-building loop (which has already called `_mark_unexplained()` on every foreign event) and strictly before the WR-06 `transaction.atomic()` block — this is what preserves per-event reporting/non-zero-exit while skipping only the write, and what makes the guard apply identically to the real and `--dry-run` paths without any `if dry_run:` duplication.
- Paired-notebook audit result: **no re-execution required.** Read `reconcile_campaign_runs_demo.ipynb`'s three `cutover_classical_allocations` references (one markdown cell describing the demo, two code cells calling the command with `--dry-run` and for real). Both code cells' committed output shows `candidates: 10, groups: 3, runs created: 3, ..., unexplained: 1` with the single unexplained reason being `no_source_line` (pk=334, a pre-existing junk `tmp` row with no recoverable `Source line:`). No `foreign_attribution` reason appears anywhere in the notebook's committed output, so the WR-07 guard — which only changes behavior for groups where `foreign_attribution` accounts for every event in the group — cannot alter what this notebook already shows. Confirmed by inspecting both cells' `outputs` directly (not by re-running nbconvert).

## Deviations from Plan

None — plan executed exactly as written. The guard, test class, docstring sentence, and runbook sentence all match the plan's `<behavior>`/`<action>` specifications; Task 3 required no fixes because both pinned ruff hooks were already clean on the changed files.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- WR-07 is closed: a `Source line:` group with all events foreign-attributed no longer creates a zero-event `CampaignRun`, closing the blast-radius risk of the next `reconcile_campaign_runs` sweep minting a duplicate `ALLOC:` night set over a foreign run's nights.
- `solsys_code.tests.test_cutover_classical_allocations` passes in full (19 tests), including all 7 pre-existing bonus-coverage classes (`TestForeignAttributionLeftUntouched`, `TestDryRunMatchesRealRun`, `TestCutoverSequenceContract`, `TestGroupTransactionBoundary`, `TestWindowContainmentGuard`, `TestUnknownClassicalStatusGuard`, and the 8 plan-named classes) — no pre-existing behavior changed.
- Both pinned pre-commit ruff hooks (`ruff`, `ruff-format`) are clean on the two changed Python files.
- No other module in the repo depends on `cutover_classical_allocations` at runtime — the only cross-reference is a comment in `load_telescope_runs.py` (WR-09) noting the two modules share the same status dict; confirmed unaffected by this change.

---
*Phase: quick-260913-ng8*
*Completed: 2026-09-13*

## Self-Check: PASSED

All created/modified files exist on disk and both task commits (`9ce07bd`, `bb20c2e`) are present in `git log`.
