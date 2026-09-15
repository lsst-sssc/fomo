---
phase: 35-allocation-layer-classical-cutover
plan: 14
subsystem: allocation-layer
tags: [django-management-command, dry-run, tdd, jupyter-notebook]

# Dependency graph
requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: load_telescope_runs allocation cutover with NF-21's per-line ZoneInfoNotFoundError handler (35-09), the loader demo notebook's "Per-line skip paths" section (35-11)
provides:
  - "load_telescope_runs.py's `if dry_run:` branch folds run_created/run_updated/run_unchanged only AFTER both preview_campaign_run_action() and reconcile_run(existing, dry_run=True) have returned, mirroring the real branch -- a line whose preview reconcile raises is now reported under `skipped` alone on both the dry-run and the real pass (WR-02)"
  - "TestMalformedTimezoneSkipsOneLine now also pins the dry/real counter parity invariant and the dry-run pass's skip-and-continue behaviour, not only the real pass"
  - "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb's 'Per-line skip paths' section gained an executed dry/real parity cell (16/16 code cells, 0 null execution counts)"
affects: [35-15]

# Actuals (#2632)
actuals:
  tokens: 6906
  tasks: 3
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fold-after-both-calls counter ordering: when a dry-run preview short-circuit has two sequential steps (an action classification, then a call that can raise), fold the run-level counter from the FIRST step only after the SECOND step has also returned -- otherwise a raising second step double-counts the line into both its would-be outcome bucket and the exception handler's skip bucket. Mirrors the real branch's transaction.atomic()-gated fold exactly."

key-files:
  created: []
  modified:
    - solsys_code/management/commands/load_telescope_runs.py
    - solsys_code/tests/test_load_telescope_runs.py
    - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb

key-decisions:
  - "Task 1 and Task 2 landed in the same commit (909a67a): the plan's Task 2 action item explicitly allows adding its assertions 'to the new parity test from Task 1, or as a small sibling method -- either is acceptable', and both tasks touch the same file (test_load_telescope_runs.py) with no independently-verifiable intermediate state. Wrote both the parity test and the skip-and-continue sibling test together, plus the combined class-docstring update naming both invariants (NF-21 and WR-02), and committed once."
  - "The plan's Task 1 fixture design (a schedule line whose CampaignRun already exists, timezone mistyped AFTER creation) required fixing the class's shared setUpTestData: it seeds NTT with the malformed timezone from the very start (the NF-21 skip-fixture), so the WR-02 parity test first repairs the timezone to a valid IANA zone, creates the CampaignRun with a real pass, THEN mutates it to the typo before the dry-run/real comparison -- matching 35-REVIEW.md PROBE-B's exact reproduction sequence."
  - "The notebook's new parity cell reuses the pre-existing NTT/EFOSC2 CampaignRun created earlier in the notebook (from the 'NTT EFOSC2 allocation 9-13 July' line loaded near the top) rather than inventing a new fixture, per the plan's instruction to reuse the section's existing fixture machinery."

requirements-completed: [ALLOC-04]

coverage:
  - id: D1
    description: "load_telescope_runs --dry-run and the immediately following real pass report the SAME (created, updated, unchanged, skipped) tuple for a line whose own preview reconcile_run(existing, dry_run=True) call raises -- closing WR-02's exact PROBE-B reproduction ((0, 0, 1, 1) dry vs (0, 0, 0, 1) real, now (0, 0, 0, 1) on both)."
    requirement: ALLOC-04
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_load_telescope_runs.TestMalformedTimezoneSkipsOneLine#test_dry_run_and_real_run_report_the_same_counters_for_a_skipped_line"
        status: pass
    human_judgment: false
  - id: D2
    description: "created + updated + unchanged + skipped equals lines processed on both the dry-run and real pass, for the WR-02 fixture and for the pre-existing multi-line NF-21 fixture."
    requirement: ALLOC-04
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_load_telescope_runs.TestMalformedTimezoneSkipsOneLine#test_dry_run_and_real_run_report_the_same_counters_for_a_skipped_line"
        status: pass
    human_judgment: false
  - id: D3
    description: "NF-21's skip-and-continue invariant (a bad line never aborts the whole import, the following line is still processed) now also holds on the DRY-RUN pass, which creates no CampaignRun row at all."
    requirement: ALLOC-04
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_load_telescope_runs.TestMalformedTimezoneSkipsOneLine#test_dry_run_pass_leaves_no_campaign_run_and_still_processes_the_next_line"
        status: pass
    human_judgment: false
  - id: D4
    description: "test_malformed_timezone_skips_only_its_own_line (the original NF-21 regression) still exists, unmodified, and still passes."
    requirement: ALLOC-04
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_load_telescope_runs.TestMalformedTimezoneSkipsOneLine#test_malformed_timezone_skips_only_its_own_line"
        status: pass
    human_judgment: false
  - id: D5
    description: "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb's 'Per-line skip paths' section demonstrates the dry/real parity with real executed output -- both summary lines printed under the label 'dry/real counter parity (WR-02):', every code cell carrying a non-null execution count, developer database untouched."
    requirement: ALLOC-04
    verification:
      - kind: other
        ref: "python3 JSON scan: code cells: 16, null counts: 0; grep -c 'dry/real counter parity (WR-02)' >= 2; grep -c 'Done (dry run). lines processed' >= 2; git status --short src/fomo_db.sqlite3 empty"
        status: pass
    human_judgment: false
  - id: D6
    description: "Full 5-module regression suite (allocation_projector, campaign_reconciler, cutover_classical_allocations, load_telescope_runs, observation_projector_signals) and both pinned ruff/ruff-format gates stay green after all three tasks."
    requirement: ALLOC-04
    verification:
      - kind: integration
        ref: "python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_cutover_classical_allocations solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_observation_projector_signals (219 tests, OK); pre-commit run ruff --all-files; pre-commit run ruff-format --all-files"
        status: pass
    human_judgment: false

duration: ~35min
completed: 2026-09-15
status: complete
---

# Phase 35 Plan 14: Loader Dry-Run/Real-Run Counter Parity (WR-02) Summary

**`load_telescope_runs --dry-run`'s run-level counters now fold only after the preview reconcile has returned, so a line whose preview raises is reported under `skipped` alone on both the preview and the real pass -- closing WR-02, the regression 35-09's own NF-21 fix introduced inside the same gap-closure round.**

## Performance

- **Duration:** ~35 min
- **Completed:** 2026-09-15T16:53:01Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- **WR-02 closed:** `load_telescope_runs.py`'s `if dry_run:` branch now computes `action = preview_campaign_run_action(existing, fields)`, runs the `existing is not None` / `else` reconcile arms exactly as before, and folds `run_created`/`run_updated`/`run_unchanged` from `action` only AFTER both calls have returned -- mirroring the real branch, which increments only after its `transaction.atomic()` block returns. A line whose preview `reconcile_run(existing, dry_run=True)` raises is now counted once, under `skipped`, on both passes.
- `TestMalformedTimezoneSkipsOneLine` gained `test_dry_run_and_real_run_report_the_same_counters_for_a_skipped_line`, which failed against the pre-fix tree with PROBE-B's exact divergence (`(0, 0, 1, 1) != (0, 0, 0, 1)`) and now asserts both that the dry/real tuples are equal and that `created + updated + unchanged + skipped == lines_processed` on both passes.
- `TestMalformedTimezoneSkipsOneLine` also gained `test_dry_run_pass_leaves_no_campaign_run_and_still_processes_the_next_line`, extending NF-21's skip-and-continue invariant (previously real-path only) to the dry-run pass: a preview never writes a `CampaignRun`, and a bad line never stops the following line from being previewed. The class docstring now states both invariants it pins and explains why WR-02's own coverage gap (real-path-only) is how it survived the NF-21 fix.
- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`'s existing "Per-line skip paths" section gained one markdown + code cell pair labelled `dry/real counter parity (WR-02):`, reusing the notebook's pre-existing NTT/EFOSC2 `CampaignRun` and NTT's temporarily-mutated malformed timezone. The executed output shows both summary lines now agreeing: `Done (dry run). lines processed: 1, created: 0, updated: 0, unchanged: 0, skipped: 1` and `Done. lines processed: 1, created: 0, updated: 0, unchanged: 0, skipped: 1`. Regenerated whole via `jupyter nbconvert --to notebook --execute --inplace` against the notebook's existing scratch-database routing; 16/16 code cells have non-null execution counts; `src/fomo_db.sqlite3` untouched.
- Full 5-module regression suite (219 tests) and both pinned `ruff`/`ruff-format` gates pass clean after all three tasks.

## Task Commits

1. **Task 1 + Task 2: Fold the dry-run counters only after the preview reconcile has returned; confirm NF-21's skip-and-continue invariant on the dry-run pass** - `909a67a` (fix)
2. **Task 3: Extend and regenerate the paired loader demo notebook** - `9bb2b64` (docs)

**Plan metadata:** pending (this commit)

_TDD note: Task 1's regression test was confirmed to fail against the pre-fix tree with the exact `(0, 0, 1, 1) != (0, 0, 0, 1)` divergence PROBE-B reproduced (RED), before the production fix was applied (GREEN); both landed in the same commit per this plan's per-task-not-per-RED-GREEN-commit convention, matching 35-09's own precedent for closely-coupled test+fix pairs._

## Files Created/Modified
- `solsys_code/management/commands/load_telescope_runs.py` - reordered the `if dry_run:` branch so `run_created`/`run_updated`/`run_unchanged` are folded from `action` only after both `preview_campaign_run_action()` and the `existing is not None`/`else` reconcile arms have returned; added a comment naming WR-02 and 35-REVIEW.md.
- `solsys_code/tests/test_load_telescope_runs.py` - new module-level `_RUN_SUMMARY_RE`/`_parse_run_summary()` helper; `TestMalformedTimezoneSkipsOneLine` gained `test_dry_run_and_real_run_report_the_same_counters_for_a_skipped_line` and `test_dry_run_pass_leaves_no_campaign_run_and_still_processes_the_next_line`; class docstring extended to name both invariants (NF-21, WR-02).
- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` - one new markdown + code cell pair in the "Per-line skip paths" section demonstrating the dry/real parity with real executed output; whole notebook regenerated by execution (16/16 code cells, 0 null execution counts).

## Decisions Made
- Combined Task 1 and Task 2 into a single commit (`909a67a`): the plan's own Task 2 action item allows landing its assertions inside Task 1's parity test or as a sibling method in the same file, and both tasks have no independently meaningful intermediate state -- writing and committing them together avoided an artificial split of one coherent test-file change.
- Fixed the WR-02 parity test's fixture setup to first repair `setUpTestData`'s malformed NTT timezone to a valid IANA zone (the class fixture seeds the typo from the start, since it exists for the real-path NF-21 test), create the `CampaignRun` with a real pass, then mutate the timezone to the typo before the dry-run/real comparison -- exactly reproducing PROBE-B's sequence (Rule 1 auto-fix: the first test draft assumed the fixture started with a valid timezone, which it does not).
- Reused the notebook's existing "NTT EFOSC2 allocation 9-13 July" `CampaignRun` (created earlier in the notebook) as the WR-02 parity cell's fixture rather than seeding a new one, per the plan's instruction to reuse the section's existing fixture machinery.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Initial WR-02 test draft assumed a valid starting timezone that the shared fixture does not provide**
- **Found during:** Task 1 (writing the parity test)
- **Issue:** `TestMalformedTimezoneSkipsOneLine.setUpTestData()` seeds NTT's `Observatory.timezone` with the typo `America/Santigo` from the start (it is the class's shared NF-21 fixture). A first draft of the WR-02 parity test tried to create the `existing` `CampaignRun` via a real pass over this already-broken timezone, which raised immediately and left no row to base the dry-run/real comparison on.
- **Fix:** The test now repairs NTT's timezone to `America/Santiago` first, runs a real pass to create the row, then mutates it to the typo before the dry-run/real comparison -- matching 35-REVIEW.md PROBE-B's exact sequence.
- **Files modified:** solsys_code/tests/test_load_telescope_runs.py
- **Verification:** Both new tests pass; `test_malformed_timezone_skips_only_its_own_line` (unmodified) still passes.
- **Committed in:** 909a67a (Task 1+2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in test fixture setup, caught before commit via the RED confirmation run)
**Impact on plan:** No scope creep; the fix only corrected the new test's own fixture sequencing to match the plan's specified reproduction.

## Issues Encountered
None beyond the auto-fixed test-fixture sequencing issue above. `pre-commit`'s `ruff-format` hook reformatted the new notebook cell's source on the first commit attempt (a single multi-line `re.compile(...)` call collapsed to one line), exactly as the plan warned (citing 35-10's precedent); re-staged and re-committed with no functional change.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- WR-02 -- the last of the three WARNING-severity regressions from 35-REVIEW.md iteration 5 that were in this plan's scope -- is closed in code, with a regression test that failed against the pre-fix tree with the exact PROBE-B reproduction.
- Plan 35-15 (per its stated `depends_on` on this plan) can now write its runbook sentence describing the restored dry-run/real-run parity invariant for `load_telescope_runs`.
- CR-01 (BLOCKER) and WR-01/WR-03/WR-04 from 35-REVIEW.md iteration 5 are NOT in this plan's scope -- untouched by this dispatch, as instructed.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-15*

## Self-Check: PASSED
- FOUND: solsys_code/management/commands/load_telescope_runs.py
- FOUND: solsys_code/tests/test_load_telescope_runs.py
- FOUND: docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
- FOUND: .planning/phases/35-allocation-layer-classical-cutover/35-14-SUMMARY.md
- FOUND: commit 909a67a
- FOUND: commit 9bb2b64
