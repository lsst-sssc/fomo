---
phase: 36-unattended-operation
plan: 09
subsystem: unattended-operation
tags: [check_unattended, output-streams, stdout, stderr, gap-closure, django-management-command, runbook]

# Dependency graph
requires:
  - phase: 36-07
    provides: the 9-numbered-step fresh-host setup procedure step 6 (the preflight step) inserts a passage into
  - phase: 36-08
    provides: the flat LCO_API_KEY step-2 wording Task 1 preserves the operator's provenance addendum inside
provides:
  - "Command.handle() in check_unattended.py writes each result line exactly once, to standard output when it passed and to standard error otherwise -- never both -- flushing standard output first so a merged destination (a terminal, or any 2>&1) still reads in check order"
  - "A test module that can express this defect class at all: _run_merged() (one io.StringIO bound as both stdout= and stderr=) and _result_lines(), a TestResultStreamRouting class with an exactly-once case, a multi-failure boundary case, a no-escape-bytes case, and a routing case, and seven revised presence tests that assert against the merged capture instead of two separate ones"
  - "docs/runbooks/telescope_runs_calendar.rst step 6 states which stream carries what, that each line is written once, what a standard-output-only redirect silently drops, and shows the merged 2>&1 form"
  - "The operator's own step-2 wording for where the LCO/SOAR API key comes from is committed, with its doubled article and missing verb fixed and its trailing whitespace stripped"
affects: [37-status-vocabulary-public-tallies-provenance-blind-gaps, ship-decision-for-v2.4]

# Actuals (#2632) — pairs with the plan's `estimate` to calibrate future estimates.
# Same estimateTokens scale (chars/4 over the realized diff), never a harness token count.
actuals:
  tokens: 4082
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single-sink test fixture for stream-routing regressions: a helper that passes ONE io.StringIO as both stdout= and stderr= to call_command(), modeling the terminal/2>&1 condition a command's real operator runs under, distinct from a helper that hands the command two separate sinks -- a suite that only ever exercises two separate sinks cannot see a dual-write-to-both-streams defect at all."
    - "Flush-before-cross-stream-write when standard output and standard error can share one destination file: block-buffered stdout lines can otherwise be hoisted below line-buffered stderr lines that logically preceded them once both are appended into the same log."

key-files:
  created: []
  modified:
    - solsys_code/management/commands/check_unattended.py
    - solsys_code/tests/test_check_unattended.py
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "Task 1 (committing the operator's uncommitted step-2 wording) runs FIRST and as its own commit, before Task 2 or Task 3 touch any file -- git add stages whole files, so committing the fix/runbook work first would have swept the operator's hand-written UAT wording into an unrelated commit and lost its provenance."
  - "Task 2 combines the test-module fix and the command fix into ONE fix(36-09): commit rather than separate test(...)/feat(...) commits -- the plan's own instruction treats them as one behavioral change whose tests describe its code, overriding the general TDD 2-3-commit convention for this specific gap-closure task."
  - "workflow.tdd_mode is false for this project (confirmed via config-get, matching the precedent already recorded in 36-01 through 36-04's SUMMARYs), so `gsd_run check tdd-red-evidence` was not run -- its underlying parser (parseNodeTestSummary/tapFailedTestNames) targets `node --test` TAP output and has no Python/Django adapter in this toolchain. RED was verified manually instead: all three new duplicate/routing-sensitive tests were run against the pre-fix command and failed for the stated reasons before any implementation change, exactly mirroring the manual-RED precedent 36-04-SUMMARY.md documents."
  - "Seven existing presence tests (two flock tests, four warning tests, one facility-credentials test) were switched from the two-sink _run() helper to the merged-sink _run_merged() helper, and the three assertions that had pinned the identical non-ok line in BOTH separate captures were deleted -- the routing contract they encoded as an (incorrect) expectation is now pinned once, by the dedicated TestResultStreamRouting.test_warning_and_passing_lines_route_to_separate_streams test, so a future routing regression fails one test instead of silently emptying seven presence assertions."

patterns-established:
  - "TestResultStreamRouting groups every merged-destination and stream-routing test for this command in one class, with a class docstring naming the gap (G-36-5) and the UAT round that found it, so a future reader locates the whole regression-coverage set in one place rather than scattered across the presence tests it made routing-agnostic."

requirements-completed: [SCHED-08, SCHED-10]

coverage:
  - id: D1
    description: "check_unattended.py's Command.handle() writes each result line exactly once -- to standard output when it passed, to standard error otherwise, flushing standard output first -- closing G-36-5 (the watched_proposals warning, and any non-ok line, rendering twice on a terminal or under 2>&1)"
    requirement: "SCHED-10"
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_check_unattended (48 tests, all pass, OK) -- includes TestResultStreamRouting's exactly-once, multi-failure-boundary, no-escape-bytes and routing cases, plus seven revised presence tests"
        status: pass
      - kind: other
        ref: "python manage.py check_unattended on the real checkout: stdout+stderr read together and piped through sort | uniq -d print nothing (no duplicate result line); stdout carries 9 passing lines, 0 warning/failure lines, and the cron-line block; stderr carries 0 passing lines and exactly the one watched_proposals warning this host produces; neither redirected file contains an ESC byte"
        status: pass
    human_judgment: false
  - id: D2
    description: "docs/runbooks/telescope_runs_calendar.rst step 6 of 'Setting it up on a fresh host' states which stream carries what, that each line is written once, what a standard-output-only redirect drops, and shows the merged 2>&1 form -- so a fresh-host operator reading the runbook alone knows how to capture a complete, correctly-ordered preflight log"
    requirement: "SCHED-08"
    verification:
      - kind: other
        ref: "awk-sliced fresh-host subsection contains 'standard output', 'standard error', '2>&1' and 'check_unattended'; a read-only docutils parse of the whole page and the sphinx-build hook both report the page clean; plan 36-07's presence-and-order gate and plan 36-08's flat-setting gate both still pass over the same slice"
        status: pass
    human_judgment: true
    rationale: "Whether the added passage 'reads well' in the surrounding voice and genuinely helps an operator is a judgment call automated checks can only proxy (token presence, hunk position, parse cleanliness) -- a human should skim the rendered step once."
  - id: D3
    description: "The operator's own hand-written step-2 wording for where the LCO/SOAR API key comes from (added during UAT round 3) is committed as its own docs(36-09) commit, with the doubled article and missing verb fixed and trailing whitespace stripped, before any other change touches that file"
    requirement: "SCHED-10"
    verification:
      - kind: other
        ref: "collapsed-newline grep confirms 'in the top right corner' (not the doubled-article form) and 'This should never be in the crontab line' (not the verbless form); git diff -U0 68fff91 shows exactly one hunk for this commit, positioned inside step 2; no trailing whitespace on any added line"
        status: pass
    human_judgment: false

# Metrics
duration: 42min
completed: 2026-09-18
status: complete
---

# Phase 36 Plan 09: Fix duplicated check_unattended result lines (G-36-5) Summary

**`Command.handle()` now writes each preflight result line to exactly one stream chosen by status -- standard output when it passed, standard error otherwise -- closing the UAT round 3 defect where a terminal or `2>&1` rendered every warning and failure twice, the second copy red.**

## Performance

- **Duration:** 42 min
- **Started:** 2026-09-18T19:39:00Z (approx, from plan dispatch)
- **Completed:** 2026-09-18T20:20:47Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Fixed `Command.handle()`'s emission loop in `check_unattended.py`: a passing result line goes to standard output only; a warning or failure line flushes standard output first and then goes to standard error only -- never both, closing G-36-5 (the watched_proposals warning that doubled during UAT round 3).
- Gave the test module the ability to see this defect class at all: `_run_merged()` (one `io.StringIO` as both `stdout=` and `stderr=`) and `_result_lines()`, a new `TestResultStreamRouting` class (exactly-once, multi-failure-boundary, no-escape-bytes, and routing cases), and seven presence tests switched to assert against the merged capture -- the module went from 44 to 48 tests, all passing.
- Documented the routing in `docs/runbooks/telescope_runs_calendar.rst` step 6: which stream carries what, that each line is written once, what a bare `>` redirect silently drops, and the merged `2>&1` form to use instead -- the same redirect the committed crontab template already appends with.
- Committed the operator's own uncommitted step-2 wording (added by hand during UAT round 3, naming the LCO Observation Portal's 'Profile' link as the API key source) as its own commit, first, with its doubled article and missing verb fixed and trailing whitespace stripped.

## Task Commits

Each task was committed atomically:

1. **Task 1: Commit the operator's step-2 API-key wording, typos fixed and whitespace stripped** - `db8a1cc` (docs)
2. **Task 2: Emit each result line once, to one stream — and give the test module a way to see it** - `b5f40a2` (fix) -- combines the test-module fix and the command fix in one commit, per this task's own instruction (the two are one behavioral change)
3. **Task 3: Say in the runbook which stream carries what, and what a bare redirect drops** - `be955cf` (docs)

_Note: Task 2 is `tdd="true"` but `workflow.tdd_mode` is `false` for this project (see Deviations below), so it produced one `fix(36-09):` commit rather than separate `test(...)`/`feat(...)` commits, per the task's own explicit instruction._

## Files Created/Modified

- `solsys_code/management/commands/check_unattended.py` - `Command.handle()`'s result-emission loop now branches on status: `ok` writes to `self.stdout` only; everything else flushes `self.stdout` then writes to `self.stderr` only, with a comment explaining why the flush is load-bearing.
- `solsys_code/tests/test_check_unattended.py` - Added `_run_merged()` and `_result_lines()` helpers; added `TestResultStreamRouting` (4 new tests); revised 7 existing presence tests (2 flock, 4 warning, 1 facility-credentials) to assert against the merged capture and deleted the 3 assertions that had pinned the identical line in both separate captures; module grew from 44 to 48 tests.
- `docs/runbooks/telescope_runs_calendar.rst` - Step 2: committed the operator's own API-key-source wording with typos fixed. Step 6: added a passage on stream routing, the merged `2>&1` redirect, and what a bare `>` drops.

## Decisions Made

- Task 1 (the operator's step-2 wording) runs first and as its own commit, before any other file change, so `git add` staging whole files cannot sweep the operator's UAT wording into an unrelated commit.
- Task 2's test fix and command fix are one `fix(36-09):` commit, not split into `test(...)`/`feat(...)`, per the plan's explicit instruction that they are one behavioral change.
- `workflow.tdd_mode` is `false` (confirmed via `config-get`), matching the precedent recorded in 36-01 through 36-04's SUMMARYs, so `gsd_run check tdd-red-evidence` was not run — its TAP parser targets `node --test` output and has no Python/Django adapter in this toolchain. RED was verified manually: all three new tests (`test_watched_proposals_warning_appears_exactly_once_in_merged_capture`, `test_multiple_non_ok_results_produce_no_duplicate_lines`, `test_warning_and_passing_lines_route_to_separate_streams`) were run against the pre-fix command and failed for the stated reasons (2 occurrences instead of 1; 13 unique lines instead of 10 under a forced multi-failure run; the warning line present in stdout when it should only be in stderr) before any implementation change was made.
- The three assertions that had pinned the identical non-ok line in both separate captures (the two flock tests, the heartbeat warning test) were deleted; the routing contract they encoded is now pinned once, by the dedicated routing test, so a future routing regression fails one test instead of silently emptying seven presence assertions.

## Deviations from Plan

None - plan executed exactly as written, including the manual-RED substitution for `gsd_run check tdd-red-evidence`, which the plan's own reference material (tdd.md, inherited via the execution context) permits when `workflow.tdd_mode` is `false` and which this same phase's prior plans (36-01 through 36-04) already established as the working precedent for this project.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Paired-Docs Note

Per CLAUDE.md's paired-docs rule, `check_unattended.py` maps explicitly to this runbook's "How do I run everything unattended?" section (specifically, the fresh-host setup procedure's step 6), not to a notebook. No notebook is owed by any task in this plan.

`deploy/cron/fomo.crontab.example` was checked (not assumed) for any comment describing the preflight's own output routing, as distinct from `run_unattended`'s schedule-line log redirect. It makes no such claim — its comments describe only where `check_unattended` resolves values FROM and where `run_unattended`'s own log goes — so it was left untouched.

## Next Phase Readiness

G-36-5 is closed: `python manage.py check_unattended`, run on this checkout as during UAT round 3, no longer prints any duplicated result line, in a merged capture or across the real two-stream run. Plan 36-07's presence-and-order gate and plan 36-08's flat-setting gate both still pass over the fresh-host subsection they cover. `solsys_code/unattended.py`, `notifications.py`, `run_unattended.py`, every other management command, `test_unattended.py`, and the crontab template are all untouched (`git diff --name-only 68fff91 -- solsys_code/ deploy/ src/` lists only the two files Task 2 owns). `ruff`, `ruff-format`, and the 48-test suite are all green. Re-verification (which reads this plan's `gap_ids: [G-36-5]`) can reconcile `36-UAT.md` and `36-VERIFICATION.md` — this plan intentionally left both untouched, per its own prohibitions.

---
*Phase: 36-unattended-operation*
*Completed: 2026-09-18*

## Self-Check: PASSED

- FOUND: solsys_code/management/commands/check_unattended.py
- FOUND: solsys_code/tests/test_check_unattended.py
- FOUND: docs/runbooks/telescope_runs_calendar.rst
- FOUND commit: db8a1cc
- FOUND commit: b5f40a2
- FOUND commit: be955cf
