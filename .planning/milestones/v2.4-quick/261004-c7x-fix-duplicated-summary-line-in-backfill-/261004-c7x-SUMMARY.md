---
phase: 261004-c7x
plan: 01
subsystem: management-commands
tags: [backfill_lco_observation_records, stdout, bugfix]
requirements: [ALLOC-06]
key-files:
  modified:
    - solsys_code/management/commands/backfill_lco_observation_records.py
    - solsys_code/tests/test_backfill_lco_observation_records.py
decisions:
  - Keep the return value, drop the explicit write; BaseCommand.execute() prints the returned string once.
status: complete
actuals:
  tokens: 2000
  tasks: 2
  commits: 1
plan_head_before: f383fe27da61f00f92fc84852d63357c2d02e63d
plan_head_after: 98af555
---

# Quick Task 261004-c7x: backfill summary line printed once

`backfill_lco_observation_records` now prints its summary line exactly once (last line of stdout) in real and `--dry-run` modes. `handle()` still returns the string, so `call_command()` callers and tests are unchanged.

## What changed

- `handle()` no longer writes `summary` to `self.stdout` before `return summary`; the docstring `Returns:` section explains that `BaseCommand.execute()` prints it once.
- New test `TestBackfillSystemLinks.test_summary_line_is_printed_exactly_once` (dry-run pass, then real pass): summary is a non-empty `str`, occurs once in stdout, is the last line, and has the expected `Would create:` / `Created:` prefix.

## Evidence

- RED (before fix), both subtests: `AssertionError: 2 != 1` at `test_backfill_lco_observation_records.py:636` (`self.assertEqual(stdout.count(summary), 1, stdout)`), summary shown twice.
- GREEN: module `python manage.py test solsys_code.tests.test_backfill_lco_observation_records` ran 26 tests, OK.
- Full suite `python manage.py test solsys_code --exclude-tag=ephemeris_segfault`: Ran 1952 tests in 1428.6s, OK (baseline 1951 + the new test); no pre-existing failures.
- `pre-commit run ruff --all-files` and `ruff-format --all-files`: Passed. The commit's pre-commit hooks (ruff, Sphinx, unit tests) all passed.

## Paired-doc re-check

No notebook or runbook change needed. The scan of `docs/notebooks/pre_executed/*.ipynb` printed `[]` (no code cell names the command or carries `unmatched target:` output), and `docs/runbooks/telescope_runs_calendar.rst` shows the summary example once (`grep -c 'already existed: 12'` = 1), so the fix brings code into line with the runbook.

## Deviations from Plan

**[Rule 3 - Blocking] Missing generated `src/fomo/_version.py` in the worktree.** It is gitignored (setuptools_scm output) so the fresh worktree lacked it and `manage.py` failed to import settings. Copied the file from the main checkout; it is ignored by git and not committed.

The plan's expectation of branch `issue37-telescope-runs-calendar` was superseded by the worktree branch `worktree-agent-af9d61bef3c76f925` (required by the orchestrator's HEAD check). Staging was by explicit path; the commit contains exactly the two planned files. The operator's uncommitted edits were not present in the worktree.

## Commit

- 98af555: fix(261004-c7x): print the backfill_lco_observation_records summary line once

## Known Stubs

None.

## Self-Check: PASSED

Both modified files present; commit 98af555 exists (`git log`); commit touched only the two planned files.
