---
phase: quick-260903-ik7
plan: 01
subsystem: solsys_code/management/commands
tags: [backfill, lco, dry-run, observation-records, django-management-command]
status: complete

dependency-graph:
  requires:
    - quick-260903-h1v (backfill_lco_observations command, first shipped)
  provides:
    - Accurate --dry-run counters for backfill_lco_observations, matching a real pass
    - embedded blocks / fallback lookups needed schedule-path counters (both modes)
    - _changed_record_fields shared four-field comparison helper
  affects:
    - solsys_code/management/commands/backfill_lco_observations.py
    - solsys_code/tests/test_backfill_lco_observations.py
    - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
    - docs/runbooks/telescope_runs_calendar.rst

tech-stack:
  added: []
  patterns:
    - Single comparison helper (_changed_record_fields) called by both the write branch
      and the dry-run branch, so updated-vs-unchanged can never drift between modes.
    - compare_schedule flag: schedule fields are compared only when the request's schedule
      was actually resolved (embedded observations block present), never against a None
      the dry-run fallback-skip produced.
    - Per-invocation de-dup set for the dry-run target counter, matching what a real run's
      save-then-match sequence naturally produces.

key-files:
  created: []
  modified:
    - solsys_code/management/commands/backfill_lco_observations.py
    - solsys_code/tests/test_backfill_lco_observations.py
    - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
    - docs/runbooks/telescope_runs_calendar.rst

decisions:
  - "Kept the summary as a single f-string with per-field ternaries (matching the original
    code's style) rather than branching into two separate f-string blocks for dry-run vs.
    real -- this keeps the label logic co-located per counter instead of duplicated across
    two near-identical blocks."
  - "The dry-run target de-dup set (dry_run_target_names_seen) is scoped to the whole
    command invocation, not per-RequestGroup, matching the plan's exact wording and a real
    run's actual behavior (a target saved once during a run is matched by fuzzy-name
    lookup on every subsequent request in the same run, not just within one group)."

metrics:
  duration: ~55min
  completed: 2026-09-03

actuals:
  tokens: 10348
  tasks: 3
  commits: 3
---

# Quick Task 260903-ik7: Fix backfill_lco_observations dry-run summary Summary

Replaced `backfill_lco_observations --dry-run`'s structurally-always-zero
created/updated/unchanged/targets/groups counters with real counts derived from actual
database reads, added `embedded blocks` / `fallback lookups needed` schedule-path counters
to both modes, and re-executed the paired demo notebook and runbook section to match.

## Task Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Accurate dry-run counters end to end, proven on one path | `be5f16c` | `solsys_code/management/commands/backfill_lco_observations.py`, `solsys_code/tests/test_backfill_lco_observations.py` |
| 2 | Expand test coverage to every counter in both modes | `15eefba` | `solsys_code/tests/test_backfill_lco_observations.py` |
| 3 | Re-execute the paired demo notebook and correct the runbook section | `ec11123` | `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb`, `docs/runbooks/telescope_runs_calendar.rst` |

## Files Created/Modified

- `solsys_code/management/commands/backfill_lco_observations.py` — added
  `_changed_record_fields()` module-level helper (single four-field comparison, shared by
  both branches); extended `_resolve_schedule()` to a 4-tuple returning whether the
  request payload carried an embedded `observations` block; rewrote the write branch to
  call the helper instead of an inline four-`if` block; rewrote the dry-run per-request
  branch to read the existing record with `.first()` and derive created/updated/unchanged
  from `_changed_record_fields(..., compare_schedule=embedded)`; added the
  `dry_run_target_names_seen` per-invocation de-dup set for the dry-run target counter;
  incremented `groups_created`/`groups_reused` in the dry-run group branch; added
  `embedded_blocks`/`fallback_lookups_needed` counters incremented on every request that
  reaches schedule resolution; rebuilt the summary f-string with the new would-forms and
  the two new counters; updated the `Command` class docstring with the fallback-schedule
  caveat.
- `solsys_code/tests/test_backfill_lco_observations.py` — added the module-level
  `_expected_summary()` helper (11-parameter, labels spelled out literally, no import from
  the command module); added 7 new tests (would-update, unchanged on the embedded path,
  unchanged on the fallback path, target de-dup within a group, groups
  would-create-then-would-reuse, mixed schedule-path counters in dry-run, mixed
  schedule-path counters in real-run) plus the Task 1 end-to-end proof test; strengthened
  the pre-existing `test_dry_run_writes_nothing_but_reports_summary` to assert its exact
  summary line. 24 tests total (up from 17).
- `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` — updated the
  markdown cell introducing the dry-run pass to explain the corrected counters and the
  fallback-schedule caveat; re-executed in place via `jupyter nbconvert --to notebook
  --execute --inplace` with no changes to the fixture, mocking helpers, or cleanup cell.
- `docs/runbooks/telescope_runs_calendar.rst` — "How do I backfill ObservationRecords
  without a campaign?" section: the `--dry-run` paragraph now states dry-run counts match
  a real pass with the fallback caveat; the "Scheduled times" paragraph documents the two
  new schedule-path counters; the single example summary literal block is replaced with
  two blocks (a real-run line and a dry-run line), each showing every counter in its
  correct label form.

## Decisions Made

- Single f-string with per-field ternaries preserved (not split into two summary
  constructions) — keeps each counter's dry-run/real label pairing visually adjacent.
- The dry-run target de-dup set is invocation-scoped, not group-scoped, matching a real
  run's actual save-then-match behavior across the whole command run.
- No `save()`/`update_fields` change to the write branch's `record.save()` call — kept as
  a plain save (no `update_fields`) exactly as before, so the model's auto-now `modified`
  field still updates, preserving the notebook's no-churn demonstration.

## Deviations from Plan

None - plan executed exactly as written. All `<verify>` automated checks passed on first
or second attempt (one test needed a `reset_mock()` call added mid-Task-2 to isolate the
dry-run pass's zero-calls assertion from the preceding real-run setup pass — an
in-test-authoring fix, not a plan deviation, tracked here for completeness rather than as
a Rule 1/2/3 auto-fix against production code).

## Issues Encountered

- `pre-commit`'s `ruff-format` hook reformatted the command module and test file on the
  first commit attempt of Tasks 1 and 2 (wrapping a couple of lines that exceeded the
  formatter's preferred wrap points even though they were under 120 columns). Re-staged
  and re-committed each time per the standard hook-failure retry protocol; no manual edits
  needed beyond re-running `git add`.

## Notebook's Printed Counters (h1v open question)

The regenerated dry-run cell (cell 8) prints:

```
requestgroups seen: 1, would create: 2, would update: 0, unchanged: 0, skipped: 0,
targets would create: 1, groups would create: 1, groups would reuse: 0,
embedded blocks: 0, fallback lookups needed: 2, block lookups failed: n/a (dry-run)
```

- The same-target de-duplication changed the notebook's target count: without it, the
  two-request group sharing `DEMO_TARGET_NAME` would have reported `targets would create:
  2` under `--dry-run`; with it, it correctly reports `targets would create: 1`, matching
  the real pass's `targets created: 1` in cell 10's output.
- `embedded blocks: 0, fallback lookups needed: 2` — this is the first real evidence
  toward the h1v SUMMARY's open question about which D-B schedule path the LCO portal
  actually exercises. The notebook's hand-built fixture never carried an embedded
  `observations` block on either request, so both fall back to the live per-request
  lookup. A future run against the real portal (proposal KEY2026B-004 or similar) should
  be compared against this 0/2 split to see whether the real portal embeds observation
  blocks in its `RequestGroup` payload at all, or whether the fallback lookup is always
  required in practice.

## Threat Flags

None — every threat register mitigation from `260903-ik7-PLAN.md`'s `<threat_model>` was
implemented as specified: the dry-run branch performs no writes (verified by exact-line
tests and zero-DB-write assertions across all dry-run tests), the summary counters are
pinned by exact-full-line assertions in both modes, no new information is added to
stdout/stderr beyond integers and fixed strings, the schedule-path counters add zero HTTP
calls, and the sibling command/test module were verified untouched (`git diff --quiet`)
before every commit.

## Self-Check: PASSED

- `solsys_code/management/commands/backfill_lco_observations.py` — FOUND, contains
  `_changed_record_fields`, `embedded_blocks`, `fallback_lookups_needed`.
- `solsys_code/tests/test_backfill_lco_observations.py` — FOUND, 24 `def test_` methods,
  `_expected_summary` helper present.
- `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` — FOUND, re-executed
  output contains `targets would create: 1`, `fallback lookups needed: 2`, `block lookups
  failed: n/a (dry-run)`.
- `docs/runbooks/telescope_runs_calendar.rst` — FOUND, contains `fallback lookups needed`
  (2 occurrences) and `targets would create` (1 occurrence).
- Commit `be5f16c` — FOUND in `git log --oneline`.
- Commit `15eefba` — FOUND in `git log --oneline`.
- Commit `ec11123` — FOUND in `git log --oneline`.
- Full plan-level `<verification>` re-run: `python manage.py test
  solsys_code.tests.test_backfill_lco_observations` (24 tests, OK); neighboring LCO/
  calendar modules (96 tests, OK); `python manage.py help backfill_lco_observations`
  registers with 4 command-specific flags; `pre-commit run ruff --all-files` and
  `pre-commit run ruff-format --all-files` clean; `pre-commit run sphinx-build --all-files`
  clean; `git status --porcelain` on the sibling command/test module and on
  campaign-related modules both empty.
