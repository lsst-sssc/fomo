---
phase: 261001-smo
plan: 01
subsystem: calendar / campaign attribution
tags: [F5, attribution-bridge, container-runs, reconciler, notebook, runbook]
status: complete
requirements: [ANNOT-01, ANNOT-02, TRIG-02]
commits: 3
plan_head_before: b44376c800de6cfdfac8f48df96505efd3a62c5b
plan_head_after: f8ee7f32b999640687aec0885ba6dd058d84ed51
key-files:
  modified:
    - solsys_code/campaign_reconciler.py
    - solsys_code/allocation_projector.py
    - solsys_code/tests/test_allocation_projector.py
    - solsys_code/tests/test_allocation_projector_signals.py
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
    - docs/runbooks/telescope_runs_calendar.rst
    - .planning/v2.4-INTENT-REVIEW.md (working tree only, uncommitted)
actuals:
  tasks: 3
  commits: 3
---

# Phase 261001-smo Plan 01: F5 container-run attribution Summary

Class-wide, satellite and queue-sourced runs now attribute their linked LCO/SOAR record events (set `CalendarEventMeta.run`) on reconcile and on link/record save/delete, through the single existing D-08 bridge.

## What changed

- `campaign_reconciler._reconcile_container()` now calls a new `_write_container_event()` (the old body, unchanged), then the bridge `_sync_observation_attribution()` via a function-local import, folding refusals into `blocked`. The bridge runs even when the run's own `RUN:{pk}` key is blocked; dry runs attribute nothing.
- `allocation_projector.reproject_allocation_if_dispatched()` keeps the approval gate; per-night runs still go to `project_allocation()`, other runs get the bridge alone. This single entry point covers both link receivers and the record-save path.
- Docstrings updated (bridge, entry point, both receivers, reconciler module).
- 12 new tests; the 10 named per-night tests are unchanged in body (AST pin gate passed).

## Commits

- 6494eab test(261001-smo): pin attribution of linked observation events on queue/class-wide runs (F5)
- f01b753 fix(261001-smo): run the attribution bridge for container runs on reconcile and on link save/delete (F5)
- f8ee7f3 docs(261001-smo): demo container-run attribution on the scratch copy; runbook says linked entries carry their campaign on every run type

## Test counts

- Baseline, two edited modules (`test_allocation_projector` + `test_allocation_projector_signals`): Ran 134, OK.
- After: 146 (134 + 12 new). The five-module run in Task 1 (those two plus `test_campaign_reconciler`, `test_observation_projector_signals`, `test_reconcile_campaign_runs`) was Ran 257, OK.
- Full suite `python manage.py test solsys_code --exclude-tag=ephemeris_segfault`: Ran 1833 tests, OK (skipped=1), exit 0 (previous baseline 1821 + 12).

## RED result

12 new tests: exactly 9 failed on assertions (attribution, `blocked`, missing WARNING log) and 3 guards passed (tests 2, C, D), as predicted. One fixture slip was fixed before the RED commit (`_make_record()` takes keyword-only arguments, so `*self._night_2_block()` became explicit `scheduled_start=`/`scheduled_end=`).

## Notebook

`reconcile_campaign_runs_demo.ipynb` was re-executed top to bottom (24 code cells, execution counts 1..24, no error output) on its `fomo-notebook-db-` scratch copy, which the last cell removed. Real-data line as executed:

`Linked LCO/SOAR records on this database copy: 182 -- own event attributed to the linking run: 182, attributed to a different run: 0, unattributed: 0 (of which on runs the reconciler does not skip: 0), no event: 0`

The synthetic demo printed all four transitions and `PASS: F5`. The namespace-isolation diff still reads `Differences found: 0`.

The copy did NOT predate the repair: the cron reads this checkout, and the fix was on disk before the 21:15 PDT tick, so by the time of the notebook run the live database already held the 181 attributions (read-only count 181 for runs 69-75). The cutover sweep therefore had nothing left to adopt on the copy; the real-data line is a post-repair confirmation, and the sentence in the cell is worded conditionally for that reason.

## Deviations from Plan

None to production behaviour. Small notes:

- The first runbook lead-in wrapped "for every kind of run" across a line break, which broke the plan's contiguous-words gate; re-wrapped so the words sit on one line (no content change).
- The plan's Task 2 runbook gate script searches for the heading text 'public tally show?', which is not in the file, so that exact script raises ValueError; the equivalent placement check (new paragraph between the two named paragraphs, required tokens present) was run instead and passed.
- One transient Sphinx failure (a missing autoapi file, a race) occurred on a first `pre-commit run sphinx-build`; the rerun and the commit's own hook passed.
- No pre-existing test was changed.

## Known Stubs

None.

## Threat Flags

None.

## Pending for the operator

The F5 "Fix landed" note is in `.planning/v2.4-INTENT-REVIEW.md`, left uncommitted beside the operator's own edits. The live confirmation (chip shows `KEY2026B-004_targets` on the observation entries) and ticking the F5 checkbox are the operator's.

## Self-Check: PASSED

Commits 6494eab, f01b753, f8ee7f3 exist; the six code/doc paths are modified; `.planning/v2.4-INTENT-REVIEW.md` is modified and unstaged.
