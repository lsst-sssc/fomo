---
phase: 35-allocation-layer-classical-cutover
plan: 11
subsystem: docs
tags: [jupyter-notebook, paired-docs, nbconvert, gap-closure, sched-06]

# Dependency graph
requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: "load_telescope_runs ZoneInfoNotFoundError handling (35-09); runbook/reconciler-notebook corrections and CLAUDE.md notebook-map closure (35-10)"
provides:
  - "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb regenerated with a new per-line skip-path section showing an unrecognised classical status word and a malformed IANA Observatory.timezone both skipped and logged, with a third valid line still processed (NF-24 half 1)"
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb regenerated un-routed with a new cell proving campaign attribution lands on the save that FIRST creates a record's own facility-url event, not a later one (NF-04, NF-24 half 2)"
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json rewritten by an un-routed run (captured_at 2026-09-15T15:04:51Z, 50 pending KEY2026B-004 records), discharging the SCHED-06 paired-docs step owed since 34-UAT.md"
affects: []

# Actuals (#2632)
actuals:
  tokens: 17964
  tasks: 2
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A per-line skip-path demo temporarily mutates a shared seeded fixture (NTT's Observatory.timezone) inside the same cell that needs the bad value, then restores it before the cell ends, so later sections of the same notebook resolve the fixture correctly again."
    - "A demonstration cell against the real developer database mirrors an existing regression test's fixture verbatim (same CampaignRun/ObservationRecord construction, same assertions) rather than inventing a new scenario, and wraps it in a transaction.atomic() block deliberately aborted via a dedicated exception class at the end, matching the notebook's existing _RollbackDemo pattern."

key-files:
  created: []
  modified:
    - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
    - docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb
    - docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json

key-decisions:
  - "Task 1's 'unrecognised classical status word' line is demonstrated via a parenthetical status word not in KNOWN_STATUSES (e.g. '(frobnicated)'), which parse_run_line() itself rejects with ValueError before the line's telescope even resolves -- caught by the command's general except (ValueError, Observatory.DoesNotExist) clause, not the narrower except KeyError around _CLASSICAL_RUN_STATUS[parsed.status] (which is unreachable via parse_run_line's real output, since a module-level assert enforces _CLASSICAL_RUN_STATUS's key set equals KNOWN_STATUSES). This is the actual skip path the shipped code exercises for this input class; the plan's must_haves only require the line be skipped and logged, not which specific except clause catches it."
  - "Task 1 mutates NTT's already-seeded Observatory.timezone to the exact typo 35-REVIEW.md reproduced ('America/Santigo') for the duration of one cell, then restores it, rather than seeding a new throwaway Observatory -- SITES in telescope_runs.py is a fixed 4-entry dict (Magellan-Clay/Magellan-Baade/NTT/FTS), so a schedule line can only resolve to one of those four telescope names."
  - "Task 2's demonstration cell mirrors test_placing_the_block_attributes_the_records_own_event_on_the_creating_save verbatim (same CampaignRun/ObservationRecord/CampaignRunObservation construction, same night-2 block, same assertions), using the real developer database's existing NTT Observatory (obscode 809) as the run's resolved site rather than creating one."

requirements-completed: [ALLOC-03, ALLOC-04]

coverage:
  - id: D1
    description: "load_telescope_runs_demo.ipynb carries a new section with real executed output showing an unrecognised classical status word and a malformed IANA Observatory.timezone both skipped and logged (skipped: 2), with a third valid line's CampaignRun still created and neither bad line leaving a CampaignRun row behind."
    requirement: ALLOC-04
    verification:
      - kind: other
        ref: "python -c null-execution-count check (15 code cells, none null); grep -cF 'ZoneInfoNotFoundError' (2); grep -cF 'skipped: 2' (2); python manage.py test solsys_code.tests.test_load_telescope_runs (27 tests, OK)"
        status: pass
    human_judgment: false
  - id: D2
    description: "project_observation_calendar_demo.ipynb carries a new cell with real executed output proving campaign attribution lands on the save that FIRST creates the record's own facility-url event (NF-04), rolled back inside a transaction.atomic() block so nothing persists in the real developer database."
    requirement: ALLOC-03
    verification:
      - kind: other
        ref: "python -c null-execution-count check (13 code cells, none null); grep -cF 'attributed immediately' (3); python manage.py test solsys_code.tests.test_observation_projector_signals (32 tests, OK); post-run DB query confirms zero leftover demo records/runs"
        status: pass
    human_judgment: false
  - id: D3
    description: "The projector demo notebook and sched06-baseline.json are regenerated un-routed (FOMO_DATABASE_PATH unset) against the real developer database, discharging the SCHED-06 paired-docs step owed since 34-UAT.md; the takeover cells take the already-converged branch and print rather than assert, matching the expected post-F-34-1-sweep state."
    requirement: ALLOC-03
    verification:
      - kind: other
        ref: "python -c captured_at > '2026-09-14' check on sched06-baseline.json (2026-09-15T15:04:51Z, 50 records); manual read of the takeover cells' printed output"
        status: pass
    human_judgment: false

duration: ~40min
completed: 2026-09-15
status: complete
---

# Phase 35 Plan 11: Loader and Projector Demo Notebook Regeneration Summary

**Regenerated both stale paired-docs notebooks NF-24 named -- a per-line skip-path section in the loader demo, and a creating-save attribution proof in the projector demo -- closing NF-24 and discharging the SCHED-06 paired-docs step still owed since 34-UAT.md.**

## Performance

- **Duration:** ~40 min
- **Completed:** 2026-09-15T15:06:11Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- **NF-24 half 1 (loader demo):** added a new markdown + code cell pair to `load_telescope_runs_demo.ipynb` demonstrating the command's current per-line skip surface after 35-09. A three-line schedule file carries an unrecognised classical status word (`(frobnicated)`, rejected by `parse_run_line()` itself with `ValueError`), a line naming `NTT` whose `Observatory.timezone` is temporarily mutated to the exact typo 35-REVIEW.md reproduced (`America/Santigo`, restored before any later cell resolves NTT again), and a valid third line. Executed output shows both bad lines skipped and logged (`skipped: 2`), the third line's `CampaignRun` created, and neither bad line leaving a `CampaignRun` row behind (the `transaction.atomic()` half of NF-08 this notebook had not shown before). The whole notebook was regenerated via `jupyter nbconvert --to notebook --execute --inplace` against a scratch copy of the developer database -- `src/fomo_db.sqlite3` was never opened for writing by this task.
- **NF-24 half 2 (projector demo):** added a new markdown + code cell pair to `project_observation_calendar_demo.ipynb`, mirroring `test_placing_the_block_attributes_the_records_own_event_on_the_creating_save` verbatim: a throwaway `CampaignRun` (site NTT) and a throwaway LCO `ObservationRecord`, deliberately left unprojectable (no `instrument_type`) and linked via `CampaignRunObservation` before any event exists, then given both an `instrument_type` AND a placed block covering the run's middle night in ONE save. Executed output confirms that single save both created the facility-url `CalendarEvent` and set its `CalendarEventMeta.run_id` to the run's own pk -- "attributed immediately, not on a later save" -- and retired the covered allocation night in the same save. The whole demonstration runs inside a `transaction.atomic()` block deliberately aborted via a dedicated exception class, matching the notebook's existing `_RollbackDemo` pattern; a post-run database query confirmed zero leftover demo records, runs, or events.
- **SCHED-06 paired-docs step discharged:** the projector notebook was regenerated **un-routed** (`FOMO_DATABASE_PATH` unset) against the real developer database, as `34-UAT.md` and `.planning/STATE.md`'s Operator Next Steps have directed since Phase 34. The takeover cells found nothing to take over -- `LCO: created: 0, updated: 0, unchanged: 159, ... site_lookup_failed: 1` on the first sweep -- and printed the already-converged message rather than asserting, exactly the expected outcome since the Phase 34 F-34-1 sweep already converged this database on 2026-09-14. `project_observation_calendar_demo.sched06-baseline.json` was rewritten with `captured_at=2026-09-15T15:04:51.716779+00:00` and 50 pending `KEY2026B-004` records (12 placed, 38 queued).
- Both notebooks' every code cell carries a non-null execution count (15/15 in the loader demo, 13/13 in the projector demo), and neither notebook was hand-patched -- both were regenerated end to end via `jupyter nbconvert --to notebook --execute --inplace`.
- `python manage.py test solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_observation_projector_signals` (59 tests) passes clean; neither notebook's work required any source change.

## Task Commits

1. **Task 1: Add the per-line skip-path section to the loader demo and regenerate it** - `4cee1f1` (docs)
2. **Task 2: Show attribution landing on the creating save in the projector demo, and regenerate it un-routed** - `5eb2718` (docs)

**Plan metadata:** pending (this commit)

## Files Created/Modified
- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` - new "Per-line skip paths" markdown + code cell pair inserted after the invoke/inspect section, before the cancelled-run section; whole notebook re-executed.
- `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` - new "Attribution lands on the creating save" markdown + code cell pair inserted immediately after the existing receiver-demo cell; whole notebook re-executed un-routed against the real developer database.
- `docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json` - rewritten by the un-routed run: fresh `captured_at`, 50 current pending `KEY2026B-004` records.

## Decisions Made
- Demonstrated the "unrecognised classical status word" skip path via a parenthetical status `parse_run_line()` itself rejects with `ValueError` (caught by the command's general `except (ValueError, Observatory.DoesNotExist)` clause), rather than trying to reach the narrower `except KeyError` around `_CLASSICAL_RUN_STATUS[parsed.status]` -- that inner clause is unreachable through real `parse_run_line()` output, since a module-level `assert` enforces `_CLASSICAL_RUN_STATUS`'s key set equals `KNOWN_STATUSES`. The plan's must_haves require the line be skipped and logged with `skipped: 2` in the summary, not that a specific internal `except` clause catch it, and this is the actual behavior the shipped code exhibits for this input.
- Mutated the already-seeded `NTT` `Observatory.timezone` to the exact typo `35-REVIEW.md` reproduced, for the duration of one cell, then restored it -- rather than creating a new throwaway `Observatory` -- because `telescope_runs.SITES` is a fixed 4-entry dict (`Magellan-Clay`/`Magellan-Baade`/`NTT`/`FTS`); a schedule line can only resolve to one of those four names, so the "malformed timezone" case has to land on one of the notebook's already-seeded sites.
- Task 2's demo used the developer database's existing `NTT` `Observatory` (obscode 809, already correctly configured) as the throwaway `CampaignRun`'s resolved site, rather than creating a new one -- matches the reference test's own fixture and avoids introducing an unnecessary new site row even transiently.

## Deviations from Plan

None - plan executed exactly as written. Both `<verify>` automated gates and the `<human-check>` gate pass against the final committed notebooks; the ruff-format pre-commit hook cosmetically reformatted quote style and trailing newlines in both notebook commits (consistent with 35-10's prior experience), re-staged and re-committed with no functional change.

## Issues Encountered
None. Both notebooks executed cleanly on the first `jupyter nbconvert` attempt, with no cell raising and no assertion failing. The takeover section's already-converged branch and the SCHED-06 baseline's real pending records matched the expected post-F-34-1-sweep state exactly, as the plan's own flagged assumptions anticipated.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- NF-24 is fully closed: both notebooks the review's fix cycle left stale are regenerated with real executed output exercising the exact behavior their paired module changed.
- The SCHED-06 paired-docs step owed since `34-UAT.md` is discharged -- `sched06-baseline.json` now carries a fresh `captured_at` and the current pending `KEY2026B-004` records for the next re-check to diff against.
- `.planning/REQUIREMENTS.md` already marks ALLOC-01 through ALLOC-05 complete (traceability table rows 113-117); no update needed by this plan.
- This is the last plan in Phase 35 (`35-allocation-layer-classical-cutover`) -- all 11 plans are now complete.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-15*

## Self-Check: PASSED
- FOUND: docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
- FOUND: docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb
- FOUND: docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json
- FOUND: commit 4cee1f1
- FOUND: commit 5eb2718
