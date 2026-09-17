---
phase: 36-unattended-operation
plan: 02
subsystem: infra
tags: [django-admin, management-command, watched-proposal, lco, discovery, tdd]

# Dependency graph
requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: the stable backfill_lco_observations.py this plan refactors and extends
provides:
  - solsys_code.models.WatchedProposal -- the admin-editable proposal list
  - solsys_code.management.commands.backfill_lco_observations.sweep_proposal() -- the
    per-proposal function both the CLI override and the bare watched-list loop call
  - solsys_code.management.commands.backfill_lco_observations.watched_rows() -- the
    ordered active-row queryset helper
affects: [36-03, 36-04, 36-05]

# Actuals (#2632)
actuals:
  tokens: 14721
  tasks: 3
  commits: 6

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "One module-level sweep_proposal(proposal, *, ..., stdout=None, stderr=None) -> str
      function both a CLI override and an unattended per-row loop call, with stdout/stderr
      sinks defaulting to io.StringIO() so it is callable with no management command at all"
    - "Per-row try/except/continue with class-name-only failure recording
      (f'failed: {type(exc).__name__}') written to a durable bookkeeping field, mirroring
      reconcile_campaign_runs.py's per-run isolation shape"

key-files:
  created:
    - solsys_code/migrations/0022_watchedproposal.py
    - solsys_code/tests/test_watched_proposal.py
  modified:
    - solsys_code/models.py
    - solsys_code/admin.py
    - solsys_code/management/commands/backfill_lco_observations.py
    - solsys_code/tests/test_backfill_lco_observations.py
    - solsys_code/tests/test_admin.py

key-decisions:
  - "sweep_proposal() accepts created_after/created_before as raw ISO-8601 strings (parsed
    internally via the existing _parse_created_bound()), not pre-parsed datetimes -- keeps
    the portal query-param string and the client-side parsed comparison sourced from one
    call inside the extracted function, so the CommandError-on-invalid-date check moved
    with the rest of the sweep logic rather than staying split across the extraction
    boundary. Command.handle() therefore now resolves only username-to-User (genuine CLI
    validation) before delegating."
  - "The watched-path's zero-rows message and its post-loop aggregate line are written
    directly via self.stdout.write(), and handle() returns None for that path instead of
    the built string -- avoids BaseCommand.execute() auto-writing the return value a
    second time, mirroring the no-double-print discipline the single-proposal summary
    already relied on."
  - "A per-row sweep failure is logged at logger.debug() with type(exc).__name__ only,
    never str(exc) -- keeps the SCHED-10/D-17 credential-safety rule intact in the debug
    log too, not just in last_run_summary/stderr, closing the one place a leaked
    credential could otherwise hide from the tests that check the other two surfaces."

patterns-established:
  - "TDD RED verified manually per-task by temporarily reverting the implementation file(s)
    to their pre-task committed state (keeping the new tests), confirming the exact
    ImportError/AttributeError/argparse failure, then restoring -- workflow.tdd_mode is
    false for this project, so gsd_run check tdd-red-evidence does not classify Django
    unittest output (precedent: plans 34-07, 36-01)."

requirements-completed: [DISCOVER-01]

coverage:
  - id: D1
    description: "WatchedProposal model, migration, and admin registration -- a staff user
      can add, deactivate and search proposal codes from the Django admin, with is_active
      toggled directly from the changelist and last_run_at/last_run_summary read-only"
    requirement: DISCOVER-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_watched_proposal.py -- TestWatchedProposalModel (6 tests)"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_admin.py -- WatchedProposalAdminTests (3 tests)"
        status: pass
      - kind: other
        ref: "python manage.py makemigrations --check --dry-run (no pending changes)"
        status: pass
    human_judgment: false
  - id: D2
    description: "sweep_proposal() extracted from Command.handle() as a behavior-preserving
      refactor -- callable directly with no management command involved, returning a
      summary byte-identical to the pre-extraction handle()"
    requirement: DISCOVER-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observations.py -- TestSweepProposalFunction
          (2 tests); all 30 pre-existing tests pass unmodified (git diff --diff-filter=D:
          0 deletions)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Bare invocation sweeps every active WatchedProposal row in proposal_code
      order, applying per-row target_list_name/attributed_to overrides, recording
      last_run_at/last_run_summary, isolating a portal/data failure to its own row with no
      leaked exception message, and reporting a healthy zero-exit no-op for an empty list"
    requirement: DISCOVER-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observations.py -- TestWatchedListSweep (5
          tests), TestPerProposalIsolation (2 tests), TestEmptyWatchedList (1 test), plus 1
          new test in TestBackfillLcoObservations (D-07 no-watched-row override)"
        status: pass
      - kind: other
        ref: "python manage.py backfill_lco_observations --dry-run against the real
          developer database (0 active WatchedProposal rows -> quiet no-op, exit 0)"
        status: pass
      - kind: other
        ref: "python manage.py backfill_lco_observations --proposal KEY2026B-004 --dry-run
          against the real developer database (override works with no WatchedProposal row)"
        status: pass
    human_judgment: false

# Metrics
duration: 21min
completed: 2026-09-17
status: complete
---

# Phase 36 Plan 2: Admin-Editable Watched-Proposal List Summary

**`backfill_lco_observations` now needs no arguments: an admin-editable `WatchedProposal` list drives a per-row-isolated sweep through an extracted `sweep_proposal()` function, replacing the required `--proposal` CLI argument.**

## Performance

- **Duration:** 21 min
- **Started:** 2026-09-17T15:23:37Z
- **Completed:** 2026-09-17T15:44:19Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments
- `WatchedProposal` model (`proposal_code` unique + stripped-on-save, `is_active`,
  `target_list_name`, `attributed_to` SET_NULL, `last_run_at`/`last_run_summary`
  bookkeeping), its migration, and a `WatchedProposalAdmin` with `is_active` listed,
  editable and filterable directly from the changelist
- `sweep_proposal()` extracted as a module-level function, callable with no management
  command involved, proven byte-identical to the pre-extraction `handle()` for both a real
  pass and a dry run, with all 30 pre-existing tests passing unmodified
- `--proposal` relaxed to an optional override; a bare invocation sweeps every active
  `WatchedProposal` row in `proposal_code` order, applies each row's overrides, records
  `last_run_at`/`last_run_summary` per row, isolates a portal/data failure to its own row
  with no leaked exception message, and reports a quiet zero-exit no-op for an empty list

## Task Commits

Each task carried `tdd="true"` and split into a `test(...)` RED commit and a `feat(...)`
GREEN commit:

1. **Task 1 RED: add failing tests for the WatchedProposal model and admin** - `25ea09d` (test)
2. **Task 1 GREEN: the WatchedProposal model, migration, and admin registration** - `e64ba98` (feat)
3. **Task 2 RED: add failing test for the sweep_proposal() extraction** - `f1c4205` (test)
4. **Task 2 GREEN: extract sweep_proposal() -- behavior-preserving refactor** - `133222f` (feat)
5. **Task 3 RED: add failing tests for the watched-list bare invocation** - `d3ba8e9` (test)
6. **Task 3 GREEN: bare invocation sweeps the watched list, one row at a time** - `d726aa4` (feat)

## Files Created/Modified
- `solsys_code/models.py` - `WatchedProposal` model
- `solsys_code/migrations/0022_watchedproposal.py` - `CreateModel` migration
- `solsys_code/admin.py` - `WatchedProposalAdmin` registration
- `solsys_code/management/commands/backfill_lco_observations.py` - `sweep_proposal()`,
  `watched_rows()`, optional `--proposal`, bare-invocation watched-list loop
- `solsys_code/tests/test_watched_proposal.py` - model tests
- `solsys_code/tests/test_admin.py` - `WatchedProposalAdminTests`
- `solsys_code/tests/test_backfill_lco_observations.py` - `TestSweepProposalFunction`,
  `TestWatchedListSweep`, `TestPerProposalIsolation`, `TestEmptyWatchedList`, plus one
  D-07 addition to `TestBackfillLcoObservations`

## Decisions Made
See `key-decisions` in frontmatter: `sweep_proposal()`'s raw-string date-bound parameters,
the watched-path's explicit-write/`return None` pattern to avoid double-printing, and
class-name-only debug logging on a per-row failure.

## Deviations from Plan

None - plan executed exactly as written. `gsd_run check tdd-red-evidence` was not run
(`workflow.tdd_mode` is `false` for this project); RED was verified manually per task by
temporarily reverting the implementation file(s) to their pre-task committed state (test
files staged/kept) and confirming the exact failure (ImportError for Tasks 1 and 2,
argparse `--proposal` required error plus 7 further failures/errors for Task 3), then
restoring the GREEN implementation and re-running the full suite.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for Plan 03 (the other three unattended runner steps, whose discovery step calls
`sweep_proposal()`/`watched_rows()` directly) and Plan 05 (the paired documentation --
`docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb`,
`docs/runbooks/telescope_runs_calendar.rst`, `docs/notebooks.rst`, `docs/installation.rst`,
`CLAUDE.md` -- which this plan's own `files_modified` frontmatter deliberately excludes;
36-05-PLAN.md's own `files_modified` confirms it owns that scope). No blockers.

---
*Phase: 36-unattended-operation*
*Completed: 2026-09-17*

## Self-Check: PASSED

All key files found on disk (`solsys_code/models.py` contains `class WatchedProposal`,
`solsys_code/migrations/0022_watchedproposal.py`, `solsys_code/admin.py` contains
`class WatchedProposalAdmin`, `solsys_code/tests/test_watched_proposal.py`); all 6 task
commit hashes (`25ea09d`, `e64ba98`, `f1c4205`, `133222f`, `d3ba8e9`, `d726aa4`) found in
`git log`. Full test module reruns
(`test_watched_proposal.py`, `test_admin.py`, `test_backfill_lco_observations.py` -- 105
tests total) and both quality gates (`pre-commit run ruff --all-files`,
`pre-commit run ruff-format --all-files`) pass at time of writing.
