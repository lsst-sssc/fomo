---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Classical Run Ingest
status: verifying
stopped_at: Completed 03-02-PLAN.md
last_updated: "2026-06-16T16:20:53.709Z"
last_activity: 2026-06-16
progress:
  total_phases: 2
  completed_phases: 2
  total_plans: 3
  completed_plans: 3
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-13)

**Core value:** A `load_telescope_runs` management command turns classical-schedule run lines into accurate, idempotent `tom_calendar.CalendarEvent`s — one per observing night — using Stage 1's `telescope_runs.SITES`/`get_site()`/`sun_event()` for sunset/sunrise times.
**Current focus:** Phase 03 — classical-calendar-ingest

## Current Position

Phase: 03
Plan: Not started
Status: Phase complete — ready for verification
Last activity: 2026-06-16

## Performance Metrics

**Velocity:**

- Total plans completed: 5
- Average duration: - min
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2 | - | - |
| 02 | 1 | ~35 min | ~35 min |
| 03 | 2 | - | - |

**Recent Trend:**

- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
| Phase 03 P02 | 404 | 1 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Sourcing: `SITES` coordinates come from `Observatory` model records (by MPC obscode), not a standalone hardcoded dict
- Scope: v1.1 covers Stage 2 only (classical run ingest); Stages 3-4 deferred
- Phase split: Phase 2 (parsing) is a prerequisite for Phase 3 (calendar ingest) — parser output tuples are the contract between them
- Testing: `Observatory`-backed `SITES`/`sun_event()` tests go in `solsys_code/tests/` (Django suite, DB access); pure-parsing logic (Phase 2) may be unit-testable under `tests/` (pytest) if it has no DB dependency
- Phase 2 (02-01): telescope resolution by prefix match against `SITES.keys()`; ambiguous prefixes (e.g. `'Magellan'`) raise `ValueError` listing all candidates rather than guessing
- Phase 3 (03-01): `Observatory.DoesNotExist` caught alongside `ValueError` in per-line handler (log+skip, not abort)
- Phase 3 (03-01): cross-month runs (`day2 < day1`) raise `ValueError` in `_iter_run_nights` and are reported+skipped — fails loudly
- [Phase ?]: Notebook outputs cleared by pre-commit hook for all .ipynb files (consistent with project convention)

### Pending Todos

None yet.

### Blockers/Concerns

- Environment blocker from v1.0 resolved (2026-06-13): `tom_catalogs` no longer referenced; `./manage.py test solsys_code` runs 79/79 OK.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260613-eb1 | Add a demo Jupyter notebook for Phase 1 (telescope_runs.py) under docs/notebooks/pre_executed/, and add a convention to .planning/PROJECT.md so future phases produce a similar demo notebook as part of their Definition of Done | 2026-06-13 | 1a36914 | [260613-eb1-add-a-demo-jupyter-notebook-for-phase-1-](./quick/260613-eb1-add-a-demo-jupyter-notebook-for-phase-1-/) |
| 260613-f7d | Modify docs/notebooks/ESO_How_to_download_data.ipynb to write downloaded files into a git-ignored docs/notebooks/data/ subdirectory, relocate the two stray ADP.*.fits files there, and exclude the newly-tracked vendored notebook from ruff (pyproject.toml) to keep `ruff check .` clean | 2026-06-13 | ef1f9b3 | [260613-f7d-modify-docs-notebooks-eso-how-to-downloa](./quick/260613-f7d-modify-docs-notebooks-eso-how-to-downloa/) |

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-06-16T16:09:19.672Z
Stopped at: Completed 03-02-PLAN.md
Resume file: None

## Operator Next Steps

- Phase 3 plan 03-01 complete. Run `/gsd-execute-phase` to execute 03-02 (integration test / demo notebook)
