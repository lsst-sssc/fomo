---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: LCO Queue Calendar Sync
status: executing
stopped_at: Phase 4 context gathered
last_updated: "2026-06-17T22:07:16.165Z"
last_activity: 2026-06-17
progress:
  total_phases: 1
  completed_phases: 1
  total_plans: 1
  completed_plans: 1
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-16)

**Core value:** A `sync_lco_observation_calendar` management command syncs FTS/MuSCAT4 LCO queue ObservationRecords to the FOMO calendar as unified CalendarEvents — window banner when unscheduled, placed block once the LCO scheduler acts, updating in place on rescheduling, and marked with a status prefix on terminal states.
**Current focus:** Phase 04 — lco-queue-sync-command

## Current Position

Phase: 04
Plan: Not started
Status: Executing Phase 04
Last activity: 2026-06-17

```
[░░░░░░░░░░░░░░░░░░░░] 0% — Phase 4 of 4 (v1.2)
```

## Performance Metrics

**Velocity:**

- Total plans completed: 6 (across v1.0 and v1.1)
- Average duration: ~35 min (Phase 2) + 7-8 min/plan (Phase 3)
- Total execution time: ~3-4 sessions

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2 | - | - |
| 02 | 1 | ~35 min | ~35 min |
| 03 | 2 | - | - |
| 04 | 1 | - | - |

## Accumulated Context

### Decisions

All decisions logged in PROJECT.md Key Decisions table.

### Pending Todos

1. Status-aware calendar event coloring (telescope/proposal-keyed, alpha by confidence) — `.planning/todos/pending/2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md`

### Blockers/Concerns

None. Environment blocker (tom_catalogs) resolved 2026-06-11 (PR #38 merged to main).

### Key Technical Notes (Phase 4)

- `parameters` on `ObservationRecord` is a `TextField` containing JSON (not a JSONField). Filtering by proposal code requires fetching `ObservationRecord.objects.filter(facility='LCO')` and parsing in Python — no DB-level JSON filtering.
- CalendarEvent upsert keyed on `url` field (`https://observe.lco.global/requestgroups/<observation_id>/`).
- Two time-source branches: `parameters['start']`/`parameters['end']` when `scheduled_start is None` (banner); `scheduled_start`/`scheduled_end` when populated (placed block).
- Terminal state constants: WINDOW_EXPIRED → `[EXPIRED]`, CANCELED → `[CANCELLED]`, FAILURE_LIMIT_REACHED / NOT_ATTEMPTED → `[FAILED]`.
- No-churn idempotency: only call `event.save()` when fields have actually changed (avoids `modified` timestamp churn on unchanged events — same pattern as Phase 3 load_telescope_runs).
- `astropy Time.to_datetime()` produces microseconds — strip with `.replace(microsecond=0)` before DB save (established pattern from Phase 3 fix, commit 437aa53).
- DB-dependent tests go in `solsys_code/tests/test_sync_lco_observation_calendar.py`, run with `./manage.py test solsys_code`.
- `ObservationRecord` lives in `tom_observations.models`; `CalendarEvent` in `tom_calendar.models`.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260613-eb1 | Add a demo Jupyter notebook for Phase 1 (telescope_runs.py) under docs/notebooks/pre_executed/, and add a convention to .planning/PROJECT.md | 2026-06-13 | 1a36914 | [260613-eb1-add-a-demo-jupyter-notebook-for-phase-1-](./quick/260613-eb1-add-a-demo-jupyter-notebook-for-phase-1-/) |
| 260613-f7d | Modify docs/notebooks/ESO_How_to_download_data.ipynb to write downloaded files into a git-ignored docs/notebooks/data/ subdirectory | 2026-06-13 | ef1f9b3 | [260613-f7d-modify-docs-notebooks-eso-how-to-downloa](./quick/260613-f7d-modify-docs-notebooks-eso-how-to-downloa/) |
| 260617-mlr | Backfill Phase 04's missing demo notebook and enforce the convention going forward | 2026-06-17 | a2f8eea | [260617-mlr-backfill-phase-04-s-missing-demo-noteboo](./quick/260617-mlr-backfill-phase-04-s-missing-demo-noteboo/) |
| 260618-h97 | Add src/data/ to .gitignore (Django MEDIA_ROOT resolves inside the git checkout; stopgap until MEDIA_ROOT is relocated outside the repo for dockerized deployment) | 2026-06-18 | d3f3727 | [260618-h97-add-src-data-to-gitignore-this-directory](./quick/260618-h97-add-src-data-to-gitignore-this-directory/) |
| 260618-lw4 | De-emphasize [QUEUED] calendar events so they never look more visually prominent than confirmed/placed events (project-level tom_calendar template override) | 2026-06-18 | 517e8bc | [260618-lw4-de-emphasize-queued-calendar-events-so-t](./quick/260618-lw4-de-emphasize-queued-calendar-events-so-t/) |
| 260618-mck | Fix insufficient contrast in [QUEUED] calendar event de-emphasis style (follow-up to 260618-lw4; forced-white event-title text was nearly invisible against the near-transparent fill, especially on the #f8f9fa other-month overflow cell) | 2026-06-18 | 5ee2dd0 | [260618-mck-fix-insufficient-contrast-in-queued-cale](./quick/260618-mck-fix-insufficient-contrast-in-queued-cale/) |

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-06-17T20:51:13.159Z
Stopped at: Phase 4 context gathered
Resume file: .planning/phases/04-lco-queue-sync-command/04-CONTEXT.md
Last activity: 2026-06-18 - Completed quick task 260618-mck: Fix insufficient contrast in [QUEUED] calendar event style

## Operator Next Steps

- Run `/gsd-plan-phase 4` to decompose Phase 4 into executable plans.
