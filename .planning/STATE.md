---
gsd_state_version: 1.0
milestone: v1.3
milestone_name: Full LCO Facility Sync
current_phase: 6
status: executing
stopped_at: Phase 6 context gathered
last_updated: "2026-06-21T00:11:49.709Z"
last_activity: 2026-06-19
last_activity_desc: "Completed quick task 260619-ml8: Fix pre-commit notebook-clear exclude path + redundant UTC offset in calendar description"
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 1
  completed_plans: 1
  percent: 33
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-19)

**Core value:** Generalize `sync_lco_observation_calendar` to correctly sync all LCO-family facilities (LCO + SOAR) for all real site codes and any combination of proposals, fixing the parameter-shape bugs found in v1.2 against real data.
**Current focus:** Phase 05 — multi-proposal-multi-facility-selection

## Current Position

Phase: 6
Plan: Not started
Status: Ready to execute
Last activity: 2026-06-19 - Completed quick task 260619-ml8: Fix pre-commit notebook-clear exclude path + redundant UTC offset in calendar description

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**

- Total plans completed: 7 (across v1.0-v1.2)
- Average duration: ~35 min (Phase 2) + 7-8 min/plan (Phase 3)
- Total execution time: ~3-4 sessions

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2 | - | - |
| 02 | 1 | ~35 min | ~35 min |
| 03 | 2 | - | - |
| 04 | 1 | - | - |
| 05 | 1 | - | - |
| 06 | TBD | - | - |
| 07 | TBD | - | - |

## Accumulated Context

### Decisions

All decisions logged in PROJECT.md Key Decisions table. Recent decisions affecting v1.3:

- Phase ordering follows research's dependency chain: query generalization (Phase 5, pure/zero-I/O) before instrument extraction (Phase 6, fallback label needs correct instrument data) before telescope-label API+fallback (Phase 7, highest-risk new I/O).
- SYNC-06..09 (partial-failure counters/reporting) folded into Phase 7 rather than a standalone phase — that's the only phase introducing the new API-failure axis they report on.

### Pending Todos

1. Status-aware calendar event coloring (telescope/proposal-keyed, alpha by confidence) — `.planning/todos/pending/2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md`

### Blockers/Concerns

- Research gap (Phase 7): exact JSON key names in `/api/requests/<id>/observations/` block responses are inferred by analogy, not confirmed against a live response — confirm against a real `observation_id` before finalizing parsing.
- Research gap (Phase 7): whether `FACILITIES['SOAR']` needs an explicit settings.py entry vs. relying on `SOARSettings` defaults — decide during Phase 5 planning.
- Research gap (Phase 7): `tlv` (Wise Observatory) appears in the webpage-sourced 8-site table but not confirmed in installed `LCOSettings.get_sites()`/`SOARSettings.get_sites()` — verify before shipping the static mapping dict.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260619-f7u | Phase 5 notebook gap: update sync_lco_observation_calendar_demo.ipynb for multi-proposal/multi-facility (SELECT-02/03/04/05); document Stage-vs-Phase numbering mapping in PROJECT.md | 2026-06-19 | 5bef02d | [260619-f7u-phase-5-notebook-gap-update-sync-lco-obs](./quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/) |
| 260619-jpr | Fix sync_lco_observation_calendar SOAR site mapping bug: add SOAR site code 'sor' to SITE_TELESCOPE_MAP, fix SOAR test fixtures and demo notebook fixtures that incorrectly reused LCO site code 'coj' for SOAR records | 2026-06-19 | caf97bd | [260619-jpr-fix-sync-lco-observation-calendar-soar-s](./quick/260619-jpr-fix-sync-lco-observation-calendar-soar-s/) |
| 260619-ml8 | Fix pre-commit jupyter-nb-clear-output exclude path mismatch (docs/pre_executed vs docs/notebooks/pre_executed) and remove redundant +00:00 UTC offset from sync_lco_observation_calendar's CalendarEvent description Window (UTC) string | 2026-06-19 | 6b1d37a | [260619-ml8-fix-pre-commit-jupyter-nb-clear-output-e](./quick/260619-ml8-fix-pre-commit-jupyter-nb-clear-output-e/) |

### Key Technical Notes (carried from Phase 4 / v1.2)

- `parameters` on `ObservationRecord` is a `TextField` containing JSON (not a JSONField) — filtering requires fetching then parsing in Python, no DB-level JSON filtering assumed safe without re-verification for `__in` queries.
- CalendarEvent upsert keyed on `url` (`LCOFacility().get_observation_url(observation_id)`, confirmed `/requests/<id>` no trailing slash).
- No-churn idempotency: only call `.save()` when fields actually changed.
- `astropy Time.to_datetime()` produces microseconds — strip with `.replace(microsecond=0)` before DB save.
- DB-dependent tests go in `solsys_code/tests/test_sync_lco_observation_calendar.py`, run with `./manage.py test solsys_code`.
- v1.2 real-data bug (drives v1.3): real records have no flat `instrument_type` or `site` key in `parameters`; multi-config `c_1..c_5_instrument_type` shape only, and `SITE_TELESCOPE_MAP` was a 2-entry unverified guess.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| todo | 2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md | pending (deliberately deferred future work) | v1.2 close |

## Session Continuity

Last session: 2026-06-20T23:31:23.939Z
Stopped at: Phase 6 context gathered
Resume file: .planning/phases/06-correct-instrument-type-extraction/06-CONTEXT.md
Last activity: 2026-06-19 — Phase 05 (multi-proposal-multi-facility-selection) executed and verified

## Operator Next Steps

- Run `/gsd-discuss-phase 6` to start Phase 6 (correct-instrument-type-extraction)
