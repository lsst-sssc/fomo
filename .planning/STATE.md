---
gsd_state_version: 1.0
milestone: v1.3
milestone_name: Full LCO Facility Sync
status: planning
last_updated: "2026-06-19T00:00:00.000Z"
last_activity: 2026-06-19
progress:
  total_phases: 3
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-19)

**Core value:** Generalize `sync_lco_observation_calendar` to correctly sync all LCO-family facilities (LCO + SOAR) for all real site codes and any combination of proposals, fixing the parameter-shape bugs found in v1.2 against real data.
**Current focus:** Phase 5 — Multi-Proposal & Multi-Facility Selection

## Current Position

Phase: 5 of 7 (Multi-Proposal & Multi-Facility Selection)
Plan: — (not yet planned)
Status: Roadmap created, ready to plan Phase 5
Last activity: 2026-06-19 — v1.3 ROADMAP.md created (Phases 5-7), 14/14 requirements mapped

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**

- Total plans completed: 6 (across v1.0-v1.2)
- Average duration: ~35 min (Phase 2) + 7-8 min/plan (Phase 3)
- Total execution time: ~3-4 sessions

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2 | - | - |
| 02 | 1 | ~35 min | ~35 min |
| 03 | 2 | - | - |
| 04 | 1 | - | - |
| 05 | TBD | - | - |
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

Last session: 2026-06-19T00:00:00.000Z
Stopped at: v1.3 ROADMAP.md created (Phases 5-7), REQUIREMENTS.md traceability updated
Resume file: None
Last activity: 2026-06-19 — Roadmap created for v1.3 Full LCO Facility Sync

## Operator Next Steps

- Run `/gsd-plan-phase 5` to plan Multi-Proposal & Multi-Facility Selection
