---
gsd_state_version: 1.0
milestone: v1.3
milestone_name: Full LCO Facility Sync
current_phase: 07
current_phase_name: live-telescope-label-resolution-with-fallback-failure-report
status: executing
stopped_at: Completed 07-01-PLAN.md
last_updated: "2026-06-23T05:27:53.728Z"
last_activity: 2026-06-23
last_activity_desc: Phase 07 execution started
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 4
  completed_plans: 3
  percent: 67
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-19)

**Core value:** Generalize `sync_lco_observation_calendar` to correctly sync all LCO-family facilities (LCO + SOAR) for all real site codes and any combination of proposals, fixing the parameter-shape bugs found in v1.2 against real data.
**Current focus:** Phase 07 — live-telescope-label-resolution-with-fallback-failure-report

## Current Position

Phase: 07 (live-telescope-label-resolution-with-fallback-failure-report) — EXECUTING
Plan: 2 of 2
Status: Ready to execute
Last activity: 2026-06-23 — Phase 07 execution started

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**

- Total plans completed: 8 (across v1.0-v1.2)
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
| 06 | 1 | - | - |
| 07 | TBD | - | - |
| Phase 06 P01 | 6min | 2 tasks | 2 files |
| Phase 07 P01 | 50min | 3 tasks | 3 files |

## Accumulated Context

### Decisions

All decisions logged in PROJECT.md Key Decisions table. Recent decisions affecting v1.3:

- Phase ordering follows research's dependency chain: query generalization (Phase 5, pure/zero-I/O) before instrument extraction (Phase 6, fallback label needs correct instrument data) before telescope-label API+fallback (Phase 7, highest-risk new I/O).
- SYNC-06..09 (partial-failure counters/reporting) folded into Phase 7 rather than a standalone phase — that's the only phase introducing the new API-failure axis they report on.
- [Phase ?]: Sentinel None + InstrumentExtractionError contract chosen over a bare exception, matching the file's existing 'return None to signal non-match' style
- [Phase ?]: Added a flat instrument_type fallback tier beyond D-01/D-02 to keep the 19 pre-existing regression tests passing for today's legacy single-config shape
- [Phase ?]: tlv dropped entirely from SITE_TELESCOPE_MAP (operator decision at 07-01 Task 1 checkpoint) -- confirmed absent from installed LCOSettings/SOARSettings; scope corrected to 7 real sites, not 8
- [Phase ?]: elp/lsc/cpt/tfn aperture-class inventory confirmed by operator (LCO staff) at the same checkpoint -- both 1m0 and 0m4 entries added per site, no [ASSUMED] tag needed

### Pending Todos

1. Status-aware calendar event coloring (telescope/proposal-keyed, alpha by confidence) — `.planning/todos/pending/2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md`
2. Extract site/telescope mapping and instrument extraction into own module (revisit after Phase 7 ships) — `.planning/todos/pending/2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`

### Blockers/Concerns

- Research gap (Phase 7): exact JSON key names in `/api/requests/<id>/observations/` block responses are inferred by analogy, not confirmed against a live response — confirm against a real `observation_id` before finalizing parsing.
- Research gap (Phase 7): whether `FACILITIES['SOAR']` needs an explicit settings.py entry vs. relying on `SOARSettings` defaults — decide during Phase 5 planning.
- ~~Research gap (Phase 7): `tlv` (Wise Observatory) appears in the webpage-sourced 8-site table but not confirmed in installed `LCOSettings.get_sites()`/`SOARSettings.get_sites()` — verify before shipping the static mapping dict.~~ Resolved at the 07-01 Task 1 checkpoint: `tlv` dropped entirely (confirmed absent from both installed `get_sites()` implementations); verified dict covers the 7 real sites instead of 8. See 07-01-SUMMARY.md.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260619-f7u | Phase 5 notebook gap: update sync_lco_observation_calendar_demo.ipynb for multi-proposal/multi-facility (SELECT-02/03/04/05); document Stage-vs-Phase numbering mapping in PROJECT.md | 2026-06-19 | 5bef02d | [260619-f7u-phase-5-notebook-gap-update-sync-lco-obs](./quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/) |
| 260619-jpr | Fix sync_lco_observation_calendar SOAR site mapping bug: add SOAR site code 'sor' to SITE_TELESCOPE_MAP, fix SOAR test fixtures and demo notebook fixtures that incorrectly reused LCO site code 'coj' for SOAR records | 2026-06-19 | caf97bd | [260619-jpr-fix-sync-lco-observation-calendar-soar-s](./quick/260619-jpr-fix-sync-lco-observation-calendar-soar-s/) |
| 260619-ml8 | Fix pre-commit jupyter-nb-clear-output exclude path mismatch (docs/pre_executed vs docs/notebooks/pre_executed) and remove redundant +00:00 UTC offset from sync_lco_observation_calendar's CalendarEvent description Window (UTC) string | 2026-06-19 | 6b1d37a | [260619-ml8-fix-pre-commit-jupyter-nb-clear-output-e](./quick/260619-ml8-fix-pre-commit-jupyter-nb-clear-output-e/) |
| 260620-v9x | Phase 6 notebook gap: update sync_lco_observation_calendar_demo.ipynb to demonstrate EXTRACT-01/EXTRACT-02 (SOAR multi-config, MUSCAT per-channel, malformed/extraction_failed) | 2026-06-21 | d41cdc7 | [260620-v9x-update-docs-notebooks-pre-executed-sync-](./quick/260620-v9x-update-docs-notebooks-pre-executed-sync-/) |
| 260621-foo | Document demo-notebook-companion convention in CLAUDE.md (planner/plan-checker/executor/verifier responsibilities) to prevent recurrence of the Phase 5/6 notebook-scoping gap | 2026-06-21 | fe378de | [260621-foo-add-a-claude-md-convention-requiring-dem](./quick/260621-foo-add-a-claude-md-convention-requiring-dem/) |

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

Last session: 2026-06-23T05:27:53.692Z
Stopped at: Completed 07-01-PLAN.md
Resume file: None
Last activity: 2026-06-21 — Phase 06 UAT complete (4/4 passed) and security audit verified (06-SECURITY.md, threats_open: 0)

## Operator Next Steps

- Plan 07-01 complete (TELESCOPE-01, TELESCOPE-02, SYNC-08, SYNC-09 validated). Plan 07-02 (Wave 2) is ready to execute — it wires the live-API + fallback decision tree into `Command.handle()`/`_build_event_fields`/`_title_for`, adds the `telescope_api_failed` counter, and updates the demo notebook again for the new observable behavior.
- Note for phase completion: ROADMAP.md/CONTEXT.md's original "8 real LCO-network sites" framing should be reconciled with the operator-approved 7-site correction documented in 07-01-SUMMARY.md.
