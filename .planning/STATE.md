---
gsd_state_version: 1.0
milestone: v1.3
milestone_name: Full LCO Facility Sync
status: Awaiting next milestone
stopped_at: v1.3 milestone closed and archived
last_updated: "2026-06-24T17:14:39.424Z"
last_activity: 2026-06-24 — Milestone v1.3 completed and archived
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 5
  completed_plans: 5
  percent: 100
current_phase: null
current_phase_name: null
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-24 after v1.3 milestone close)

**Core value:** v1.3 shipped — generalized `sync_lco_observation_calendar` to correctly sync all LCO-family facilities (LCO + SOAR) for all real site codes and any combination of proposals. Next milestone's core value not yet defined.
**Current focus:** Planning next milestone (`/gsd-new-milestone`)

## Current Position

Phase: Milestone v1.3 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-06-24 — Milestone v1.3 completed and archived

## Performance Metrics

**Velocity:**

- Total plans completed: 9 (across v1.0-v1.2)
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
| Phase 07 P02 | ~50min | 3 tasks | 3 files |
| Phase 07.1 P01 | 25min | 3 tasks | 3 files |
| 07.1 | 1 | - | - |

## Accumulated Context

### Decisions

All v1.3 decisions logged in PROJECT.md's Key Decisions table (Phases 5-07.1 entries, backfilled at milestone close). Cleared here now that v1.3 is shipped — see PROJECT.md for the durable record.

### Pending Todos

1. Status-aware calendar event coloring (telescope/proposal-keyed, alpha by confidence) — `.planning/todos/pending/2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md`
2. Extract site/telescope mapping and instrument extraction into own module (revisit after Phase 7 ships) — `.planning/todos/pending/2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`

### Blockers/Concerns

None open. All three Phase 7 research gaps (JSON key names in the observations API response, `FACILITIES['SOAR']` settings entry, `tlv` site-table discrepancy) were resolved during Phase 5/7 implementation and verified live — see PROJECT.md Key Decisions and `07-01-SUMMARY.md`/`07-02-SUMMARY.md` for details.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260619-f7u | Phase 5 notebook gap: update sync_lco_observation_calendar_demo.ipynb for multi-proposal/multi-facility (SELECT-02/03/04/05); document Stage-vs-Phase numbering mapping in PROJECT.md | 2026-06-19 | 5bef02d | [260619-f7u-phase-5-notebook-gap-update-sync-lco-obs](./quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/) |
| 260619-jpr | Fix sync_lco_observation_calendar SOAR site mapping bug: add SOAR site code 'sor' to SITE_TELESCOPE_MAP, fix SOAR test fixtures and demo notebook fixtures that incorrectly reused LCO site code 'coj' for SOAR records | 2026-06-19 | caf97bd | [260619-jpr-fix-sync-lco-observation-calendar-soar-s](./quick/260619-jpr-fix-sync-lco-observation-calendar-soar-s/) |
| 260619-ml8 | Fix pre-commit jupyter-nb-clear-output exclude path mismatch (docs/pre_executed vs docs/notebooks/pre_executed) and remove redundant +00:00 UTC offset from sync_lco_observation_calendar's CalendarEvent description Window (UTC) string | 2026-06-19 | 6b1d37a | [260619-ml8-fix-pre-commit-jupyter-nb-clear-output-e](./quick/260619-ml8-fix-pre-commit-jupyter-nb-clear-output-e/) |
| 260620-v9x | Phase 6 notebook gap: update sync_lco_observation_calendar_demo.ipynb to demonstrate EXTRACT-01/EXTRACT-02 (SOAR multi-config, MUSCAT per-channel, malformed/extraction_failed) | 2026-06-21 | d41cdc7 | [260620-v9x-update-docs-notebooks-pre-executed-sync-](./quick/260620-v9x-update-docs-notebooks-pre-executed-sync-/) |
| 260621-foo | Document demo-notebook-companion convention in CLAUDE.md (planner/plan-checker/executor/verifier responsibilities) to prevent recurrence of the Phase 5/6 notebook-scoping gap | 2026-06-21 | fe378de | [260621-foo-add-a-claude-md-convention-requiring-dem](./quick/260621-foo-add-a-claude-md-convention-requiring-dem/) |
| 260623-ocs | Fix T-07-03 security spec gap in sync_lco_observation_calendar: read resolved API block via .get() instead of bracket indexing in _build_event_fields, None-guard _aperture_class_from_telescope_code, add regression test for a block missing 'site'/'telescope' falling back (not skipped) | 2026-06-24 | 2fa0300 | [260623-ocs-fix-t-07-03-security-spec-gap-in-sync-lc](./quick/260623-ocs-fix-t-07-03-security-spec-gap-in-sync-lc/) |
| 260623-su3 | Fix SITE_TELESCOPE_MAP completeness gap from Phase 7 UAT Test 1: add missing ('coj','1m0'), ('coj','0m4'), ('ogg','0m4') entries confirmed against https://lco.global/observatory/sites/mpccodes/, add regression test for the 3 newly-mapped pairs | 2026-06-24 | 5583400 | [260623-su3-fix-site-telescope-map-completeness-gap-](./quick/260623-su3-fix-site-telescope-map-completeness-gap-/) |

### Key Technical Notes (carried from Phase 4 / v1.2)

- `parameters` on `ObservationRecord` is a `TextField` containing JSON (not a JSONField) — filtering requires fetching then parsing in Python, no DB-level JSON filtering assumed safe without re-verification for `__in` queries.
- CalendarEvent upsert keyed on `url` (`LCOFacility().get_observation_url(observation_id)`, confirmed `/requests/<id>` no trailing slash).
- No-churn idempotency: only call `.save()` when fields actually changed.
- `astropy Time.to_datetime()` produces microseconds — strip with `.replace(microsecond=0)` before DB save.
- DB-dependent tests go in `solsys_code/tests/test_sync_lco_observation_calendar.py`, run with `./manage.py test solsys_code`.
- v1.2 real-data bug (drives v1.3): real records have no flat `instrument_type` or `site` key in `parameters`; multi-config `c_1..c_5_instrument_type` shape only, and `SITE_TELESCOPE_MAP` was a 2-entry unverified guess.

### Roadmap Evolution

- Phase 07.1 inserted after Phase 7: Close gap: TELESCOPE-03/04/SYNC-06 — SOAR fallback label is facility-unaware (URGENT)

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| todo | 2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md | pending (deliberately deferred future work) | v1.2 close |
| todo | 2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md | pending (deliberately deferred future work) | v1.3 close |

## Session Continuity

Last session: 2026-06-24T14:34:31.323Z
Stopped at: Completed 07.1-01-PLAN.md (phase 07.1 complete, ready for verification)
Resume file: 
None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
