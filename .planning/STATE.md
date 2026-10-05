---
gsd_state_version: "1.0"
milestone: v2.4
milestone_name: Observation-First Calendar
current_phase: "37.1"
current_phase_name: "Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)"
status: executing
stopped_at: Completed 37.1-11-PLAN.md
last_updated: "2026-10-05T12:26:31.898Z"
state_head: 9a1c3923dccb482e8807c37e71638cd0ae14fc31
progress:
  total_phases: 6
  completed_phases: 37
  total_plans: 74
  completed_plans: 73
  percent: 99
last_activity: 2026-10-05
last_activity_desc: Phase 37.1 gaps-only round 4 (37.1-11) executed; verification gaps_found 92/93
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-18 — after Phase 36 complete)

**Core value:** The calendar is driven by what actually happened — one event per `ObservationRecord`, narrowing on every save with no operator action; allocations project intent nights until a real observation retires them; campaigns annotate, never own.
**Current focus:** Phase 37.1 — Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)

## Current Position

Phase: 37.1 (Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)) — READY TO EXECUTE
Plan: 11 of 11 (37.1-11 complete)
Status: All 11 plans executed — round-4 verification 2026-10-05 `gaps_found` 92/93 (docs-only: runbook recheck paragraph + help/docstrings/notebook cell 39 promise the runner retries a failed recheck lookup unconditionally; false for an unwatched `--proposal`, 37.1-REVIEW round-4 WR-01). Next: `/gsd-plan-phase 37.1 --gaps` (or `/gsd-code-review 37.1 --fix` for WR-01)

## Performance Metrics

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 37.1 P11 | 38 min | 3 tasks | 7 files |

## Decisions

- [Phase 37.1]: 37.1-11: WR-01 fixed (not accepted): a non-list /observations/ reply raises UnexpectedBlockPayloadError on FOMO's facility; select_schedule_block() stays tolerant; supersedes T-37.1-48/A-2 with T-37.1-50
- [Phase 37.1]: 37.1-11: a failed --recheck-unscheduled lookup marks the record only when the ordinary gate would skip it (_failed_lookup_needs_marker); a dry run never marks

## Session

**Last session:** 2026-10-05T04:01:42.500Z
**Stopped at:** Completed 37.1-11-PLAN.md
**Resume file:** None
