---
gsd_state_version: "1.0"
milestone: v2.4
milestone_name: Observation-First Calendar
current_phase: "37.1"
current_phase_name: "Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)"
status: executing
stopped_at: Completed 37.1-11-PLAN.md
last_updated: "2026-10-05T04:01:42.908Z"
state_head: 92812f9beff4969b573a7a710d170dcd0894fc34
progress:
  total_phases: 6
  completed_phases: 37
  total_plans: 73
  completed_plans: 73
  percent: 100
last_activity: 2026-10-05
last_activity_desc: Completed 37.1-11 (WR-01/WR-02 fixes, WR-03 runbook gap, paired notebook)
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-18 — after Phase 36 complete)

**Core value:** The calendar is driven by what actually happened — one event per `ObservationRecord`, narrowing on every save with no operator action; allocations project intent nights until a real observation retires them; campaigns annotate, never own.
**Current focus:** Phase 37.1 — Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)

## Current Position

Phase: 37.1 (Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)) — EXECUTING
Plan: 11 of 11 (37.1-11 complete)
Status: All 11 plans executed — ready for `/gsd-verify-work` re-verification of Phase 37.1

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
