---
gsd_state_version: "1.0"
milestone: v2.4
milestone_name: Observation-First Calendar
current_phase: "37.1"
current_phase_name: "Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)"
status: verifying
stopped_at: Completed 37.1-13-PLAN.md
last_updated: "2026-10-05T18:20:07.483Z"
state_head: 1db93a4364b8062bf8b182d7eb7ea301599ac97c
progress:
  total_phases: 6
  completed_phases: 38
  total_plans: 75
  completed_plans: 75
  percent: 100
last_activity: 2026-10-05
last_activity_desc: Phase 37.1 gap-closure round 6 (37.1-13, G-37.1-6 FAILED-with-data blocks) executed; awaiting the developer's live-host re-run
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-18 — after Phase 36 complete)

**Core value:** The calendar is driven by what actually happened — one event per `ObservationRecord`, narrowing on every save with no operator action; allocations project intent nights until a real observation retires them; campaigns annotate, never own.
**Current focus:** Phase 37.1 — Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)

## Current Position

Phase: 37.1 (Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)) — EXECUTING
Plan: 13 of 13
Status: Phase complete — ready for verification. 37.1-13 complete (FAILED-with-data blocks counted, IN-08..IN-11 and WR-04 fixed); G-37.1-6 and G-37.1-1-alloc stay failed until the developer's live-host re-run (UAT test 6, then test 7)

## Performance Metrics

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 37.1 P11 | 38 min | 3 tasks | 7 files |
| Phase 37.1 P12 | 59 min | 3 tasks | 4 files |
| Phase 37.1 P13 | 11 min | 3 tasks | 11 files |

## Decisions

- [Phase 37.1]: 37.1-11: WR-01 fixed (not accepted): a non-list /observations/ reply raises UnexpectedBlockPayloadError on FOMO's facility; select_schedule_block() stays tolerant; supersedes T-37.1-48/A-2 with T-37.1-50
- [Phase 37.1]: 37.1-11: a failed --recheck-unscheduled lookup marks the record only when the ordinary gate would skip it (_failed_lookup_needs_marker); a dry run never marks
- [Phase 37.1]: 37.1-12: WR-01 and IN-04 closed docs-only (developer decision 2026-10-05): runner retry of a failed-recheck record is stated only for an active watched proposal; an unwatched --proposal code is retried by a manual re-run (T-37.1-64 accepted); stderr hint declined
- [Phase 37.1]: 37.1-13 A-17: time_completed counts only as an int or float (not a bool) above 0; numeric strings, None, missing, NaN, zero and negative read as no data
- [Phase 37.1]: 37.1-13 A-18: an embedded FAILED block is judged by its own configuration_statuses; one with none gives no times and no live lookup
- [Phase 37.1]: 37.1-13 A-19: WR-04 marked fixed alongside IN-08..IN-11 (open count 24 -> 19); G-37.1-6 and G-37.1-1-alloc stay failed until the developer's live re-run

## Session

**Last session:** 2026-10-05T18:20:07.373Z
**Stopped at:** Completed 37.1-13-PLAN.md
**Resume file:** None
