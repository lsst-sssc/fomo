---
gsd_state_version: "1.0"
milestone: v2.4
milestone_name: Observation-First Calendar
current_phase: "37.1"
current_phase_name: "Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)"
status: executing
stopped_at: Completed 37.1-12-PLAN.md
last_updated: "2026-10-05T14:06:07.045Z"
state_head: 7e8bae08a6a7421813ceb877b8531db0af49e6e0
progress:
  total_phases: 6
  completed_phases: 38
  total_plans: 74
  completed_plans: 74
  percent: 100
last_activity: 2026-10-05
last_activity_desc: Phase 37.1 gaps-only round 5 (37.1-12, docs-only) executed; verification human_needed 102/102
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-18 — after Phase 36 complete)

**Core value:** The calendar is driven by what actually happened — one event per `ObservationRecord`, narrowing on every save with no operator action; allocations project intent nights until a real observation retires them; campaigns annotate, never own.
**Current focus:** Phase 37.1 — Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)

## Current Position

Phase: 37.1 (Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)) — AWAITING HUMAN VERIFICATION
Plan: 12 of 12 (37.1-12 complete)
Status: All 12 plans executed — round-5 verification 2026-10-05 `human_needed` 102/102 (truth 93 closed by 37.1-12; round-5 review 0 critical / 0 warning / 10 info; full suite 2000 + 40 OK). Five human items are tests 6-10 of 37.1-UAT.md: two live-host re-runs, the TOM-untouched judgment check, the pop-up/pager check, and a decision on review notes IN-08 to IN-11. Next: `/gsd-verify-work 37.1`

## Performance Metrics

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 37.1 P11 | 38 min | 3 tasks | 7 files |
| Phase 37.1 P12 | 59 min | 3 tasks | 4 files |

## Decisions

- [Phase 37.1]: 37.1-11: WR-01 fixed (not accepted): a non-list /observations/ reply raises UnexpectedBlockPayloadError on FOMO's facility; select_schedule_block() stays tolerant; supersedes T-37.1-48/A-2 with T-37.1-50
- [Phase 37.1]: 37.1-11: a failed --recheck-unscheduled lookup marks the record only when the ordinary gate would skip it (_failed_lookup_needs_marker); a dry run never marks
- [Phase 37.1]: 37.1-12: WR-01 and IN-04 closed docs-only (developer decision 2026-10-05): runner retry of a failed-recheck record is stated only for an active watched proposal; an unwatched --proposal code is retried by a manual re-run (T-37.1-64 accepted); stderr hint declined

## Session

**Last session:** 2026-10-05T14:06:06.656Z
**Stopped at:** Completed 37.1-12-PLAN.md
**Resume file:** None
