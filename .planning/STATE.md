---
gsd_state_version: "1.0"
milestone: v2.4
milestone_name: Observation-First Calendar
current_phase: "37.1"
current_phase_name: "Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)"
status: executing
stopped_at: Completed 37.1-16-PLAN.md
last_updated: "2026-10-06T04:32:03.116Z"
state_head: df1adb5bcebea06afde4a35902ab52ae2b74f177
progress:
  total_phases: 6
  completed_phases: 38
  total_plans: 78
  completed_plans: 78
  percent: 100
last_activity: 2026-10-06
last_activity_desc: Phase 37.1 plan 37.1-16 executed (WR-21 code fix, WR-22 wording, WR-23, IN-31 to IN-34); awaiting verification
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-18 — after Phase 36 complete)

**Core value:** The calendar is driven by what actually happened — one event per `ObservationRecord`, narrowing on every save with no operator action; allocations project intent nights until a real observation retires them; campaigns annotate, never own.
**Current focus:** Phase 37.1 — Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)

## Current Position

Phase: 37.1 (Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)) — EXECUTING
Plan: 16 of 16
Status: All plans executed; ALLOC-06 awaits verification and the developer's live-host re-run (UAT tests 6 and 7)

## Performance Metrics

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 37.1 P11 | 38 min | 3 tasks | 7 files |
| Phase 37.1 P12 | 59 min | 3 tasks | 4 files |
| Phase 37.1 P13 | 11 min | 3 tasks | 11 files |
| Phase 37.1 P14 | 11 min | 3 tasks | 12 files |
| Phase 37.1 P15 | 15 min | 3 tasks | 15 files |
| Phase 37.1 P16 | 39 min | 3 tasks | 12 files |

## Decisions

- [Phase 37.1]: 37.1-11: WR-01 fixed (not accepted): a non-list /observations/ reply raises UnexpectedBlockPayloadError on FOMO's facility; select_schedule_block() stays tolerant; supersedes T-37.1-48/A-2 with T-37.1-50
- [Phase 37.1]: 37.1-11: a failed --recheck-unscheduled lookup marks the record only when the ordinary gate would skip it (_failed_lookup_needs_marker); a dry run never marks
- [Phase 37.1]: 37.1-12: WR-01 and IN-04 closed docs-only (developer decision 2026-10-05): runner retry of a failed-recheck record is stated only for an active watched proposal; an unwatched --proposal code is retried by a manual re-run (T-37.1-64 accepted); stderr hint declined
- [Phase 37.1]: 37.1-13 A-17: time_completed counts only as an int or float (not a bool) above 0; numeric strings, None, missing, NaN, zero and negative read as no data
- [Phase 37.1]: 37.1-13 A-18: an embedded FAILED block is judged by its own configuration_statuses; one with none gives no times and no live lookup
- [Phase 37.1]: 37.1-13 A-19: WR-04 marked fixed alongside IN-08..IN-11 (open count 24 -> 19); G-37.1-6 and G-37.1-1-alloc stay failed until the developer's live re-run
- [Phase 37.1]: 37.1-14: placed block wins -- select_schedule_block() returns the last PENDING block before the started tier (developer decision 2026-10-05, WR-19 option a); a record keeps carrying one block
- [Phase 37.1]: 37.1-14 A-24: an IN_PROGRESS block yields to a PENDING block in either order (pinned by test)
- [Phase 37.1]: select_schedule_block() takes keyword-only request_finished (default False); finished means the request state is one of the facility's terminal observing states, so a block that took data outranks a leftover PENDING block at the finishing tick (WR-20)
- [Phase 37.1]: A finished request whose only timed block is a leftover PENDING block keeps that block's times (A-33), flagged for the developer
- [Phase 37.1]: resolve_observed_site() passes is_request_finished(record.status, facility): a completed request is finished (A-35)
- [Phase 37.1]: 37.1-16: WR-21 fixed by code (developer decision 2026-10-05): resolve_placement_block() returns the block whose start is the record's stored scheduled_start, falling back to the rule only when none matches; ties go to the rule among the tied blocks, else the last (A-43, A-44) — The telescope shown on the calendar must belong to the block the record's times came from, whichever rule stored them
- [Phase 37.1]: 37.1-16: WR-22 stays wording only; the per-record correction is update_observation_status() then removing the three observed-site parameters (A-45); ALLOC-06 stays open until the developer's live-host re-run — The recheck flag never revisits a record holding both times; its behaviour is fenced by an AST probe against d6b105b

## Session

**Last session:** 2026-10-06T04:31:46.750Z
**Stopped at:** Completed 37.1-16-PLAN.md
**Resume file:** None
