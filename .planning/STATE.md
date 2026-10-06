---
gsd_state_version: "1.0"
milestone: v2.4
milestone_name: Observation-First Calendar
current_phase: "37.1"
status: completed
stopped_at: Phase 37.1 complete — all phases complete
last_updated: "2026-10-06T18:27:14.920Z"
state_head: fa94cc6cb9cb03f5d5929fcd6cbab3c8bcec18c3
progress:
  total_phases: 6
  completed_phases: 39
  total_plans: 79
  completed_plans: 79
  percent: 100
last_activity: 2026-10-06
last_activity_desc: Phase 37.1 complete — round-11 verification passed 186/186; UAT complete (16 tests, every gap resolved); ALLOC-06 Complete; milestone v2.4 ready for /gsd-complete-milestone
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-06 — after Phase 37.1 complete)

**Core value:** The calendar is driven by what actually happened — one event per `ObservationRecord`, narrowing on every save with no operator action; allocations project intent nights until a real observation retires them; campaigns annotate, never own.
**Current focus:** Milestone v2.4 — all phases complete; next step is `/gsd-complete-milestone v2.4`

## Current Position

Phase: 37.1
Plan: Not started
Status: All phases complete

## Performance Metrics

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 37.1 P11 | 38 min | 3 tasks | 7 files |
| Phase 37.1 P12 | 59 min | 3 tasks | 4 files |
| Phase 37.1 P13 | 11 min | 3 tasks | 11 files |
| Phase 37.1 P14 | 11 min | 3 tasks | 12 files |
| Phase 37.1 P15 | 15 min | 3 tasks | 15 files |
| Phase 37.1 P16 | 39 min | 3 tasks | 12 files |
| Phase 37.1 P17 | 13 min | 3 tasks | 8 files |

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
- [Phase 37.1]: 37.1-17: WR-24 fixed by "Add fallback record" (developer decision 2026-10-06): backfill notebook cell d4a7c2e1 passes every record stored start and record 900664 shows the no-match fallback with executed output; WR-25 accepted as is (ledger skipped); IN-35 to IN-38 fixed as wording only; ALLOC-06 stays open pending the live-host re-run
- [Phase 37.1]: UAT 2026-10-06: live Didymos re-run found 4 blocks and ALLOC:1:* events went 14 -> 10 (test 11), closing G-37.1-6 and G-37.1-1-alloc; ALLOC-06 Complete
- [Phase 37.1]: UAT 2026-10-06: A-24 and A-33 acknowledged as they stand (test 14); projector cell 7e7bd66e's pointer amended to name the no-match record and the notebook re-executed by the developer (tests 15-16, a842a97, IN-44 fixed)
- [Phase 37.1]: A resolved UAT gap names one plan file in resolved_by (e.g. 37.1-13-PLAN.md); the list form is not read by the completion check, other fix plans go in also_fixed_by

## Session

**Last session:** 2026-10-06T18:28:23.000Z
**Stopped at:** Phase 37.1 complete — all v2.4 phases complete, ready to complete the milestone
**Resume file:** None
