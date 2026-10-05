---
gsd_state_version: "1.0"
milestone: v2.4
milestone_name: Observation-First Calendar
current_phase: "37.1"
current_phase_name: "Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)"
status: executing
stopped_at: Completed 37.1-14-PLAN.md
last_updated: "2026-10-05T22:26:16.968Z"
state_head: 203d99e8cd3cac170241fe638d80226cdb2362f1
progress:
  total_phases: 6
  completed_phases: 38
  total_plans: 76
  completed_plans: 76
  percent: 100
last_activity: 2026-10-05
last_activity_desc: Phase 37.1 gaps-only round 7 (37.1-14) executed; verification gaps_found 131/132 (observation_blocks.py module docstring, IN-30); WR-20 awaits a developer decision
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-18 — after Phase 36 complete)

**Core value:** The calendar is driven by what actually happened — one event per `ObservationRecord`, narrowing on every save with no operator action; allocations project intent nights until a real observation retires them; campaigns annotate, never own.
**Current focus:** Phase 37.1 — Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)

## Current Position

Phase: 37.1 (Close gap: ALLOC-06 — exact-identity system links on ingest (intent review Q1) (INSERTED)) — VERIFICATION GAPS
Plan: 14 of 14
Status: All 14 plans executed — round-7 verification 2026-10-05 `gaps_found` 131/132 (one docs gap: the observation_blocks.py module docstring, lines 9-11, still says a block that stopped early retires its night exactly like a COMPLETED one; review IN-30). Round-7 review 0 critical / 1 warning / 2 info; WR-20 (the placed-block rule reads block states only, so a request that expires while the portal still lists a never-run PENDING block would keep that block's night) needs a developer decision. Full suite 2030 + 40 OK at d8e4f62; security 80/80 threats closed; Nyquist validated (40 tasks); UI review 24/24 unchanged. ALLOC-06 reverted from Complete. G-37.1-6 and G-37.1-1-alloc stay failed until the developer's live-host re-run (UAT test 6, then test 7). Next: `/gsd-plan-phase 37.1 --gaps`

## Performance Metrics

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 37.1 P11 | 38 min | 3 tasks | 7 files |
| Phase 37.1 P12 | 59 min | 3 tasks | 4 files |
| Phase 37.1 P13 | 11 min | 3 tasks | 11 files |
| Phase 37.1 P14 | 11 min | 3 tasks | 12 files |

## Decisions

- [Phase 37.1]: 37.1-11: WR-01 fixed (not accepted): a non-list /observations/ reply raises UnexpectedBlockPayloadError on FOMO's facility; select_schedule_block() stays tolerant; supersedes T-37.1-48/A-2 with T-37.1-50
- [Phase 37.1]: 37.1-11: a failed --recheck-unscheduled lookup marks the record only when the ordinary gate would skip it (_failed_lookup_needs_marker); a dry run never marks
- [Phase 37.1]: 37.1-12: WR-01 and IN-04 closed docs-only (developer decision 2026-10-05): runner retry of a failed-recheck record is stated only for an active watched proposal; an unwatched --proposal code is retried by a manual re-run (T-37.1-64 accepted); stderr hint declined
- [Phase 37.1]: 37.1-13 A-17: time_completed counts only as an int or float (not a bool) above 0; numeric strings, None, missing, NaN, zero and negative read as no data
- [Phase 37.1]: 37.1-13 A-18: an embedded FAILED block is judged by its own configuration_statuses; one with none gives no times and no live lookup
- [Phase 37.1]: 37.1-13 A-19: WR-04 marked fixed alongside IN-08..IN-11 (open count 24 -> 19); G-37.1-6 and G-37.1-1-alloc stay failed until the developer's live re-run
- [Phase 37.1]: 37.1-14: placed block wins -- select_schedule_block() returns the last PENDING block before the started tier (developer decision 2026-10-05, WR-19 option a); a record keeps carrying one block
- [Phase 37.1]: 37.1-14 A-24: an IN_PROGRESS block yields to a PENDING block in either order (pinned by test)

## Session

**Last session:** 2026-10-05T22:26:16.859Z
**Stopped at:** Completed 37.1-14-PLAN.md
**Resume file:** None
