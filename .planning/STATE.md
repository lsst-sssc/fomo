---
gsd_state_version: "1.0"
milestone: v2.4
milestone_name: Observation-First Calendar
status: Awaiting next milestone
stopped_at: v2.4 milestone completed and archived
last_updated: "2026-10-07T04:07:04.948Z"
last_activity: 2026-10-06
last_activity_desc: Milestone v2.4 completed and archived
state_head: f45e17ee803ad93c736f5371c7fb4908091a5493
progress:
  total_phases: 6
  completed_phases: 39
  total_plans: 79
  completed_plans: 79
  percent: 100
current_phase: "37.1"
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-06 — after the v2.4 milestone)

**Core value:** The calendar is driven by what actually happened — one event per `ObservationRecord`, narrowing on every save with no operator action; allocations project intent nights until a real observation retires them; campaigns annotate, never own.
**Current focus:** Planning the next milestone — v2.4 shipped 2026-10-06; next step is `/gsd-new-milestone`

## Current Position

Phase: Milestone v2.4 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-10-06 — Milestone v2.4 completed and archived

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

Cleared at the v2.4 close; the full decision log is in `.planning/PROJECT.md` (Key Decisions) and the v2.4 phase records under `.planning/milestones/v2.4-phases/`.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first. (Before this close, 7 quick tasks
and 4 debug sessions the audit flagged were found to be complete and closed in f45e17e rather than deferred.)

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| deferred_items | 33/deferred-items.md: flaky `test_observatory_create_form_submits_to_observatory_url` (live MPC call timed out) | acknowledged | 2026-10-06 | v2.4 |
| deferred_items | 37/deferred-items.md: flaky `test_observatory_create_form_submits_to_observatory_url` (order-dependent Playwright failure in full run) | acknowledged | 2026-10-06 | v2.4 |
| seeds | SEED-003 | dormant | 2026-10-06 | v2.4 |
| seeds | SEED-004 | dormant | 2026-10-06 | v2.4 |
| seeds | SEED-261007-5pe | dormant | 2026-10-06 | v2.4 |
| seeds | SEED-261007-j63 | dormant | 2026-10-06 | v2.4 |
| todos | 2026-09-01-add-ttl-cache-to-attribution-banner-count.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-09-30-fetch-lco-observation-blocks-in-bulk-per-proposal.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-02-load-telescope-runs-skip-comment-lines-and-warn-on-a-bare-pr.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-02-run-pre-executed-demo-notebooks-against-a-scratch-db-copy-ne.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-a-failed-or-aborted-record-keeps-its-last-scheduled-window-i.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-decide-whether-campaignrun-run-status-needs-an-awarded-and-i.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-delete-the-reconciler-s-own-stale-run-pk-container-on-a-cont.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-explain-campaignrun-telescope-class-setting-it-on-a-site-res.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-give-the-campaign-gap-analysis-a-start-end-date-control.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-isolate-the-campaign-table-query-count-test-from-the-shared.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-keep-a-request-s-site-restriction-and-show-it-in-the-event-t.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-link-each-campaign-table-row-to-its-run-or-give-the-target-i.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-mark-site-lookups-as-not-attempted-in-project-observation-ca.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-report-system-link-outcomes-in-the-discovery-step-s-tick-sum.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-revisit-a-human-confirmed-allocation-night-is-not-retired-if.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-say-in-watchedproposal-attributed-to-help-text-and-the-runbo.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-show-a-proposal-level-unused-nights-figure-once-per-proposal.md | (presence-only) | 2026-10-06 | v2.4 |

## Session

**Last session:** 2026-10-06T18:28:23.000Z
**Stopped at:** v2.4 milestone completed and archived
**Resume file:** None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
