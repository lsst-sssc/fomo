---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Campaign Coordination for Rare/Urgent Objects
current_phase: 14
current_phase_name: campaign-data-model-bootstrap-import
status: verifying
stopped_at: Completed 14-03-PLAN.md (CAMP-05 demo notebook + fixture)
last_updated: "2026-07-03T07:43:55.177Z"
last_activity: 2026-07-02
last_activity_desc: Phase 14 execution started
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 3
  completed_plans: 3
  percent: 25
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-02 — v2.0 milestone opened)

**Core value:** When the next 4I-class object appears, FOMO replaces the ad-hoc Google Sheet as the community's campaign-coordination hub — target-linked observing runs, submission with oversight, and a per-object campaign view.
**Current focus:** Phase 14 — campaign-data-model-bootstrap-import

## Current Position

Phase: 14 (campaign-data-model-bootstrap-import) — EXECUTING
Plan: 3 of 3
Status: Phase complete — ready for verification
Last activity: 2026-07-02 — Phase 14 execution started
Progress: [░░░░░░░░░░] 0/4 phases

## Roadmap Summary (v2.0)

| Phase | Goal | Requirements | Deferrable |
|-------|------|--------------|------------|
| 14. Campaign Data Model & Bootstrap Import | `CampaignRun` model + 3I/ATLAS CSV import validated against real data | CAMP-01..05 | No |
| 15. Per-Campaign Table View (Read Path) | Spreadsheet-replacement table of all runs for a campaign, PII-gated | VIEW-01..04 | No |
| 16. Submission Form, Approval Queue & Calendar Projection | Community intake + staff approval gate; approved runs project onto the calendar | SUBMIT-01..05, CAL-01..03 | No |
| 17. Coverage-Gap Analysis | Ephemeris-aware observable-but-unclaimed dates | GAP-01, GAP-02 | **Yes — to v2.1** |

Coverage: 19/19 v1 requirements mapped, no orphans.

## Performance Metrics

**Velocity:**

- Prior milestone plans completed: 14 (v1.3-v1.4); v1.6 added 3 plans across Phases 11-12; v1.7 shipped Phase 13 (2 plans)
- Average duration: ~15 min/plan (v1.6 range: ~8-24 min)
- Total execution time: see per-phase breakdown in shipped milestone archives

**By Phase (v2.0):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 14 | TBD | - | - |
| 15 | TBD | - | - |
| 16 | TBD | - | - |
| 17 | TBD | - | - |
| Phase 14 P01 | 24min | 3 tasks | 3 files |
| Phase 14 P02 | 6min | 3 tasks | 3 files |
| Phase 14 P03 | 25min | 2 tasks | 2 files |

## Accumulated Context

### Decisions

All v1.0-v1.7 decisions logged in PROJECT.md Key Decisions table.

**v2.0 roadmap decisions:**

- Four-phase structure (14-17) for the 19 v1 requirements, aligned with `coarse` granularity. Research suggested a 5-phase split (model+import → table → form+approval → calendar projection → gap); calendar projection (CAL-01..03) was folded into the form+approval phase (Phase 16) because the projection is triggered by the approval action itself and reuses `insert_or_create_calendar_event()` unchanged — it is not a separable deliverable.
- Phase ordering: model+import first (validates schema against real messy CSV before any UI, echoing the v1.2→v1.3 lesson), read path (table) before write path (form) so staff see data working before the public form goes live, calendar projection triggered inside the approval phase, coverage-gap last.
- Phase 17 (coverage-gap, GAP-01/02) ordered last and explicitly deferrable to v2.1 per milestone scope. GAP-01 is a phase-time research spike (dark-window-only vs. target-altitude filtering) that gates GAP-02's approach and the `ephem_utils`/SPICE-cost decision.
- PII policy (contact person/email gated to authenticated staff, verified by anonymous-client test) and demo-notebook PII strategy (synthetic/redacted fixture, CAMP-05) are carried as phase-discussion decisions flagged by research; VIEW-03 and CAMP-05 encode them as hard requirements.
- [Phase 14]: CampaignRun.campaign FK uses on_delete=PROTECT (not CASCADE/SET_NULL) since it's required (null=False) -- prevents accidental loss of campaign history if a TargetList is ever deleted
- [Phase 14]: site_raw stored as CharField(max_length=255), not TextField, matching Observatory.name/short_name convention for short strings
- [Phase 14]: No DB-level UniqueConstraint on the natural key -- follows CalendarEvent precedent of app-level get_or_create only (deferred to Plan 02's import command)
- [Phase 14]: resolve_site length-checks and blank-checks the raw Site Code before any tier attempt, so an oversized/blank code is flagged for review with site=None rather than truncated or fabricated (D-08/D-09/Pitfall 2)
- [Phase 14]: parse_obs_window uses three narrowly-scoped regexes (not a permissive general date parser) so a stray date-range or garbage UT Time Range cell never succeeds into a wrong-but-plausible time
- [Phase 14]: insert_or_create_campaign_run omits 'modified' from update_fields since CampaignRun has no auto-now timestamp field, unlike insert_or_create_calendar_event
- [Phase 14]: Demo notebook seeds real MPC obscodes (F65/309/705) locally via update_or_create so import_campaign_csv's site resolution never makes a live MPC API call, matching load_telescope_runs_demo.ipynb's established seeding convention
- [Phase 14]: Approval-lifecycle demo cell constructs CampaignRun rows directly via .objects.create() (not the CSV import, which always writes approved per D-03) to exercise pending_review -> approved/rejected

### Pending Todos

- `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` — rename
  `calendar_utils.py`'s cross-module-consumed underscore-prefixed helpers
  (`_derive_telescope`, `_extract_instrument`, `_resolve_placement_block`,
  `_coarse_telescope_label`, `_aperture_class_from_telescope_code`) to reflect that the
  module is now a real shared API (3 consumers); low-priority style cleanup found while
  verifying the 2026-06-23 extraction todo was complete.

- Carried-forward items in Deferred Items below.

### Blockers/Concerns

None. Roadmap created; Phase 14 ready to plan via `/gsd-plan-phase 14`.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| requirement | ESO-10 (`sync_eso_observation_calendar` command) | v2 — unblocked by Phase 13's Bypass verdict; out of scope for v2.0 (unrelated to campaign coordination) | v1.7 close |
| requirement | ESO-11 (paired ESO demo notebook) | v2 — unblocked by Phase 13's Bypass verdict; out of scope for v2.0 | v1.7 close |
| todo | `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — extract site/telescope mapping and instrument extraction into own module | Deliberately deferred; no second consumer yet | v1.7 close |
| seed | SEED-001 — file upstream `tom_eso` feature requests | Dormant; trigger is TOM Toolkit maintainer bandwidth or a future ESO milestone start | v1.7 close |
| seed | SEED-002 — ESO ObservationRecord-centric future intent | Dormant; unrelated to v2.0 campaign-coordination scope | v1.7 close |

## Session Continuity

Last session: 2026-07-03T07:43:55.168Z
Stopped at: Completed 14-03-PLAN.md (CAMP-05 demo notebook + fixture)
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 14`
