---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Campaign Coordination for Rare/Urgent Objects
current_phase: 14
status: planning
stopped_at: Phase 14 planned (3 plans, 3 waves) - ready to execute
last_updated: "2026-07-02T20:45:00.000Z"
last_activity: 2026-07-02
last_activity_desc: Phase 14 planned (3 plans across 3 waves); ready to execute
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 3
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-02 — v2.0 milestone opened)

**Core value:** When the next 4I-class object appears, FOMO replaces the ad-hoc Google Sheet as the community's campaign-coordination hub — target-linked observing runs, submission with oversight, and a per-object campaign view.
**Current focus:** Phase 14 — Campaign Data Model & Bootstrap Import (planned, 3 plans across 3 waves, ready to execute)

## Current Position

Phase: 14 — Campaign Data Model & Bootstrap Import (planned, not yet executed)
Plan: 3 plans (14-01, 14-02, 14-03) across 3 waves — see .planning/phases/14-campaign-data-model-bootstrap-import/
Status: Planning complete (research + pattern-mapping + plan-checker all passed); ready to execute Phase 14
Last activity: 2026-07-02 — Phase 14 planned (CampaignRun model, import command, demo notebook)
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

## Accumulated Context

### Decisions

All v1.0-v1.7 decisions logged in PROJECT.md Key Decisions table.

**v2.0 roadmap decisions:**

- Four-phase structure (14-17) for the 19 v1 requirements, aligned with `coarse` granularity. Research suggested a 5-phase split (model+import → table → form+approval → calendar projection → gap); calendar projection (CAL-01..03) was folded into the form+approval phase (Phase 16) because the projection is triggered by the approval action itself and reuses `insert_or_create_calendar_event()` unchanged — it is not a separable deliverable.
- Phase ordering: model+import first (validates schema against real messy CSV before any UI, echoing the v1.2→v1.3 lesson), read path (table) before write path (form) so staff see data working before the public form goes live, calendar projection triggered inside the approval phase, coverage-gap last.
- Phase 17 (coverage-gap, GAP-01/02) ordered last and explicitly deferrable to v2.1 per milestone scope. GAP-01 is a phase-time research spike (dark-window-only vs. target-altitude filtering) that gates GAP-02's approach and the `ephem_utils`/SPICE-cost decision.
- PII policy (contact person/email gated to authenticated staff, verified by anonymous-client test) and demo-notebook PII strategy (synthetic/redacted fixture, CAMP-05) are carried as phase-discussion decisions flagged by research; VIEW-03 and CAMP-05 encode them as hard requirements.

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

Last session: 2026-07-02T19:30:38.483Z
Stopped at: Phase 14 context gathered
Resume file: .planning/phases/14-campaign-data-model-bootstrap-import/14-CONTEXT.md

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 14`
