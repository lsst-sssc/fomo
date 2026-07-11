---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: Uncertain Scheduling & Site Disambiguation
current_phase: 21
current_phase_name: site-disambiguation-submitter-contact-opt-in
status: executing
stopped_at: Phase 21 UI-SPEC approved
last_updated: "2026-07-11T11:49:47.654Z"
last_activity: 2026-07-11
last_activity_desc: Phase 21 execution started
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 14
  completed_plans: 11
  percent: 75
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-05 — v2.1 milestone opened)

**Core value:** Campaign coordination handles the real 3I/ATLAS sheet's harder rows — space-mission observations whose exact observing night isn't known yet, only a window or a still-pending schedule — while closing out submitter contact opt-in (VIEW-05) and a real staff-facing site-disambiguation UI.
**Current focus:** Phase 21 — site-disambiguation-submitter-contact-opt-in

## Current Position

Phase: 21 (site-disambiguation-submitter-contact-opt-in) — EXECUTING
Plan: 2 of 4
Status: Ready to execute
Last activity: 2026-07-11 — Phase 21 execution started

## Roadmap Summary (v2.1)

| Phase | Goal | Requirements |
|-------|------|--------------|
| 18. Uncertain-Scheduling Investigation Spike | Settle window schema, TBD natural key, CSV range/TBD parsing rules, and fuzzy-match library against real 3I sheet rows before implementation | SCHED-01 |
| 19. Window-Schema Migration | Replace single-night `obs_date`/`ut_start`/`ut_end` with a nullable `window_start`/`window_end` pair; migrate existing rows with no data loss | SCHED-02..05 |
| 20. Range/TBD Import & Asset-Aware Coverage Gap | Import range/TBD `Obs. Date` rows into the window representation; make coverage-gap analysis distinguish ground vs. space-mission runs | IMPORT-01..02, ASSET-01..02 |
| 21. Site Disambiguation & Submitter Contact Opt-In | Staff-facing fuzzy-match site-resolution UI in the approval queue; submitter contact opt-in flag | SITE-01..03, VIEW-05 |

Coverage: 13/13 v1 requirements mapped, no orphans.

**Dependency spine:** 18 (spike) → 19 (window migration, largest blast radius) → 20 (import + asset-gap consumers). Phase 21 (site UI + opt-in) is structurally independent of the scheduling work and depends only on Phase 18's fuzzy-library decision — can run in parallel with 19-20.

## Roadmap Summary (v2.0 — shipped 2026-07-05)

| Phase | Goal | Requirements | Deferrable |
|-------|------|--------------|------------|
| 14. Campaign Data Model & Bootstrap Import | `CampaignRun` model + 3I/ATLAS CSV import validated against real data | CAMP-01..05 | No |
| 15. Per-Campaign Table View (Read Path) | Spreadsheet-replacement table of all runs for a campaign, PII-gated | VIEW-01..04 | No |
| 16. Submission Form, Approval Queue & Calendar Projection | Community intake + staff approval gate; approved runs project onto the calendar | SUBMIT-01..05, CAL-01..03 | No |
| 17. Coverage-Gap Analysis | Ephemeris-aware observable-but-unclaimed dates | GAP-01, GAP-02 | **Yes — shipped anyway** |

Coverage: 19/19 v1 requirements mapped, no orphans.

## Performance Metrics

**Velocity:**

- Prior milestone plans completed: 14 (v1.3-v1.4); v1.6 added 3 plans across Phases 11-12; v1.7 shipped Phase 13 (2 plans); v2.0 shipped 13 plans across Phases 14-17
- Average duration: ~15 min/plan (v1.6 range: ~8-24 min)
- Total execution time: see per-phase breakdown in shipped milestone archives

**By Phase (v2.0):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 14 | 3 | - | - |
| 15 | 2 | - | - |
| 16 | 5 | - | - |
| 17 | 3 | - | - |
| Phase 14 P01 | 24min | 3 tasks | 3 files |
| Phase 14 P02 | 6min | 3 tasks | 3 files |
| Phase 14 P03 | 25min | 2 tasks | 2 files |
| Phase 15 P01 | 25min | 3 tasks | 8 files |
| Phase 15 P02 | 15min | 3 tasks | 5 files |
| Phase 16 P01 | 8min | 2 tasks | 4 files |
| Phase 16 P02 | 26min | 2 tasks | 5 files |
| Phase 16 P03 | 21min | 2 tasks | 5 files |
| Phase 16 P04 | 8min | 2 tasks | 4 files |
| Phase 16 P05 | 16min | 2 tasks | 2 files |
| Phase 17 P01 | 21min | 3 tasks | 3 files |
| Phase 17 P02 | 25min | 3 tasks | 5 files |
| Phase 17 P03 | 15min | 3 tasks | 4 files |
| Phase 18 P01 | 32min | 3 tasks | 2 files |
| Phase 18 P02 | 12min | 2 tasks | 3 files |
| 18 | 2 | - | - |
| Phase 19 P01 | 20min | 2 tasks | 3 files |
| Phase 19 P02 | 10min | 2 tasks | 2 files |
| Phase 19 P03 | ~20min | 3 tasks | 6 files |
| Phase 19 P04 | 20min | 2 tasks | 4 files |
| 19 | 4 | - | - |
| Phase 20 P01 | 20min | 2 tasks | 3 files |
| Phase 20 P02 | 10min | 2 tasks | 5 files |
| Phase 20 P03 | 20min | 2 tasks | 3 files |
| Phase 20 P04 | 22min | 1 tasks | 2 files |
| 20 | 4 | - | - |
| Phase 21 P01 | 8min | 3 tasks | 3 files |

## Accumulated Context

### Decisions

All v1.0-v1.7 decisions logged in PROJECT.md Key Decisions table.

**v2.1 roadmap decisions:**

- Four-phase structure (18-21) for the 13 v1 requirements, aligned with `coarse` granularity. The pre-correction research SUMMARY suggested a 7-phase split (spike + obscode-widening + schema + gap + CSV + site + opt-in). Two compressions were applied: (a) the standalone "`Observatory.obscode` max-length widening" phase was dropped entirely — the operator-caught post-research correction established that real space-observatory MPC codes (250/274/289) are standard 3-char codes that already fit `max_length=4`; obscode widening is very likely NOT needed, so it becomes a spike question (default answer: no) inside Phase 18 rather than its own phase; (b) the two schema-consumer phases (CSV range/TBD import + asset-aware gap analysis) were folded into a single Phase 20, since both are consumers of the Phase 19 window schema, can run concurrently, and neither is large enough to stand alone under coarse granularity.
- Phase ordering follows the research dependency spine: spike first (settles window schema, TBD natural key, CSV parsing rules, fuzzy-library choice), then the window-schema migration (largest blast radius) as its own phase before any consumer touches the new schema, then the consumers (import + gap), with the independent site-UI + opt-in track able to run in parallel.
- Phase 21 (SITE-01..03 + VIEW-05) grouped together because both are structurally independent of the scheduling-representation work — they touch `Observatory` resolution and the submission form, not the window schema. Per research, the `CampaignRunDecisionView.post()` re-resolution guard (SITE-03) must ship in the same phase as the new fuzzy-match resolution UI (SITE-01/02), never split.
- `Observatory.obscode` length widening is explicitly out of scope (Out of Scope table in REQUIREMENTS.md) unless the Phase 18 spike finds a real code that doesn't fit.

**Carried from v2.0 (still live for v2.1):**

- [Phase 14]: parse_obs_window uses three narrowly-scoped regexes (not a permissive general date parser) so a stray date-range or garbage UT Time Range cell never succeeds into a wrong-but-plausible time — Phase 20's range/TBD parsing must extend this pattern-per-shape approach, not replace it with a generic parser.
- [Phase 14]: resolve_site length-checks and blank-checks the raw Site Code before any tier attempt, flagging oversized/blank codes for review with site=None rather than fabricating (D-08/D-09/Pitfall 2) — quick task 260705-l1v extended this "never fabricate" invariant; Phase 21's fuzzy-match UI must preserve it (never auto-select).
- [Phase 17]: Multi-target campaign target=None CampaignRuns are collected into a separate unattributed_runs list, never counted as claiming either target's date — Phase 20's asset-aware claimed_dates() rewrite must preserve this bucketing.
- [Phase 18]: rapidfuzz package legitimacy confirmed by human (Task 1 checkpoint approved) — the automated SUS verdict was a documented download-lookup false-positive
- [Phase 18]: probe script fuzzy_match_probe.py never staged/committed (per D-08); only 18-DECISION.md is a committed deliverable from this plan
- [Phase 18]: resolve_site() cannot currently resolve 250/274/289 via its MPC Tier 2 path due to a null-longitude TypeError in MPCObscodeFetcher.to_observatory() for satellite-type MPC records (real live-test finding, flagged for Phase 19/21 awareness, no fix in this phase)
- [Phase 18]: SCHED-01 criterion 1 (window schema): confirmed as-is - nullable window_start/window_end DateField pair, validated against every real cell shape
- [Phase 18]: SCHED-01 criterion 4 (fuzzy library): split verdict - difflib primary/default, rapidfuzz deferred until a real advantage is demonstrated
- [Phase 18]: SCHED-01 criterion 5 (obscode widening): no widening needed, confirmed against live Observatory.obscode max_length=4
- [Phase 18]: SCHED-01 criterion 2 (TBD natural key): fold contact_person into the natural key for null-window rows via a partial/conditional UniqueConstraint (Phase 19 to design mechanism)
- [Phase 18]: SCHED-01 criterion 3 (CSV range/TBD parsing): extend parse_obs_window()'s pattern-per-shape discipline to Obs. Date, checking both Obs. Date and UT Time Range (Phase 20 to implement)
- [Phase 19]: Resolved-window UniqueConstraint keys on all four fields (campaign, telescope_instrument, window_start, window_end), not window_start alone, so a future date range starting the same day as an existing single-night row won't false-collide
- [Phase 19]: TBD UniqueConstraint deliberately excludes window_start/window_end from its fields tuple (always NULL under its own condition) and keys on contact_person instead
- [Phase 19]: [Phase 19] Deleted _observing_night_date() outright rather than deprecating it -- window_start/window_end are already plain dates, no time-of-day-to-night-boundary conversion needed
- [Phase 19]: [Phase 19] Deleted test_ut_start_only_keys_to_site_local_observing_night rather than renaming it -- the code path it tested no longer exists under the window schema
- [Phase 19]: render_window_start() returns the literal '-&gt;' HTML entity (not a plain hyphen/en-dash) for a range row, per D-05's exact wording
- [Phase 19]: TestApproval/TestCalendarNoChurn each got their own scoped ground-Observatory fixture (not added to the shared CampaignApprovalTestBase) so D-06's site-required calendar projection doesn't break TestApprovalSiteResolution's Observatory.objects.count()==0 assertions
- [Phase 19]: [Phase 19] import_campaign_csv's window-key collision check now runs for every row (not just ut_needs_review fallback rows), since window_start collapses to date granularity -- any same-telescope/same-date pair collides on the natural key — window_start is a DateField; time-of-day no longer disambiguates the natural key
- [Phase 19]: [Phase 19] Applied migration 0004_campaignrun_window_schema to the real dev DB (src/fomo_db.sqlite3), previously deferred by Plan 01, so the import_campaign_csv_demo notebook could execute against the live schema — Notebook connects directly to the dev DB, not a test DB; backfill/dedup outcome matched Plan 01's smoke test exactly (16 -> 14 rows)
- [Phase 19]: [Phase 19] Demo notebook's approval-lifecycle cell switched from unconditional CampaignRun.objects.create() to update_or_create() keyed on (campaign, telescope_instrument, contact_person) — Plan 01's new partial TBD UniqueConstraint on those same fields made repeat notebook execution crash with IntegrityError once the dev DB was actually migrated
- [Phase 20]: claimed_dates() computes ground-vs-space classification once from the site parameter before the per-run loop (is_space_mission), never per-row from run.site, preserving the PII-minimizing .only('pk','window_start','window_end') queryset — Avoids N+1 reads and keeps the existing PII-minimization invariant intact (Pitfall 3)
- [Phase 20]: pending_narrowing_runs is a distinct bucket from undated_runs -- TBD runs always land in undated_runs regardless of site type; only a space-mission run with an un-narrowed range lands in pending_narrowing_runs — D-09 explicit distinction between 'no info at all' and 'a real space-mission run with a range, just not scheduled tight enough yet'
- [Phase 20]: original_obs_date_raw is CharField(max_length=255), not TextField -- mirrors site_raw exactly per RESEARCH field-type resolution
- [Phase 20]: render_window_start()'s TBD-badge tooltip resolves original_obs_date_raw via Accessor(...).resolve(record, quiet=True) or '', guarding for blank, and interpolates via format_html positional argument (never mark_safe/f-string) -- reusable pattern for future dict-vs-model tooltips on this table
- [Phase 20]: parse_obs_window() order-of-attempts parser (exact date -> full-date range regex -> compact-range regex with rollover -> TBD catch-all) never raises for any Obs. Date input (D-13) -- stdlib date()'s own ValueError validation is the never-raise mechanism, no manual day-count guard needed
- [Phase 20]: import_campaign_csv's natural key branches on window_start is None (resolved-window vs TBD lookup/collision-key shape), matching CampaignRun.Meta.constraints' two partial UniqueConstraints exactly -- contact_person is pulled out of the unconditional fields dict and only set via lookup for the TBD branch
- [Phase 20]: range/TBD rows skip UT-Time-Range parsing entirely (ut_start=ut_end=None, ut_needs_review=False) per RESEARCH.md's A1 assumption -- no current CampaignRun field stores these values for a multi-night window
- [Phase 20]: Demo notebook's new range/TBD demonstration cell placed after the existing generic inspection cell (not immediately after the import call_command cell) so it reuses the notebook's already-established CampaignRun.objects.get(...) query style
- [Phase 21]: [Phase 21 P01]: old_names included as one whole string (not split) per RESEARCH.md Open Question 2 recommendation
- [Phase 21]: [Phase 21 P01]: local Observatory candidates merge in after the MPC pool via dict.setdefault(), so an already-vetted local record's display string wins any first-seen collision over raw MPC bulk data

### Pending Todos

- `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` — rename
  `calendar_utils.py`'s cross-module-consumed underscore-prefixed helpers
  (`_derive_telescope`, `_extract_instrument`, `_resolve_placement_block`,
  `_coarse_telescope_label`, `_aperture_class_from_telescope_code`) to reflect that the
  module is now a real shared API (3 consumers); low-priority style cleanup found while
  verifying the 2026-06-23 extraction todo was complete.

- Carried-forward items in Deferred Items below.

### Blockers/Concerns

None. Roadmap created; Phase 18 ready to plan via `/gsd-plan-phase 18`.

### Quick Tasks Completed

| # | Description | Date | Commit | Status | Directory |
|---|-------------|------|--------|--------|-----------|
| 260705-l1v | Fix approval-queue site-visibility gap: show site_raw in the pending CampaignRun approval queue and stop the approval endpoint from fabricating placeholder Observatory rows for unresolvable free-text site names (found during v2.0 manual UAT) | 2026-07-05 | 959a78d | Verified | [260705-l1v-fix-approval-queue-site-visibility-gap-s](./quick/260705-l1v-fix-approval-queue-site-visibility-gap-s/) |

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| requirement | ESO-10 (`sync_eso_observation_calendar` command) | v2 — unblocked by Phase 13's Bypass verdict; out of scope for v2.1 (unrelated to uncertain scheduling) | v1.7 close |
| requirement | ESO-11 (paired ESO demo notebook) | v2 — unblocked by Phase 13's Bypass verdict; out of scope for v2.1 | v1.7 close |
| requirement | SCHED-06 (progressive-disclosure window-narrowing UI) | v2 — deferred until the window schema is proven against real re-imported data | v2.1 requirements |
| requirement | SUBMIT-06/07 (trusted-PI self-approval; submission status lookup) | v2 — not committed to a milestone | v2.0 close |
| todo | `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — extract site/telescope mapping and instrument extraction into own module | Deliberately deferred; no second consumer yet | v1.7 close |
| todo | `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` — rename `calendar_utils.py`'s private helpers to reflect shared-module status | Low-priority style cleanup; no functional impact | v2.0 close |
| seed | SEED-001 — file upstream `tom_eso` feature requests | Still dormant | v2.0 close |
| seed | SEED-002 — ESO ObservationRecord-centric future intent | Still dormant | v2.0 close |

## Session Continuity

Last session: 2026-07-11T11:46:15.658Z
Stopped at: Phase 21 UI-SPEC approved
Resume file: .planning/phases/21-site-disambiguation-submitter-contact-opt-in/21-UI-SPEC.md

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 18`
