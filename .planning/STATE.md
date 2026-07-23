---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: Uncertain Scheduling & Site Disambiguation
status: Awaiting next milestone
stopped_at: "Completed quick task 260723-02e: Add --campaign flag to load_telescope_runs, associating CalendarEvent.target_list with a TargetList"
last_updated: "2026-07-23T07:25:25.179Z"
last_activity: 2026-07-23
last_activity_desc: "Completed quick task 260722-uyz: Populate CalendarEvent.target_list in sync_lco_observation_calendar from the record's Target's campaign TargetList membership"
current_phase: 25
current_phase_name: e.g. Gemini FT-115-style awarded allocations
progress:
  total_phases: 8
  completed_phases: 8
  total_plans: 26
  completed_plans: 26
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-18 — v2.1 milestone closed, full evolution review performed)

**Core value:** Campaign coordination handles the real 3I/ATLAS sheet's harder rows — space-mission observations whose exact observing night isn't known yet, only a window or a still-pending schedule — while closing out submitter contact opt-in (VIEW-05) and a real staff-facing site-disambiguation UI.
**Current focus:** Awaiting next milestone — v2.1 shipped 2026-07-18 (8 phases, 26 plans, 13/13 v1 requirements)

## Current Position

Phase: Milestone v2.1 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-07-23 - Completed quick task 260722-uyz: Populate CalendarEvent.target_list in sync_lco_observation_calendar from the record's Target's campaign TargetList membership

## Roadmap Summary (v2.1 — shipped 2026-07-18)

| Phase | Goal | Requirements |
|-------|------|--------------|
| 18. Uncertain-Scheduling Investigation Spike | Settle window schema, TBD natural key, CSV range/TBD parsing rules, and fuzzy-match library against real 3I sheet rows before implementation | SCHED-01 |
| 19. Window-Schema Migration | Replace single-night `obs_date`/`ut_start`/`ut_end` with a nullable `window_start`/`window_end` pair; migrate existing rows with no data loss | SCHED-02..05 |
| 20. Range/TBD Import & Asset-Aware Coverage Gap | Import range/TBD `Obs. Date` rows into the window representation; make coverage-gap analysis distinguish ground vs. space-mission runs | IMPORT-01..02, ASSET-01..02 |
| 21. Site Disambiguation & Submitter Contact Opt-In | Staff-facing fuzzy-match site-resolution UI in the approval queue; submitter contact opt-in flag | SITE-01..03, VIEW-05 |
| 22. Site Matching at Submission & Unmatched-Site Resolution | Live in-browser fuzzy search (public form + approval queue) and a "Sites Needing Review" resolution surface — closes the Phase 21 functionality gap | none mapped (added mid-milestone) |
| 23. Weather/Storm Cancellation Handling | Staff can mark an approved run cancelled/weathered; calendar syncs `[CANCELLED]`/`[WEATHERED]` in place | none mapped (organic phase) |
| 24. Operator and Usage Runbook Documentation | Task-oriented Sphinx runbook for all five management commands + Phase 23 staff actions | none mapped (docs-only) |
| 25. Range-Window CalendarEvent Projection | Approved range-window runs (e.g. real Gemini FT-115) project per-night calendar events instead of staying invisible; backfill command for already-approved runs | none mapped (gap-closure phase from `/gsd-debug`) |

Coverage: 13/13 v1 requirements mapped, no orphans. Full phase detail archived at `.planning/milestones/v2.1-ROADMAP.md`; requirements archived at `.planning/milestones/v2.1-REQUIREMENTS.md`.

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
| Phase 21 P02 | 14min | 3 tasks | 7 files |
| Phase 21 P03 | 21min | 3 tasks | 3 files |
| Phase 21 P04 | 13min | 2 tasks | 3 files |
| 21 | 4 | - | - |
| Phase 22 P01 | 20min | - tasks | - files |
| Phase 22 P01 | 20min | 2 tasks | 5 files |
| Phase 22 P02 | 15min | 2 tasks | 4 files |
| Phase 22 P03 | 35min | 2 tasks | 4 files |
| 22 | 6 | - | - |
| 23 | 3 | - | - |
| 25 | 2 | - | - |
| 24 | 1 | - | - |
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 23 P01 | 15min | 2 tasks | 3 files |
| Phase 23 P02 | 10min | 3 tasks | 5 files |
| Phase 25 P01 | 25min | 3 tasks | 2 files |
| Phase 25 P02 | 20min | 2 tasks | 2 files |
| Phase 24 P01 | 10min | 3 tasks | 3 files |
| Phase quick-260722-tkt P01 | 25min | 3 tasks | 2 files |
| Phase quick-260722-twe P01 | 4min | 2 tasks | 2 files |
| Phase quick-260722-uhh P01 | 12min | 2 tasks | 2 files |
| Phase quick-260722-ux0 P01 | ~15min | 3 tasks | 2 files |
| Phase quick-260722-uyz P01 | ~20min | 3 tasks | 3 files |
| Phase 260723-02e P01 | 22min | 3 tasks | 3 files |

## Accumulated Context

### Roadmap Evolution

- Phase 22 added (2026-07-14): Site Matching at Submission and Unmatched-Site Resolution Workflow — closes the Phase 21 functionality gap. Decisions confirmed with operator: (a) the public submission form's Observing site field gets HTMX live-search autocomplete (new endpoint running `fuzzy_match_candidates()` over `build_site_candidates()`), also replacing the approval queue's static per-row datalist; (b) "site failure never blocks approval" is kept, with a new "Sites needing review" surface for approved runs with `site_needs_review=True` whose resolution triggers the deferred CalendarEvent projection.
- Phase 24 added (2026-07-17): Operator and usage runbook documentation for the telescope-runs-calendar management commands and staff workflows (load_telescope_runs, sync_lco_observation_calendar, sync_gemini_observation_calendar, import_campaign_csv, Phase 23's approval-queue status-change actions) — raised during PR #41/#43 split review: design docs (docs/design/*.rst) and demo notebooks existed, but no general, discoverable how-to-run documentation did. Scoped to publish operator-facing usage docs beyond design rationale and `--help` text.
- Phase 25 added (2026-07-17): Range-window CalendarEvent projection — closes the diagnosed gap where approved, site-resolved range-window CampaignRuns (e.g. the real GS-2026A-FT-115 Gemini FT allocation) never get a CalendarEvent, verified via `/gsd-debug` (`.planning/debug/range-window-calendar-event.md`, diagnose-only, root cause + before/after spec, no code changed). Root cause: Phase 19 D-06's guard was a behavior-preservation deferral, not a considered decision; Phase 23's `TestGeminiFtScenario` re-encoded the deferred behavior as contract. Fix scope per the spec: drop the guard's `window_start == window_end` clause (add a `window_end` truthiness check instead), give the ground branch multi-day date-math (satellite branch is already correct), and deliberately revise the 4 Phase 19/23 test assertions that currently assert zero events for range runs.

### Decisions

All v1.0-v2.1 decisions logged in PROJECT.md Key Decisions table. The exhaustive per-plan v2.1 decision log previously kept here (roadmap-structure decisions, and one bullet per Phase 18-25 plan) has been cleared now that v2.1 has shipped and closed — nothing is lost: the milestone-level decisions are summarized in PROJECT.md's Key Decisions table (backfilled at close for Phases 18/19/20/21/23/24, which already had rows for 14/22/25), and the full fine-grained per-plan log remains verbatim in each phase's archived `PATTERNS.md`/`SUMMARY.md` under `.planning/milestones/v2.1-phases/`.

- [Phase quick-260722-tkt]: Field Targets created by --create-missing-targets are always type=SIDEREAL (fixed-sky pointings), distinct from the campaign's non-sidereal moving-object target by design
- [Phase quick-260722-tkt]: created_targets counter only reflects actually-persisted creations (0 in --dry-run); per-request stdout line still reports would-create/would-reuse intent
- [Phase quick-260722-twe]: epoch/pm_ra/pm_dec/parallax (from LCO wire keys epoch/proper_motion_ra/proper_motion_dec/parallax) are only set on newly-built field Targets, never on a reused existing Target — reuse never overwrites existing Target metadata
- [Phase quick-260722-uhh]: Target's admin URL/reverse name is `tom_targets_basetarget_changelist`, not `tom_targets_target_changelist` — `Target = get_target_model_class()` resolves to `BaseTarget` (no `TARGET_MODEL_CLASS` override in settings.py); tests derive the app_label/model_name dynamically rather than hardcoding either string
- [Phase quick-260722-ux0]: `facility.update_observation_status()` uses its own module-level `make_request` binding (`tom_observations.facilities.ocs.make_request`), separate from the one a caller module imports and patches — a test suite calling code that transitively invokes `update_observation_status()` must patch `LCOFacility.update_observation_status` itself (or the `ocs` module's `make_request`), not the caller's imported `make_request` name, or it will make a real live HTTP call
- [Phase quick-260722-uyz]: sync_lco_observation_calendar never populated CalendarEvent.target_list since its original Phase 04 implementation (confirmed via git log -p) — fixed by deriving it from record.target.targetlist_set.order_by('name').first() (deterministic alphabetically-first pick when a Target is in 2+ campaigns, None if in none); applies uniformly to both LCO and SOAR records since they share _build_event_fields()
- [Phase ?]: Quick 260723-02e: load_telescope_runs --campaign duplicates only the explicit-name TargetList lookup branch of backfill's _resolve_campaign (no interactive prompt); 'target_list' key always present in the fields dict for consistent no-churn FK diffing

### Pending Todos

- `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` — rename
  `calendar_utils.py`'s cross-module-consumed underscore-prefixed helpers
  (`_derive_telescope`, `_extract_instrument`, `_resolve_placement_block`,
  `_coarse_telescope_label`, `_aperture_class_from_telescope_code`) to reflect that the
  module is now a real shared API (3 consumers); low-priority style cleanup found while
  verifying the 2026-06-23 extraction todo was complete.

- Carried-forward items in Deferred Items below.

### Blockers/Concerns

None. v2.1 shipped 2026-07-18; awaiting `/gsd-new-milestone` to start the next cycle.

### Quick Tasks Completed

| # | Description | Date | Commit | Status | Directory |
|---|-------------|------|--------|--------|-----------|
| 260705-l1v | Fix approval-queue site-visibility gap: show site_raw in the pending CampaignRun approval queue and stop the approval endpoint from fabricating placeholder Observatory rows for unresolvable free-text site names (found during v2.0 manual UAT) | 2026-07-05 | 959a78d | Verified | [260705-l1v-fix-approval-queue-site-visibility-gap-s](./quick/260705-l1v-fix-approval-queue-site-visibility-gap-s/) |
| 260711-o71 | Measure solsys_code test coverage, add permanent CR-01/CR-02 regression tests to test_campaign_approval.py (closing the gap left by the phase 21 verifier's temporary tests), re-measure and report the diff | 2026-07-11 | adcd59a | Complete | [260711-o71-measure-current-test-coverage-for-solsys](./quick/260711-o71-measure-current-test-coverage-for-solsys/) |
| 260714-ilz | Close date-format gap on public campaign-run submission form: obs_date now accepts single date/range/blank via parse_obs_window(), closing the hard Django date-validation failure that blocked multi-night range submissions (SUBMIT-01) | 2026-07-14 | f7b3ca0 | Complete | [260714-ilz-close-date-format-gap-on-public-campaign](./quick/260714-ilz-close-date-format-gap-on-public-campaign/) |
| 260719-d18 | Fix calendar data-url missing utc_offset query param causing timezone selection to reset on calRefresh | 2026-07-19 | ec9afc6 | Complete | [260719-d18-fix-calendar-data-url-missing-utc-offset](./quick/260719-d18-fix-calendar-data-url-missing-utc-offset/) |
| 260714-jpd | Register CampaignRun and CalendarEventTelescopeLabel in solsys_code/admin.py: approval_status read-only (no admin bypass of CampaignRunDecisionView.post()'s calendar projection + D-06 guard), contact PII excluded from the change-list but editable in detail, proven via a new admin test-client suite | 2026-07-14 | b6ae100 | Complete | [260714-jpd-add-calendareventtelescopelabel-and-camp](./quick/260714-jpd-add-calendareventtelescopelabel-and-camp/) |
| 260716-h8c | Backfill Observatory.timezone from lat/lon in MPCObscodeFetcher.to_observatory() (Tier-2 MPC site-code lookup) using timezonefinder, closing the CR-01 gap where Tier-2-resolved sites always got a blank timezone and needed a manual admin edit before a Sites Needing Review calendar-projection retry could succeed | 2026-07-16 | 75962de | Complete | [260716-h8c-backfill-observatory-timezone-from-lat-l](./quick/260716-h8c-backfill-observatory-timezone-from-lat-l/) |
| 260716-js7 | Add a client-side confirm-before-approve guard on the approval queue's Pending Review row: nudge staff before they Approve an unresolved Observing Site, mirroring the existing Reject confirmation pattern (D-06 preserved, no server-side change) | 2026-07-16 | 24d1d94 | Complete | [260716-js7-give-staff-clear-inline-feedback-guardra](./quick/260716-js7-give-staff-clear-inline-feedback-guardra/) |
| 260717-iae | Wire the five existing pre-executed demo notebooks into docs/notebooks.rst's Sphinx toctree so they appear in the published Notebooks section (previously orphaned, no toctree reference) | 2026-07-17 | 6b3c145 | Complete | [260717-iae-wire-the-existing-pre-executed-demo-note](./quick/260717-iae-wire-the-existing-pre-executed-demo-note/) |
| 260718-dih | Fix PR review findings from .planning/Findings.md: guard the unreverted calendar-sync loop in CampaignRunDecisionView._set_run_status with a non-reverting try/except, make parse_run_line fail fast on cross-month run ranges instead of the loader rejecting them later, anchor the partial-night token match with fullmatch, add regression tests for all three, and correct Findings.md's line-number citations | 2026-07-18 | 01dbc2a | Complete | [260718-dih-fix-pr-review-findings-unguarded-calenda](./quick/260718-dih-fix-pr-review-findings-unguarded-calenda/) |
| 260722-hpw | Fix import_campaign_csv to skip leading comment/blank rows before the real CSV header, so it can consume the real 3I/ATLAS sheet export unchanged | 2026-07-22 | 990bfb9 | Complete | [260722-hpw-fix-import-campaign-csv-to-skip-leading-](./quick/260722-hpw-fix-import-campaign-csv-to-skip-leading-/) |
| 260722-tkt | Add opt-in --create-missing-targets flag to backfill_lco_observation_records: auto-create-or-reuse a SIDEREAL field Target from the request's RA/Dec, add it to the campaign, and process the request normally instead of skipping it | 2026-07-22 | 73581b0 | Complete | [260722-tkt-add-create-missing-targets-flag-to-backf](./quick/260722-tkt-add-create-missing-targets-flag-to-backf/) |
| 260722-twe | Extend backfill_lco_observation_records --create-missing-targets to also pull epoch/pm_ra/pm_dec/parallax from the LCO request target dict when present, mapped onto newly-built field Targets only (reused Targets untouched) | 2026-07-22 | ba59d0f | Complete | [260722-twe-extend-backfill-lco-observation-records-](./quick/260722-twe-extend-backfill-lco-observation-records-/) |
| 260722-uhh | Register a custom Django admin for tom_targets' Target model in solsys_code/admin.py with list_filter on type (sidereal vs non-sidereal), so staff can filter Targets by type in the admin change-list | 2026-07-22 | fac8a61 | Complete | [260722-uhh-register-a-custom-django-admin-for-tom-t](./quick/260722-uhh-register-a-custom-django-admin-for-tom-t/) |
| 260722-ux0 | Fix backfill_lco_observation_records: refresh scheduled_start/scheduled_end via facility.update_observation_status() immediately after creating a new ObservationRecord, closing the perpetual [QUEUED] calendar-title bug for backfilled terminal records | 2026-07-23 | 6c5b205 | Complete | [260722-ux0-fix-backfill-lco-observation-records-pop](./quick/260722-ux0-fix-backfill-lco-observation-records-pop/) |
| 260722-uyz | Fix sync_lco_observation_calendar: populate CalendarEvent.target_list from the record's Target's campaign TargetList membership (deterministic first match by name), closing a gap present since the command's original Phase 04 implementation | 2026-07-23 | ac5f0ac | Complete | [260722-uyz-fix-sync-lco-observation-calendar-set-ca](./quick/260722-uyz-fix-sync-lco-observation-calendar-set-ca/) |
| 260723-02e | Add optional --campaign flag to load_telescope_runs: resolve a tom_targets.TargetList once upfront (explicit-name-only, fail-fast) and associate it with every created/updated CalendarEvent.target_list, matching backfill_lco_observation_records and sync_lco_observation_calendar precedent | 2026-07-23 | b3c4cd8 | Complete | [260723-02e-add-an-optional-campaign-flag-to-load-te](./quick/260723-02e-add-an-optional-campaign-flag-to-load-te/) |

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

Last session: 2026-07-23T07:25:25.155Z
Stopped at: Completed quick task 260723-02e: Add --campaign flag to load_telescope_runs, associating CalendarEvent.target_list with a TargetList
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
