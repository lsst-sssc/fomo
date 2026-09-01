---
gsd_state_version: 1.0
milestone: v2.2
milestone_name: One Canonical Run Record
status: Awaiting next milestone
stopped_at: Phase 30 complete — all phases complete
last_updated: "2026-09-01T10:20:57.258Z"
last_activity: 2026-09-01
last_activity_desc: Milestone v2.2 completed and archived
state_head: 6fa1844657448a12c4e5f48d2a128837ac438a60
progress:
  total_phases: 6
  completed_phases: 6
  total_plans: 33
  completed_plans: 33
  percent: 100
current_phase: 30
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-01 — v2.2 milestone archived, full evolution review complete)

**Core value:** An observing run exists once, as a `CampaignRun`, and everything else is derived from it — the calendar events that show it, the observation records that realise it, and the coverage-gap analysis that counts it.
**Current focus:** Planning next milestone — awaiting `/gsd-new-milestone`

## Current Position

Phase: Milestone v2.2 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-09-01 — Milestone v2.2 completed and archived

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
| 30 | 4 | - | - |
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
| Phase quick-260724-tiz P01 | ~12min | 2 tasks | 4 files |
| Phase quick-260724-vb0 P01 | ~45min | 3 tasks | 4 files |
| Phase 26 P01 | 50min | 3 tasks | 8 files |
| Phase 26 P02 | ~35min | 3 tasks | 1 files |
| Phase 26-canonical-record-spike P03 | 90min | 3 tasks | 4 files |
| Phase 26-canonical-record-spike P04 | 90min | 3 tasks | 1 files |
| Phase 26 P05 | 55min | 3 tasks | 5 files |
| Phase 27 P01 | 25min | 3 tasks | 5 files |
| Phase Phase 27 P02 P02 | 25min | 3 tasks | 4 files |
| Phase 27 P03 | 20min | 3 tasks | 10 files |
| Phase 27 P04 | 45min | 3 tasks | 6 files |
| Phase 27 P05 | 35min | 3 tasks | 7 files |
| Phase 27 P06 | 70min | 3 tasks | 7 files |
| Phase 29 P06 | 75min | 3 tasks | 4 files |
| Phase 30 P01 | 8min | 3 tasks | 4 files |
| Phase 30 P02 | 10min | 3 tasks | 4 files |
| Phase 30 P03 | 10min | 3 tasks | 3 files |
| Phase 30 P04 | 15min | 3 tasks | 6 files |

## Accumulated Context

### Roadmap Evolution

- Phase 27.1 inserted after Phase 27 (2026-07-30) (URGENT): Close gap: staff surfaces and data-integrity risks from the canonical run record. Sources: `27-UAT.md` (5 passed / 2 issues → 3 gaps — no nav path to the Sites Needing Review queue, the event modal rendering its own multi-line `{# #}` header as literal text, and an illegible admin run picker) plus `27-VERIFICATION.md` review warnings WR-01 (CSV re-import can silently revert a `repair_stale_campaign_run_sites` fix), WR-03 (`source` freely editable in admin) and WR-04 (a TBD run renders "(None–None)" in the public modal). Scheduled before Phase 28 because the admin FK picker is the only mechanism that creates run↔event links until the attribution queue ships. **WR-02 deliberately excluded** as a stale finding: the verification report calls "never clears `telescope_class` on site resolution" an invariant violation, but `models.py:213-219` documents the opposite invariant and the user already rejected code-review finding CR-01 which proposed clearing it.
- Phase 22 added (2026-07-14): Site Matching at Submission and Unmatched-Site Resolution Workflow — closes the Phase 21 functionality gap. Decisions confirmed with operator: (a) the public submission form's Observing site field gets HTMX live-search autocomplete (new endpoint running `fuzzy_match_candidates()` over `build_site_candidates()`), also replacing the approval queue's static per-row datalist; (b) "site failure never blocks approval" is kept, with a new "Sites needing review" surface for approved runs with `site_needs_review=True` whose resolution triggers the deferred CalendarEvent projection.
- Phase 24 added (2026-07-17): Operator and usage runbook documentation for the telescope-runs-calendar management commands and staff workflows (load_telescope_runs, sync_lco_observation_calendar, sync_gemini_observation_calendar, import_campaign_csv, Phase 23's approval-queue status-change actions) — raised during PR #41/#43 split review: design docs (docs/design/*.rst) and demo notebooks existed, but no general, discoverable how-to-run documentation did. Scoped to publish operator-facing usage docs beyond design rationale and `--help` text.
- Phase 25 added (2026-07-17): Range-window CalendarEvent projection — closes the diagnosed gap where approved, site-resolved range-window CampaignRuns (e.g. the real GS-2026A-FT-115 Gemini FT allocation) never get a CalendarEvent, verified via `/gsd-debug` (`.planning/debug/range-window-calendar-event.md`, diagnose-only, root cause + before/after spec, no code changed). Root cause: Phase 19 D-06's guard was a behavior-preservation deferral, not a considered decision; Phase 23's `TestGeminiFtScenario` re-encoded the deferred behavior as contract. Fix scope per the spec: drop the guard's `window_start == window_end` clause (add a `window_end` truthiness check instead), give the ground branch multi-day date-math (satellite branch is already correct), and deliberately revise the 4 Phase 19/23 test assertions that currently assert zero events for range runs.
- Phase 30 added: v2.2 Tech-Debt Cleanup: repo-wide ruff pass, runbook prose fixes, attribution candidate approval_status filter
- Phase 30 edited: edited fields: goal, depends_on, success_criteria (added), scope list (added), locked-context pointer (added), paired-docs. Rewritten from 30-CONTEXT.md per D-11 — the original goal named the WR-09/WR-10 runbook fixes and a repo-wide ruff pass, all three of which discuss-phase verified as already closed, and specified excluding PENDING_REVIEW from attribution candidates, which D-01 reversed.

### Decisions

All v1.0-v2.2 decisions logged in PROJECT.md's Key Decisions table. The exhaustive per-plan v2.2 decision log previously kept here (roadmap-structure decisions, and one bullet per Phase 26-30 plan plus the 2026-07/08 quick tasks) has been cleared now that v2.2 has shipped and closed — nothing is lost: the milestone-level decisions are summarized in PROJECT.md's Key Decisions table (9 rows backfilled at close for Phases 26/27/27.1/28/29, plus the 3 rows Phase 30 added at its own completion), and the full fine-grained per-plan log remains verbatim in each phase's archived `PATTERNS.md`/`SUMMARY.md` under `.planning/milestones/v2.2-phases/` and `.planning/milestones/v2.2-quick/`.

### Pending Todos

- `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` — rename
  `calendar_utils.py`'s cross-module-consumed underscore-prefixed helpers
  (`_derive_telescope`, `_extract_instrument`, `_resolve_placement_block`,
  `_coarse_telescope_label`, `_aperture_class_from_telescope_code`) to reflect that the
  module is now a real shared API (3 consumers); low-priority style cleanup found while
  verifying the 2026-06-23 extraction todo was complete.

- Carried-forward items in Deferred Items below.

### Blockers/Concerns

None blocking. v2.2 "One Canonical Run Record" shipped and closed 2026-09-01 (6 phases, 33 plans, 24/24 requirements). One non-blocking follow-up carried into the next milestone: `import_campaign_csv.py`'s `site_needs_review` is computed from the pre-preservation `telescope_class` value rather than the post-guard value (30-REVIEW.md WR-01) — recommend a future quick task.

### Quick Tasks Completed

| # | Description | Date | Commit | Status | Directory |
|---|-------------|------|--------|--------|-----------|

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
| quick_task | `260613-eb1-add-a-demo-jupyter-notebook-for-phase-1-` | Completed (has SUMMARY.md); un-archived, no explicit status field | v2.2 close |
| quick_task | `260619-jpr-fix-sync-lco-observation-calendar-soar-s` | Completed (has SUMMARY.md); un-archived, no explicit status field | v2.2 close |
| quick_task | `260722-hpw-fix-import-campaign-csv-to-skip-leading-` | Completed (has SUMMARY.md); un-archived, no explicit status field | v2.2 close |
| quick_task | `260725-kn4-guard-mpcobscodefetcher-and-to-earth-loc` | Completed (has SUMMARY.md); un-archived, no explicit status field | v2.2 close |
| quick_task | `260726-fqb-map-jpl-horizons-naif-observer-notation-` | Completed (has SUMMARY.md); un-archived, no explicit status field | v2.2 close |
| quick_task | `260726-kdp-close-operator-runbook-drift-and-broaden` | Completed (has SUMMARY.md); un-archived, no explicit status field | v2.2 close |
| quick_task | `260805-qdc-fix-t-29-19-phase-29-security-audit-rout` | Completed (has SUMMARY.md); un-archived, no explicit status field | v2.2 close |
| quick_task | `260805-sgf-split-campaignrun-telescope-instrument-i` | Completed (has SUMMARY.md); un-archived, no explicit status field | v2.2 close |
| quick_task | `260805-tad-fix-window-shape-dispatch-in-the-calenda` | Completed (has SUMMARY.md); un-archived, no explicit status field | v2.2 close |
| quick_task | `260806-lgo-mark-recon-04-as-complete-in-planning-re` | Completed (has SUMMARY.md); un-archived, no explicit status field | v2.2 close |
| quick_task | `260806-ol7-build-a-new-demo-notebook-or-extend-an-e` | Completed (has SUMMARY.md); un-archived, no explicit status field | v2.2 close |
| todo | `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — still pending, no second consumer yet | Deliberately deferred (carried again) | v2.2 close |
| seed | SEED-001 — file upstream `tom_eso` feature requests | Still dormant | v2.2 close |
| seed | SEED-002 — ESO ObservationRecord-centric future intent | Still dormant | v2.2 close |
| deferred_items | Phase 26 `deferred-items.md`: pre-existing repo-wide ruff/format drift | Resolved — Phase 30 (30-02) root-caused as unpinned dev-ruff vs. pinned pre-commit ruff; pinned dev extra, no reformat needed | v2.2 close |
| deferred_items | Phase 27 `deferred-items.md`: pre-existing ruff format drift | Resolved — same root cause fixed by Phase 30 (30-02) | v2.2 close |
| deferred_items | Phase 27.1 `deferred-items.md`: pre-existing repo-wide ruff/format drift | Resolved — same root cause fixed by Phase 30 (30-02) | v2.2 close |
| deferred_items | Phase 14 (archived v2.0) `deferred-items.md`: pre-existing ruff check/format findings | Resolved — same root cause fixed by Phase 30 (30-02) | v2.2 close |
| deferred_items | Phase 15 (archived v2.0) `deferred-items.md`: pre-existing repo-wide ruff check/format failures | Resolved — same root cause fixed by Phase 30 (30-02) | v2.2 close |
| deferred_items | Phase 04 (archived v1.2) `deferred-items.md`: ruff format --check pre-existing findings | Resolved — same root cause fixed by Phase 30 (30-02) | v2.2 close |

## Session Continuity

Last session: 2026-09-01T04:20:04.776Z
Stopped at: Phase 30 complete — all phases complete
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
