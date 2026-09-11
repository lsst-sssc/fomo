---
gsd_state_version: 1.0
milestone: v2.4
milestone_name: Observation-First Calendar
current_phase: 34
current_phase_name: The Observation Projector & Trigger
status: verifying
stopped_at: Completed 34-04-PLAN.md
last_updated: "2026-09-11T04:57:59.810Z"
last_activity: 2026-09-10
last_activity_desc: Phase 34 execution started
state_head: f0f09d4506a14495cf6b1a22461ffc85aff8e161
progress:
  total_phases: 5
  completed_phases: 1
  total_plans: 15
  completed_plans: 15
  percent: 20
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-10 — after Phase 33 complete)

**Core value:** The calendar is driven by what actually happened — one event per `ObservationRecord`, narrowing on every save with no operator action; allocations project intent nights until a real observation retires them; campaigns annotate, never own.
**Current focus:** Phase 34 — The Observation Projector & Trigger

## Current Position

Phase: 34 (The Observation Projector & Trigger) — EXECUTING
Plan: 4 of 4
Status: Phase complete — ready for verification
Last activity: 2026-09-10 — Phase 34 execution started

## Roadmap Summary (v2.4 — in progress, started 2026-09-03)

| Phase | Goal | Requirements |
|-------|------|--------------|
| 33. Series Identity & Reconciler Inversion | Give `CalendarEventMeta` the real link fields the base layer needs, and turn the campaign reconciler from an owner into an annotator so the two layers can run side by side | PROJ-04, ANNOT-01, ANNOT-02 |
| 34. The Observation Projector & Trigger | Every LCO/SOAR observation record draws and keeps current its own calendar event on every save, with a sweep as backstop; the old LCO sync command is retired in its favour | PROJ-01..03, PROJ-05, PROJ-06, TRIG-01..03, SCHED-06, ANNOT-03 |
| 35. Allocation Layer & Classical Cutover | An allocation projects its own sunset→sunrise intent nights and hands each night over when a real observation links to it; `load_telescope_runs` writes allocations instead of events | ALLOC-01..05 |
| 36. Unattended Operation | The sweep, the discovery backfill and the reconciler run on the real host on a cron + `flock` schedule against an admin-editable watched-proposal list, with failures visible and no credential logged | SCHED-08..10, DISCOVER-01 |
| 37. Status Vocabulary, Public Tallies & Provenance-Blind Gaps | One status vocabulary, an ongoing public tally per run and campaign, unused awarded nights that look unused, and coverage gaps that count every observation | STATUS-01..02, TALLY-01..03, UNUSED-01, GAPB-01 |

Coverage: 29/29 v1 requirements mapped, no orphans, no duplicates. Phase numbering continues from the superseded v2.3 (last phase: 32). Full phase detail, locked constraints and paired-docs scope in `.planning/ROADMAP.md`.

**Superseded predecessor:** v2.3's roadmap (Phases 31-35, ADAPT-*/OUTCOME-*) is archived at `.planning/milestones/v2.3-ROADMAP.md`. Phase 31's four verdicts and plan 32-01 Tasks 1–2 are kept as v2.4's allocation-without-campaign foundation; ADAPT-01..06 and OUTCOME-01..04 are dropped, not deferred.

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
| 31 | 6 | - | - |
| 33 | 11 | - | - |
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
| Phase 31 P01 | 25min | 3 tasks | 6 files |
| Phase 31 P02 | 20min | 2 tasks | 2 files |
| Phase 31 P03 | 25min | 3 tasks | 2 files |
| Phase 31 P04 | 20min | 3 tasks | 1 files |
| Phase 31 P05 | ~25min | 2 tasks | 2 files |
| Phase 33 P01 | 73min | 3 tasks | 7 files |
| Phase 33-series-identity-reconciler-inversion P03 | 40min | 3 tasks | 6 files |
| Phase 33 P02 | ~39min | 3 tasks | 6 files |
| Phase 33 P04 | 28min | 3 tasks | 10 files |
| Phase 33-series-identity-reconciler-inversion P05 | 44min | 3 tasks | 3 files |
| Phase 33 P06 | 32min | 3 tasks | 7 files |
| Phase 33 P07 | 26min | 3 tasks | 4 files |
| Phase 33 P08 | 40min | 4 tasks | 7 files |
| Phase 33 P09 | 38min | 3 tasks | 3 files |
| Phase 33-series-identity-reconciler-inversion P11 | ~30 min | 3 tasks | 5 files |
| Phase 33-series-identity-reconciler-inversion P10 | 70min | 3 tasks | 9 files |
| Phase 34 P01 | 54min | 3 tasks | 7 files |
| Phase 34 P02 | 52min | 3 tasks | 12 files |
| Phase 34 P03 | 38min | 2 tasks | 6 files |
| Phase 34 P04 | 40min | 2 tasks | 11 files |

## Accumulated Context

### Roadmap Evolution

- v2.4 roadmap created (2026-09-03): 5 phases (33-37), 29/29 requirements mapped. Phase numbering continues from the superseded v2.3 (last phase: 32); v2.3's ADAPT-*/OUTCOME-* work is dropped, not re-planned. Structure is driven by the spike-established ordering, not by category grouping: the reconciler inversion (ANNOT-01/02) plus the `CalendarEventMeta` link fields (PROJ-04) form Phase 33 and land **before** the base layer and the campaign layer ever run side by side, because the reconciler's adopt/re-key/detach paths read `CalendarEventMeta.run` as ownership and would otherwise steal base-layer events (spike 002's named landmine). The projector, its `post_save` trigger and its sweep share Phase 34, and ANNOT-03 (retiring `sync_lco_observation_calendar`) rides with them because the takeover is a plain update in the same key namespace and can only be proven equivalent once the projector and sweep exist; SCHED-06 sits there too as the verification-over-time requirement that closes spike 004's PARTIAL verdict against the real `KEY2026B-004` nights. The allocation layer and the `load_telescope_runs` cutover (ALLOC-01..05) follow in Phase 35 because the handoff rule needs observation events to hand over to, and ALLOC-04/05 additionally depend on Phase 31's SCHEMA-03 finding that a classical `source_identifier` needs a facility-specific key. Unattended operation (Phase 36) comes after the sweep command exists, since that is what cron invokes. The public tallies, the status vocabulary and provenance-blind gap analysis close the milestone in Phase 37 because they all read the projected events and the `CalendarEventMeta` links the earlier phases create — and Phase 34's provisional title prefixes are deliberately left for Phase 37 to settle. No investigation-only phase was added: the five spikes already did that work.

- v2.3 roadmap created (2026-09-01): 5 phases (31-35), 22/22 requirements mapped. Phase numbering continues from v2.2's last phase (30). Structure derived from `research/SUMMARY.md`'s recommended 8-phase sequence, compressed to 5 under the `coarse` granularity setting: the scheduling-mechanism spike (SCHED-07) is folded into the schema spike phase as an independent parallel track rather than standing alone (both are investigation-only, and running them together preserves the spike-before-implementation discipline for each); the shared `write_and_reconcile_campaign_run()` helper is groundwork inside Phase 32 rather than a phase with no requirements of its own; and the three adapters share one phase, shipping simplest-first (classical → LCO → Gemini) as ordered plans so each still validates the shared pattern before the next facility's identity scheme is attempted. Hard dependency order from research is respected: schema spike gates the adapters; adapters gate outcome propagation (nothing to read without confirmed `CampaignRunObservation` links); the scheduler entry point comes after everything it orchestrates; carried-forward work (STATUS/GAPB/UNUSED) comes last, once a `CampaignRun` exists for every ingest path. SCHED-06 sits with outcome propagation because the narrowing UI is that feature's visible face (pipeline stages 3→4).
- Phase 27.1 inserted after Phase 27 (2026-07-30) (URGENT): Close gap: staff surfaces and data-integrity risks from the canonical run record. Sources: `27-UAT.md` (5 passed / 2 issues → 3 gaps — no nav path to the Sites Needing Review queue, the event modal rendering its own multi-line `{# #}` header as literal text, and an illegible admin run picker) plus `27-VERIFICATION.md` review warnings WR-01 (CSV re-import can silently revert a `repair_stale_campaign_run_sites` fix), WR-03 (`source` freely editable in admin) and WR-04 (a TBD run renders "(None–None)" in the public modal). Scheduled before Phase 28 because the admin FK picker is the only mechanism that creates run↔event links until the attribution queue ships. **WR-02 deliberately excluded** as a stale finding: the verification report calls "never clears `telescope_class` on site resolution" an invariant violation, but `models.py:213-219` documents the opposite invariant and the user already rejected code-review finding CR-01 which proposed clearing it.
- Phase 22 added (2026-07-14): Site Matching at Submission and Unmatched-Site Resolution Workflow — closes the Phase 21 functionality gap. Decisions confirmed with operator: (a) the public submission form's Observing site field gets HTMX live-search autocomplete (new endpoint running `fuzzy_match_candidates()` over `build_site_candidates()`), also replacing the approval queue's static per-row datalist; (b) "site failure never blocks approval" is kept, with a new "Sites needing review" surface for approved runs with `site_needs_review=True` whose resolution triggers the deferred CalendarEvent projection.
- Phase 24 added (2026-07-17): Operator and usage runbook documentation for the telescope-runs-calendar management commands and staff workflows (load_telescope_runs, sync_lco_observation_calendar, sync_gemini_observation_calendar, import_campaign_csv, Phase 23's approval-queue status-change actions) — raised during PR #41/#43 split review: design docs (docs/design/*.rst) and demo notebooks existed, but no general, discoverable how-to-run documentation did. Scoped to publish operator-facing usage docs beyond design rationale and `--help` text.
- Phase 25 added (2026-07-17): Range-window CalendarEvent projection — closes the diagnosed gap where approved, site-resolved range-window CampaignRuns (e.g. the real GS-2026A-FT-115 Gemini FT allocation) never get a CalendarEvent, verified via `/gsd-debug` (`.planning/debug/range-window-calendar-event.md`, diagnose-only, root cause + before/after spec, no code changed). Root cause: Phase 19 D-06's guard was a behavior-preservation deferral, not a considered decision; Phase 23's `TestGeminiFtScenario` re-encoded the deferred behavior as contract. Fix scope per the spec: drop the guard's `window_start == window_end` clause (add a `window_end` truthiness check instead), give the ground branch multi-day date-math (satellite branch is already correct), and deliberately revise the 4 Phase 19/23 test assertions that currently assert zero events for range runs.
- Phase 30 added: v2.2 Tech-Debt Cleanup: repo-wide ruff pass, runbook prose fixes, attribution candidate approval_status filter
- Phase 30 edited: edited fields: goal, depends_on, success_criteria (added), scope list (added), locked-context pointer (added), paired-docs. Rewritten from 30-CONTEXT.md per D-11 — the original goal named the WR-09/WR-10 runbook fixes and a repo-wide ruff pass, all three of which discuss-phase verified as already closed, and specified excluding PENDING_REVIEW from attribution candidates, which D-01 reversed.
- Phase 32 edited (2026-09-03, discuss-phase): resolved the pending retarget todo (`2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md`, gap G-31-3) at discussion time rather than deferring it further. ADAPT-03 retargeted from Gemini to SOAR (folded into `sync_lco_observation_calendar` under a new `SOAR_QUEUE` source value) as the facility proving the pattern generalises to real read-back; Gemini's own write path kept in scope as new requirement ADAPT-06, explicitly caveated as unable to support Phase 33 outcome propagation. Coverage moves from 22/22 to 23/23. ROADMAP.md's Phase 32 goal/success-criteria/locked constraints and REQUIREMENTS.md's ADAPT-03/ADAPT-06 text updated to match; full rationale in `32-CONTEXT.md` and `32-DISCUSSION-LOG.md`.

### Decisions

All v1.0-v2.2 decisions logged in PROJECT.md's Key Decisions table. The exhaustive per-plan v2.2 decision log previously kept here (roadmap-structure decisions, and one bullet per Phase 26-30 plan plus the 2026-07/08 quick tasks) has been cleared now that v2.2 has shipped and closed — nothing is lost: the milestone-level decisions are summarized in PROJECT.md's Key Decisions table (9 rows backfilled at close for Phases 26/27/27.1/28/29, plus the 3 rows Phase 30 added at its own completion), and the full fine-grained per-plan log remains verbatim in each phase's archived `PATTERNS.md`/`SUMMARY.md` under `.planning/milestones/v2.2-phases/` and `.planning/milestones/v2.2-quick/`.

v2.3 roadmap-structure decisions (2026-09-01):

- **Both spikes share Phase 31.** The schema/identity spike (SCHEMA-01..03) and the scheduling-mechanism spike (SCHED-07) are independent of each other, so they run as parallel tracks in one investigation-only phase. This keeps each one's findings ahead of the phase that implements them (Phase 32 and Phase 34 respectively), which folding SCHED-07 into Phase 34 as a first plan would not.
- **The shared write-and-reconcile helper is not its own phase.** It carries no requirement of its own and exists only to stop the same pattern being written three times; it is Phase 32's groundwork, with the classical adapter as its first consumer.
- **The three adapters share one phase, ordered plans.** Research's simplest-first sequencing (classical → LCO → Gemini) is preserved as plan ordering inside Phase 32 rather than as three phases, per the `coarse` granularity setting.
- **ADAPT-04 (per-adapter no-churn) and ADAPT-05 (cutover sequencing) are mapped to Phase 32 only**, not repeated per adapter — they are cross-cutting guarantees of the same phase, and the coverage rule is one requirement to exactly one phase.
- **SCHED-06 sits with outcome propagation (Phase 33), not with the other carried-forward work**, because the narrowing UI is what outcome propagation looks like on screen (four-stage pipeline stages 3→4) and depends on OUTCOME-01..04 existing.
- [Phase 31]: SCHEMA-01/02 evidence gathered: 0/49 CampaignRun rows have a null campaign FK today; 4 pre-existing telinst/window tuples already collide across different real campaigns, falsifying Option B's zero-risk premise; Option A loses all duplicate protection for non-campaign rows. — Recommendation between the three D-05 candidate shapes deferred to plan 31-02's checkpoint; this plan only gathers evidence.
- [Phase 31]: SCHEMA-01/02: Task 1 checkpoint chose nullable-fk (Option A - make CampaignRun.campaign nullable) over single-sentinel and per-proposal-placeholder; SCHEMA-02 locks a source_identifier CharField with a partial UniqueConstraint as the write-time identity surface, disjoint from and additive alongside both existing partial constraints.
- [Phase 31]: SCHEMA-03: classical adapter's 5-minute telescope/instrument/start_time tolerance match is NOT sufficient as a write-time identity surface on its own (two proposals sharing telescope/instrument/night collide); source_identifier's synthesized key inherits the same gap; a proposal code is not currently a reliable fallback (present in only 1/3 real sample lines, unparseable where seen)
- [Phase 31]: SCHED-07: cron+flock inside the FOMO container confirmed against real host facts - flock present, 0/3 existing FOMO cron entries guarded, heartbeat egress confirmed (HTTP 301); no container build definition exists in the repo, container/AWS scopes remain unconfirmed
- [Phase 31]: Phase 31 published both spike verdicts (schema/identity SCHEMA-01..03, scheduling SCHED-07) to docs/design/run_identity_and_unattended_invocation_spike.rst, closing roadmap Success Criterion 5's first half; Sphinx build and targeted 6-module regression both green, source tree/test suite proven unchanged.
- [Phase 31, gap-closure plan 31-06]: UAT gap G-31-3 closed — corrected both committed artifacts' facility framing: Gemini has no facility read-back (`GEMFacility`'s status/URL methods are hardcoded stubs; `sync_gemini_observation_calendar` never imports that class and only replays FOMO's own prior submissions), SOAR is the facility that actually has one (inherits a real portal read path from `LCOFacility`, already handled inside the existing LCO sync command). All four verdicts unchanged. Three Phase 32/33 consequences (missing `SOAR_QUEUE` vocabulary value, ADAPT-03 re-target, Phase 33 Gemini-infeasibility caveat) recorded as a pending todo, not actioned here.
- [Phase 33]: Phase 33 Plan 01: retired _adopted_event_for_night() outright rather than leaving it as dead code; _attributed_nights() carries no blank-url restriction so a facility-URL-keyed attributed event is skipped too; campaign_decoration() builds its campaign-table link with reverse() in Python so a null campaign_id returns table_url=None instead of NoReverseMatch; event_title() dropped its campaign-name branch entirely -- the decoration tag is now the single campaign label everywhere.
- [Phase 33]: 33-02: marker glyph is a single flag character styled by .cal-campaign-chip (currentColor + flex-shrink:0), never a new color constant — keeps the chip from competing with the entry's own accessible foreground or the proposal fill
- [Phase 33]: 33-02: CampaignRunTable row_attrs resolves pk via Accessor(...).resolve(record, quiet=True), returning None (never 'run-None') so django-tables2 drops the id attribute for an unresolvable pk — works identically for staff model-instance rows and non-staff .values() dict rows
- [Phase 33]: 33-04: unlink_event_from_run() in campaign_utils.py is now the single writer that clears a CalendarEventMeta attribution -- run, confirmed_by and confirmed_at together -- with a null-run guard that returns 0 before any queryset is built (T-33-21); all three existing clear-the-link call sites (undo view, reconciler detach, admin clear branch) route through it. The reconciler's detach step now also clears the audit stamps (D-16, closing a stale-confirmation leak) via a local import that breaks the circular dependency with campaign_utils' own top-level import of campaign_reconciler.
- [Phase 33-series-identity-reconciler-inversion]: Phase 33 Plan 05: D-04's real-database diff proof split across two notebook cells to satisfy the plan's cell-ordering verify script; skip-rule demo deletes this run's own already-created event rather than mutating classical_run's window (which is part of its own natural-key lookup).
- [Phase 33]: 33-06: CR-01 fixed by moving the tr:target style into tom_common/base.html's empty additional_css block -- a top-level node in a template that extends is silently discarded by Django's ExtendsNode.
- [Phase 33]: 33-06: WR-08's positional-page-resolution fix deferred -- would add a per-event ordered query, contradicting plan 33-02's no-per-event-query must-have; the gap is pinned by a test instead.
- [Phase 33]: Phase 33: 33-07: UNLINK_CLEARED_FIELDS is the single declaration of what clearing a campaign attribution means, consumed by unlink_event_from_run()'s bulk .update() and CalendarEventMetaAdmin.save_model()'s in-memory clear via a function-local import (WR-02).
- [Phase 33]: Phase 33: 33-07: unlink_event_from_run() rejects str/bytes events arguments with TypeError, closing the per-character event__in expansion hole (WR-04).
- [Phase 33]: Phase 33: 33-07: CalendarEventMetaInline's docstring corrected to describe Django's actual rendering -- fk_name='run' produces a hidden InlineForeignKeyField for parent linkage, never an editable widget (WR-06 code-side half).
- [Phase 33]: Adopted the local-noon anchored _observing_night(), superseding 26-DECISION.md D-10's plain site-local .date() derivation (checkpoint resolved by user: noon-anchor). — Matches the anchor sun_event() itself uses; closes CR-02's duplicate-night/uncovered-night defect.
- [Phase 33]: Made the D-01/ANNOT-01 skip in _reconcile_classical_nights() unconditional on attribution alone, with a counted/logged detach of any superseded RUN:-keyed event. — Closes CR-03: reconcile-then-attribute and attribute-then-reconcile orderings now converge on the same result, per 29-REVIEW.md CR-01's detach-never-delete rule.
- [Phase 33]: 33-09: FOMO_DATABASE_PATH env-var branch in settings.py lets demo notebooks run against a scratch DB copy; event pk 335 deleted (not re-attached) since its demo campaign is removed; task2's whole-file grep gate is over-scoped vs. its own must-haves -- satisfied narrowly at the public-table cell, documented as a deviation.
- [Phase 33]: 33-11: calendar modal handlers migrated inline to bootstrap.Modal.getOrCreateInstance(...).show() (not upstream's showModal() indirection) since the partial is itself htmx-swapped
- [Phase 33]: 33-11: campaign_decoration() guarded with isinstance(event, CalendarEvent) to fix a pre-existing AttributeError on the create-event form path (Rule 1 deviation)
- [Phase 33]: Plan 33-10: human-confirmation guard (_stale_attributions()) added only to campaign_reconciler.py, never to campaign_utils.unlink_event_from_run()/UNLINK_CLEARED_FIELDS -- Phase 28's human-initiated callers keep clearing a confirmed row; the automated sweep now defers to a prior human decision and reports detach_declined.
- [Phase 33]: Plan 33-10: _reconcile_classical_nights() reordered so ownership (_may_write()) is decided before a night's skip/attribution outcome (WR-13) -- a night both attributed to this run and contested by a foreign RUN:-keyed attribution now reports blocked, not skipped, and never detaches the foreign attribution.
- [Phase 34]: Phase 34 Plan 01: facility_for() is the sole path to a facility instance (never a shared LCOFacility()/SOARFacility() instance); the 'inconsistent' stage is projectable per D-13 (spans the request window directly, gets the [?] marker) rather than raising.
- [Phase 34]: Phase 34 Plan 01: wiring the post_save receiver globally broke 6 pre-existing tests (test_sync_lco_observation_calendar.py x4, test_campaign_attribution.py x1, test_campaign_attribution_views.py x1) whose fixtures assumed ObservationRecord.save() had no calendar side effect; fixed by disconnecting the projector's receiver around just the affected fixture-creation calls, leaving campaign_attribution.py and the retired sync command untouched (both owned by plan 34-02).
- [Phase 34]: 34-02: project_queryset()'s pre_fields_hook extension point lets the sweep's one-time observed-site lookup run mid-record without a second loop or a network call inside the projector module. — Keeps TRIG-02's no-network-call guarantee on observation_projector.py intact while still letting the sweep perform a real portal call between capturing the pre-sweep snapshot and building the intended fields.
- [Phase 34]: 34-02: sync_lco_observation_calendar retired outright (D-18) after a 38-row behaviour classification (16 covered, 10 migrated, 12 retired with reason) proved no behaviour was silently dropped. — ANNOT-03 requires one writer per source in the same key namespace; a classification table with a named destination or reason for every retired test is the audit trail that makes the deletion safe rather than a silent drop.
- [Phase 34]: Phase 34 Plan 03: observation_status_legend() is a fixed, hand-maintained marker vocabulary rather than derived from _TERMINAL_PREFIXES/status_border_css() -- deriving it risks ring-vs-label drift; Phase 37 owns the final wording.
- [Phase 34]: Phase 34 Plan 03: observation_series_decoration()'s docstring was rewritten to describe its no-write guarantee in prose after its first draft (mirroring campaign_decoration()'s literal .save()/.update()/.create()/get_or_create() phrasing) tripped the plan's own verify grep, which counts those substrings from the function's def line to end of file and cannot distinguish docstring text from code.
- [Phase 34]: Phase 34 Plan 04: project_observation_calendar_demo.ipynb runs against the real developer database (not a scratch copy) since SCHED-06's baseline must be captured over the same database a later re-execution re-checks; the receiver-demo section stays side-effect-free via a transaction.atomic() rollback. — Unlike this repo's other pre_executed/ notebooks, a scratch copy discarded at the end of the run would leave nothing for a post-observing-nights re-check to diff against.

### Pending Todos

- `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` — rename
  `calendar_utils.py`'s cross-module-consumed underscore-prefixed helpers
  (`_derive_telescope`, `_extract_instrument`, `_resolve_placement_block`,
  `_coarse_telescope_label`, `_aperture_class_from_telescope_code`) to reflect that the
  module is now a real shared API (3 consumers); low-priority style cleanup found while
  verifying the 2026-06-23 extraction todo was complete.

- `2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md` — move
  `_reconcile_classical_nights()`'s `sun_event()` call inside the `existing is None`
  branch so idempotent sweeps stop paying per-night astropy solar scans for results
  that are discarded (finding F2, 2026-09-01 branch review).

- `2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md` — add the
  `is_offered_candidate()` server-side guard to `AttributionDecisionView._dismiss()`,
  matching every confirm path, so a stale/tampered staff POST can't persist a dismissal
  for a never-offered pair or report false success for nonexistent pks (finding F4,
  2026-09-01 branch review).

- `2026-09-01-add-ttl-cache-to-attribution-banner-count.md` — wrap
  `orphans_needing_attribution_count()` in a short-TTL cache (campaign_gap.py pattern)
  so the campaign-list page stops rebuilding both attribution backlogs per request
  (finding F1, downgraded Medium→Low after measuring 23 ms / 64 queries at 31 orphans
  on the dev DB; opportunistic fix, 2026-09-01 branch review).

- `2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md` —
  ADAPT-03/Phase 32 should target SOAR, not Gemini (GEMFacility has no queue read-back),
  and Phase 33's outcome propagation is structurally impossible for Gemini; found via
  gap G-31-3 and diagnosed in `.planning/debug/gemini-vs-soar-facility-scope.md`.

- Carried-forward items in Deferred Items below.

### Blockers/Concerns

None blocking. v2.2 "One Canonical Run Record" shipped and closed 2026-09-01 (6 phases, 33 plans, 24/24 requirements). One non-blocking follow-up carried into the next milestone: `import_campaign_csv.py`'s `site_needs_review` is computed from the pre-preservation `telescope_class` value rather than the post-guard value (30-REVIEW.md WR-01) — recommend a future quick task.

Carried forward from Phase 33 (completed 2026-09-10) into the phases that own them — none blocks Phase 34 planning:

- **[Phase 34]** `CalendarEventMeta.observation_group` has no declared ordering on its reverse manager; if Phase 34's projector adds a reader over that relation it must set an ordering or add a shuffled-insertion test (UAT decision 2026-09-09, test 3, option A — absence-by-grep evidence accepted for now).
- **[Phase 34]** PROJ-04's shared-title-stem clause is still open — Phase 33 shipped only the carrier fields.
- **[Phase 35]** Leftover `RUN:{pk}:{date}` duplicates on a night after the reconciler's detach (WR-09: the `CalendarEvent` row survives by design) are Phase 35 SC 5's responsibility.
- **[Phase 37 or later]** `campaign_decoration()`'s `#run-{pk}` anchor only lands on the campaign table's first page (>25 runs — 33-06 WR-08); pinned as a tested limitation rather than fixed, because computing the page would add a per-event query.
- The stale v2.3 note about "Phase 33's aggregation rule" is gone: OUTCOME-01..04 were dropped with v2.3, and v2.4's Phase 33 had no aggregation rule.

Phase 31's scheduling-track host-facts gap (previously listed here) is resolved: the spike got real answers from the operator (cron + `flock -n`, confirmed present) — see SCHED-07 in PROJECT.md Key Decisions.

### Quick Tasks Completed

| # | Description | Date | Commit | Status | Directory |
|---|-------------|------|--------|--------|-----------|
| 260903-h1v | Add backfill_lco_observations management command: backfill ObservationRecords, missing non-sidereal Targets, and ObservationGroups from the LCO portal by proposal code | 2026-09-03 | f874531 | complete | [260903-h1v-add-backfill-lco-observations-management](./quick/260903-h1v-add-backfill-lco-observations-management/) |
| 260903-ik7 | Fix backfill_lco_observations --dry-run summary: wire would-create/update/unchanged, target and group counters, add embedded-block vs fallback-lookup counters | 2026-09-03 | ec11123 | complete | [260903-ik7-fix-backfill-lco-observations-dry-run-su](./quick/260903-ik7-fix-backfill-lco-observations-dry-run-su/) |
| 260903-jid | Fix backfill_lco_observations doubled summary line: drop explicit stdout write, keep return so Django prints it once | 2026-09-03 | 699908a | complete | [260903-jid-fix-backfill-lco-observations-doubled-su](./quick/260903-jid-fix-backfill-lco-observations-doubled-su/) |
| 260903-kpy | Collect every Target touched by a backfill_lco_observations sweep into a <proposal>_targets TargetList (create-or-reuse, idempotent, dry-run aware, --target-list override) | 2026-09-03 | 1082550 | complete | [260903-kpy-collect-every-target-touched-by-a-backfi](./quick/260903-kpy-collect-every-target-touched-by-a-backfi/) |

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| requirement | ESO-10 (`sync_eso_observation_calendar` command) | v2 — unblocked by Phase 13's Bypass verdict; explicitly out of scope for v2.3 (LCO/SOAR/Gemini only) | v1.7 close |
| requirement | ESO-11 (paired ESO demo notebook) | v2 — unblocked by Phase 13's Bypass verdict; explicitly out of scope for v2.3 | v1.7 close |
| requirement | SCHED-06 (progressive-disclosure window-narrowing UI) | **Un-deferred — now in v2.3 scope, mapped to Phase 33**, re-scoped against the v2.2 four-stage window pipeline | v2.1 requirements |
| requirement | SUBMIT-06/07 (trusted-PI self-approval; submission status lookup) | v2 — deferred again at v2.3 requirements; unrelated to automatic run sync | v2.0 close |
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

Last session: 2026-09-11T04:57:59.527Z
Stopped at: Completed 34-04-PLAN.md
Resume file: None

## Operator Next Steps

- Phase 33 complete (2026-09-10). Start Phase 34 "The Observation Projector & Trigger" with `/gsd-discuss-phase 34` — no CONTEXT.md exists yet for it.
- Phase 34 is now safe to build: the reconciler annotates only (ANNOT-01), the `observation_record`/`observation_group` carrier fields it writes to exist (PROJ-04), and campaign decoration survives base re-projection (ANNOT-02). Its paired-docs scope is large (new sweep notebook, migrating `sync_lco_observation_calendar_demo.ipynb`, the runbook's LCO sync section) — plan it in from the start.
- `.planning/REQUIREMENTS.md`: `phase.complete` flagged 5 REQ-IDs present in the body but missing from the Traceability table (UPSTREAM-01, ESO-10, ESO-11, SUBMIT-06, SUBMIT-07 — all deferred/out-of-milestone items); add them manually when next editing that file.
