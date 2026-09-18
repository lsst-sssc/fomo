---
gsd_state_version: "1.0"
milestone: v2.4
milestone_name: Observation-First Calendar
current_phase: 36
current_phase_name: Unattended Operation
status: executing
stopped_at: Completed 36-09-PLAN.md (gap closure for G-36-5)
last_updated: "2026-09-18T20:22:18.675Z"
last_activity: 2026-09-18
last_activity_desc: Phase 36 execution started
state_head: be955cf59521ebf7e75f18a9381ee701bdf26308
progress:
  total_phases: 5
  completed_phases: 35
  total_plans: 52
  completed_plans: 52
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-16 — after Phase 35 complete)

**Core value:** The calendar is driven by what actually happened — one event per `ObservationRecord`, narrowing on every save with no operator action; allocations project intent nights until a real observation retires them; campaigns annotate, never own.
**Current focus:** Phase 36 — Unattended Operation

## Current Position

Phase: 36 (Unattended Operation) — EXECUTING
Plan: 2 of 9
Status: Ready to execute
Last activity: 2026-09-18 — Phase 36 execution started

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
| 34 | 7 | - | - |
| 35 | 25 | - | - |
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
| Phase 34 P05 | 24min | 2 tasks | 3 files |
| Phase 34-the-observation-projector-trigger P06 | ~10min | 2 tasks | 1 files |
| Phase 34 P07 | ~50min | 3 tasks | 3 files |
| Phase 35 P01 | 53min | 3 tasks | 8 files |
| Phase 35 P02 | 90min | 3 tasks | 5 files |
| Phase 35 P03 | 21min | 2 tasks | 5 files |
| Phase 35 P04 | 77min | 3 tasks | 8 files |
| Phase 35 P05 | 25min | 3 tasks | 7 files |
| Phase 35 P06 | 95min | 3 tasks | 10 files |
| Phase 35 P07 | 195min | 3 tasks | 4 files |
| Phase 35 P08 | 30min | 3 tasks | 2 files |
| Phase 35 P09 | 75min | 3 tasks | 5 files |
| Phase 35 P10 | 50min | 3 tasks | 3 files |
| Phase 35 P11 | 40min | 2 tasks | 3 files |
| Phase 35 P12 | ~40min | 3 tasks | 2 files |
| Phase 35 P13 | 20min | 3 tasks | 3 files |
| Phase 35 P14 | 35min | 3 tasks | 3 files |
| Phase 35 P15 | ~35min | 2 tasks | 2 files |
| Phase 35 P16 | ~10min | 2 tasks | 2 files |
| Phase 35 P17 | 35min | 2 tasks | 4 files |
| Phase 35 P18 | 35min | 3 tasks | 3 files |
| Phase 35 P19 | 18min | 3 tasks | 6 files |
| Phase 35 P20 | 65min | 3 tasks | 2 files |
| Phase 35 P21 | 41min | 2 tasks | 4 files |
| Phase 35 P22 | ~30min | 2 tasks | 2 files |
| Phase 35 P23 | 85min | 3 tasks | 6 files |
| Phase 35 P24 | 75min | 3 tasks | 4 files |
| Phase 35 P25 | 90min | 2 tasks | 2 files |
| Phase 36 P01 | 24min | 3 tasks | 8 files |
| Phase 36 P02 | 21min | 3 tasks | 7 files |
| Phase 36 P03 | 25min | 3 tasks | 2 files |
| Phase 36 P04 | 18min | 3 tasks | 3 files |
| Phase 36 P05 | 33min | 3 tasks | 5 files |
| Phase 36-unattended-operation P06 | 30min | 3 tasks | 6 files |
| Phase 36 P07 | 25 min | 3 tasks | 3 files |
| Phase 36 P08 | ~15min | 3 tasks | 4 files |
| Phase 36 P09 | 42min | 3 tasks | 3 files |

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
- [Phase 34]: [Phase 34-05] coerce_schedule_datetime() raises ValueError for an unusable schedule value rather than returning None — stage_for() already classified a non-None schedule field as a placed block (D-10); degrading it to None would draw a queued-looking event over the wrong window, so raising keeps the record a D-13 unprojectable one instead, with the save never aborted.
- [Phase 34]: [Phase 34-05] Task 2 pins the coercion contract with tests only -- no calendar_utils.py change, since Task 1's GREEN commit already implemented coerce_schedule_datetime() correctly
- [Phase 34]: 34-06: SCRATCH_DB_OVERRIDE pattern lets project_observation_calendar_demo.ipynb run against either the real developer database or a FOMO_DATABASE_PATH-routed scratch copy, guarding both the resolved-database assert and the SCHED-06 baseline JSON write on the same flag.
- [Phase 34]: 34-06: G-34-2 closed with live proof -- a real updatestatus run against a scratch copy of the developer database logged zero unprojectable lines, and the following dry-run sweep reported updated: 0, unprojectable: 0 for LCO; the 33 stale LCO events on the real developer database were left untouched for the operator's own SCHED-06 re-check (34-UAT.md Test 4).
- [Phase 34]: [Phase 34] 34-07: gsd_run check tdd-red-evidence could not classify Task 3's RED phase (TAP parser is Node-test-specific, does not recognize Django's unittest output); workflow.tdd_mode is false for this project, so RED was verified manually from the real named-assertion failure instead of the tool-mediated gate.
- [Phase 34]: [Phase 34] 34-07: one nbconvert re-execution retry was required after a real network flake (a retried site lookup succeeded on the second sweep, producing updated: 1) tripped the pre-existing convergence assert; re-cloned fresh from src/fomo_db.sqlite3 and re-executed once more per the plan's own re-run rule, converging cleanly.

Phase 34 decisions (2026-09-12; full rows in PROJECT.md Key Decisions):

- [Phase 34]: every projector receiver swallows and logs (`type(exc).__name__` only); a half-set schedule projects as `[?]` rather than raising — TRIG-02 held on the real `updatestatus` path once G-34-2 (portal ISO strings on the in-memory record) was fixed by strict `coerce_schedule_datetime()`.
- [Phase 34]: observed-site resolution is sweep-only, once per record; the receiver never calls the portal. A freshly COMPLETED record reads `[O] 1m0` until the next sweep.
- [Phase 34]: series decoration is display-time, read-only, from `CalendarEventMeta` links, gated on authenticated viewer + run visibility.
- [Phase 34]: notebook takeover + SCHED-06 baseline run against the real developer DB; re-executions scratch-routed with baseline-write and non-vacuous-takeover guards (G-34-3).
- [Phase 34 UAT]: `updatestatus` skips terminal-state records, so the 14 events that went stale under the pre-fix receiver need one sweep (F-34-1) — G-34-1 withdrawn; SCHED-06 closed on 4378332/4378046 narrowing via the receiver alone.
- [Phase 35]: [Phase 35]: 35-01: retired_nights() counts every night in the retired set as +1 in ReconcileResult.retired even when nothing existed yet to delete -- the plan's own Task 2 Test 1 requires retired == 1 on a run's very first reconcile when its linked record was already placed before the run ever reconciled.
- [Phase 35]: [Phase 35]: 35-01: project_allocation()'s D-14 convergence tracks a local-only retired_urls set (never returned) to stop a dry-run preview from double-counting a night the per-night loop already reported as retired -- real (non-dry) mode never needs this since the event is already deleted from the DB by the time convergence runs.
- [Phase 35]: [Phase 35]: 35-02: TestClassicalStage1's 5 tests with a confirmed test_allocation_projector.py counterpart are retired; the 2 without one (site-local key-date round-trip, mid-loop sun_event ValueError propagation) are kept, migrated to the ALLOC: key form.
- [Phase 35]: [Phase 35]: 35-02: TestAttributedNightSkip and TestObservingNightBoundary retired outright -- _attributed_nights() is dead code after 35-01 (defined, never called); TestObservingNightBoundary's coverage is fully duplicated by test_allocation_projector.TestAllocationNightBoundary.
- [Phase 35]: [Phase 35]: 35-02: found, not fixed (test-only plan scope) -- campaign_views._resolve_site()'s 'no new entries' message still names result.skipped_nights, which is now permanently 0 after the D-05 handoff superseded the skip-the-night rule; recommend a follow-up quick task.
- [Phase 35]: [Phase 35]: 35-03: night_start_utc/night_end_utc are TimeFields, not integer minutes-after-midnight -- they round-trip cleanly from the classical loader's own (hour, minute) integer parse and are directly admin-editable.
- [Phase 35]: [Phase 35]: 35-03: no admin.py change needed for the two new sub-night fields -- CampaignRunAdmin declares no explicit fields/fieldsets list, so both are editable by default.
- [Phase 35]: [Phase 35]: 35-03: _span_needs_remint() never re-mints on a null sub-night field (its expected boundary requires an uncheckable sun_event() call, which D-13 forbids for an existing night); a set field's expected boundary is computed directly with zero astropy cost.
- [Phase 35]: 35-04: receiver_on_run_observation_delete() uses Django's post_delete origin kwarg (not a plain CampaignRun existence check) to detect a run delete cascade -- empirically verified this Django version fires a CASCADE child's post_delete before the parent row's own DELETE, so the plan's own 'run is already gone by then' assumption was wrong. — Prevents leaking freshly re-minted ALLOC: nights moments before the run itself is deleted.
- [Phase 35]: 35-04: Task 2's linked-run re-project tests live in test_observation_projector_signals.py (new TestLinkedRunReproject class), matching the plan's own file assignment, not test_allocation_projector_signals.py where they were first drafted.
- [Phase 35]: Proposal-token syntax is a bracketed [proposal] token, consumed first in parse_run_line(), before the status grammar.
- [Phase 35]: source_identifier is derived from the run's own stored window_start/window_end, never the raw line day range, so a re-import recomputes a byte-identical key.
- [Phase 35]: A cancelled classical run's event description now also carries the shared writer's 'Run status: Cancelled' line (documented divergence from pre-cutover output).
- [Phase 35]: 35-06: legacy_deleted's delete branch is scoped by URL shape alone (bare vs. date-bearing), not by the run's current dispatch branch -- confirmed correct against real data, where several legacy_deleted rows belonged to still-allocation-dispatched runs, not just the 8 single-night runs D-10 sends to a container.
- [Phase 35]: 35-06: project_allocation() now returns legacy_urls_claimed so campaign_reconciler's dry-run preview never double-counts a night the per-night loop already accounted for -- found only by the real-database four-step proof run (Task 3), not by any unit test in this plan.
- [Phase 35]: 35-06: found, not fixed -- 8 of the developer database's 16 RUN:{pk} containers carried a stale pre-Phase-33 campaign-label title prefix, corrected by this plan's first post-D-12 reconcile sweep; a pre-existing staleness finding, not a Phase 35 defect.
- [Phase 35]: Phase 35 Plan 07: the reconciler demo's cutover section runs before the fixture/dispatch demo (opposite of the plan's literal action-item order) so its before-state capture reflects the pristine developer database, reproducing plan 35-06's exact real numbers (241/56/16/10/0/45 -> 233/0/16/1/57/48) rather than a diluted diff.
- [Phase 35]: Phase 35 Plan 07: deleting a CampaignRunObservation link restores its allocation night automatically via D-11's post_delete receiver, with no explicit reconcile_run() call needed -- the reconciler demo's observation-handoff cell was corrected mid-execution to assert this no-op convergence rather than a fresh creation.
- [Phase 35]: [Phase 35]: 35-08: cutover_classical_allocations's duplicate_identity guard now queries CampaignRun.objects.filter(source_identifier=key) and recovers the claimant's own stored Source line: via _extract_source_line() -- an in-process seen_keys dict alone cannot protect a find-or-update that matches against the database (NF-19 BLOCKER).
- [Phase 35]: [Phase 35]: 35-08: seen_keys[key] is now claimed only after a group's convertibility gates (campaign mismatch, unknown status, all-foreign-attributed) have already passed, not at the moment its identity key first resolves, so an unconvertible group can no longer poison a convertible sibling's duplicate_identity error (IN-02).
- [Phase 35]: [Phase 35]: 35-09: except ZoneInfoNotFoundError placed ahead of (ValueError, Observatory.DoesNotExist) on load_telescope_runs' per-line try -- clause order, not breadth, decides which handler sees the exception since ZoneInfoNotFoundError subclasses KeyError (NF-21).
- [Phase 35]: [Phase 35]: 35-09: _raise_if_set_window_inverted() is a shared astropy-free guard called from both project_allocation() dry-run short-circuits (re-mint and create branches), replacing the create branch's inline duplicate -- a third caller of _mint_fields() has one guard to reuse instead of a third inline copy (NF-20).
- [Phase 35]: [Phase 35]: 35-09: legacy_urls_claimed.add() moved ahead of the takeover branch's _may_write() check, mirroring the retired branch's NF-09 fix -- a blocked legacy takeover event is claimed on every decision, not only the re-key path, so campaign_reconciler's foreign fold never double-counts it (NF-22).
- [Phase 35]: 35-10: rewrapped runbook duplicate_identity phrase onto single unwrapped lines so the plan's exact-phrase grep -cF gate matches per-line (RST rendering unaffected).
- [Phase 35]: 35-10: moved the notebook's existing duplicate-run-identity demo cell's fixture cleanup into the new second-invocation cell so both cells share the same live fixture, per the plan's literal requirement.
- [Phase 35]: [Phase 35]: 35-11: unrecognised-status skip path demonstrated via parse_run_line()'s own ValueError (a parenthetical status not in KNOWN_STATUSES), not the narrower except KeyError around _CLASSICAL_RUN_STATUS[parsed.status] -- that inner clause is unreachable via real parse_run_line() output since a module-level assert enforces its key set equals KNOWN_STATUSES.
- [Phase 35]: [Phase 35]: 35-11: malformed-timezone skip path demonstrated by temporarily mutating the already-seeded NTT Observatory's timezone (restored before any later cell resolves NTT again), since telescope_runs.SITES is a fixed 4-entry dict and a schedule line can only resolve to one of its four names.
- [Phase 35]: [Phase 35]: 35-12: inverted the cutover's database-scoped duplicate_identity guard predicate to a direct inequality (CR-01, BLOCKER) -- a claimant with no recoverable Source line: marker is now refused, never find-and-updated, since observation_details is admin/CSV/form-writable and its absence is not evidence of agreement.
- [Phase 35]: [Phase 35]: 35-12: TestDryRunAndRealRunAgree's _make_all_three_preconditions_fixture() needed a matching Source line: marker on its pre-existing CampaignRun after the CR-01 inversion, or the fixture's group was refused under duplicate_identity before its own key_collision/window_mismatch preconditions were ever reached (Rule 1 auto-fix).
- [Phase 35]: 35-13: _raise_if_set_window_inverted() now short-circuits only when BOTH sub-night fields are null, matching _span_needs_remint(); a half-null run on the re-mint branch resolves its missing boundary from existing.start_time/end_time rather than sun_event(), closing the third iteration of the dry-run/real-run inversion-guard parity bug (NF-10 -> NF-20 -> WR-01).
- [Phase 35]: 35-13: _stale_dated_events()'s claimed_legacy_urls Args description replaced with the same four-outcome, load-bearing-in-real-mode wording as the two already-corrected copies (NF-09/NF-17/NF-22), closing WR-03 -- the third and last stale copy of that contract.
- [Phase 35]: 35-14: load_telescope_runs.py's dry-run branch now folds run_created/run_updated/run_unchanged only after both preview_campaign_run_action() and reconcile_run(existing, dry_run=True) have returned, mirroring the real branch -- closing WR-02, the counter-parity regression 35-09's NF-21 fix introduced.
- [Phase 35]: [Phase 35]: 35-15: closed 35-VERIFICATION.md gap 2 -- rewrote the duplicate_identity runbook guarantee to state the Source line: marker precondition (no hedge words) instead of the unconditional promise that was still false on the CR-01 path; added the loader dry/real counter-parity sentence; regenerated reconcile_campaign_runs_demo.ipynb by re-execution so its committed output carries 35-12's corrected CommandError text.
- [Phase 35]: 35-16: reverted _raise_if_set_window_inverted()'s stored-boundary fallback to a two-parameter (run, night) signature -- the guard resolves both sub-night boundaries only from the run's own fields and returns silently when either is unknown, closing the fourth iteration of the dry-run/real-run inversion-guard parity bug (NF-10 -> NF-20 -> WR-01).
- [Phase 35]: 35-17: Narrowed the loader's create-arm dry-run claim (PROBE-P5) and documented the cutover's matching-marker re-run gotcha (PROBE-P4) rather than adding new write paths or refusal branches -- both are text-only corrections pinned by new tests.
- [Phase 35]: [Phase 35]: 35-18: narrowed the runbook's loader claim to the existing-run arm and added the brand-new-line's limitation, extended both duplicate_identity definitions with the second (database-claimant) cause, added a cutover 'Re-run gotcha' note, and regenerated both notebooks by re-execution against plans 35-16/35-17's fixes -- closing the third and intended-final gap-closure round for Phase 35.
- [Phase 35]: [Phase 35]: 35-19: implemented the design_rationale's provenance-recording fix for CR-01 (minted_sub_night_window on CalendarEventMeta) instead of the review's naive per-sweep sun_event() call, which would have broken TestNoSunEventRecompute's D-13 astropy-budget pin.
- [Phase 35]: [Phase 35]: 35-19: Rule 1 deviation -- rebuilt test_cutover_classical_allocations.py's TestCutoverSequenceContract convertible-group fixture with real sun_event()-derived boundaries; the shared round-hour convention sat outside CR-01's one-minute tolerance and the fix correctly re-minted it as a genuinely-stale legacy night.
- [Phase 35]: [Phase 35]: 35-20: closed CR-01/CR-03 (35-REVIEW.md iteration 8) via _remint_decline_reason() (shared _clearable_declined_and_unattributed() rule plus a re-mint-local staff-state check on observation_record/observation_group/is_verified) and a compute-before-destroy + scoped transaction.atomic() reorder in the re-mint branch, per design_rationale's decline-not-preserve, no-foreign-arm design rather than the review's literal fix snippet.
- [Phase 35]: 35-21 closed CR-02 (35-REVIEW.md iteration 8): the mint-provenance token now carries a version marker and run.site_id, not just the sub-night pair, so a CampaignRun.site correction on an already-projected run re-mints instead of reporting unchanged forever.
- [Phase 35]: [Phase 35]: 35-22: seeded a second real ground site (E10, Siding Spring) in the notebook's site-correction cell rather than reassigning to the existing satellite fixture -- a satellite reassignment would exercise dispatch-branch re-classification, not CR-02's provenance-token fix.
- [Phase 35]: [Phase 35]: 35-22: the declined-re-mint notebook cell seeds a dedicated single-night CampaignRun rather than reusing classical_run, mirroring TestRemintHumanConfirmationGuard's own fixture shape and avoiding compounded state across demos.
- [Phase 35]: [Phase 35]: 35-22 closes gap-closure round 5: the runbook's retired (5 reasons)/detach_declined (2 outcomes + remedy) documentation and a re-executed reconcile_campaign_runs_demo.ipynb now describe and prove plan 35-20's CR-01 guard and plan 35-21's CR-02 token together, discharging CLAUDE.md's paired-docs rule at round granularity.
- [Phase 35]: 35-23: CR-04 closed via a two-way split on the re-mint decline -- falls through to the plain-update path instead of continue-ing, so a declined night still receives its ordinary title/description/target_list refresh. — The decline must refuse only the destructive half (boundary rewrite); the update path is how a staff mark_cancelled action reaches an allocation night at all.
- [Phase 35]: 35-23: CR-05 closed by applying _clearable_declined_and_unattributed() directly to the retirement branch's own existing.delete(), proven on both the sweep and the no-sweep receiver path; deliberately does NOT reuse _remint_decline_reason() -- only confirmed_by declines a retirement, never is_verified=False or an observation link. — The re-mint branch destroys a row it intends to re-create and owes its contents a decision; the retirement branch removes a night genuinely superseded by the linked observation, so extending the veto there would leave a permanent duplicate night.
- [Phase 35]: 35-23: WR-06 closed by splitting detach_declined into detach_declined (confirmed_by-only legacy/allocation retire declines) and remint_declined (a separate re-mint decline, three causes), keeping detach_declined's existing message byte-identical. — One counter carrying two meanings made both printed operator messages false for one of them; splitting also removes CR-04's counter ambiguity.
- [Phase 35]: 35-24: WR-05 closed -- a fully-set sub-night pair never re-mints on a site correction (step 2 short-circuit), so the plain-update path now refreshes only the dark-window line, bounded to one sun_event(kind='dark') call per night per correction and never in --dry-run.
- [Phase 35]: 35-24: WR-07 qualified rather than fixed -- a declined-and-unrecorded night resolves once PER SWEEP (not once ever); WR-01's dry-run repetition stays separately open. WR-08 closed on the model and _remint_decline_reason(); the runbook half is plan 35-25's.
- [Phase 35]: 35-24: the escalated decision (35-VERIFICATION.md HVR#1) closed via a v3 token carrying a site-position fingerprint (site_id + SHA-256 lat/lon/altitude/timezone digest) alongside site_id -- an in-place Observatory correction now re-mints instead of reading unchanged forever.
- [Phase 35]: 35-25: extended the existing declined-re-mint notebook cell to also demonstrate CR-04's title-refresh property (same fixture, same save() call), rather than adding a duplicate cell pair
- [Phase 35]: 35-25: seeded dedicated Observatory/CampaignRun rows for both the CR-05 retirement-guard demo and the escalated-decision (in-place site-definition correction) demo, to avoid conflating multiple corrections' before/after state on shared fixtures
- [Phase 35]: 35-25 closes gap-closure round 6: the runbook's seven-passage counter-section rewrite and a re-executed reconcile_campaign_runs_demo.ipynb now describe and prove all four of this round's behaviour changes (CR-04, CR-05, WR-06 from plan 35-23; WR-05, WR-08 and the escalated decision from plan 35-24), discharging CLAUDE.md's paired-docs rule at round granularity

Phase 35 close decisions (UAT 2026-09-16; full rows in PROJECT.md Key Decisions):

- [Phase 35 UAT]: G-35-4 (in-place `Observatory` position/timezone correction must re-mint projected nights, with a >1 minute boundary-difference threshold so a trivial coordinate tweak never churns the calendar) — owner said "fix in round 6"; diagnosis (`.planning/debug/observatory-edit-leaves-nights-stale.md`) showed plans 35-23/35-24 had already shipped exactly that, threshold included (`_UNRECORDED_PROVENANCE_TOLERANCE`, `allocation_projector.py:77`); closed as already satisfied, no round 7.
- [Phase 35 UAT]: the declined-re-mint provenance-token seam (a declined re-mint records the run's current token onto a night whose `start_time` is still the pre-edit value; `allocation_projector.py:1441`, 35-REVIEW.md iteration 10 WR-01) deferred as a follow-up — reachable only by a sub-night edit and an in-place `Observatory` correction landing in the same sweep, no observable consequence.
- [Phase 35 UAT]: CR-05's narrowing of Success Criterion 3 / ALLOC-03 accepted as written — a night whose companion row a person has confirmed is never deleted on link (`detach_declined`, warning, runbook remedy); no rewording of the criterion requested.
- [Phase 35 UAT]: UAT Test 5 recorded as `pass` with the deferral text as its resolution rather than `skipped`, because `gsd_run phase uat-passed` counts a skipped test as a blocker even when the workflow's own deferred-follow-up rule produced it.
- [Phase 36]: 36-01: notifications.notify_staff() implements its own fail_silently try/except around send_mail() rather than delegating to send_mail()'s own parameter, so the outage-tolerance contract is identical regardless of which layer fails or is mocked. — Discovered while writing Task 3's mail-outage test -- patching send_mail() directly bypasses Django's internal per-backend fail_silently handling.
- [Phase 36]: 36-01: unattended.py never calls django.urls.reverse() -- a management-command-only process has not yet loaded the URL conf, and reverse() would trigger a full resolution that imports solsys_code.views via calendar_urls.py, reintroducing the ~1.6 GB SPICE-kernel import on every cron tick. — The failure email's admin/calendar links use notifications.absolute_url() with hardcoded paths instead of reverse()'d ones.
- [Phase 36]: [Phase 36]: 36-02: sweep_proposal() accepts created_after/created_before as raw ISO-8601 strings (parsed internally), not pre-parsed datetimes -- keeps the extracted function self-contained and Command.handle() reduced to CLI-only concerns (username resolution).
- [Phase 36]: [Phase 36]: 36-02: the watched-path's zero-rows/aggregate messages are written directly via self.stdout.write() with handle() returning None, avoiding BaseCommand.execute() double-printing the return value.
- [Phase 36]: [Phase 36]: 36-02: a per-row sweep failure is logged at logger.debug() with type(exc).__name__ only, never str(exc) -- keeps SCHED-10/D-17 credential-safety intact in the debug log too.
- [Phase 36]: 36-03: step functions never call call_command()/django.core.management.call_command -- each step imports and calls the underlying module function directly, verified by a source-count probe plus a negative test patching call_command at its own definition site.
- [Phase 36]: check_unattended's check_email() returns two CheckResults (EMAIL_BACKEND, staff_recipients), keeping the six-callable verify probe stable across Task 2's cron_line()/--send-test-email additions — Plan text describes check_email() as 'hard, two results'; cron_line() and _send_test_email() are deliberately non-check helpers outside the six named check_ functions
- [Phase 36]: 36-05: added a previously-missing backfill_lco_observations cheat-sheet row rather than blocking on the plan's 'update' framing — No such row existed at all; the command was already fully documented elsewhere, so adding the row closed a pre-existing gap rather than introducing new scope.
- [Phase 36]: 36-05: the runbook's locking guarantee for run_unattended is stated narrowly and verified against unattended.py's command_lock() call sites -- two ticks (incl. --step <name>) never overlap, but a direct manage.py invocation of the underlying sweep command is not locked against a tick — Cross-checked directly against source and step docstrings rather than assuming the discretion note's original aspiration held, so the runbook does not promise a guarantee the code does not provide.
- [Phase 36-unattended-operation]: Corrected heartbeat alert-window guidance to name both the check's expected ping interval (Period) and grace time (Grace), everywhere it appears — G-36-3 found the runbook named only the grace knob, leaving Period at its 1-day vendor default and disabling the dead-man layer for about a day; the fix (36-06) propagates the two-knob guidance to the runbook, crontab template, runner docstrings, preflight output, and the verification record
- [Phase 36]: [Phase 36-07]: Moved the stale Test 6 runbook-sufficiency entry out of re_verification.human_items_closed_by_uat into human_items_still_open with a RE-OPENED-by-G-36-1 marker — Makes 'no longer closed' unambiguous by list membership rather than relying on a reader noticing an appended marker string.
- [Phase 36]: [Phase 36-07]: Kept the SC5/truth-35 evidence cells' base verdict as VERIFIED with a parenthetical human-item qualifier, matching the exact form SC3 already uses, instead of downgrading the marker — The structural half (9 numbered steps, correct order) is genuinely machine-verified; only sufficiency-at-point-of-use is a human item.
- [Phase 36]: Promoted the single flat LCO_API_KEY setting to feed both the LCO and SOAR facility entries rather than adding a second SOAR_API_KEY (36-08). — One credential authenticates against the same LCO Observation Portal for both facilities; a second name would let the two drift.
- [Phase 36]: workflow.tdd_mode is false; RED for Task 2 verified manually per 36-01..36-04 precedent — gsd tdd-red-evidence's TAP parser targets node --test output with no Python/Django adapter

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

Carried forward from Phase 35 (completed 2026-09-16) — none blocks Phase 36 planning:

- **[Resolved 2026-09-16 — quick task `260916-o6n`]** `35-REVIEW.md` iteration 10 **CR-01** (critical): the retirement decline plan 35-23 added (CR-05) kept a human-confirmed `ALLOC:` night alive but ended in an unconditional `continue`, so the surviving night never received its title/description/target_list refresh — `mark_cancelled` never reached it. Fixed in `3a38858`/`c26cb97`: the decline now falls through to a shared `_refresh_labels()` (fields built once by `_label_fields()`, used by the decline branch, the dry-run preview and the real write), records no provenance token and makes no `sun_event()` call on that path, reuses `updated`/`unchanged` (no new counter; `ReconcileResult` unchanged); `TestDeclinedRetirementStillUpdatesLabels` (7 tests incl. receiver path, dry-run parity and a non-vacuous no-`sun_event` pin); runbook `detach_declined` section states the counter pair; notebook cell 20 demonstrates the `[CANCELLED]` refresh with executed output.
- **[Follow-up — deferred at UAT 2026-09-16]** `35-REVIEW.md` iteration 10 **WR-01** / `35-UAT.md` Test 5: a declined re-mint whose site also moved records the run's *current* provenance token onto a night whose `start_time` is still the pre-edit value (`_record_sub_night_provenance()` call at `allocation_projector.py:1441`; the comment at `:1362-1368` is false on that path). Reachable only when a sub-night edit and an in-place `Observatory` correction land in the same sweep; no observable consequence. Guard the call on "the re-mint was not declined" when next in that code.
- **[Advisory — `35-VERIFICATION.md` advisory #1, `35-REVIEW.md` iteration 10 WR-04]** the step-4 staleness warning at `allocation_projector.py:739-750` says "unrecorded-provenance night" on both entry paths of the boundary comparison, so an operator who just corrected a site position is pointed at the runbook's legacy-audit reason (5) instead of the site-definition-correction paragraph. Branch the wording on the entry path (a `token_trusted`-and-fingerprint-differed flag is already in scope); `_UNRECORDED_PROVENANCE_TOLERANCE` is now too narrow a name for its two callers. Notebook cell 22 prints the old wording.
- **[Review residue, `35-REVIEW.md` iteration 10]** WR-02 (the `v2`→`v3` token bump cannot re-audit a fully-set sub-night run, so WR-05's dark-window refresh is unreachable for nights that already exist), WR-03 (`project_allocation()` docstring still states the D-13 absolute), WR-05 (two over-broad runbook claims), WR-06 (no notebook cell for the fully-set dark-window refresh), WR-07 (`load_telescope_runs` night summary omits both decline counters), and the carried-forward WR-08..WR-11 / IN-01 / IN-03 (each with a one-line disposition in `35-23-PLAN.md`'s `<review_dispositions>` ledger). Triage when the CR-01 quick task is in the same code.
- **[Phase 37]** Two UAT-deferred behaviour ideas for the status/vocabulary work: failed or aborted records should keep their last scheduled / partly executed window (expired ones keep the original window, as today); `--dry-run` should show `site_lookups` as not attempted (e.g. `n/a (dry run)`) rather than `0`.
- **[Phase 37 or later]** `campaign_decoration()`'s `#run-{pk}` anchor only lands on the campaign table's first page (>25 runs — 33-06 WR-08); pinned as a tested limitation rather than fixed, because computing the page would add a per-event query.
- **[Bookkeeping]** `.planning/REQUIREMENTS.md`: `phase.complete` again flagged 5 REQ-IDs present in the body but missing from the Traceability table (UPSTREAM-01, ESO-10, ESO-11, SUBMIT-06, SUBMIT-07 — all deferred/out-of-milestone); add them when next editing that file.
- Resolved in Phase 35 (removed from this list): the 14 stale LCO events / un-routed projector-notebook re-execution owed from `34-UAT.md` (discharged by 35-11 on 2026-09-15, `sched06-baseline.json` rewritten, database already converged by the F-34-1 sweep), and the leftover `RUN:{pk}:{date}` duplicates after a detach (SC 5 verified — 56 → 0 on the real database, 48 rekeyed + 8 legacy_deleted).

Phase 31's scheduling-track host-facts gap (previously listed here) is resolved: the spike got real answers from the operator (cron + `flock -n`, confirmed present) — see SCHED-07 in PROJECT.md Key Decisions.

### Quick Tasks Completed

| # | Description | Date | Commit | Status | Directory |
|---|-------------|------|--------|--------|-----------|
| 260903-h1v | Add backfill_lco_observations management command: backfill ObservationRecords, missing non-sidereal Targets, and ObservationGroups from the LCO portal by proposal code | 2026-09-03 | f874531 | complete | [260903-h1v-add-backfill-lco-observations-management](./quick/260903-h1v-add-backfill-lco-observations-management/) |
| 260903-ik7 | Fix backfill_lco_observations --dry-run summary: wire would-create/update/unchanged, target and group counters, add embedded-block vs fallback-lookup counters | 2026-09-03 | ec11123 | complete | [260903-ik7-fix-backfill-lco-observations-dry-run-su](./quick/260903-ik7-fix-backfill-lco-observations-dry-run-su/) |
| 260903-jid | Fix backfill_lco_observations doubled summary line: drop explicit stdout write, keep return so Django prints it once | 2026-09-03 | 699908a | complete | [260903-jid-fix-backfill-lco-observations-doubled-su](./quick/260903-jid-fix-backfill-lco-observations-doubled-su/) |
| 260903-kpy | Collect every Target touched by a backfill_lco_observations sweep into a <proposal>_targets TargetList (create-or-reuse, idempotent, dry-run aware, --target-list override) | 2026-09-03 | 1082550 | complete | [260903-kpy-collect-every-target-touched-by-a-backfi](./quick/260903-kpy-collect-every-target-touched-by-a-backfi/) |
| 260911-9rd | Close Phase 34 review finding CR-01 by analysis: record the LCO/SOAR shared request-ID rationale in 34-REVIEW-FIX.md and add an event_url() regression test | 2026-09-11 | 0a87174 | complete | [260911-9rd-close-phase-34-review-finding-cr-01-by-a](./quick/260911-9rd-close-phase-34-review-finding-cr-01-by-a/) |
| 260913-ng8 | Fix 35-REVIEW.md WR-07: skip the CampaignRun write in cutover_classical_allocations for an all-foreign-attributed group | 2026-09-13 | bb20c2e | complete | [260913-ng8-fix-35-review-md-wr-07-skip-the-campaign](./quick/260913-ng8-fix-35-review-md-wr-07-skip-the-campaign/) |
| 260913-npq | Fix 35-REVIEW.md WR-11: cutover_classical_allocations detects ALLOC: key collisions (in-run and existing-url) and reports them as key_collision; runbook pins cutover-before-import ordering | 2026-09-13 | 338625d | complete | [260913-npq-fix-35-review-md-wr-11-make-cutover-clas](./quick/260913-npq-fix-35-review-md-wr-11-make-cutover-clas/) |
| 260913-rmd | Fix 35-REVIEW.md NF-03: resolve sub-night boundaries against the site's own UTC night span (three bands) instead of the sign of its UTC offset | 2026-09-13 | 09104f5 | complete | [260913-rmd-fix-35-review-md-nf-03-replace-the-sign-](./quick/260913-rmd-fix-35-review-md-nf-03-replace-the-sign-/) |
| 260913-ti3 | Fix 35-REVIEW.md NF-02: hoist the cutover per-event preconditions so --dry-run and the real pass agree on every check, count and exit status; add the window_mismatch reason | 2026-09-13 | bc15c4d | complete | [260913-ti3-fix-35-review-md-nf-02-hoist-the-cutover](./quick/260913-ti3-fix-35-review-md-nf-02-hoist-the-cutover/) |
| 260913-ti1 | Fix 35-REVIEW.md NF-01/NF-06/NF-09 (+NF-07 docs): one total-partition helper makes writable-but-unattributed events deletable at all four stale-event paths; _may_write now agrees with writable_allocation_events; declined legacy event counted once | 2026-09-13 | a0834b3 | complete | [260913-ti1-fix-35-review-md-nf-01-nf-06-nf-09-make-](./quick/260913-ti1-fix-35-review-md-nf-01-nf-06-nf-09-make-/) |
| 260916-o6n | Fix 35-REVIEW.md iteration 10 CR-01: a declined (human-confirmed) allocation-night retirement now falls through to a shared _refresh_labels() so mark_cancelled reaches it; 7 tests, runbook detach_declined counter pair, reconciler notebook re-executed | 2026-09-17 | c26cb97 | — | [260916-o6n-fix-35-review-md-iteration-10-cr-01-give](./quick/260916-o6n-fix-35-review-md-iteration-10-cr-01-give/) |
| 260918-bn7 | Document the WR-17 suppression-state fallback and the WR-16 lock-held exit-code normalization in the unattended runbook section | 2026-09-18 | 6ded6b4 | — | [260918-bn7-document-the-wr-17-suppression-state-fal](./quick/260918-bn7-document-the-wr-17-suppression-state-fal/) |

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

Last session: 2026-09-18T20:22:18.586Z
Stopped at: Completed 36-09-PLAN.md (gap closure for G-36-5)
Resume file: None

## Operator Next Steps

- Phase 35 complete (2026-09-16); the owner-chosen quick task `260916-o6n` for `35-REVIEW.md` iteration 10 CR-01 landed the same day (`3a38858`, `c26cb97`), so no critical review finding is open against Phase 35.
- Start Phase 36 "Unattended Operation" with `/gsd-discuss-phase 36` — no phase directory or CONTEXT.md exists yet. Its inputs are in place: the projector sweep (`project_observation_calendar`), the discovery backfill (`backfill_lco_observations`) and the reconciler (`reconcile_campaign_runs`) all exist as zero-required-argument commands; Phase 31 settled cron + `flock -n` on the real host; `DISCOVER-01` replaces `backfill_lco_observations`' `--proposal` arguments with an admin-editable watched-proposal list. Paired docs: `backfill_lco_observations_demo.ipynb` and a new unattended-operation runbook section.
- `.planning/REQUIREMENTS.md`: `phase.complete` flagged 5 REQ-IDs present in the body but missing from the Traceability table (UPSTREAM-01, ESO-10, ESO-11, SUBMIT-06, SUBMIT-07 — all deferred/out-of-milestone items); add them manually when next editing that file.
