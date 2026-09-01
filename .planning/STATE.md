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

See: .planning/PROJECT.md (updated 2026-09-01 — Phase 30 complete, milestone v2.2 finished)

**Core value:** An observing run exists once, as a `CampaignRun`, and everything else is derived from it — the calendar events that show it, the observation records that realise it, and the coverage-gap analysis that counts it.
**Current focus:** v2.2 complete — awaiting `/gsd-complete-milestone`

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

- [Phase 30]: REJECTED-run attribution exclusion enforced once via a shared `_ATTRIBUTION_INELIGIBLE_APPROVAL_STATUSES` constant read by both eligibility gates, not filtered separately at each display surface
- [Phase 30]: The ruff/format "drift" three phases (26, 27, 27.1) each logged and deferred was a misdiagnosis — repo was always clean under the pinned version; root-caused to CLAUDE.md documenting a bare unpinned `ruff` invocation, fixed by pinning `pyproject.toml` + routing the documented command through `pre-commit run` (no repo-wide reformat)
- [Phase 30]: `preserve_telescope_class` guard mirrors `preserve_site`'s shape exactly (decision computed beside inputs, widened pop, summary counter, per-row stderr line) rather than inventing a new pattern

Full rationale for each in PROJECT.md's Key Decisions table (Phase 30 rows).

All v1.0-v2.1 decisions logged in PROJECT.md Key Decisions table. The exhaustive per-plan v2.1 decision log previously kept here (roadmap-structure decisions, and one bullet per Phase 18-25 plan) has been cleared now that v2.1 has shipped and closed — nothing is lost: the milestone-level decisions are summarized in PROJECT.md's Key Decisions table (backfilled at close for Phases 18/19/20/21/23/24, which already had rows for 14/22/25), and the full fine-grained per-plan log remains verbatim in each phase's archived `PATTERNS.md`/`SUMMARY.md` under `.planning/milestones/v2.1-phases/`.

- [Phase quick-260722-tkt]: Field Targets created by --create-missing-targets are always type=SIDEREAL (fixed-sky pointings), distinct from the campaign's non-sidereal moving-object target by design
- [Phase quick-260722-tkt]: created_targets counter only reflects actually-persisted creations (0 in --dry-run); per-request stdout line still reports would-create/would-reuse intent
- [Phase quick-260722-twe]: epoch/pm_ra/pm_dec/parallax (from LCO wire keys epoch/proper_motion_ra/proper_motion_dec/parallax) are only set on newly-built field Targets, never on a reused existing Target — reuse never overwrites existing Target metadata
- [Phase quick-260722-uhh]: Target's admin URL/reverse name is `tom_targets_basetarget_changelist`, not `tom_targets_target_changelist` — `Target = get_target_model_class()` resolves to `BaseTarget` (no `TARGET_MODEL_CLASS` override in settings.py); tests derive the app_label/model_name dynamically rather than hardcoding either string
- [Phase quick-260722-ux0]: `facility.update_observation_status()` uses its own module-level `make_request` binding (`tom_observations.facilities.ocs.make_request`), separate from the one a caller module imports and patches — a test suite calling code that transitively invokes `update_observation_status()` must patch `LCOFacility.update_observation_status` itself (or the `ocs` module's `make_request`), not the caller's imported `make_request` name, or it will make a real live HTTP call
- [Phase quick-260722-uyz]: sync_lco_observation_calendar never populated CalendarEvent.target_list since its original Phase 04 implementation (confirmed via git log -p) — fixed by deriving it from record.target.targetlist_set.order_by('name').first() (deterministic alphabetically-first pick when a Target is in 2+ campaigns, None if in none); applies uniformly to both LCO and SOAR records since they share _build_event_fields()
- [Phase ?]: Quick 260723-02e: load_telescope_runs --campaign duplicates only the explicit-name TargetList lookup branch of backfill's _resolve_campaign (no interactive prompt); 'target_list' key always present in the fields dict for consistent no-churn FK diffing
- [Phase ?]: [Quick 260723-r5g]: sync_lco_observation_calendar's LCOFacility/SOARFacility expose no get_successful_observing_states() method — the successful-terminal state set must be derived as get_terminal_observing_states() minus get_failed_observing_states()
- [Phase ?]: [Quick 260724-tiz]: Added a separate TELESCOPE_PALETTE (brighter dark-surface set) rather than modifying PROPOSAL_PALETTE; per-telescope stripe re-implemented as a ::before pseudo-element (cal-event-classical + --tel-color) to avoid colliding with status_border_css's box-shadow ring on the same style attribute
- [Phase ?]: [Quick 260724-vb0]: Split TELESCOPE_PALETTE into two parallel palettes (TELESCOPE_PALETTE for legend vs white, TELESCOPE_STRIPE_PALETTE for stripe vs #5a6268 gray fill) since no 8-hue palette can clear 3:1 against both backgrounds; fixed the stripe's remaining white-facing edge with a one-sided opaque STRIPE_OUTER_EDGE_COLOR line rather than a hue change
- [Phase ?]: D-02 verdict confirmed-with-additions: the rename's two class-name imports (admin.py, sync_lco_observation_calendar.py) are the only real-code risks and both fail loudly, but the admin reverse-URL name and the four test modules' own class-name references are two more consumer sites the original four-point checklist missed
- [Phase ?]: Phase 26 evidence posture differs from Phase 18: writes for real against a disposable scratch DB file copy (tmp/26-spike-db-copy.sqlite3) rather than rolling back transaction.atomic() blocks against the live DB
- [Phase 26]: D-10 site-local-night derivation used simple timezone conversion + .date() (not a noon-anchored night-boundary heuristic), matching CONTEXT.md's own D-10 illustration
- [Phase 26]: Measured gap: CampaignRun pk=1's real site (Observatory obscode E10) has a blank timezone field in the dev DB; D-11 prototype substitutes Australia/Sydney explicitly and flags this as a Phase 27 pre-migration backfill item
- [Phase ?]: source vocabulary locked at six values (five roadmap values + LEGACY); source/telescope_class stay out of both existing CampaignRun partial unique constraints
- [Phase ?]: Reconciler event key locked: RUN:{run_pk}:{date} with {date} always the site-local observing night, not the naive UTC date
- [Phase ?]: Adopt-vs-gap-fill write strategy (D-11) deliberately deferred to Phase 29 per human decision at the 26-03 task-1 checkpoint, not locked
- [Phase ?]: Migration shape locked: RenameModel CalendarEventTelescopeLabel->CalendarEventMeta then three AddField ops; rename checklist is six integration points, not four
- [Phase 26-canonical-record-spike]: Bare RUN:1 span key is measurably stable under a window-narrowing stage transition; the rejected per-night RUN:1:{date} key is not (one key orphaned) -- direct code-level answer for queue-run projection key form
- [Phase 26-canonical-record-spike]: RECON-07 baseline splits 8 QUEUE / 11 CLASSICAL / 0 SPACE of 19 runs -- queue-run projection affects a substantial minority, not a corner case, of the flagship visibility criterion
- [Phase 26-canonical-record-spike]: No verdict on span/none/per-night chosen here -- deliberately left for plan 26-05 task 1, mirroring D-11's write-strategy deferral
- [Phase 26]: Queue-run projection settled (human decision): a queue-scheduled run gets a bare RUN:{run_pk} whole-window container event coexisting with its real ObservationRecord-derived CalendarEvents, which already narrow/refine as observations are scheduled and observed (verified against sync_lco_observation_calendar.py, not assumed)
- [Phase 26]: D-05's 80x5=400 class-wide fan-out figure does not survive -- pk=29/pk=30 are both QUEUE run-type, so both take the settled bare-container form (1 event, not 80, not 400); the site-fanout half of D-05 stands unchanged
- [Phase 27-01]: _observations_block_response() stays owned by test_sync_lco_observation_calendar.py (still used by many command-behaviour tests there); test_calendar_utils.py imports it rather than duplicating it
- [Phase 27-01]: derive_telescope_class's aperture regex uses one generic metre-phrase pattern (digit-must-precede-'m') instead of enumerating literal phrases -- this ordering is what rejects MuSCAT4's trailing digit without a special case
- [Phase 27-01]: D-12's subset-assertion test computes calendar_utils' aperture-class set by calling aperture_class_from_telescope_code() on real codes rather than hardcoding the set literal a second time
- [Phase 27-02]: D-22 mutation proof run manually: flipping create_placeholder to True fabricated a placeholder Observatory on network failure; reverted and confirmed byte-identical
- [Phase 27-02]: Live repair (Task 2) intentionally produced no git commit -- dev DB is gitignored; evidence is the before/after table in the SUMMARY
- [Phase 27-02]: MPC Obscodes API was reachable during the live run: HST (pk 8,12) and Swift (pk 13) resolved via genuine tier-2 lookups, creating 2 new real Observatory rows
- [Phase 27-03]: related_name='telescope_label_meta' left byte-identical; run FK uses SET_NULL (not CASCADE) since the companion row also carries is_verified history
- [Phase 27-03]: Migration 0009 (AddField run) kept separate from 0008 (RenameModel) so a rename regression and a new-field regression can never be confused for each other
- [Phase ?]: [Phase 27-04]: Migration 0010/0011 header comments rephrased to avoid literal AddField/CreateModel/RunPython.noop tokens in prose, so exact-count acceptance-criteria greps pass without a grep-literalism footnote
- [Phase ?]: [Phase 27-04]: ObservationRecord test fixtures use a separate record_owner user distinct from the confirmed_by user under test, since ObservationRecord.user is on_delete=DO_NOTHING and deleting a still-referenced user fails SQLite's deferred FK check
- [Phase 27-05]: Superuser (not merely is_staff=True) fixtures needed for save_formset inline tests -- DeleteProtectedModelForm.has_changed() gates on the inline model's own add/change permission
- [Phase 27-05]: telescope_class non-staff visibility (D-18) proven at the .values() queryset level, not as a rendered CampaignRunTable column -- campaign_tables.py is out of this plan's scope
- [Phase ?]: [Phase 27-06]: import_campaign_csv writes source=CSV_IMPORT and derives telescope_class via the shared calendar_utils.derive_telescope_class() helper, gated on site is None -- neither field enters the natural-key lookup
- [Phase ?]: [Phase 27-06]: PROJECT.md's stale Phase 25 pk=34 claim is date-pinned (2026-07-18) rather than deleted, preserving the pk=34 occurrence count; 26-CONTEXT.md's D-11 owned-nights framing gets a dated forward-pointer instead of a rewrite
- [Phase ?]: [Phase 29-06]: User-directed deviation added CampaignRun.Source.ESO_QUEUE (migration 0014) since real 3I/ATLAS ESO VLT queue rows had no matching source value -- not Rule 1/2/3, explicit user choice among 3 presented options
- [Phase ?]: [Phase 29-06]: Real dev-DB RECON-07 baseline measured as 26 approved/resolved/windowed 3I/ATLAS rows (10 QUEUE/11 CLASSICAL/5 SPACE) today, not the 19 (8/11/0) 26-DECISION.md originally cited -- Phase 27's live site-repair work resolved 4 satellite rows' sites after that spike's probe date
- [Phase 30]: Reused one shared frozenset constant (_ATTRIBUTION_INELIGIBLE_APPROVAL_STATUSES) across both eligibility gates rather than inlining the status literal twice — D-03 anti-drift rationale: one place to add a future disqualifying status
- [Phase 30]: Event-path fixture runs carry target=run_target while the record-orphan fixture uses a deliberately separate field_target in the same campaign — Proves the record-path tests cannot pass by accident via a reintroduced target-FK-equality check, the standing prohibition in _eligible_runs_for_record's docstring
- [Phase 30]: D-06/D-07: pinned pyproject.toml's ruff dev extra to 0.2.1 and routed CLAUDE.md's documented lint/format gate through pre-commit, closing the phantom ruff drift Phases 26/27/27.1 each independently logged
- [Phase 30]: D-10: repaired campaign_reconciler.py's five stale docstring references to deleted functions and corrected 26-DECISION.md's stuck-in-progress header
- [Phase 30]: [Phase 30-03]: D-04 telescope_class guard mirrors preserve_site -- preserve_telescope_class computed immediately after the derivation it gates on (not before, unlike preserve_site), a strict superset of the old blanking-only condition
- [Phase 30]: [Phase 30-03]: New tests placed in a sibling TestReImportTelescopeClassPreservation class rather than appended to TestReImportSitePreservation; test_telescope_class_never_blanked_by_reimport left byte-identical
- [Phase 30]: D-08/D-09/D-11: reconciled all five v2.2 phase VALIDATION.md files (nyquist_compliant: true) and amended v2.2-MILESTONE-AUDIT.md with true item dispositions; every closed tech-debt item cites where it was closed

### Pending Todos

- `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` — rename
  `calendar_utils.py`'s cross-module-consumed underscore-prefixed helpers
  (`_derive_telescope`, `_extract_instrument`, `_resolve_placement_block`,
  `_coarse_telescope_label`, `_aperture_class_from_telescope_code`) to reflect that the
  module is now a real shared API (3 consumers); low-priority style cleanup found while
  verifying the 2026-06-23 extraction todo was complete.

- Carried-forward items in Deferred Items below.

### Blockers/Concerns

None blocking. v2.2 "One Canonical Run Record" shipped 2026-09-01 (Phase 30, its last phase, completed 4/4 plans, verification 12/12 must-haves, regression gate 358/358 tests); awaiting `/gsd-complete-milestone`. One non-blocking follow-up recorded, not milestone-blocking: `import_campaign_csv.py`'s `site_needs_review` is computed from the pre-preservation `telescope_class` value rather than the post-guard value (30-REVIEW.md WR-01) — recommend a future quick task.

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
