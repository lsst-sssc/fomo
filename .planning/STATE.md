---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Campaign Coordination for Rare/Urgent Objects
current_phase: 17
status: verifying
stopped_at: Completed 17-03-PLAN.md
last_updated: "2026-07-05T04:42:59.530Z"
last_activity: 2026-07-05
last_activity_desc: Completed quick task 260705-l1v: Fix approval-queue site-visibility gap
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 13
  completed_plans: 13
  percent: 100
current_phase_name: coverage-gap-analysis-deferrable-to-v2-1
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-02 — v2.0 milestone opened)

**Core value:** When the next 4I-class object appears, FOMO replaces the ad-hoc Google Sheet as the community's campaign-coordination hub — target-linked observing runs, submission with oversight, and a per-object campaign view.
**Current focus:** Phase 17 — coverage-gap-analysis-deferrable-to-v2-1

## Current Position

Phase: 17
Plan: Not started
Status: Phase complete — ready for verification
Last activity: 2026-07-05 — Completed quick task 260705-l1v: Fix approval-queue site-visibility gap
Progress: [███████░░░] 3/4 phases

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
- [Phase 15]: render_run_status/render_approval_status resolve the raw field value via Accessor(record) rather than trusting django-tables2's value kwarg, since django-tables2 auto-calls get_FOO_display() for model-instance rows before invoking a custom render_ method — django-tables2 handed staff requests an already-humanized label and anonymous requests the raw code, breaking the RunStatus/ApprovalStatus TextChoices lookup for staff -- contradicts 15-RESEARCH.md Pitfall 2's stated assumption
- [Phase 15]: reverse('tom_targets:detail', ...) resolves via Django's application-namespace fallback even though the registered instance namespace is 'targets' -- confirmed live, no adjustment needed
- [Phase 15]: campaign_links.html applies mr-2 mb-2 unconditionally to every link rather than only wrapping 2+ campaigns in a flex div -- simpler, harmless for the single-link case, matches the plan's literal Task 2 action text
- [Phase 16]: CampaignRunSubmissionForm is a plain forms.Form, not ModelForm, because CampaignRun.telescope_instrument has no blank=True on the model -- a ModelForm would wrongly force it required, contradicting D-05
- [Phase 16]: EMAIL_BACKEND placed before the local_settings.py import block so a production override always wins
- [Phase 16]: CampaignRun.objects.create() wrapped in transaction.atomic() savepoint inside the IntegrityError handler -- without it, the caught exception poisons the outer request/test transaction and the form re-render raises TransactionManagementError instead of showing the friendly duplicate-submission error
- [Phase 16]: _notify_staff wraps reverse('campaigns:approval_queue') in try/except NoReverseMatch with a hardcoded fallback path, since that URL name is added by Plan 03 (Wave 3) which has not landed yet at Plan 02's execution point in the wave sequence
- [Phase 16]: CSRF token for the approval-queue Actions column's per-row mini-forms is minted via get_token(request) inside ApprovalQueueTable.render_actions (request passed as an explicit __init__ kwarg), keeping {% render_table %} intact instead of switching to a manual template row-loop
- [Phase 16]: decided_qs is materialized to a list and passed order_by=() before construction of ApprovalQueueTable, fixing a 'Cannot reorder a query once a slice has been taken' crash caused by the inherited CampaignRunTable.Meta.order_by=('-obs_date',) default sort colliding with the already-sliced [:20] queryset
- [Phase 16]: [Phase 16 P04]: Three pre-existing Phase 15 anonymous-client tests switched to the staff client since D-09 legitimately changes anonymous visibility and those tests exercise generic table mechanics (pagination/run_status coverage/filter semantics) unrelated to approval-status gating
- [Phase 16]: [Phase 16 P05]: Fix scoped entirely to ApprovalQueueTable.Meta (exclude + sequence) -- CampaignRunTable untouched, preserving Phase 15 D-09 spreadsheet-parity; sequence uses the '...' ellipsis token instead of enumerating all remaining columns
- [Phase 17]: Multi-target campaign target=None CampaignRuns are collected into a separate unattributed_runs list, never counted as claiming either target's date (Pitfall 4)
- [Phase 17]: Single-target campaigns: claimed_dates() does not filter by target at all -- the single target is implied and target=None is the common real-data case, matching import_campaign_csv's D-07 precedent
- [Phase 17]: gap_analysis_available(campaign) placed as a module-level function (not a classmethod) in campaign_views.py so Plan 03's CampaignRunTableView can import and reuse it directly for button-gating
- [Phase 17]: CampaignGapAnalysisView validates target/site membership via raw request.GET.get() + explicit queryset checks (not solely the form's ModelChoiceField validation), preserving 400 Bad Request semantics required by T-17-01
- [Phase ?]: [Phase 17 P03] CampaignRunTableView.get_context_data() now supplies gap_analysis_available to context, reusing the Plan 02 gap_analysis_available(campaign) helper rather than duplicating its target-count/resolved-site logic (Rule 2 deviation).
- [Phase ?]: [Phase 17 P03] CampaignGapAnalysisView's IDOR 400 branches re-render the gap-analysis template with status=400 and idor_error=True instead of a bare HttpResponseBadRequest, so the UI-SPEC's alert-danger error copy can show (Rule 2 deviation); verified this doesn't break Plan 02's IDOR test, full solsys_code suite re-confirmed 326/326 green.

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

### Quick Tasks Completed

| # | Description | Date | Commit | Status | Directory |
|---|-------------|------|--------|--------|-----------|
| 260705-l1v | Fix approval-queue site-visibility gap: show site_raw in the pending CampaignRun approval queue and stop the approval endpoint from fabricating placeholder Observatory rows for unresolvable free-text site names (found during v2.0 manual UAT) | 2026-07-05 | 959a78d | Verified | [260705-l1v-fix-approval-queue-site-visibility-gap-s](./quick/260705-l1v-fix-approval-queue-site-visibility-gap-s/) |

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| requirement | ESO-10 (`sync_eso_observation_calendar` command) | v2 — unblocked by Phase 13's Bypass verdict; out of scope for v2.0 (unrelated to campaign coordination) | v1.7 close |
| requirement | ESO-11 (paired ESO demo notebook) | v2 — unblocked by Phase 13's Bypass verdict; out of scope for v2.0 | v1.7 close |
| todo | `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — extract site/telescope mapping and instrument extraction into own module | Deliberately deferred; no second consumer yet | v1.7 close |
| seed | SEED-001 — file upstream `tom_eso` feature requests | Dormant; trigger is TOM Toolkit maintainer bandwidth or a future ESO milestone start | v1.7 close |
| seed | SEED-002 — ESO ObservationRecord-centric future intent | Dormant; unrelated to v2.0 campaign-coordination scope | v1.7 close |
| todo | `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` — rename `calendar_utils.py`'s private helpers to reflect shared-module status | Low-priority style cleanup; no functional impact | v2.0 close |
| seed | SEED-001 — file upstream `tom_eso` feature requests (re-acknowledged) | Still dormant at v2.0 close | v2.0 close |
| seed | SEED-002 — ESO ObservationRecord-centric future intent (re-acknowledged) | Still dormant at v2.0 close | v2.0 close |

## Session Continuity

Last session: 2026-07-04T22:49:26.740Z
Stopped at: Completed 17-03-PLAN.md
Resume file: None

## Operator Next Steps

- Phase 16 (all 5 plans, including gap closure 16-05) is complete. Plan Phase 17 (Coverage-Gap Analysis, deferrable to v2.1) with `/gsd-plan-phase 17` when ready, or defer per milestone scope.
