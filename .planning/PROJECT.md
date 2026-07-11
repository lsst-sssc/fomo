# Telescope Runs Calendar — Stages 1, 2 & 3

## What This Is

A helper module and management commands for FOMO that:

1. (`solsys_code/telescope_runs.py`) resolves a telescope name to its observing site (via the `Observatory` model, by MPC obscode) and computes dip-corrected UTC sunset, sunrise, and -15° dark-window crossing times for a given date — Stage 1.
2. (`solsys_code/management/commands/load_telescope_runs.py`) parses classical-schedule run lines and idempotently creates one `tom_calendar.CalendarEvent` per observing night, populated with sunset/sunrise times and the -15° dark window — Stage 2.
3. (`solsys_code/management/commands/sync_lco_observation_calendar.py`) syncs LCO queue ObservationRecords (FTS/MuSCAT4) to the calendar as unified CalendarEvents — starting as a scheduling-window banner and updating in place to the placed block once the LCO scheduler acts — Stage 3.
4. A visual clarity layer (v1.4): `CalendarEventTelescopeLabel` sidecar model records live-verified vs. fallback telescope-label resolution; `calendar_display_extras` template-tag library provides proposal-keyed color (sha256 → 8-color colorblind-vetted palette), status box-shadow rings, and a click-to-filter legend.
5. (`solsys_code/management/commands/sync_gemini_observation_calendar.py`) syncs submitted Gemini ToO `ObservationRecord`s to `CalendarEvent` window banners — using explicit `windowDate`/`windowTime`/`windowDuration` parameters when present and ToO-type-derived defaults (`Rap:` +24 h; `Std:` +24 h to +7 d) when not — with per-record credential scrubbing, idempotent no-churn find-or-create, and a pre-executed demo notebook — Stage 3b (v1.5).
6. (`solsys_code/calendar_utils.py`) shared utility module holding `SITE_TELESCOPE_MAP`, `_extract_instrument`, `insert_or_create_calendar_event()`, and related helpers — extracted from `sync_lco_observation_calendar.py` in v1.6 so all three management commands share one canonical implementation. Calendar event title text is now WCAG-AA-compliant against all palette backgrounds (`text_color_for_bg`), and `CalendarEventTelescopeLabel` data is loaded in a single prefetch query rather than one per event (`fomo_render_calendar` wrapper view, v1.6).
7. (`solsys_code/models.py:CampaignRun`, `campaign_utils.py`, `campaign_tables.py`, `campaign_views.py`, `campaign_gap.py`) a second, separate feature area — **campaign coordination** — added in v2.0: a `CampaignRun` model (linked to a campaign `TargetList`) with the full 3I-sheet field inventory and independent lifecycle/approval status fields; a bootstrap CSV import command validated against the real 3I/ATLAS coordination sheet; a PII-gated per-campaign table (spreadsheet replacement); a public submission form with a staff approval queue whose approved runs project onto the shared calendar (`CAMPAIGN:{pk}` `CalendarEvent`s, reusing `insert_or_create_calendar_event()`); and ephemeris-aware coverage-gap analysis showing observable-but-unclaimed dates. This makes FOMO the community's campaign-coordination hub for rare/urgent objects (e.g. interstellar visitors like 3I/ATLAS), not just a calendar-sync tool.

This is a Stages-1-through-3b-complete implementation of the "telescope runs on the calendar" feature (issue #37), with a calendar visual clarity layer added in v1.4, Gemini ToO calendar sync added in v1.5, shared-utility refactoring + WCAG/N+1 polish added in v1.6, and an ESO/VLT feasibility spike in v1.7. v2.0 added a second, independent feature area — campaign coordination for rare/urgent objects — reusing the calendar-sync infrastructure but serving a distinct community-coordination use case. Stage 4 (full observation-record sync for all facilities) remains future work.

## Current State

**Shipped:**
- ✅ v1.0 "Site/Ephemeris Helper" — 2026-06-14 (Phase 1)
- ✅ v1.1 "Classical Run Ingest" — 2026-06-16 (Phases 2-3)
- ✅ v1.2 "LCO Queue Calendar Sync" — 2026-06-17 (Phase 4)
- ✅ v1.3 "Full LCO Facility Sync" — 2026-06-24 (Phases 5-7, 07.1) — multi-proposal/multi-facility, correct instrument extraction, live telescope-label resolution + facility-aware coarse fallback
- ✅ v1.4 "Calendar Visual Clarity" — 2026-06-26 (Phases 8-9) — `CalendarEventTelescopeLabel` sidecar model, dashed-border + tooltip for fallback labels, proposal-keyed color palette, status box-shadow rings, `[QUEUED]` override fix, click-to-filter legend
- ✅ v1.5 "Gemini Calendar Sync" — 2026-06-27 (Phase 10) — `sync_gemini_observation_calendar` management command syncing Gemini ToO ObservationRecords to CalendarEvent window banners with per-record password scrubbing, ToO-type window derivation, and no-churn idempotency
- ✅ v1.6 "Tech Debt & Display Polish" — 2026-06-29 (Phases 11-12) — `calendar_utils.py` shared utility module with `SITE_TELESCOPE_MAP`/`_extract_instrument`/`insert_or_create_calendar_event` (REFAC-01/02); `text_color_for_bg` WCAG template tag (DISPLAY-08); `fomo_render_calendar` wrapper view eliminating N+1 query (DISPLAY-09)
- ✅ v1.7 "ESO/VLT Calendar Sync — Feasibility Spike" — 2026-07-02 (Phase 13) — investigation-only: live Paranal (VLT) P2 API probe confirmed credentials obtainable/usable (ESO-01), captured real `getOB()`/`getNightExecutions()` shapes (ESO-02), confirmed a headless `FACILITIES['ESO']`-style credential path bypassing `ESOProfile`/session decryption (ESO-03); decision doc recommends **Bypass** — sync straight from `p2api` to `CalendarEvent`, skipping `ObservationRecord` for ESO (ESO-04) — with a future-sync sketch (ESO-05) in `.planning/phases/13-eso-feasibility-spike/13-DECISION.md` and `docs/design/eso_feasibility_spike.rst`. No sync command shipped; La Silla P2 connectivity confirmed reachable via a direct `p2api` bypass of `tom_eso`'s broken `ESOAPI`/`p1api` wrapper, though La-Silla-sourced OB data remains unconfirmed.
- ✅ v2.0 "Campaign Coordination for Rare/Urgent Objects" — 2026-07-05 (Phases 14-17) — `CampaignRun` model + real 3I/ATLAS CSV bootstrap import (Phase 14); PII-gated per-campaign spreadsheet-replacement table (Phase 15); public submission form + staff approval queue with calendar projection (Phase 16); ephemeris-aware coverage-gap analysis (Phase 17). 19/19 v1 requirements shipped. A manual-UAT gap found just before close (pending approval-queue rows showed a blank Site column, and approving an unresolvable free-text site silently fabricated a placeholder `Observatory` row) was fixed via quick task `260705-l1v` prior to shipping.

**Working code:**
- `solsys_code/telescope_runs.py`: `SITES`, `get_site()`, `horizon_dip()`, `sun_event()`, `ParsedRun`, `parse_run_line()`, `KNOWN_STATUSES`
- `solsys_code/management/commands/load_telescope_runs.py`: `load_telescope_runs` BaseCommand
- `solsys_code/calendar_utils.py`: `SITE_TELESCOPE_MAP`, `_extract_instrument` (c_1..c_5 multi-config), `insert_or_create_calendar_event()`, `_coarse_telescope_label()`, and 8 related helpers — shared by all three sync commands
- `solsys_code/management/commands/sync_lco_observation_calendar.py`: `sync_lco_observation_calendar` BaseCommand (multi-proposal/multi-facility)
- `solsys_code/models.py`: `CalendarEventTelescopeLabel` (OneToOneField sidecar on `tom_calendar.CalendarEvent`)
- `solsys_code/migrations/0001_calendareventtelescopelabel.py`: first real solsys_code migration
- `solsys_code/templatetags/calendar_display_extras.py`: `proposal_color`, `status_border_css`, `visible_proposals` template tags; `PROPOSAL_PALETTE`, `NEUTRAL_SLOT_COLOR`, `CLASSICAL_SCHEDULE_LABEL` constants
- `solsys_code/tests/test_telescope_runs.py`: 26 tests
- `solsys_code/tests/test_load_telescope_runs.py`: 6 tests
- `solsys_code/tests/test_sync_lco_observation_calendar.py`: 49 tests (incl. sidecar write, verified/fallback/no-churn)
- `solsys_code/tests/test_calendar_display_extras.py`: 27 tests (ProposalColorTest, StatusBorderCssTest, VisibleProposalsTest, TextColorForBgTest — DISPLAY-08)
- `solsys_code/tests/test_calendar_template.py`: 17 tests (DISPLAY-04/05/06/07 + Phase 8 dashed-border + DISPLAY-08/09 inline color + N+1 regression)
- `solsys_code/calendar_urls.py`: FOMO-local calendar URL conf shadowing `tom_calendar.urls` for `/calendar/` — routes root to `fomo_render_calendar`
- `solsys_code/templatetags/calendar_display_extras.py`: now also `text_color_for_bg`, `_relative_luminance` (WCAG 2.1 formula)
- `solsys_code/views.py`: now also `fomo_render_calendar` (DISPLAY-09 prefetch + Count annotation)
- `docs/notebooks/pre_executed/telescope_runs_demo.ipynb`: Stage 1 demo
- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`: Stage 2 demo
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`: Stage 3 demo (updated through v1.4)
- `solsys_code/management/commands/sync_gemini_observation_calendar.py`: `sync_gemini_observation_calendar` BaseCommand (GEM ToO sync, credential-safe, no-churn)
- `solsys_code/tests/test_sync_gemini_observation_calendar.py`: 15 tests (all 10 GEM-* requirements)
- `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb`: Stage 3b demo (4 D-06 scenarios)
- `solsys_code/models.py`: now also `CampaignRun` (`ApprovalStatus`/`RunStatus` TextChoices) — v2.0 (Phase 14)
- `solsys_code/migrations/0002_campaignrun.py`, `0003_campaignrun_natural_key_unique_constraint.py`: `CampaignRun` table + DB-level natural-key uniqueness — v2.0 (Phase 14)
- `solsys_code/campaign_utils.py`: `resolve_site` (3-tier), `parse_obs_window` (best-effort UT parsing with am/pm handling), `map_observation_status`, `insert_or_create_campaign_run` — v2.0 (Phase 14)
- `solsys_code/management/commands/import_campaign_csv.py`: bootstrap CSV import command (skip-and-log, site resolution, single-target auto-assignment, natural-key collision disambiguation) — v2.0 (Phase 14)
- `solsys_code/tests/test_campaign_models.py`, `test_import_campaign_csv.py`: 39 tests — v2.0 (Phase 14)
- `docs/notebooks/pre_executed/fixtures/campaign_sample.csv`, `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`: synthetic PII-free fixture + paired demo notebook — v2.0 (Phase 14)
- `solsys_code/campaign_tables.py`, `campaign_filters.py`, `campaign_views.py`, `campaign_urls.py`: `CampaignRunTable`, `CampaignRunFilterSet`, `CampaignRunTableView`/`CampaignListView`, `campaigns` URL namespace — v2.0 (Phase 15)
- `src/templates/campaigns/campaign_list.html`, `campaignrun_table.html`; `src/templates/solsys_code/partials/campaign_links.html`, `campaigns_nav_link.html`: read-path + navigation templates — v2.0 (Phase 15)
- `solsys_code/apps.py`: second `target_detail_buttons()` entry + new `nav_items()` hook; `src/templatetags/solsys_code_extras.py`: `campaign_links`/`campaigns_nav_link` inclusion tags — v2.0 (Phase 15)
- `solsys_code/tests/test_campaign_views.py`: 16 tests — v2.0 (Phase 15)
- `solsys_code/mixins.py`: `StaffRequiredMixin`; `solsys_code/campaign_forms.py`: `CampaignRunSubmissionForm` (plain `forms.Form` + honeypot `alt_contact_info`) — v2.0 (Phase 16)
- `solsys_code/campaign_views.py`: now also `CampaignRunSubmissionView`, `ApprovalQueueView`, `CampaignRunDecisionView`, `_notify_staff`; `solsys_code/campaign_tables.py`: now also `ApprovalQueueTable` (`Meta.exclude`/`Meta.sequence` trims 3 structurally-blank columns and leads with `actions` — 16-05 gap closure); `campaign_urls.py`: `submit`/`submission_thanks`/`approval_queue`/`decide` URL names — v2.0 (Phase 16)
- `src/templates/campaigns/campaignrun_submit_form.html`, `submission_thanks.html`, `approval_queue.html`: submission + staff approval-queue templates — v2.0 (Phase 16)
- `solsys_code/tests/test_campaign_forms.py`, `test_campaign_submission.py`, `test_campaign_approval.py`: 40 tests — v2.0 (Phase 16)
- `solsys_code/campaign_gap.py`: `observable_dates`/`claimed_dates`/`get_or_compute_gap` coverage-gap computation core, cached (TTL) set-difference of observable vs. claimed nights — v2.0 (Phase 17)
- `solsys_code/campaign_forms.py`: now also `CampaignGapAnalysisForm`; `campaign_views.py`: now also `CampaignGapAnalysisView`, `gap_analysis_available()`; `campaign_urls.py`: `gap_analysis` URL — v2.0 (Phase 17)
- `src/templates/campaigns/campaignrun_gap_analysis.html`: coverage-gap page; `campaignrun_table.html`: D-14-gated "Show Coverage Gaps" button — v2.0 (Phase 17)
- `solsys_code/campaign_tables.py` (`render_site` fallback), `campaign_utils.py` (`resolve_site` gains keyword-only `create_placeholder`), `campaign_views.py` (approval opts out of placeholder creation): approval-queue site-visibility gap fix — quick task `260705-l1v` (2026-07-05)
- **All 332 `./manage.py test solsys_code` tests pass (v2.0 milestone target features all shipped, plus the pre-close quick-task fix).**
- `solsys_code/models.py` (`CampaignRun.window_start`/`window_end` replace `obs_date`/`ut_start`/`ut_end`, two partial `UniqueConstraint`s), `solsys_code/migrations/0004_campaignrun_window_schema.py` (backfill → dedup → constraint-swap hard-cutover migration), `campaign_gap.py`/`campaign_tables.py`/`campaign_views.py`/`campaign_forms.py`/`import_campaign_csv.py` (all consumers rewritten window-native) — v2.1 (Phase 19). Code review flagged one Critical follow-up (CR-01: the migration's dedup step only covers the TBD-branch collision, not the structurally identical resolved-window collision the backfill also creates); independently confirmed via live-DB query to have caused no actual data loss in this run, but tracked as a real robustness gap — see `.planning/phases/19-window-schema-migration/19-REVIEW.md`, fix via `/gsd-code-review 19 --fix`.
- `solsys_code/campaign_utils.py` (`parse_obs_window()` range/TBD parsing), `solsys_code/campaign_gap.py` (asset-aware `claimed_dates()` — ground windows claim every date, space-mission runs claim none until narrowed to `window_start == window_end`), `solsys_code/management/commands/import_campaign_csv.py` (range/TBD rows import instead of skip) — v2.1 (Phase 20).
- `solsys_code/solsys_code_observatory/utils.py` (`MPCObscodeFetcher.query_all()` bulk fetch), `solsys_code/campaign_utils.py` (`build_site_candidates()`/`fuzzy_match_candidates()` — cached `difflib` fuzzy-match against the live MPC obscode list), `solsys_code/campaign_tables.py` (`ApprovalQueueTable.render_site()` inline datalist + `render_actions()` single-form refactor), `solsys_code/campaign_views.py` (`CampaignRunDecisionView.post()` D-06 clobber guard + `site_selection` resolution, once-per-request candidate pool), `solsys_code/solsys_code_observatory/views.py` (`CreateObservatory` `?obscode=`/`?next=` round-trip), `solsys_code/models.py`/`migrations/0007_campaignrun_contact_public_opt_in.py`/`campaign_forms.py` (`CampaignRun.contact_public_opt_in` opt-in flag + submission-form checkbox + `Case`/`When` queryset-level PII gate) — v2.1 (Phase 21). Deep code review found and fixed 2 critical bugs before phase close: fuzzy-matched name/short_name candidates couldn't actually resolve on approve (the raw display text was never mapped back to its obscode before `resolve_site()`), and the `CreateObservatory` `?next=`/`?obscode=` round-trip was unreachable because the real template dropped both query params — see `21-REVIEW.md`/`21-REVIEW-FIX.md`. This was the last phase of v2.1 — all target features (window-schema migration, range/TBD import, asset-aware coverage gap, site-disambiguation UI, contact opt-in) are now shipped.
- **All 417 `./manage.py test solsys_code` tests pass.**

## Core Value

Stage 1 (v1.0): Sun-event times accurate to within 2 minutes of the Las Campanas skycalc reference tool — the foundation that Stages 2-4 build on. Also: validated the GSD discuss→plan→execute→verify loop end-to-end on this codebase.

Stage 2 (v1.1): A `load_telescope_runs` management command turns classical-schedule run lines into accurate, idempotent `tom_calendar.CalendarEvent`s — one per observing night — using Stage 1's `SITES`/`get_site()`/`sun_event()` for sunset/sunrise times.

Stage 3 (v1.2): A `sync_lco_observation_calendar` management command syncs LCO queue ObservationRecords (FTS/MuSCAT4) to the calendar — one CalendarEvent per record, keyed on the LCO portal URL, transitioning from a scheduling-window banner (`parameters['start'`/`'end']`) to a placed block (`scheduled_start`/`scheduled_end`) as the scheduler acts, and updating in place if the block is rescheduled.

v2.0 (Campaign Coordination): When the next 4I-class object appears, FOMO replaces the ad-hoc Google Sheet as the community's campaign-coordination hub — target-linked observing runs, submission with oversight, and a per-object campaign view. This is now co-equal with the calendar-sync value above, not a extension of it — the two feature areas share infrastructure (`insert_or_create_calendar_event()`, `Observatory`) but serve distinct use cases (routine facility sync vs. ad-hoc community coordination for rare objects).

## Current Milestone: v2.1 Uncertain Scheduling & Site Disambiguation

**Goal:** Campaign coordination handles the real 3I/ATLAS sheet's harder rows — space-mission observations (e.g. the Carrie Holt/Martin Cordiner JWST rows) whose exact observing night isn't known yet, only a window or a still-pending schedule — while also closing out the two loose ends v2.0 shipped with: submitter contact opt-in (VIEW-05) and a real staff-facing site-disambiguation UI (the natural next step after quick task `260705-l1v`'s visibility fix).

**Target features:**
- **Range-first `CampaignRun` scheduling** — replace the single `obs_date`/`ut_start`/`ut_end` fields with a window representation (a single classically-scheduled night becomes a 1-day window); this is a real schema migration touching every existing row, not an additive bolt-on field.
- **Ground vs. space-mission asset distinction** — reuse the existing `Observatory.observations_type` (`SATELLITE_OBSTYPE`) rather than adding a new field to `CampaignRun`; a run's "is this a space mission" status is derived from its resolved `site`.
- **CSV import handles range/TBD dates** — `import_campaign_csv`'s `parse_obs_window()` currently requires an exact `YYYY-MM-DD` and skips any row that doesn't parse (a true natural-key failure per D-05) — a range like "Aug 1-15" or a "TBD pending Cycle 2" cell must import into the new window representation instead of being silently dropped.
- **Coverage-gap analysis is asset-aware** — a ground-based run's window claims every date within it (conservative, avoids double-booking); a space-mission run claims nothing until its schedule narrows to something concrete (exact narrowing/refinement trigger is a spike question).
- **Site-disambiguation UI in the approval queue** — the Site column becomes an inline dropdown of fuzzy-matched `Observatory` candidates (matched against name/short_name/old_names when the typed text doesn't resolve via `resolve_site()`'s existing tier 1/2), plus a free-text fallback to resolve-to-existing or explicitly create a new `Observatory`. Never auto-fabricates a placeholder (consistent with quick task `260705-l1v`) — an unresolved site with no good candidate just stays unresolved until a human acts.
- **VIEW-05** — a single combined opt-in flag on the submission form (default opt-out); when set, `contact_person`/`contact_email` become visible to anonymous visitors on the per-campaign table, same as staff already see.
- **Phase-time investigation spike** — settles the exact window schema, the range/TBD parsing rules, and the fuzzy-match approach against real 3I sheet rows before implementation lands, in the same milestone (not deferred).

**Key context:**
- **Correction (operator-caught, post-research):** real space telescopes have standard, short MPC obscodes — 250 = Hubble, 274 = JWST, 289 = Nancy Grace Roman (all 3 characters, per the official MPC Observatory Codes list). The `'500@-170'` string in `resolve_site()`'s docstring/comments (and repeated throughout this milestone's research) is JPL Horizons/SPICE observer notation (NAIF SPK ID), **not an MPC obscode** — `Observatory.obscode`'s `CharField(max_length=4)` very likely does NOT need widening. The spike should confirm `resolve_site()` resolves these real codes correctly, treating "does obscode length need to change" as a spike question with a default answer of no, not a presumed blocker.
- The `CampaignRun` natural-key `UniqueConstraint` is `(campaign, telescope_instrument, ut_start)` — a run with no fixed start time (TBD scheduling) breaks this key's assumption; the spike must decide the replacement natural key for window/TBD rows.
- This directly extends quick task `260705-l1v` (approval-queue site-visibility fix, 2026-07-05): that fix stopped auto-fabrication and surfaced the raw typed text; this milestone adds the actual resolution UI staff need to act on it.

## v2.0 Campaign Coordination for Rare/Urgent Objects — SHIPPED 2026-07-05

**Goal:** When the next 4I-class object appears, FOMO replaces the ad-hoc Google Sheet as the community's campaign-coordination hub — target-linked observing runs, submission with oversight, and a per-object campaign view.

**Target features:**
- Campaign-run data model + 3I bootstrap import — target-linked run records with lifecycle status (planned → observed → reduced → published), filters/bandpass, outcome, publication plans, collaboration flag, contact info (PII-guarded); validated by a one-off CSV import of the real 3I/ATLAS coordination sheet
- Per-target campaign table view — the spreadsheet-replacement display: all campaign runs for an object in one table, linked from the target page / calendar
- Submission form + approval queue — community-facing intake (PIs + external observers) with admin review before runs go public (per seed `target-linked-run-submission-form.md`)
- Coverage-gap analysis — ephemeris-aware view of observable-but-unclaimed dates; FOMO's differentiator over any spreadsheet; scoped last so it can defer to v2.1 if needed

**Key context:**
- Reference model: the real 3I/ATLAS campaign Google Sheet (field inventory captured in the enriched seed)
- Contact emails stored but auth-gated for display (FOMO is `OPEN` targets / `AUTH_STRATEGY='READ_ONLY'`) — exact policy settled during phase discussion
- ESO seeds (SEED-001/SEED-002) do not match this scope and stay dormant; ESO-10/ESO-11 remain deferred

**Phase 14 — Campaign Data Model & Bootstrap Import — COMPLETE (2026-07-03):**
- ✅ CAMP-01: `CampaignRun` model with the full 3I-sheet field inventory, required `campaign` (`TargetList`) FK
- ✅ CAMP-02: Optional `target` FK; single-target campaigns auto-assign their one `Target` to every imported row
- ✅ CAMP-03: Lifecycle + approval status as two independent controlled-vocabulary fields (`approval_status`: 3 values, `run_status`: 8 values) per D-02 — not a single flat vocabulary (ROADMAP/REQUIREMENTS wording corrected to match)
- ✅ CAMP-04: `import_campaign_csv` bootstrap-import command — skip-and-log natural-key handling, 3-tier site resolution (D-08), created/updated/unchanged/skipped/site_needs_review summary
- ✅ CAMP-05: Paired demo notebook + synthetic PII-free fixture, executed with committed output, demonstrates the approval lifecycle
- Post-plan deep code review found and fixed 2 critical data-correctness bugs before phase close: PM/AM UT-time markers were parsed but never applied (silently 12h wrong), and the `(campaign, telescope_instrument, ut_start)` natural key collided for distinct rows sharing an unparseable UT Time Range (silent row loss). Both fixed with regression tests, plus 9 warning-level hardening fixes (network timeouts, race protection, PII-safe logging, DB-level uniqueness constraint, status-mapping negation awareness, CSV header validation) — see `14-REVIEW.md`/`14-REVIEW-FIX.md`.
- 242 `./manage.py test solsys_code` tests pass; `ruff check .`/`ruff format --check .` clean on all Phase 14 files.

**Phase 15 — Per-Campaign Table View (Read Path) — COMPLETE (2026-07-03):**
- ✅ VIEW-01: `CampaignRunTableView` (django-tables2 `SingleTableMixin` + django-filter `FilterView`) — sortable/paginated (25/page, default `-obs_date`) table of every `CampaignRun` for a campaign
- ✅ VIEW-02: `SolsysCodeConfig.target_detail_buttons()` second entry links a target-detail page to each campaign it belongs to (via `TargetList` membership); new `SolsysCodeConfig.nav_items()` hook adds a global "Campaigns" navbar entry (first `nav_items()` consumer in FOMO)
- ✅ VIEW-03: `get_queryset()` returns a `.values(*ALLOWED_FIELDS_FOR_NON_STAFF)`-restricted queryset for non-staff so `contact_person`/`contact_email` are never fetched by SQL for anonymous requests, proven by an anonymous-client test; visible to staff
- ✅ VIEW-04: `CampaignRunFilterSet` — `run_status` multi-select (OR semantics), `open_to_collaboration` boolean filter
- Verification independently re-ran the full test suite (`manage.py test solsys_code` — 258/258 pass) and the phase-specific module (16/16), confirmed PII-gating at the SQL level (not template-only), and cross-checked every commit hash — see `15-VERIFICATION.md`. Code review found no critical issues (2 warnings/3 info, non-blocking) — see `15-REVIEW.md`.
- 258 `./manage.py test solsys_code` tests pass; `ruff check .`/`ruff format --check .` clean on all Phase 15 files.

**Phase 16 — Submission Form, Approval Queue & Calendar Projection (Write Path) — COMPLETE (2026-07-04):**
- ✅ SUBMIT-01: `CampaignRunSubmissionView` (public `FormView`) creates a `PENDING_REVIEW` `CampaignRun` from a validated submission; discoverable via "Submit a Run" entry buttons on the campaigns list and per-campaign table pages
- ✅ SUBMIT-02: Non-staff visitors see approved and rejected rows on the per-campaign table; only `pending_review` is hidden (D-09)
- ✅ SUBMIT-03: `CampaignRunDecisionView` performs an atomic conditional `.update()` keyed on `(pk, approval_status=PENDING_REVIEW)`; a double-approve/-reject is a proven no-op
- ✅ SUBMIT-04: Hidden honeypot field `alt_contact_info` — a tripped submission creates no `CampaignRun`, sends no email, and returns the same thanks page as a genuine submission
- ✅ SUBMIT-05: Genuine submissions email every `is_staff` user with a non-empty email; subject/body contain no PII
- ✅ CAL-01/CAL-02: Approving a run with telescope + `ut_start`/`ut_end` creates/updates a `CalendarEvent` keyed `CAMPAIGN:{pk}` via `insert_or_create_calendar_event`, `target_list` set to the campaign
- ✅ CAL-03: Re-approving an already-approved run creates no duplicate event and no `modified` churn
- Post-plan deep code review found and fixed one critical data-consistency bug before phase close: the calendar-projection side effects (site resolution + `CalendarEvent` write) ran outside the atomic status-update transaction, so a mid-flow failure left a run permanently stuck `APPROVED` with no calendar event and no recovery path (the double-approve guard made it unrecoverable via the UI). Fixed by reverting `approval_status` back to `PENDING_REVIEW` on any post-update failure, with logging. Two further warnings fixed (misleading message for a nonexistent `pk`; dead `NoReverseMatch` fallback). One warning — the public form's `campaign` field listing every `TargetList`, not just active campaigns — was deliberately left open as a product-scope decision (would require a schema change to distinguish "campaign" `TargetList`s and break the bootstrap-new-campaign submission path) — see `16-REVIEW.md`/`16-REVIEW-FIX.md`.
- Verification independently re-ran the phase's 4 test modules (58/58), re-verified the fix chain by reproducing the original failure mode against a live test DB, and confirmed requirement traceability for all 8 IDs — see `16-VERIFICATION.md`.
- 300 `./manage.py test solsys_code` tests pass; `ruff check .`/`ruff format --check .` clean on all Phase 16 files.

**Phase 17 — Coverage-Gap Analysis (Deferrable to v2.1) — COMPLETE (2026-07-05):**
- ✅ GAP-01: Dark-window-only observability decision (`17-GAP-01-DECISION.md`) — reuses `telescope_runs.sun_event()`, no true altitude/airmass filtering, no module-scope dependency on the heavy SPICE-loading ephemeris module
- ✅ GAP-02: `campaign_gap.py` composes `sun_event()` (observable dates) with `CampaignRun` queries (claimed dates) into a cached set-difference; `CampaignGapAnalysisView` exposes it as a public, GET-triggered, server-validated endpoint with a campaign-scoped selection form and a `campaignrun_gap_analysis.html` page matching the UI-SPEC contract; a D-14-gated "Show Coverage Gaps" button was added to the campaign table page
- Post-plan deep code review found and fixed 2 critical bugs before phase close: a non-numeric `target`/`site` GET param crashed the view with an unhandled 500 instead of the documented 400, and `claimed_dates()` crashed with an unhandled `ValueError` for any site with a blank `timezone` (reachable via normal production data, since auto-created `Observatory` rows from `resolve_site()` never set one). Both fixed with regression-safe guards; 5 further warnings fixed (PII-safe `.only()` queryset for the public cached view, date-range floor, form-based `end_date` parsing instead of hand-parsed GET data, a redundant-query TOCTOU cleanup, and a documented range-scope invariant) — see `17-REVIEW.md`/`17-REVIEW-FIX.md`.
- Verification independently re-ran the full test suite, and re-reproduced both critical-bug scenarios directly against the fixed code (confirmed 400 instead of 500, and 200 instead of 500) rather than trusting the fix report's narrative — see `17-VERIFICATION.md`. Human-verify checkpoint (page + button visual/interaction review) approved with no issues.
- 326 `./manage.py test solsys_code` tests pass; `ruff check .`/`ruff format --check .` clean on all Phase 17 files.
- This was the last phase of v2.0 — all target features (data model, table view, submission/approval, coverage-gap analysis) are now shipped.

**Prior milestones (v1.0-v1.7):**

**v1.7 ESO/VLT Calendar Sync — Feasibility Spike — COMPLETE (2026-07-02):**
- ✅ ESO-01/02/03: real Paranal P2 API investigation confirmed credentials obtainable/usable, captured verbatim `getOB()`/`getNightExecutions()` response shapes, confirmed a viable headless `FACILITIES['ESO']`-style credential path
- ✅ ESO-04: decision doc recommends **Bypass** (sync straight from `p2api` to `CalendarEvent`, skipping `ObservationRecord` for ESO), rationale tied directly to ESO-01/02/03
- ✅ ESO-05: future-sync sketch (synthetic `ESO:{p2_environment}/{obId}` key, banner-only vs. status-aware options, 12-code `obStatus` vocabulary) scoped as input to a future implementation milestone
- No sync command shipped (investigation-only, per milestone scope); `eso_p2_probe.py` was a throwaway, git-excluded, never-committed script (D-09)
- Bonus finding: La Silla P2 connectivity is reachable via a direct `p2api.ApiConnection('production_lasilla', ...)` bypass of `tom_eso`'s `ESOAPI`/`p1api` wrapper (which unconditionally — and incorrectly for La Silla — requires a Phase-1 connection); La-Silla-sourced OB data itself remains unconfirmed (the one live test returned a Paranal-instrument run)

**v1.6 Tech Debt & Display Polish — COMPLETE (2026-06-29):**
- ✅ REFAC-01/02: `calendar_utils.py` created; all three commands use `insert_or_create_calendar_event()`; "upsert" jargon removed from docs
- ✅ DISPLAY-08: `text_color_for_bg` WCAG 2.1 relative-luminance template tag; all 8 palette entries + NEUTRAL_SLOT_COLOR return `#fff`; `#ffffff` returns `#000`; proven by `TextColorForBgTest`
- ✅ DISPLAY-09: `fomo_render_calendar` wrapper view with `prefetch_related('telescope_label_meta')` + `Count('todos', filter=Q(is_completed=False))` annotation; `/calendar/` URL shadows upstream; N+1 regression test via `CaptureQueriesContext` green
- 194 `./manage.py test solsys_code` tests pass; `ruff` clean

## Requirements

### Validated

- ✓ `Observatory` model stores MPC-obscode-keyed site `lat`/`lon`/`altitude` with geodetic/geocentric conversion helpers — existing
- ✓ `tom_calendar.models.CalendarEvent` has the fields needed to represent a telescope run — existing
- ✓ `tom_observations.models.ObservationRecord` carries `scheduled_start`/`scheduled_end`/status for real observation blocks — existing, used by Stage 4
- ✓ `get_site()` resolves a telescope name to an `Observatory` record + `EarthLocation` + timezone — v1.0 (SITE-01, SITE-02)
- ✓ `sun_event(site, date, kind)` returns UTC sunset, sunrise, and -15° dark crossings with dip correction — v1.0 (EPHEM-01, EPHEM-02)
- ✓ Horizon-dip helper returns 1.44° ± 0.02° at 2402 m — v1.0 (EPHEM-03)
- ✓ Las Campanas sunset/sunrise for June 2026 agree with skycalc to within 2 minutes — v1.0 (EPHEM-04)
- ✓ Astronomical twilight (-18°) for Las Campanas on 10 June 2026 agrees with skycalc to within 2 minutes — v1.0 (EPHEM-05)
- ✓ `America/Santiago` / `Australia/Sydney` DST offsets correct — v1.0 (EPHEM-06)
- ✓ Observatory records exist for Magellan (Las Campanas), NTT (La Silla), FTS (Siding Spring) — v1.0 (SITE-03)
- ✓ **PARSE-01**: Parse classical run line into `ParsedRun(telescope, instrument, status, year, month, day1, day2)`, both date-range orderings — v1.1 (Phase 2)
- ✓ **PARSE-02**: Hyphenated instrument names parse as single token — v1.1 (Phase 2)
- ✓ **PARSE-03**: No-year defaults to current year; late-December rolls to next year — v1.1 (Phase 2)
- ✓ **INGEST-01**: `load_telescope_runs` expands `S..E` into `E - S + 1` nightly CalendarEvents (`start_time = sunset(d)`, `end_time = sunrise(d+1)`) — v1.1 (Phase 3)
- ✓ **INGEST-02**: Each event sets `telescope`/`instrument`/`title` and `description` with -15° dark window, status, and original run line text — v1.1 (Phase 3)
- ✓ **INGEST-03**: Running the command twice on the same file creates no duplicate CalendarEvents — v1.1 (Phase 3)
- ✓ **SELECT-01**: `sync_lco_observation_calendar --proposal <code>` syncs all `ObservationRecord(facility='LCO')` matching `parameters['proposal']` — v1.2 (Phase 4)
- ✓ **SYNC-01**: One `CalendarEvent` per matching record, keyed on `url` = `LCOFacility().get_observation_url(observation_id)` — v1.2 (Phase 4)
- ✓ **SYNC-02**: When `scheduled_start` is `None`, event times come from `parameters['start']`/`parameters['end']` (window banner); title is `[QUEUED]`-prefixed — v1.2 (Phase 4)
- ✓ **SYNC-03**: When `scheduled_start`/`scheduled_end` are populated, event times are set from those values (placed block replaces banner) — v1.2 (Phase 4)
- ✓ **SYNC-04**: Re-running after rescheduling updates the existing event in place, no duplicates, no `modified` churn on unchanged records — v1.2 (Phase 4)
- ✓ **SYNC-05**: `telescope`, `instrument`, `proposal` on `CalendarEvent` are populated from the record — v1.2 (Phase 4)
- ✓ **TERM-01**: Terminal-failure states (WINDOW_EXPIRED/CANCELED/FAILURE_LIMIT_REACHED/NOT_ATTEMPTED) get a `[EXPIRED]`/`[CANCELLED]`/`[FAILED]` title prefix; event is retained; `COMPLETED` gets a clean title — v1.2 (Phase 4)
- ✓ **SELECT-02**: `--proposal A,B,C` syncs records matching exactly A/B/C with no substring leakage (e.g. no match on `AB`) — v1.3 (Phase 5)
- ✓ **SELECT-03**: `--proposal ALL` (any casing) syncs every LCO + SOAR record regardless of proposal — v1.3 (Phase 5)
- ✓ **SELECT-04**: A single run produces correct CalendarEvents for both LCO and SOAR records, dispatched via an eager `{'LCO': LCOFacility(), 'SOAR': SOARFacility()}` dict — v1.3 (Phase 5)
- ✓ **SELECT-05**: SOAR records are dispatched through a `SOARFacility` instance, never a reused `LCOFacility` instance, proven by a discriminating spy test — v1.3 (Phase 5)
- ✓ **EXTRACT-01**: Instrument type is extracted by scanning `c_1_instrument_type`..`c_5_instrument_type` for the configuration with a populated exposure time, replacing the v1.2 flat-key assumption that doesn't exist in real data — v1.3 (Phase 6)
- ✓ **EXTRACT-02**: Extraction is verified against SOAR's multi-configuration shape (spectrum/arc/lamp-flat) and LCO MUSCAT's per-channel exposure-key shape, never mistaking a calibration/non-science config for the meaningful one — v1.3 (Phase 6)
- ✓ **TELESCOPE-01**: Verified static site/telescope mapping dict, keyed on `siteid-enclid-telid`, covers all real LCO-network sites (replaces the 2-site `[ASSUMED]` `SITE_TELESCOPE_MAP`) — v1.3 (Phase 7; coj/ogg gaps found in UAT fixed via quick task 260623-su3)
- ✓ **TELESCOPE-02**: Per-record LCO API call resolves the actual site/enclosure/telescope and maps it through the verified dict — v1.3 (Phase 7)
- ✓ **TELESCOPE-03**: A failed/timed-out/unmapped per-record API call falls back to a coarse instrument-class label (`1m0`/`0m4`/`2m0`/`4m0`) instead of skipping the record, for both LCO and SOAR facilities — v1.3 (Phase 7; SOAR was facility-unaware until Phase 07.1 closed the v1.3 milestone-audit gap)
- ✓ **TELESCOPE-04**: A fallback-labeled event is distinguishable from a verified-label event via a clean `[UNVERIFIED] <coarse-label> <instrument>` title, for both LCO and SOAR — v1.3 (Phase 7; SOAR's doubled raw-instrument title fixed in Phase 07.1)
- ✓ **SYNC-06**: Per-record telescope-API failures are tracked as a distinct `telescope_api_failed` counter, separate from `skipped`, for both LCO and SOAR — v1.3 (Phase 7; SOAR zero-coverage gap closed in Phase 07.1)
- ✓ **SYNC-07**: A per-record API failure does not abort the run or skip the record — the record still gets a `CalendarEvent` (fallback-labeled), and the rest of the batch continues — v1.3 (Phase 7)
- ✓ **SYNC-08**: The per-record API call uses an explicit timeout and a single attempt (no retry/backoff loop) — v1.3 (Phase 7)
- ✓ **SYNC-09**: Error/exception output from a failed API call never includes raw response body or credential content — v1.3 (Phase 7)
- ✓ **DISPLAY-01**: `CalendarEventTelescopeLabel` sidecar model (OneToOneField PK on `tom_calendar.CalendarEvent`) records live-verified vs. fallback telescope-label outcome; `sync_lco_observation_calendar` writes it via `update_or_create`; classical-schedule events have no row (template treats missing row as "verified") — v1.4 (Phase 8)
- ✓ **DISPLAY-02**: Dashed-border + native-tooltip visual cue in `calendar.html` distinguishes fallback-labeled events from verified ones, discoverable without reading title text — v1.4 (Phase 8)
- ✓ **DISPLAY-03**: Hovering a fallback-labeled event shows a tooltip with verification detail — v1.4 (Phase 8)
- ✓ **DISPLAY-04**: `CalendarEvent` color hashed deterministically from normalized proposal into a curated 8-color colorblind-vetted palette; same proposal renders identically across telescopes, restarts, and htmx re-renders; empty proposal gets dedicated neutral slot — v1.4 (Phase 9)
- ✓ **DISPLAY-05**: `[QUEUED]` template override that discarded proposal color with flat grey removed; queued events retain proposal-keyed background — v1.4 (Phase 9)
- ✓ **DISPLAY-06**: Status box-shadow rings (queued 2px, terminal-failure 3px) layered orthogonally on top of proposal color, composed with Phase 8 dashed border without collision — v1.4 (Phase 9)
- ✓ **DISPLAY-07**: Footer legend maps proposal codes to rendered colors with collision grouping; click-to-filter JS IIFE toggles highlight/dim on the calendar grid client-side, survives htmx month swaps — v1.4 (Phase 9)
- ✓ **GEM-SELECT-01**: `sync_gemini_observation_calendar` syncs all `ObservationRecord(facility='GEM')` records — v1.5 (Phase 10)
- ✓ **GEM-WINDOW-01**: Each synced record becomes one `CalendarEvent`; window from `windowDate`/`windowTime`/`windowDuration` when present — v1.5 (Phase 10)
- ✓ **GEM-WINDOW-02**: Records without explicit window fall back to ToO-type-derived window anchored on `ObservationRecord.created` (`Rap:` → +24 h, `Std:` → +24 h to +7 d); neither → skip with counter — v1.5 (Phase 10)
- ✓ **GEM-KEY-01**: Idempotency key (`CalendarEvent.url`) constructed as `GEM:{prog}/{observation_id}` — v1.5 (Phase 10)
- ✓ **GEM-TELE-01**: `telescope` derived from program prefix (`GS-*` → `Gemini South`, `GN-*` → `Gemini North`) — v1.5 (Phase 10)
- ✓ **GEM-INSTR-01**: `instrument` from settings description (strip `Std:`/`Rap:` prefix), fallback to obs code — v1.5 (Phase 10)
- ✓ **GEM-PROP-01**: `proposal` set from `params['prog']` — v1.5 (Phase 10)
- ✓ **GEM-STATUS-01**: `[ON_HOLD]` title prefix when `ready=false`; clean title otherwise — v1.5 (Phase 10)
- ✓ **GEM-NOCHURN-01**: Re-running creates no duplicates, no `modified` churn on unchanged records — v1.5 (Phase 10)
- ✓ **GEM-SECURE-01**: `password` field never logged or exposed during sync — v1.5 (Phase 10)
- ✓ Extract `SITE_TELESCOPE_MAP` + `_extract_instrument` into `solsys_code/calendar_utils.py`; `insert_or_create_calendar_event()` helper extracted and used by all three sync commands — v1.6 (Phase 11)
- ✓ **DISPLAY-08**: WCAG 2.1 relative-luminance `text_color_for_bg` template tag; all 8 `PROPOSAL_PALETTE` entries + `NEUTRAL_SLOT_COLOR` return `#fff`; `#ffffff` → `#000` — v1.6 (Phase 12)
- ✓ **DISPLAY-09**: `fomo_render_calendar` wrapper view with `prefetch_related('telescope_label_meta')` + `Count` annotation; N+1 regression test via `CaptureQueriesContext` green — v1.6 (Phase 12)
- ✓ **ESO-01**: Confirmed valid ESO P2 API production credentials (Paranal/VLT) are obtainable and usable, with connection evidence — v1.7 (Phase 13)
- ✓ **ESO-02**: Captured real `getOB()`/`getNightExecutions()` response shapes verbatim (`obStatus='P'` and `'M'` cases), redacted per D-04 — v1.7 (Phase 13)
- ✓ **ESO-03**: Confirmed a viable headless credential-sourcing path (direct `ESOAPI(...)` construction from env-var-supplied credentials, bypassing `ESOProfile`/session decryption) — v1.7 (Phase 13)
- ✓ **ESO-04**: Decision doc recommends exactly one option — **Bypass** — with rationale tied directly to ESO-01/02/03 — v1.7 (Phase 13)
- ✓ **ESO-05**: Future-sync sketch (synthetic key, banner-only vs. status-aware, 12-code `obStatus` vocabulary) scoped as input to a future milestone — v1.7 (Phase 13)
- ✓ **CAMP-01**: `CampaignRun` model stores an observing run linked to a campaign `TargetList` (required FK) with the full 3I-sheet field inventory — v2.0 (Phase 14)
- ✓ **CAMP-02**: A `CampaignRun` can optionally record the specific `Target` observed; single-target campaigns auto-assign it without setting it manually — v2.0 (Phase 14)
- ✓ **CAMP-03**: Lifecycle status (planned → observed → reduced → published) plus approval state (pending review / approved / rejected) as two independent controlled-vocabulary fields (D-02) — v2.0 (Phase 14)
- ✓ **CAMP-04**: Operator can bootstrap-import the real 3I/ATLAS sheet CSV via a management command with per-row skip-and-log error handling and a created/updated/skipped summary — v2.0 (Phase 14)
- ✓ **CAMP-05**: The import command's paired demo notebook contains no real PII — runs against a synthetic/redacted fixture — v2.0 (Phase 14)
- ✓ **VIEW-01**: User can view a per-campaign table of all its runs (sortable/paginated), replacing the spreadsheet — v2.0 (Phase 15)
- ✓ **VIEW-02**: User can reach a target's campaigns from its target-detail page; navbar exposes a campaigns entry — v2.0 (Phase 15)
- ✓ **VIEW-03**: Contact person/email are visible only to authenticated staff — excluded from view context for anonymous requests and proven by an anonymous-client test — v2.0 (Phase 15)
- ✓ **VIEW-04**: User can filter the table by lifecycle status and the open-to-collaboration flag — v2.0 (Phase 15)
- ✓ **SUBMIT-01**: Community member can submit a run via a public form, discoverable from the campaigns list and per-campaign table pages — v2.0 (Phase 16)
- ✓ **SUBMIT-02**: Non-staff visitors see approved and rejected rows on the per-campaign table; only `pending_review` is hidden — v2.0 (Phase 16)
- ✓ **SUBMIT-03**: Staff approve/reject via an atomic conditional update; a double-decision is a proven no-op — v2.0 (Phase 16)
- ✓ **SUBMIT-04**: Hidden honeypot field silently absorbs spam submissions (no create, no email, same thanks page) — v2.0 (Phase 16)
- ✓ **SUBMIT-05**: A genuine submission emails every staff user with a non-empty email; no PII in subject/body — v2.0 (Phase 16)
- ✓ **CAL-01/CAL-02**: Approving a run with telescope + start/end time creates/updates a `CalendarEvent` via `insert_or_create_calendar_event`, keyed `CAMPAIGN:{pk}` — v2.0 (Phase 16)
- ✓ **CAL-03**: Re-approving an already-approved run creates no duplicate event and no `modified` churn — v2.0 (Phase 16)
- ✓ **GAP-01**: Coverage-gap analysis observability approach decided (dark-window-only, not true altitude/airmass filtering) with rationale citing pre-milestone research — v2.0 (Phase 17)
- ✓ **GAP-02**: User can see which observable nights for a campaign target and site are not yet claimed by any run, via a GET-triggered, cached, server-validated view — v2.0 (Phase 17)
- ✓ **SCHED-01**: Phase-time investigation spike settled all five scheduling-design decisions against the real 3I/ATLAS sheet rows: window schema confirmed as the nullable `window_start`/`window_end` `DateField` pair; TBD natural key folds `contact_person` into a partial/conditional `UniqueConstraint`; CSV range/TBD parsing rules enumerated per real cell shape (extends `_HHMM_RANGE`/`_APPROX_HOUR`/`_BARE_HOUR_UTC`); fuzzy-match library split verdict is **difflib primary** (no live-test evidence favored `rapidfuzz`'s extra dependency); `Observatory.obscode` needs no widening (250/274/289 all fit `max_length=4`) — but the spike also surfaced a real, previously-unknown bug: `resolve_site()` currently **cannot** resolve any of the three standard space-observatory codes because `MPCObscodeFetcher.to_observatory()` raises an unguarded `TypeError` on the MPC API's `null` longitude for satellite-type records (flagged for Phase 19/21, not fixed here — investigation-only phase). See `.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md` and `docs/design/uncertain_scheduling_spike.rst` — v2.1 (Phase 18)
- ✓ **SCHED-02**: `CampaignRun`'s `obs_date`/`ut_start`/`ut_end` single-night representation replaced by a nullable `window_start`/`window_end` pair; a classically-scheduled single night is `window_start == window_end` — v2.1 (Phase 19)
- ✓ **SCHED-03**: A `CampaignRun` can be saved in a "TBD" state (both window fields null), distinct from a resolved window — v2.1 (Phase 19)
- ✓ **SCHED-04**: Two distinct TBD rows for the same campaign + telescope neither silently merge nor duplicate — closed via a partial `UniqueConstraint` on `(campaign, telescope_instrument, contact_person)` scoped to `window_start IS NULL` — v2.1 (Phase 19)
- ✓ **SCHED-05**: Every existing `CampaignRun` row migrated with no data loss (`window_start == window_end == former obs_date`) — migration `0004_campaignrun_window_schema.py` combines backfill → dedup → constraint-swap in load-bearing order. Code review's CR-01 (dedup asymmetry between TBD and resolved-window branches) was independently confirmed to have caused no actual data loss in this migration run, but remains a real robustness gap for richer future datasets — v2.1 (Phase 19)
- ✓ **ASSET-01**: A `CampaignRun`'s ground vs. space-mission classification is derived from its resolved site's `Observatory.observations_type` (`SATELLITE_OBSTYPE`) — no new field on `CampaignRun` — v2.1 (Phase 20)
- ✓ **ASSET-02**: Coverage-gap analysis claims every date in a ground-based run's window; a space-mission run claims no dates until its window narrows to a single concrete night — v2.1 (Phase 20)
- ✓ **IMPORT-01**: `import_campaign_csv`/`parse_obs_window` accepts a date range or a TBD-style free-text cell and imports the row into the window representation, instead of raising and skipping it as a natural-key failure — v2.1 (Phase 20)
- ✓ **IMPORT-02**: A row whose `Obs. Date` text still can't be parsed gets a "needs review" flag and is included in the import summary, never silently dropped — v2.1 (Phase 20)
- ✓ **SITE-01**: When a submitted `site_raw` doesn't resolve via `resolve_site()`'s tier 1/tier 2, the approval queue's Site column presents a dropdown of fuzzy-matched `Observatory` candidates for staff to pick from — v2.1 (Phase 21)
- ✓ **SITE-02**: Staff can type a code directly and resolve it to an existing `Observatory` or explicitly create a new one; no placeholder `Observatory` is ever auto-fabricated — v2.1 (Phase 21)
- ✓ **SITE-03**: Approving a run whose site a staff member already manually resolved is never silently re-resolved/overwritten (fixes the `CampaignRunDecisionView` clobbering bug) — v2.1 (Phase 21)
- ✓ **VIEW-05**: Submitter can opt in (single combined flag, default opt-out) to public display of `contact_person`/`contact_email` on the per-campaign table; unset behaves exactly as today (staff-only) — v2.1 (Phase 21)

### Active

None — all v2.1 target features shipped (Phases 18-21). Milestone close via `/gsd-complete-milestone` is the next step, not yet run.

Not yet committed to this milestone (still candidates for a future one): Stage 4 full observation-record sync for all facilities; ESO-10/ESO-11 (`sync_eso_observation_calendar` + paired notebook, unblocked by Phase 13's Bypass verdict); SUBMIT-06/07 (trusted-PI self-approval, submission status lookup) — see `.planning/STATE.md` Deferred Items and `.planning/milestones/v2.0-REQUIREMENTS.md` v2 Requirements for full detail.

### Out of Scope

- Gemini facility support — different base class (`BaseRoboticObservationFacility`), stub `get_observation_url()` (no portal URL to key the idempotent sync on), different parameter keys and terminal-states vocabulary than LCO
- ESO/NTT *classical* scheduling — already handled by Stage 2 (`load_telescope_runs`); never goes through `ObservationRecord`/queue sync. (ESO/VLT *ObservationRecord*/queue sync is now in scope for v1.7 — see Current Milestone above.)
- Reworking `tom_observations`' existing astroplan-based visibility/airmass plots — separate, not touched by this feature
- Distinguishing Magellan Baade vs Clay in `telescope` field — open item; bare `'Magellan'` is deliberately ambiguous (both at Las Campanas, same ephemeris)
- Replacing `SITES`'s hardcoded telescope-name → obscode mapping with a data-driven `Observatory.short_name` lookup — Stage 2+ consideration, not required for Stage 2 success criteria

## Context

- **Codebase**: FOMO (Django + TOM Toolkit), Solar System follow-up coordination.
- **Design doc**: `docs/design/telescope_runs_calendar.rst` — full feasibility study and 4-stage implementation plan for issue #37.
- **Experiment doc**: `docs/design/gsd_experiment.rst` — rationale for using this feature as a GSD trial.
- **Site coordinate sourcing**: coordinates come from `Observatory` records (by MPC obscode), not hardcoded constants.
- **Two-test-suite split**: `solsys_code/` Django app tests run via `./manage.py test solsys_code`; pure-Python helpers under `tests/` run via `python -m pytest`.
- **SPICE kernel side effect**: `telescope_runs.py` avoids importing `solsys_code.ephem_utils` (triggers ~1.6 GB SPICE kernel download).
- **Environment**: `tomtoolkit==3.0a9`/`tom_catalogs` mismatch (v1.0 blocker) resolved by PR #38 (merged 2026-06-11).
- **Current codebase state (as of v1.6 close)**: 194 tests passing under `./manage.py test solsys_code`; `ruff check .` and `ruff format --check .` clean. New in v1.4: `solsys_code/models.py` (`CalendarEventTelescopeLabel`), `solsys_code/migrations/0001_calendareventtelescopelabel.py` (first real solsys_code migration), `solsys_code/templatetags/` package (`calendar_display_extras.py`), `src/templates/tom_calendar/partials/calendar.html` (rewritten event branches + footer legend). New in v1.5: `solsys_code/management/commands/sync_gemini_observation_calendar.py`, `solsys_code/tests/test_sync_gemini_observation_calendar.py` (15 tests), `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb`. New in v1.6: `solsys_code/calendar_utils.py` (12 extracted symbols + `insert_or_create_calendar_event`), `solsys_code/calendar_urls.py` (full namespace replacement for `/calendar/`), `text_color_for_bg`/`_relative_luminance` in `calendar_display_extras.py` (DISPLAY-08), `fomo_render_calendar` wrapper view in `views.py` (DISPLAY-09). 44 commits across Phases 11-12 (2026-06-27 → 2026-06-29); 39 files changed, +4,586/-405 lines.
- **v1.2 correctness bug found against real data (drives v1.3)**: checked real `ObservationRecord` rows in this dev DB (pk=1 obs_id=3780553 PENDING, pk=2 obs_id=3781325 COMPLETED, both proposal `LTP2025A-004`). Neither has a `site` key in `parameters`, and neither has a flat `instrument_type` key — real LCO submissions use multi-configuration cadence requests (`c_1_instrument_type`..`c_5_instrument_type`, each with `c_N_ic_1..5_*` exposure settings); only the configuration(s) with a populated `exposure_time` are "meaningful" (in both records checked, only `c_1` was populated). `SITE_TELESCOPE_MAP`'s 2-entry `coj`/`ogg` dict was also `[ASSUMED]`/web-search-only, never confirmed against real data. v1.2's shipped command would silently `KeyError`/skip every real record in this database.
- **LCO site -> MPC code reference table** (from https://lco.global/observatory/sites/mpccodes/), basis for the v1.3 verified mapping dict: ogg/Haleakala (F65,T04,T03), elp/McDonald (V37,V39,V38,V45,V47), lsc/Cerro Tololo (W85,W86,W87,W89,W79), cpt/Sutherland (K91,K92,K93,L09), coj/Siding Spring (Q58,Q59,Q63,Q64,E10), tfn/Tenerife (Z31,Z24,Z21,Z17), tlv/Wise Observatory (097), sor/SOAR Cerro Pachon (I33). A bare 3-letter site code is 1-to-many against MPC codes; the fully-qualified `siteid-enclid-telid` code (e.g. `coj-clma-2m0a` -> `E10` -> "FTS") is 1-to-1.

## Constraints

- **Astronomy library**: `astropy` for sun-position calculations.
- **Timezones**: `zoneinfo` (stdlib, `tzdata` installed).
- **Data source**: Site coordinates from `Observatory` model records (MPC obscode lookup).
- **Precision**: Sunset/sunrise must match Las Campanas skycalc to ≤ 2 minutes; horizon dip at 2402 m must be 1.44° ± 0.02°.
- **DB precision**: `astropy Time.to_datetime()` produces microseconds — strip with `.replace(microsecond=0)` before any DB key lookup.
- **Testing**: DB-dependent tests in `solsys_code/tests/`, run with `./manage.py test solsys_code`. Quality gates: `ruff check .` and `ruff format --check .`.

### Stage vs Phase numbering

This project tracks progress with two intentionally different numbering schemes,
and they do not line up one-to-one:

- **Stage** is the issue #37 feature-stage grouping, used in notebook headers and
  the "What This Is" list above: Stage 1 = site/ephemeris helper; Stage 2 =
  classical run ingest; Stage 3 = LCO queue sync, now being generalized to
  LCO + SOAR across v1.3's Phases 5-7; Stage 4 = future full observation-record
  sync for all facilities.
- **Phase** is the GSD execution-phase count — the `NN-name` directories under
  `.planning/phases/`.

The two schemes are different granularities on purpose: Stage 2 spans GSD
Phases 2-3 (parsing in Phase 2, ingest in Phase 3), and Stage 3 corresponds to
Phase 4 and is being extended by Phases 5-7 (multi-proposal/multi-facility
selection, instrument-type extraction, telescope-label resolution). A notebook
header that says "Stage N" predates this clarification and is intentionally
left as-is — it is not meant to imply "Phase N".

## Demo Notebooks

Each phase ships a demo notebook under `docs/notebooks/pre_executed/`. Notebooks require manual execution to see outputs (pre-commit hook clears all `.ipynb` output cells — consistent project convention). Use the Django setup boilerplate from the Django setup boilerplate section below before importing any model.

### Django setup boilerplate for notebooks

```python
import os
import sys
from pathlib import Path

import django

repo_root = str(Path.cwd().resolve().parents[2])  # adjust depth to repo root
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'src.fomo.settings')
os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'true')

django.setup()
```

Without the `sys.path` fix, imports fail with `ModuleNotFoundError: No module named 'src'`. Without `DJANGO_ALLOW_ASYNC_UNSAFE`, ORM calls raise `SynchronousOnlyOperation`.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Source `SITES` coordinates from `Observatory` model by MPC obscode | Avoids duplicating lat/lon/altitude already modeled in `solsys_code_observatory`; `tom_observations.facilities.lco` is incomplete/inconsistent for this purpose | Implemented in Phase 01 via `get_site()` and `Observatory.to_earth_location()` |
| Scope Stage 1 GSD run to a single self-contained unit | Per `gsd_experiment.rst` — trial the GSD workflow before committing to the full 4-stage feature | Phase 01 completed end-to-end; 9/9 requirements validated; GSD loop validated |
| DB-dependent tests go in `solsys_code/tests/` (Django suite) | Consistent with existing two-suite split; pure-Python `tests/` suite has no DB access | Implemented in Phase 01 (`test_telescope_runs.py`), Phase 03 (`test_load_telescope_runs.py`) |
| Telescope token resolved by prefix match against `SITES.keys()` | Exact match wins; 2+ matches raise `ValueError` listing candidates — no silent guessing (D-01) | Implemented in Phase 02; bare `'Magellan'` correctly raises `ValueError` naming both Clay/Baade |
| Three date-range regex patterns tried in order | month-after-range → cross-month → month-before-range; covers all 3 sample formats | Implemented in Phase 02; all 4 success criteria pass |
| CalendarEvent create-or-update keyed on `(telescope, instrument, start_time)` via `get_or_create` + conditional save | Idempotent re-run; no `modified`-timestamp churn on unchanged events (D-04) | Implemented in Phase 03; INGEST-03 validated by test and UAT |
| `astropy Time.to_datetime()` microsecond-strip | `to_datetime()` produces sub-second precision that breaks `get_or_create` key matching on re-run | Fixed in Phase 03 code review (commit `437aa53`); `.replace(microsecond=0)` before DB save |
| Per-line `(ValueError, Observatory.DoesNotExist)` handler (log+skip) | Both are data/setup issues for that line; abort would discard all subsequent valid lines | Implemented in Phase 03 (D-02); skipped lines reported with line number + original text |
| `CalendarEvent.url` keyed on `LCOFacility().get_observation_url(observation_id)`, not the literal `requestgroups/<id>/` string from the original ROADMAP wording | Real method returns `/requests/<id>` (no trailing slash); using the wrong literal would silently break find-or-create matching against real LCO data (D-01) | Implemented in Phase 04; confirmed live via `LCOFacility().get_observation_url('12345')`; `grep -c requestgroups` on source = 0 |
| Terminal-state title prefix trigger uses `get_failed_observing_states()` (4 states), not `get_terminal_observing_states()` (5 states = those 4 + `COMPLETED`) | Research correction (D-06): the wrong helper would wrongly prefix `COMPLETED` records, which should get a clean title | Implemented in Phase 04; verified live (4-vs-5 state sets) plus a dedicated COMPLETED-gets-clean-title test |
| No-churn create-or-update compares all 7 changeable fields before `.save()` | Avoids `modified`-timestamp churn on unchanged records, same pattern as Phase 03's `load_telescope_runs` | Implemented in Phase 04 (SYNC-04); verified by a test asserting bit-for-bit-identical `modified` on an unchanged re-run |
| Status-aware `CalendarEvent` coloring deferred rather than built alongside the narrower `[QUEUED]` de-emphasis fix | Visual-design decision (telescope/proposal-keyed hash + status opacity), not just engineering; user wanted to explore "striping" as an alternative before committing | Narrower de-emphasis fix shipped (260618-lw4/mck); fuller scheme captured as a pending todo for a future milestone |
| `FACILITIES['SOAR']` mirrors `FACILITIES['LCO']` literally (same hardcoded `api_key`/`portal_url`), not a new env-var-backed entry | D-04/D-05: SOAR authenticates against the same LCO Observation Portal; narrower reading avoids migrating `LCO`'s existing credential handling within this phase's query/selection/dispatch scope | Implemented in Phase 05; `FACILITIES['LCO']` byte-for-byte unchanged, verified live |
| Eager `{'LCO': LCOFacility(), 'SOAR': SOARFacility()}` dispatch dict built once per run, not lazily per record | Fixes the SELECT-05 bug (one `LCOFacility()` reused for every record); avoids per-record instantiation cost; both keys always present (D-06) | Implemented in Phase 05; proven by a discriminating spy test (SOAR spy called, LCO spy not called for a SOAR record) |
| Sentinel `None` + `InstrumentExtractionError` contract for `_extract_instrument`, not a bare exception | Matches the file's existing "return `None` to signal non-match" style; keeps malformed-record handling consistent with the rest of the command | ✓ Good — implemented in Phase 06; dedicated `extraction_failed` counter, no regression in the 19 pre-existing tests |
| Kept a flat `instrument_type` fallback tier beyond the `c_1`/`c_2` multi-config scan | Preserves the 19 pre-existing regression tests that exercise today's legacy single-config DB shape | ✓ Good — implemented in Phase 06 |
| `tlv` (Wise Observatory) dropped entirely from `SITE_TELESCOPE_MAP` | Operator-confirmed at the 07-01 Task 1 checkpoint: absent from both installed `LCOSettings.get_sites()`/`SOARSettings.get_sites()`; shipping a guessed entry was rejected | ✓ Good — implemented in Phase 07; scope corrected to the 7 real, installed-library-confirmed sites |
| `elp`/`lsc`/`cpt`/`tfn` get both `1m0` and `0m4` aperture-class entries | Operator (LCO staff) confirmed both aperture classes exist at each of those sites — no `[ASSUMED]` tag needed | ✓ Good — implemented in Phase 07 |
| Live-API failure/timeout AND a successfully-returned-but-unmapped `(site, telescope_code)` pair share the same `telescope_api_failed` counter and `[UNVERIFIED]` prefix | 07-RESEARCH.md Pitfall 4: both are the same user-visible degrade signal; splitting them into two differently-labeled failure classes adds complexity without operator value | ✓ Good — implemented in Phase 07 |
| D-09: terminal-failure title prefix beats `[UNVERIFIED]`; the two are mutually exclusive | `[UNVERIFIED]` only ever applies to a placed (non-terminal) record, matching Phase 4's existing terminal-prefix-wins precedent — avoids a new combination rule | ✓ Good — implemented in Phase 07 |
| `_coarse_telescope_label(instrument_type, facility_name)` — 2-arg signature, SOAR detected via `facility_name.upper() == 'SOAR'` (exact match, not substring) | The v1.3 milestone audit found the 1-arg version silently fell through to the raw instrument string for SOAR (`'SOAR_GHTS_REDCAM'[:3]` isn't a recognized aperture prefix); needed the call-site's facility context to special-case SOAR, not pattern-match on the instrument string | ✓ Good — implemented in Phase 07.1; SOAR unconditionally returns `'4m0'`, LCO branch byte-for-byte unchanged |
| Call site `_build_event_fields` passes `record.facility` (the string) into `_coarse_telescope_label`, not the in-scope `LCOFacility`/`SOARFacility` instance | The function needs the facility *name* to branch on, not a credentialed facility object — passing the instance would be a type mismatch and an unnecessary credential-bearing object threaded through a pure labeling function | ✓ Good — implemented in Phase 07.1; closes the doubled-title defect (`[UNVERIFIED] SOAR_GHTS_REDCAM SOAR_GHTS_REDCAM` → `[UNVERIFIED] 4m0 SOAR_GHTS_REDCAM`) |
| `CalendarEventTelescopeLabel` uses `OneToOneField(primary_key=True)` — sidecar shares the FK as its PK | Extends `tom_calendar.CalendarEvent` (a third-party model) without touching its migrations or schema; reverse accessor `event.telescope_label_meta` is a single-row read, not a queryset | ✓ Good — implemented in Phase 08; solsys_code's first real migration |
| Sidecar `update_or_create` kept as a standalone statement, never merged into the existing `CalendarEvent` fields dict | Folds into the no-churn comparison pipeline would require comparing and diffing an extra model; standalone keeps the no-churn discipline isolated | ✓ Good — implemented in Phase 08; existing `CalendarEvent` no-churn test unchanged |
| Template treats missing sidecar row as "verified" by documented default | `load_telescope_runs` events have no sidecar; defaulting to verified (not fallback) avoids misleading the operator about classically-scheduled events | ✓ Good — implemented in Phase 08; documented in template comment |
| `proposal_color` uses sha256 (not Python's built-in `hash()`) + `.strip().upper()` normalization | Built-in `hash()` is process-salted in CPython 3.3+ — different colors on every restart. sha256 is deterministic across restarts, hosts, and Python versions | ✓ Good — implemented in Phase 09; `grep -c 'hash(' calendar_display_extras.py` = 0 |
| `PROPOSAL_PALETTE` order locked verbatim from 09-UI-SPEC.md (8 colorblind-vetted, white-text-AA hex values) | Palette order determines which proposal gets which color; changing it after deployment recolors all existing events; lock it early | ✓ Good — implemented in Phase 09; order is a named constant, not derived |
| `visible_proposals` groups by resolved color hex, not by raw proposal string | Two proposals that hash to the same palette slot (collision) share one swatch; keying on the string would create two identical-color swatches — misleading | ✓ Good — implemented in Phase 09; D-04 collision-grouping design decision |
| Status box-shadow rings (queued/terminal) composed as a prefix to the existing inline style, not replacing the Phase 8 dashed border | D-09: the two visual signals are orthogonal (label verification vs. event status); composing avoids the `{status_border} border: 2px dashed...` ordering problem by always appending | ✓ Good — implemented in Phase 09; Pitfall 3 composition test green |
| Click-to-filter JS IIFE placed inside the `#calendar-partial` fragment (before its closing `</div>`), not in the page `<head>` | Pitfall 5: htmx `outerHTML` swap replaces the fragment including any `<script>` inside it — the IIFE re-executes on each month swap and resets `activeProposal` to null, preventing stale filter state | ✓ Good — implemented in Phase 09; documented in template comment; human-verified in UAT |
| `update_fields=changed` (list of field names) for no-churn save, not unconditional `save()` | Prevents modifying `CalendarEvent.modified` on unchanged fields while satisfying GEM-NOCHURN-01; pattern re-used from Phase 03's `load_telescope_runs` | ✓ Good — implemented in Phase 10; GEM-NOCHURN-01 verified by test |
| `safe_params` strips `password` key as first statement in each loop iteration, before any logging or exception paths | D-04: ensures no code path can accidentally log or persist the credential, even via an unexpected exception | ✓ Good — implemented in Phase 10; GEM-SECURE-01 verified by 15/15 passing tests |
| `site_key`/`telescope` determination placed BEFORE the `try/except` block, not inside it | A `KeyError` on `obsid` lookup inside `try` would cause a `NameError` in the `except` clause that references `site_key` in `counters[site_key]['skipped']`; placing it before avoids the undefined-variable trap | ✓ Good — implementation refinement found during Task 2; no regressions |
| Raw-fallback branch for GEM-INSTR-01: explicit window + unknown obs code → `instrument = obs_code` | D-01 skip path (no window → skip) only applies when no explicit window is present; an explicit window with an unknown obs code still deserves a CalendarEvent, using the raw obs code as a readable fallback | ✓ Good — implemented in Phase 10; covered by test |
| `insert_or_create_calendar_event` uses `event.save()` (not `event.save(update_fields=changed)`) on update | `update_fields` silently skips `auto_now` fields (`CalendarEvent.modified`), breaking tests that assert the timestamp updates after a write; plain `event.save()` matches original LCO sync behavior | ✓ Good — Phase 11 fix commit 3fb5ad7; deviation from plan caught by test failures during worktree recovery |
| Absolute import style (`from solsys_code.calendar_utils import ...`) throughout all three commands | Plan 11-01 originally specified relative imports; Plan 11-02 explicitly accepted absolute; functional behavior identical — consistency chosen over plan wording | ✓ Good — Phase 11; all three commands use absolute imports uniformly |
| `calendar_urls.py` is a full replacement of `tom_calendar.urls` (all 6 URL names), not a single-route shadow | When FOMO's namespace only registered the root URL name, all `calendar:create-event` / `calendar:update-event` etc. reversals raised `NoReverseMatch` — Django resolves the first-registered namespace and expects all names to be there | ✓ Good — implemented in Phase 12; W005 warning is expected/harmless; all 6 reversals resolve correctly through FOMO namespace |
| TDD RED/GREEN gate enforced for Phase 12 Task 1 (`text_color_for_bg`) | Follows the pre-existing DISPLAY-08 research decision to validate the WCAG formula via test before wiring it into the template | ✓ Good — RED commit `d79a734`, GREEN commit `cda8789`; confirmed via git log |
| ESO-04 verdict is Bypass, not Bridge | All Plan 01 evidence came from direct `p2api`/`ESOAPI` reads (`getOB`, `getOBExecutions`, `getNightExecutions`); Bridge's premise — patching `tom_eso` to hand-create `ObservationRecord` rows — was never exercised, per the phase's D-08 read-only guardrail | ✓ Good — implemented in Phase 13; rationale traces each of ESO-01/02/03 explicitly in `13-DECISION.md` |
| `eso_p2_probe.py` kept as a throwaway, git-excluded script, never a committed deliverable | D-09: the investigation itself is exploratory (scratch script/shell session), not a shippable module — only the decision docs are committed | ✓ Good — implemented in Phase 13; registered in `.git/info/exclude`, zero write-style `p2api` calls, never staged |
| La Silla `production_lasilla` failure root-caused to `tom_eso.eso_api.ESOAPI` unconditionally requiring a `p1api` connection (whose `API_URL` lacks a La Silla entry), not genuine account/API inaccessibility | Live follow-up test (`p2api.ApiConnection('production_lasilla', ...)` bypassing `ESOAPI`/`p1api`) connected without error, confirming the wrapper-bug diagnosis; the operator also confirmed working La Silla web-portal access with the same credentials | ✓ Good — documented in Phase 13's `13-DECISION.md`; La-Silla-sourced OB data itself remains unconfirmed (the one live test returned a Paranal-instrument run) |
| Status split into two independent `TextChoices` fields (`approval_status`: 3 values, `run_status`: 8 values) rather than one flat vocabulary (D-02) | A DDT/proposal request's real-world outcome can be pending independently of admin review state — a flat vocabulary can't represent both dimensions at once | ✓ Good — implemented in Phase 14; ROADMAP/REQUIREMENTS wording corrected post-verification to match |
| Natural key `(campaign, telescope_instrument, ut_start)` for idempotent re-import, backed by a DB-level `UniqueConstraint` (D-04, added post-review as WR-05) | `get_or_create` alone is only race-safe with a real DB constraint on the lookup fields; without one, concurrent/overlapping imports could create duplicate rows | ✓ Good — implemented in Phase 14; deep code review (CR-02) additionally found the unparseable-UT-time fallback could collide two distinct rows onto the same key — fixed with a deterministic per-batch disambiguating offset |
| Single-target campaigns auto-assign their one `Target` to every imported row (D-07); re-imports always reset `target` to the auto-resolved value | Matches the real 3I/ATLAS sheet's common case (comet-only campaign); re-import overwrite behavior is documented (WR-07) rather than tracking manual-correction provenance, which would be a larger scope increase | ✓ Good — implemented in Phase 14 |
| 3-tier site resolution — existing `Observatory` → live MPC API → flagged placeholder (D-08) — length/blank-checked against `Observatory.obscode.max_length` before any tier is attempted | Never fabricate or truncate a site code; blank/oversized codes are flagged `site_needs_review` with `site=None` rather than guessing | ✓ Good — implemented in Phase 14; hardened post-review (WR-01/02/03/04) against network timeouts, malformed API responses, and unhandled race conditions |
| Deep (not standard) code review depth on Phase 14 caught 2 critical, reproduced data-correctness bugs (PM/AM time markers silently discarded; natural-key collision silently merging distinct rows) that all task-level self-checks and 227 passing tests had missed | Both bugs sat directly on the phase's CAMP-04 purpose — validating the schema against the real, messy sheet before a live import; a shallower review pass would likely not have traced the "never raise, never fabricate, never silently merge" invariants against actual behavior | ✓ Good — both fixed with regression tests before phase completion (commits `70f6ef3`, `dab0314`); 242/242 tests pass post-fix |
| `resolve_site()` gained a keyword-only `create_placeholder` parameter (default `True`) rather than changing tier-3 behavior globally | Manual UAT just before v2.0 close found the approval endpoint silently fabricated placeholder `Observatory` rows for unresolvable public free-text site names (e.g. 'DCT'); the already-vetted CSV import path (Phase 14) legitimately wants tier-3's placeholder-creation behavior unchanged, so the fix had to be call-site-scoped, not a blanket behavior change | ✓ Good — implemented via quick task `260705-l1v` (2026-07-05); approval now leaves `site=None, site_needs_review=True` with no bogus Observatory row, CSV path unaffected, 332/332 tests pass |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-11 — Phase 21 (site disambiguation & submitter contact opt-in) complete, SITE-01/02/03 and VIEW-05 validated. Code review found and fixed 2 critical bugs before ship (fuzzy-matched candidates couldn't resolve on approve; CreateObservatory's `?next=`/`?obscode=` round-trip was dropped by the real template). This was the last phase of v2.1 Uncertain Scheduling & Site Disambiguation — all target features shipped; milestone close (`/gsd-complete-milestone`) not yet run.*
