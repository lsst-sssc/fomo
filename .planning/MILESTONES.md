# Milestones

## v2.0 Campaign Coordination for Rare/Urgent Objects (Shipped: 2026-07-05)

**Phases completed:** 4 phases, 13 plans, 33 tasks
**Closeout type:** override_closeout (4 acknowledged pre-existing items — 2 pending todos, 2 dormant ESO seeds, none introduced by v2.0; see STATE.md Deferred Items). One real gap found during pre-close manual UAT (approval-queue site-visibility + silent placeholder-Observatory fabrication) was fixed via quick task `260705-l1v` before shipping, not deferred.

**Key accomplishments:**

- `CampaignRun` Django model with two-field TextChoices status vocabulary (3-value approval, 8-value run status), required campaign FK, nullable target/site FKs, migration applied, and 6 model-level tests green.
- `campaign_utils.py` (3-tier site resolution, best-effort UT-time parsing, status translation, no-churn create-or-update) plus the `import_campaign_csv` management command, both covered by 20 passing Django tests with the MPC Obscodes API fully mocked.
- Synthetic, PII-free `campaign_sample.csv` fixture plus a paired, executed `import_campaign_csv_demo.ipynb` demonstrating the bootstrap import's created/updated/skipped summary, auto-target resolution, and the `pending_review` -> `approved`/`rejected` approval lifecycle -- all offline, satisfying CAMP-05.
- Anonymous-accessible, PII-gated `django-tables2`/`django-filter` table listing every `CampaignRun` for a campaign, plus a campaigns list page — first real consumer of both libraries in FOMO.
- Per-campaign "View {name} Runs" links on target-detail pages via a second `target_detail_buttons()` entry, plus FOMO's first `AppConfig.nav_items()` navbar hook for a global "Campaigns" entry -- completing VIEW-02.
- Plain `forms.Form` submission form with a non-raising HiddenInput honeypot, a dispatch-level `is_staff` gate mixin, and a console `EMAIL_BACKEND` -- the three self-contained leaf dependencies for Phase 16's submission/approval write path.
- `CampaignRunSubmissionView` (FormView) wired at `campaigns:submit`, backed by a `transaction.atomic()`-guarded `.objects.create()` that turns Pitfall 4's natural-key collision into a friendly form error, a honeypot short-circuit that returns the identical thanks redirect (SUBMIT-04), and a PII-free staff-notification email (SUBMIT-05).
- Staff-gated two-section approval queue (pending actionable / recently-decided read-only), a POST-only atomic approve/reject endpoint whose double-approve is a proven no-op, and a `CAMPAIGN:{pk}` `CalendarEvent` projection on successful approve routed through the shared `insert_or_create_calendar_event()` helper.
- Non-staff visitors to a per-campaign table now see approved and rejected runs but never `pending_review` ones (queryset-level `.exclude()`, mirroring the existing D-13 discipline); "Submit a Run" buttons and a staff-only "N pending review" banner close the discoverability loop for the form (Plan 02) and approval queue (Plan 03).
- `ApprovalQueueTable.Meta` gains `exclude`/`sequence` so Approve/Reject leads column 1 and three structurally-blank post-observation columns are dropped, while `CampaignRunTable` stays byte-for-byte spreadsheet-parity.
- Pure-logic `campaign_gap.py` module composing `telescope_runs.sun_event()` (observable side) with a `CampaignRun` query (claimed side) into a cached set-difference, plus the GAP-01 dark-window-only decision artifact.
- `CampaignGapAnalysisView` wires Plan 01's `get_or_compute_gap` into a public, GET-triggered, cached endpoint with a campaign-scoped selection form and server-side IDOR re-validation of target/site pks.
- Gap-analysis page (`campaignrun_gap_analysis.html`) and D-14-gated "Show Coverage Gaps" button on the campaign table, rendered verbatim to the UI-SPEC copywriting contract, human-verified and approved.

---

## v1.7 ESO/VLT Calendar Sync — Feasibility Spike (Shipped: 2026-07-02)

**Phases completed:** 1 phases, 2 plans, 5 tasks

**Key accomplishments:**

- Live Paranal (VLT) production P2 API investigation confirms credentials work, real OB status/execution shapes are captured, and headless credential-sourcing via env-var-backed ESOAPI is a viable path — while La Silla's production_lasilla environment fails via tom_eso's ESOAPI wrapper specifically (root-caused to a p1api gap, not genuine API inaccessibility).
- Recommends Bypass (sync straight from p2api to CalendarEvent, skipping ObservationRecord) for a future ESO/VLT calendar sync, grounded directly in Plan 01's real Paranal P2 API evidence, with a durable docs/design/eso_feasibility_spike.rst summary for future milestones.

---

## v1.6 Tech Debt & Display Polish (Shipped: 2026-06-29)

**Phases completed:** 2 phases (11–12), 3 plans, 44 commits | 39 files | +4,586 / -405 lines
**Timeline:** 2026-06-27 → 2026-06-29
**Closeout type:** override_closeout (1 acknowledged todo — site/telescope extraction refactor, delivered by Phase 11 but tracking file not closed; see STATE.md Deferred Items)

**Key accomplishments:**

- `solsys_code/calendar_utils.py` created with 12 extracted symbols (`SITE_TELESCOPE_MAP`, `_extract_instrument`, `_coarse_telescope_label`, `insert_or_create_calendar_event()`, and 8 helpers) — REFAC-01 delivered; all symbols importable independently of any management command.
- All three management commands (`load_telescope_runs`, `sync_lco_observation_calendar`, `sync_gemini_observation_calendar`) refactored to delegate CalendarEvent create-or-update to shared `insert_or_create_calendar_event()` helper; duplicated logic removed; "upsert" replaced with plain English in design docs and MILESTONES.md — REFAC-02 delivered.
- `text_color_for_bg` WCAG 2.1 template tag added to `calendar_display_extras.py`: computes white/black text color from relative luminance against any proposal palette background; all 8 `PROPOSAL_PALETTE` entries + `NEUTRAL_SLOT_COLOR` return `#fff`; `#ffffff` → `#000`; TDD RED/GREEN gate enforced — DISPLAY-08 delivered.
- `fomo_render_calendar` wrapper view with `prefetch_related('telescope_label_meta')` + `Count` annotation eliminates N+1 query per event; `calendar_urls.py` full namespace replacement ensures all `calendar:*` URL reversals resolve; N+1 regression test via `CaptureQueriesContext` — DISPLAY-09 delivered.
- Full test suite: 194 `./manage.py test solsys_code` tests pass; `ruff check .` and `ruff format --check .` clean.

---

## v1.5 Gemini Calendar Sync (Shipped: 2026-06-27)

**Phases completed:** 1 phase (Phase 10), 2 plans

**Key accomplishments:**

- `sync_gemini_observation_calendar` management command syncing GEM ObservationRecords to CalendarEvents with per-record password scrubbing, ToO-type window derivation from `FACILITIES['GEM']['programs']`, and no-churn `get_or_create(url=) + save(update_fields=changed)` idiom — 15/15 tests passing.
- Pre-executed demo notebook confirming all four D-06 scenarios (explicit window, Rap: derived, Std: derived, ON_HOLD + idempotent re-run) with no credential leakage; CLAUDE.md companion-notebook list extended to four entries.

Known deferred items at close: 1 (see STATE.md Deferred Items — site/telescope extraction refactor, pending since v1.3)

---

## v1.4 Calendar Visual Clarity (Shipped: 2026-06-26)

**Phases completed:** 2 phases, 4 plans, 5 tasks

**Key accomplishments:**

- Added `CalendarEventTelescopeLabel` OneToOneField sidecar model (solsys_code's first real model/migration) and a standalone `update_or_create` write in `sync_lco_observation_calendar.py` that persists the live-verified-vs-fallback telescope-label outcome per `CalendarEvent`.
- Added a dashed-border + native-tooltip render branch to both the all-day and timed event loops in `calendar.html`, plus the first `calendar.html` view-level rendering test in this codebase, proving fallback-labeled events are visually distinguishable and verified/no-row events are unaffected.
- New `calendar_display_extras` template-tag library with `proposal_color` (sha256 → 8-color colorblind-vetted palette), `status_border_css` (locked CSS literals), and `visible_proposals` (collision-grouped legend aggregation) — replacing the pk-based color system.
- Rewrote `calendar.html` event branches: proposal-keyed color, fixed `[QUEUED]` grey-override, status box-shadow rings composed with Phase 8 dashed border, footer legend with click-to-filter JS IIFE surviving htmx month swaps.

---

## v1.3 Full LCO Facility Sync (Shipped: 2026-06-24)

**Phases completed:** 4 phases, 5 plans, 14 tasks

**Key accomplishments:**

- Generalized `sync_lco_observation_calendar` to accept a comma-list/ALL `--proposal` argument and dispatch LCO and SOAR `ObservationRecord`s through their own facility instance, fixing the SELECT-05 single-shared-`LCOFacility()` dispatch bug.
- Replaced the flat `parameters['instrument_type']` read in `sync_lco_observation_calendar.py` with a `c_1..c_5` multi-config scanner that distinguishes SOAR's SPECTRUM science config from its ARC/LAMP_FLAT calibration configs and detects LCO MUSCAT's per-channel exposure shape, adding a dedicated `extraction_failed` counter for fully-malformed records.
- Migrated SITE_TELESCOPE_MAP to a verified 7-site (site, aperture_class) dict and added `_resolve_placement_block`/`_aperture_class_from_telescope_code`/2-arg `_derive_telescope` for single-attempt, timeout-bounded, never-leaking LCO Observation Portal API resolution.
- Replaced the flat `parameters['site']` shim with a live-API + coarse-fallback decision tree, an `[UNVERIFIED]` title prefix with D-09-resolved priority, and a per-facility `telescope_api_failed` counter -- completing Phase 7's user-visible behavior.
- Made `_coarse_telescope_label` facility-aware so a placed SOAR record's API-failure fallback resolves to `'4m0'` instead of the raw `'SOAR_GHTS_REDCAM'` string, closing the doubled `[UNVERIFIED] SOAR_GHTS_REDCAM SOAR_GHTS_REDCAM` title defect found in the v1.3 milestone audit.

---

## v1.2 LCO Queue Calendar Sync (Shipped: 2026-06-18)

**Phases completed:** 1 phases, 1 plans, 3 tasks

**Key accomplishments:**

- `sync_lco_observation_calendar` management command syncs LCO ObservationRecords to CalendarEvents via TDD, keyed on the real `LCOFacility().get_observation_url()` portal URL, with no-churn create-or-update and a terminal-failure title prefix system that correctly excludes COMPLETED (D-06 research correction).

---

## v1.1 Classical Run Ingest (Shipped: 2026-06-16)

**Phases completed:** 2 phases, 3 plans, 5 tasks

**Key accomplishments:**

- `ParsedRun` dataclass + `parse_run_line()` parser handles all 3 classical-schedule date-range formats (month-before/after-range, cross-month), hyphenated instruments, year defaulting, and telescope prefix-match resolution with descriptive ValueError for ambiguous names.
- `load_telescope_runs` Django management command expands parsed run date ranges into idempotent nightly `CalendarEvent`s using `sun_event()` for accurate UTC sunset/sunrise — creating or updating via `get_or_create` keyed on `(telescope, instrument, start_time)` with conditional save.
- 6-test `TestLoadTelescopeRuns` suite covers INGEST-01/02/03 plus per-line error handling and no-churn idempotency; all 95 `./manage.py test solsys_code` tests pass.
- 6/6 UAT scenarios confirmed live on dev DB; demo notebook `load_telescope_runs_demo.ipynb` confirmed executable end-to-end.

---

## 1.0 Site/Ephemeris Helper (Shipped: 2026-06-14)

**Phases completed:** 1 phases, 2 plans, 4 tasks

**Key accomplishments:**

- Observatory model gains a timezone field and to_earth_location(), migration 0002 seeds 4 telescope sites (Magellan-Clay/Baade, NTT, FTS), and a new telescope_runs.py computes dip-corrected sunset/sunrise (-(0.833+dip)) and -15deg dark-window UTC crossing times via astropy get_sun/AltAz with coarse-scan + bisection root-finding.
- Extended test_telescope_runs.py with skycalc-accuracy validation for 4 June 2026 Las Campanas nights, a -18deg astronomical-twilight cross-check matching 19:16/06:08 Santiago local to the second, and zoneinfo DST-offset tests for Santiago/Sydney - all passing with ruff check/format clean.

---
