# Milestones

## v2.1 Uncertain Scheduling & Site Disambiguation (Shipped: 2026-07-18)

**Phases completed:** 8 phases, 26 plans, 60 tasks
**Closeout type:** override_closeout (4 pre-existing acknowledged items — 2 pending todos, 2 dormant ESO/VLT seeds, none introduced by v2.1, all already tracked in STATE.md's Deferred Items table since v1.7/v2.0 close; see STATE.md Deferred Items). Two debug-session bookkeeping items found during pre-close audit (a stale `diagnosed` status on a session Phase 25 had already resolved, and a false-positive flag on the knowledge-base index file) were fixed inline before shipping, not deferred. All 13 v1 requirements shipped (100%); all 8 phases verified `passed`.

**Key accomplishments:**

- Locked all five SCHED-01 decisions (window schema, TBD natural key, CSV range/TBD parsing rules, fuzzy-match library split verdict, no obscode widening) into 18-DECISION.md's Recommendation section and a new durable `docs/design/uncertain_scheduling_spike.rst`, each recommendation tied directly to a Plan 01 Finding.
- CampaignRun's obs_date/ut_start/ut_end replaced by a nullable window_start/window_end DateField pair, enforced by two partial UniqueConstraints, via one combined non-reversible migration that backfills and dedupes existing rows before swapping constraints.
- campaign_gap.claimed_dates() rewritten to claim every date in an inclusive window_start/window_end range directly, replacing the obs_date/ut_start-derived night-boundary logic and deleting the now-dead `_observing_night_date()` helper.
- campaign_tables.py/campaign_views.py/campaign_forms.py rewritten against window_start/window_end: a combined TBD/single-date/range window column with cross-backend nulls-last sort, a D-06 hybrid ground-vs-space calendar projection on approve, and a submission form collapsed to a single observing date.
- import_campaign_csv now keys its natural-key lookup on window_start (single-night collapse), replaces the sub-second collision-offset hack (impossible on a DateField) with a log-and-skip duplicate handler, and its paired demo notebook is regenerated against the real, now-migrated dev DB.
- claimed_dates() now distinguishes ground vs. space-mission CampaignRuns — space-mission runs with an un-narrowed window claim nothing and surface in a new pending_narrowing_runs gap-page alert, computed once from the site parameter without widening the PII-minimizing queryset.
- CampaignRun gains original_obs_date_raw/window_needs_review fields (migration 0006, applied to the dev DB) and the campaign table's TBD badge now shows the raw sheet text as an HTML-escaped hover tooltip.
- `parse_obs_window()` now parses full-date and compact rollover ranges and never raises for any Obs. Date input (7-tuple TBD contract); `import_campaign_csv` persists both range and flagged-TBD rows instead of skipping them, branching its natural key to match the model's two partial UniqueConstraints exactly.
- Extended `campaign_sample.csv` with a date-range and a TBD row, then regenerated `import_campaign_csv_demo.ipynb` with a new committed-output cell demonstrating IMPORT-01's resolved multi-night window and IMPORT-02's flagged-TBD-with-preserved-raw-text import path end-to-end against the migrated dev DB.
- Bulk MPC obscode fetch (`MPCObscodeFetcher.query_all()`) feeding a 24h-cached, local+MPC merged candidate pool (`build_site_candidates()`) and a `difflib`-based fuzzy matcher (`fuzzy_match_candidates()`), plus the Wave-0 `TestSiteFuzzyMatch` scaffold with a reusable bulk-MPC fixture.
- A default-opt-out `contact_public_opt_in` checkbox on the public submission form drives a Case/When queryset annotation that exposes `contact_person`/`contact_email` to anonymous visitors only for opted-in `CampaignRun` rows, gated at the SQL SELECT.
- Inline `<input list=...>`/`<datalist>` site-disambiguation control wired into the staff approval queue's Site column, submitting into a single collapsed per-row `<form>` via the HTML5 `form=` attribute -- no new endpoint, no JavaScript.
- A `if run.site is None:` guard closes the SITE-03 clobbering bug in `CampaignRunDecisionView.post()`, wires the staff-submitted `site_selection` field into approve-time resolution (SITE-02), and extends `CreateObservatory` with a `?obscode=` prefill + validated `?next=` redirect so the "Create new Observatory" link round-trips back to the approval queue.
- Anonymous, throttled HTMX live-search endpoint (`campaigns:site_search`) with a substring-first-then-difflib site matcher, backing Plan 02's public form and approval-queue widgets and Plan 03's sites-needing-review row.
- Wired Plan 01's `campaigns:site_search` live-search endpoint into both site-entry surfaces: the public submission form's `site_raw` field (D-09, no create-new link) and the approval-queue pending row's inline site input (D-10, replacing the static datalist while keeping the Create-new-Observatory escape hatch), using the htmx-grammar-corrected `input[this.value.length >= 2] changed delay:300ms` trigger consistently on both.
- Closes the last Phase 21 gap: a third "Sites Needing Review" table on the approval-queue page lists approved runs with an unresolved site, and a new `resolve_site` decision action resolves the site via a concurrency-safe conditional claim and retroactively fires the deferred CalendarEvent projection only after it succeeds.
- SiteSearchView.get() now resolves its search term from `q`, then `site_raw`, then `site_selection` — a single view-side fallback chain that restores live-search rendering on the public submission form and both approval-queue widgets, with zero widget/template changes.
- Wrapped the Sites Needing Review section in a border-warning Bootstrap 4 card with an "action required" header, without reordering D-07's locked pending/decided/review document order
- Closes UAT gap 2B: a Sites Needing Review row for a tier-3 PLACEHOLDER Observatory (e.g. `Observatory(obscode='DCT')`, name `NEEDS REVIEW: DCT`, blank timezone) now shows the live-search correction widget and can be replaced through the UI, while D-06 racing/never-re-resolve protection, CR-01's genuine-site retry state, WR-01's read-only-table suppression, and D-09's never-fabricate invariant all stay intact.
- Cancelled classical-schedule runs now render a `[CANCELLED] {telescope} {instrument}` title on the calendar, computed fresh every ingest and reverting cleanly when the status word is removed.
- Staff can now mark an APPROVED CampaignRun cancelled or weathered from a new Decided-table action, which updates the linked CAMPAIGN:{pk} calendar event in place with a distinct `[CANCELLED]`/`[WEATHERED]` title prefix and terminal box-shadow ring, without ever fabricating an event for a range/TBD/unresolved-site run.
- Proved resolve_site('I11') resolves Gemini South (ground, real timezone) and that the real GS-2026A-FT-115 range-window Gemini run flows through the exact same approve/mark-status mechanism as any Magellan run, with zero CalendarEvents fabricated at any step (D-06/D-07) -- no production code added.
- Task-oriented Sphinx operator runbook (docs/runbooks/telescope_runs_calendar.rst) covering all five telescope-runs-calendar management commands plus the approval-queue mark_cancelled/mark_weather_failure staff actions, a five-command cheat-sheet, and a troubleshooting section built from real observed failure modes — wired into docs/index.rst's toctree and cross-referenced with a new Django-onboarding subsection in docs/installation.rst.
- Approved, site-resolved range-window CampaignRuns now project one dip-corrected CalendarEvent per night (ground) or one whole-day-span event (satellite), replacing the silent zero-event guard; a shared title helper keeps the window-context suffix intact through status changes.
- A one-off `backfill_range_calendar_events` management command finds already-APPROVED, site-resolved range-window `CampaignRun`s with no existing calendar event and projects them by delegating entirely to Plan 01's rewritten `_project_calendar_event()`, closing the gap left by projection only firing on the approve/resolve_site POST actions.

---

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
