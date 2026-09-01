# Milestones

## v2.2 One Canonical Run Record (Shipped: 2026-09-01)

**Phases completed:** 6 phases, 33 plans, 96 tasks

**Key accomplishments:**

- Read-only D-04/SPIKE-02 evidence against the real dev DB, plus a hand-authored `CalendarEventMeta` rename migration applied cleanly (zero row loss) and measured twice against the existing 265-test suite on a scratch branch, confirming D-02's prediction with two additional consumers.
- Executed SPIKE-01's `IntegrityError` coexistence check and SPIKE-03's three-way D-11 prototype against `CampaignRun` pk=1's real 15-night window, both measuring exactly as predicted (15/15/26 event counts, 4 uncovered nights), bracketed by a human-verified `/calendar/` dev-server load.
- Measured, D-11-grade evidence for the queue-run projection question 26-VERIFICATION.md left open: all 31 CampaignRun rows categorized (12 QUEUE/12 CLASSICAL/7 SPACE, RECON-07 split 8/11/0), the shipped `campaign_gap.claimed_dates()` over-claim reproduced with a file:line citation, and three candidate calendar-projection strategies (span/none/per-night) built and counted against pk=1's real 15-night window and its real 11 LCO events, with the bare `RUN:1` key proven stable under a window-narrowing stage transition and the rejected per-night candidate proven unstable (one orphaned key) -- no verdict stated, leaving the choice to plan 26-05.
- Locked the SPIKE-03 gap 26-VERIFICATION.md found: a queue-scheduled `CampaignRun` gets a reconciler-owned whole-window `RUN:{run_pk}` container event coexisting with its real `ObservationRecord`-derived `CalendarEvent`s (verified against live `sync_lco_observation_calendar.py` source to already narrow/refine as observations are scheduled and observed) -- a human decision against plan 26-04's measured evidence, mirrored into `26-DECISION.md`, the durable `docs/design/` page, `ROADMAP.md`, and `REQUIREMENTS.md`, with D-11's write-strategy deferral left untouched as the sole remaining open item.
- Renamed calendar_utils.py's five cross-module helpers to public names, added the single `derive_telescope_class()` helper (D-20) with its D-12 subset-assertion test suite, and relocated the six calendar_utils-owned tests out of test_sync_lco_observation_calendar.py.
- One-time repair command re-resolved 6 of 9 approved site-less CampaignRuns (JWST offline via alias, HST/Swift via live MPC tier-2) with create_placeholder=False as the D-22 fail-safe, plus a coordinate-derived Observatory.timezone backfill migration closing the E10 (Siding Spring) gap.
- Renamed CalendarEventTelescopeLabel to CalendarEventMeta and gave it a nullable SET_NULL run FK to CampaignRun, via hand-authored migrations 0008/0009 applied against the real dev DB with all 11 companion rows and their is_verified history intact.
- `CampaignRun` now records its ingest source and telescope-class allocation (both `TextChoices`, six and four values respectively) and owns a `CampaignRunObservation` link model to `ObservationRecord`, with migrations 0010/0011 applied against the real dev DB and the D-16 telescope_class backfill landing exactly the three predicted rows (JUICE->SPACE, LCO 1m->1m0, LCO 2m->2m0).
- Two editable admin inlines with save_formset attribution stamping give staff their first write path for observation-record attributions; telescope_class joins the public non-staff surface while source stays staff-only; and a calendar-event-modal template override links an event back to its owning, publicly-visible run.
- Every CSV-imported CampaignRun now records source=csv_import and a derived telescope_class through the one shared calendar_utils.derive_telescope_class() helper; the paired demo notebook and operator runbook were updated and regenerated to match; and the phase's three folded planning-doc corrections (a stale Phase 25 claim, a pre-domain-correction framing note, and a falsified 26-DECISION premise) landed as date-pinned/forward-pointing edits rather than silent rewrites.
- Reordered the approval queue's Sites Needing Review card to render first, and added a staff-only "Possible campaign run match" hint in the calendar-event modal for unlinked events with a HIGH-band attribution candidate.
- Converted three broken multi-line `{# #}` Django comment blocks in `event_form.html` to real `{% comment %}` blocks, gated the run-window render on `window_start` so a TBD run no longer prints `(None-None)` in the public calendar modal, and added a permanent repo-wide sweep test plus four render-level regression tests to make the defect class un-recur.
- CampaignRun/CalendarEventMeta gained discriminating `__str__` labels (verified 44/44 and 11/11 distinct against the live dev DB), a searchable `CalendarEventMetaAdmin.run` autocomplete picker, and an instance-level admin lock that makes `source` non-overwritable on an already-approved WEB run.
- Staff can now reach the "Sites Needing Review" queue from the campaign list whenever it has rows, even with zero pending submissions -- via a single shared `runs_needing_site_review()` definition and a nested, precedence-safe `{% if %}` gate.
- `import_campaign_csv` now preserves a resolved site and telescope_class across a re-import whose CSV cell doesn't itself resolve, closing the WR-01 operational data-integrity risk from Phase 27, with 8 new regression tests, updated runbook/notebook documentation, and a corrected WR-02 verification entry.
- Widened `CampaignRunAdmin.get_readonly_fields`'s source-provenance lock from `(APPROVED, WEB)` to every `source == WEB` row at any approval status, closing the two-step edit-then-approve bypass WR-03 identified and completing Phase 27.1 success criterion 6.
- Two typed per-pair dismissal models (CalendarEventDismissal, ObservationRecordDismissal), CalendarEventMeta.confirmed_by/confirmed_at audit fields, and admin-side stamping on a genuine run-link transition -- the schema every later plan in this phase reads or writes.
- A new peer module (`campaign_attribution.py`) computing scored, evidence-carrying `(orphan, CampaignRun)` candidates via a pure weighted sum over date-overlap/instrument-similarity/telescope-match, gated by a single campaign/target boundary hard gate, with the criterion-5 `FTS/MuSCAT4` vs `2M0-SCICAM-MUSCAT` pair proven to land in the High band.
- AttributionDecisionView's five POST actions (confirm / confirm_selected / dismiss / undo_confirmation / undo_dismissal), each re-deriving eligibility server-side via the Phase 28-02 matcher before writing, plus AttributionQueueView's GET context assembly and the campaign-list banner count -- proven by 793 passing tests (full solsys_code suite, excluding the pre-existing test_views.TestEphemeris ASSIST segfault).
- Staff attribution queue page with evidence-column worklists, confidence-band badges, a D-09 checkbox gate, a campaign-list banner, and the operator runbook's attribution-pass section.
- Closed both BLOCKER gaps from 28-VERIFICATION.md: the rendered Confirm button now submits without a dismissal reason (formnovalidate), and the standalone CalendarEventMeta admin page now stamps confirmed_by/confirmed_at server-side and refuses hand-typed values, exactly as the inline path already did.
- Gated `_undo_confirmation()`'s dismissal write on the link-clearing write's `changed_count`, and rebound `sole_high_candidate_pk` to the full uncapped candidate list in both backlog builders — closing 28-VERIFICATION.md's WR-01/WR-02 anti-patterns and correcting the IN-01 docstring/implementation mismatch, with no behavior change on IN-01.
- `campaign_reconciler.py`'s `reconcile_run()` -- one idempotent per-run function projecting queue/class-wide/satellite runs to a single bare `RUN:{pk}` container and classical runs to per-night `RUN:{pk}:{date}` events, with a RECON-05 ownership guard and RECON-06 dry-run support.
- `_adopted_event_for_night()` re-keys a `load_telescope_runs`-created classical event already attributed to a run (via `CalendarEventMeta.run`) to `RUN:{pk}:{date}` in place instead of minting a duplicate, and the classical/queue branches are now proven end-to-end by 15 new tests.
- `reconcile_campaign_runs` -- the single idempotent sweep that loops `reconcile_run()` over every `CampaignRun`, retiring the backfill-command-per-gap pattern, with `--dry-run` parity and per-run failure isolation proven by 4 command-level tests.
- `campaign_views.py`'s four staff actions (approve/resolve_site/mark_cancelled/mark_weather_failure) now call `campaign_reconciler.reconcile_run()` exclusively; the retired `_project_calendar_event()`/`_calendar_event_title()` projection code and the `backfill_range_calendar_events` command are deleted, and the approval-queue test suite (124 tests) is rewritten onto `RUN:` keys and the reconciler's own title/prefix builders.
- Rewrote `docs/runbooks/telescope_runs_calendar.rst` to retire every trace of `backfill_range_calendar_events` and document `reconcile_campaign_runs` in its place, corrected the "Campaign run block" section's now-superseded manual-only claim, and shipped a new pre-executed `reconcile_campaign_runs_demo.ipynb` proving the dry-run/real-sweep/idempotency contract end-to-end.
- A REJECTED CampaignRun is excluded from `_eligible_runs_for_event`/`_eligible_runs_for_record` via one named `frozenset` constant, pinned by 8 new tests and demonstrated in the campaign-lifecycle demo notebook's real executed output.
- Pinned ruff's dev-extra dependency and CLAUDE.md's documented lint/format command to the exact version pre-commit already enforces, closing the phantom drift Phases 26/27/27.1 each independently logged, and repaired two cosmetic bookkeeping items (stale docstring references, a stuck-in-progress decision-doc header) flagged by the milestone audit.
- Closed 27-REVIEW WR-01's remaining half by mirroring `preserve_site`'s exact shape for `telescope_class`: a re-import can no longer silently replace a hand-corrected class with a different derived one, every such firing is named on stderr and counted as `telescope_class_preserved` in the summary, and the runbook's re-import gotcha note now says so too.
- All five v2.2 phase `VALIDATION.md` files (26, 27, 27.1, 28, 29) are now `status: validated`/`nyquist_compliant: true`, and `.planning/v2.2-MILESTONE-AUDIT.md` records every tech-debt item's true disposition — including the ruff-drift misdiagnosis, three items closed by this Phase 30, two already closed before it, and the D-11 roadmap-correction note.

Known verification overrides: 20 newly acknowledged, 0 carried forward from a prior close (see STATE.md Deferred Items). All 6 phases (26, 27, 27.1, 28, 29, 30) verified `phase_complete`/`passed`; all 24 v2.2 requirements checked off — the acknowledged items were 11 completed-but-unarchived quick tasks, 1 deliberately deferred todo, 2 dormant seeds, and 6 historical `deferred-items.md` log entries recording a ruff/format drift that Phase 30 (30-02) root-caused and fixed (it was a phantom drift from an unpinned dev ruff version, not dirty code).

---

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
