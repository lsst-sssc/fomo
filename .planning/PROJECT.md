# Telescope Runs Calendar — Stages 1, 2 & 3

## What This Is

A helper module and management commands for FOMO that:

1. (`solsys_code/telescope_runs.py`) resolves a telescope name to its observing site (via the `Observatory` model, by MPC obscode) and computes dip-corrected UTC sunset, sunrise, and -15° dark-window crossing times for a given date — Stage 1.
2. (`solsys_code/management/commands/load_telescope_runs.py`) parses classical-schedule run lines and idempotently creates one `tom_calendar.CalendarEvent` per observing night, populated with sunset/sunrise times and the -15° dark window — Stage 2.
3. (`solsys_code/management/commands/sync_lco_observation_calendar.py`) syncs LCO queue ObservationRecords (FTS/MuSCAT4) to the calendar as unified CalendarEvents — starting as a scheduling-window banner and updating in place to the placed block once the LCO scheduler acts — Stage 3.
4. A visual clarity layer (v1.4): `CalendarEventTelescopeLabel` sidecar model records live-verified vs. fallback telescope-label resolution; `calendar_display_extras` template-tag library provides proposal-keyed color (sha256 → 8-color colorblind-vetted palette), status box-shadow rings, and a click-to-filter legend.
5. (`solsys_code/management/commands/sync_gemini_observation_calendar.py`) syncs submitted Gemini ToO `ObservationRecord`s to `CalendarEvent` window banners — using explicit `windowDate`/`windowTime`/`windowDuration` parameters when present and ToO-type-derived defaults (`Rap:` +24 h; `Std:` +24 h to +7 d) when not — with per-record credential scrubbing, idempotent no-churn find-or-create, and a pre-executed demo notebook — Stage 3b (v1.5).
6. (`solsys_code/calendar_utils.py`) shared utility module holding `SITE_TELESCOPE_MAP`, `_extract_instrument`, `insert_or_create_calendar_event()`, and related helpers — extracted from `sync_lco_observation_calendar.py` in v1.6 so all three management commands share one canonical implementation. Calendar event title text is now WCAG-AA-compliant against all palette backgrounds (`text_color_for_bg`), and `CalendarEventTelescopeLabel` data is loaded in a single prefetch query rather than one per event (`fomo_render_calendar` wrapper view, v1.6).

This is a Stages-1-through-3b-complete implementation of the "telescope runs on the calendar" feature (issue #37), with a calendar visual clarity layer added in v1.4, Gemini ToO calendar sync added in v1.5, and shared-utility refactoring + WCAG/N+1 polish added in v1.6. Stage 4 (full observation-record sync for all facilities) remains future work.

## Current State

**Shipped:**
- ✅ v1.0 "Site/Ephemeris Helper" — 2026-06-14 (Phase 1)
- ✅ v1.1 "Classical Run Ingest" — 2026-06-16 (Phases 2-3)
- ✅ v1.2 "LCO Queue Calendar Sync" — 2026-06-17 (Phase 4)
- ✅ v1.3 "Full LCO Facility Sync" — 2026-06-24 (Phases 5-7, 07.1) — multi-proposal/multi-facility, correct instrument extraction, live telescope-label resolution + facility-aware coarse fallback
- ✅ v1.4 "Calendar Visual Clarity" — 2026-06-26 (Phases 8-9) — `CalendarEventTelescopeLabel` sidecar model, dashed-border + tooltip for fallback labels, proposal-keyed color palette, status box-shadow rings, `[QUEUED]` override fix, click-to-filter legend
- ✅ v1.5 "Gemini Calendar Sync" — 2026-06-27 (Phase 10) — `sync_gemini_observation_calendar` management command syncing Gemini ToO ObservationRecords to CalendarEvent window banners with per-record password scrubbing, ToO-type window derivation, and no-churn idempotency
- ✅ v1.6 "Tech Debt & Display Polish" — 2026-06-29 (Phases 11-12) — `calendar_utils.py` shared utility module with `SITE_TELESCOPE_MAP`/`_extract_instrument`/`insert_or_create_calendar_event` (REFAC-01/02); `text_color_for_bg` WCAG template tag (DISPLAY-08); `fomo_render_calendar` wrapper view eliminating N+1 query (DISPLAY-09)

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
- **All 194 `./manage.py test solsys_code` tests pass (Phase 12 complete).**

## Core Value

Stage 1 (v1.0): Sun-event times accurate to within 2 minutes of the Las Campanas skycalc reference tool — the foundation that Stages 2-4 build on. Also: validated the GSD discuss→plan→execute→verify loop end-to-end on this codebase.

Stage 2 (v1.1): A `load_telescope_runs` management command turns classical-schedule run lines into accurate, idempotent `tom_calendar.CalendarEvent`s — one per observing night — using Stage 1's `SITES`/`get_site()`/`sun_event()` for sunset/sunrise times.

Stage 3 (v1.2): A `sync_lco_observation_calendar` management command syncs LCO queue ObservationRecords (FTS/MuSCAT4) to the calendar — one CalendarEvent per record, keyed on the LCO portal URL, transitioning from a scheduling-window banner (`parameters['start'`/`'end']`) to a placed block (`scheduled_start`/`scheduled_end`) as the scheduler acts, and updating in place if the block is rescheduled.

## Current Milestone: v1.7 ESO/VLT Calendar Sync

**Goal:** Add ESO/VLT ObservationRecord sync to the calendar, closing the last unhandled configured facility (`tom_eso.eso.ESOFacility`) and completing Stage 4 of issue #37.

**Target features:**
- `sync_eso_observation_calendar` management command following the LCO/Gemini pattern (one `CalendarEvent` per synced record, idempotent no-churn create-or-update)
- Scope refined by research into what `tom_eso`'s `ESOFacility` actually exposes for OB (Observation Block) execution/status data (Phase 2 status) — this is not yet confirmed and may constrain what "synced" can mean for ESO records

**Prior milestones (v1.0-v1.6):**

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

### Active

<!-- No active requirements — v1.6 milestone complete -->

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
*Last updated: 2026-06-29 after Phase 12 — v1.6 Tech Debt & Display Polish complete*
