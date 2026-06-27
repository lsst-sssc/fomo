# Milestones

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
- `load_telescope_runs` Django management command expands parsed run date ranges into idempotent nightly `CalendarEvent`s using `sun_event()` for accurate UTC sunset/sunrise — upsert via `get_or_create` keyed on `(telescope, instrument, start_time)` with conditional save.
- 6-test `TestLoadTelescopeRuns` suite covers INGEST-01/02/03 plus per-line error handling and no-churn idempotency; all 95 `./manage.py test solsys_code` tests pass.
- 6/6 UAT scenarios confirmed live on dev DB; demo notebook `load_telescope_runs_demo.ipynb` confirmed executable end-to-end.

---

## 1.0 Site/Ephemeris Helper (Shipped: 2026-06-14)

**Phases completed:** 1 phases, 2 plans, 4 tasks

**Key accomplishments:**

- Observatory model gains a timezone field and to_earth_location(), migration 0002 seeds 4 telescope sites (Magellan-Clay/Baade, NTT, FTS), and a new telescope_runs.py computes dip-corrected sunset/sunrise (-(0.833+dip)) and -15deg dark-window UTC crossing times via astropy get_sun/AltAz with coarse-scan + bisection root-finding.
- Extended test_telescope_runs.py with skycalc-accuracy validation for 4 June 2026 Las Campanas nights, a -18deg astronomical-twilight cross-check matching 19:16/06:08 Santiago local to the second, and zoneinfo DST-offset tests for Santiago/Sydney - all passing with ruff check/format clean.

---
