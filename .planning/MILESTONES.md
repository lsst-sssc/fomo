# Milestones

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
