# Roadmap: Telescope Runs Calendar — Stage 1 (Site/Ephemeris Helper)

## Overview

A single, self-contained phase delivers `solsys_code/telescope_runs.py`: a
`SITES` registry that resolves Magellan/NTT/FTS to `Observatory`-backed
`EarthLocation` + timezone data, and a `sun_event()` function that computes
dip-corrected sunset/sunrise and -15° dark-window crossings, validated against
LCO skycalc. Given the small, cohesive scope (one module, 9 tightly coupled
requirements, coarse granularity), this ships as one phase.

## Phases

- [x] **Phase 1: Site & Ephemeris Helper** - Resolve telescope sites via `Observatory` and compute dip-corrected sun-event times validated against skycalc (completed 2026-06-12)

## Phase Details

### Phase 1: Site & Ephemeris Helper

**Goal**: Given a telescope name (Magellan, NTT, FTS) and a date, the system can resolve the observing site (location + timezone) from the `Observatory` model and compute accurate UTC sunset, sunrise, and -15° dark-window times, dip-corrected for site altitude and validated against LCO skycalc.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: SITE-01, SITE-02, SITE-03, EPHEM-01, EPHEM-02, EPHEM-03, EPHEM-04, EPHEM-05, EPHEM-06
**Success Criteria** (what must be TRUE):

  1. `SITES` resolves "Magellan", "NTT", and "FTS" to an `Observatory`-backed `astropy.coordinates.EarthLocation` (lat/lon/altitude from the matching MPC-obscode `Observatory` record) plus the correct IANA timezone (`America/Santiago` for Magellan/NTT, `Australia/Sydney` for FTS)
  2. `Observatory` records for Magellan (Las Campanas), NTT (La Silla), and FTS (Siding Spring) exist with correct MPC obscodes and lat/lon/altitude, created via the existing CreateObservatory form
  3. A horizon-dip helper returns 1.44° ± 0.02° for an altitude of 2402 m (Las Campanas)
  4. `sun_event(site, date, 'sun')` (or equivalent) returns dip- and refraction/semidiameter-corrected UTC sunset/sunrise for Las Campanas on June 2026 sample nights (1, 10, 20, 30) that agree with LCO skycalc to within 2 minutes
  5. `sun_event(site, date, 'dark')` returns the -15° dark-window crossing times, and the -18° astronomical twilight crossings for Las Campanas on 10 June 2026 agree with skycalc's twi.end/twi.beg (19:16/06:08 local) to within 2 minutes; `America/Santiago`/`Australia/Sydney` resolve to the correct UTC offsets across their respective DST boundaries (UTC-4/-3 for Santiago in June/January, UTC+10/+11 for Sydney in July/January)

**Plans**: 2 plans
Plans:
**Wave 1**

- [x] 01-01-PLAN.md — Walking Skeleton: extend Observatory (timezone field + to_earth_location), seed migration, telescope_runs.py (SITES/get_site/horizon_dip/sun_event) + DB tests [SITE-01/02/03, EPHEM-01/02/03]

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — Validation slice: skycalc agreement for 4 June dates, -18° twilight cross-check, DST resolution, ruff gates [EPHEM-04/05/06]

## Progress

**Execution Order:**
Phase 1 only.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Site & Ephemeris Helper | 2/2 | Complete   | 2026-06-12 |
