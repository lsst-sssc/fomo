# Requirements: Telescope Runs Calendar — Stage 1 (Site/Ephemeris Helper)

**Defined:** 2026-06-12
**Core Value:** Stage 1's ephemeris helper must produce sun-event times within 2 minutes of Las Campanas skycalc, sourced via the `Observatory` model — and be built end-to-end through GSD's discuss/plan/execute/verify loop.

## v1 Requirements

### Site Registry

- [x] **SITE-01**: A `SITES` lookup resolves a telescope name (Magellan, NTT, FTS) to its `Observatory` record (by MPC obscode) and constructs an `astropy.coordinates.EarthLocation` from that record's lat/lon/altitude
- [x] **SITE-02**: The `SITES` lookup provides the correct IANA timezone for each site (`America/Santiago` for Magellan/NTT, `Australia/Sydney` for FTS)
- [x] **SITE-03**: `Observatory` records exist (via the existing CreateObservatory form) for Magellan (Las Campanas), NTT (La Silla), and FTS (Siding Spring) with correct MPC obscodes and lat/lon/altitude

### Ephemeris

- [x] **EPHEM-01**: `sun_event(site, date, kind)` returns UTC sunset and sunrise times with the refraction+semidiameter correction (-0.833°) and altitude-dependent horizon-dip correction (`dip = 1.76' * sqrt(h_metres)`) applied
- [x] **EPHEM-02**: `sun_event(site, date, 'dark')` returns UTC crossing times for the -15° dark window
- [x] **EPHEM-03**: The horizon-dip helper returns 1.44° ± 0.02° for an altitude of 2402 m (Las Campanas)
- [x] **EPHEM-04**: Computed Las Campanas sunset/sunrise (dip-corrected) for June 2026 sample nights (1, 10, 20, 30) agree with Las Campanas skycalc to within 2 minutes
- [x] **EPHEM-05**: Computed astronomical twilight (-18°) for Las Campanas on 10 June 2026 agrees with skycalc's twi.end/twi.beg (19:16/06:08 local) to within 2 minutes
- [x] **EPHEM-06**: `America/Santiago` resolves to UTC-4 in June and UTC-3 in January; `Australia/Sydney` resolves to UTC+10 in July and UTC+11 in January

## v2 Requirements

None — this GSD run is intentionally scoped to Stage 1 only. Stages 2-4 are
tracked in `docs/design/telescope_runs_calendar.rst` and listed under Out of
Scope below; they would form a future GSD milestone if Stage 1 succeeds.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Stage 2 — `load_telescope_runs` classical ingest command | Deferred to a future GSD run; depends on Stage 1 |
| Stage 3 — FTS/MuSCAT4 queue window banners | Deferred; FTS queue input format is still an open item in the design doc |
| Stage 4 — observation-record sync to calendar | Deferred; depends on Stages 1-3 |
| `tom_calendar` UI/template changes or DB migrations | Not needed — Stage 1 is a pure helper module |
| Reworking `tom_observations`' astroplan-based visibility/airmass plots | Confirmed separate concern (different purpose/horizon convention); not touched |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SITE-01 | Phase 1 | Complete |
| SITE-02 | Phase 1 | Complete |
| SITE-03 | Phase 1 | Complete |
| EPHEM-01 | Phase 1 | Complete |
| EPHEM-02 | Phase 1 | Complete |
| EPHEM-03 | Phase 1 | Complete |
| EPHEM-04 | Phase 1 | Complete |
| EPHEM-05 | Phase 1 | Complete |
| EPHEM-06 | Phase 1 | Complete |

**Coverage:**

- v1 requirements: 9 total
- Mapped to phases: 9
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-12*
*Last updated: 2026-06-12 after initial definition*
