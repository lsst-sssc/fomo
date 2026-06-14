# Requirements: Telescope Runs Calendar — Stage 2 (Classical Run Ingest)

**Defined:** 2026-06-13
**Core Value:** A `load_telescope_runs` management command turns classical-schedule run lines into accurate, idempotent `tom_calendar.CalendarEvent`s — one per observing night — using Stage 1's `telescope_runs.SITES`/`get_site()`/`sun_event()` for sunset/sunrise times.

## v1 Requirements

### Run Line Parsing

- [x] **PARSE-01**: Parse a classical run line (e.g. `NTT EFOSC2 allocation 9-13 July`, `Magellan IMACS 13-19 July (proposed)`, `Magellan Proto-Lightspeed Jul 8-12 (proposed)`) into `(telescope, instrument, status, year, month, day1, day2)`, handling both month-before-range (`9-13 July`) and month-after-range (`Jul 8-12`) date orderings
- [x] **PARSE-02**: Hyphenated instrument names (e.g. `Proto-Lightspeed`) parse correctly as a single instrument token
- [x] **PARSE-03**: A run line with no year defaults to the current year; a run starting in late December rolls into the next calendar year

### Classical Ingest

- [ ] **INGEST-01**: `load_telescope_runs` management command expands a parsed run `S..E` into `E - S + 1` nightly `CalendarEvent`s (one per evening date `d`, `start_time = sunset(d)`, `end_time = sunrise(d+1)`), using `telescope_runs.get_site()`/`sun_event()`
- [ ] **INGEST-02**: Each created event sets `telescope`/`instrument` from the parsed line, with a glanceable `title` and a `description` containing the -15° dark window times plus the original run line text
- [ ] **INGEST-03**: Running the command twice on the same input file does not create duplicate `CalendarEvent`s (idempotent)

## v2 Requirements

None — this milestone is intentionally scoped to Stage 2 only. Stages 3-4 are
tracked in `docs/design/telescope_runs_calendar.rst` and listed under Out of
Scope below; they would form a future milestone if Stage 2 succeeds.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Stage 3 — FTS/MuSCAT4 queue window banners | Deferred; FTS queue input format is still an open item in the design doc |
| Stage 4 — observation-record sync to calendar | Deferred; depends on Stages 1-3 |
| `tom_calendar` UI/template changes or DB migrations | Not needed — `CalendarEvent` already has the required fields (per design doc) |
| Distinguishing Magellan Baade vs Clay in `telescope` | Open item from design doc; ephemeris is identical, both at Las Campanas |
| Replacing `SITES`'s hardcoded telescope-name -> obscode mapping with a data-driven `Observatory.short_name` lookup | Flagged as a Stage 2+ consideration in the design doc, but not required for Stage 2's success criteria; may be picked up if it blocks parsing new telescope names |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PARSE-01 | Phase 2 | Complete |
| PARSE-02 | Phase 2 | Complete |
| PARSE-03 | Phase 2 | Complete |
| INGEST-01 | Phase 3 | Pending |
| INGEST-02 | Phase 3 | Pending |
| INGEST-03 | Phase 3 | Pending |
