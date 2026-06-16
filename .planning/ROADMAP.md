# Roadmap: Telescope Runs Calendar

## Milestones

- ✅ **v1.0 Site/Ephemeris Helper** — Phase 1 (shipped 2026-06-14) — see [milestones/1.0-ROADMAP.md](milestones/1.0-ROADMAP.md)
- **v1.1 Classical Run Ingest** — Phases 2-3 (in progress)

## Phases

<details>
<summary>✅ v1.0 Site/Ephemeris Helper (Phase 1) — SHIPPED 2026-06-14</summary>

- [x] Phase 1: Site & Ephemeris Helper (2/2 plans) — completed 2026-06-12

</details>

### v1.1 Classical Run Ingest

- [x] **Phase 2: Run Line Parsing** - Parse classical-schedule run lines into structured (telescope, instrument, status, year, month, day1, day2) tuples (completed 2026-06-14)
- [x] **Phase 3: Classical Calendar Ingest** - `load_telescope_runs` management command expands parsed runs into idempotent nightly CalendarEvents (completed 2026-06-16)

## Phase Details

### Phase 2: Run Line Parsing

**Goal**: A run-line parser turns free-text classical-schedule lines (the three sample formats) into structured tuples ready for calendar-event expansion.
**Depends on**: Phase 1 (Site & Ephemeris Helper) — provides `telescope_runs.SITES` telescope names this parser must recognize
**Requirements**: PARSE-01, PARSE-02, PARSE-03
**Success Criteria** (what must be TRUE):

  1. Parsing `NTT EFOSC2 allocation 9-13 July` and `Magellan IMACS 13-19 July (proposed)` (month-before-range) yields the correct `(telescope, instrument, status, year, month, day1, day2)` tuple
  2. Parsing `Magellan Proto-Lightspeed Jul 8-12 (proposed)` (month-after-range, hyphenated instrument) yields `instrument='Proto-Lightspeed'` and the correct date fields
  3. A run line with no year present defaults `year` to the current year
  4. A run line whose date range starts in late December produces `year` = current year + 1 (roll-over to next calendar year)

**Plans**: 1 plan
Plans:

- [x] 02-01-PLAN.md — Add ParsedRun dataclass + parse_run_line() parser to telescope_runs.py, with tests covering all 4 success criteria

### Phase 3: Classical Calendar Ingest

**Goal**: Running `load_telescope_runs` against a file of classical run lines populates the calendar with one accurate, idempotent `CalendarEvent` per observing night for each parsed run.
**Depends on**: Phase 2 (Run Line Parsing) — consumes its parsed tuples; also depends on Phase 1's `telescope_runs.get_site()`/`sun_event()` for sunset/sunrise times
**Requirements**: INGEST-01, INGEST-02, INGEST-03
**Success Criteria** (what must be TRUE):

  1. Ingesting `NTT EFOSC2 allocation 9-13 July` creates 5 `CalendarEvent`s — one per night, `E - S + 1` inclusive. (Per Phase 2's CONTEXT.md D-01, the two `Magellan ...` sample lines raise `ValueError` on bare-`Magellan` telescope-name ambiguity and are Phase 2 error-path fixtures, not Phase 3 ingest fixtures — they are excluded from this phase's numeric success criteria.)
  2. Each created event has `start_time` = dip-corrected sunset of its evening date and `end_time` = sunrise of the following morning (both UTC, `end_time > start_time`, duration 8-15 hours)
  3. Each event's `telescope`/`instrument` fields and a glanceable `title` are set from the parsed run line, and `description` contains both the -15° dark-window times and the original run-line text
  4. Running `load_telescope_runs` twice on the same input file produces no duplicate `CalendarEvent`s

**Plans**: 2 plansPlans:
**Wave 1**

- [x] 03-01-PLAN.md — Add `load_telescope_runs` management command (test-first) that expands parsed runs into idempotent nightly CalendarEvents (INGEST-01/02/03) — completed 2026-06-16

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 03-02-PLAN.md — Phase 03 demo notebook (pre_executed) showing the ingest command and resulting CalendarEvents (Definition of Done)

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|-----------------|--------|-----------|
| 1. Site & Ephemeris Helper | v1.0 | 2/2 | Complete | 2026-06-12 |
| 2. Run Line Parsing | v1.1 | 1/1 | Complete   | 2026-06-14 |
| 3. Classical Calendar Ingest | v1.1 | 2/2 | Complete   | 2026-06-16 |
