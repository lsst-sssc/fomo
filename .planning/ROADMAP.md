# Roadmap: Telescope Runs Calendar

## Milestones

- ✅ **v1.0 Site/Ephemeris Helper** — Phase 1 (shipped 2026-06-14) — see [milestones/1.0-ROADMAP.md](milestones/1.0-ROADMAP.md)
- ✅ **v1.1 Classical Run Ingest** — Phases 2-3 (shipped 2026-06-16) — see [milestones/v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)
- 🔄 **v1.2 LCO Queue Calendar Sync** — Phase 4 (in progress)

## Phases

<details>
<summary>✅ v1.0 Site/Ephemeris Helper (Phase 1) — SHIPPED 2026-06-14</summary>

- [x] Phase 1: Site & Ephemeris Helper (2/2 plans) — completed 2026-06-12

</details>

<details>
<summary>✅ v1.1 Classical Run Ingest (Phases 2-3) — SHIPPED 2026-06-16</summary>

- [x] Phase 2: Run Line Parsing (1/1 plans) — completed 2026-06-14
- [x] Phase 3: Classical Calendar Ingest (2/2 plans) — completed 2026-06-16

</details>

### v1.2 LCO Queue Calendar Sync

- [ ] **Phase 4: LCO Queue Sync Command** - Management command syncs FTS/MuSCAT4 ObservationRecords to CalendarEvents with banner, placed-block, and terminal-state handling

## Phase Details

### Phase 4: LCO Queue Sync Command

**Goal**: Users can run `sync_lco_observation_calendar --proposal <code>` to sync FTS/MuSCAT4 queue records to the FOMO calendar, with each ObservationRecord represented as a CalendarEvent that transitions from an unscheduled banner to a placed block as the LCO scheduler acts, and is marked with a status prefix on reaching a terminal state
**Depends on**: Phase 3 (CalendarEvent model and DB infrastructure established)
**Requirements**: SELECT-01, SYNC-01, SYNC-02, SYNC-03, SYNC-04, SYNC-05, TERM-01
**Success Criteria** (what must be TRUE):

  1. Running the command with `--proposal PROPOSAL2025A-001` produces one CalendarEvent per matching LCO ObservationRecord, with `url` set to `https://observe.lco.global/requestgroups/<id>/`, `telescope` and `instrument` populated from parameters, and no CalendarEvent created for non-matching records
  2. For an unscheduled record (`scheduled_start` is None), the CalendarEvent's `start_time`/`end_time` match `parameters['start']`/`parameters['end']` and the title indicates queue/unscheduled status
  3. After the LCO scheduler places the block (populating `scheduled_start`/`scheduled_end`), re-running the command updates the existing CalendarEvent's times to the placed values — no new event is created and `modified` is not updated for records whose data has not changed
  4. For a record in a terminal state (WINDOW_EXPIRED, CANCELED, FAILURE_LIMIT_REACHED, NOT_ATTEMPTED), the CalendarEvent title is prefixed with `[EXPIRED]`, `[CANCELLED]`, or `[FAILED]` and the event is retained (not deleted)
  5. All `./manage.py test solsys_code` tests pass (including new tests for SYNC-01 through SYNC-05 and TERM-01) and `ruff check .` / `ruff format --check .` are clean

> **Planning note (D-01):** Success criterion 1's literal `url` format `https://observe.lco.global/requestgroups/<id>/` is corrected by CONTEXT.md D-01 — the real `LCOFacility().get_observation_url()` returns `https://observe.lco.global/requests/<id>` (no trailing slash, `/requests/`). The upsert-by-`url` behavior is unchanged; only the concrete string differs.

> **Planning note (D-06, research correction):** `get_terminal_observing_states()` returns 5 states (the 4 failure states + `COMPLETED`). TERM-01's prefix table only covers the 4 failure states. Locked decision: use `get_failed_observing_states()` as the prefix trigger; `COMPLETED` records get a clean (no-prefix) title.
**Plans**: 1 plan

- [ ] 04-01-PLAN.md — `sync_lco_observation_calendar` command: `--proposal` selection, banner→placed CalendarEvent upsert keyed on `url`, terminal-state title prefixes, no-churn idempotency

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Site & Ephemeris Helper | v1.0 | 2/2 | Complete | 2026-06-12 |
| 2. Run Line Parsing | v1.1 | 1/1 | Complete | 2026-06-14 |
| 3. Classical Calendar Ingest | v1.1 | 2/2 | Complete | 2026-06-16 |
| 4. LCO Queue Sync Command | v1.2 | 0/1 | Not started | - |
