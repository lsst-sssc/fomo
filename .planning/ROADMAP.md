# Roadmap: Telescope Runs Calendar

## Milestones

- ✅ **v1.0 Site/Ephemeris Helper** — Phase 1 (shipped 2026-06-14) — see [milestones/1.0-ROADMAP.md](milestones/1.0-ROADMAP.md)
- ✅ **v1.1 Classical Run Ingest** — Phases 2-3 (shipped 2026-06-16) — see [milestones/v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 LCO Queue Calendar Sync** — Phase 4 (shipped 2026-06-18) — see [milestones/v1.2-ROADMAP.md](milestones/v1.2-ROADMAP.md)
- 🔄 **v1.3 Full LCO Facility Sync** — Phases 5-7 (in progress) — see [milestones/v1.3-ROADMAP.md](milestones/v1.3-ROADMAP.md)

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

<details>
<summary>✅ v1.2 LCO Queue Calendar Sync (Phase 4) — SHIPPED 2026-06-18</summary>

- [x] Phase 4: LCO Queue Sync Command (1/1 plans) — completed 2026-06-17

</details>

### v1.3 Full LCO Facility Sync

- [ ] **Phase 5: Multi-Proposal & Multi-Facility Selection** - `sync_lco_observation_calendar` syncs any combination of proposals (or `ALL`) across both LCO and SOAR records, each authenticated against its own facility instance
- [ ] **Phase 6: Correct Instrument-Type Extraction** - The command extracts the real instrument type from multi-configuration parameter shapes, never mistaking a calibration config for the science one
- [ ] **Phase 7: Live Telescope-Label Resolution with Fallback & Failure Reporting** - Placed records get a verified site/telescope label via per-record API call, falling back to a coarse instrument-class label (with visible, counted, non-fatal degrade) when that call fails

## Phase Details

### Phase 5: Multi-Proposal & Multi-Facility Selection

**Goal**: An operator can run `sync_lco_observation_calendar` against any combination of proposals (or the whole network) across both LCO and SOAR records in one invocation, with each record authenticated against the correct facility
**Depends on**: Phase 4 (existing single-proposal, LCO-only command)
**Requirements**: SELECT-02, SELECT-03, SELECT-04, SELECT-05
**Success Criteria** (what must be TRUE):

  1. Running with `--proposal A,B,C` syncs `ObservationRecord`s matching any of the 3 codes, and does not match on partial/single-character substrings of those codes
  2. Running with `--proposal ALL` syncs every LCO-family (`LCO` + `SOAR`) `ObservationRecord` regardless of its proposal code
  3. A single run produces correct `CalendarEvent`s for both `facility='LCO'` and `facility='SOAR'` records together, without requiring two separate invocations
  4. A SOAR record is processed using a `SOARFacility`/SOAR-credentialed instance, never a reused `LCOFacility()` instance — verified by a test asserting the per-facility instance dict dispatches by each record's own `facility` value**Plans**: 1 plan
- [ ] 05-01-PLAN.md — Generalize sync_lco_observation_calendar to multi-proposal (comma-list/ALL) + multi-facility (LCO+SOAR) per-record dispatch

### Phase 6: Correct Instrument-Type Extraction

**Goal**: The command always identifies the scientifically meaningful instrument configuration for a record, regardless of which LCO-family facility submitted it
**Depends on**: Phase 5 (queryset already covers LCO + SOAR, so both shapes are exercised together)
**Requirements**: EXTRACT-01, EXTRACT-02
**Success Criteria** (what must be TRUE):

  1. For a record with a single populated `c_N_instrument_type`/exposure-time config (today's real LCO data shape), the extracted instrument type matches that config — unchanged from today's correct cases
  2. For a SOAR-shaped record with multiple populated configs (e.g. spectrum + arc + lamp-flat), the extracted instrument type is the science spectrum config, never the arc or lamp-flat calibration config
  3. For an LCO MUSCAT-shaped record with per-channel (`_g`/`_r`/`_i`/`_z`) exposure keys, the extracted instrument type correctly reflects the populated channel(s) instead of returning nothing or raising

**Plans**: TBD

### Phase 7: Live Telescope-Label Resolution with Fallback & Failure Reporting

**Goal**: Every synced record gets a telescope label — the verified, API-resolved one when possible, a clearly-marked coarse fallback when not — and a degraded API call never aborts the run, hides its own failure, or leaks credentials
**Depends on**: Phase 6 (fallback label is derived from the now-correct instrument type)
**Requirements**: TELESCOPE-01, TELESCOPE-02, TELESCOPE-03, TELESCOPE-04, SYNC-06, SYNC-07, SYNC-08, SYNC-09
**Success Criteria** (what must be TRUE):

  1. For a placed (scheduled) record, the command calls the LCO API for that record's actual site/enclosure/telescope and maps the fully-qualified code through a verified static dict covering all real LCO-network sites, producing the correct telescope label
  2. When that per-record API call fails, times out, or returns a code absent from the verified dict, the record still gets a `CalendarEvent` — labeled with a coarse instrument-class token (`1m0`/`0m4`/`2m0`) and a description noting the API lookup failed — and the run continues to the next record
  3. The run's summary line reports a fallback/API-failure count distinct from the existing `skipped_count`, so an operator can see how many records got a degraded label without it being conflated with hard skips
  4. The per-record API call uses an explicit timeout and makes a single attempt (no retry loop) — confirmed by a test that mocks a slow/failing response and asserts no second call is made
  5. No logged error or exception message from a failed API call contains the raw response body or API key/credential content — confirmed by a test asserting the log output for a simulated failure is a fixed, generic message

**Plans**: TBD

Full detail also lives in [milestones/v1.3-ROADMAP.md](milestones/v1.3-ROADMAP.md) (current milestone). Completed-milestone phase detail lives in their respective `milestones/*-ROADMAP.md` files linked above.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Site & Ephemeris Helper | v1.0 | 2/2 | Complete | 2026-06-12 |
| 2. Run Line Parsing | v1.1 | 1/1 | Complete | 2026-06-14 |
| 3. Classical Calendar Ingest | v1.1 | 2/2 | Complete | 2026-06-16 |
| 4. LCO Queue Sync Command | v1.2 | 1/1 | Complete | 2026-06-17 |
| 5. Multi-Proposal & Multi-Facility Selection | v1.3 | 0/1 | Planned | - |
| 6. Correct Instrument-Type Extraction | v1.3 | 0/TBD | Not started | - |
| 7. Live Telescope-Label Resolution with Fallback & Failure Reporting | v1.3 | 0/TBD | Not started | - |
