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

See [milestones/v1.3-ROADMAP.md](milestones/v1.3-ROADMAP.md) for full Phase 5-7 goal/requirements/success-criteria detail (current milestone). Completed-milestone phase detail lives in their respective `milestones/*-ROADMAP.md` files linked above.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Site & Ephemeris Helper | v1.0 | 2/2 | Complete | 2026-06-12 |
| 2. Run Line Parsing | v1.1 | 1/1 | Complete | 2026-06-14 |
| 3. Classical Calendar Ingest | v1.1 | 2/2 | Complete | 2026-06-16 |
| 4. LCO Queue Sync Command | v1.2 | 1/1 | Complete | 2026-06-17 |
| 5. Multi-Proposal & Multi-Facility Selection | v1.3 | 0/TBD | Not started | - |
| 6. Correct Instrument-Type Extraction | v1.3 | 0/TBD | Not started | - |
| 7. Live Telescope-Label Resolution with Fallback & Failure Reporting | v1.3 | 0/TBD | Not started | - |
