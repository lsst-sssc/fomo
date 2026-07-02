# Roadmap: Telescope Runs Calendar

## Milestones

- ✅ **v1.0 Site/Ephemeris Helper** — Phase 1 (shipped 2026-06-14) — see [milestones/1.0-ROADMAP.md](milestones/1.0-ROADMAP.md)
- ✅ **v1.1 Classical Run Ingest** — Phases 2-3 (shipped 2026-06-16) — see [milestones/v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 LCO Queue Calendar Sync** — Phase 4 (shipped 2026-06-18) — see [milestones/v1.2-ROADMAP.md](milestones/v1.2-ROADMAP.md)
- ✅ **v1.3 Full LCO Facility Sync** — Phases 5-7, 07.1 (shipped 2026-06-24) — see [milestones/v1.3-ROADMAP.md](milestones/v1.3-ROADMAP.md)
- ✅ **v1.4 Calendar Visual Clarity** — Phases 8-9 (shipped 2026-06-26) — see [milestones/v1.4-ROADMAP.md](milestones/v1.4-ROADMAP.md)
- ✅ **v1.5 Gemini Calendar Sync** — Phase 10 (shipped 2026-06-27) — see [milestones/v1.5-ROADMAP.md](milestones/v1.5-ROADMAP.md)
- ✅ **v1.6 Tech Debt & Display Polish** — Phases 11-12 (shipped 2026-06-29) — see [milestones/v1.6-ROADMAP.md](milestones/v1.6-ROADMAP.md)
- 🚧 **v1.7 ESO/VLT Calendar Sync — Feasibility Spike** — Phase 13 (in progress)

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

<details>
<summary>✅ v1.3 Full LCO Facility Sync (Phases 5-7, 07.1) — SHIPPED 2026-06-24</summary>

- [x] Phase 5: Multi-Proposal & Multi-Facility Selection (1/1 plans) — completed 2026-06-19
- [x] Phase 6: Correct Instrument-Type Extraction (1/1 plans) — completed 2026-06-21
- [x] Phase 7: Live Telescope-Label Resolution with Fallback & Failure Reporting (2/2 plans) — completed 2026-06-24
- [x] Phase 07.1: Close gap: TELESCOPE-03/04/SYNC-06 — SOAR fallback label is facility-unaware (INSERTED) (1/1 plans) — completed 2026-06-24

</details>

<details>
<summary>✅ v1.4 Calendar Visual Clarity (Phases 8-9) — SHIPPED 2026-06-26</summary>

- [x] Phase 8: Telescope Label Verification Sidecar (2/2 plans) — completed 2026-06-25
- [x] Phase 9: Proposal Color & Status Visual Treatment (2/2 plans) — completed 2026-06-26

</details>

<details>
<summary>✅ v1.5 Gemini Calendar Sync (Phase 10) — SHIPPED 2026-06-27</summary>

- [x] Phase 10: Gemini Calendar Sync Command (2/2 plans) — completed 2026-06-27

</details>

<details>
<summary>✅ v1.6 Tech Debt & Display Polish (Phases 11-12) — SHIPPED 2026-06-29</summary>

- [x] Phase 11: Code Refactoring (2/2 plans) — completed 2026-06-27
- [x] Phase 12: Display Polish (1/1 plans) — completed 2026-06-28

</details>

### 🚧 v1.7 ESO/VLT Calendar Sync — Feasibility Spike (In Progress)

**Milestone Goal:** Determine whether/how ESO/VLT observation sync can work at all, given that the installed `tom_eso==0.2.4` cannot create `ObservationRecord` rows or report status through the standard TOM facility API. Produce a written Bridge / Bypass / Not-Yet-Feasible decision doc grounded in the real ESO P2 API. This is investigation-only — no `sync_eso_observation_calendar` command is built in this milestone.

- [x] **Phase 13: ESO Feasibility Spike** - Investigate ESO P2 credentials + real API data shapes and produce a Bridge/Bypass/Not-Yet-Feasible decision doc (completed 2026-07-02)

## Phase Details

### Phase 13: ESO Feasibility Spike

**Goal**: Answer "can ESO/VLT observation sync work at all, and if so how?" by probing the real ESO P2 API for OB status/execution data and the headless-credential situation, then writing a decision doc recommending Bridge, Bypass, or Not Yet Feasible. No sync command is implemented this milestone; the deliverable is a written, evidence-grounded recommendation that seeds a future milestone's requirements.
**Depends on**: Phase 12 (v1.6 complete)
**Requirements**: ESO-01, ESO-02, ESO-03, ESO-04, ESO-05
**Success Criteria** (what must be TRUE):

  1. The decision doc states explicitly whether valid ESO P2 API credentials (production, demo/sandbox, or a captured fixture) were obtainable and usable for this investigation, with the supporting evidence (ESO-01).
  2. At least one real `p2api` response for an OB's status/execution data is captured verbatim in the doc — an actual `obStatus` value and/or a `getOBExecutions()`/`getNightExecutions()` response — documented, not guessed. If credentials proved unobtainable, the doc records that blocker explicitly and captures the closest available reference shape (ESO-02).
  3. The doc states a viable (or explicitly non-viable) credential-sourcing path for a headless Django management command, accounting for ESO's per-user, session-bound, Fernet-encrypted `ESOProfile` auth and the absent `FACILITIES['ESO']` settings fallback (ESO-03).
  4. The decision doc explicitly recommends exactly one of Bridge / Bypass / Not Yet Feasible, with rationale tied directly to the ESO-01 through ESO-03 findings (ESO-04).
  5. If the recommendation is feasible (Bridge or Bypass), the doc sketches what "synced" could reasonably mean for a future ESO sync command (e.g. banner-only window vs. status-aware), scoped as input to a future milestone's requirements rather than implemented here (ESO-05).

**Plans**: 2/2 plans complete

- [x] 13-01-PLAN.md
- [x] 13-02-PLAN.md

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Site & Ephemeris Helper | v1.0 | 2/2 | Complete | 2026-06-12 |
| 2. Run Line Parsing | v1.1 | 1/1 | Complete | 2026-06-14 |
| 3. Classical Calendar Ingest | v1.1 | 2/2 | Complete | 2026-06-16 |
| 4. LCO Queue Sync Command | v1.2 | 1/1 | Complete | 2026-06-17 |
| 5. Multi-Proposal & Multi-Facility Selection | v1.3 | 1/1 | Complete | 2026-06-19 |
| 6. Correct Instrument-Type Extraction | v1.3 | 1/1 | Complete | 2026-06-21 |
| 7. Live Telescope-Label Resolution with Fallback & Failure Reporting | v1.3 | 2/2 | Complete | 2026-06-24 |
| 07.1. Close gap: SOAR fallback label is facility-unaware | v1.3 | 1/1 | Complete | 2026-06-24 |
| 8. Telescope Label Verification Sidecar | v1.4 | 2/2 | Complete | 2026-06-25 |
| 9. Proposal Color & Status Visual Treatment | v1.4 | 2/2 | Complete | 2026-06-26 |
| 10. Gemini Calendar Sync Command | v1.5 | 2/2 | Complete | 2026-06-27 |
| 11. Code Refactoring | v1.6 | 2/2 | Complete | 2026-06-27 |
| 12. Display Polish | v1.6 | 1/1 | Complete | 2026-06-28 |
| 13. ESO Feasibility Spike | v1.7 | 2/2 | Complete   | 2026-07-02 |

Full phase detail for all shipped milestones lives in their respective `milestones/*-ROADMAP.md` archive files linked above.
