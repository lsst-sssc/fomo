# Roadmap: Telescope Runs Calendar

## Milestones

- ✅ **v1.0 Site/Ephemeris Helper** — Phase 1 (shipped 2026-06-14) — see [milestones/1.0-ROADMAP.md](milestones/1.0-ROADMAP.md)
- ✅ **v1.1 Classical Run Ingest** — Phases 2-3 (shipped 2026-06-16) — see [milestones/v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 LCO Queue Calendar Sync** — Phase 4 (shipped 2026-06-18) — see [milestones/v1.2-ROADMAP.md](milestones/v1.2-ROADMAP.md)
- ✅ **v1.3 Full LCO Facility Sync** — Phases 5-7, 07.1 (shipped 2026-06-24) — see [milestones/v1.3-ROADMAP.md](milestones/v1.3-ROADMAP.md)
- ✅ **v1.4 Calendar Visual Clarity** — Phases 8-9 (shipped 2026-06-26) — see [milestones/v1.4-ROADMAP.md](milestones/v1.4-ROADMAP.md)
- ✅ **v1.5 Gemini Calendar Sync** — Phase 10 (shipped 2026-06-27) — see [milestones/v1.5-ROADMAP.md](milestones/v1.5-ROADMAP.md)
- ✅ **v1.6 Tech Debt & Display Polish** — Phases 11-12 (shipped 2026-06-29) — see [milestones/v1.6-ROADMAP.md](milestones/v1.6-ROADMAP.md)
- ✅ **v1.7 ESO/VLT Calendar Sync — Feasibility Spike** — Phase 13 (shipped 2026-07-02) — see [milestones/v1.7-ROADMAP.md](milestones/v1.7-ROADMAP.md)
- ✅ **v2.0 Campaign Coordination for Rare/Urgent Objects** — Phases 14-17 (shipped 2026-07-05) — see [milestones/v2.0-ROADMAP.md](milestones/v2.0-ROADMAP.md)
- 🚧 **v2.1 Uncertain Scheduling & Site Disambiguation** — Phases 18-21 (in progress)

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

<details>
<summary>✅ v1.7 ESO/VLT Calendar Sync — Feasibility Spike (Phase 13) — SHIPPED 2026-07-02</summary>

- [x] Phase 13: ESO Feasibility Spike (2/2 plans) — completed 2026-07-02

</details>

<details>
<summary>✅ v2.0 Campaign Coordination for Rare/Urgent Objects (Phases 14-17) — SHIPPED 2026-07-05</summary>

- [x] Phase 14: Campaign Data Model & Bootstrap Import (3/3 plans) — completed 2026-07-03
- [x] Phase 15: Per-Campaign Table View (Read Path) (2/2 plans) — completed 2026-07-03
- [x] Phase 16: Submission Form, Approval Queue & Calendar Projection (Write Path) (5/5 plans) — completed 2026-07-04
- [x] Phase 17: Coverage-Gap Analysis (Deferrable to v2.1, shipped anyway) (3/3 plans) — completed 2026-07-04

</details>

### 🚧 v2.1 Uncertain Scheduling & Site Disambiguation (Phases 18-21) — IN PROGRESS

- [x] **Phase 18: Uncertain-Scheduling Investigation Spike** - Settle window schema, TBD natural key, CSV range/TBD parsing rules, and fuzzy-match library against real 3I sheet rows before implementation (completed 2026-07-09)
- [ ] **Phase 19: Window-Schema Migration** - Replace single-night `obs_date`/`ut_start`/`ut_end` with a nullable `window_start`/`window_end` pair, migrating existing rows with no data loss
- [ ] **Phase 20: Range/TBD Import & Asset-Aware Coverage Gap** - Import range/TBD `Obs. Date` rows into the window representation and make coverage-gap analysis distinguish ground vs. space-mission runs
- [ ] **Phase 21: Site Disambiguation & Submitter Contact Opt-In** - Staff-facing fuzzy-match site-resolution UI in the approval queue plus a submitter contact opt-in flag

## Phase Details

### Phase 18: Uncertain-Scheduling Investigation Spike

**Goal**: Settle the uncertain-scheduling design decisions against the real 3I/ATLAS sheet rows before any implementation lands — this is the phase-time spike the milestone deliberately includes rather than defers, and it gates the schema everything downstream depends on.
**Depends on**: Nothing (first phase of v2.1)
**Requirements**: SCHED-01
**Success Criteria** (what must be TRUE):

  1. A decision doc records the final window field schema (`window_start`/`window_end` nullable `DateField` pair) validated against real 3I sheet rows
  2. The replacement natural key for TBD rows (null `window_start`) is decided and documented, including how the SQLite/PostgreSQL NULL-uniqueness gap will be closed (partial/conditional constraint)
  3. The CSV range/TBD text-parsing rules are enumerated from the actual cell shapes present in the real 3I sheet (one rule per real shape), not guessed generically
  4. The fuzzy-match library choice (`rapidfuzz` vs. stdlib `difflib`) is made with a recorded rationale from match-quality testing against real messy site-name input
  5. `resolve_site()` is confirmed to resolve real space-observatory MPC codes (250=Hubble, 274=JWST, 289=Nancy Grace Roman) correctly, with a documented verdict on whether `Observatory.obscode` length needs changing at all (default answer: no)

**Plans:** 2/2 plans complete

Plans:
**Wave 1**

- [x] 18-01-PLAN.md — Live rapidfuzz-vs-difflib + resolve_site() probe against the real 3I/ATLAS CSV; capture redacted Findings (SCHED-01 criteria 2-5) into 18-DECISION.md

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 18-02-PLAN.md — Complete 18-DECISION.md Recommendation for all 5 criteria; write durable docs/design/uncertain_scheduling_spike.rst

### Phase 19: Window-Schema Migration

**Goal**: Replace `CampaignRun`'s single-night `obs_date`/`ut_start`/`ut_end` representation with a nullable window (`window_start`/`window_end`), migrating every existing row with no data loss. This is the largest-blast-radius change in the milestone and lands as its own phase before any consumer (CSV import, gap analysis) is touched.
**Depends on**: Phase 18 (spike settles the window schema and TBD natural key)
**Requirements**: SCHED-02, SCHED-03, SCHED-04, SCHED-05
**Success Criteria** (what must be TRUE):

  1. A `CampaignRun` can be saved with `window_start == window_end`, representing a single classically-scheduled night
  2. A `CampaignRun` can be saved in a "TBD" state (both window fields null), distinct from a resolved window
  3. Two distinct TBD rows for the same campaign + telescope neither silently merge nor silently duplicate under the DB constraint
  4. Every existing `CampaignRun` row survives the migration with `window_start == window_end == former obs_date` — no data loss
  5. The existing per-campaign table, approval queue, and coverage-gap pages still render correctly against the new window fields

**Plans:** 1/4 plans executed

Plans:
**Wave 1**

- [x] 19-01-PLAN.md — Schema: CampaignRun window_start/window_end fields + two partial UniqueConstraints + single combined migration 0004 (backfill + generic dedup) + model tests (SCHED-02/03/04/05, D-01/02/07/08)

**Wave 2** *(all depend on 19-01; no file overlap, run in parallel)*

- [ ] 19-02-PLAN.md — Coverage-gap consumer: rewrite claimed_dates() to iterate the window range, delete _observing_night_date()
- [ ] 19-03-PLAN.md — Views/display/forms/projection: window display column (D-03/05), nulls-last sort (D-04), D-06 hybrid ground/space calendar projection, PII allowlist swap, collapsed submission form
- [ ] 19-04-PLAN.md — CSV import: window natural-key lookup + collision rethink + regenerated demo notebook

### Phase 20: Range/TBD Import & Asset-Aware Coverage Gap

**Goal**: Make the new window schema usable end-to-end for the harder 3I/ATLAS rows — import range and TBD `Obs. Date` cells into the window representation instead of dropping them, and rewrite coverage-gap analysis so ground runs and space-mission runs claim dates differently. Both are consumers of the Phase 19 schema and can proceed once it lands.
**Depends on**: Phase 19 (needs the window schema and asset-type derivation)
**Requirements**: IMPORT-01, IMPORT-02, ASSET-01, ASSET-02
**Success Criteria** (what must be TRUE):

  1. Importing a CSV row with a date range (e.g. "Aug 1-15") creates a run with the matching window instead of being skipped as a natural-key failure
  2. Importing a row with still-unparseable date text (e.g. "TBD pending Cycle 2") creates a TBD run flagged "needs review" and listed in the import summary, never silently dropped
  3. Coverage-gap analysis marks every date within a ground-based run's window as claimed (conservative, avoids double-booking)
  4. A space-mission run (its resolved site's `Observatory.observations_type == SATELLITE_OBSTYPE`) claims no dates until its window narrows to a single concrete night, at which point that night is claimed

**Plans**: TBD

### Phase 21: Site Disambiguation & Submitter Contact Opt-In

**Goal**: Give staff a real site-disambiguation UI in the approval queue (the natural next step after quick task `260705-l1v`'s visibility fix) and let submitters opt into public contact display. Both are structurally independent of the scheduling-representation work — they touch `Observatory` resolution and the submission form, not the window schema.
**Depends on**: Phase 18 (spike decides the fuzzy-match library); independent of Phases 19-20
**Requirements**: SITE-01, SITE-02, SITE-03, VIEW-05
**Success Criteria** (what must be TRUE):

  1. When a submitted site doesn't resolve via `resolve_site()`'s exact-match or live-MPC-API tiers, the approval queue's Site column presents a dropdown of fuzzy-matched `Observatory` candidates for staff to pick from
  2. Staff can type a code directly and resolve it to an existing `Observatory` or explicitly create a new one; no placeholder `Observatory` is ever auto-fabricated (consistent with quick task `260705-l1v`)
  3. Approving a run whose site a staff member already manually resolved does not silently re-resolve or overwrite that choice
  4. A submitter who opts in (single combined flag, default opt-out) has their `contact_person`/`contact_email` shown publicly on the per-campaign table; leaving it unset keeps them staff-only exactly as today

**Plans**: TBD
**UI hint**: yes

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
| 13. ESO Feasibility Spike | v1.7 | 2/2 | Complete | 2026-07-02 |
| 14. Campaign Data Model & Bootstrap Import | v2.0 | 3/3 | Complete | 2026-07-03 |
| 15. Per-Campaign Table View (Read Path) | v2.0 | 2/2 | Complete | 2026-07-03 |
| 16. Submission Form, Approval Queue & Calendar Projection | v2.0 | 5/5 | Complete | 2026-07-04 |
| 17. Coverage-Gap Analysis (Deferrable to v2.1) | v2.0 | 3/3 | Complete | 2026-07-04 |
| 18. Uncertain-Scheduling Investigation Spike | v2.1 | 2/2 | Complete    | 2026-07-09 |
| 19. Window-Schema Migration | v2.1 | 1/4 | In Progress|  |
| 20. Range/TBD Import & Asset-Aware Coverage Gap | v2.1 | 0/? | Not started | - |
| 21. Site Disambiguation & Submitter Contact Opt-In | v2.1 | 0/? | Not started | - |

Full phase detail for all shipped milestones lives in their respective `milestones/*-ROADMAP.md` archive files linked above.

## Current Milestone

**v2.1 Uncertain Scheduling & Site Disambiguation** — Phases 18-21, roadmap created 2026-07-05.

**Dependency spine:** Phase 18 (spike) settles the schema → Phase 19 (window migration, largest blast radius) → Phase 20 (import + asset-gap consumers of the new schema). Phase 21 (site UI + contact opt-in) is structurally independent of the scheduling work and depends only on Phase 18's fuzzy-library decision, so it can run in parallel with Phases 19-20.

Next: `/gsd-plan-phase 18`
