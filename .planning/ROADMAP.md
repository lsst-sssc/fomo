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

### 🚧 v2.1 Uncertain Scheduling & Site Disambiguation (Phases 18-22) — IN PROGRESS

- [x] **Phase 18: Uncertain-Scheduling Investigation Spike** - Settle window schema, TBD natural key, CSV range/TBD parsing rules, and fuzzy-match library against real 3I sheet rows before implementation (completed 2026-07-09)
- [x] **Phase 19: Window-Schema Migration** - Replace single-night `obs_date`/`ut_start`/`ut_end` with a nullable `window_start`/`window_end` pair, migrating existing rows with no data loss (completed 2026-07-09)
- [x] **Phase 20: Range/TBD Import & Asset-Aware Coverage Gap** - Import range/TBD `Obs. Date` rows into the window representation and make coverage-gap analysis distinguish ground vs. space-mission runs (completed 2026-07-10)
- [x] **Phase 21: Site Disambiguation & Submitter Contact Opt-In** - Staff-facing fuzzy-match site-resolution UI in the approval queue plus a submitter contact opt-in flag (completed 2026-07-11)
- [x] **Phase 22: Site Matching at Submission and Unmatched-Site Resolution Workflow** - HTMX live fuzzy site search on the public submission form and approval queue, plus a "Sites needing review" resolution surface for approved runs with unresolved sites (with deferred calendar projection on resolve) (completed 2026-07-15)

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

**Plans:** 4/4 plans complete

Plans:
**Wave 1**

- [x] 19-01-PLAN.md — Schema: CampaignRun window_start/window_end fields + two partial UniqueConstraints + single combined migration 0004 (backfill + generic dedup) + model tests (SCHED-02/03/04/05, D-01/02/07/08)

**Wave 2** *(all depend on 19-01; no file overlap, run in parallel)*

- [x] 19-02-PLAN.md — Coverage-gap consumer: rewrite claimed_dates() to iterate the window range, delete _observing_night_date()
- [x] 19-03-PLAN.md — Views/display/forms/projection: window display column (D-03/05), nulls-last sort (D-04), D-06 hybrid ground/space calendar projection, PII allowlist swap, collapsed submission form
- [x] 19-04-PLAN.md — CSV import: window natural-key lookup + collision rethink + regenerated demo notebook

### Phase 20: Range/TBD Import & Asset-Aware Coverage Gap

**Goal**: Make the new window schema usable end-to-end for the harder 3I/ATLAS rows — import range and TBD `Obs. Date` cells into the window representation instead of dropping them, and rewrite coverage-gap analysis so ground runs and space-mission runs claim dates differently. Both are consumers of the Phase 19 schema and can proceed once it lands.
**Depends on**: Phase 19 (needs the window schema and asset-type derivation)
**Requirements**: IMPORT-01, IMPORT-02, ASSET-01, ASSET-02
**Success Criteria** (what must be TRUE):

  1. Importing a CSV row with a date range (e.g. "Aug 1-15") creates a run with the matching window instead of being skipped as a natural-key failure
  2. Importing a row with still-unparseable date text (e.g. "TBD pending Cycle 2") creates a TBD run flagged "needs review" and listed in the import summary, never silently dropped
  3. Coverage-gap analysis marks every date within a ground-based run's window as claimed (conservative, avoids double-booking)
  4. A space-mission run (its resolved site's `Observatory.observations_type == SATELLITE_OBSTYPE`) claims no dates until its window narrows to a single concrete night, at which point that night is claimed

**Plans:** 4/4 plans complete

Plans:
**Wave 1**

- [x] 20-01-PLAN.md — Asset-aware coverage gap: `claimed_dates()` ground-vs-space branch + `pending_narrowing_runs` bucket + gap-page alert (ASSET-01/02, D-09/D-10)
- [x] 20-02-PLAN.md — Schema: `original_obs_date_raw` + `window_needs_review` fields + migration 0006 + TBD-badge tooltip (D-01/D-02/D-04/D-08)

**Wave 2** *(depends on 20-02)*

- [x] 20-03-PLAN.md — CSV range/TBD parsing (`parse_obs_window` D-11/D-12/D-13) + `import_campaign_csv` row-loop rewrite (D-05/D-06/D-07, Pitfall 2) (IMPORT-01/02)

**Wave 3** *(depends on 20-03)*

- [x] 20-04-PLAN.md — Demo notebook: range/TBD import demonstration with executed output (IMPORT-01/02)

### Phase 21: Site Disambiguation & Submitter Contact Opt-In

**Goal**: Give staff a real site-disambiguation UI in the approval queue (the natural next step after quick task `260705-l1v`'s visibility fix) and let submitters opt into public contact display. Both are structurally independent of the scheduling-representation work — they touch `Observatory` resolution and the submission form, not the window schema.
**Depends on**: Phase 18 (spike decides the fuzzy-match library); independent of Phases 19-20
**Requirements**: SITE-01, SITE-02, SITE-03, VIEW-05
**Success Criteria** (what must be TRUE):

  1. When a submitted site doesn't resolve via `resolve_site()`'s exact-match or live-MPC-API tiers, the approval queue's Site column presents a dropdown of fuzzy-matched `Observatory` candidates for staff to pick from
  2. Staff can type a code directly and resolve it to an existing `Observatory` or explicitly create a new one; no placeholder `Observatory` is ever auto-fabricated (consistent with quick task `260705-l1v`)
  3. Approving a run whose site a staff member already manually resolved does not silently re-resolve or overwrite that choice
  4. A submitter who opts in (single combined flag, default opt-out) has their `contact_person`/`contact_email` shown publicly on the per-campaign table; leaving it unset keeps them staff-only exactly as today

**Plans:** 4/4 plans complete

Plans:
**Wave 1**

- [x] 21-01-PLAN.md — Site-disambiguation backend: `MPCObscodeFetcher.query_all()` bulk fetch, cached `build_site_candidates()` + `fuzzy_match_candidates()`, `TestSiteFuzzyMatch` scaffold (SITE-01, D-01/02/03)
- [x] 21-02-PLAN.md — VIEW-05 contact opt-in: `contact_public_opt_in` field + migration 0007 + submission-form checkbox + `Case`/`When` per-row PII gating (VIEW-05, D-07/08)

**Wave 2** *(depends on 21-01)*

- [x] 21-03-PLAN.md — Approval-queue site UI: single-`<form>` `render_actions()` refactor + `render_site()` input/datalist override + once-per-request candidate pool wiring (SITE-01, D-04)

**Wave 3** *(depends on 21-02, 21-03)*

- [x] 21-04-PLAN.md — Decision + create-new: `post()` D-06 clobber guard + `site_selection` read + `CreateObservatory` `?obscode=`/`?next=` support (SITE-02/03, D-05/06)

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
| 19. Window-Schema Migration | v2.1 | 4/4 | Complete    | 2026-07-09 |
| 20. Range/TBD Import & Asset-Aware Coverage Gap | v2.1 | 4/4 | Complete    | 2026-07-10 |
| 21. Site Disambiguation & Submitter Contact Opt-In | v2.1 | 4/4 | Complete    | 2026-07-11 |

Full phase detail for all shipped milestones lives in their respective `milestones/*-ROADMAP.md` archive files linked above.

## Current Milestone

**v2.1 Uncertain Scheduling & Site Disambiguation** — Phases 18-22, roadmap created 2026-07-05; Phase 22 added 2026-07-14 to close the Phase 21 site-matching functionality gap.

**Dependency spine:** Phase 18 (spike) settles the schema → Phase 19 (window migration, largest blast radius) → Phase 20 (import + asset-gap consumers of the new schema). Phase 21 (site UI + contact opt-in) is structurally independent of the scheduling work and depends only on Phase 18's fuzzy-library decision, so it can run in parallel with Phases 19-20.

Next: `/gsd-plan-phase 18`

### Phase 22: Site Matching at Submission and Unmatched-Site Resolution Workflow

**Goal:** Close the Phase 21 functionality gap: submitters and staff get live fuzzy matching against the merged local `Observatory` + full MPC candidate pool wherever a site is entered, and approved runs with unresolved sites get a resolution workflow instead of a dead end. (1) The public 'Submit an Observing Run' form's Observing site field becomes an HTMX live-search autocomplete backed by a new endpoint running `fuzzy_match_candidates()` over `build_site_candidates()`, replacing the bare free-text CharField; the same live-search widget replaces the approval queue's static per-row datalist (currently only the ≤5 fuzzy matches of the originally-submitted `site_raw`). (2) Post-approval resolution: keep "site failure never blocks approval", but add a "Sites needing review" surface listing approved runs with `site_needs_review=True`, with the same inline resolve input; resolving a site then triggers the deferred CalendarEvent projection that approval skipped.
**Requirements**: TBD
**Depends on:** Phase 21 (reuses `resolve_site`/`build_site_candidates`/`fuzzy_match_candidates` and the approval-queue decide flow)
**Plans:** 6/6 plans complete

Plans:
**Wave 1**

- [x] 22-01-PLAN.md — Live-search foundation: `substring_or_fuzzy_match_candidates()` + per-IP throttle, anonymous GET-only `SiteSearchView` HTML-fragment endpoint (`campaigns:site_search`), suggestion partial, new `test_campaign_site_search.py` (D-01..D-05)

**Wave 2** *(depends on 22-01)*

- [x] 22-02-PLAN.md — Widget wiring: public form `site_raw` HTMX live-search (no create-new link, D-09) + approval-queue pending-row datalist→live-widget swap keeping the Create-new link (D-10); text-only fill, resolution stays at approval (D-06)

**Wave 3** *(depends on 22-02)*

- [x] 22-03-PLAN.md — Post-approval resolution: `_project_calendar_event()` extraction (approve revert-path unchanged), `resolve_site` decision action with non-reverting failure path + D-06 guard, third "Sites Needing Review" table with Resolve action (D-07/D-08)

**Gap Closure (UAT 2026-07-15)**

*Gap-closure Wave 1*

- [x] 22-04-PLAN.md — Query-param fix (UAT tests 1 & 3): `SiteSearchView.get()` resolves the search term from `q` → `site_raw` → `site_selection` so the widgets' htmx `hx-get` requests (keyed by the input's own `name`) actually render suggestions; view-only, no widget markup change (D-03/D-09/D-10)
- [x] 22-05-PLAN.md — Sites Needing Review visual grouping (UAT test 2A): presentation-only card/section styling so the actionable section is distinct from the historical "Recently Decided" table, preserving D-07's locked document order (D-07)

*Gap-closure Wave 2* *(depends on 22-04, 22-05 — shared files)*

- [x] 22-06-PLAN.md — Placeholder-site correction (UAT test 2B): `NEEDS_REVIEW_NAME_PREFIX` + `is_placeholder_observatory()` helper, `render_site()` shows the correction widget for a placeholder site, `_resolve_site()` replaces a placeholder site via a pre-read-site-keyed conditional claim — D-06 racing/never-re-resolve, CR-01 non-revert, and D-09 never-fabricate all preserved (D-06/D-08/D-09)

### Phase 23: Weather/Storm Cancellation Handling

**Goal:** Give staff a way to mark scheduled telescope time as weathered-out/cancelled and have that status visibly reflected wherever it's tracked. Classical-schedule CalendarEvents (load_telescope_runs, e.g. Magellan Baade/Clay) currently recognize a 'cancelled' status word but only embed it as inert description text with zero visual differentiation on the calendar -- needs a visible cancelled/weathered treatment analogous to the LCO sync's [CANCELLED]/[EXPIRED] title-prefix mechanism. CampaignRun.run_status already has CANCELLED and WEATHER_TECH_FAILURE choices but is only editable via Django admin -- needs a staff-facing way to set it from the approval queue or per-campaign table, with the calendar-side CalendarEvent kept in sync when a run's status changes post-approval. Triggered by a real incoming storm expected to affect two scheduled Magellan runs (Baade IMACS 17-18 July, Clay Lightspeed 18-20 July) plus a Gemini FT program (GS-2026A-FT-115, 13-16 July) noted informally in Didymos_runs pending this feature.
**Requirements**: none mapped (organic phase outside v2.1 REQ scope; effective requirement set is CONTEXT.md D-01..D-07)
**Depends on:** Phase 22
**Plans:** 1/3 plans executed

Plans:
**Wave 1** *(parallel — no file overlap)*

- [x] 23-01-PLAN.md — Classical-schedule `[CANCELLED]` title prefix in load_telescope_runs + tests + demo notebook (D-01/D-02)
- [ ] 23-02-PLAN.md — CampaignRun status-change staff UI (`_set_run_status` + Decided-table action) + in-place calendar sync + `[WEATHERED]` box-shadow ring (D-03/D-04/D-05)

**Wave 2** *(depends on 23-02 — shares test_campaign_approval.py + exercises `_set_run_status()`)*

- [ ] 23-03-PLAN.md — Gemini FT-115 informational run + I11 site resolution + end-to-end no-special-casing scenario (D-06/D-07)
