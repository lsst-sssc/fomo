# Roadmap: Telescope Runs Calendar

## Milestones

- ✅ **v1.0 Site/Ephemeris Helper** — Phase 1 (shipped 2026-06-14) — see [milestones/1.0-ROADMAP.md](milestones/1.0-ROADMAP.md)
- ✅ **v1.1 Classical Run Ingest** — Phases 2-3 (shipped 2026-06-16) — see [milestones/v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 LCO Queue Calendar Sync** — Phase 4 (shipped 2026-06-18) — see [milestones/v1.2-ROADMAP.md](milestones/v1.2-ROADMAP.md)
- ✅ **v1.3 Full LCO Facility Sync** — Phases 5-7, 07.1 (shipped 2026-06-24) — see [milestones/v1.3-ROADMAP.md](milestones/v1.3-ROADMAP.md)
- 🚧 **v1.4 Calendar Visual Clarity** — Phases 8-9 (in progress, started 2026-06-24)

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

### v1.4 Calendar Visual Clarity (Phases 8-9) — IN PROGRESS

- [x] **Phase 8: Telescope Label Verification Sidecar** - Add a `CalendarEventTelescopeLabel` sidecar model recording fallback-vs-verified telescope-label resolution, with a visual cue and tooltip in the calendar UI. (completed 2026-06-25)
- [x] **Phase 9: Proposal Color & Status Visual Treatment** - Replace `pk`-based `CalendarEvent` color with a deterministic proposal-keyed palette, fix the `[QUEUED]` grey override, layer a status visual treatment on top, and add an on-page legend. (completed 2026-06-26)

## Phase Details

### Phase 8: Telescope Label Verification Sidecar

**Goal**: Operators can tell, directly in the calendar UI, whether a synced event's telescope label was live-verified against the LCO API or fallback-guessed, without reading title text.
**Depends on**: Phase 7 / 07.1 (telescope_api_failed signal, TELESCOPE-03/04) — already shipped
**Requirements**: DISPLAY-01, DISPLAY-02, DISPLAY-03 (REQUIREMENTS.md numbering: telescope-label-verification sidecar, visual cue, tooltip)
**Success Criteria** (what must be TRUE):

  1. After running `sync_lco_observation_calendar`, every synced `CalendarEvent` has an associated `CalendarEventTelescopeLabel` row correctly marked verified or fallback, matching that record's actual `telescope_api_failed` outcome.
  2. Events created by `load_telescope_runs` (classical schedule) have no sidecar row and render as "verified" by the template's documented default — no behavior change to that command.
  3. On the calendar page, a fallback-labeled event is visually distinguishable (border/badge) from a verified one, discoverable without opening the event or reading its title.
  4. Hovering a fallback-labeled event shows a tooltip with the verification detail (not just the visual cue alone).
  5. Re-running the sync command on unchanged records does not create duplicate sidecar rows and does not churn `CalendarEvent.modified` (existing no-churn contract preserved).

**Plans**: 2/2 plans complete
**Wave 1**

- [x] 08-01-PLAN.md — Sidecar model + first migration + sync_lco_observation_calendar write + tests + demo notebook (DISPLAY-01)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 08-02-PLAN.md — calendar.html dashed-border cue + hover tooltip (all-day + timed) + rendering test (DISPLAY-02, DISPLAY-03)

**UI hint**: yes

### Phase 9: Proposal Color & Status Visual Treatment

**Goal**: A calendar viewer can identify which proposal an event belongs to by color alone (consistent across telescopes and re-renders) and can distinguish queued/placed/terminal-failure status visually, not just by reading title-prefix text.
**Depends on**: Phase 8 (shares `calendar.html` and, if the sketch session decides to reuse it, the `calendar_display_extras.py` template-tag module/visual vocabulary from Phase 8)
**Requirements**: DISPLAY-04, DISPLAY-05, DISPLAY-06, DISPLAY-07 (REQUIREMENTS.md numbering: proposal-keyed color, `[QUEUED]` override fix, status visual treatment, on-page legend)
**Success Criteria** (what must be TRUE):

  1. Two events with the same normalized proposal string render the same color, regardless of telescope/site, across htmx month-grid re-renders and process restarts; events with an empty proposal get a dedicated neutral slot rather than a hash-of-empty-string color.
  2. A `[QUEUED]` event still shows its proposal color (dimmed/bordered as appropriate), instead of today's flat grey that discards it.
  3. Queued, placed, and terminal-failure events are visually distinguishable from each other via a status treatment (mechanism chosen via `/gsd:sketch` during phase planning) layered on top of proposal color, for both all-day and timed events; the existing `[QUEUED]`/`[UNVERIFIED]`/terminal-prefix text remains as an accessible fallback.
  4. A viewer can look at an on-page legend and match a rendered color to its proposal code without hovering or clicking into any event.
  5. Clicking a legend entry highlights that proposal's events and dims the rest of the calendar grid, client-side with no page reload; clicking again clears the highlight. (Scope added during Phase 9 discussion, 2026-06-25 — see DISPLAY-07 in REQUIREMENTS.md.)

**Plans**: 2/2 plans complete
**Wave 1**

- [x] 09-01-PLAN.md — calendar_display_extras template-tag module (proposal_color/status_border_css/visible_proposals) + Wave 0 unit tests (DISPLAY-04, DISPLAY-06, DISPLAY-07)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 09-02-PLAN.md — calendar.html rewrite: proposal color + box-shadow status + [QUEUED] fix, footer legend + click-to-filter, integration tests (DISPLAY-04, DISPLAY-05, DISPLAY-06, DISPLAY-07)

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
| 8. Telescope Label Verification Sidecar | v1.4 | 2/2 | Complete    | 2026-06-25 |
| 9. Proposal Color & Status Visual Treatment | v1.4 | 2/2 | Complete   | 2026-06-26 |

Full phase detail (goals, success criteria, plans) for all shipped milestones lives in their respective `milestones/*-ROADMAP.md` archive files linked above.
