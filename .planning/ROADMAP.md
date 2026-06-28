# Roadmap: Telescope Runs Calendar

## Milestones

- ✅ **v1.0 Site/Ephemeris Helper** — Phase 1 (shipped 2026-06-14) — see [milestones/1.0-ROADMAP.md](milestones/1.0-ROADMAP.md)
- ✅ **v1.1 Classical Run Ingest** — Phases 2-3 (shipped 2026-06-16) — see [milestones/v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 LCO Queue Calendar Sync** — Phase 4 (shipped 2026-06-18) — see [milestones/v1.2-ROADMAP.md](milestones/v1.2-ROADMAP.md)
- ✅ **v1.3 Full LCO Facility Sync** — Phases 5-7, 07.1 (shipped 2026-06-24) — see [milestones/v1.3-ROADMAP.md](milestones/v1.3-ROADMAP.md)
- ✅ **v1.4 Calendar Visual Clarity** — Phases 8-9 (shipped 2026-06-26) — see [milestones/v1.4-ROADMAP.md](milestones/v1.4-ROADMAP.md)
- ✅ **v1.5 Gemini Calendar Sync** — Phase 10 (shipped 2026-06-27) — see [milestones/v1.5-ROADMAP.md](milestones/v1.5-ROADMAP.md)
- 🚧 **v1.6 Tech Debt & Display Polish** — Phases 11-12 (in progress)

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

### 🚧 v1.6 Tech Debt & Display Polish (In Progress)

**Milestone Goal:** Clear all deferred technical debt and display polish items accumulated across v1.3–v1.5, leaving the codebase clean before Stage 4.

- [x] **Phase 11: Code Refactoring** - Extract shared telescope-mapping and calendar-event utility code into standalone modules; update all three commands to use them; remove "upsert" from live docs (completed 2026-06-27)
- [ ] **Phase 12: Display Polish** - WCAG-AA-compliant text color per palette background; batch-load telescope-label data to eliminate N+1 query in calendar template

## Phase Details

Full phase detail (goals, success criteria, plans) for Phases 8-10 lives in their respective milestone archive files linked in the Milestones section above.

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

### Phase 10: Gemini Calendar Sync Command

**Goal**: Operators can sync submitted Gemini ToO `ObservationRecord`s to `CalendarEvent` window banners with a single management command — idempotent, credential-safe, and exercised by a runnable demo notebook.
**Depends on**: Phase 9 (shares `CalendarEvent` model, no-churn idempotency pattern, and management-command conventions established through Phases 3-9)
**Requirements**: GEM-SELECT-01, GEM-WINDOW-01, GEM-WINDOW-02, GEM-KEY-01, GEM-TELE-01, GEM-INSTR-01, GEM-PROP-01, GEM-STATUS-01, GEM-NOCHURN-01, GEM-SECURE-01
**Success Criteria** (what must be TRUE):

  1. `./manage.py sync_gemini_observation_calendar` exits cleanly and creates one `CalendarEvent` per `ObservationRecord(facility='GEM')`, with `CalendarEvent.url` set to `GEM:{prog}/{observation_id}` and `proposal` set from `params['prog']`.
  2. Events with `windowDate`/`windowTime`/`windowDuration` parameters have `start_time`/`end_time` from those values; events without use ToO-type-derived windows (`Rap:` → `[created, created+24h]`, `Std:` → `[created+24h, created+7d]`); records with neither an explicit window nor a resolvable ToO type appear in the `skipped` counter and produce no `CalendarEvent`.
  3. `telescope` is `'Gemini South'` for `GS-*` programs and `'Gemini North'` for `GN-*`; `instrument` strips any `Std:`/`Rap:` prefix from the obs-code settings description (falling back to raw obs code); records with `params['ready'] == 'false'` get a `[ON_HOLD]` title prefix, all others get a clean title.
  4. Re-running the command on the same set of records produces no duplicate `CalendarEvent` rows and leaves `CalendarEvent.modified` unchanged on every record that had no field change.
  5. The `password` key from `parameters` is absent from all log output, exception tracebacks, and `CalendarEvent` fields; the demo notebook `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb` executes end-to-end without error using synthetic fixtures.

**Plans**: 2 plans

**Wave 1**

- [x] 10-01-PLAN.md — `sync_gemini_observation_calendar` command + full Django test suite (all 10 GEM-* requirements)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 10-02-PLAN.md — pre-executed demo notebook covering the four D-06 scenarios (GEM-WINDOW-01/02, GEM-STATUS-01, GEM-NOCHURN-01, GEM-SECURE-01)

### Phase 11: Code Refactoring

**Goal**: Shared telescope-mapping and calendar-event creation logic is extracted into importable utility modules that all three management commands use, with no duplicated implementation and no "upsert" jargon remaining in live docs or comments.
**Depends on**: Phase 10
**Requirements**: REFAC-01, REFAC-02
**Success Criteria** (what must be TRUE):

  1. `SITE_TELESCOPE_MAP`, `_extract_instrument`, and related LCO/SOAR helpers are importable from a new standalone `solsys_code/` module without importing the management command file
  2. All three management commands (`load_telescope_runs`, `sync_lco_observation_calendar`, `sync_gemini_observation_calendar`) delegate their CalendarEvent create-or-update logic to `insert_or_create_calendar_event()`; the prior duplicated code blocks are absent from each command file
  3. The word "upsert" does not appear in `docs/design/telescope_runs_calendar.rst` or `.planning/MILESTONES.md` (replaced with plain English or the function name)
  4. All `./manage.py test solsys_code` tests pass with no behavior change — the refactor is behavior-neutral

**Plans**: 2 plans

**Wave 1**

- [x] 11-01-PLAN.md — Create `solsys_code/calendar_utils.py` (REFAC-01 telescope-mapping extractions + `insert_or_create_calendar_event`); refactor `sync_lco_observation_calendar` to use it (REFAC-01, REFAC-02)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 11-02-PLAN.md — Refactor `load_telescope_runs` + `sync_gemini_observation_calendar` to use `insert_or_create_calendar_event`; replace "upsert" in `.rst` and `MILESTONES.md` (REFAC-02)

### Phase 12: Display Polish

**Goal**: Calendar event text is WCAG-AA-compliant against every palette background, and the calendar template loads telescope-label data without an N+1 query per event.
**Depends on**: Phase 11
**Requirements**: DISPLAY-08, DISPLAY-09
**Success Criteria** (what must be TRUE):

  1. Every calendar event's title text renders in white or black — whichever achieves WCAG AA 4.5:1 contrast against its proposal palette background — with the choice computed from relative luminance, not hardcoded per palette entry
  2. All 8 colors in `PROPOSAL_PALETTE` pass WCAG AA 4.5:1 contrast when paired with their computed text color, verifiable by the test suite
  3. `CalendarEventTelescopeLabel` data for all visible calendar events is loaded in a single prefetch query rather than one query per event, regardless of how many events are on the calendar
  4. The full test suite passes with all existing tests preserved and new behavior covered

**Plans**: 1 plan

**Wave 1**

- [ ] 12-01-PLAN.md — `text_color_for_bg` WCAG text-color tag (DISPLAY-08) + FOMO `fomo_render_calendar` wrapper view with prefetch + Count annotation (DISPLAY-09) + calendar.html edits + unit & integration tests

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
| 9. Proposal Color & Status Visual Treatment | v1.4 | 2/2 | Complete    | 2026-06-26 |
| 10. Gemini Calendar Sync Command | v1.5 | 2/2 | Complete    | 2026-06-27 |
| 11. Code Refactoring | v1.6 | 2/2 | Complete   | 2026-06-27 |
| 12. Display Polish | v1.6 | 0/1 | Not started | - |

Full phase detail for all shipped milestones lives in their respective `milestones/*-ROADMAP.md` archive files linked above.
