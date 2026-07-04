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
- 🚧 **v2.0 Campaign Coordination for Rare/Urgent Objects** — Phases 14-17 (in progress, opened 2026-07-02)

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

### 🚧 v2.0 Campaign Coordination for Rare/Urgent Objects (Phases 14-17) — IN PROGRESS

- [x] **Phase 14: Campaign Data Model & Bootstrap Import** - `CampaignRun` model + one-off 3I/ATLAS CSV import validated against real data (completed 2026-07-03)
- [x] **Phase 15: Per-Campaign Table View (Read Path)** - Spreadsheet-replacement table of all runs for a campaign, PII-gated (completed 2026-07-03)
- [x] **Phase 16: Submission Form, Approval Queue & Calendar Projection (Write Path)** - Community intake with staff approval gate; approved runs project onto the calendar (completed 2026-07-04)
- [ ] **Phase 17: Coverage-Gap Analysis (Deferrable to v2.1)** - Ephemeris-aware observable-but-unclaimed dates; the differentiator over any spreadsheet

## Phase Details

_Detail sections below cover the active v2.0 milestone. Full phase detail for all
shipped milestones lives in their respective `milestones/*-ROADMAP.md` archive files
linked in the Milestones section above._

### Phase 14: Campaign Data Model & Bootstrap Import

**Goal**: A `CampaignRun` model exists — linked to a campaign `TargetList`, carrying the full 3I-sheet field inventory and a combined lifecycle/approval status — and the real 3I/ATLAS coordination sheet can be imported into it.
**Depends on**: Nothing (first phase of milestone)
**Requirements**: CAMP-01, CAMP-02, CAMP-03, CAMP-04, CAMP-05
**Success Criteria** (what must be TRUE):

  1. A `CampaignRun` record stores its campaign `TargetList`, an optional observed `Target`, two independent controlled-vocabulary fields for lifecycle/approval status (approval status + run status, per discuss-phase decision D-02), and the full 3I field inventory (telescope/instrument, site, obs date + UT range, filters/bandpass, observation details, weather, outcome, publication plans, collaboration flag, comments, contact person/email).
  2. A single-target campaign works without ever setting the optional observed `Target`.
  3. Operator can run a management command that imports the real 3I/ATLAS sheet CSV, reporting a created/updated/skipped summary; unparseable rows are skipped and logged without aborting the run.
  4. The import command's paired demo notebook runs end-to-end against a synthetic/redacted fixture with no real PII committed to git history.

**Plans**: 3/3 plans complete
**Wave 1**

- [x] 14-01-PLAN.md — CampaignRun model + migration + model tests (CAMP-01/02/03)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 14-02-PLAN.md — campaign_utils helpers + import_campaign_csv command + tests (CAMP-02/04)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 14-03-PLAN.md — synthetic PII-free fixture + paired demo notebook (CAMP-05)

### Phase 15: Per-Campaign Table View (Read Path)

**Goal**: A coordinator can see every run for a campaign in one sortable, filterable table that replaces the shared spreadsheet — with contact details visible only to staff.
**Depends on**: Phase 14
**Requirements**: VIEW-01, VIEW-02, VIEW-03, VIEW-04
**Success Criteria** (what must be TRUE):

  1. User can view a per-campaign table listing all of its runs, sortable and paginated.
  2. User can reach a campaign's table from the relevant target-detail page, and a navbar entry exposes campaigns.
  3. Contact person/email are excluded from view context for anonymous requests (proven by an anonymous-client test) and shown only to authenticated staff.
  4. User can filter the table by lifecycle status and by the open-to-collaboration flag.

**Plans**: 2/2 plans complete
**UI hint**: yes

**Wave 1**

- [x] 15-01-PLAN.md — per-campaign table read path: CampaignRunTable + FilterSet + views + urls + templates + tests (VIEW-01/03/04)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 15-02-PLAN.md — navigation & target-detail integration: apps.py hooks + inclusion tags + partials + tests (VIEW-02)

### Phase 16: Submission Form, Approval Queue & Calendar Projection (Write Path)

**Goal**: Community members (PIs and external observers) can submit runs that stay hidden until a staff member approves them, and an approved run with a telescope and date range appears on the shared calendar.
**Depends on**: Phase 15
**Requirements**: SUBMIT-01, SUBMIT-02, SUBMIT-03, SUBMIT-04, SUBMIT-05, CAL-01, CAL-02, CAL-03
**Success Criteria** (what must be TRUE):

  1. A community member can submit a run via a web form with campaign (`TargetList`) mandatory and every other field optional.
  2. A new submission is pending and invisible on public views until a staff member approves it.
  3. Staff can approve or reject pending runs, and approval is atomic — a double-approve is a proven no-op.
  4. Honeypot bot submissions are dropped without processing, and staff receive an email notification when a genuine submission lands.
  5. Approving a run that has a telescope + date range creates or updates a paired `CalendarEvent` (keyed `CAMPAIGN:{pk}` via `insert_or_create_calendar_event()`, `target_list` set to the campaign's list) with no duplicate events and no `modified` churn on re-approval or unchanged edits.

**Plans**: 5/5 plans complete
**UI hint**: yes

**Wave 1**

- [x] 16-01-PLAN.md — StaffRequiredMixin + CampaignRunSubmissionForm (plain forms.Form + honeypot) + EMAIL_BACKEND (SUBMIT-01/04)

**Wave 2** *(blocked on Wave 1)*

- [x] 16-02-PLAN.md — CampaignRunSubmissionView + submit/thanks URLs + templates + staff email notify + tests (SUBMIT-01/04/05)

**Wave 3** *(blocked on Wave 2)*

- [x] 16-03-PLAN.md — ApprovalQueueView + atomic approve/reject decision endpoint + calendar projection + ApprovalQueueTable + tests (SUBMIT-03/CAL-01/02/03)

**Wave 4** *(blocked on Wave 3)*

- [x] 16-04-PLAN.md — D-09 non-staff visibility filter + Submit-a-Run entry buttons + staff pending banner + tests (SUBMIT-01/02)

**Gap closure** *(post-UAT)*

- [x] 16-05-PLAN.md — Trim/reorder ApprovalQueueTable columns (Meta.exclude + sequence, actions-first) so Approve/Reject is reachable without scrolling; closes UAT Test 14 (SUBMIT-03)

### Phase 17: Coverage-Gap Analysis (Deferrable to v2.1)

**Goal**: A user can see which observable nights for a campaign target and site are not yet claimed by any run — FOMO's differentiator over any spreadsheet. **This phase is explicitly deferrable to v2.1 if the milestone runs long**; it is ordered last so it can be cut without disturbing the launch-critical Phases 14-16. GAP-01 is a phase-time research spike that gates GAP-02's implementation approach.
**Depends on**: Phase 16 (needs both the claimed-runs data and the campaign table stable)
**Requirements**: GAP-01, GAP-02
**Success Criteria** (what must be TRUE):

  1. A phase-time research spike produces an explicit decision — dark-window-only vs. target-altitude filtering — settling the `ephem_utils`/SPICE-cost tradeoff before any implementation begins.
  2. User can view observable-but-unclaimed dates for a campaign target + site, computed on explicit request or from a cache.
  3. The gap computation never runs inline in the table view and never imports `ephem_utils` at module scope (lightweight `telescope_runs.py` helpers or a lazy import only).

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
| 14. Campaign Data Model & Bootstrap Import | v2.0 | 3/3 | Complete    | 2026-07-03 |
| 15. Per-Campaign Table View (Read Path) | v2.0 | 2/2 | Complete    | 2026-07-03 |
| 16. Submission Form, Approval Queue & Calendar Projection | v2.0 | 5/5 | Complete   | 2026-07-04 |
| 17. Coverage-Gap Analysis (Deferrable to v2.1) | v2.0 | 0/? | Not started | - |

Full phase detail for all shipped milestones lives in their respective `milestones/*-ROADMAP.md` archive files linked above. Active v2.0 phase detail is in the **Phase Details** section above.

## Current Milestone

**v2.0 Campaign Coordination for Rare/Urgent Objects** — Phases 14-17, opened 2026-07-02.

When the next 4I-class object appears, FOMO replaces the ad-hoc Google Sheet as the
community's campaign-coordination hub — target-linked observing runs, submission with
oversight, and a per-object campaign view. Phases 14-16 are launch-critical; Phase 17
(coverage-gap analysis) is the differentiator and is deferrable to v2.1 if time runs short.

Next: `/gsd-plan-phase 14`
