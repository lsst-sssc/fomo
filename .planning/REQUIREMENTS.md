# Requirements: FOMO v2.0 Campaign Coordination for Rare/Urgent Objects

**Defined:** 2026-07-02
**Core Value:** When the next 4I-class object appears, FOMO replaces the ad-hoc Google Sheet as the community's campaign-coordination hub — target-linked observing runs, submission with oversight, and a per-object campaign view.

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### Campaign Data Model & Bootstrap Import (CAMP)

- [x] **CAMP-01**: `CampaignRun` model stores an observing run linked to a campaign `TargetList` (required FK) with the 3I-sheet field inventory: telescope/instrument, site code, obs date + UT range, filters/bandpass, observation details, weather, outcome, publication plans, collaboration flag, comments, contact person/email
- [x] **CAMP-02**: A `CampaignRun` can optionally record the specific `Target` observed (a member of the campaign's list); single-target campaigns work without setting it
- [x] **CAMP-03**: `CampaignRun` carries a lifecycle status (planned → observed → data reduced → published) plus an approval state (pending review / approved / rejected) as two independent controlled-vocabulary fields (approval status + run status), per discuss-phase decision D-02 (a flat vocabulary can't represent a pending real-world outcome independently of admin review state)
- [x] **CAMP-04**: Operator can bootstrap-import the real 3I/ATLAS sheet CSV via a management command with per-row skip-and-log error handling and a summary count (created/updated/skipped)
- [x] **CAMP-05**: The import command's paired demo notebook contains no real PII — it runs against a synthetic/redacted fixture (CLAUDE.md convention satisfied without committing real emails to git history)

### Campaign Display (VIEW)

- [x] **VIEW-01**: User can view a per-campaign table of all its runs (sortable/paginated), replacing the spreadsheet
- [x] **VIEW-02**: User can reach a target's campaigns from its target-detail page; navbar exposes a campaigns entry
- [x] **VIEW-03**: Contact person/email are visible only to authenticated staff — excluded from view context for anonymous requests and proven by an anonymous-client test
- [x] **VIEW-04**: User can filter the table by lifecycle status and the open-to-collaboration flag

### Community Submission & Approval (SUBMIT)

- [ ] **SUBMIT-01**: Community member can submit a run via a web form — campaign (TargetList) mandatory, all other fields optional
- [ ] **SUBMIT-02**: New submissions are pending and invisible on public views until approved
- [ ] **SUBMIT-03**: Staff can review and approve/reject pending runs; approval is atomic (double-approve is a no-op, proven by test)
- [ ] **SUBMIT-04**: The public form carries a honeypot field; bot submissions are dropped without processing
- [ ] **SUBMIT-05**: Staff receive an email notification when a new submission lands

### Calendar Projection (CAL)

- [ ] **CAL-01**: Approving a run that has telescope + date range creates/updates a paired `CalendarEvent` via `insert_or_create_calendar_event()` keyed `CAMPAIGN:{pk}` (no collisions with facility syncs)
- [ ] **CAL-02**: The paired `CalendarEvent.target_list` is set to the campaign's `TargetList` (native linkage — `CalendarEvent` has a `TargetList` FK but no `Target` FK)
- [ ] **CAL-03**: Re-approving or editing an unchanged run causes no duplicate events and no `modified` churn

### Coverage-Gap Analysis (GAP) — deferrable to v2.1 if time runs short

- [ ] **GAP-01**: Phase-time research spike decides dark-window-only vs. target-altitude filtering (the `ephem_utils`/SPICE cost decision) before implementation
- [ ] **GAP-02**: User can view observable-but-unclaimed dates for a campaign target + site; computed on explicit request or cached, never inline in the table view, never importing `ephem_utils` at module scope

## v2 Requirements

Deferred to a future release. Tracked but not in current roadmap.

### Community Submission & Approval

- **SUBMIT-06**: Trusted/known PIs can bypass the admin approval queue (self-service approval; needs its own submitter-trust design pass)
- **SUBMIT-07**: Submitter can check the status of their submission via a private link (received / approved / rejected)

### Campaign Display

- **VIEW-05**: Submitter can opt in to public display of their contact details on the campaign table

### ESO (carried from v1.7 close, unrelated to this milestone)

- **ESO-10**: `sync_eso_observation_calendar` management command (unblocked by Phase 13's Bypass verdict; see SEED-002 for medium-term ObservationRecord-centric intent)
- **ESO-11**: Paired ESO demo notebook

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Data-file/product upload & hosting (ExoFOP-style) | Turns FOMO into a data-storage system; the 3I sheet is metadata-only coordination |
| Bot-ingestion API for non-synced facilities (TNS-style) | Campaign runs exist precisely for the non-synced-facility case; a push API is a different product |
| Bookable/lockable coverage slots | Coverage-gap analysis is advisory display, not a reservation system |
| Third-party moderation packages (django-moderation etc.) | Single `status` field + admin action suffices for one linear-lifecycle model; packages are generic multi-model frameworks |
| Proposal-code validation against facility databases | Free-text is enough; validation adds facility coupling with no coordination value |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CAMP-01 | Phase 14 | Complete |
| CAMP-02 | Phase 14 | Complete |
| CAMP-03 | Phase 14 | Complete |
| CAMP-04 | Phase 14 | Complete |
| CAMP-05 | Phase 14 | Complete |
| VIEW-01 | Phase 15 | Complete |
| VIEW-02 | Phase 15 | Complete |
| VIEW-03 | Phase 15 | Complete |
| VIEW-04 | Phase 15 | Complete |
| SUBMIT-01 | Phase 16 | Pending |
| SUBMIT-02 | Phase 16 | Pending |
| SUBMIT-03 | Phase 16 | Pending |
| SUBMIT-04 | Phase 16 | Pending |
| SUBMIT-05 | Phase 16 | Pending |
| CAL-01 | Phase 16 | Pending |
| CAL-02 | Phase 16 | Pending |
| CAL-03 | Phase 16 | Pending |
| GAP-01 | Phase 17 | Pending |
| GAP-02 | Phase 17 | Pending |

**Coverage:**

- v1 requirements: 19 total
- Mapped to phases: 19 ✓ (Phases 14-17)
- Unmapped: 0 ✓

---
*Requirements defined: 2026-07-02*
*Last updated: 2026-07-02 after roadmap creation (Phases 14-17 mapped; 19/19 coverage)*
