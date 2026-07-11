# Requirements: FOMO v2.1 Uncertain Scheduling & Site Disambiguation

**Defined:** 2026-07-05
**Core Value:** Campaign coordination handles the real 3I/ATLAS sheet's harder rows — space-mission observations (e.g. the Carrie Holt/Martin Cordiner JWST rows) whose exact observing night isn't known yet, only a window or a still-pending schedule — while closing out submitter contact opt-in (VIEW-05) and a real staff-facing site-disambiguation UI.

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### Scheduling Window Model (SCHED)

- [x] **SCHED-01**: A phase-time investigation spike settles the window field schema (`window_start`/`window_end` nullable `DateField` pair), the replacement natural key for TBD rows (no fixed start time), the CSV range/TBD text-parsing rules, the fuzzy-match library choice (`rapidfuzz` vs. stdlib `difflib`), and confirms whether `resolve_site()` correctly resolves real space-observatory MPC codes (250=Hubble, 274=JWST, 289=Nancy Grace Roman — standard 3-char codes; `Observatory.obscode` widening is presumed NOT needed per post-research correction) — before implementation begins
- [x] **SCHED-02**: `CampaignRun`'s `obs_date`/`ut_start`/`ut_end` (single-night representation) is replaced by a window (`window_start`/`window_end`); a classically-scheduled single night is represented as `window_start == window_end`
- [x] **SCHED-03**: A `CampaignRun` can exist with no window at all yet ("TBD" state — both `window_start`/`window_end` null), distinct from having a resolved window
- [x] **SCHED-04**: The natural-key `UniqueConstraint` no longer silently allows duplicate TBD rows (the SQLite/PostgreSQL NULL-uniqueness gap is closed via a partial/conditional constraint)
- [x] **SCHED-05**: Existing `CampaignRun` rows migrate to the new window representation with no data loss (single night → `window_start == window_end == obs_date`)

### Asset-Type Distinction & Coverage-Gap (ASSET)

- [x] **ASSET-01**: A `CampaignRun`'s "ground" vs. "space-mission" classification is derived from its resolved site's `Observatory.observations_type` (`SATELLITE_OBSTYPE`) — no new field on `CampaignRun`
- [x] **ASSET-02**: Coverage-gap analysis claims every date in a ground-based run's window; a space-mission run claims no dates until its window narrows to a single concrete night (`window_start == window_end`)

### CSV Import (IMPORT)

- [x] **IMPORT-01**: `import_campaign_csv`/`parse_obs_window` accepts a date range (e.g. "Aug 1-15") or a TBD-style free-text cell (e.g. "TBD pending Cycle 2") and imports the row into the new window representation, instead of raising and skipping it as a natural-key failure
- [x] **IMPORT-02**: A row whose `Obs. Date` text still can't be parsed after the above gets a "needs review" flag and is included in the import summary, never silently dropped

### Site Disambiguation (SITE)

- [x] **SITE-01**: When a submitted `site_raw` doesn't resolve via `resolve_site()`'s existing tier 1 (exact `Observatory` match) or tier 2 (live MPC Obscodes API), the approval queue's Site column presents a dropdown of fuzzy-matched `Observatory` candidates for staff to pick from
- [ ] **SITE-02**: Staff can type a code directly and resolve it to an existing `Observatory` or explicitly create a new one, if no fuzzy-matched candidate is correct; no placeholder `Observatory` is ever auto-fabricated (consistent with quick task `260705-l1v`)
- [ ] **SITE-03**: Approving a run whose site a staff member already manually resolved does not get silently re-resolved/overwritten by `CampaignRunDecisionView`'s automatic `resolve_site()` call (fixes a real clobbering bug found during v2.1 research)

### Campaign Display (VIEW)

- [x] **VIEW-05**: Submitter can opt in (single combined flag, default opt-out) to public display of their `contact_person`/`contact_email` on the per-campaign table; unset behaves exactly as today (staff-only)

## v2 Requirements

Deferred to a future release. Tracked but not in current roadmap.

### Campaign Display

- **SCHED-06**: Progressive-disclosure UI showing a run's window visibly narrowing over time (TBD → range → exact night) in the per-campaign table — deferred until the window schema is proven against real re-imported data

### Prior deferred items (carried from v2.0 close, still not committed to a milestone)

- **Stage 4**: full observation-record sync for all facilities
- **ESO-10/ESO-11**: `sync_eso_observation_calendar` command + paired notebook (unblocked by Phase 13's Bypass verdict)
- **SUBMIT-06/07**: trusted-PI self-service approval bypass; submission status lookup via private link

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Full JWST-APT-style multi-state visit-status vocabulary (14 states) | Duplicates existing `run_status`/`approval_status` fields; solves a scheduling-*pipeline* problem FOMO doesn't have (FOMO is a passive coordination hub, not a scheduler) |
| Continuous confidence-score field for date certainty | No reference system (JWST, HST, ToO literature) uses this; the window's presence/absence/width already encodes certainty; a float invites false precision with no data to support it |
| STScI APT/Visit-Status scraping or sync integration | No public bulk API exists for this; mirrors the exact over-scope anti-pattern already flagged in v2.0 research (a generic bot-ingestion layer is a different product) |
| Reusing Gemini's Rap:/Std: ToO trigger-relative window pattern for community-submitted rows | Wrong shape — Gemini's windows are computed from a known trigger timestamp (`ObservationRecord.created`); 3I/ATLAS space-mission rows have no trigger data at all, just human-typed free text |
| `Observatory.obscode` length widening | Presumed unnecessary post-correction — real space-observatory MPC codes (250/274/289) are standard 3-character codes; only revisit if the Phase 18 spike finds a real code that doesn't fit |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SCHED-01 | Phase 18 | Complete |
| SCHED-02 | Phase 19 | Complete |
| SCHED-03 | Phase 19 | Complete |
| SCHED-04 | Phase 19 | Complete |
| SCHED-05 | Phase 19 | Complete |
| ASSET-01 | Phase 20 | Complete |
| ASSET-02 | Phase 20 | Complete |
| IMPORT-01 | Phase 20 | Complete |
| IMPORT-02 | Phase 20 | Complete |
| SITE-01 | Phase 21 | Complete |
| SITE-02 | Phase 21 | Pending |
| SITE-03 | Phase 21 | Pending |
| VIEW-05 | Phase 21 | Complete |

**Coverage:**

- v1 requirements: 13 total
- Mapped to phases: 13 ✓
- Unmapped: 0

---
*Requirements defined: 2026-07-05*
*Last updated: 2026-07-05 after roadmap creation (Phases 18-21 mapped)*
