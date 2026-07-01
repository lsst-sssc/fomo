# Requirements: Telescope Runs Calendar — Stages 1, 2 & 3

**Defined:** 2026-07-01
**Core Value:** Determine whether/how ESO/VLT observation sync can work at all, given research found the installed `tom_eso==0.2.4` cannot create `ObservationRecord` rows or report status through the standard TOM facility API. Produce a decision doc against real ESO P2 credentials — no sync command is built in this milestone.

## v1 Requirements

Requirements for milestone v1.7 (ESO/VLT Calendar Sync — Feasibility Spike).

### ESO Feasibility

- [ ] **ESO-01**: Confirm whether valid ESO P2 API credentials (production, demo/sandbox, or a captured fixture) are actually obtainable and usable for this investigation
- [ ] **ESO-02**: Using the real `p2api` client (already a `tom_eso` dependency), capture an actual OB's status/execution data shape — `obStatus` value(s) and/or a `getOBExecutions()`/`getNightExecutions()` response — documented verbatim, not guessed
- [ ] **ESO-03**: Determine a viable credential-sourcing path for a headless Django management command, given ESO auth today is per-user, session-bound, and Fernet-encrypted (`ESOProfile`) with no `FACILITIES['ESO']` settings fallback
- [ ] **ESO-04**: Produce a written decision doc recommending one of: Bridge (patch/work around `tom_eso` so it populates real `ObservationRecord` rows, then sync as usual), Bypass (sync straight from `p2api` to `CalendarEvent`, skipping `ObservationRecord` for ESO), or Not Yet Feasible (with the specific blocker) — with rationale tied to ESO-01 through ESO-03's findings
- [ ] **ESO-05**: If ESO-04 concludes feasible, sketch what "synced" could reasonably mean for a future ESO sync command (e.g. banner-only window vs. status-aware), scoped as input to a future milestone's requirements — not implemented here

## v2 Requirements

Deferred to a future milestone, contingent on this spike's outcome.

### ESO Sync Implementation

- **ESO-10**: `sync_eso_observation_calendar` management command implementing the Bridge or Bypass approach chosen in ESO-04
- **ESO-11**: Paired demo notebook per CLAUDE.md's companion-notebook convention

## Out of Scope

Explicitly excluded from v1.7. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Implementing `sync_eso_observation_calendar` | This milestone is investigation-only; implementation is deferred to a future milestone once ESO-04's decision is known |
| Patching/forking `tom_eso` upstream | A possible outcome of the Bridge option, not something to build speculatively before the decision is made |
| VLT UT1-4 telescope-level disambiguation | `tom_eso`'s `get_observing_sites()` doesn't expose it today; deferred pending ESO-02's findings on what the real API returns |
| Any code changes to `solsys_code/calendar_utils.py` or existing sync commands | Research confirmed the existing `insert_or_create_calendar_event()` pattern is already facility-agnostic; no changes needed until an ESO command is actually built |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ESO-01 | TBD | Pending |
| ESO-02 | TBD | Pending |
| ESO-03 | TBD | Pending |
| ESO-04 | TBD | Pending |
| ESO-05 | TBD | Pending |
