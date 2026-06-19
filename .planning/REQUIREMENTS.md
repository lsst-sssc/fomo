# Requirements: Telescope Runs Calendar — v1.3 Full LCO Facility Sync

**Defined:** 2026-06-19
**Core Value:** Generalize `sync_lco_observation_calendar` to correctly sync all LCO-family facilities (LCO + SOAR) for all real site codes and any combination of proposals, fixing the parameter-shape bugs found in v1.2 against real data.

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### Proposal & Facility Selection

- [x] **SELECT-02**: `--proposal` accepts a comma-separated list of proposal codes (e.g. `--proposal A,B,C`), syncing records matching any of them
- [x] **SELECT-03**: `--proposal ALL` syncs every LCO-family `ObservationRecord` regardless of proposal code
- [x] **SELECT-04**: Sync covers both `facility='LCO'` and `facility='SOAR'` records in a single run
- [x] **SELECT-05**: Each record is processed using the facility instance/credentials matching its own `facility` value (LCO vs SOAR), never a single shared facility instance reused across both

### Instrument Extraction

- [ ] **EXTRACT-01**: Instrument type is extracted by scanning `c_1_instrument_type`..`c_5_instrument_type` for the configuration with a populated exposure time, replacing the v1.2 flat-key assumption that doesn't exist in real data
- [ ] **EXTRACT-02**: Extraction is verified against SOAR's multi-configuration shape (e.g. spectrum/arc/lamp-flat) and LCO MUSCAT's per-channel exposure-key shape, not just the single-populated-config shape seen in this DB's 2 real LCO records, so a calibration/non-science config is never mistaken for the meaningful one

### Telescope Label Resolution

- [ ] **TELESCOPE-01**: A verified static site/telescope mapping dict, keyed on the fully-qualified LCO code (`siteid-enclid-telid`), covers all real LCO-network sites — replacing v1.2's 2-site `[ASSUMED]` `SITE_TELESCOPE_MAP`
- [ ] **TELESCOPE-02**: For a record with a placed (scheduled) block, the sync calls the LCO Observation Portal API for that record to resolve the actual site/enclosure/telescope, and maps it to a label via the verified dict
- [ ] **TELESCOPE-03**: If the per-record API call fails, times out, or returns a code not in the verified dict, the sync falls back to a coarse instrument-class label (`1m0`/`0m4`/`2m0` parsed from the instrument type) instead of skipping the record
- [ ] **TELESCOPE-04**: A fallback-labeled event is distinguishable from a verified-label event (the coarse class is visible as the telescope token; the description states the API lookup failed) — a label flip between runs is visible information, not silently hidden churn

### Partial-Failure Handling & Reporting

- [ ] **SYNC-06**: Per-record telescope-API failures are tracked as a distinct counter (e.g. fallback count) from `skipped_count`, and reported in the run's summary line
- [ ] **SYNC-07**: A per-record API failure does not abort the run or skip the record — the record still gets a `CalendarEvent` (with the fallback label), and the rest of the batch continues to sync
- [ ] **SYNC-08**: The per-record API call uses an explicit timeout and a single attempt (no retry/backoff loop); a failure is allowed to resolve itself on the next scheduled run
- [ ] **SYNC-09**: Error/exception output from a failed API call never includes raw response body or credential content (preventing the LCO API key from leaking into logs)

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Calendar Display

- **DISPLAY-01**: Status-aware `CalendarEvent` coloring (telescope/proposal-keyed hash, opacity by status) — already deferred per `.planning/todos/pending/2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md`
- **DISPLAY-02**: A dedicated field distinguishing a fallback-resolved telescope label from a verified one, beyond title/description text — only worth it if operators report confusion from the text-only approach

### Facility Expansion

- **FACILITY-01**: Gemini facility support — different base class (`BaseRoboticObservationFacility`), stub `get_observation_url()` (no portal URL to key the idempotent sync on), different parameter keys/terminal-states vocabulary than LCO
- **FACILITY-02**: Retry/backoff for the per-record LCO API call — only worth revisiting if real operational experience shows a persistently high fallback rate not actually caused by genuine API/network failures

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Gemini facility support | Different base class, no usable portal URL for the idempotent `url`-keyed sync, different parameter/terminal-state vocabulary |
| ESO/NTT facility support | Classically scheduled, already handled by Stage 2 (`load_telescope_runs`); never goes through `ObservationRecord`/queue sync |
| Status-aware `CalendarEvent` coloring | Visual-design decision deferred at v1.2 close; requires a project-level `tom_calendar` template override |
| Retry/backoff on the per-record API call | Adds complexity for a best-effort label; the next scheduled run is the natural retry |
| Caching/memoizing per-record API results | Premature optimization; risks staleness bugs (site placement can change between scheduler runs) worse than the latency it would save |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SELECT-02 | Phase 5 | Complete |
| SELECT-03 | Phase 5 | Complete |
| SELECT-04 | Phase 5 | Complete |
| SELECT-05 | Phase 5 | Complete |
| EXTRACT-01 | Phase 6 | Pending |
| EXTRACT-02 | Phase 6 | Pending |
| TELESCOPE-01 | Phase 7 | Pending |
| TELESCOPE-02 | Phase 7 | Pending |
| TELESCOPE-03 | Phase 7 | Pending |
| TELESCOPE-04 | Phase 7 | Pending |
| SYNC-06 | Phase 7 | Pending |
| SYNC-07 | Phase 7 | Pending |
| SYNC-08 | Phase 7 | Pending |
| SYNC-09 | Phase 7 | Pending |

**Coverage:**

- v1 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-19*
*Last updated: 2026-06-19 after v1.3 ROADMAP.md creation (Phases 5-7)*
