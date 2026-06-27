# Requirements: Telescope Runs Calendar — Stage 4 Gemini

**Defined:** 2026-06-26
**Core Value:** A `sync_gemini_observation_calendar` command that syncs submitted Gemini ToO ObservationRecords to CalendarEvent window banners, using explicit submission windows when present and ToO-type-derived defaults when not.

## v1.5 Requirements

Requirements for Gemini calendar sync milestone. Each maps to roadmap phases.

### Gemini Calendar Sync

- [ ] **GEM-SELECT-01**: `sync_gemini_observation_calendar` command syncs all `ObservationRecord(facility='GEM')` records
- [ ] **GEM-WINDOW-01**: Each synced record creates one `CalendarEvent`; `start_time`/`end_time` from `windowDate`/`windowTime`/`windowDuration` parameters when present
- [ ] **GEM-WINDOW-02**: Records without explicit window fall back to a ToO-type-derived window anchored on `ObservationRecord.created`: `Rap:` prefix → `[created, created + 24h]`; `Std:` prefix → `[created + 24h, created + 7d]`; type unresolvable → skip with `skipped` counter
- [ ] **GEM-KEY-01**: `CalendarEvent.url` set to `f"GEM:{prog}/{observation_id}"` — stable, unique, never empty
- [ ] **GEM-TELE-01**: `telescope` field derived from program ID prefix: `GS-*` → `'Gemini South'`, `GN-*` → `'Gemini North'`
- [ ] **GEM-INSTR-01**: `instrument` field from settings description for the obs code (strip `Std:`/`Rap:` prefix); fallback to raw obs code if description not found in settings
- [ ] **GEM-PROP-01**: `proposal` field set from `params['prog']`
- [ ] **GEM-STATUS-01**: `[ON_HOLD]` title prefix when `params['ready'] == 'false'`; clean title (`{telescope} {instrument} ToO`) otherwise
- [ ] **GEM-NOCHURN-01**: Re-running the command on the same records creates no duplicate `CalendarEvent`s and does not update `modified` on unchanged records
- [ ] **GEM-SECURE-01**: The `password` key in `parameters` is never logged, printed, or included in any exception message or `CalendarEvent` field

## Future Requirements

### GOATS / GPP Integration

- **GEM-GPP-01**: When GOATS/GPP integration is available, replace window-banner approach with real `scheduled_start`/`scheduled_end` from Gemini scheduler
- **GEM-GPP-02**: Replace constructed `GEM:{prog}/{obs_id}` key with real portal URL from `GEMFacility.get_observation_url()` once un-stubbed

## Out of Scope

| Feature | Reason |
|---------|--------|
| `CalendarEventTelescopeLabel` sidecar for Gemini events | Telescope is deterministic from program prefix — missing-row = "verified" convention applies |
| Live Gemini ODB status polling | `GEMFacility.get_observation_status()` is a stub returning empty state |
| GOATS / GPP integration | Not installed; requires Python < 3.11; future work if FOMO migrates |
| Gemini archive data retrieval | Out of scope for calendar sync |
| Multi-obsid overlap deduplication | Acceptable wrinkle — multiple obsids per submission each get their own event with overlapping derived windows |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| GEM-SELECT-01 | Phase 10 | Pending |
| GEM-WINDOW-01 | Phase 10 | Pending |
| GEM-WINDOW-02 | Phase 10 | Pending |
| GEM-KEY-01 | Phase 10 | Pending |
| GEM-TELE-01 | Phase 10 | Pending |
| GEM-INSTR-01 | Phase 10 | Pending |
| GEM-PROP-01 | Phase 10 | Pending |
| GEM-STATUS-01 | Phase 10 | Pending |
| GEM-NOCHURN-01 | Phase 10 | Pending |
| GEM-SECURE-01 | Phase 10 | Pending |

**Coverage:**
- v1.5 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-26*
*Last updated: 2026-06-26 — traceability updated after roadmap creation (all 10 mapped to Phase 10)*
