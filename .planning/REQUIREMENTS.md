# Requirements: Telescope Runs Calendar — Stages 1, 2 & 3

**Defined:** 2026-06-27
**Core Value:** Clear all deferred technical debt and display polish items accumulated across v1.3–v1.5, leaving the codebase clean before Stage 4.

## v1.6 Requirements

### Refactoring

- [ ] **REFAC-01**: `SITE_TELESCOPE_MAP`, `_extract_instrument`, and related LCO/SOAR helpers are importable from a standalone `solsys_code/` module, not from `sync_lco_observation_calendar`
- [ ] **REFAC-02**: All three management commands (`load_telescope_runs`, `sync_lco_observation_calendar`, `sync_gemini_observation_calendar`) use `insert_or_create_calendar_event()` for their CalendarEvent create-or-update logic; the duplicated pattern is removed from each command; "upsert" in `docs/design/telescope_runs_calendar.rst` and `.planning/MILESTONES.md` replaced with plain English or `insert_or_create_calendar_event`

### Display

- [ ] **DISPLAY-08**: Calendar event title text renders in white or black based on the relative luminance of the proposal palette background, meeting WCAG AA 4.5:1 contrast ratio against every palette color
- [ ] **DISPLAY-09**: `CalendarEventTelescopeLabel` data for visible calendar events is loaded in a single prefetch query, not per-event — the N+1 pattern is eliminated from the calendar template

## Future Requirements

### Stage 4

- Extend facility sync beyond LCO/SOAR/Gemini to all remaining facilities (full issue #37 Stage 4)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Stage 4 full observation-record sync | Future milestone; scope is tech debt only |
| Reworking astroplan visibility/airmass plots | Separate feature, unrelated to calendar sync |
| Distinguishing Magellan Baade vs Clay | Deliberately ambiguous; both at Las Campanas, same ephemeris |
| `Observatory.short_name` data-driven lookup for SITES | Stage 2+ consideration, not required |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| REFAC-01 | Phase 11 | Pending |
| REFAC-02 | Phase 11 | Pending |
| DISPLAY-08 | Phase 12 | Pending |
| DISPLAY-09 | Phase 12 | Pending |

**Coverage:**
- v1.6 requirements: 4 total
- Mapped to phases: 4
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-27*
*Last updated: 2026-06-27 — Phase assignments added (Phase 11: REFAC-01, REFAC-02; Phase 12: DISPLAY-08, DISPLAY-09)*
