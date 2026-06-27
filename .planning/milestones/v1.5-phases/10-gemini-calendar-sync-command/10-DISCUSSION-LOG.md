# Phase 10: Gemini Calendar Sync Command - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-26
**Phase:** 10-Gemini Calendar Sync Command
**Areas discussed:** ToO-type fallback when settings missing, Multi-obsid submissions, Demo notebook fixture design, Command output format

---

## ToO-type fallback when settings missing

| Option | Description | Selected |
|--------|-------------|----------|
| Skip + skipped counter | Per GEM-WINDOW-02: no CalendarEvent, increment 'skipped', log warning naming missing prog/obs_code | ✓ |
| Wide fallback window + warn | Create event with 7-day window anchored on record.created, log warning | |
| You decide | Implement per GEM-WINDOW-02 as written (skip + counter) with descriptive log | |

**User's choice:** Skip + skipped counter
**Notes:** Skip is the right call — an event with unknown time bounds on the calendar would mislead operators. The warning log should name the specific `prog` and `obs_code` missing from settings so the operator knows exactly what to add to `FACILITIES['GEM']['programs']`.

---

## Multi-obsid submissions

### Does FOMO ever submit multiple obsids at once?

| Option | Description | Selected |
|--------|-------------|----------|
| Never — always single obsid per trigger | Always single, can assert len==1 | |
| Yes, occasionally 2 obsids at once | Standard + rapid of the same template | ✓ |
| Unsure | Defensive: use first entry + warn if >1 | |

**User's choice:** Yes, occasionally 2 obsids at once (standard + rapid of the same template)

### Which obsid to use when params['obsid'] has >1 entry?

| Option | Description | Selected |
|--------|-------------|----------|
| Try Std./Rap. matching from observation_id | Match each ObservationRecord.observation_id to the obsid list via suffix | Attempted |
| First entry + warning log | Use params['obsid'][0] + log the full list | Deferred to Claude |
| You decide | Claude's discretion | ✓ |

**User's choice:** Initially wanted matching from observation_id; after confirming that `observation_id` is only the numeric suffix (not the full `GS-2025A-T-001-MM-7` string), deferred to Claude's discretion.

**Notes:** Claude's call — use `params['obsid'][0]` for both instrument lookup and ToO-type detection. Log a WARNING listing the full obsid list when >1 entries found. The instrument label is typically identical for Std./Rap. variants of the same physical instrument; the window assignment may be off on the second record (documented limitation, not fixable from stored data alone).

---

## Demo notebook fixture design

### Program ID style

| Option | Description | Selected |
|--------|-------------|----------|
| Placeholder IDs (GS-2026A-T-999) | Clearly fake IDs | |
| Real FOMO program IDs (masked) | Actual program structure with passwords blanked | |
| You decide | Claude's discretion on fixture shape | |

**User's choice (free text):** Use realistic program names from ToO programs at `https://www.gemini.edu/observing/schedules-and-queue/queue-summary-bands-dd-lp-ft-pw?semester=2026A&site=South&queue=SQ` but change the program number at the end to `999`.

**Notes:** Draw the program-type letter and semester from the actual 2026A Gemini South ToO queue listing; end in `999`. This gives realistic-looking IDs (not `GS-YYYYS-T-NNN`) that still can't be confused with real submissions.

### Scenarios to cover

| Scenario | Description | Selected |
|----------|-------------|----------|
| Explicit windowDate/windowTime window | Primary happy path (GEM-WINDOW-01) | ✓ |
| Rap: derived window (24h) | No explicit window, Rap: obs code (GEM-WINDOW-02) | ✓ |
| Std: derived window (7d) | No explicit window, Std: obs code (GEM-WINDOW-02) | ✓ |
| ON_HOLD + idempotent re-run | ready='false' → [ON_HOLD]; re-run = no new events, no modified churn | ✓ |

**User's choice:** All four scenarios selected.
**Notes:** Settings fixture should patch `FACILITIES['GEM']['programs']` with at least one Rap: and one Std: obs code for the notebook program. Password placeholder: `'password': '[redacted]'`.

---

## Command output format

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror LCO format | Per-facility lines: 'Gemini South: created: N, updated: N, unchanged: N, skipped: N' | ✓ |
| Simpler single line | 'Done. created: N, ...' — no per-facility breakdown | |
| Per-site breakdown | Separate South / North lines (same as Mirror LCO, but explicitly split) | |

**User's choice:** Mirror LCO format
**Notes:** Operators already familiar with the LCO sync output; consistent format reduces cognitive load. No `extraction_failed` or `telescope_api_failed` counters needed — those are LCO-specific. Separate South/North lines naturally fall out of per-site counter tracking.

---

## Claude's Discretion

- **Multi-obsid handling**: Use `params['obsid'][0]` for instrument/ToO-type lookup. Log WARNING with full obsid list when >1 entries found.
- **Password scrubbing mechanism** (not discussed explicitly, derived from GEM-SECURE-01): strip `password` key from params dict immediately at record load time, before any logging or field derivation.
- **`--proposal` filter flag**: May add if it fits naturally given the LCO analog, but not required.

## Deferred Ideas

- **`--proposal` filter flag** — analogous to LCO sync; noted for potential inclusion or future phase.
- **GOATS / GPP integration** (GEM-GPP-01/02) — requires Python < 3.11 and GOATS not installed; future work.
- **Live Gemini ODB status polling** — `GEMFacility.get_observation_status()` is a stub; future work if/when Gemini ODB API becomes accessible.
