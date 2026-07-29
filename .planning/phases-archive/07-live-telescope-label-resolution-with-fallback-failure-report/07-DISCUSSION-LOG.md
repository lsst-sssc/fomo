# Phase 7: Live Telescope-Label Resolution with Fallback & Failure Reporting - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-21
**Phase:** 7-Live Telescope-Label Resolution with Fallback & Failure Reporting
**Areas discussed:** Resolution timing, Static dict granularity, Fallback visibility, API timeout value

---

## Resolution timing

| Option | Description | Selected |
|--------|-------------|----------|
| Placed records only | Matches ROADMAP criterion 1 literally; banner records get the coarse fallback immediately, no API call, not counted as a failure | ✓ |
| Every record, placed or not | Attempt the API call regardless of scheduling state | |

**User's choice:** Placed records only (Recommended)
**Notes:** Confirmed with a follow-up question that a banner-stage record's coarse label does not increment the new SYNC-06 fallback counter.

| Option | Description | Selected |
|--------|-------------|----------|
| Uncounted | The fallback/API-failure counter only increments on an actual failed/timed-out/unmapped API call for a placed record | ✓ |
| Counted alongside real failures | Every record currently showing the coarse label increments the same counter | |

**User's choice:** Uncounted (Recommended)
**Notes:** Avoids conflating "not scheduled yet" with "API broke."

---

## Static dict granularity

| Option | Description | Selected |
|--------|-------------|----------|
| Collapse per site | One label per site, mirroring the existing 'FTS'/'Magellan' collapsing precedent | |
| Distinct per telescope | Each fully-qualified siteid-enclid-telid code gets its own unique label | |
| Other (freeform) | Collapse per-site AND per-telescope-class, so 1m0/0m4/2m0 stay distinct within a site | ✓ |

**User's choice:** Collapse per-site and per-telescope-class so aperture classes stay distinct
**Notes:** Middle ground between the two presented options — group by (site, aperture class) pair, not by individual dome.

| Option | Description | Selected |
|--------|-------------|----------|
| "LSC-1m0" style | Site code (uppercased) + hyphen + aperture class | ✓ |
| "LSC 1m0" (space) | Same content, space-separated | |
| Site-only when only one class present | Bare site code when a site has only one class, suffix only for multi-class sites | |

**User's choice:** "LSC-1m0" style (Recommended)

| Option | Description | Selected |
|--------|-------------|----------|
| Keep FTS/FTN/SOAR as-is | Preserve the 3 existing brand-name labels unchanged | |
| Migrate everything to SITECODE-class | Rename coj/ogg/sor entries too for full consistency, accepting a one-time label change | ✓ |

**User's choice:** Migrate everything to SITECODE-class
**Notes:** User explicitly accepted the one-time visible label change on historical CalendarEvents at coj/ogg/sor for full dict consistency. Confirmed this does not touch the separate `telescope_runs.py:SITES` dict used by Stage 1/2.

---

## Fallback visibility

| Option | Description | Selected |
|--------|-------------|----------|
| Add a title prefix | Mirrors the existing [QUEUED]/[EXPIRED]/[CANCELLED]/[FAILED] convention since the calendar day view only shows the truncated title at a glance | ✓ |
| No title change | Rely only on telescope field + description per TELESCOPE-04's literal wording | |

**User's choice:** Add a title prefix (Recommended)
**Notes:** Calendar template (`tom_calendar/partials/calendar.html`) confirmed to only render `event.title` (truncated 16-18 chars) in the day view.

| Option | Description | Selected |
|--------|-------------|----------|
| Placed-failure only | Only add the marker when a placed record's API call actually failed | ✓ |
| Any coarse-labeled record | Apply to every record currently showing the coarse class, banner or placed | |

**User's choice:** Placed-failure only (Recommended)
**Notes:** Avoids redundancy with the existing [QUEUED] prefix on banner-stage records.

| Option | Description | Selected |
|--------|-------------|----------|
| [UNVERIFIED] | Describes the label's confidence state, consistent with TELESCOPE-04's framing | ✓ |
| [API FAIL] | Names the cause directly | |
| [FALLBACK] | Generic term matching the SYNC-06 counter's name | |

**User's choice:** [UNVERIFIED] (Recommended)

---

## API timeout value

| Option | Description | Selected |
|--------|-------------|----------|
| 5 seconds | Tight enough not to stall a batch sync, generous enough for a simple lookup | |
| 10 seconds | More tolerant of a slow response before falling back | ✓ |
| You decide | Leave the exact number to the researcher/planner | |

**User's choice:** 10 seconds
**Notes:** Diverged from Claude's recommended 5s default.

---

## Claude's Discretion

- Exact label string for SOAR's real aperture class (confirm against `tom_observations.facilities.soar` or real data).
- Exact per-site aperture-class inventory for the 5 newly-added sites (elp, lsc, cpt, tfn, tlv).
- How `[UNVERIFIED]` combines with terminal-state prefixes if a record reaches a terminal state after an API-failure fallback was already applied (D-09, open).
- Whether the new fallback counter is reported per-facility in the run summary (strongly suggested by existing Phase 5/6 precedent, but not explicitly discussed).
- Exact helper/method structure for the new per-record API call (extend `get_observation_status()` vs. add a new method).

## Deferred Ideas

None — discussion stayed within phase scope. The status-aware CalendarEvent coloring todo (`2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md`) was reviewed (matched at score 0.9 via keyword overlap) but confirmed out of scope — a UI/template coloring change, distinct from this phase's label-resolution/fallback-reporting scope. Left pending/deferred unchanged.
