# Phase 23: Weather/Storm Cancellation Handling - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-16
**Phase:** 23-weather-storm-cancellation-handling-give-staff-a-way-to-mark
**Areas discussed:** Classical-run cancellation trigger, Cancelled-event visual treatment, CampaignRun status entry point + calendar sync, Gemini scope, Prefix split, Action placement

---

## Classical-run cancellation trigger (Trigger)

| Option | Description | Selected |
|--------|-------------|----------|
| Re-run load_telescope_runs | Edit source schedule file to add the recognized 'cancelled' status word/parenthetical and re-ingest — parser and idempotent update path already exist | ✓ |
| New standalone command | Dedicated management command to flip a CalendarEvent's status without touching the source file | |
| Calendar UI action | Button/action directly on the calendar page — biggest scope increase | |

**User's choice:** Re-run load_telescope_runs (Recommended)
**Notes:** No additional command or UI needed for classical events — this phase only wires the already-embedded status text into a visible treatment.

---

## Cancelled/weathered visual treatment (Visual)

| Option | Description | Selected |
|--------|-------------|----------|
| Title prefix, matching LCO pattern | [CANCELLED]/[WEATHERED] title prefix, reusing LCO/SOAR sync's existing terminal-state mechanism | ✓ |
| Dashed/dimmed styling only | Visual-only, like the existing fallback-label dashed border | |
| Both title prefix and dimmed styling | Most visible, but needs care to avoid collision with Phase 8/9 visual layers | |

**User's choice:** Title prefix, matching LCO pattern (Recommended)
**Notes:** Consistent vocabulary with the existing calendar's terminal-state convention.

---

## CampaignRun status entry point + calendar sync (Campaign Sync)

| Option | Description | Selected |
|--------|-------------|----------|
| Approval queue action + update calendar marker | New status-change action on the approval queue, updates the existing CAMPAIGN:{pk} CalendarEvent in place | ✓ |
| Approval queue action + delete calendar event | Same action, but removes the CalendarEvent entirely | |
| Per-campaign table inline edit, no calendar sync | Simplest, but doesn't satisfy the "kept in sync" goal | |

**User's choice:** Approval queue action + update calendar marker (Recommended)
**Notes:** Keeps the visual record that time was lost to weather; consistent with the classical-event treatment.

---

## Gemini FT program scope (Gemini Scope)

**Initial framing (before user clarification):** presented as a binary in/out-of-scope choice. User asked a clarifying factual question instead of answering directly: does `ObservationRecord(facility='GEM')` support queue-style observing with an awarded-hours total across a window, similar to LCO/FTS?

**Investigation:** Checked `tom_observations/facilities/gemini.py` (site-packages). Confirmed `GEMFacility`/`GEMObservationForm` models exactly ONE submitted ToO-style request with ONE window (`windowDate`/`windowTime`/`windowDuration`) per `ObservationRecord` — no concept of an aggregate awarded-hours allocation executed as multiple visits. Real submission requires actual Gemini ODB API credentials and a registered program ID; this DB currently has 0 GEM `ObservationRecord`s.

**User's follow-up:** wants the "window of possibility" visible for campaign planning, explicitly acknowledging real ODB access/submission is out of scope (most people, including the user, lack Gemini ODB access).

**Reframing:** recognized that `CampaignRun.window_start`/`window_end` (Phase 19 schema) already models exactly this shape ("we know roughly when, not exactly when"), and the "Didymos 2026" `TargetList` (pk=1) already exists in the DB with 0 runs. Proposed representing GS-2026A-FT-115 as an informational `CampaignRun` row under that campaign — no new Gemini integration code needed.

| Option | Description | Selected |
|--------|-------------|----------|
| Real ObservationRecord/ODB sync | Extend sync_gemini_observation_calendar-style integration to cover FT programs | ✗ (out of scope) |
| Informational CampaignRun window entry | Represent FT-115's window via existing CampaignRun/window_start/window_end machinery, no ODB sync | ✓ (in scope) |

**User's choice:** "yes, lock it in" — confirmed the reframed split (informational CampaignRun entry in scope; real ODB sync out of scope).
**Notes:** This ties directly into the phase's core mechanism — once run_status changes are staff-settable (D-04), the same action applies uniformly to this Gemini row.

---

## Prefix split for CampaignRun terminal states (Prefix Split)

| Option | Description | Selected |
|--------|-------------|----------|
| Two distinct prefixes | [CANCELLED] for cancelled/withdrawn, [WEATHERED] for weather/tech loss — matches LCO's 4-distinct-prefix precedent | ✓ |
| One shared [CANCELLED] label | Simpler, but distinction only visible in the run_status badge, not on the calendar | |

**User's choice:** Two distinct prefixes (Recommended)
**Notes:** Exact wording of the WEATHER_TECH_FAILURE prefix left to Claude's discretion.

---

## Action placement for run_status change (Action Placement)

| Option | Description | Selected |
|--------|-------------|----------|
| New action row on the Decided table | Structural change — Decided table currently renders zero actions (show_actions=False) | ✓ |
| Somewhere else / you decide | Defer exact placement to the planner | |

**User's choice:** New action row on the Decided table (Recommended)
**Notes:** Exact control shape (dropdown vs per-status buttons) left to Claude's discretion.

---

## Claude's Discretion

- Exact wording/format of the [WEATHERED]-style prefix for WEATHER_TECH_FAILURE.
- Exact UI control shape for the Decided-table status-change action.
- Whether new prefixes need adding to `calendar_display_extras.py`'s `_TERMINAL_PREFIXES` tuple to also pick up the Phase 8/9 status box-shadow ring treatment.
- Site resolution for the Gemini CampaignRun entry — whether "Gemini-South" already resolves via the existing MPC candidate pool or needs a new local Observatory row.

## Deferred Ideas

- Real Gemini `ObservationRecord`/ODB API sync for FT-115 or any Gemini program — explicitly out of scope for this phase (D-06). A future phase's concern if genuine proposal-level allocation tracking via the real Gemini API is ever wanted.
