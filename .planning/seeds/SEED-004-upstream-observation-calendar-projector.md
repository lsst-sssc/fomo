---
id: SEED-004
status: dormant
planted: 2026-09-03T21:30:00.000Z
planted_during: v2.3 Phase 32 — /gsd-explore session on observation-first calendar layering
trigger_when: >
  When BOTH hold: (1) the /gsd-spike proving ObservationRecord -> tom_calendar.CalendarEvent
  projection (per-record, ObservationGroup identity, observation_change_state hook trigger,
  allocation-night retirement) has passed against the real KEY2026B-004 records, AND
  (2) it is known whether tom_calendar is maintained by the TOM Toolkit org
  (see .planning/research/questions.md) — that decides whether the projector is contributed
  to tom_calendar, to tom_observations, or published as a standalone tom_* plugin.
scope: medium
---

# SEED-004: Upstream the observation -> calendar projector

## Idea

The base-layer projector described in
`.planning/notes/observation-first-calendar-layering.md` is deliberately campaign-free: one
`tom_calendar.CalendarEvent` per `ObservationRecord`, grouped by `ObservationGroup`,
narrowing from the request window to the placed block to the observed block as TOM's own
`updatestatus` refreshes `scheduled_start/end`, triggered by the existing
`observation_change_state` hook. Nothing in it is specific to FOMO or to Solar System
targets, and TOM Toolkit currently has no ObservationRecord -> calendar feature at all
(tom_calendar is a generic manual event calendar with no ObservationRecord link).

Two companion pieces are equally generic and already exist in FOMO:

- `backfill_lco_observations` — import a proposal's existing observations from an OCS
  portal into ObservationRecords/ObservationGroups/Targets (TOM only creates records on
  submission).
- the telescope-name -> observing-site -> sunset/sunrise-bounded block helper
  (`solsys_code/telescope_runs.py`).

## Why later, not now

The projector must first prove itself in FOMO (the spike), and the contribution target
depends on who maintains tom_calendar. Joins the existing upstream track (SEED-001,
SEED-002).

## What "acting on it" looks like

- Confirm tom_calendar maintainership; pick the contribution target.
- Extract the projector + hook receiver + sweep command into a facility-agnostic module
  (OCS today; the block-selection rule is already the same one `LCOFacility.get_observation_status`
  uses).
- Open an issue/PR upstream with the FOMO implementation as the reference.
