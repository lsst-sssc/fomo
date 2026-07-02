---
gsd_state_version: 1.0
milestone: v1.7
milestone_name: ESO/VLT Calendar Sync — Feasibility Spike
current_phase: 13
current_phase_name: ESO Feasibility Spike
status: executed
stopped_at: Phase 13 execution complete — both plans done, ESO-04 verdict is Bypass; ready for phase-goal verification
last_updated: "2026-07-01T22:37:31.533Z"
last_activity: 2026-07-01
last_activity_desc: Plan 13-02 complete (ESO-04 Bypass recommendation + ESO-05 future-sync sketch + docs/design/eso_feasibility_spike.rst); La Silla direct-p2api bypass confirmed connecting
progress:
  total_phases: 1
  completed_phases: 1
  total_plans: 2
  completed_plans: 2
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-01 — v1.7 milestone opened)

**Core value:** Determine whether/how ESO/VLT observation sync can work at all, given the installed `tom_eso==0.2.4` cannot create `ObservationRecord` rows or report status through the standard TOM facility API. Produce a Bridge/Bypass/Not-Yet-Feasible decision doc against real ESO P2 credentials — no sync command is built this milestone.
**Current focus:** Phase 13 — ESO Feasibility Spike

## Current Position

Phase: 13 (ESO Feasibility Spike) — EXECUTED (both plans complete, awaiting phase-goal verification)
Plan: 2 of 2 (both complete)
Status: Ready for phase-goal verification
Last activity: 2026-07-01 — Plan 13-02 complete (ESO-04 verdict: Bypass; ESO-05 future-sync sketch; durable .rst summary)

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**

- Prior milestone plans completed: 14 (v1.3-v1.4); v1.6 added 3 plans across Phases 11-12
- Average duration: ~15 min/plan (v1.6 range: ~8-24 min)
- Total execution time: see per-phase breakdown below

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 11 | 2 | ~15 min | ~8 min |
| 12 | 1 | - | - |
| 13 | TBD | - | - |

## Accumulated Context

### Decisions

All v1.0-v1.6 decisions logged in PROJECT.md Key Decisions table.

**v1.7 roadmap decisions:**

- Single-phase structure (Phase 13) chosen for the whole spike: the five requirements form one tightly-coupled investigation loop — ESO-02's real-data capture is gated on ESO-01's credentials, and ESO-04's decision doc synthesizes all of ESO-01..ESO-03 — so credential/access work is not meaningfully separable from data-gathering-and-decision work. `coarse` granularity reinforces folding investigation-only work into one phase.
- Investigation-only milestone: success criteria are written as documented findings and an explicit decision (not "code implemented"), because no `sync_eso_observation_calendar` command ships this milestone.
- Plan 13-01 (commit `48b800d`, corrected `7594910`): Paranal (VLT) production credentials confirmed obtainable/usable (ESO-01); real `getOB()`/`getNightExecutions()` shapes captured, redacted per D-04 (ESO-02); headless credential-sourcing via env-var-backed `ESOAPI(...)` confirmed viable, bypassing `ESOProfile`/session decryption (ESO-03). La Silla (`production_lasilla`) fails via `tom_eso`'s `ESOAPI` wrapper (`P1Error`) — root-caused to `p1api`'s `API_URL` lacking a La Silla entry, while `p2api`'s own `API_URL` does support it; a direct `p2api.ApiConnection('production_lasilla', ...)` bypass (confirmed live, see below) connects successfully, though the run it returned was a Paranal-instrument run already captured under `production` — connectivity confirmed, La-Silla-sourced OB data not yet confirmed.
- Plan 13-02 (commits `7a52db1`/`9a3c8a3`/`7ea0974`/`63cd325`): **ESO-04 verdict is Bypass** — sync straight from `p2api` to `CalendarEvent`, skipping `ObservationRecord` for ESO entirely — grounded directly in Plan 01's evidence (all captured data came from direct P2 API reads, never from `ObservationRecord` creation, so Bridge's premise was never exercised). ESO-05 future-sync sketch: reuse `insert_or_create_calendar_event()` unchanged; synthetic key `ESO:{p2_environment}/{obId}`; banner-only sync recommended as MVP floor, status-aware (using the captured 12-code `obStatus` vocabulary) as a should-have. D-11 Bridge effort-sizing explicitly not applicable (verdict is Bypass). Durable summary written to `docs/design/eso_feasibility_spike.rst`.

### Pending Todos

None.

### Blockers/Concerns

None. Phase 13 execution is complete pending phase-goal verification (`/gsd-execute-phase 13` verify step / `gsd-verifier`).

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| requirement | ESO-10 (`sync_eso_observation_calendar` command) | v2 — contingent on Phase 13 decision | v1.7 open |
| requirement | ESO-11 (paired ESO demo notebook) | v2 — contingent on Phase 13 decision | v1.7 open |

## Session Continuity

Last session: 2026-07-01T22:37:31.518Z
Stopped at: Phase 13 execution complete (both plans); ready for phase-goal verification
Resume file: .planning/phases/13-eso-feasibility-spike/13-DECISION.md

## Operator Next Steps

- Run phase-goal verification for Phase 13 (`gsd-verifier` via
  `/gsd-execute-phase 13`) to confirm the 5 ROADMAP success criteria are met
  and close out the phase.
- Optional future follow-up (not blocking): determine whether the account
  genuinely has La-Silla-sourced OBs/runs distinct from Paranal — the direct
  `p2api.ApiConnection('production_lasilla', ...)` bypass connects
  successfully, but the one run it returned was a Paranal-instrument
  (`UT1`/`FORS2`) run already seen under `production`.
