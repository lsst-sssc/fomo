---
phase: 13-eso-feasibility-spike
plan: 02
subsystem: docs
tags: [eso, tom_eso, p2api, feasibility-spike, decision-doc, calendar-sync]

# Dependency graph
requires:
  - phase: 13-eso-feasibility-spike (Plan 01)
    provides: ESO-01/02/03 evidence — real Paranal P2 credential access, captured OB status/execution data shapes, headless credential-sourcing confirmation
provides:
  - "Completed ESO-04 recommendation: Bypass (sync straight from p2api to CalendarEvent, skipping ObservationRecord for ESO), grounded in ESO-01/02/03 findings"
  - "Completed ESO-05 future-sync sketch: banner-only vs status-aware options, synthetic ESO:{p2_environment}/{obId} idempotency key, reusable insert_or_create_calendar_event() landing point, 12-code obStatus vocabulary"
  - "Durable summary docs/design/eso_feasibility_spike.rst for future milestones to reference without digging into .planning/"
affects: [future ESO/VLT calendar sync implementation milestone (ESO-10, ESO-11 deferred requirements)]

# Tech tracking
tech-stack:
  added: []
  patterns: [evidence-then-recommendation decision doc structure (mirrors research/SUMMARY.md), durable RST design-doc summary alongside a full-detail planning-doc companion]

key-files:
  created:
    - docs/design/eso_feasibility_spike.rst
  modified:
    - .planning/phases/13-eso-feasibility-spike/13-DECISION.md

key-decisions:
  - "Verdict is Bypass, not Bridge: Plan 01's evidence was gathered entirely via direct p2api/ESOAPI reads (getOB, getOBExecutions, getNightExecutions), never via ObservationRecord creation — Bypass is the only option this spike's real-data path actually demonstrated end-to-end."
  - "D-11 Bridge effort-sizing is explicitly omitted since the verdict is Bypass, not Bridge (D-11 only applies when Bridge is recommended)."
  - "La Silla's revised finding (direct p2api.ApiConnection bypass of tom_eso's ESOAPI/p1api wrapper) reinforces the Bypass verdict — the same bypass-shaped strategy already works for the La Silla connection problem."

patterns-established:
  - "Future ESO sync command reuses insert_or_create_calendar_event() unchanged and follows Gemini's GEM:{prog}/{observation_id} synthetic-key precedent as ESO:{p2_environment}/{obId}."

requirements-completed: [ESO-04, ESO-05]

coverage:
  - id: D1
    description: "13-DECISION.md ESO-04 Recommendation section completed with a single bold Bypass verdict, rationale explicitly tied to ESO-01/02/03 findings"
    requirement: "ESO-04"
    verification:
      - kind: other
        ref: "grep -Eq 'Bridge|Bypass|Not Yet Feasible' .planning/phases/13-eso-feasibility-spike/13-DECISION.md"
        status: pass
    human_judgment: true
    rationale: "Whether the verdict is genuinely justified by the ESO-01/02/03 evidence (not just present as text) requires human judgment of the rationale's soundness."
  - id: D2
    description: "13-DECISION.md ESO-05 Future-sync sketch completed (banner-only vs status-aware, synthetic key, reusable landing point, obStatus vocabulary); D-11 effort-sizing correctly omitted since verdict is Bypass not Bridge"
    requirement: "ESO-05"
    verification:
      - kind: other
        ref: "grep -q '## Future-sync sketch (ESO-05)' .planning/phases/13-eso-feasibility-spike/13-DECISION.md"
        status: pass
    human_judgment: true
    rationale: "Whether the sketch is genuinely scoped as future-input (not implemented) and technically sound requires human judgment."
  - id: D3
    description: "docs/design/eso_feasibility_spike.rst created alongside telescope_runs_calendar.rst, stating the same Bypass verdict up front, with a list-table method-availability/status matrix, and no credential-adjacent content"
    verification:
      - kind: other
        ref: "test -f docs/design/eso_feasibility_spike.rst && grep -Eq 'Bridge|Bypass|Not Yet Feasible' docs/design/eso_feasibility_spike.rst && grep -q 'list-table' docs/design/eso_feasibility_spike.rst && head -2 docs/design/eso_feasibility_spike.rst | grep -q '===='"
        status: pass
    human_judgment: true
    rationale: "Whether the .rst genuinely matches telescope_runs_calendar.rst's skeleton style and contains no credential-adjacent content requires human visual/content review."

duration: 15min
completed: 2026-07-01
status: complete
---

# Phase 13 Plan 02: ESO Feasibility Spike — Decision Synthesis Summary

**Recommends Bypass (sync straight from p2api to CalendarEvent, skipping ObservationRecord) for a future ESO/VLT calendar sync, grounded directly in Plan 01's real Paranal P2 API evidence, with a durable docs/design/eso_feasibility_spike.rst summary for future milestones.**

## Performance

- **Duration:** ~15 min
- **Completed:** 2026-07-01T22:30:55Z
- **Tasks:** 2/2 completed
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments

- Completed `13-DECISION.md`'s `## Recommendation (ESO-04)` section with a single bold **Bypass** verdict, with rationale explicitly tracing back to each of ESO-01, ESO-02, and ESO-03's findings from Plan 01 — showing that Plan 01's entire evidence-gathering path (direct `p2api`/`ESOAPI` reads: `getOB()`, `getOBExecutions()`, `getNightExecutions()`) is exactly the Bypass data path, never touching `ObservationRecord` creation (Bridge's premise).
- Completed the `## Future-sync sketch (ESO-05)` section: reusable `insert_or_create_calendar_event()` landing point, synthetic `ESO:{p2_environment}/{obId}` idempotency key (following Gemini's `GEM:{prog}/{observation_id}` precedent), banner-only vs. status-aware sync options (status-aware is real-data-supported by ESO-02's captured `'P'` and `'M'` obStatus values, but requires an open "which nights to poll" policy decision), and the full 12-code ESO P2 `obStatus` vocabulary table. Explicitly noted D-11's Bridge effort-sizing estimate does not apply since the verdict is Bypass.
- Created `docs/design/eso_feasibility_spike.rst`, following `telescope_runs_calendar.rst`'s exact skeleton (title underlined with `=`, `Background`/`Key finding`/list-table/`Future scope` sections with `-` underlines), stating the same Bypass verdict as a bold one-liner up front, so future milestones can reference the verdict without digging into `.planning/`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Complete 13-DECISION.md — ESO-04 recommendation and ESO-05 future-sync sketch** - `7a52db1` (docs)
2. **Task 2: Write the durable summary docs/design/eso_feasibility_spike.rst** - `9a3c8a3` (docs)

_No TDD tasks in this plan (documentation-only plan, no application code)._

## Files Created/Modified

- `.planning/phases/13-eso-feasibility-spike/13-DECISION.md` - Recommendation (ESO-04: Bypass) and Future-sync sketch (ESO-05) sections completed; status line updated to reflect completion.
- `docs/design/eso_feasibility_spike.rst` - New durable design-doc summary, following the `telescope_runs_calendar.rst` skeleton, stating the same Bypass verdict, including a method-availability/status list-table and the 12-code ESO P2 `obStatus` vocabulary.

## Decisions Made

- **Bypass, not Bridge:** all of Plan 01's captured evidence (ESO-01/02/03) came from direct `p2api`/`ESOAPI` reads against real OB/execution data — never from creating an `ObservationRecord` row. Bridge's premise (patching `tom_eso` so it populates real `ObservationRecord` rows, then syncing as usual) was never exercised or evidenced in this spike, per the phase's D-08 read-only guardrail. Bypass is the option this spike's real-data path actually demonstrates end-to-end, and it also generalizes cleanly to the La Silla connection workaround identified in Plan 01 (bypassing `tom_eso.eso_api.ESOAPI`/`p1api` to call `p2api.ApiConnection('production_lasilla', ...)` directly).
- **D-11 effort-sizing omitted:** per the phase's D-11 decision, the Bridge effort-sizing estimate (which `tom_eso` methods would need real implementations, small-patch/moderate-fork/larger-undertaking sizing) is only required when the verdict is Bridge. Since the verdict is Bypass, this section explicitly states the estimate does not apply, rather than silently omitting it without explanation.
- **Status-aware sync flagged as real-data-supported but not the MVP floor:** ESO-02 captured both a pre-execution (`'P'`) and post-execution (`'M'`) real `obStatus` value, proving status-aware sync is reachable — but the "which night(s) to poll per OB" policy question is genuinely open (no single "current status" endpoint exists in the P2 API), so the sketch recommends banner-only as the safer MVP floor and status-aware as a should-have layered on top, consistent with the FEATURES.md research's original P1/P2 prioritization.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. This plan is documentation-only.

## Next Phase Readiness

Phase 13 (ESO Feasibility Spike) is now complete: both plans (13-01 evidence-gathering, 13-02 decision synthesis) are committed, and the milestone's core deliverable — a defensible Bridge/Bypass/Not-Yet-Feasible recommendation grounded in real ESO P2 API evidence — exists in both the full-detail `13-DECISION.md` and the durable `docs/design/eso_feasibility_spike.rst`. The deferred requirements ESO-10 (`sync_eso_observation_calendar` command) and ESO-11 (paired ESO demo notebook), noted in STATE.md's Deferred Items, can now proceed in a future milestone against this spike's Bypass recommendation — the future-sync sketch (synthetic key, banner-only vs. status-aware options, obStatus vocabulary) is ready to seed that milestone's requirements. No blockers.

---
*Phase: 13-eso-feasibility-spike*
*Completed: 2026-07-01*
