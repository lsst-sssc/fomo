# Phase 17: Coverage-Gap Analysis - GAP-01 Decision

**Decided:** 2026-07-04
**Status:** Reached during discuss-phase (17-CONTEXT.md D-01/D-02); documented here after the
fact as this phase's citable decision artifact, per GAP-01's success criterion.

## Decision

Coverage-gap "observability" is **dark-window-only**: a date counts as observable when the site
has a non-zero astronomical (-15 degree) dark window that night, computed by reusing
`solsys_code/telescope_runs.py`'s existing `sun_event()`/`get_site()` helpers. This phase does
**not** implement true target-altitude/airmass filtering (which would require
`solsys_code.ephem_utils`'s REBOUND/ASSIST/sorcha pipeline) — that approach is explicitly
rejected for this phase, not merely deferred.

## Rationale

Pre-milestone research already unanimously recommended this approach before phase discussion
began: `.planning/research/ARCHITECTURE.md`'s explicit "Decision" section, `PITFALLS.md`,
`SUMMARY.md`, and `STACK.md` all independently concluded that `telescope_runs.sun_event()` is
sufficient and that importing `ephem_utils` for this feature would be disproportionate. Re-running
that same research question at plan time would have surfaced no new information — the phase
discussion (17-CONTEXT.md D-01) treated it as settled and focused effort on implementation
details (view/cache design) instead.

## Consequences

- No module-scope dependency on the heavy SPICE-loading ephemeris module is introduced anywhere
  in this phase's code (`campaign_gap.py`, its tests, or later plans' views/templates).
- The only ephemeris cost is `sun_event()`'s measured ~520ms/call (17-RESEARCH.md Pitfall 1),
  which directly shapes the cache-with-TTL and 90/180-day date-range-clamp design implemented in
  this plan and the view/UX design in Plans 02-03 (a 90-day default window costs ~47s
  synchronous on a cache miss; the 180-day hard cap bounds the worst case at ~94s).
