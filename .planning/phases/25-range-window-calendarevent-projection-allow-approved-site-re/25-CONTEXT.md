# Phase 25: Range-window CalendarEvent Projection - Context

**Gathered:** 2026-07-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix `_project_calendar_event()` (`solsys_code/campaign_views.py`) so an approved,
site-resolved range-window `CampaignRun` (e.g. the real GS-2026A-FT-115 Gemini South
allocation, `window_start` != `window_end`) becomes visible on the calendar instead of
being silently invisible forever. This closes a gap diagnosed via `/gsd-debug`
(`.planning/debug/range-window-calendar-event.md`, diagnose-only, no code changed) — the
guard's `window_start == window_end` clause was a Phase 19 D-06 behavior-preservation
deferral, not a considered decision, and Phase 23's `TestGeminiFtScenario` later
re-encoded that deferral as if it were the intended contract.

**This discussion materially expanded the technical shape of the fix beyond the debug
spec's original recommendation** — see Decisions below. TBD runs (`window_start` is
`None`), unresolved-site runs (`site` is `None`), and missing-`telescope_instrument` runs
all stay excluded exactly as today; this phase does not touch that part of the guard's
logic, only the `window_start == window_end` equality clause.

Not in scope: any new visual/UI treatment beyond title text (no new box-shadow ring,
no calendar template change); changing `mark_weather_failure`/`mark_cancelled`'s
approval/business-logic guards; the satellite branch's existing whole-day-span
construction (already correct, stays as-is — see Decisions).

</domain>

<decisions>
## Implementation Decisions

### Guard fix (carried forward from the debug spec — locked, not re-discussed)
- **D-01:** `_project_calendar_event()`'s guard changes from requiring
  `run.window_start == run.window_end` to requiring both `run.window_start` and
  `run.window_end` to be truthy (drop the equality clause, add a `window_end`
  truthiness check — the latter is REQUIRED, not optional, or a half-open window with
  `window_end = None` would reach the date-math with an unusable `None`). TBD runs
  (`window_start` is `None`), unresolved-site runs, and missing-`telescope_instrument`
  runs are unaffected — the existing truthiness checks for those already exclude them.
  Full rationale, provenance (Phase 19 D-06), and the confirmation that
  `CalendarEvent.start_time`/`end_time` support this (no model limitation) are in
  `.planning/debug/range-window-calendar-event.md`'s Resolution section — read that
  file in full before planning.

### Ground-branch date math — SUPERSEDES the debug spec's "Option A/B" recommendation
- **D-02:** Ground-based range-window runs get **one `CalendarEvent` per night**, each
  with real dip-corrected sunset/sunrise from `sun_event()` — mirroring
  `load_telescope_runs`' existing `INGEST-01` precedent exactly (`E - S + 1` nightly
  events for a classical-schedule date range). The debug spec's two options (a single
  whole-day-span event, or a single event with dip-corrected first/last-night edges)
  are both **rejected** — the user explicitly chose per-night expansion instead,
  reasoning that the codebase already has this exact idiom for multi-night ranges and a
  single blunt-boundary span would be a regression from that existing physical-accuracy
  standard. **This is a bigger change than the debug spec scoped** (see D-03/D-04 for
  the consequences) — the researcher and planner must treat "one event per night" as
  the locked requirement, not a discretionary implementation choice.
- **D-03:** Because a `CampaignRun` can now own **multiple** `CalendarEvent`s instead of
  exactly one, the existing `CAMPAIGN:{pk}` single-event `url` key (used everywhere
  today: `_project_calendar_event()`'s create path, `_set_run_status()`'s
  existence-check-and-update path, and the `TestCalendarProjection`/`TestGeminiFtScenario`
  test assertions) needs a **new per-night key scheme**. No existing convention in this
  codebase keys multiple `CalendarEvent`s under one parent identity with a `CAMPAIGN:`
  prefix (the closest precedent, `load_telescope_runs`, doesn't use a `url` lookup key
  at all — it keys on `{telescope, instrument, start_time}`). Recommended default,
  left as **Claude's Discretion** for the researcher/planner to confirm or refine:
  `CAMPAIGN:{pk}:{date.isoformat()}` (e.g. `CAMPAIGN:34:2026-07-13`) — keeps the
  existing `CAMPAIGN:{pk}` prefix greppable/filterable while disambiguating nights.
  A single-night run (`window_start == window_end`) should very likely KEEP the
  existing bare `CAMPAIGN:{pk}` key unchanged (no need to touch working, already-tested
  single-night behavior) — confirm this during planning.
- **D-04:** `_set_run_status()` (Phase 23's `mark_weather_failure`/`mark_cancelled`
  handler) currently does a single existence-check
  (`CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').exists()`) then updates that
  ONE event's title/description via `insert_or_create_calendar_event()`. This must
  become "find and update EVERY `CalendarEvent` belonging to this run" (e.g.
  `.filter(url__startswith=f'CAMPAIGN:{run.pk}:')` for multi-night runs, in addition to
  the existing exact-match check for single-night runs) so the `[CANCELLED]`/
  `[WEATHERED]` prefix (Phase 23 D-03) lands on every night's event, not just one. The
  debug spec's claim "`mark_weather_failure`/`mark_cancelled` need NO change" is
  **superseded** by this decision — it needs a real rewrite of the lookup/update loop,
  though its guard logic (`approval_status == APPROVED`, conditional `.update()`,
  never-fabricate-for-a-run-with-no-events) is unaffected and must be preserved exactly.

### Satellite branch — unchanged (not re-discussed, confirmed as still correct)
- **D-05:** The satellite branch's existing whole-day-span construction
  (`window_start` 00:00 UTC → `window_end` 23:59 UTC, ONE event) stays exactly as-is.
  Satellites have no physical "night"/dip-corrected-sunset concept the way ground sites
  do, so per-night expansion (D-02) does not apply there — this asymmetry is
  intentional, not an oversight. Confirm during research that the satellite branch
  is genuinely unreachable for the disputed real case (Gemini I11 is
  `OPTICAL_OBSTYPE`/ground, not satellite) so this stays a documentation note, not a
  functional change.

### Title/description format
- **D-06:** Each per-night event's title gets a window-context suffix distinguishing it
  from an ordinary single-night classical run, e.g.
  `f'{campaign.name}: {telescope_instrument} (window {window_start}..{window_end})'`
  — applied identically to every night's event within the same run (not per-night
  numbering like "Night 2 of 4"). Rationale: unlike `load_telescope_runs`' classical
  nightly events (which are separately, individually confirmed nights and don't need
  this framing), each night here is part of one *awarded allocation* the calendar
  viewer should be able to recognize as connected — the suffix carries that meaning
  even though the events are otherwise visually indistinguishable from separate nights.
  Single-night runs (`window_start == window_end`) keep their existing unsuffixed title
  format unchanged.

### Backfill for already-approved runs
- **D-07:** A **one-off management command** finds already-`APPROVED` `CampaignRun`s
  with a resolved, ground-based site and a range window (`window_start` !=
  `window_end`, both non-null) that have no existing `CAMPAIGN:{pk}*` `CalendarEvent`,
  and runs them through the same (now-fixed) projection logic. This is what makes the
  real GS-2026A-FT-115 row (`CampaignRun` pk=34 in this dev DB, already `APPROVED`
  under the OLD zero-event behavior) actually get its events once this phase ships —
  without this command, pk=34 would silently stay eventless forever, since projection
  only fires on the approve/`resolve_site` POST actions, never retroactively. Rejected
  alternatives: a Django data migration (side-effecting business logic like
  site-resolution-adjacent projection doesn't belong in a migration, and can't be
  safely re-run if something's missed); manual-only (leaves the user's actual
  motivating case, FT-115, unresolved after this phase ships — defeats the point).
  Command name/location/flags are Claude's Discretion.

### Claude's Discretion
- Exact per-night `CalendarEvent.url` key format (D-03) — `CAMPAIGN:{pk}:{date}` is the
  recommended default; confirm/refine during research.
- Whether single-night runs keep the bare `CAMPAIGN:{pk}` key unchanged, or also move
  to the new per-night scheme for consistency (D-03 leans toward "keep unchanged" to
  avoid touching working, already-tested behavior, but confirm during planning).
- Exact backfill management command name, location, and whether it supports a dry-run
  flag (D-07).
- The exact `_set_run_status()` rewrite shape (single combined queryset filter using
  `Q(url=f'CAMPAIGN:{pk}') | Q(url__startswith=f'CAMPAIGN:{pk}:')`, vs. two separate
  filter calls) (D-04).
- Which exact test assertions need which new values — the debug spec's list of "tests
  that must flip 0→1" now needs `count == 1` corrected to `count == (number of nights
  in the window)` for ground range-window cases; re-derive the full list from
  `test_campaign_approval.py`'s current state during planning rather than trusting the
  debug spec's exact counts verbatim (the debug spec assumed single-event ground
  projection, which D-02 supersedes).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The diagnosis this phase implements
- `.planning/debug/range-window-calendar-event.md` — **MANDATORY, read in full.** The
  `/gsd-debug` diagnosis: confirmed root cause, source-verified guard/branch line
  numbers, Phase 19 D-06 provenance, and the original before/after spec. Treat its
  guard-fix recommendation (D-01 above) as still authoritative; treat its ground-branch
  date-math recommendation (Option A/B) as **superseded** by this file's D-02-D-04 —
  the user chose per-night expansion instead during this discussion, which the debug
  session did not anticipate.

### Existing code this phase changes
- `solsys_code/campaign_views.py` — `_project_calendar_event()` (~lines 392-455, the
  guard at ~412, satellite branch ~420-429, ground branch ~442-453 that D-02 rewrites
  for per-night expansion); `_set_run_status()` (~lines 708-762, the
  `CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').exists()` single-event
  lookup D-04 must generalize to multiple events); `_RUN_STATUS_CALENDAR_PREFIX`,
  `_ACTION_TO_RUN_STATUS` (Phase 23 module-level dicts, unaffected).
- `solsys_code/calendar_utils.py` — `insert_or_create_calendar_event()` (the shared
  no-churn create-or-update helper both the per-night create path and the
  backfill command must route through; its `lookup` dict is what makes the new
  per-night `url` key scheme (D-03) mechanically work).

### The precedent this phase's per-night decision mirrors
- `solsys_code/management/commands/load_telescope_runs.py` — `INGEST-01`'s existing
  `E - S + 1` nightly-`CalendarEvent` expansion for a classical-schedule date range;
  D-02 explicitly copies this idiom for `CampaignRun` range windows. Note this command
  does NOT use a `url`-keyed lookup (`{telescope, instrument, start_time}` instead) —
  D-03's new `url` key scheme has no direct precedent to copy verbatim, only the
  per-night *expansion* idea.
- `solsys_code/telescope_runs.py` — `sun_event()`, `horizon_dip()` (the dip-corrected
  sunset/sunrise math D-02's per-night events reuse, same as the ground branch already
  calls today for single-night runs).

### Phase 19 origin of the guard being changed
- `.planning/phases/19-window-schema-migration/19-CONTEXT.md` (D-06, ~lines 82-84) and
  `19-RESEARCH.md` — the original behavior-preservation decision and its explicit
  "range rows only display, not project, in this phase" deferral (19-CONTEXT.md
  ~lines 23-25) that this phase now resolves.

### Tests this phase must deliberately revise
- `solsys_code/tests/test_campaign_approval.py` — `TestCalendarProjection` (the range
  no-event test around line ~349, and the TBD/no-telescope/sun-event-ValueError tests
  that must STAY at count==0), `TestRunStatusChange` (the range mark-status test around
  line ~450), `TestGeminiFtScenario` (Phase 23 Plan 03's Gemini FT-115 end-to-end
  scenario, ~lines 2162-2208 — every "count stays 0" assertion needs re-deriving per
  D-02's per-night model, not just flipped to 1), and the range-resolve test around
  line ~997. Full detail (line numbers, exact assertion text) in the debug file's
  `Resolution.next_planning_phase` section, but re-verify against the current file
  state during planning — do not trust stale line numbers blindly.

### Project conventions
- `CLAUDE.md` — `solsys_code/telescope_runs.py` and `load_telescope_runs.py` are two of
  the four modules with a MANDATORY paired demo notebook
  (`docs/notebooks/pre_executed/telescope_runs_demo.ipynb`,
  `load_telescope_runs_demo.ipynb`). This phase does NOT modify either module's
  behavior (D-02 only reuses `sun_event()`/`horizon_dip()` as-is from
  `campaign_views.py`, a module NOT in that list) — confirm during planning that no
  paired-notebook update is actually triggered; if research finds this phase does end
  up changing `telescope_runs.py` or `load_telescope_runs.py` behavior, the paired
  notebook becomes a must-have, not optional, per CLAUDE.md's documented incident
  history (quick tasks `260619-f7u`/`260620-v9x`).
- `CLAUDE.md` — Django test factories: any new `Target` fixture in tests must use
  `NonSiderealTargetFactory` (not `SiderealTargetFactory`).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `sun_event()` / `horizon_dip()` (`telescope_runs.py`) — already called by the ground
  branch for single-night projection; D-02's per-night expansion calls the same
  function once per date in the window instead of once.
- `insert_or_create_calendar_event()` (`calendar_utils.py`) — the no-churn
  create-or-update helper; D-02's per-night creation and D-04's `_set_run_status()`
  multi-event update both route through it, one call per event.
- `load_telescope_runs.py`'s date-range-expansion loop shape (`E - S + 1` nights) — the
  direct pattern to copy for iterating `window_start..window_end` inclusive.

### Established Patterns
- No-churn idempotency (SYNC-04-style): re-running projection/backfill must never
  create duplicates or touch `modified` on unchanged events — `insert_or_create_calendar_event()`
  already guarantees this per-call; D-02/D-07 just need to call it once per
  (run, night) pair consistently.
- "Site failure/TBD never blocks approval, never fabricates an event" (Phase 22/23) —
  this phase's guard change (D-01) must preserve this: TBD and unresolved-site runs
  still get zero events, approval still succeeds regardless.
- Existence-guarded calendar sync (Phase 23 D-05) — `_set_run_status()` never creates
  an event for a run that never had one; D-04's multi-event rewrite must preserve this
  guard, just generalized from "one event" to "zero or more events."

### Integration Points
- `_project_calendar_event()`'s ground branch (~442-453) — rewritten to loop over
  `window_start..window_end` inclusive, calling `sun_event()` + `insert_or_create_calendar_event()`
  once per night, using the new D-03 per-night `url` key.
- `_set_run_status()` (~708-762) — the single `.filter(url=...).exists()` check becomes
  a queryset covering both the legacy single-night key and the new per-night-prefixed
  keys, iterating to update every matching event.
- New backfill management command (D-07) — a new file under
  `solsys_code/management/commands/`, calling the same rewritten
  `_project_calendar_event()` (or a shared helper it delegates to) for each qualifying
  already-approved `CampaignRun`.

</code_context>

<specifics>
## Specific Ideas

The user's motivating real-world case: the real GS-2026A-FT-115 `CampaignRun` (pk=34 in
this dev DB) is Gemini South GMOS-S, 6.50 awarded hours, window 2026-07-13..2026-07-16
(4 nights), site I11 (Gemini South, ground/OPTICAL), already `APPROVED`. After this
phase ships, the backfill command (D-07) should give it 4 per-night `CalendarEvent`s,
each dip-corrected for that specific night, each titled with the window-context suffix
(D-06) so campaign followers can see it was awarded time — and if it's later marked
weathered/cancelled, all 4 events pick up the `[WEATHERED]`/`[CANCELLED]` prefix (D-04).

</specifics>

<deferred>
## Deferred Ideas

- **Any new calendar visual/UI treatment beyond title text** (e.g. a distinct
  box-shadow ring or color for "part of a multi-night awarded window" vs. a genuinely
  separate classical night) — out of scope for this phase; D-06's title suffix is the
  full extent of the visual differentiation. A future phase could revisit this if the
  title suffix proves insufficient in practice.
- **Extending per-night expansion to the satellite branch** — explicitly rejected
  (D-05); satellites have no physical-night concept, so this isn't deferred as a future
  idea, just noted as intentionally out of scope permanently unless a future need
  emerges.

### Reviewed Todos (not folded)
- **"Extract site/telescope mapping and instrument extraction into own module"**
  (`.planning/todos/pending/2026-06-23-...`) — weak match (score 0.6), already
  reviewed and rejected as not-relevant in Phases 13/18/21/22 (concerns
  `calendar_utils.py`'s LCO/Gemini-sync module, not `campaign_views.py`'s projection
  guard). Still not relevant; not folded.
- **"Rename calendar_utils.py private helpers to reflect shared-module API"**
  (`.planning/todos/pending/2026-07-02-...`) — weak match (score 0.6), same reasoning
  as above; unrelated to this phase's `campaign_views.py`/test-revision scope. Not
  folded.

</deferred>

---

*Phase: 25-range-window-calendarevent-projection-allow-approved-site-re*
*Context gathered: 2026-07-17*
