# Phase 23: Weather/Storm Cancellation Handling - Context

**Gathered:** 2026-07-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Give staff a way to mark scheduled telescope time as weathered-out/cancelled and have that
status visibly reflected wherever it's tracked. Two currently-disconnected subsystems are in
scope: (1) classical-schedule `CalendarEvent`s (Magellan/NTT/FTS via `load_telescope_runs`),
which already recognize a `cancelled` status word but only embed it as inert description
text with zero visual differentiation on the calendar; (2) `CampaignRun.run_status`, which
already has `CANCELLED`/`WEATHER_TECH_FAILURE` choices but is only editable via Django admin
and has no calendar-sync behavior at all. A third, narrower piece is in scope: representing a
Gemini Fast-Turnaround program's observing window informationally via the existing
`CampaignRun` window-schema (no real Gemini ODB/`ObservationRecord` integration).

Triggered by a real incoming storm expected to affect two scheduled Magellan runs
(Baade IMACS 17-18 July, Clay Lightspeed 18-20 July) and a Gemini FT program
(GS-2026A-FT-115, 13-16 July).

</domain>

<decisions>
## Implementation Decisions

### Classical-schedule cancellation (load_telescope_runs)
- **D-01:** Staff mark a classical run cancelled by editing the source schedule file to add
  the already-recognized `cancelled` status word/parenthetical (`_resolve_status()`'s
  `KNOWN_STATUSES`), then re-running `load_telescope_runs` against it. This is the existing
  idempotent create-or-update path (`insert_or_create_calendar_event`) — this phase does NOT
  add a new command or UI action for classical events, only wires the already-embedded
  `Status: cancelled` value into a visible treatment.
- **D-02:** Visual treatment is a title prefix, matching the LCO/SOAR sync's existing pattern
  (`WINDOW_EXPIRED`→`[EXPIRED]`, `CANCELED`→`[CANCELLED]`, etc. in
  `sync_lco_observation_calendar.py`). `KNOWN_STATUSES` currently has only one relevant word
  (`cancelled`) — classical events get a single `[CANCELLED]` prefix; there is no
  classical-schedule equivalent of `WEATHER_TECH_FAILURE` (see D-03 for why that distinction
  only applies to `CampaignRun`).

### CampaignRun.run_status staff UI + calendar sync
- **D-03:** `CANCELLED` and `WEATHER_TECH_FAILURE` get two DISTINCT title prefixes on the
  linked `CalendarEvent` (e.g. `[CANCELLED]` vs `[WEATHERED]` — exact wording is Claude's
  discretion, but they must render differently), not one shared label. Precedent: LCO's own
  mapping uses 4 distinct prefixes for 4 distinct meanings
  (`EXPIRED`/`CANCELLED`/`FAILED`/`FAILED`), so a shared label for two semantically different
  states would be a regression from that existing convention.
- **D-04:** The status-change action lives on the approval queue's **Decided** table (already-
  approved runs). This is a real structural change: `ApprovalQueueTable`'s Decided table
  currently renders `show_actions=False` and `render_actions()` returns `''` unconditionally
  for it — this phase adds the first action any Decided row has ever had. Exact control shape
  (dropdown+submit vs per-status buttons) is Claude's discretion.
- **D-05:** Setting `run_status` to `CANCELLED`/`WEATHER_TECH_FAILURE` on an already-approved
  run **updates the existing `CAMPAIGN:{pk}` `CalendarEvent` in place** (title/description),
  reusing `insert_or_create_calendar_event()`'s no-churn update path — it does NOT delete the
  event. Keeps the visual record that time was lost to weather, consistent with the classical-
  event treatment (D-02) rather than silently freeing the calendar slot.

### Gemini FT program visibility (informational only)
- **D-06:** GS-2026A-FT-115 is represented as an informational `CampaignRun` row under the
  existing "Didymos 2026" `TargetList` (pk=1 in this DB, currently 0 runs), using the
  `window_start`/`window_end` pair from Phase 19's window schema — NOT a real Gemini
  `ObservationRecord`/ODB API submission. `tom_observations.facilities.gemini.GEMFacility`
  models one submitted ToO-style request with exactly ONE window per record
  (`windowDate`/`windowTime`/`windowDuration`) — it has no concept of "N hours awarded,
  executed as several visits scattered across a window," and requires real Gemini ODB
  credentials + a registered program ID to submit through. Real ODB sync stays explicitly
  OUT of scope for this phase (see Deferred Ideas).
- **D-07:** This CampaignRun entry is subject to the SAME run_status mechanism as D-03/D-04/
  D-05 — if the storm also wipes FT-115, staff mark it the same way as the Magellan runs, no
  special-casing.

### Claude's Discretion
- Exact wording/format of the `[WEATHERED]`-style prefix for `WEATHER_TECH_FAILURE` (D-03).
- Exact UI control shape for the Decided-table status-change action (dropdown, per-status
  buttons, etc.) (D-04).
- Whether the new `[WEATHERED]`/`[CANCELLED]` prefixes need to be added to
  `calendar_display_extras.py`'s `_TERMINAL_PREFIXES` tuple to also pick up the existing
  status box-shadow ring treatment (Phase 8/9), or whether a title prefix alone satisfies
  "visually reflected" for this phase — investigate during research/planning.
- Site resolution for the Gemini CampaignRun entry (D-06) — whether "Gemini-South" already
  resolves via the existing MPC candidate pool / `resolve_site()` (Phase 21/22 machinery) or
  needs a new local `Observatory` row created first.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Classical-schedule status + calendar sync
- `solsys_code/telescope_runs.py` — `KNOWN_STATUSES`, `_resolve_status()` (status word/
  parenthetical parsing, already recognizes `cancelled`)
- `solsys_code/management/commands/load_telescope_runs.py` — `Status: {parsed.status}` is
  currently embedded in `description` only, no title/visual handling
- `solsys_code/calendar_utils.py` — `insert_or_create_calendar_event()` (shared no-churn
  create-or-update helper, reused by all three sync commands; this phase's D-01/D-05 both
  route through it)

### LCO title-prefix precedent (the pattern D-02/D-03 extend)
- `solsys_code/management/commands/sync_lco_observation_calendar.py` — the
  `WINDOW_EXPIRED`/`CANCELED`/`FAILURE_LIMIT_REACHED`/`NOT_ATTEMPTED` → `[EXPIRED]`/
  `[CANCELLED]`/`[FAILED]`/`[FAILED]` mapping and its title-prefix application
- `solsys_code/templatetags/calendar_display_extras.py` — `_TERMINAL_PREFIXES = ('[EXPIRED]',
  '[CANCELLED]', '[FAILED]')`, consumed by the status box-shadow ring logic (Phase 9);
  relevant to the Claude's Discretion item on whether new prefixes need adding here too
- `src/templates/tom_calendar/partials/calendar.html` — composition of the dashed fallback-
  label border (Phase 8) with the status box-shadow rings (Phase 9); any new visual treatment
  must not collide with either

### CampaignRun run_status + approval queue
- `solsys_code/models.py` — `CampaignRun.RunStatus` (`CANCELLED`, `WEATHER_TECH_FAILURE`
  choices already exist; `approval_status` is the separate, unrelated admin-review field)
- `solsys_code/campaign_views.py` — `CampaignRunDecisionView.post()`,
  `_project_calendar_event()` (the existing approve/resolve-time projection this phase's D-05
  must integrate with for the Decided-table status-change path)
- `solsys_code/campaign_tables.py` — `ApprovalQueueTable.render_actions()` /
  `render_site()` (Decided table's current `show_actions=False` → `''` early-return that D-04
  changes for the first time)
- `solsys_code/admin.py` — `CampaignRunAdmin.readonly_fields = ['approval_status']` (confirms
  `run_status` is currently editable ONLY here; D-04 adds the first non-admin entry point)

### Gemini scope boundary
- `tom_observations/facilities/gemini.py` (site-packages, not project-owned) —
  `GEMObservationForm.observation_payload()` / `GEMFacility` — confirms the one-record-one-
  window ToO submission model that D-06 explicitly does NOT use
- `solsys_code/management/commands/sync_gemini_observation_calendar.py` — the real sync path
  this phase does NOT invoke (kept as a reference for what "in scope" deliberately avoids)
- `solsys_code/campaign_utils.py` — `resolve_site()`, `build_site_candidates()` (Phase 21/22
  site-resolution machinery the Gemini CampaignRun's site field should go through)

### Real-world trigger data
- `Didymos_runs` (repo root) — the two real classical Magellan lines plus the informational
  Gemini FT-115 note added ahead of this phase's discussion

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `insert_or_create_calendar_event()` (`calendar_utils.py`) — shared no-churn create-or-update
  helper already used by all three sync commands; both D-01 (classical re-ingest) and D-05
  (CampaignRun status-change update) should route through it rather than hand-rolling saves.
- LCO's status→prefix dict pattern (`sync_lco_observation_calendar.py`) — directly reusable
  shape for both the classical `[CANCELLED]` mapping (D-02) and the CampaignRun
  `[CANCELLED]`/`[WEATHERED]` mapping (D-03).
- `CampaignRun.window_start`/`window_end` (Phase 19 schema) — already the correct shape for
  D-06's Gemini window entry; no new model field needed.
- `Observatory` / `resolve_site()` / `build_site_candidates()` (Phase 21/22) — existing
  site-resolution UI/logic the Gemini CampaignRun's site should reuse, not duplicate.

### Established Patterns
- Terminal-state title prefixes are the codebase's established idiom for "this
  scheduled/synced item didn't happen as planned" (LCO sync, Phase 4/7) — this phase extends
  that idiom to two more subsystems rather than inventing a new visual language.
- `ApprovalQueueTable`'s `show_actions`/`mode` gating (Phase 22, WR-01 hardening) is the
  established pattern for controlling which rows get interactive controls — D-04's new
  Decided-table action should follow the same gating discipline (read-only variants of this
  table, if any exist, must not gain the action).

### Integration Points
- Classical (D-01/D-02): `load_telescope_runs.handle()` → `insert_or_create_calendar_event()`
  → new title-prefix logic before the existing `title = f'{parsed.telescope} {parsed.instrument}'`
  line.
- Campaign (D-04/D-05): new Decided-table action → `CampaignRunDecisionView.post()` (new
  branch alongside `approve`/`reject`/`resolve_site`) → `run.run_status` write →
  `_project_calendar_event()`-adjacent update of the existing `CAMPAIGN:{pk}` event's
  title/description.
- Gemini (D-06): a `CampaignRun` row created under the "Didymos 2026" `TargetList` (pk=1),
  going through the same `approval_status`/`run_status` lifecycle as any other campaign run —
  no new code path, just new data.

</code_context>

<specifics>
## Specific Ideas

Real-world trigger: a storm heading for Chile is expected to wipe out two scheduled Magellan
runs — Baade IMACS 17-18 July and Clay Lightspeed 18-20 July (`BoN-0626` window) — both
already present as classical-schedule lines in `Didymos_runs`. A Gemini FT program
(GS-2026A-FT-115, PI Thomas-Osip, "Constraining the ongoing evolution of the Didymos binary
asteroid system ahead of Hera's arrival", GMOS-S, 6.50 awarded hours, window 13-16 July) is
also affected and is noted informationally in the same file pending this phase.

Concrete field values for the Gemini `CampaignRun` entry (D-06), for the planner to use as a
seed/test fixture: `campaign` = "Didymos 2026" (`TargetList` pk=1), `telescope_instrument` =
"Gemini-South GMOS-S" (exact string TBD by planner), `window_start` = 2026-07-13,
`window_end` = 2026-07-16, `contact_person` = "Thomas-Osip", `run_status` = `REQUESTED`
(not yet observed at context-gathering time).

</specifics>

<deferred>
## Deferred Ideas

- **Real Gemini `ObservationRecord`/ODB API sync for FT-115 (or any Gemini program)** —
  explicitly out of scope for this phase per D-06. `sync_gemini_observation_calendar.py`
  already exists for genuinely-submitted Gemini programs; extending it to represent
  proposal-level "hours awarded, not yet scheduled to specific visits" allocations (rather
  than single-window ToO submissions) would be real new scope for a future phase, not a
  detail of this one.

[No pending todos were folded — the two matches from `cross_reference_todos` (extraction/
renaming refactors) were purely keyword-coincidental and unrelated to this phase's domain.]

</deferred>

---

*Phase: 23-weather-storm-cancellation-handling-give-staff-a-way-to-mark*
*Context gathered: 2026-07-16*
