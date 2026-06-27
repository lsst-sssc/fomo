# Phase 8: Telescope Label Verification Sidecar - Context

**Gathered:** 2026-06-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Operators can tell, directly in the calendar UI, whether a synced event's telescope label was
live-verified against the LCO API or fallback-guessed, without reading title text. This phase delivers:
a `CalendarEventTelescopeLabel` sidecar model (`OneToOneField(primary_key=True)` on `CalendarEvent`)
written by `sync_lco_observation_calendar.py`; a visual cue on the calendar grid distinguishing
fallback-labeled events from verified ones; and a hover tooltip surfacing the verification detail.
`load_telescope_runs.py` (classical schedule) needs zero code change — no sidecar row, "verified" by
documented default. Covers DISPLAY-01/02/03 per REQUIREMENTS.md's traceability table (note:
REQUIREMENTS.md's own section header for these three calls the group "DISPLAY-02" — the Phase-level
DISPLAY-01/02/03 numbering in the Traceability table is the one to follow).

</domain>

<decisions>
## Implementation Decisions

### Sidecar model field shape
- **D-01:** `CalendarEventTelescopeLabel` stores `is_verified: BooleanField(default=True)` only — no
  separate `reason`/`detail` field. Matches research's (`STACK.md`) literal recommendation, and matches
  the existing `telescope_api_failed` contract in `sync_lco_observation_calendar.py`, which already
  deliberately funnels API-timeout and successfully-returned-but-unmapped-code into one shared signal
  (Phase 07 Key Decision: "both are the same user-visible degrade signal; splitting them into two
  differently-labeled failure classes adds complexity without operator value"). The tooltip (D-04) is
  therefore one fixed sentence, not per-failure-reason text.

### Visual cue style
- **D-02:** Fallback-labeled events get a **dashed border**; verified events keep a solid border (or no
  border change from today). This is research's lead recommendation (`FEATURES.md`, matches the Outlook
  tentative-booking precedent) and applies to both the all-day (`cal-event-all-day`) and timed
  (`cal-event-timed`) render branches in `src/templates/tom_calendar/partials/calendar.html`.
- **D-03:** **Dash-style is reserved for Phase 8's verification signal.** Phase 9's status visual
  treatment (queued/placed/terminal-failure, decided via `/gsd:sketch` during Phase 9 planning) MUST use
  a different border property — color, thickness, or double-border — not dash-style, so the two phases'
  border-based signals don't collide on the same CSS property. Record this as an explicit constraint for
  Phase 9's CONTEXT.md / planning.

### Tooltip wording
- **D-04:** Tooltip text uses generic plain-language "estimate" framing, not a terse one-liner and not
  per-failure-reason detail (consistent with D-01's boolean-only field and CLAUDE.md's plain-English
  convention). Working text: *"Telescope label is an estimate — could not be verified against the LCO
  API; showing a coarse fallback label (1m0/0m4/2m0/4m0)."* Exact final copy is Claude's discretion at
  plan/execute time as long as it stays plain-language, explains *why* (API verification failed) and
  *what* the label is (a coarse fallback), and never references internal field/exception names.

### Claude's Discretion
- Exact tooltip copy (within D-04's framing constraint).
- Whether the visual cue (dashed border) is implemented via a template-tag-driven CSS class or an inline
  conditional `style=` attribute — match whatever pattern the existing `[QUEUED]` override and any new
  `calendar_display_extras.py` module (if created this phase) already use.
- N+1 mitigation for the sidecar's reverse `OneToOneField` accessor read in the month-grid loop is
  **out of scope for this phase** — already locked as deferred to v2 per `REQUIREMENTS.md` DISPLAY-09
  ("current calendar-event volume doesn't justify the added scope; revisit if volume grows"). Accept the
  per-event reverse-accessor read as-is; do not build a batching template tag in Phase 8.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/ROADMAP.md` (Phase 8 section) — goal, success criteria, depends-on (Phase 7/07.1)
- `.planning/REQUIREMENTS.md` (DISPLAY-01/02/03, "Telescope Label Verification" section) — locked
  requirement text; DISPLAY-09 (N+1 batching tag deferred to v2) is the source of D-final decision above
- `.planning/PROJECT.md` — Key Decisions table (Phase 07/07.1 `telescope_api_failed`/coarse-label
  contract this phase's sidecar write depends on), Current Milestone section

### Research (all written 2026-06-24 for this milestone, HIGH confidence)
- `.planning/research/SUMMARY.md` — executive summary, phase-ordering rationale
- `.planning/research/STACK.md` — sidecar model code shape, exact `update_or_create` integration
  pattern, "no sidecar row for classical-schedule events" recommendation
- `.planning/research/ARCHITECTURE.md` — exact integration line numbers in
  `sync_lco_observation_calendar.py`, data-flow diagrams, anti-patterns to avoid (signal-based write,
  editing `tom_calendar` directly)
- `.planning/research/FEATURES.md` — status/verification visual-language options table, dependency notes
  on shared border vocabulary between DISPLAY-01 (Phase 9) and DISPLAY-02 (this phase)
- `.planning/research/PITFALLS.md` — Pitfalls 3-5 (no-churn conflation, staleness contract, N+1) are
  directly relevant to this phase's write/read integration

### Existing code (read directly this session)
- `solsys_code/management/commands/sync_lco_observation_calendar.py:385-470,590-626` — exact write site
  (`telescope_api_failed` computed at L470, popped at L604, `get_or_create` at L615 — sidecar write goes
  immediately after)
- `solsys_code/models.py` — currently all-comment; this phase adds its first real model
- `solsys_code/migrations/` — currently `__init__.py`-only; this phase adds its first real migration
- `src/templates/tom_calendar/partials/calendar.html` — the only customization seam (full-copy template
  override); existing `[QUEUED]` grey-override pattern at lines ~158-162 is the precedent for
  conditional-class/style branching
- Installed `tom_calendar` package (`models.py`, `views.py`) — `CalendarEvent` field shape confirmed,
  `render_calendar()` confirmed to have no `extra_context`/queryset hook (template-only customization)

### Process convention
- `CLAUDE.md` — demo-notebook-companion convention (any behavior change to
  `sync_lco_observation_calendar.py` requires `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`
  in `files_modified` with a regenerated, re-executed cell); plain-English-over-jargon convention (used
  for D-04's tooltip framing)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_build_event_fields()` in `sync_lco_observation_calendar.py` already computes `telescope_api_failed`
  (the exact DISPLAY-02 signal) per record — this phase persists it, doesn't recompute it.
- The `[QUEUED]` grey-override conditional-branch pattern in `calendar.html` (`{% if event.title|slice... %}`)
  is the established precedent for adding another conditional visual branch (dashed border) without
  introducing a new templating mechanism.

### Established Patterns
- `OneToOneField(primary_key=True)` sidecar model is the standard, already-research-confirmed Django
  pattern for adding a field to a third-party model (`CalendarEvent`) without touching its migrations.
- `update_or_create()` colocated immediately after the existing `get_or_create()`/diff/`save()` block —
  kept as a separate statement, never folded into the `fields` dict or `changed` comparison (Pitfall 3).
- No Python-level hook exists on `tom_calendar.views.render_calendar()` — any read-side change
  (sidecar read, visual cue) must live in the template-override layer, confirmed by direct read.

### Integration Points
- Write side: one new line in `sync_lco_observation_calendar.py`'s per-record loop, right after
  `event, created = CalendarEvent.objects.get_or_create(...)`.
- Read side: `{{ event.telescope_label_meta.is_verified|default:True }}` direct template attribute read
  (reverse O2O accessor) — no new template tag needed for the boolean itself, though the dashed-border
  CSS class/style and the tooltip text are conditional template logic in `calendar.html`.
- `load_telescope_runs.py`: confirmed no integration needed — no telescope-label resolution concept
  exists there at all.

</code_context>

<specifics>
## Specific Ideas

- Tooltip working text: *"Telescope label is an estimate — could not be verified against the LCO API;
  showing a coarse fallback label (1m0/0m4/2m0/4m0)."* (D-04; exact final copy is Claude's discretion).
- Dashed border for fallback, solid for verified, on both all-day and timed event render branches (D-02).

</specifics>

<deferred>
## Deferred Ideas

- N+1 batching template tag for the sidecar's reverse-accessor read — already deferred to v2 as
  DISPLAY-09 in `REQUIREMENTS.md`; not re-opened here (D-final decision above just confirms accept-as-is
  for this phase).
- Per-reason tooltip detail (distinguishing API-timeout vs unmapped-code) — considered in the "Sidecar
  field shape" discussion and explicitly not pursued (D-01); would require splitting the existing shared
  `telescope_api_failed` signal, which Phase 07's Key Decisions deliberately avoided.

### Reviewed Todos (not folded)
- `2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md` — reviewed; not folded.
  This is Phase 9's scope (`CalendarEvent.color`/status treatment), a different mechanism than this
  phase's label-verification sidecar.
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — reviewed; not folded.
  Unrelated refactor work, deliberately deferred; no connection to the sidecar model.

</deferred>

---

*Phase: 8-Telescope Label Verification Sidecar*
*Context gathered: 2026-06-24*
