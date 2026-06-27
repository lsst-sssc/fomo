# Phase 9: Proposal Color & Status Visual Treatment - Context

**Gathered:** 2026-06-25
**Status:** Ready for planning

<domain>
## Phase Boundary

A calendar viewer can identify which proposal an event belongs to by color alone (consistent across
telescopes and re-renders) and can distinguish queued/placed/terminal-failure status visually, not just
by reading title-prefix text. This phase delivers: a deterministic, colorblind-vetted, curated-palette
hash of the normalized `proposal` string replacing today's meaningless `pk`-based `CalendarEvent.color`;
a fix to the existing `[QUEUED]` override (`calendar.html:158-161`) so it dims/borders the proposal color
instead of discarding it with flat grey; a status visual treatment (mechanism chosen via `/gsd:sketch`
during phase planning, layered orthogonally on top of — not instead of — the proposal color); and an
on-page legend mapping proposal codes to colors, now expanded during this discussion to also support
click-to-filter highlighting. Covers DISPLAY-04/05/06/07 per REQUIREMENTS.md's traceability table.

</domain>

<decisions>
## Implementation Decisions

### Legend design (DISPLAY-07)
- **D-01:** The legend lives in the **existing footer row** (`calendar.html` ~line 198, the
  `d-flex justify-content-between align-items-center mt-1` row that already renders the target-list
  color key via `target_list_block.html` and the UTC-offset selector) — not a new row, not a
  collapsible panel. Mirror `target_list_block.html`'s exact swatch pattern: a colored `▌` bullet
  (`<span class="cal-event-bullet" style="color: {{ swatch_color }};">▌</span>`) followed by the
  proposal code as plain text.
- **D-02:** The legend lists only proposals with at least one event **visible in the currently-rendered
  month** — computed from the day-cell events already in the render context, not a separate
  all-history query.
- **D-03 (scope expansion, confirmed by user):** Each legend entry is **clickable** and toggles a
  **client-side highlight/dim filter**: clicking a proposal's legend entry full-opacity-highlights that
  proposal's events on the grid and dims all others; clicking again clears the highlight. No page
  reload, no htmx round-trip, no URL/query-param change — pure CSS/JS state toggle. This expands
  DISPLAY-07 beyond its original "identification only" wording; **REQUIREMENTS.md and ROADMAP.md have
  already been updated** (2026-06-25) to reflect this — DISPLAY-07's text now includes the toggle
  behavior, and Phase 9's ROADMAP success criteria gained a 5th item for it. No further requirements
  amendment is needed before planning.
- **D-04 (collision handling):** If two or more proposals hash to the same palette color, the legend
  groups them under **one swatch** listing all colliding proposal codes (e.g. `▌ LTP2025A-004, LTP2025B-012`)
  rather than rendering separate rows that would visually imply more distinct colors than the palette
  actually has.

### Empty-proposal / classical-event treatment
- **D-05:** Events with no proposal (`load_telescope_runs` classical-schedule events, `proposal=''`)
  get a dedicated neutral palette slot. **Claude's discretion** on the exact neutral color (e.g. a
  mid-grey), as long as it's a deliberate slot in the new curated palette, not a hash-of-empty-string
  fallback and not necessarily identical to today's ad hoc styling.
- **D-06:** The legend includes an entry for this neutral slot, labeled something like
  **"Classical schedule"** (exact copy is Claude's discretion) — every color a viewer sees in the
  current month should have a matching legend entry, including the neutral one.

### Color collision / overflow behavior
- **D-07:** Two or more simultaneously-active proposals landing on the same palette color (a hash
  collision against the small ~8-9-color curated palette) is **accepted as-is** — no larger palette,
  no collision-detection fallback pattern, no warning. Title text and the legend (per D-04) already
  disambiguate. Matches research's framing: color is the primary-but-not-sole identity signal.

### Status visual treatment mechanism (explicitly NOT decided here)
- **D-08:** The exact status-treatment mechanism (border-color/thickness/double-border — research
  favors border-style, see `FEATURES.md`) is **deferred to a `/gsd:sketch` session during Phase 9
  planning**, per the locked instruction in `PROJECT.md`'s Current Milestone section and
  `ROADMAP.md`'s Phase 9 success criterion #3. This was deliberately not opened as a discuss-phase gray
  area.
- **D-09 (carried forward, locked, do not re-open):** Per Phase 8's `08-CONTEXT.md` D-03 — **dash-style
  is reserved for Phase 8's verification signal** (fallback-vs-verified telescope label). Phase 9's
  status treatment MUST use a different border property (color, thickness, or double-border), never
  dash-style, so the two phases' border-based signals don't collide on the same CSS property. The
  `/gsd:sketch` session's option set is therefore narrowed to exclude dashed borders.

### Claude's Discretion
- Exact neutral-slot color for no-proposal events (D-05).
- Exact legend label copy for the neutral slot (D-06), e.g. "Classical schedule".
- Implementation mechanism for the click-to-filter toggle (D-03) — plain CSS class toggle via a small
  inline `<script>`/Alpine-style attribute, or whatever lightweight client-side pattern is idiomatic for
  this template; no new JS framework/dependency.
- Palette source (literal hex/rgba vs. any other representation) — research (`STACK.md`) already
  strongly recommends a project-local literal palette, independent of `tom_calendar.utils.BOOTSTRAP_COLORS`,
  to survive the pinned `tomtoolkit==3.0.0a9` → future Bootstrap5 rename; this is a technical
  implementation detail, not reopened as a user decision.

### Folded Todos
- `2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md` ("Status-aware calendar
  event coloring, telescope/proposal-keyed, alpha by confidence") — folded. This todo is the direct
  origin of DISPLAY-04/05/06/07 and is already `resolves_phase: 9` in its own frontmatter. Its specific
  note that the user previously suggested **"striping" as an alternative/addition to opacity for
  status** is carried forward as an explicit option to weigh in the `/gsd:sketch` session (D-08),
  alongside research's border-style recommendation — not decided here.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/ROADMAP.md` (Phase 9 section) — goal, success criteria (now 5, after D-03's scope
  expansion), depends-on (Phase 8)
- `.planning/REQUIREMENTS.md` (DISPLAY-04/05/06/07, "Calendar Color & Status" section) — locked
  requirement text; DISPLAY-07 amended 2026-06-25 to include the click-to-filter toggle (D-03); DISPLAY-08
  (WCAG contrast-aware text color) and DISPLAY-09 (N+1 batching tag) are v2-deferred, out of this phase's
  scope
- `.planning/PROJECT.md` — Current Milestone section, Key Decisions table
- `.planning/phases/08-telescope-label-verification-sidecar/08-CONTEXT.md` — D-03 (dash-style reserved
  for Phase 8) is a hard constraint on this phase's status-treatment sketch session (D-09 above)
- `.planning/todos/pending/2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md` —
  folded todo (see Folded Todos); carries the user's original "striping" suggestion

### Research (all written 2026-06-24 for this milestone, HIGH confidence)
- `.planning/research/SUMMARY.md` — executive summary, phase-ordering rationale
- `.planning/research/STACK.md` — template-tag mechanism (`calendar_display_extras.py`,
  `proposal_color` simple_tag), literal-palette-not-BOOTSTRAP_COLORS recommendation, sidecar-vs-template
  integration points, exact line references for the `[QUEUED]` override bug
- `.planning/research/ARCHITECTURE.md` — integration points, anti-patterns to avoid
- `.planning/research/FEATURES.md` — "Status Visual Language: Options for the `/gsd:sketch` Session"
  table (lines 42-60) — border-style (recommended), opacity, stripe/hatching, icon/glyph options with
  tradeoffs; Feature Dependencies section on DISPLAY-01/02 shared visual vocabulary
- `.planning/research/PITFALLS.md` — N+1, no-churn conflation, staleness contract pitfalls

### Existing code (read directly this session)
- `src/templates/tom_calendar/partials/calendar.html:158-205` — the `[QUEUED]` override bug (replaces
  `event.color` with flat grey instead of dimming it); the existing footer-row legend precedent
  (`target_list_block.html` include + UTC-offset `<select>`) that D-01 reuses
- `src/templates/tom_calendar/partials/target_list_block.html` — the exact swatch markup
  (`<span class="cal-event-bullet" style="color: {{ tl_color }};">▌</span>`) D-01 mirrors for the
  proposal legend
- Installed `tom_calendar` package (`models.py:52-53`, `utils.py`) — confirms `CalendarEvent.color` is
  a read-only `BOOTSTRAP_COLORS[self.pk % 9]` property, not overridable per-instance; must be bypassed
  entirely via a new template tag, not patched
- `solsys_code/management/commands/sync_lco_observation_calendar.py` — `_FAILURE_PREFIX_BY_STATUS`,
  `_failure_prefix()`, `[QUEUED]`/`[UNVERIFIED]`/terminal-prefix title vocabulary the status treatment
  must remain visually-additive to (text stays as accessible fallback per D-08/FEATURES.md)

### Process convention
- `CLAUDE.md` — plain-English-over-jargon convention; this phase touches no module from the
  demo-notebook-companion list (`telescope_runs.py`, `load_telescope_runs.py`,
  `sync_lco_observation_calendar.py`), so no notebook update is required by that convention — confirm
  this holds at planning time since it only touches `calendar.html`/template tags

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `target_list_block.html`'s swatch pattern (colored `▌` bullet + label) — direct precedent for the
  new proposal legend (D-01), no new visual component needed.
- The existing footer row (`calendar.html` ~line 198) already holds one legend-like element
  (target-list color key) alongside the UTC-offset selector — the proposal legend slots in next to it.
- `_FAILURE_PREFIX_BY_STATUS` / title-prefix vocabulary (`[QUEUED]`/`[EXPIRED]`/`[CANCELLED]`/`[FAILED]`)
  is the existing, fully-accessible signal the new status border-treatment layers on top of, per
  research's explicit recommendation to keep text as the fallback channel.

### Established Patterns
- Django template tag library mirroring `tom_calendar`'s own `calendar_tags.py` (`target_list_color`
  simple_tag) is the established, idiomatic mechanism for computing values `{{ event.color }}` alone
  can't express — `STACK.md` recommends a new `calendar_display_extras.py` module following this exact
  shape.
- Project-level full-copy template override (`src/templates/tom_calendar/partials/calendar.html`) is
  the only stable customization seam — confirmed (again, for this phase) that `tom_calendar.views.render_calendar()`
  has no Python-level hook.

### Integration Points
- Read side only this phase (no new model, no migration) — color and status are both computable
  in-template from existing `CalendarEvent` fields (`proposal`, `title` prefix vocabulary), unlike
  Phase 8's sidecar model which needed a DB write.
- The new legend's click-to-filter toggle (D-03) is pure front-end (CSS class toggle, no server
  round-trip) — does not touch `sync_lco_observation_calendar.py`, `load_telescope_runs.py`, or any
  management command.

</code_context>

<specifics>
## Specific Ideas

- Legend swatch markup should look and feel identical to the existing target-list color key
  (`▌` bullet + label), just keyed on proposal instead of target list, living in the same footer row.
- Click a legend entry → that proposal's events go full-opacity/highlighted, everything else on the
  grid dims; click again → clears back to normal. No navigation, no reload.
- A collision between two proposals' colors shows as one swatch with both proposal codes listed next
  to it (e.g. `▌ LTP2025A-004, LTP2025B-012`), not two visually-identical separate rows.
- Classical-schedule (no-proposal) events get their own neutral slot in the palette, with a
  "Classical schedule"-style legend entry — exact wording is Claude's discretion.

</specifics>

<deferred>
## Deferred Ideas

None beyond what's captured above — the click-to-filter idea that might have looked like scope creep
was instead explicitly absorbed into Phase 9's scope (D-03) at the user's request, with REQUIREMENTS.md
and ROADMAP.md amended accordingly rather than deferred.

### Reviewed Todos (not folded)
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — reviewed again (also
  reviewed-not-folded in Phase 8); unrelated refactor work (SITE_TELESCOPE_MAP/instrument extraction is
  Phase 7 scope), no connection to color/status visual treatment.

</deferred>

---

*Phase: 9-Proposal Color & Status Visual Treatment*
*Context gathered: 2026-06-25*
