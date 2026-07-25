---
phase: quick-260724-osc
plan: 01
subsystem: ui
tags: [django-templates, calendar, template-tags, color-coding]

# Dependency graph
requires:
  - phase: quick-260723-02e
    provides: CalendarEvent.telescope field already populated by load_telescope_runs
provides:
  - telescope_color simple_tag (deterministic per-telescope hex color, reuses PROPOSAL_PALETTE)
  - visible_classical_telescopes simple_tag (classical-schedule-only telescope legend data)
  - neutral_slot_color assignment tag (exposes NEUTRAL_SLOT_COLOR to templates)
  - per-telescope left-edge stripe on classical-schedule all-day calendar events
  - display-only telescope legend section in calendar.html
affects: [calendar, telescope-runs-calendar]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "telescope_color mirrors proposal_color's exact normalization (.strip().upper()) + sha256 hash-into-PROPOSAL_PALETTE approach, sharing one palette instead of introducing a second one"
    - "border-left composed into an existing inline style attribute via a guarded {% if %} fragment, so unrelated style properties (background-color, status_border, dashed border) are never clobbered"
    - "classical-vs-proposal detection reuses the already-computed bg_color compared against a template-exposed neutral_slot_color constant, rather than re-deriving the distinction a second way"

key-files:
  created: []
  modified:
    - solsys_code/templatetags/calendar_display_extras.py
    - solsys_code/tests/test_calendar_display_extras.py
    - src/templates/tom_calendar/partials/calendar.html
    - solsys_code/tests/test_calendar_template.py

key-decisions:
  - "telescope_color falls back to NEUTRAL_SLOT_COLOR for blank/None telescopes (same as proposal_color's own defensive fallback) rather than a distinct sentinel — so a classical event with no telescope set still renders a stripe, just one that visually blends into the neutral fill (same color), which is harmless and consistent with the existing pattern."
  - "The telescope legend uses a new non-interactive CSS class (cal-legend-telescope) with no data-proposal attribute, explicitly so it cannot be picked up by the existing click-to-filter JS which keys off .cal-legend-swatch and data-proposal."

requirements-completed: [QUICK-260724-osc]

coverage:
  - id: D1
    description: "Classical-schedule (empty-proposal) all-day events render a per-telescope colored left-edge stripe over an unchanged neutral-gray fill"
    requirement: QUICK-260724-osc
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_display_extras.py::TelescopeColorTest"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py::CalendarTemplateTest::test_osc_classical_event_renders_telescope_stripe"
        status: pass
    human_judgment: false
  - id: D2
    description: "Two different telescopes render two different stripe colors; the same telescope always renders the same color"
    requirement: QUICK-260724-osc
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_display_extras.py::TelescopeColorTest::test_same_input_same_output"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_calendar_display_extras.py::VisibleClassicalTelescopesTest::test_groups_by_color_with_collision_handling"
        status: pass
    human_judgment: false
  - id: D3
    description: "Proposal-having events render NO telescope stripe"
    requirement: QUICK-260724-osc
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_display_extras.py::VisibleClassicalTelescopesTest::test_only_classical_events_contribute"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py::CalendarTemplateTest::test_osc_proposal_having_event_has_no_telescope_stripe"
        status: pass
    human_judgment: false
  - id: D4
    description: "A display-only legend section decodes the visible telescope stripe colors, without hooking into the click-to-filter JS"
    requirement: QUICK-260724-osc
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py::CalendarTemplateTest::test_osc_telescope_legend_renders_when_classical_event_visible"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py::CalendarTemplateTest::test_osc_telescope_legend_is_not_click_to_filter_wired"
        status: pass
    human_judgment: false

duration: 7min
completed: 2026-07-24
status: complete
---

# Quick Task 260724-osc: Per-Telescope Left-Edge Stripe Color Summary

**Classical-schedule calendar events now show a deterministic per-telescope colored left-edge stripe (reusing the existing PROPOSAL_PALETTE hash approach) plus a display-only decoding legend, composed alongside the existing proposal-fill/status-ring visuals without touching the click-to-filter JS.**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-07-24T17:53:39-07:00
- **Completed:** 2026-07-24T18:00:01-07:00
- **Tasks:** 2 completed
- **Files modified:** 4

## Accomplishments
- Added `telescope_color` and `visible_classical_telescopes` template tags (plus a small `neutral_slot_color` assignment tag) to `calendar_display_extras.py`, mirroring `proposal_color`/`visible_proposals`'s exact normalization, hashing, and dual dict/attribute-day support.
- Wired a `border-left: 4px solid <hex>;` stripe into the all-day event loop in `calendar.html`, scoped via `{% if bg_color == neutral_color %}` so only classical-schedule (empty-proposal) events get it, composed into both the dashed (unverified) and non-dashed style branches without clobbering the existing `background-color`/`color`/`status_border`/dashed-border declarations.
- Added a `cal-legend-telescope` legend section next to the existing proposal legend, rendering `{% visible_classical_telescopes weeks %}` output — deliberately given no `data-proposal` attribute and a distinct class from `.cal-legend-swatch` so it cannot hook into the click-to-filter `<script>` block.
- Added 12 new unit tests (`TelescopeColorTest`, `VisibleClassicalTelescopesTest`, `NeutralSlotColorTagTest`) and 4 new integration tests (`CalendarTemplateTest`) covering stripe presence/absence and legend rendering/isolation.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add telescope_color + visible_classical_telescopes tags with unit tests** - `3a7c90c` (feat)
2. **Task 2: Wire the stripe + legend into calendar.html with integration tests** - `0acd715` (feat)

**Plan metadata:** pending (docs: complete plan) — orchestrator commits `.planning/` docs separately per this task's constraints.

_Note: Task 1 is a `tracer` task per plan frontmatter; its automated `<verify>` (40/40 tests) passed cleanly and the plan is `autonomous: true`, so execution proceeded directly into Task 2's expansion without a separate human-verify pause._

## Files Created/Modified
- `solsys_code/templatetags/calendar_display_extras.py` - added `telescope_color`, `visible_classical_telescopes`, `neutral_slot_color` simple_tags
- `solsys_code/tests/test_calendar_display_extras.py` - added `TelescopeColorTest`, `VisibleClassicalTelescopesTest`, `NeutralSlotColorTagTest` classes
- `src/templates/tom_calendar/partials/calendar.html` - wired the border-left stripe into the all-day loop, added the `cal-legend-telescope` legend section and its CSS
- `solsys_code/tests/test_calendar_template.py` - added a classical-with-telescope fixture and 4 integration tests

## Decisions Made
- `telescope_color` reuses `PROPOSAL_PALETTE` rather than a second palette, per plan instruction — keeps the two color-coding systems visually consistent and avoids doubling the WCAG-AA-vetted color set.
- Chose `border-left` composition via a guarded inline-style fragment (rather than a wrapper `<div>` or CSS pseudo-element) to keep the change minimal and confined to the existing `.cal-event-all-day` style attribute, matching how `status_border` is already composed.

## Deviations from Plan

None - plan executed exactly as written. The plan explicitly permitted adding `neutral_slot_color` and its test during either Task 1 or Task 2 ("it is acceptable to add the tag here and its test alongside"); it was added during Task 1 for convenience since Task 1 wasn't yet committed at that point — not a deviation, an explicitly sanctioned choice point in the plan text.

## Issues Encountered
- Initial version of the "telescope legend is not click-to-filter wired" integration test matched the `.cal-legend-telescope` CSS *class definition* inside the `<style>` block first (not the rendered `<span>` markup), causing a false failure. Fixed by scoping the search to start after `</style>`. Resolved before commit — no code change needed, test-only fix.
- All existing classical-schedule fixtures (`all_day_fallback`, `all_day_verified`, `all_day_no_row`, `no_proposal_event`) default `telescope=''`, so they also now render a `border-left: 4px solid #5a6268;` stripe (same color as the neutral fill, effectively invisible but present in markup) — this is expected, correct behavior per `telescope_color`'s NEUTRAL_SLOT_COLOR fallback (mirrors `proposal_color`), not a defect.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Stage 1 (`solsys_code/telescope_runs.py`) and Stages 2-4 groundwork are unaffected — this quick task was a display-layer polish item unrelated to the `telescope_runs.py`/`load_telescope_runs`/`sync_lco_observation_calendar`/`sync_gemini_observation_calendar` demo-notebook-paired modules (none of them were touched), so no demo notebook update was required.

---
*Phase: quick-260724-osc*
*Completed: 2026-07-24*

## Self-Check: PASSED

- FOUND: solsys_code/templatetags/calendar_display_extras.py
- FOUND: solsys_code/tests/test_calendar_display_extras.py
- FOUND: src/templates/tom_calendar/partials/calendar.html
- FOUND: solsys_code/tests/test_calendar_template.py
- FOUND: .planning/quick/260724-osc-add-per-telescope-left-edge-stripe-color/260724-osc-SUMMARY.md
- FOUND commit: 3a7c90c (feat(260724-osc): add telescope_color + visible_classical_telescopes tags)
- FOUND commit: 0acd715 (feat(260724-osc): wire per-telescope stripe + legend into calendar.html)
