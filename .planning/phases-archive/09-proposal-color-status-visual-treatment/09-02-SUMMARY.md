---
phase: 09
plan: "02"
subsystem: calendar-template
tags: [calendar, template, proposal-color, status-ring, legend, click-to-filter, DISPLAY-04, DISPLAY-05, DISPLAY-06, DISPLAY-07]
dependency_graph:
  requires:
    - Plan 01 (solsys_code.templatetags.calendar_display_extras: proposal_color, status_border_css, visible_proposals)
  provides:
    - Rendered proposal-keyed color on all-day pill background and timed bullet (DISPLAY-04)
    - [QUEUED] fix: queued all-day events retain proposal background-color (DISPLAY-05)
    - Status box-shadow rings (queued 2px, terminal 3px) composed with Phase 8 dashed border (DISPLAY-06)
    - Footer proposal legend with click-to-filter IIFE surviving htmx month swaps (DISPLAY-07)
  affects:
    - All calendar views (calendar.html is the partial rendered by render_calendar())
    - Integration test suite (test_calendar_template.py)
tech_stack:
  added: []
  patterns:
    - Django simple_tag with `as` variable assignment (proposal_color/status_border_css called per event)
    - CSS box-shadow composition with existing border style (D-08/D-09 non-collision)
    - Vanilla JS IIFE inside htmx-swapped fragment for click-to-filter (Pitfall 5 htmx survival)
    - data-proposal keyed on resolved color hex (not raw proposal string) for collision grouping (D-04)
key_files:
  created: []
  modified:
    - src/templates/tom_calendar/partials/calendar.html
    - solsys_code/tests/test_calendar_template.py
    - solsys_code/templatetags/calendar_display_extras.py
decisions:
  - "data-proposal keyed on color hex (bg_color) not raw proposal string — enables collision grouping via strict JS equality (resolved_decisions 1)"
  - "data-proposal on OUTER .cal-event-all-day-row and .cal-event-timed divs — JS targets .cal-event descendants (resolved_decisions 2)"
  - "status_border in style before dashed-border literal: 'background-color: X; status_border dashed...' — both are independent CSS properties, neither overwrites the other"
  - "visible_proposals duck-types day objects: isinstance(dict) branch for real view, attribute branch for SimpleNamespace unit test stubs"
metrics:
  duration_minutes: 18
  completed_date: "2026-06-26"
  tasks_completed: 3
  files_created: 0
  files_modified: 3
status: complete
---

# Phase 09 Plan 02: Calendar Template Visual Treatment Summary

Rewrote calendar.html event branches with proposal-keyed color and status box-shadow, fixed the [QUEUED] grey override, added the footer legend with htmx-surviving click-to-filter — and fixed a latent dict/attribute access bug in visible_proposals.

## What Was Built

Modified `src/templates/tom_calendar/partials/calendar.html` (Task 1+2) and extended `solsys_code/tests/test_calendar_template.py` (Task 3):

**Template changes (Tasks 1 & 2):**

- Updated `{% load %}` line to include `calendar_display_extras`.
- All-day branch: per-event `{% proposal_color event.proposal as bg_color %}` and `{% status_border_css event.title as status_border %}` computed before the outer row div. The three-way `{% if [QUEUED] %}{% elif is_verified %}{% else %}` chain replaced with a two-way `{% if is_verified == False %}` branch that concatenates `background-color: bg_color; status_border border: 2px dashed ...` (composed, not mutually exclusive). `data-proposal="{{ bg_color }}"` on the outer `.cal-event-all-day-row` div (D-04 collision grouping, resolved_decisions 2).
- Timed branch: same per-event compute; `data-proposal="{{ bg_color }}"` on both `.cal-event-timed` div variants; `{{ status_border }}` in the style; proposal-color `▌` bullet `<span class="cal-event-bullet" style="color: {{ bg_color }};">` added after the target_list include.
- Legend CSS (5 rules): `.cal-legend-swatch`, `.cal-legend-swatch.is-active`, `.cal-event` transition, `#calendar-partial.cal-filtering .cal-event`, `#calendar-partial.cal-filtering .cal-event.cal-filter-match`.
- Footer legend: `{% visible_proposals weeks as proposal_legend %}` loop with `.cal-legend-swatch` spans; `data-proposal="{{ entry.color }}"` on each swatch (color-keyed to match events).
- Inline `<script>` IIFE inside `#calendar-partial` (before closing `</div>`): re-executes on each htmx swap, resets `activeProposal` to null, handles swatch click → toggle `.cal-filtering`/`.cal-filter-match`/`.is-active`.

**Test changes (Task 3):**

- Added marker constants: `QUEUED_BOX_SHADOW`, `TERMINAL_BOX_SHADOW`, `OLD_QUEUED_GREY`, `NEUTRAL_HEX`.
- Added 5 Phase 9 fixtures: `queued_event`, `terminal_event`, `timed_with_proposal`, `no_proposal_event`, `queued_fallback_timed` (with `is_verified=False` sidecar).
- Updated `num_fallback_day_cell_occurrences` from 3 to 4 (queued_fallback_timed adds 1 day-cell occurrence of dashed marker).
- Added 10 test methods covering DISPLAY-04/05/06/07 including Pitfall-3 composition regression test.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Rewrite event render branches + visible_proposals bug fix | cbf0077 | src/templates/tom_calendar/partials/calendar.html, solsys_code/templatetags/calendar_display_extras.py |
| 2 | Footer legend, CSS rules, inline filter IIFE | cbf0077 | src/templates/tom_calendar/partials/calendar.html (same commit — tasks 1+2 implemented together) |
| 3 | Extend integration tests for DISPLAY-04/05/06/07 | d66f227 | solsys_code/tests/test_calendar_template.py, solsys_code/templatetags/calendar_display_extras.py |

## Verification Results

- `./manage.py test solsys_code.tests.test_calendar_template solsys_code.tests.test_calendar_display_extras -v 2`: 36/36 GREEN
- `./manage.py test solsys_code`: 171/171 GREEN (no regressions)
- `ruff check solsys_code/templatetags/calendar_display_extras.py solsys_code/tests/test_calendar_template.py`: clean
- `ruff format --check ...`: clean
- `grep -c 'background-color: rgba(0, 0, 0, 0.45)' src/templates/tom_calendar/partials/calendar.html`: 0 (DISPLAY-05 override removed)
- `grep -n 'calendar_display_extras' src/templates/tom_calendar/partials/calendar.html`: line 123 (load tag present)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `visible_proposals` used attribute access on dict-based day objects**
- **Found during:** Task 1 — pre-existing Phase 8 tests failed immediately on template render
- **Issue:** The `visible_proposals` tag used `day.all_day_events` and `day.events` (attribute access). The real `tom_calendar` view passes day objects as dicts (`{"all_day_events": [...], "events": [...], ...}`). Attribute access on a dict raises `AttributeError`. The Plan 01 unit tests used `SimpleNamespace` stubs (attribute access works), so the bug was masked in unit testing but surfaced in integration testing.
- **Fix (attempt 1):** Changed to `day['all_day_events']` / `day['events']` (dict subscript). This fixed the integration tests but broke 5 Plan 01 unit tests that use `SimpleNamespace` (subscript on SimpleNamespace raises `TypeError`).
- **Fix (attempt 2, final):** Added `isinstance(day, dict)` branch: dict access for real view days, attribute access for SimpleNamespace unit test stubs. Both test suites pass.
- **Files modified:** `solsys_code/templatetags/calendar_display_extras.py`
- **Commits:** cbf0077 (attempt 1), d66f227 (final fix)

## Known Stubs

None — all four DISPLAY requirements render concrete values from locked constants and real DB-backed event data. No placeholder text or hardcoded empty responses.

## Threat Surface Scan

No new network endpoints, auth paths, or schema changes introduced. The template changes interpolate only `bg_color` (palette hex from `proposal_color`) and `status_border` (box-shadow literal from `status_border_css`) into `style` attributes — never the raw `proposal` or `title` field strings (T-09-01 control). `entry.label` in the legend is rendered through Django's default autoescaping (no `|safe`, T-09-02 control). The inline IIFE reads only `dataset.proposal` (computed hex) and toggles CSS classes; no `innerHTML`, `eval`, or network calls (T-09-03 accepted as low risk).

## Self-Check: PASSED

- src/templates/tom_calendar/partials/calendar.html: FOUND
- solsys_code/tests/test_calendar_template.py: FOUND
- solsys_code/templatetags/calendar_display_extras.py: FOUND (modified)
- Commit cbf0077: FOUND (template rewrite)
- Commit d66f227: FOUND (integration tests + final bug fix)
