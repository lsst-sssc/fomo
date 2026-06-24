---
phase: quick-260618-mck
plan: 01
status: complete
subsystem: calendar-ui
tags: [django-templates, tom_calendar, ui, css]
provides:
  - Legible [QUEUED] all-day calendar event box on both white and #f8f9fa day-cell backgrounds
affects: [tom_calendar, calendar-ui]
tech-stack:
  added: []
  patterns: [inline-style-override-for-vendor-template, literal-rgba-color-no-css-var-tokens]
key-files:
  created: []
  modified: [src/templates/tom_calendar/partials/calendar.html]
key-decisions: []
duration: ~5min
completed: 2026-06-18
---

# Quick Task 260618-mck: Fix insufficient contrast in [QUEUED] calendar style Summary (Minimal)

**Strengthened the `[QUEUED]` all-day event inline style from a near-invisible `rgba(0,0,0,0.06)` fill / dashed `0.35`-alpha border to a `rgba(0,0,0,0.45)` mid-gray fill / solid `0.55`-alpha border, making the forced-white title text legible on both white in-month and `#f8f9fa` overflow day cells while staying visibly muted versus solid `event.color` confirmed/placed blocks.**

## Performance
- **Duration:** ~5 min
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- `[QUEUED]` box fill changed from `rgba(0, 0, 0, 0.06)` to `rgba(0, 0, 0, 0.45)` — composites to a mid-gray (~#8c8c8c on white, ~#898a8b on #f8f9fa) that gives the `.cal-event-all-day` forced `color: #fff !important` title strong contrast on both day-cell backgrounds.
- Border changed from `1px dashed rgba(0, 0, 0, 0.35)` to `1px solid rgba(0, 0, 0, 0.55)` so the box outline reads against the light-gray overflow cell too.
- No `var(--...)` Bootstrap color tokens introduced — confirmed only pre-existing `.cal-day` `<style>` block rules (`--light`, `--secondary`, `--primary`, `--white`) remain, none inside the `[QUEUED]` inline style.
- Change confined to exactly one line; the else branch (solid `event.color` for confirmed/placed events), the timed-event block, and all other files are untouched.

## Task Commits
1. **Task 1: Strengthen the [QUEUED] de-emphasis style for contrast on white AND #f8f9fa** - `5ee2dd0`

## Files Created/Modified
- `src/templates/tom_calendar/partials/calendar.html` - `[QUEUED]` branch `cal-event-all-day` inline style fill/border strengthened for contrast.

## Deviations from Plan

None - plan executed exactly as written.

## Next Phase Readiness
Stage 1 contrast fix complete. Manual render verification (operator note from plan): open the calendar month view and confirm the `[QUEUED] FTS 2M0-...` box is clearly legible (white title on mid-gray box) on both a white in-month cell and a `#f8f9fa` other-month overflow cell, and that it still looks more muted than a solid `event.color` block. No further automated action needed for this quick task.

## Self-Check: PASSED
- FOUND: src/templates/tom_calendar/partials/calendar.html
- FOUND: 5ee2dd0
