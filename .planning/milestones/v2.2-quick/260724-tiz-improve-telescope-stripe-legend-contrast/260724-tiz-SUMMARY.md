---
phase: quick-260724-tiz
plan: 01
subsystem: calendar-display
tags: [contrast-fix, css, telescope-legend]
dependency-graph:
  requires: [quick-260724-osc]
  provides: [TELESCOPE_PALETTE, cal-event-classical, cal-legend-chip]
  affects: [calendar_display_extras, calendar.html]
tech-stack:
  added: []
  patterns:
    - "CSS pseudo-element (::before) for a decorative stripe, keeping it off the same
      style-attribute property namespace as an unrelated box-shadow ring"
key-files:
  created: []
  modified:
    - solsys_code/templatetags/calendar_display_extras.py
    - solsys_code/tests/test_calendar_display_extras.py
    - src/templates/tom_calendar/partials/calendar.html
    - solsys_code/tests/test_calendar_template.py
decisions:
  - "Kept PROPOSAL_PALETTE and proposal_color completely untouched; added a
    second, brighter TELESCOPE_PALETTE constant rather than modifying the shared
    dark palette, per the user's explicit rejection of lightening the gray fill."
  - "Stripe implemented as a ::before pseudo-element (cal-event-classical class +
    --tel-color custom property) instead of an inline border-left, so it cannot
    collide with status_border_css's box-shadow ring on the same style attribute."
metrics:
  duration: ~12 min
  completed: 2026-07-24
status: complete
---

# Phase quick-260724-tiz Plan 01: Improve telescope stripe/legend contrast Summary

Follow-up contrast fix to quick-260724-osc's per-telescope stripe: switched telescope_color()
to a new pre-validated brighter TELESCOPE_PALETTE, re-implemented the stripe as a CSS
pseudo-element (6px, with a light inset seam) to avoid an inline-style collision with the
status-ring box-shadow, and enlarged both the proposal and telescope legend swatches from a
thin `▌` glyph into filled 12px rounded chips.

## What Was Built

- **`TELESCOPE_PALETTE`** (`solsys_code/templatetags/calendar_display_extras.py`): a new
  8-hex brighter dark-surface categorical palette, added immediately after `PROPOSAL_PALETTE`
  (left untouched). `telescope_color()` now hashes into `TELESCOPE_PALETTE` instead of
  `PROPOSAL_PALETTE`; docstring corrected to match. Hashing/normalization logic
  (`.strip().upper()`, `hashlib.sha256`, `NEUTRAL_SLOT_COLOR` fallback) unchanged.
- **CSS pseudo-element stripe** (`src/templates/tom_calendar/partials/calendar.html`): added
  `.cal-event-classical` (`position: relative; padding-left: 9px;`) and its `::before` rule
  (6px `width`, `background-color: var(--tel-color)`, `box-shadow: inset -1px 0 0
  rgba(255, 255, 255, 0.55)` as the light seam, `border-radius: 3px 0 0 3px`). Both all-day
  event branches (verified/unverified) now conditionally add the `cal-event-classical` class
  and a `--tel-color: {{ tel_color }};` custom property to the div when `bg_color ==
  neutral_color`, replacing the old inline `border-left: 4px solid {{ tel_color }};` fragment.
  No inline `border-left` remains anywhere in the template.
- **`.cal-legend-chip`**: a shared 12px filled rounded-swatch class, used in place of the `▌`
  glyph in both the proposal legend loop (inside the unchanged `.cal-legend-swatch` /
  `data-proposal` click-to-filter wrapper) and the telescope legend loop (inside the unchanged
  `.cal-legend-telescope` wrapper).
- **Tests**: updated `test_calendar_display_extras.py`'s palette-membership assertion to
  `TELESCOPE_PALETTE`; updated `test_calendar_template.py`'s stripe assertions to check for the
  `cal-event-classical` class and `--tel-color` custom property (and their absence on
  proposal-having events) instead of the old `border-left` string; added a new test asserting
  both legends render `.cal-legend-chip` spans with the correct `background-color`.

## Deviations from Plan

None - plan executed exactly as written.

## Verification

- `python manage.py test solsys_code.tests.test_calendar_display_extras
  solsys_code.tests.test_calendar_template` → 63 tests, OK.
- `ruff check` / `ruff format --check` on all 4 touched files → clean (repo-wide ruff findings
  in unrelated notebooks/settings.py are pre-existing and out of scope for this plan).

## Self-Check: PASSED

All 4 modified files present on disk; both task commits (`d8b2cc9`, `9f7bfae`) found in git log.
