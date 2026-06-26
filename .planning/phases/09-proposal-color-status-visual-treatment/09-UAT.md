---
status: testing
phase: 09-proposal-color-status-visual-treatment
source: [09-VERIFICATION.md]
started: 2026-06-25T00:00:00Z
updated: 2026-06-25T00:00:00Z
---

## Current Test

number: 1
name: Click-to-Filter — highlight matching events on legend swatch click
expected: |
  Load the calendar with >= 2 distinct proposals visible. Click a legend swatch.
  That proposal's events stay full opacity; all others dim (~0.18 opacity, grayscale).
  Clicked swatch gets bold + underline (is-active). Click same swatch again → all events
  restored to full opacity, is-active removed.
awaiting: user response

## Tests

### 1. Click-to-Filter: highlight matching events on legend swatch click
expected: |
  Load the calendar page with at least two distinct proposals visible in the current month.
  Click a legend swatch (.cal-legend-swatch) for one proposal.
  That proposal's events remain at full opacity; all other .cal-event elements dim to ~0.18
  opacity with grayscale. The clicked swatch gains bold text and underline (.is-active).
  Clicking the same swatch again removes .cal-filtering from #calendar-partial, removes
  .cal-filter-match from all events, and removes .is-active from all swatches.
result: [pending]

### 2. Click-to-Filter: htmx month-swap survival
expected: |
  With the calendar showing, navigate to Prev or Next month via the buttons.
  After the htmx outerHTML swap completes, click a legend swatch.
  The inline <script> re-executes (it is inside the swapped #calendar-partial fragment).
  activeProposal resets to null on each swap. Click-to-filter works identically on the
  re-rendered month.
result: [pending]

### 3. Colorblind accessibility of PROPOSAL_PALETTE
expected: |
  With >= 5 proposals visible on the calendar, run through a CVD simulator (e.g. Coblis)
  for deuteranopia and protanopia. All 8 palette entries (#005f9e, #a34000, #5b2080,
  #006b4e, #9e1c1c, #006b6b, #6b2060, #7a4500) remain mutually distinguishable under
  both deficiency simulations.
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
