---
status: complete
phase: 09-proposal-color-status-visual-treatment
source: [09-VERIFICATION.md]
started: 2026-06-25T00:00:00Z
updated: 2026-06-26T00:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Click-to-Filter — highlight matching events on legend swatch click
expected: |
  Load the calendar with >= 2 distinct proposals visible. Click a legend swatch.
  That proposal's events stay full opacity; all others dim (~0.18 opacity, grayscale).
  Clicked swatch gets bold + underline (is-active). Click same swatch again → all events
  restored to full opacity, is-active removed.
result: pass

### 2. Click-to-Filter: htmx month-swap survival
expected: |
  Navigate Prev/Next. After htmx swap, click a swatch. JS re-executes, activeProposal
  resets to null, filtering works on the re-rendered month.
result: pass

### 3. Colorblind accessibility of PROPOSAL_PALETTE
expected: |
  With >= 5 proposals visible, run through CVD simulator (deuteranopia + protanopia).
  All 8 palette entries remain mutually distinguishable.
result: skipped
reason: deferred — user opted not to run simulation at this time

## Summary

total: 3
passed: 2
issues: 0
pending: 0
skipped: 1
blocked: 0

## Gaps
