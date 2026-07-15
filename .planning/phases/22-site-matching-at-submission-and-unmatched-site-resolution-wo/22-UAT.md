---
status: testing
phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
source: [22-VERIFICATION.md]
started: 2026-07-15T18:00:00Z
updated: 2026-07-15T18:00:00Z
---

## Current Test

number: 1
name: Public form live-search fires, debounces, and fills in a real browser
expected: |
  An hx-get fires to /campaigns/site-search/, a suggestion list appears below the field showing
  both Faulkes sites as 'Display Name (obscode)', and clicking one fills the input with that
  exact text. Typing 1 character does nothing; no request fires before 2 characters.
awaiting: user response

## Tests

### 1. Public form live-search fires, debounces, and fills in a real browser
expected: On the public 'Submit an Observing Run' form, typing 'faulkes' (2+ chars) into the
  Observing site field and waiting ~300ms fires an hx-get to /campaigns/site-search/; a
  suggestion list appears below the field showing both Faulkes sites as 'Display Name
  (obscode)'; clicking one fills the input with that exact text. Typing 1 character does
  nothing — no request fires before 2 characters.
result: [pending]

### 2. Multi-row approval-queue widgets don't cross-fill between rows
expected: In the staff approval queue, using the pending-row site input and the Sites Needing
  Review row's site input to search and pick a suggestion fills the correct row's input (not a
  different row's); the 'Create new Observatory' link still works; submitting resolves/approves
  as expected.
result: [pending]

### 3. Sites Needing Review resolve UX for the CR-01 blank-timezone fix
expected: Resolving a Sites Needing Review row end-to-end against a real (or realistic Tier-2)
  MPC obscode with a blank Observatory.timezone shows a warning, keeps the row in the table, and
  a later Resolve retries — the banner and row persistence behave as CR-01's fix describes when
  driven through the actual UI, not just the regression test's direct POST.
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
