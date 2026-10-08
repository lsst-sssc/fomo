---
status: testing
phase: 39-calendar-write-access
source: [39-VERIFICATION.md]
started: 2026-10-08T21:14:30Z
updated: 2026-10-08T21:14:30Z
---

## Current Test

number: 1
name: Visitor affordance (judgment-tier prohibition, 39-02)
expected: |
  Open /calendar/ logged out, click an entry and hover over the day cells. Nothing looks editable and nothing prompts a login. Decide whether the day-cell hover tint (.cal-day:hover) is acceptable for visitors or should be scoped to logged-in users.
awaiting: user response

## Tests

### 1. Visitor affordance (judgment-tier prohibition, 39-02)
expected: Open /calendar/ logged out, click an entry and hover over the day cells. Nothing looks editable and nothing prompts a login. Decide whether the day-cell hover tint (.cal-day:hover) is acceptable for visitors or should be scoped to logged-in users. (Verifier's non-authoritative probe: the anonymous card has no form/input/select/textarea/hx-post/write URL; the month view's only control is the utc_offset display select.)
result: [pending]

### 2. Concurrency backstop truth (39-01 A12)
expected: The guard keeps no state between requests, so concurrent or interrupted anonymous requests are each refused independently. Accept the structural evidence (calendar_access.py code AST-identical to 798dfe9; module level holds only a tuple of string constants; no global/nonlocal) as sufficient, or ask for a concurrency test.
result: [pending]

### 3. Runbook paragraph readability (39-04 Task 1 human-check)
expected: docs/runbooks/telescope_runs_calendar.rst lines 2576-2605 read as plain operator guidance, name both refusal paths without jargon beyond "CSRF check", and state the self-registration acceptance with its quoted reason. Approved at the 39-04 Task 1 checkpoint on 2026-10-08; confirm, and decide at the same time whether to add the WR-01 sentence (the create and edit addresses show a bare copy of the form after login; go back to the calendar page instead) and the IN-02 note about the login-page flash message.
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
