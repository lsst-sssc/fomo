---
status: testing
phase: 36-unattended-operation
source: [36-VERIFICATION.md]
started: 2026-09-18T21:10:00Z
updated: 2026-09-18T21:10:00Z
---

<!-- Round 4. Rounds 1-3 (30 tests, 29 passed, 1 issue -> G-36-5) are preserved in this
     file's git history. G-36-5 was closed by gap-closure plan 36-09 (commits db8a1cc,
     b5f40a2, be955cf) and re-verified in 36-VERIFICATION.md round 4: 88/88 must-haves,
     the operator's own reproduction re-run at HEAD 93ef89c prints the watched_proposals
     warning exactly once. The two items below are the only remaining human judgments. -->

## Current Test

number: 1
name: Skim the step-6 stream-routing passage added by plan 36-09
expected: |
  The added stream-routing paragraph (docs/runbooks/telescope_runs_calendar.rst:1620-1638,
  inside step 6 of "Setting it up on a fresh host") reads in the surrounding operator voice,
  and a fresh-host operator finishes step 6 knowing that passing lines go to standard
  output, that warnings and failures go to standard error instead, that each line is
  written once, that a bare `>` silently drops every warning and failure, and that `2>&1`
  puts the whole report in one file in check order. Every factual claim in it is already
  verified against the code -- this is a prose-quality and point-of-use-sufficiency
  judgment only.
awaiting: user response

## Tests

### 1. Skim the step-6 stream-routing passage added by plan 36-09
expected: The added paragraph at docs/runbooks/telescope_runs_calendar.rst:1620-1638 reads in the surrounding operator voice; after step 6 an operator knows passing lines go to stdout, warnings/failures go to stderr instead, each line is written once, a bare `>` drops every warning and failure, and `2>&1` puts the whole report in one file in check order. Ideally read by the same person who ran round-3 Tests 2 and 3, since that read-through predates this passage.
result: [pending]

### 2. Confirm the six judgment-tier prohibition verdicts for plan 36-09
expected: The "Plan 36-09 Prohibitions" table in 36-VERIFICATION.md shows all six judged Satisfied on the evidence given -- only three files changed; the operator's UAT wording survived into db8a1cc with its provenance intact; no explicit style argument reaches the standard-error write; no credential value in any committed file; the G-36-1/G-36-3/G-36-4 gates all re-run green; 36-UAT.md and 36-VERIFICATION.md untouched by the plan itself. Confirm or dispute each verdict.
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
