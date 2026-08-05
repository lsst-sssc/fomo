---
status: testing
phase: 29-the-reconciler
source: [29-VERIFICATION.md]
started: 2026-08-05T22:35:00Z
updated: 2026-08-05T22:35:00Z
---

## Current Test

number: 1
name: Live-browser confirmation of /calendar/ rendering and the Campaign run pop-up block
expected: |
  Calendar renders the two key families visibly distinct — each queue-scheduled 3I/ATLAS run
  as one whole-window entry, each classical run as one entry per observing night. Clicking a
  reconciler-owned entry shows a "Campaign run" block naming the run, its window and its
  status, with no manual admin linking needed.
awaiting: user response

## Tests

### 1. Live-browser confirmation of /calendar/ rendering and the Campaign run pop-up block
expected: Calendar renders the two key families visibly distinct; the pop-up block appears
  automatically for reconciler-owned events, naming the run/window/status.
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
