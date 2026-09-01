---
status: complete
phase: 29-the-reconciler
source: [29-VERIFICATION.md]
started: 2026-08-05T22:35:00Z
updated: 2026-08-05T23:10:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Live-browser confirmation of /calendar/ rendering and the Campaign run pop-up block
expected: Calendar renders the two key families visibly distinct; the pop-up block appears
  automatically for reconciler-owned events, naming the run/window/status.
result: pass
notes: |
  Confirmed on /calendar/ for July 2025: RUN:29 (LCO 1m queue run) renders as a single
  whole-window entry spanning 2025-07-05..2025-09-22; RUN:9 and RUN:22 (classical) render as
  separate per-night entries on 2025-07-03/07-04. Follow-up question about RUN:3 (ESO VLT
  FORS2) resolved: site correctly maps to MPC 309 (Paranal); the whole-night (00:00-23:59)
  display is correct because RUN:3's source is `eso_queue`, which the reconciler dispatches
  to the whole-window container branch regardless of site resolution (campaign_reconciler.py
  reconcile_run(), QUEUE_SOURCES branch) -- not a bug.

## Summary

total: 1
passed: 1
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
