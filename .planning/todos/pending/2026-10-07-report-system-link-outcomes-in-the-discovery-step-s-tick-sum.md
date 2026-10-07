---
created: 2026-10-07T02:59:35.000Z
title: "Report system-link outcomes in the discovery step's tick summary"
area: unattended-discovery
severity: minor
files:
  - solsys_code/unattended.py:427
  - solsys_code/campaign_system_links.py
audit_acknowledged:
  milestone: v2.4
  at: 2026-10-06
---

## Problem

`step_discovery` builds its `StepResult` from swept/failed counts only; a skipped or failed ALLOC-06 system link shows only as a captured INFO line, not in the summary an operator reads nor in the `failed` flag.

Source: `.planning/v2.4-INTENT-REVIEW.md`, v2.4 milestone audit, finding 1 (v2.4 intent review walkthrough, 2026-10-06).

## Solution

Carry `linked` / `links skipped` / `links failed` counts into the discovery summary; decide whether a link failure marks the step failed.
