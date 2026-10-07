---
created: 2026-10-07T02:59:35.000Z
title: "A failed or aborted record keeps its last scheduled window instead of the original request window"
area: observation-projector
severity: minor
files:
  - solsys_code/calendar_utils.py
  - solsys_code/observation_projector.py
---

## Problem

A failure-state record's event falls back to the original request window; one that had been placed should keep its last scheduled block. Expired records keep the original window.

Source: `.planning/v2.4-INTENT-REVIEW.md`, Q6 (34-UAT test 2 note) (v2.4 intent review walkthrough, 2026-10-06).

## Solution

In the stage rule, prefer the stored placed block for failed/aborted records; keep the request window for window-expired ones.
