---
created: 2026-10-07T02:59:35.000Z
title: "Decide whether CampaignRun.run_status needs an awarded-and-in-progress value"
area: campaign-runs
severity: minor
files:
  - solsys_code/models.py
---

## Problem

`RunStatus` is the v2.2 one-off lifecycle (requested → planned → observed …). A semester-long queue allocation accruing observations nightly is neither `planned` nor `observed`. Per-target runs mostly avoid it for KEY2026B-004. `run_status` stays a staff decision (TALLY-03).

Source: `.planning/v2.4-INTENT-REVIEW.md`, F3 (v2.4 intent review walkthrough, 2026-10-06).

## Solution

Either add an in-progress value (and say who sets it), or document `planned` as "awarded, ongoing".
