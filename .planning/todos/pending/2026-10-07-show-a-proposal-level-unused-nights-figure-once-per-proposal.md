---
created: 2026-10-07T02:59:35.000Z
title: "Show a proposal-level unused-nights figure once per proposal, not on every run row"
area: campaign-tally
severity: minor
files:
  - solsys_code/campaign_tally.py:406
  - solsys_code/campaign_tables.py
audit_acknowledged:
  milestone: v2.4
  at: 2026-10-06
---

## Problem

`_apply_unused_fields()` gives every run the whole proposal's unused-nights estimate, so all seven KEY2026B-004 rows read `[U] 14` (confirmed live 2026-10-06); the roll-up correctly counts it once.

Source: `.planning/v2.4-INTENT-REVIEW.md`, F4 (v2.4 intent review walkthrough, 2026-10-06).

## Solution

Render the estimate once per proposal code, or label the per-row figure as the proposal's ("proposal: 14").
