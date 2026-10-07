---
created: 2026-10-07T02:59:35.000Z
title: "Link each campaign table row to its run, or give the target its own column"
area: campaign-table
severity: cosmetic
files:
  - solsys_code/campaign_tables.py
  - src/templates/campaigns/campaignrun_table.html
---

## Problem

Campaign table rows are neither links nor show a run id; a row is identifiable only by the target folded into `telescope_instrument` to satisfy `unique_campaign_run_resolved_window`.

Source: `.planning/v2.4-INTENT-REVIEW.md`, F15 (v2.4 intent review walkthrough, 2026-10-06).

## Solution

Link the row (or its instrument cell) to the run, and/or add a target column; then the per-target runs need not carry the target in `telescope_instrument`.
