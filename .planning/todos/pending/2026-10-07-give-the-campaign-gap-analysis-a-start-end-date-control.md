---
created: 2026-10-07T02:59:35.000Z
title: "Give the campaign gap analysis a start/end date control"
area: campaign-gap
severity: minor
files:
  - solsys_code/campaign_views.py
  - src/templates/campaigns/campaignrun_gap_analysis.html
---

## Problem

The gap analysis looks forward from today with no date control, so a past campaign (Didymos July 2026) cannot be checked; GAPB-01 could not be confirmed live.

Source: `.planning/v2.4-INTENT-REVIEW.md`, GAPB-01 check (v2.4 intent review walkthrough, 2026-10-06).

## Solution

Add a date range (defaulting to today onward), so past windows can be inspected.
