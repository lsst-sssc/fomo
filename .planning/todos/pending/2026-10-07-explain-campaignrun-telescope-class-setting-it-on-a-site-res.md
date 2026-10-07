---
created: 2026-10-07T02:59:35.000Z
title: "Explain CampaignRun.telescope_class: setting it on a site-resolved run switches it to one container and deletes its nights"
area: campaign-runs
severity: minor
files:
  - solsys_code/models.py:415
  - solsys_code/admin.py
  - docs/runbooks/telescope_runs_calendar.rst
---

## Problem

Setting *Telescope class allocation* on a site-resolved run (natural reading: "it is a 2m0 proposal") routes it to the whole-window container on the next reconcile and deletes its per-night `ALLOC:` events, with nothing at the edit point saying so. Hit live on run 1, 2026-10-02.

Source: `.planning/v2.4-INTENT-REVIEW.md`, F6 (v2.4 intent review walkthrough, 2026-10-06).

## Solution

`help_text` on the field and admin form: for class-wide, site-less allocations only; on a site-resolved run it replaces the per-night events with one container. A runbook line beside the `run_status` edit notes. Consider a `clean()` warning (not an error — CR-01) when both `site` and `telescope_class` are set.
