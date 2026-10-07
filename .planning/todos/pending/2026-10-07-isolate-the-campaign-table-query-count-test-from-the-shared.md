---
created: 2026-10-07T02:59:35.000Z
title: "Isolate the campaign table query-count test from the shared file cache"
area: tests
severity: minor
files:
  - solsys_code/tests/test_campaign_views.py
---

## Problem

`TestCampaignRunTableProgressColumn.test_page_query_count_grows_by_a_bounded_per_row_amount_not_unboundedly` calls `cache.clear()` on the real shared file cache, which parallel workers and the cron tick also use; it raced once under `--parallel 4` (2 != 3), passing alone and on re-run.

Source: `.planning/v2.4-INTENT-REVIEW.md`, 261006-nga executor note (v2.4 intent review walkthrough, 2026-10-06).

## Solution

Put the class under `@override_settings(CACHES=TEST_CACHES)` (a local-memory cache).
