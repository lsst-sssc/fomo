---
created: 2026-09-01T17:08:10.247Z
title: Add TTL cache to attribution banner count
area: general
severity: minor
files:
  - solsys_code/campaign_views.py:254
  - solsys_code/campaign_attribution.py:773
  - solsys_code/campaign_gap.py:28
  - solsys_code/tests/test_campaign_attribution_views.py
---

## Problem

Finding F1 from the 2026-09-01 review of the `issue37-code-only` branch (artifact:
https://claude.ai/code/artifact/98811990-b03d-44e7-b9eb-48f967942c5c), downgraded
Medium -> Low after measurement — this is an opportunistic fix, not urgent.

`CampaignListView.get_context_data()` (`campaign_views.py:254`) calls
`campaign_attribution.orphans_needing_attribution_count()` on every campaign-list
page load, for every visitor including anonymous, uncached. That function builds
BOTH full attribution backlogs (scoring, sorting, evidence strings) just to take
`len()` — ~2 SQL queries + Python scoring per unattributed orphan CalendarEvent /
ObservationRecord.

Measured on the real dev DB (2026-09-01, 94 events / 13 records / 49 runs -> 31
orphans): 23 ms and 64 SQL queries per page load — imperceptible today. Cost
scales with the unattributed-orphan population (~0.7 ms + 2 queries each), NOT
with campaigns or runs; at expected usage (1-3 campaigns/yr, 5-50 runs each) it
stays under ~0.25 s even after a busy year with the attribution queue never
drained. Residual risk: an internet-facing instance (FOMO targets are OPEN /
read-only) where crawlers hit /campaigns/ repeatedly, and an unboundedly-growing
orphan population if the attribution queue is never worked.

## Solution

Add a short TTL cache around the banner count, reusing the established
`campaign_gap.py` low-level-cache pattern (`django.core.cache`, cf.
`GAP_CACHE_TTL_SECONDS` at `campaign_gap.py:28`). Sketch:

- In `campaign_attribution.py`, wrap `orphans_needing_attribution_count()` (and
  optionally `unattributable_orphan_count()`) in a cache-get-or-compute with a
  short TTL (~5-15 min) and a module-level key constant.
- Keep `AttributionQueueView` reading the FRESH value (or accept the small TTL
  staleness there too — decide during implementation; the queue page's own
  worklists are already computed fresh, so a slightly stale header count is
  probably fine, but the D-15 "is_drained" done-state signal reads it).
- A confirm/dismiss/undo action in `AttributionDecisionView` should invalidate
  the cache key so the banner doesn't promise a backlog that was just drained
  (mirror how staleness is handled, or simply delete the key in each action).
- Add a test asserting the second call within the TTL issues no backlog queries
  (`CaptureQueriesContext` precedent in test_calendar_template.py's N+1 test)
  and that a decision action invalidates the count.

Also worth folding in while there: `event_attribution_backlog()` /
`record_attribution_backlog()` call `candidates_for_event/record()` WITHOUT the
`dismissed_run_ids` prefetch argument those functions provide precisely for
batch callers (see the docstring at `campaign_attribution.py:557`) — passing a
single prefetched dismissal map would drop one query per orphan even on cache
misses.
