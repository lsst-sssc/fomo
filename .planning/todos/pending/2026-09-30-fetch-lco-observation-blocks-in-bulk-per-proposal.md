---
created: 2026-09-30T13:00:00.000Z
title: Fetch LCO observation blocks in bulk per proposal instead of one call per request
area: unattended-discovery
severity: minor
files:
  - solsys_code/management/commands/backfill_lco_observations.py
  - solsys_code/calendar_utils.py:293
  - solsys_code/management/commands/project_observation_calendar.py:43
audit_acknowledged:
  milestone: v2.4
  at: 2026-10-06
---

## Problem

Intent-review finding F2 (`.planning/v2.4-INTENT-REVIEW.md`, "Found during the walkthrough"),
option B — deferred after option A (state-gated fallback lookups) was chosen for the
pre-close fix.

Discovery resolves each request's schedule with one live call to
`/api/requests/{id}/observations/` (`_resolve_schedule` →
`facility.get_observation_status()`), because the `/api/requestgroups/` list payload carries
no `observations` blocks. The sweep's one-time observed-site lookup
(`calendar_utils.resolve_placement_block`) hits the same endpoint per record. Cost is
O(requests) per proposal per tick: ~1.3 s per call on the live host.

Option A removes the calls for terminal records, so steady state is ~PENDING + new per tick.
What remains is still one call per PENDING/new request, and a large burst on the first sweep
of any newly watched proposal (181 calls for a 2-month-old proposal; ~1,600 for an
18-month-old one).

## Proposed fix

Replace per-request lookups with one paginated listing of the proposal's blocks —
`/api/observations/?proposal=<code>` (optionally `request_group_id=` per group, or
`start_after=` bounded to the sweep window) — and match blocks to requests locally by
`request` id, applying the same first-COMPLETED-else-last-PENDING rule as `_select_block`.
Serve `resolve_placement_block` from the same fetched set so the sweep's site lookup shares
it. O(pages) per proposal, not O(requests).

**Verify first:** which filters `/api/observations/` actually accepts on the production
portal (proposal, request_group_id, request_id, start_after/before, state) and its page size,
before planning. Keep the per-request call as the fallback path for a block the listing did
not return.

## Acceptance

- First sweep of a proposal with N requests makes ceil(N_blocks / page_size) + 1 portal calls,
  not N.
- `fallback lookups needed` in `last_run_summary` drops to the number of requests the bulk
  listing could not resolve.
- Dry-run/real-run `updated`/`unchanged` agreement (T-ik7-02) holds; paired
  `backfill_lco_observations_demo.ipynb` updated.
