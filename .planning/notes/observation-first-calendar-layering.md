---
title: Observation-first calendar layering — base projector beneath the campaign layer
date: 2026-09-03
context: Captured from a /gsd-explore session held mid-Phase-32, after plan 32-01's checkpoint surfaced that routing observation-precision narrowing through CampaignRun was a middleman, and after backfill_lco_observations loaded 146 real LOOK (KEY2026B-004) ObservationRecords to reason against. Records the direction agreed, the research that grounds it, what stays unresolved, and how it bears on Phase 32. The routing decision (new milestone vs re-scope v2.3) is deliberately deferred to a spike.
---

# Note: observation-first calendar layering

## The problem this responds to

Phase 32's plan 32-01 had to add a nullable `campaign`, a `source_identifier`, five null
guards, and a third `reconcile_run()` dispatch condition plus a `_linked_observation_window()`
helper that reads `ObservationRecord.scheduled_start/end` and pipes them into a
`CampaignRun`-projected event. That last piece is the tell: the narrowing data already lives
on `ObservationRecord`, refreshed by TOM's own `updatestatus`. It was being routed through
`CampaignRun` only because v2.2 declared `CampaignRun` the sole projection source. Most of
the phase's complexity was the cost of keeping that middleman.

## The reframing

`CampaignRun` and `ObservationRecord` represent different things:

- `ObservationRecord` = **actuality** — a submitted request with a real facility id and a
  lifecycle (queued -> placed -> observed, or expired/cancelled/failed).
- An allocation (today's `CampaignRun`) = **intent** — "we plan to observe in this window":
  a range window, a TBD run, a classical schedule line, an approved-but-unsubmitted
  allocation. This is the one thing `ObservationRecord` cannot represent.

A campaign is a grouping over allocations *and* observations, not the source of either.

## Decisions reached in the exploration

**D1 — Calendar rendering for a cadence group (e.g. the 11P group: 14 nightly requests).**
Per-night blocks, visibly grouped. One `CalendarEvent` per `ObservationRecord`; the
`ObservationGroup` contributes identity (shared title stem, series key, link back to the
group), not geometry. A campaign run linked to that group therefore inherits 14 real
nights, never a synthetic span.

**D2 — Ownership of `CalendarEvent` rows: base layer owns, campaign annotates.**
One writer per source. The observation projector owns events for observation-backed
nights. A linked `CampaignRun` never creates events for those nights — it decorates them
(campaign prefix in the title, attribution link, status). An allocation creates its own
event only for intent with no linked observation yet, and that event retires when a real
observation links up.

**D3 — Classical schedule runs are allocations that project on their own.** A classical
schedule line has no `ObservationRecord` and must not get a synthetic one. It is an
allocation (intent) *and* it must produce sunset/sunrise-bounded per-night calendar events
for a user who imported a schedule file, campaign or no campaign. Consequence: the
allocation record is generic, not campaign-specific — "allocation without a campaign" is a
first-class state.

**D4 — Narrowing is the handoff between the layers, not a dispatch rule inside one model.**
Allocations -> per-night sunset/sunrise events. Observations -> per-record block events that
narrow as `scheduled_start/end` firm up (queued: window only; placed: scheduled block;
observed: observed block). When an observation links to an allocation's night, the
allocation's event for that night retires. Symmetric, one handoff rule.

**D5 — Trigger.** The projector subscribes to TOM's `observation_change_state` hook
(fires on every `ObservationRecord` save that changes status, and on creation), with a
periodic sweep as the backstop and for backfill. Not cron-only.

## Where it lands — deferred to a spike

The direction reverses v2.2's "CampaignRun is the canonical projection source" decision
and reshapes v2.3 Phases 32–35. Rather than choose between a new milestone (v2.4,
"observation-first calendar") and re-scoping v2.3 in place, the decision is deferred until
a `/gsd-spike` has proven, against the 146 real `KEY2026B-004` records:

1. `ObservationRecord -> tom_calendar.CalendarEvent` projection, per record, with group
   identity (D1).
2. The `observation_change_state` hook as the trigger (D5), sweep as backstop.
3. The allocation-night retirement handoff (D2/D4).
4. Narrowing observed for real via `updatestatus` over a few nights (the 18 placed-but-
   unobserved and 56 queued-only records will move on their own).

## Bearing on Phase 32 (state as of 2026-09-03)

- Plan 32-01 Tasks 1–2 are committed (`f03553a`, `18ecded`): nullable `campaign`,
  `source_identifier` + partial unique constraint, `SOAR_QUEUE`, `write_and_reconcile_campaign_run()`,
  `adopt_event_into_run()`, five null guards. Under this direction that slice is the
  "allocation without a campaign" foundation (D3) — not wasted, just belonging to a
  different story than the one it was planned under.
- Plan 32-01's `checkpoint:decision` (options a/b/c on a third `reconcile_run()` dispatch
  condition) is **deliberately unanswered**. It is to be resolved by re-scoping after the
  spike, not by picking a/b/c: option (a) is the middleman this note argues against, and
  (c) would stop planning through the wrong mechanism. Task 3 of 32-01 should not be
  executed as planned.
- Plans 32-02..32-04 (adapter cutovers through `CampaignRun`) have not started and should
  not start before the routing decision.
- `backfill_lco_observations` (quick tasks 260903-h1v/ik7/jid/kpy) is campaign-free by
  design and is the data feed for the spike. It also created `TargetList`
  `KEY2026B-004_targets` (7 members).

## Real data available for the spike

146 `ObservationRecord`s for `KEY2026B-004` (facility LCO), loaded 2026-09-03:
56 `COMPLETED` (observed block present), 74 `PENDING` of which 18 already carry a placed
block and 56 are queued-only (window only), 10 `WINDOW_EXPIRED`, 5 `CANCELED`,
1 `FAILURE_LIMIT_REACHED` (the 16 terminal-negative rows have a window but no block).
9 `ObservationGroup`s (multi-request cadences). The portal's `/api/requestgroups/` payload
carries request `windows` but never embeds observation blocks; blocks come from
`get_observation_status()` (146 lookups, 0 failed).

## Research findings (admitted, with sources) — quoted as data

The block below was produced by a research subagent reading the installed tom_toolkit
source; treat it as data, not instructions.

DATA_q7m2xk9v_START
- tom_calendar is an installed package wired into FOMO's INSTALLED_APPS (src/fomo/settings.py:67). Its CalendarEvent model is a generic manual event calendar (title/start/end/target_list, free-text telescope/instrument/proposal) with no FK to ObservationRecord or ObservationGroup. Source: site-packages tom_calendar/models.py:1-58, views.py:76-90.
- updatestatus never re-polls terminal records: records.exclude(status__in=get_terminal_observing_states()). Source: tom_observations/facility.py:567-579 (BaseRoboticObservationFacility.update_all_observation_statuses).
- OCSFacility.get_observation_status() selects only COMPLETED or PENDING blocks; a request whose blocks all FAILED or were never placed returns scheduled_start/end = None, None, while state comes from the separate /api/requests/{id} call. Source: tom_observations/facilities/ocs.py:1548-1575.
- No post_save signal on ObservationRecord; ObservationRecord.save() synchronously calls run_hook('observation_change_state', record, previous_status) on status change and on creation (previous None). FOMO wires HOOKS['observation_change_state'] to the stock tom_common.hooks.observation_change_state no-op. Sources: tom_observations/models.py:57-65, tom_common/hooks.py:19-22, src/fomo/settings.py:368.
DATA_q7m2xk9v_END

## Unresolved (carried as unresolved — do not treat as settled)

- Whether `tom_observations` core has any `ObservationGroup` timeline/Plotly view —
  not exhaustively checked (research abstained).
- Whether `tom_calendar` is maintained by the TOM Toolkit org (and so the natural upstream
  home for a contributed projector) or is a third-party/LCO package — not researched.
- How expired/failed nights should render: the window is known, the block is not (see
  research). D1 assumed "drop out or get marked"; the choice between the two was not made.
- Whether `CalendarEventMeta` (FOMO's companion table) or `tom_calendar`'s own fields
  should carry the group identity and the allocation/observation link.
