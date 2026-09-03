# Spike Manifest

## Ideas

### observation-first-calendar
Prove that a campaign-free base-layer projector can turn `tom_observations.ObservationRecord`
rows into `tom_calendar.CalendarEvent` rows — one event per record, with `ObservationGroup`
series identity — narrowing automatically from the request window to the placed block to the
observed block as TOM's own `updatestatus` refreshes `scheduled_start/end`; that the trigger can
be event-driven (TOM hook or Django signal) with a sweep as backstop; and that an allocation
(a `CampaignRun` with no campaign) retires its own per-night event when a real observation
links to that night. Real data: the 146 `KEY2026B-004` records loaded by
`backfill_lco_observations` on 2026-09-03. Design context: `.planning/notes/observation-first-calendar-layering.md`
(decisions D1–D5). The spike's outcome decides whether this becomes a new milestone (v2.4) or
a re-scope of v2.3 Phases 32–35.

**Requirements:**
- Per-night blocks, visibly grouped: one CalendarEvent per ObservationRecord; the ObservationGroup contributes identity (shared title stem, series key, link), never geometry. (D1)
- Base layer owns observation-backed CalendarEvents; a linked CampaignRun only annotates them. An allocation creates its own event only for intent with no linked observation, and that event retires when one links up. (D2)
- A classical schedule run is an allocation that must project sunset/sunrise-bounded nights on its own, campaign or no campaign; never a synthetic ObservationRecord. (D3)
- Narrowing is the allocation→observation handoff plus the observation projector re-deriving from current record state; no third dispatch condition inside the campaign reconciler. (D4)
- Trigger must be event-driven per record save with a sweep as backstop, not cron-only. (D5)
- Spike code must not modify campaign_reconciler.py, campaign_utils.py, the adapters, or plan 32-01's committed slice, and must never write to CalendarEvent rows in the reconciler's `RUN:` namespace.
- Terminal-negative records (WINDOW_EXPIRED / CANCELED / FAILURE_LIMIT_REACHED) are marked on the calendar, never silently dropped.
- Any Target fixture uses `tom_targets.tests.factories.NonSiderealTargetFactory`.
- Trigger = a FOMO-owned Django `post_save` receiver on `ObservationRecord` (catches schedule-only placement saves and the real `updatestatus` path); TOM's `observation_change_state` hook is optional for transition semantics only; a sweep command remains the backstop for bulk `update()`/`bulk_create()` paths. (spike 001, refines D5)
- The per-save projection must be idempotent, no-churn (write only on change), and cheap — it runs inside the caller's transaction on every save.

## Spikes

| # | Idea | Name | Type | Validates | Verdict | Tags |
|---|------|------|------|-----------|---------|------|
| 001a | observation-first-calendar | trigger-tom-hook | comparison | Given a PENDING record, when scheduled_start/end change with no status change and save() runs, then observation_change_state fires | PARTIAL ⚠ — fires on creation/status change only; misses placement | trigger, tom-hooks, updatestatus |
| 001b | observation-first-calendar | trigger-django-post-save | comparison | Same, via a FOMO post_save receiver; also fires from the real updatestatus path | VALIDATED ✓ WINNER — fires on every save() incl. placement and updatestatus; silent on queryset.update() | trigger, django-signals, updatestatus |
