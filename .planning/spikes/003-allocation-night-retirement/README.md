---
spike: 003
idea: observation-first-calendar
name: allocation-night-retirement
type: standard
validates: "Given a campaign=None CampaignRun allocation with its own per-night sunset-to-sunrise events, when a real ObservationRecord is linked to one of its nights, then that night's allocation event retires, the record's base-layer event stands unchanged, the run is attributed to it via CalendarEventMeta.run only, and the whole thing is idempotent and reversible"
verdict: VALIDATED
related: [002]
tags: [allocation, CampaignRun, handoff, narrowing, CalendarEventMeta, sun_event, single-writer]
---

# Spike 003: Allocation-night retirement (the layer handoff)

## What This Validates

Given an allocation — a `CampaignRun` with `campaign=None` (the shape plan 32-01 Task 1 made
legal), a resolved site, and a 3-night window — with one sunset-to-sunrise event per night
under its own `ALLOC:{run.pk}:{night}` key, when a real `ObservationRecord` is linked to it via
`CampaignRunObservation`, then:

- the allocation event for **that night** retires (is deleted) and the other nights stay;
- the record's base-layer event (from spike 002) is **not written** by the campaign side —
  attribution goes through `CalendarEventMeta.run` only (decision D2: annotate, never adopt);
- re-projecting is idempotent; unlinking restores the allocation night (reversible);
- the reconciler's `RUN:` rows are untouched; everything rolls back clean.

This is decision D4 made mechanical: narrowing from intent to actuality is the handoff between
the two projectors, not a dispatch rule inside one.

## Research

- `solsys_code/telescope_runs.py:251` — `sun_event(site, date, kind='sun') -> (setting, rising)`
  as astropy `Time` (UTC); needs `site.timezone`. `Observatory K92` (Sutherland-LCO B,
  `Africa/Johannesburg`) is a real LCO 1 m site with a timezone in this DB.
- `solsys_code/models.py:223-224` — `CampaignRun.window_start/window_end` are `DateField`s
  (null together); `CampaignRunObservation(run, observation_record, confirmed_by, confirmed_at)`
  is the exact-identity link. `unique_campaign_run_source_identifier` lets the spike key its
  run by `SPIKE003:{observation_id}`.
- `CalendarEventMeta.run` docstring (`models.py:11-74`) currently *means* "owning campaign
  run" — the Phase 29 reconciler reads it as ownership. This spike deliberately uses it as an
  attribution link instead (the Phase 28 confirmation queue already does the same).

## How to Run

```
python manage.py shell -c "import runpy; runpy.run_path('.planning/spikes/003-allocation-night-retirement/spike.py', run_name='__main__')"
```
Requires spike 002's sweep to have run (it needs the record's base event). Rolled back entirely.

## What to Expect

Four steps and a `summary:` line with every boolean `true`: 3 allocation events, then 2 after
the link (the record's night retired), `unchanged: 2, retired: 0` on the idempotent re-project,
3 again after unlinking, base event fields unchanged with `meta_run == run.pk`, `RUN:` count
unchanged, and nothing left in the DB afterwards.

## Observability

`forensic-log.json` — fixture (record, night, run, window, site), each step's counters and
the `ALLOC:` URL list, the base event snapshot after linking, and the summary booleans.

## Investigation Trail

1. Chose record 4378323 (11P, queued, window 2026-09-03 18:07 -> 09-04 02:37 UTC; night
   2026-09-03) and built a 3-night allocation 09-02..09-04 at K92 for the same target.
2. Step 1 projected 3 `ALLOC:` events via `sun_event` (sunset->sunrise at Sutherland).
3. Step 2 created the link and re-projected: `retired: 1`, exactly `ALLOC:59:2026-09-03`;
   the base event's title/span/`modified` were byte-identical before and after — the campaign
   side wrote only `CalendarEventMeta.run`.
4. Step 3 re-projected with no change: `unchanged: 2, retired: 0`.
5. Step 4 deleted the link and re-projected: `created: 1`, back to 3 nights.
6. Rollback verified: no run, no link, no meta, no `ALLOC:` rows remain; `RUN:` 74 -> 74.

## Results

**Verdict: VALIDATED.** Every summary boolean true (`forensic-log.json`).

What it says about the design:

- The handoff is a ~40-line rule: "an allocation night with a linked record has no allocation
  event". No new model, no new field; `CampaignRunObservation` is already the link.
- **Attribution must be a link, not a text write.** Writing a campaign prefix into the base
  event's title would be overwritten by the next base-layer re-projection (it rebuilds
  title/description from record state). `CalendarEventMeta.run` survives that; any campaign
  decoration belongs at render time, derived from the meta.
- **Landmine (repeat of 002's):** the existing reconciler reads `CalendarEventMeta.run` as
  *ownership* and has adopt/re-key/detach paths built on that reading. Under D2 the field's
  meaning becomes *attribution*. The build must retire those paths (or scope them to the
  reconciler's own `RUN:`/`ALLOC:`-style keys) before both projectors run together.
- **Night key:** the spike keys nights by the UTC date of the record's span start. The build
  must key by the site-local observing night (the same date `sun_event` takes) — the 32-01
  plan's Chilean/Australian must-have is exactly this edge, and it is not tested here.
- **Cost:** `sun_event` is an astropy solar scan per night (~0.3-1 s). The classical
  reconciler already pays it; the pending todo about skipping it on idempotent sweeps applies
  to the allocation projector too.

Surprises: none mechanical — the surprising thing is how little code the handoff needs once
the two projectors own disjoint key namespaces.
