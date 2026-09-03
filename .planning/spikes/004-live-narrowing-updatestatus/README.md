---
spike: 004
idea: observation-first-calendar
name: live-narrowing-updatestatus
type: standard
validates: "Given the 74 PENDING KEY2026B-004 records with base-layer events, when TOM's own updatestatus runs on later nights with no spike code involved, then their events narrow (window -> placed block -> observed block) or get marked expired, with no code change"
verdict: PARTIAL
related: [002]
tags: [narrowing, updatestatus, live-data, observability, cadence]
---

# Spike 004: Live narrowing via TOM's `updatestatus` (multi-night)

## What This Validates

Given the 74 `PENDING` KEY2026B-004 records (56 queued-only, 18 placed) and their base-layer
events from spike 002, when TOM's own `updatestatus` runs on later nights — no spike code in
the loop — then the events narrow from request window to placed block to observed block, or
get marked expired/cancelled, with no code change.

This cannot be closed in one session: it needs real nights to pass. Today's work sets up the
baseline and a re-check that reports exactly what moved.

## Research

- `tom_observations/facility.py:567-579` — `update_all_observation_statuses()` polls every
  non-terminal record for the facility, one portal call each, and calls `record.save()`
  through `update_observation_status()` (555-565). Sized against the live DB: **74
  non-terminal LCO records, all of them KEY2026B-004** — so a run costs 74 calls and touches
  nothing else.
- `ocs.py:1548-1575` — placed blocks come back as `PENDING` blocks with times; observed as
  `COMPLETED`; failed-only as `None, None` with the request's state.
- Spike 001b — `post_save` fires from this path. But the receiver only exists while a spike
  script is running; FOMO's `apps.ready()` does not install it. So a standalone `updatestatus`
  process will update records **without** re-projecting events, and the re-check must run the
  sweep first (the backstop path). That is itself a useful demonstration of D5's two halves.

## How to Run

```
# (done today) baseline snapshot — no DB writes
python manage.py shell -c "import runpy; runpy.run_path('.planning/spikes/004-live-narrowing-updatestatus/recheck.py', run_name='__main__')"

# on a later night, TOM's own refresh (74 portal calls, no spike code)
python manage.py updatestatus

# then the re-check: sweeps (backstop), diffs against the baseline, appends to recheck-history.json
python manage.py shell -c "import runpy; runpy.run_path('.planning/spikes/004-live-narrowing-updatestatus/recheck.py', run_name='__main__')"
```

## What to Expect

Baseline (captured 2026-09-03T23:46Z): `stages {'terminal-negative': 16, 'observed': 56,
'queued': 56, 'placed': 18}`. After a night or two: `transitions` such as
`queued -> placed`, `placed -> observed`, `queued -> terminal-negative`, with per-record
examples showing `span_hours` dropping from ~8-17 h (window) to ~0.3 h (block) and titles
moving `[QUEUED]` -> `[SCHEDULED]` -> clean, or -> `[EXPIRED]`. The sweep counters show how
many events the backstop had to update.

## Observability

- `baseline.json` — per-record status, stage, schedule, event span/title/hours at capture.
- `recheck-history.json` — one entry per re-check: sweep counters, stage totals, transition
  counts, narrowed/advanced/expired counts, up to 8 worked examples.

## Investigation Trail

1. Confirmed `updatestatus`'s scope on the live DB (74 records, all this proposal).
2. Captured the baseline: 146 records, stage counts matching spike 002's sweep exactly.
3. No re-check yet — waiting on real nights. The 11P/10P/220P cadences submitted 2026-09-03
   have windows every night through mid-September, so the first re-check after one or two
   nights should already show `queued -> placed` and `queued -> observed` transitions.

## Results

**Verdict: PARTIAL (by construction today).** Observability and the re-check are in place;
the live evidence accrues over the coming week. To close: run `updatestatus` + the re-check on
two or three later nights, paste the printed transition summary into this section, and flip the
verdict to VALIDATED if events narrowed with no code change (or record what didn't).

Design notes already visible:
- The receiver must be installed by the app (`apps.ready()`), not by a script, or every
  external `updatestatus` run leaves events stale until the next sweep — the exact gap D5's
  "sweep as backstop" covers, but it should be the exception, not the norm.
- A sweep over 146 records costs ~0.1-0.3 s (spike 002), so running it right after
  `updatestatus` in the same cron slot is a perfectly good interim design.
