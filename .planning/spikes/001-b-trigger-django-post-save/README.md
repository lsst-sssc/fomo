---
spike: 001b
idea: observation-first-calendar
name: trigger-django-post-save
type: comparison
validates: "Given a PENDING ObservationRecord, when its scheduled_start/scheduled_end change with no status change and save() runs (directly or via updatestatus), then a FOMO-owned Django post_save receiver fires with the record"
verdict: VALIDATED
related: [001a]
tags: [trigger, django-signals, post_save, updatestatus, narrowing]
---

# Spike 001b: Trigger via a Django `post_save` receiver on `ObservationRecord`

## What This Validates

Given a `PENDING` `ObservationRecord`, when its `scheduled_start`/`scheduled_end` change with no
status change and `save()` runs — directly, or through TOM's `update_observation_status()` —
then a FOMO-owned `post_save` receiver connected to `ObservationRecord` fires with the record.
Also probes the two edges a projector must know about: `queryset.update()` (bypasses signals)
and `save(update_fields=...)` (does the receiver learn which fields changed?).

Comparison partner: 001a (TOM's `observation_change_state` hook).

## Research

Same grounding as 001a (`models.py:58-66`, `facility.py:555-579`). Django's `post_save` is a
built-in model signal: it fires for every `Model.save()` regardless of which fields changed, with
`created`, `raw` and `update_fields` kwargs; it does **not** fire for `QuerySet.update()`,
`bulk_create()` or `bulk_update()`. Nothing in TOM prevents a downstream app from connecting a
receiver to `ObservationRecord`.

## How to Run

```
python manage.py shell -c "import runpy; runpy.run_path('.planning/spikes/001-b-trigger-django-post-save/spike.py', run_name='__main__')"
```

All writes are rolled back; the receiver is connected with a `dispatch_uid` and disconnected in a
`finally`, so nothing leaks into the process or the database.

## What to Expect

A six-row table and a `summary:` JSON line. Decisive rows: S1 and S4 fire; S5 does not; S6 reports
`update_fields == ['scheduled_end']`.

## Observability

`forensic-log.json` — every scenario with the exact receiver invocations (pk, observation_id,
status, created, raw, update_fields, scheduled_start), ISO timestamps.

## Investigation Trail

1. Connected `receiver` to `post_save` for `ObservationRecord` (`weak=False`, `dispatch_uid`).
2. Same two real queued-only records as 001a.
3. S1 schedule-only save -> fired (`created=False`, `update_fields=None`).
4. S2 status change -> fired. (TOM's stock hook also logged "from PENDING to COMPLETED" here,
   confirming the two mechanisms coexist without interfering.)
5. S3 creation -> fired with `created=True`.
6. S4 real `update_observation_status()` path, placed block, unchanged state -> fired
   (`update_fields=None`, because TOM calls a bare `record.save()`).
7. S5 `QuerySet.update(scheduled_end=...)` -> did **not** fire. Documents why a sweep backstop
   is still required for any bulk path.
8. S6 `save(update_fields=['scheduled_end'])` -> fired and the receiver saw
   `update_fields=['scheduled_end']`, so callers that pass `update_fields` give the projector a
   cheap "did the schedule change?" test; TOM's own path does not, so the projector must
   re-derive from the whole record (decision D4 already assumes this).

## Results

**Verdict: VALIDATED — WINNER of the 001 comparison.** 6/6 scenarios matched expectation
(`forensic-log.json` summary: `schedule_only_fires: true`, `updatestatus_placed_block_fires: true`,
`queryset_update_fires: false`, `update_fields_seen_in_s6: ["scheduled_end"]`).

- A `post_save` receiver catches every save a projector cares about, including the placement
  step TOM's hook misses and the real `updatestatus` path.
- The receiver must be idempotent and cheap: it fires on every save, including saves that change
  nothing relevant (S6-style partial saves, re-saves). Re-deriving the event from current record
  state and writing only on change (the reconciler's existing no-churn discipline) handles this.
- Bulk paths (`update()`, `bulk_create()`) are invisible to it; the sweep command remains the
  backstop and the backfill path. `backfill_lco_observations` uses `get_or_create`/`save`, so it
  *does* trigger the receiver per record — a fresh backfill projects itself.
- Signals fire inside the caller's transaction; the receiver's own writes roll back with it.
  Fine for correctness, but a projector should not do slow work (sun calculations) inline here —
  it should mark-and-defer or keep the per-record projection cheap.

**Design consequence for D5:** trigger = Django `post_save` on `ObservationRecord` (FOMO-owned,
connected in `solsys_code.apps.ready()`), sweep as backstop. TOM's hook is optional for
transition-specific semantics, not the primary trigger.
