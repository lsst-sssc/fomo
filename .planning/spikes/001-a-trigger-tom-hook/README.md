---
spike: 001a
idea: observation-first-calendar
name: trigger-tom-hook
type: comparison
validates: "Given a PENDING ObservationRecord, when its scheduled_start/scheduled_end change with no status change and save() runs, then TOM's observation_change_state hook fires with the record"
verdict: PARTIAL
related: [001b]
tags: [trigger, tom-hooks, observation_change_state, updatestatus, narrowing]
---

# Spike 001a: Trigger via TOM's `observation_change_state` hook

## What This Validates

Given a `PENDING` `ObservationRecord`, when its `scheduled_start`/`scheduled_end` change with no
status change and `save()` runs, then `settings.HOOKS['observation_change_state']` fires with the
record — i.e. can TOM's own hook be the event-driven trigger for calendar narrowing (decision D5)?

Comparison partner: 001b (Django `post_save` receiver). Same six-ish scenarios, different mechanism.

## Research

No external dependencies; grounded in the installed TOM source (read before coding):

- `tom_observations/models.py:58-66` — `ObservationRecord.save()` calls
  `run_hook('observation_change_state', self, presave_status)` **only when `status` changed**, and
  `run_hook(..., None)` on creation. Nothing else fires it.
- `tom_common/hooks.py:9-23` — `run_hook` resolves `settings.HOOKS[name]` by dotted path at call
  time (`import_method` = `rsplit('.',1)` + `import_module` + `getattr`), so a spike can register a
  receiver at runtime by putting its directory on `sys.path` and mutating `settings.HOOKS`.
- `tom_observations/facility.py:555-565` — `update_observation_status()` sets `status`,
  `scheduled_start`, `scheduled_end` and calls `record.save()`; `update_all_observation_statuses()`
  (567-579) excludes terminal states and calls it per record. This is the `updatestatus` path.

## How to Run

```
python manage.py shell -c "import runpy; runpy.run_path('.planning/spikes/001-a-trigger-tom-hook/spike.py', run_name='__main__')"
```

All writes run inside a transaction that is rolled back; the real KEY2026B-004 rows are untouched
(verified after the run: 146 records, 56 queued-only, 0 `spike-001*` rows).

## What to Expect

A five-row table (scenario / fired / expected / ok) and a `summary:` JSON line. The decisive rows
are S1 and S4: whether a schedule-only save fires the hook.

## Observability

`forensic-log.json` — every scenario with the exact hook invocations (record, status,
previous_status, scheduled_start) and the fixture used, ISO timestamps.

## Investigation Trail

1. Registered a logging receiver as `hook_receiver.receiver` via `settings.HOOKS` (the stock
   handler is a logger no-op, so FOMO today gets nothing from this hook).
2. Picked two real queued-only `PENDING` records (scheduled_start null) from KEY2026B-004.
3. S1: set scheduled_start/end, status unchanged, `save()` -> hook did **not** fire.
4. S2: status PENDING->COMPLETED, `save()` -> fired, `previous_status='PENDING'`.
5. S3: created a record (NonSiderealTargetFactory target) -> fired, `previous_status=None`.
6. S4: drove the real `LCOFacility().update_observation_status()` path with
   `get_observation_status` monkeypatched to return a placed block and an **unchanged** state
   (no network) -> hook did **not** fire.
7. S5: same path with state COMPLETED -> fired.

## Results

**Verdict: PARTIAL.** 5/5 scenarios matched expectation (`forensic-log.json` summary:
`schedule_only_fires: false`, `updatestatus_placed_block_fires: false`).

- The hook is a reliable trigger for **status transitions and creation** — enough for
  "observed" and "expired/cancelled" narrowing and for projecting a newly backfilled record.
- It **misses the most common narrowing step**: the scheduler placing a block while the request
  stays `PENDING`. That save changes only `scheduled_start/end`, and `save()` does not call the
  hook for it (`models.py:62`). A projector triggered only by this hook would show the placed
  block late — at the next status change or the next sweep.
- Hook payload is `(record, previous_status)`: no field-change information, so a receiver must
  re-derive the event from the whole record anyway (which is what decision D4 wants).

**Comparison outcome:** loses to 001b for narrowing. Still useful as an *optional* secondary
signal if status-transition semantics (previous status) are wanted, but it cannot be the only
trigger. Surprise: FOMO currently wires this hook to TOM's stock logger and gets no behaviour
from it at all.
