---
spike: 002
idea: observation-first-calendar
name: observation-projector
type: standard
validates: "Given the 146 real KEY2026B-004 ObservationRecords, when projected by a campaign-free projector, then each has exactly one tom_calendar.CalendarEvent with a stage-correct span, terminal-negative nights are marked not dropped, the 9 groups carry series identity, a second run creates/updates nothing, and a single record save re-projects its own event through post_save with no sweep"
verdict: VALIDATED
related: [001b, 003]
tags: [projector, tom_calendar, ObservationRecord, ObservationGroup, narrowing, post_save, idempotency]
---

# Spike 002: Observation projector (ObservationRecord -> tom_calendar.CalendarEvent)

## What This Validates

Given the 146 real `KEY2026B-004` `ObservationRecord`s loaded by `backfill_lco_observations`,
when projected by a campaign-free base-layer projector, then:

1. every record has exactly one `CalendarEvent` with a stage-correct span, and the 16
   terminal-negative records are marked rather than dropped;
2. the 9 `ObservationGroup`s' events carry visible series identity;
3. a second projection run creates and updates nothing (idempotent, no churn);
4. mutating one `PENDING` record's `scheduled_start/end` and saving it re-projects its event
   through `post_save` with no sweep (decision D5 as refined by spike 001b).

## Research

Grounded in FOMO's own code and the installed TOM (read before coding, all reused read-only):

- `solsys_code/calendar_utils.py` — `insert_or_create_calendar_event(lookup, fields)` is the
  no-churn create/update/unchanged helper all three sync commands already share (keyed by
  `url`); `record_time_window(record)` already encodes the stage rule (request window from
  `parameters['start'/'end']` while both scheduled fields are null, else the block; raises on
  a half-set pair); `extract_instrument` / `coarse_telescope_label` give instrument and
  aperture-class labels with no network.
- `solsys_code/management/commands/sync_lco_observation_calendar.py` — `_failure_prefix()` and
  `_FAILURE_PREFIX_BY_STATUS` (`[EXPIRED]`, `[CANCELLED]`, `[FAILED]`) are the existing
  terminal-failure vocabulary; `_title_for` shows `[QUEUED]` for a no-block non-successful
  record and a clean title otherwise.
- `tom_observations/facilities/ocs.py:1406` — `get_observation_url(observation_id)` =
  `portal_url + /requests/{id}`; the existing LCO sync already keys its events by this URL,
  so the projector reuses that namespace (10 such rows exist for other requests; none for
  this proposal — see `forensic-log-sweep.json` baseline).
- `src/fomo/urls.py:30` — FOMO mounts its own `solsys_code.calendar_urls` at `/calendar/`,
  shadowing tomtoolkit 3.0's `tom_calendar` routes (DISPLAY-09).
- `tomtoolkit-3.0.1.dist-info/RECORD` — 38 `tom_calendar/` entries: **`tom_calendar` ships
  inside tomtoolkit itself** (TOM-org maintained). Answers the second open question in
  `.planning/research/questions.md`.

| Approach | Pros | Cons | Status |
|---|---|---|---|
| Reuse `insert_or_create_calendar_event` + `record_time_window` (chosen) | Same no-churn contract and stage rule the existing commands use; ~150 lines total | Inherits `record_time_window`'s KeyError on records lacking `parameters['start'/'end']` | Built |
| Fresh projector with its own compare/save loop | No coupling to sync_lco internals | Second copy of the no-churn contract to keep in sync | Rejected |

## How to Run

```
# persists 146 events (run twice internally; verifies points 1-3)
python manage.py shell -c "import runpy; runpy.run_path('.planning/spikes/002-observation-projector/sweep.py', run_name='__main__')"
# point 4, rolled back
python manage.py shell -c "import runpy; runpy.run_path('.planning/spikes/002-observation-projector/signal_demo.py', run_name='__main__')"
# look at them: python manage.py runserver, then http://127.0.0.1:8000/calendar/ (Aug/Sep 2026)
# remove exactly the spike's rows
python manage.py shell -c "import runpy; runpy.run_path('.planning/spikes/002-observation-projector/cleanup.py', run_name='__main__')"
```

## What to Expect

Sweep: `run 1: {'created': 146}`, `run 2: {'unchanged': 146}`, `stages: {'placed': 18, 'queued': 56,
'observed': 56, 'terminal-negative': 16}`, `stage-correct spans: 146/146`, `terminal marked: 16`, all 9
groups fully series-titled, `RUN: namespace untouched: True (74 -> 74)`. Signal demo: the chosen
queued record's event moves from an ~8.5 h `[QUEUED]` window to a 19-minute `[SCHEDULED]` block on a
schedule-only save, to a clean title on `COMPLETED`, and back after rollback.

## Observability

- `forensic-log-sweep.json` — baseline counts, both runs' counters and timings, per-record
  `(observation_id, status, stage, action)` rows, the point-1/2/3 checks, namespace isolation.
- `forensic-log-signal.json` — the four snapshots of the demo record's event (title, span,
  modified) and the derived booleans.

## Investigation Trail

1. Surveyed the live `CalendarEvent` table first: 94 rows — 74 `RUN:` (campaign reconciler), 10
   portal-URL rows from the old LCO sync (other requests), 9 blank, 1 `LEGACY`; none for this
   proposal. Chose the portal URL as the key so the old sync's rows and the projector's rows are
   one namespace, and `RUN:` is never touched.
2. Built `projector.py`: `stage_for` (queued / placed / observed / completed-no-block /
   terminal-negative / inconsistent, from fields only), `series_for` (group members numbered by
   window start), `event_fields_for`, `project_record` (never raises), `project_queryset`,
   and a `post_save` receiver with `connect()/disconnect()`.
3. Sweep run 1: 146 created, 0.29 s including series computation; run 2: 146 unchanged, 0.12 s.
4. Verified spans against the records directly (not against the projector's own output):
   146/146. Verified every terminal-negative title starts with a bracketed prefix: 16/16.
5. Verified every member of every group carries `· {group name} i/n`: 9/9 groups fully titled.
   The 11P group turns out to have 28 members, not the 14 visible in the truncated
   `reqgroup_2682493.json` — the JSON dump was cut off at 2000 lines.
6. Signal demo on record 4378323 (first queued member of the 11P group): schedule-only save ->
   `updated (placed)`, title `[QUEUED]` -> `[SCHEDULED]`, span 18:07-02:37 -> 23:40-23:59;
   status save -> `updated (observed)`, clean title; TOM's own hook also logged the status
   change alongside, confirming the two mechanisms coexist; rollback restored the sweep state.
7. Added `[SCHEDULED]` as a new prefix for placed-but-not-yet-observed blocks — the existing
   vocabulary had nothing between `[QUEUED]` and clean. A spike choice, flagged for the status-
   vocabulary phase.

## Results

**Verdict: VALIDATED.** All four points hold on real data; evidence in the two forensic logs.

What the spike says about the design:

- **Point 1 (spans, marking):** `record_time_window` already *is* the stage rule; the projector
  adds classification and a prefix. The 16 terminal-negative nights render as their request
  window with `[EXPIRED]`/`[CANCELLED]`/`[FAILED]` — "marked, not dropped" costs nothing.
- **Point 2 (series identity):** works as a title suffix + description line, but it is a
  stopgap: `tom_calendar.CalendarEvent` has no group field, and FOMO's `CalendarEventMeta`
  only has `run`. The real build needs a proper carrier (an `observation_group` FK on
  `CalendarEventMeta`, or an `observation_record` FK from which the group is derived). Also:
  the backfill's group names already end in ` (portal id)`, so titles currently show the id
  twice (`… (2682493) 1/28`) — pick one.
- **Point 3 (idempotency):** inherited from `insert_or_create_calendar_event`; a full sweep of
  146 records costs ~0.1-0.3 s, so per-save projection is cheap enough to run inline.
- **Point 4 (trigger):** the `post_save` receiver re-projects on the placement save that
  TOM's hook cannot see (001a), and the projector's "never raises" contract keeps a bad
  record from breaking the save that triggered it.
- **Single writer held:** 0 `CalendarEventMeta` rows created (absent meta == not owned by any
  run, D2's ownership rule verbatim), `RUN:` rows untouched. Landmine for the build: the
  campaign reconciler's *adopt* path re-keys non-`RUN:` events into `RUN:` — under D2 that
  path must be inverted (annotate, don't adopt) or it will steal these events the first time
  a linked run reconciles.

Surprises: titles are long (target + aperture + instrument + series ≈ 70-90 chars) and will be
truncated in month cells — the calendar UI needs a compact form; group sizes are larger than
assumed (28/30/28/16 for the September cadences); a `completed-no-block` stage exists in the
classifier but no record hit it, so "COMPLETED with no block" is either impossible for OCS or
just absent from this proposal.

Unresolved question 2 answered: `tom_calendar` is part of tomtoolkit 3.0.1 (TOM-org
maintained) — the natural upstream home for a projector is tomtoolkit itself, not a
third-party package.
