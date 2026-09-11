---
status: testing
phase: 34-the-observation-projector-trigger
source: [34-VERIFICATION.md]
started: 2026-09-11T04:44:59Z
updated: 2026-09-11T05:38:38Z
---

## Current Test

number: 1
name: Calendar month view — marker legend, status rings, month-cell titles, series block
expected: |
  Open the calendar month view in a browser (`python manage.py runserver`, then the calendar
  page) on a month containing KEY2026B-004 entries. The legend lists [Q] Queued, [S] Scheduled,
  [O] Observed, [X] Window expired, [C] Cancelled, [F] Failed, [?] Inconsistent record. [Q] chips
  carry the dark queued ring and [X]/[C]/[F]/[?] the red terminal ring, while [S]/[O] carry none.
  Each month-cell title reads legibly within its truncation budget — the marker and telescope
  token are both visible. Opening the modal for a record that belongs to an ObservationGroup
  shows an "Observation series" block with the group name, "Night n of N", and working links,
  rendered beside (not overwriting) the campaign block.
awaiting: user response

## Tests

### 1. Calendar month view — marker legend, status rings, month-cell titles, series block
expected: See Current Test above. Visual appearance, ring contrast against real chip colours and
month-cell legibility are judgment calls; 296 automated tests confirm the markup is produced but
cannot confirm it reads well.
result: [pending]

### 2. Two interleaved saves of the same LCO ObservationRecord leave the event matching the final persisted state
expected: Drive two concurrent/interleaved saves of one LCO ObservationRecord (e.g. two
`updatestatus` runs overlapping, or two request threads saving the same record). The surviving
CalendarEvent's span and title match the record's final persisted `scheduled_start` /
`scheduled_end` / `status` — no event describes a superseded intermediate state. (Declared
`verification: backstop` in 34-01; no automated test exercises concurrency.)
result: [pending]

### 3. A sweep interrupted partway leaves correct events and the re-run converges with no repair
expected: Interrupt `python manage.py project_observation_calendar` partway (Ctrl-C mid-run)
against the developer database, then re-run it to completion. Every record processed before the
interrupt still carries a correct event; the re-run reports `created: 0, updated: 0` for the
already-processed records with no cleanup step. (Declared `verification: backstop` in 34-02.)
result: [pending]

### 4. SCHED-06 — a pending KEY2026B-004 record narrows over real nights with nobody running anything
expected: From the baseline below, run ONLY `python manage.py updatestatus` over several real
observing nights — never the sweep. Then re-execute
`docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` end to end and diff its
SCHED-06 section against `project_observation_calendar_demo.sched06-baseline.json`. Read the
re-execution's OWN first sweep summary line before anything else: if it reports
`created: 0, updated: 0` for the narrowed records, the `post_save` receiver — not the sweep —
did the narrowing (the notebook runs two real sweeps before it re-captures the baseline, so the
snapshot alone cannot prove which writer narrowed the events). At least one record has moved
queued → placed (or placed → observed) with its event span/title following. Fill in the dated
re-check row below and flip the verdict from PARTIAL. This is a verification-over-time item and
is expected to stay pending until real observing nights have elapsed.
result: [pending]

## SCHED-06: live narrowing over real observing nights

Spike 004 left SCHED-06 as a PARTIAL verdict -- the projector and sweep exist and are
tested, but nothing had yet proven a real, pending `KEY2026B-004` record narrow on the
calendar purely from `python manage.py updatestatus`, with no sweep run in between.
Plan 34-04 Task 1 captured the baseline this section tracks; the verdict closes only
when the re-check below is filled in.

### Baseline

- **Captured at:** 2026-09-11T04:44:59.526430+00:00 (UTC)
- **Proposal:** `KEY2026B-004`
- **Pending record count at baseline:** 74 (56 `queued`, 18 `placed`)
- **Baseline artifact:** `docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json`
  (per-record `observation_id`, target name, status, `scheduled_start`/`scheduled_end`,
  and the projected event's start/end/title, all as of the baseline capture)
- **Notebook cell:** `project_observation_calendar_demo.ipynb`, "SCHED-06 baseline: the
  pending `KEY2026B-004` records (D-20)"

### The rule between baseline and re-check

Over the coming nights, run **only**:

```console
$ python manage.py updatestatus
```

Nothing else -- and specifically **not** `python manage.py project_observation_calendar`
(the sweep). Running the sweep in between would let a bulk write path narrow the
records instead of the `post_save` receiver alone, which is the opposite of what
SCHED-06 needs proven. If the sweep is run for any other operational reason before the
re-check below happens, note it in the table and treat the verdict as still open rather
than closed by a mixed cause.

### Dated re-check table

Fill in one row per re-check. A re-check re-executes
`project_observation_calendar_demo.ipynb`
(`jupyter nbconvert --to notebook --execute --inplace
docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`) and diffs its
SCHED-06 section against the baseline JSON above -- committing the re-executed notebook
each time. Record the re-execution's first sweep summary line in the Notes column.

| Date | Nights watched (updatestatus only?) | Records narrowed queued->placed | Records narrowed placed->observed | Notes | Verdict |
|------|--------------------------------------|----------------------------------|-------------------------------------|-------|---------|
| _(pending)_ | | | | | PARTIAL (SCHED-06 still open) |

**Current verdict: PARTIAL.** SCHED-06 closes only once a re-check row above shows at
least one record narrowing with nothing but `updatestatus` run in between.

## Summary

total: 4
passed: 0
issues: 0
pending: 4
skipped: 0
blocked: 0

## Gaps
