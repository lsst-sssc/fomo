---
status: testing
phase: 34-the-observation-projector-trigger
source: [34-VERIFICATION.md]
started: 2026-09-11T04:44:59Z
updated: 2026-09-12T00:06:11Z
---

## Current Test

number: 2
name: Two interleaved saves of the same LCO ObservationRecord leave the event matching the final persisted state (re-run after G-34-2 closure)
expected: |
  Drive two overlapping `python manage.py updatestatus` runs against a COPY of the developer
  database (`FOMO_DATABASE_PATH=<absolute scratch path>`) and inspect the surviving
  CalendarEvents: each event matches its record's final persisted scheduled_start /
  scheduled_end / status, and the logs carry no `unprojectable ... AttributeError` lines.
awaiting: user response

[Test 4 (SCHED-06) is unblocked but verification-over-time — it stays pending until real
observing nights have elapsed; see its entry and the re-check table below.]

## Tests

### 1. Calendar month view — marker legend, status rings, month-cell titles, series block
expected: See Current Test above. Visual appearance, ring contrast against real chip colours and
month-cell legibility are judgment calls; 296 automated tests confirm the markup is produced but
cannot confirm it reads well.
result: pass

### 2. Two interleaved saves of the same LCO ObservationRecord leave the event matching the final persisted state
expected: Drive two concurrent/interleaved saves of one LCO ObservationRecord (e.g. two
`updatestatus` runs overlapping, or two request threads saving the same record). The surviving
CalendarEvent's span and title match the record's final persisted `scheduled_start` /
`scheduled_end` / `status` — no event describes a superseded intermediate state. (Declared
`verification: backstop` in 34-01; no automated test exercises concurrency.)
result: pending
rerun_required: "2026-09-12 re-verification (34-VERIFICATION.md, 12/14): the original run below is unusable as evidence because G-34-2 dominated it. Re-run against a scratch copy (FOMO_DATABASE_PATH) now that the receiver fix has landed; expected outcome unchanged, plus zero `unprojectable ... AttributeError` lines."
original_result: issue
reported: "Lots of output from running those commands; in tmp/project_observation_calendar_dry_run.txt -- two overlapping `updatestatus` runs logged `unprojectable observation_id=... : AttributeError` for nearly every LCO record touched, plus one `unprojectable observation_id='4378041': OperationalError`"
severity: blocker
note: |
  The interleaving itself behaved as designed: the one OperationalError is SQLite's write lock
  from the deliberate double run, project_record()'s savepoint absorbed it, and 4378041's event
  still matches its record (sweep dry run: unchanged). The AttributeError is a separate,
  deterministic receiver failure on the real `updatestatus` path -- see gap G-34-2.

- **2026-09-11:** G-34-2 closed (34-05 receiver fix, 34-06 live proof, 34-07 paired-docs
  closure) -- a real, portal-backed `updatestatus` run against a scratch copy of the
  developer database (`tmp/34-06-updatestatus.txt`) logged zero `unprojectable` lines. See
  `34-VERIFICATION.md`'s re-verification (2026-09-11T23:05:00Z) and the `closed_by`/
  `closed_verified` keys added to the G-34-2 gap entry below.

### 3. A sweep interrupted partway leaves correct events and the re-run converges with no repair
expected: Interrupt `python manage.py project_observation_calendar` partway (Ctrl-C mid-run)
against the developer database, then re-run it to completion. Every record processed before the
interrupt still carries a correct event; the re-run reports `created: 0, updated: 0` for the
already-processed records with no cleanup step. (Declared `verification: backstop` in 34-02.)
result: pass
note: |
  Run against a copy (FOMO_DATABASE_PATH=/tmp/fomo_uat_copy.sqlite3) so the real DB stayed
  untouched for SCHED-06. Real DB had 33 stale events (G-34-2); interrupted run repaired 1,
  re-run reported `created: 0, updated: 32, unchanged: 127, unprojectable: 0`
  (tmp/project_observation_calendar_rerun.txt), third dry run on the copy reported
  `updated: 0, unchanged: 159`. One `site_lookup_failed` (4276100) used the documented
  fallback label.

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
result: pending
reason: "verification-over-time: G-34-2 (the receiver failing on real `updatestatus` saves) is closed as of 2026-09-11 (34-05/34-06/34-07), so nothing blocks narrowing any more -- only real observing nights need to elapse since the 2026-09-11T04:44Z baseline."
standing_evidence: "The developer database still holds the 33 stale LCO events the broken pre-34-05 receiver left behind (the 34-07 scratch-clone run re-titled exactly those 33). The same 33 repairing themselves under `updatestatus` ALONE -- the re-execution's own first sweep line reading `created: 0, updated: 0` -- is the cleanest possible demonstration. Note an un-routed re-execution rewrites the baseline JSON in place by design (that is what `git diff` reads); never run `project_observation_calendar` against src/fomo_db.sqlite3 before then."

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
passed: 2
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps

- gap_id: G-34-3
  truth: "The committed project_observation_calendar_demo.ipynb shows a takeover that actually took something over (non-zero re-titled count, differing first/second sweep lines), and a repo-level test fails when that evidence goes empty"
  status: closed
  closed_by: ["34-07"]
  closed_verified: "34-VERIFICATION.md re-verification (2026-09-12T00:20:00Z): cell 05528b38 reports 33 of 159 events re-titled with 8 listed before -> after pairs; cell 556d2a9f shows first sweep updated: 33 vs second updated: 0; solsys_code/tests/test_projector_demo_notebook.py mutation-tested against both an emptied copy and the actual regressed notebook from 46d8390 -- both fail on the intended assertion."
  reason: "34-06's re-execution ran against a scratch copy already swept to convergence, so the committed takeover evidence read 0 re-titled (found by the 2026-09-11T23:05:00Z verification)"
  severity: major
  test: n/a

- gap_id: G-34-2
  truth: "Every real `updatestatus` save projects the record's event through the post_save receiver, so the surviving CalendarEvent matches the record's final persisted scheduled_start / scheduled_end / status"
  status: closed
  closed_by: ["34-05", "34-06", "34-07"]
  closed_verified: "34-VERIFICATION.md re-verification (2026-09-11T23:05:00Z): 'the post_save receiver no longer raises AttributeError on the real updatestatus path. calendar_utils.coerce_schedule_datetime() converts the portal's ISO strings to aware UTC datetimes; record_time_window() routes BOTH branches through it. Proven live: tmp/34-06-updatestatus.txt (a real portal-backed updatestatus run against a scratch copy) contains ZERO unprojectable lines, and the dry-run sweep that follows it (tmp/34-06-dry-run.txt) reports LCO: created: 0, updated: 0, unchanged: 159 -- the receiver, not the sweep, did all the work.'"
  reason: "User reported: two overlapping `updatestatus` runs logged `unprojectable observation_id=... : AttributeError` for nearly every LCO record; a sweep dry run afterwards shows 33 LCO events stale (`updated: 33, unchanged: 126`) because the receiver never wrote them"
  severity: blocker
  test: 2
  root_cause: "`OCSFacility.get_observation_status()` (tom_observations/facilities/ocs.py:1570, tomtoolkit 3.0.1) returns the portal's raw ISO strings for scheduled_start/scheduled_end; `BaseObservationFacility.update_observation_status()` (facility.py:563) assigns them straight onto the record and calls save(). Django persists the strings fine, but the post_save receiver runs on that same in-memory instance, so `record_time_window()` (calendar_utils.py:458) hands `str` values back and `event_fields_for()` crashes at observation_projector.py:291 with `'str' object has no attribute 'strftime'`. project_record() catches it and logs `unprojectable`, so the save succeeds and the event silently stays stale. The sweep re-fetches records from the DB (datetime fields) and so never hits it -- which is why `project_observation_calendar` and all 296 tests pass; `test_updatestatus_narrows_the_event_with_no_command_run` mocks get_observation_status with datetime objects and so misses the real contract. This also blocks Test 4 (SCHED-06): with the receiver failing on every real updatestatus save, nothing can narrow without the sweep."
  artifacts:
    - path: "solsys_code/observation_projector.py"
      issue: "event_fields_for() / stage_for() assume scheduled_start/scheduled_end are datetimes; on the post_save path after updatestatus they are ISO strings"
    - path: "solsys_code/calendar_utils.py"
      issue: "record_time_window() returns record.scheduled_start/end unchanged, so a str instance value passes through"
    - path: "solsys_code/tests/test_observation_projector_signals.py"
      issue: "test_updatestatus_narrows_the_event_with_no_command_run fakes get_observation_status with datetime values, not the portal's ISO strings"
  missing:
    - "Coerce scheduled_start/scheduled_end to aware datetimes on the projector path (e.g. in record_time_window() or event_fields_for(): accept str via datetime.fromisoformat, treat 'Z' / naive as UTC), so a post_save instance holding portal strings projects the same as a DB-fetched one"
    - "Make the updatestatus signals test feed ISO strings exactly as OCSFacility.get_observation_status returns them, so the test exercises the real contract"
    - "Paired docs per CLAUDE.md: project_observation_calendar_demo.ipynb re-executed after the fix; the 33 stale events should then be repaired by the next `updatestatus` run alone (each run re-saves every record), which is itself the SCHED-06 evidence Test 4 needs -- do NOT run the real sweep to repair them"
  debug_session: ".planning/debug/34-updatestatus-receiver-attributeerror.md"
