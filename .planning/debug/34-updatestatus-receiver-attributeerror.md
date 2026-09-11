---
status: root_cause_found
phase: 34-the-observation-projector-trigger
gap_id: G-34-2
created: 2026-09-11
---

# G-34-2: post_save receiver fails with AttributeError on every real `updatestatus` save

## Symptom

Two overlapping `python manage.py updatestatus` runs (UAT Test 2) logged
`unprojectable observation_id='...': AttributeError` for nearly every LCO record touched
(`tmp/project_observation_calendar_dry_run.txt`). A subsequent
`project_observation_calendar --dry-run` on the real DB reported
`LCO: created: 0, updated: 33, unchanged: 126` -- 33 events the receiver should have narrowed
but silently left stale. The receiver never raises (TRIG-02), so `updatestatus` itself reports
`Update completed successfully`.

## Reproduction (real path, traceback captured)

```python
import traceback, solsys_code.observation_projector as op
_orig = op.logger.warning
def loud(msg, *a, **k):
    _orig(msg, *a, **k); traceback.print_exc()
op.logger.warning = loud
from tom_observations.facility import get_service_class
get_service_class('LCO')().update_observation_status('4378029')
```

```
Traceback (most recent call last):
  File "solsys_code/observation_projector.py", line 368, in project_record
    fields, stage = event_fields_for(record, facility)
  File "solsys_code/observation_projector.py", line 291, in event_fields_for
    f'Window (UTC): {start_time.strftime("%Y-%m-%dT%H:%M:%S")} to {end_time.strftime(...)}'
AttributeError: 'str' object has no attribute 'strftime'
```

Projecting the same record from a shell (`event_fields_for` on a DB-fetched instance)
succeeds: stage=placed, title `[S] 1m0 220P` -- the failure is specific to the in-memory
instance `updatestatus` saves.

## Root cause

- `OCSFacility.get_observation_status()` (tomtoolkit 3.0.1,
  `tom_observations/facilities/ocs.py:1570`) returns the portal's raw ISO strings for
  `scheduled_start` / `scheduled_end` (`current_block['start']`).
- `BaseObservationFacility.update_observation_status()` (`facility.py:563`) assigns them
  straight onto the record and calls `save()`. Django persists the strings without coercing
  the Python attribute, so `post_save` fires with `record.scheduled_start` still a `str`.
- `record_time_window()` (`solsys_code/calendar_utils.py:458`) returns those attributes
  unchanged; `event_fields_for()` (`observation_projector.py:291`) calls `.strftime()` on
  them and crashes. `project_record()` catches it, logs `unprojectable`, and the event stays
  stale.
- The sweep re-fetches records from the DB (real `DateTimeField` values) and so never hits
  this -- which is why `project_observation_calendar` and all 296 tests pass.
  `test_updatestatus_narrows_the_event_with_no_command_run`
  (`solsys_code/tests/test_observation_projector_signals.py`) fakes `get_observation_status`
  with `datetime` objects, not the portal's ISO strings, so it never exercised the real contract.

## Consequence

SCHED-06 (UAT Test 4) cannot be satisfied until this is fixed: with the receiver failing on
every real `updatestatus` save, nothing narrows without the sweep. The 33 stale events are the
natural re-check evidence once fixed -- `updatestatus` re-saves every record each run, so the
next run alone should repair them through the receiver. Do NOT run the real sweep against
`src/fomo_db.sqlite3` to repair them; that would destroy the SCHED-06 evidence.

## Fix direction

1. On the projector path, coerce `scheduled_start` / `scheduled_end` to aware UTC datetimes
   when they arrive as `str` (ISO 8601, possibly with a trailing `Z`) -- in
   `record_time_window()` and/or `event_fields_for()` / `stage_for()`, so a post_save
   instance holding portal strings projects identically to a DB-fetched one.
2. Change the signals test to feed ISO strings exactly as `OCSFacility.get_observation_status`
   returns them (and keep a datetime case), so the test exercises the real contract.
3. CLAUDE.md paired docs: `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`
   re-executed; the sweep behaviour is unchanged so the runbook page likely needs no edit --
   confirm.
