---
phase: 04-lco-queue-sync-command
reviewed: 2026-06-17T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - solsys_code/management/commands/sync_lco_observation_calendar.py
  - solsys_code/tests/test_sync_lco_observation_calendar.py
findings:
  critical: 1
  warning: 2
  info: 2
  total: 5
status: issues_found
---

# Phase 04: Code Review Report

**Reviewed:** 2026-06-17T00:00:00Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Reviewed `sync_lco_observation_calendar.py` (the management command) and its test
suite. The happy-path logic (proposal filtering, URL-keyed get-or-create, no-churn
update detection, terminal-failure title prefixes) is well-tested and matches the
documented design (SYNC-01..05, TERM-01, D-01..D-06). However, the command's only
error handling is a `try/except (KeyError, ValueError)` wrapped around
`_build_event_fields`, and this does not cover all the ways a real
`ObservationRecord` can produce an invalid `CalendarEvent`. In particular, a record
with an inconsistent `scheduled_start`/`scheduled_end` pairing will raise an
unhandled `django.db.utils.IntegrityError` from `event.save()`/`get_or_create`,
crashing the whole command run instead of skipping just that record — verified by
checking `tom_observations.models.ObservationRecord`, where `scheduled_start` and
`scheduled_end` are independently nullable (`null=True` each), and
`tom_calendar.models.CalendarEvent.end_time`/`start_time` are non-nullable
`DateTimeField`s. There is also a discrepancy between a code comment's claim and
actual behavior in `_FAILURE_PREFIX_BY_STATUS` (see WR-01).

## Critical Issues

### CR-01: Mismatched scheduled_start/scheduled_end crashes the whole sync run

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:100-108`
**Issue:** `_time_window` branches only on `record.scheduled_start is None`. If a
record has `scheduled_start` set but `scheduled_end` is `None` (a state the DB
schema explicitly permits — `tom_observations.models.ObservationRecord` declares
both fields independently as `models.DateTimeField(null=True)`, so nothing
prevents a partially-updated record), `_time_window` returns `(scheduled_start,
None)`. This `None` then flows into `CalendarEvent(end_time=None)` via
`_build_event_fields` / `handle()`. `CalendarEvent.end_time` is declared as a
non-nullable `models.DateTimeField()` (no `null=True`) in
`tom_calendar/models.py`, so `event.save()` (in the "changed" branch) or the
`get_or_create(... defaults=fields)` call raises `django.db.utils.IntegrityError`.

The `handle()` loop only catches `(KeyError, ValueError)` around
`_build_event_fields(record, facility)` (lines 186-191); it does **not** wrap the
subsequent `CalendarEvent.objects.get_or_create(...)` / `event.save()` calls
(lines 193-205). An `IntegrityError` here is therefore unhandled and propagates out
of `handle()`, aborting the entire command — every other matching
`ObservationRecord` in the same `--proposal` run is left unsynced, not just the
bad record. This breaks the per-record isolation that the rest of the command
(and its tests, e.g. `test_skip_path_missing_site_logged_and_skipped`) is designed
around.

The same asymmetric-null gap also exists for the inverse case
(`scheduled_start=None`, `scheduled_end` set): the code only checks
`scheduled_start is None` and falls into the "unscheduled" branch, silently
discarding the populated `scheduled_end` and instead reading
`parameters['start']`/`parameters['end']` — which may not exist or may disagree
with the partially-scheduled state, producing a `KeyError`/`ValueError` (caught,
but the resulting skip is for the wrong reason) or simply incorrect calendar
times.

**Fix:** Validate both fields together and raise a caught exception type (or treat
inconsistent state explicitly) instead of letting a partial null pass through to
the model layer:
```python
def _time_window(record: ObservationRecord) -> tuple[datetime, datetime]:
    if record.scheduled_start is not None and record.scheduled_end is not None:
        return record.scheduled_start, record.scheduled_end
    if record.scheduled_start is None and record.scheduled_end is None:
        start_time = datetime.fromisoformat(record.parameters['start']).replace(tzinfo=dt_timezone.utc)
        end_time = datetime.fromisoformat(record.parameters['end']).replace(tzinfo=dt_timezone.utc)
        return start_time, end_time
    # Inconsistent state: one of scheduled_start/scheduled_end is set, the other isn't.
    raise ValueError(
        f'Inconsistent schedule state: scheduled_start={record.scheduled_start!r}, '
        f'scheduled_end={record.scheduled_end!r}'
    )
```
This keeps the failure inside the existing `except (KeyError, ValueError)` block in
`handle()`, so the record is skipped and reported rather than crashing the run.
Additionally, consider wrapping the `get_or_create`/`save()` calls themselves in a
`try/except IntegrityError` (or `django.db.transaction.atomic()` with a narrower
catch) as defense-in-depth against any other unanticipated constraint violation,
since `_build_event_fields` cannot enumerate every possible model-level
constraint.

## Warnings

### WR-01: `_FAILURE_PREFIX_BY_STATUS` comment claims auto-sync with the library that the code doesn't actually provide

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:20-32`, `35-48`
**Issue:** The comment above `_FAILURE_PREFIX_BY_STATUS` states it is "Built from
the library's own failure-state list (not re-typed) so a future library update is
picked up automatically." This is not what the code does: the dict is a hardcoded
literal with the 4 current `OCSSettings.get_failed_observing_states()` values
re-typed as keys (verified against
`tom_observations/facilities/ocs.py:118-119`). `_failure_prefix` checks membership
in the *live* `facility.get_failed_observing_states()` list, but then indexes into
the *static* local dict (`return _FAILURE_PREFIX_BY_STATUS[status]`). If the
upstream library ever adds a new failure state (or renames one of the existing
four), `status not in set(facility.get_failed_observing_states())` will be
`False` for the new status, but `_FAILURE_PREFIX_BY_STATUS[status]` will raise
`KeyError` for a status that isn't in the local dict. That `KeyError` is caught by
`handle()`'s broad `except (KeyError, ValueError)`, so the net effect of a library
update is the *opposite* of "picked up automatically" — newly-added failure-state
records are silently skipped (reported as `skipped`, with an opaque error message
that is just the missing status string) rather than synced with any prefix at all.
**Fix:** Either derive a generic fallback prefix for any failure state not in the
explicit map (so unknown failure states still sync, just without a specific
prefix), or correct the comment to state that the map must be updated manually
when the library's failure-state list changes:
```python
def _failure_prefix(status: str, facility: LCOFacility) -> str | None:
    if status not in set(facility.get_failed_observing_states()):
        return None
    return _FAILURE_PREFIX_BY_STATUS.get(status, '[FAILED]')
```

### WR-02: Unmapped LCO site code produces an unhelpful skip with a cryptic error message

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:51-63`, `127`
**Issue:** `SITE_TELESCOPE_MAP` only contains `coj` and `ogg` (acknowledged in the
module docstring as `[ASSUMED]`/unconfirmed). `_derive_telescope` raises a bare
`KeyError` for any other valid LCO site code (e.g. `lsc`, `cpt`, `elp`, `tfn`,
`sqa`, `tlv`). This is caught by `handle()`'s except block, but the resulting
stderr message is just `Skipping observation_id='X': 'tfn'` — there is no
indication to the operator that the cause is an unmapped site code requiring a
`SITE_TELESCOPE_MAP` update, as opposed to some other data problem. For a proposal
that schedules across more than the two currently-mapped sites, this will produce
confusing skip output that doesn't point at the fix.
**Fix:** Raise (or log) a more descriptive error from `_derive_telescope`:
```python
def _derive_telescope(site_code: str) -> str:
    try:
        return SITE_TELESCOPE_MAP[site_code]
    except KeyError:
        raise KeyError(f'Unmapped LCO site code {site_code!r}; add it to SITE_TELESCOPE_MAP') from None
```

## Info

### IN-01: Missing-key skip path is tested only for `site`, not for `instrument_type`/`proposal`/`start`/`end`

**File:** `solsys_code/tests/test_sync_lco_observation_calendar.py:314-330`
**Issue:** `test_skip_path_missing_site_logged_and_skipped` is the only test that
exercises the `except (KeyError, ValueError)` skip path in `handle()`, and it only
covers a missing `parameters['site']`. The same code path is reached if
`instrument_type`, `proposal`, `start`, or `end` are missing from `parameters`
(lines 127-129 in the command), or if `start`/`end` are present but not
ISO-parseable (`ValueError` from `datetime.fromisoformat`). None of these are
exercised, so a regression in `_build_event_fields`'s key access (e.g. a typo'd
key name) for any field other than `site` would not be caught by the test suite.
**Fix:** Add parametrized test cases for missing `instrument_type`/`proposal` and
for malformed `start`/`end` strings, asserting the record is skipped and reported
in stderr (mirroring the existing `site=None` test).

### IN-02: `get_or_create` is not protected against concurrent runs creating duplicate events

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:194`
**Issue:** `CalendarEvent.objects.get_or_create(url=url, defaults=fields)` relies on
`url` being a de facto unique key, but `CalendarEvent.url` (in
`tom_calendar/models.py`) has no `unique=True` constraint and no DB-level unique
index. If this command is ever run concurrently (e.g. overlapping cron
invocations, or a future async/queued trigger), two processes can both miss
finding an existing row for the same `url` and both insert, producing duplicate
`CalendarEvent` rows for the same observation. This isn't a crash risk (no
`IntegrityError` would be raised, since there's no unique constraint to violate),
but it silently breaks the SYNC-04 "no-churn idempotency" guarantee under
concurrent execution.
**Fix:** Wrap the per-record get-or-create/update in `transaction.atomic()`, and/or
document that this command must not be run concurrently for the same proposal
(e.g. via a lock file or `django-cron`'s overlap protection) until/unless a unique
constraint is added upstream to `CalendarEvent.url`.

---

_Reviewed: 2026-06-17T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
