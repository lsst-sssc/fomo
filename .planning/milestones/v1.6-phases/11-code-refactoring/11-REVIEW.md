---
phase: 11-code-refactoring
reviewed: 2026-06-27T00:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - solsys_code/calendar_utils.py
  - solsys_code/management/commands/sync_lco_observation_calendar.py
  - solsys_code/tests/test_sync_lco_observation_calendar.py
  - solsys_code/management/commands/load_telescope_runs.py
  - solsys_code/management/commands/sync_gemini_observation_calendar.py
  - docs/design/telescope_runs_calendar.rst
findings:
  critical: 1
  warning: 2
  info: 2
  total: 5
status: issues_found
---

# Phase 11: Code Review Report

**Reviewed:** 2026-06-27
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Reviewed the Phase 11 refactoring output: the shared `calendar_utils.py` module and the three
management commands that consume it, plus the design document. The refactoring correctly extracted the
`SITE_TELESCOPE_MAP`, instrument-extraction chain, `_coarse_telescope_label`, and
`insert_or_create_calendar_event` into a shared module. The logic itself is sound and the test suite
is comprehensive.

One blocker was found: a narrow but real gap in `_resolve_placement_block`'s exception handling leaves
the entire sync command vulnerable to abort when the LCO Observation Portal API returns a valid 2xx
JSON response whose body is not a list. Two warnings were found: a silent float-truncation in the
Gemini command and a full-model save that risks overwriting user-managed fields in a concurrent edit
scenario.

## Critical Issues

### CR-01: `_resolve_placement_block` raises uncaught `AttributeError`/`TypeError` for non-list 2xx JSON responses, aborting the entire sync run

**File:** `solsys_code/calendar_utils.py:153-174`

**Issue:** The try/except block (lines 153-165) only guards the `make_request` call and
`response.json()` call. The loop that consumes the parsed body is **outside** the try/except:

```python
try:
    response = make_request(...)
    blocks = response.json()                 # <-- inside guard
except (requests.exceptions.RequestException, ImproperCredentialsException,
        forms.ValidationError, ValueError):
    return None

# OUTSIDE the guard:
for block in blocks:                         # line 168 — unguarded
    if block.get('state') == 'COMPLETED':    # line 169 — unguarded
```

`ocs.make_request` (verified against the installed source) raises:
- `ImproperCredentialsException` for HTTP 401-403
- `forms.ValidationError` for HTTP 400
- `requests.exceptions.HTTPError` (a `RequestException`) via `raise_for_status()` for all other
  non-2xx codes

A **2xx response** is returned as-is. If the API ever returns a 200 with a JSON body that is not a
list — for example `{"detail": "Not found."}` for an unknown `observation_id` that happens to return
200, or a paginated wrapper `{"results": [...]}`, or JSON `null` — then:

- `for block in {"detail": "Not found."}` iterates over string keys `["detail"]`.
- `"detail".get('state')` raises `AttributeError` — strings have no `.get()`.
- `for block in None` raises `TypeError: 'NoneType' object is not iterable`.

Neither `AttributeError` nor `TypeError` is caught by `_build_event_fields`'s handler:

```python
except InstrumentExtractionError as exc: ...
except (KeyError, ValueError) as exc: ...
```

The exception propagates into `Command.handle()`, which has no blanket handler, aborting the entire
sync run mid-execution. The docstring's "Never raises" contract is violated.

**Fix:** Move the loop inside the try/except, or add a type guard before the loop:

```python
try:
    response = make_request(...)
    blocks = response.json()
    if not isinstance(blocks, list):
        return None          # non-list body treated as soft failure, per "Never raises" contract
except (requests.exceptions.RequestException, ImproperCredentialsException,
        forms.ValidationError, ValueError):
    return None

for block in blocks:
    ...
```

## Warnings

### WR-01: `int(window_duration)` silently truncates fractional hours, or raises ValueError on decimal strings, silently skipping the record

**File:** `solsys_code/management/commands/sync_gemini_observation_calendar.py:124`

**Issue:**

```python
end_time = start_time + timedelta(hours=int(window_duration))
```

`window_duration` is read from the JSONField `safe_params.get('windowDuration')`. Depending on how
the Gemini facility stores the value:

- Python `float` (e.g., `1.5` stored as JSON `1.5`): `int(1.5)` == `1` — silently drops 30 minutes
  from `end_time` with no warning.
- Python string `'1.5'` (stored as JSON `"1.5"`): `int('1.5')` raises `ValueError`, which is caught
  by the outer `except (KeyError, ValueError)` and causes the record to be silently skipped with only
  `ValueError` (no detail) logged to stderr.

In both cases the failure is silent to the operator. `timedelta` accepts `float` hours natively, so
`float()` handles all four input shapes (integer, float, `'6'`, `'6.5'`) correctly.

**Fix:**

```python
end_time = start_time + timedelta(hours=float(window_duration))
```

### WR-02: `insert_or_create_calendar_event` uses a full `event.save()` on the update path, risking overwrite of user-managed fields

**File:** `solsys_code/calendar_utils.py:323-328`

**Issue:**

```python
if changed:
    for f, v in fields.items():
        setattr(event, f, v)
    event.save()             # full model save, not update_fields
    return event, 'updated'
```

`event` was fetched during `get_or_create` (a SELECT). The subsequent `event.save()` writes back
**all** model fields, including fields not present in `fields` — notably `target_list_id` (a nullable
FK to `TargetList`). `CalendarEvent.target_list` is set by users manually through the calendar UI.

If a user attaches a `target_list` to a `CalendarEvent` between the `get_or_create` and `save()`, the
sync command's save overwrites the user's change with the stale `null` value read at `get_or_create`
time. Management commands can in principle run concurrently with user sessions.

**Fix:** Constrain the save to only the fields being managed:

```python
if changed:
    for f, v in fields.items():
        setattr(event, f, v)
    event.save(update_fields=list(fields.keys()))
    return event, 'updated'
```

## Info

### IN-01: `test_sync_07` side_effect ordering depends on undocumented `ObservationRecord` default ordering

**File:** `solsys_code/tests/test_sync_lco_observation_calendar.py:996-1033`

**Issue:** The test creates two placed records (800104, 800105) and supplies a two-element
`side_effect` list ordered to match the assumption that `ObservationRecord` default ordering is
`'-created'` (most-recently-created first). The comment in the test acknowledges this:

```python
# '-created' default ordering -> 800105 (created last) is processed first.
side_effect=[
    requests.exceptions.Timeout,                                              # for 800105
    _observations_block_response(site='coj', telescope='2m0a', ...),          # for 800104
]
```

`ObservationRecord.Meta.ordering` is defined by TOM Toolkit, not FOMO. If it changes in a future
upgrade, the `Timeout` fires on the wrong record. The overall `CalendarEvent.objects.count() == 2`
assertion still passes, but the per-record title assertions (`first_processed_event.title.startswith(
'[UNVERIFIED]')`) would silently validate the wrong record. The test would stay green while providing
false assurance.

**Fix:** Explicitly control iteration order, e.g. by passing `.order_by('observation_id')` via a
queryset patch, or by asserting both the count AND verifying which specific URL gets the `[UNVERIFIED]`
title by URL rather than by processing-order index.

### IN-02: `test_sync_04` second `call_command` makes an unguarded live network call with a 10-second timeout

**File:** `solsys_code/tests/test_sync_lco_observation_calendar.py:395-415`

**Issue:** The first `call_command` processes both records as banner-stage (no API call). Between the
two calls, `rescheduled.scheduled_start` is set, making record 800001 a placed record. The second
`call_command` has no `patch('solsys_code.calendar_utils.make_request', ...)` context, so
`_resolve_placement_block` executes with the real `make_request`. In an air-gapped CI environment
this either hits an unreachable host or receives a credentials error; either way,
`_resolve_placement_block` falls back to the coarse label after the full `_API_TIMEOUT_SECONDS = 10`
wait. The test passes (the fallback still produces an updated event), but execution takes ~10 seconds
for this one test.

**Fix:** Mock `make_request` for the second `call_command` call, either with a successful block
response or a `Timeout` side_effect (both produce an updated event; the timeout path is faster):

```python
with patch('solsys_code.calendar_utils.make_request',
           side_effect=requests.exceptions.Timeout):
    call_command('sync_lco_observation_calendar', '--proposal', 'MATCHCODE', ...)
```

---

_Reviewed: 2026-06-27_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
