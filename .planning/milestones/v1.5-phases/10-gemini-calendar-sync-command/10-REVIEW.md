---
phase: 10-gemini-calendar-sync-command
reviewed: 2026-06-27T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - solsys_code/management/commands/sync_gemini_observation_calendar.py
  - solsys_code/tests/test_sync_gemini_observation_calendar.py
  - docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb
findings:
  critical: 2
  warning: 4
  info: 2
  total: 8
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-06-27
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Three files were reviewed: the management command
`sync_gemini_observation_calendar.py`, its Django test suite, and the
pre-executed demo notebook. The core logic — no-churn get_or_create,
password stripping at D-04, window derivation branches, and counter
reporting — is structurally sound and the stated happy-path requirements
are met. Two crash paths are present that are not caught by the existing
exception handler and have no test coverage. Four additional issues
degrade correctness or violate the stated security invariant.

---

## Critical Issues

### CR-01: `IndexError` crash on empty `obsid` list escapes the exception handler

**File:** `solsys_code/management/commands/sync_gemini_observation_calendar.py:72`

**Issue:** `obs_code = obsid_list[0]` raises `IndexError` when
`obsid_list` is an empty list. `IndexError` is not included in the
`except (KeyError, ValueError)` clause at line 160, so the exception
propagates out of the for-loop entirely and aborts processing of every
remaining `ObservationRecord`. A single malformed GEM record with
`"obsid": []` silently kills the remainder of the sync run.

There is no test covering the empty-list path; this crash is entirely
undetected by the test suite.

**Fix:**

```python
# After line 65, before the multi-obsid warning block:
if not obsid_list:
    logger.warning(
        'ObservationRecord pk=%s has empty obsid list — skipping',
        record.pk,
    )
    counters[site_key]['skipped'] += 1
    continue
```

Alternatively, add `IndexError` to the except tuple at line 160, though
the explicit guard above produces a more informative log message.

---

### CR-02: `AttributeError` crash when `record.parameters` is `None`

**File:** `solsys_code/management/commands/sync_gemini_observation_calendar.py:47`

**Issue:** `record.parameters.items()` raises `AttributeError` if the
JSONField is stored as SQL `NULL` (Python `None`). Django JSONFields allow
null values by default unless `null=False` is set on the field definition.
An `ObservationRecord` created without explicit parameters, or with a
`None` value, causes the command to crash on its very first record. Like
CR-01, `AttributeError` is not caught anywhere in the loop.

**Fix:**

```python
safe_params = {k: v for k, v in (record.parameters or {}).items() if k != 'password'}
```

---

## Warnings

### WR-01: `ready == 'false'` fails silently when the JSON value is a boolean

**File:** `solsys_code/management/commands/sync_gemini_observation_calendar.py:130`

**Issue:** The on-hold title prefix is applied by `ready == 'false'`
(string comparison). If the Gemini facility code stores `ready` as a
JSON boolean `false` — which deserialises to Python `False`, not the
string `'false'` — the comparison is always `False` and the `[ON_HOLD]`
prefix is silently omitted. An observation that should appear as on-hold
on the calendar instead appears as active. The bug is invisible in tests
because `_gem_parameters` hard-codes `'ready': 'false'` (a string), so
the boolean path is never exercised.

**Fix:**

```python
title_prefix = '[ON_HOLD] ' if str(ready).lower() == 'false' else ''
```

---

### WR-02: `obsid` is not validated as a list before sequence operations

**File:** `solsys_code/management/commands/sync_gemini_observation_calendar.py:65-72`

**Issue:** `obsid_list = safe_params['obsid']` accepts whatever type is
stored in the JSON. Two failure modes follow:

1. If `obsid` is a string (e.g. `"MM"` rather than `["MM"]`):
   `len("MM") == 2 > 1` triggers the multi-obsid warning, and
   `obsid_list[0]` returns `"M"` (the first character). The instrument
   lookup then silently uses `"M"` as the obs code, failing to find it in
   settings and either skipping the record or using the raw one-character
   label as the instrument name. No error is raised.

2. If `obsid` is `None` or an integer: `len(obsid_list)` raises
   `TypeError`, which is not caught by `except (KeyError, ValueError)`,
   crashing the command.

**Fix:**

```python
obsid_list = safe_params['obsid']
if not isinstance(obsid_list, list) or not obsid_list:
    raise ValueError(f'obsid must be a non-empty list, got {type(obsid_list).__name__!r}')
```

This raises `ValueError` (already caught) for type mismatches and merges
the empty-list guard from CR-01 into one place.

---

### WR-03: Exception message indirectly exposes parameter values through `{exc}`, violating the stated security invariant

**File:** `solsys_code/management/commands/sync_gemini_observation_calendar.py:162`

**Issue:** The code comment immediately above this line says: *"Never
interpolate safe_params or record.parameters into this message
(GEM-SECURE-01)"*. However, `self.stderr.write(f'... {exc}')` does
exactly that indirectly: a `ValueError` from `datetime.strptime` embeds
the offending input value in its message string (e.g. `"time data
'2026-99-99' does not match format '%Y-%m-%d'"`). If `windowDate`,
`windowTime`, or `windowDuration` contain user-supplied or non-sanitised
values, those values reach stderr through the exception string. The
password itself is already stripped at D-04, so this is not a password
leak, but it breaks the documented invariant and could expose other
parameter values.

**Fix:**

```python
# Emit only the exception class name on stderr; log the full detail at DEBUG.
self.stderr.write(
    f'Skipping observation_id={record.observation_id!r}: {type(exc).__name__}'
)
logger.debug('Full exception for ObservationRecord %s: %s', record.pk, exc)
```

---

### WR-04: UNKNOWN-prefix skipped records are counted but never reported in the summary

**File:** `solsys_code/management/commands/sync_gemini_observation_calendar.py:59-61` and `167-178`

**Issue:** When a record's program prefix is neither `'GS-'` nor `'GN-'`,
the code does:

```python
counters.setdefault('UNKNOWN', {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0})
counters['UNKNOWN']['skipped'] += 1
continue
```

The `'UNKNOWN'` bucket is incremented, but the final summary (lines
167-178) only prints `counters["GS"]` and `counters["GN"]`. An operator
running the command sees `skipped: 0` for both sites even when
unknown-prefix records were dropped. A stderr warning is emitted per
record (line 58), but the per-run aggregate is lost. This defeats the
purpose of the summary counters for non-standard program IDs.

**Fix:** Print the UNKNOWN counter when non-zero, or fold the count into
an overall summary line:

```python
unknown_skipped = counters.get('UNKNOWN', {}).get('skipped', 0)
if unknown_skipped:
    self.stdout.write(f'Unknown prefix: skipped: {unknown_skipped}')
```

---

## Info

### IN-01: `setattr` loop sets all fields, but `update_fields` saves only the changed subset

**File:** `solsys_code/management/commands/sync_gemini_observation_calendar.py:153-155`

**Issue:**

```python
for f, v in fields.items():    # sets ALL fields on the Python object
    setattr(event, f, v)
event.save(update_fields=changed)  # writes only CHANGED fields to the DB
```

Unchanged fields are assigned their same value (a no-op in memory) and
then excluded from the SQL UPDATE via `update_fields`. The DB write is
correct and GEM-NOCHURN-01 is satisfied. The issue is readability:
iterating all fields but only saving a subset is misleading to a future
reader. Limiting the `setattr` loop to `changed` makes the intent
explicit.

**Fix:**

```python
for f in changed:
    setattr(event, f, fields[f])
event.save(update_fields=changed)
```

---

### IN-02: No-op `add_arguments` override should be removed

**File:** `solsys_code/management/commands/sync_gemini_observation_calendar.py:21-23`

**Issue:** `add_arguments` is overridden with only `pass`. The
`BaseCommand` base-class implementation is already a no-op, so this
override adds no behaviour and creates dead code. The docstring
`'Parse command line arguments.'` implies intent that is never fulfilled.

**Fix:** Remove the method entirely; `BaseCommand.add_arguments` handles
the no-argument case without an override.

---

_Reviewed: 2026-06-27_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
