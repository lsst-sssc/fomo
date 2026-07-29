---
phase: 10-gemini-calendar-sync-command
fixed_at: 2026-06-27T00:00:00Z
review_path: .planning/phases/10-gemini-calendar-sync-command/10-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 6
skipped: 0
status: all_fixed
---

# Phase 10: Code Review Fix Report

**Fixed at:** 2026-06-27
**Source review:** .planning/phases/10-gemini-calendar-sync-command/10-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 6 (2 Critical, 4 Warning)
- Fixed: 6
- Skipped: 0

## Fixed Issues

### CR-01: `IndexError` crash on empty `obsid` list escapes the exception handler

**Files modified:** `solsys_code/management/commands/sync_gemini_observation_calendar.py`
**Commit:** b5ade02
**Applied fix:** Added an explicit `if not obsid_list:` guard immediately after
`obsid_list = safe_params['obsid']`. When the list is empty, a `logger.warning`
is emitted with the record pk, the site's skipped counter is incremented, and
processing continues with the next record. This sits before the subsequent
isinstance check (added by WR-02) and the existing `len > 1` warning.

---

### CR-02: `AttributeError` crash when `record.parameters` is `None`

**Files modified:** `solsys_code/management/commands/sync_gemini_observation_calendar.py`
**Commit:** 6a14bdd
**Applied fix:** Changed `record.parameters.items()` to
`(record.parameters or {}).items()` in the D-04 password-stripping line,
so a NULL JSONField produces an empty dict rather than raising `AttributeError`
before any exception handler is in scope.

---

### WR-01: `ready == 'false'` fails silently when the JSON value is a boolean

**Files modified:** `solsys_code/management/commands/sync_gemini_observation_calendar.py`
**Commit:** aeda68b
**Applied fix:** Changed the title-prefix comparison from `ready == 'false'` to
`str(ready).lower() == 'false'`. The `ready` variable is still extracted on the
preceding line; only the comparison is changed. This ensures Python boolean `False`
(deserialised from a JSON `false`) also triggers the `[ON_HOLD]` prefix.
Note: requires human verification that `str(True).lower() != 'false'` is the
intended semantics for `ready=True` (it is, since `'true' != 'false'`).

---

### WR-02: `obsid` is not validated as a list before sequence operations

**Files modified:** `solsys_code/management/commands/sync_gemini_observation_calendar.py`
**Commit:** 1fbd948
**Applied fix:** Added an `if not isinstance(obsid_list, list):` check immediately
after extracting `obsid_list = safe_params['obsid']` and before the empty-list
guard from CR-01. A non-list value writes a diagnostic to `self.stderr` (using
only the type name, not the value, to stay consistent with GEM-SECURE-01),
increments the skipped counter, and continues. This prevents `TypeError` from
`len()` on a bare string or `None` escaping the `KeyError/ValueError` handler.

---

### WR-03: Exception message indirectly exposes parameter values through `{exc}`

**Files modified:** `solsys_code/management/commands/sync_gemini_observation_calendar.py`
**Commit:** 5d8054d
**Applied fix:** Changed the `except (KeyError, ValueError)` handler's stderr
write from `f'... {exc}'` to `f'... {type(exc).__name__}'`. The full exception
detail (including any embedded parameter values from `strptime` error messages)
is preserved at `DEBUG` level via `logger.debug('Full exception for ObservationRecord %s: %s', record.pk, exc)`.
This satisfies the GEM-SECURE-01 invariant stated in the code comment directly
above the except block.

---

### WR-04: UNKNOWN-prefix skipped records are counted but never reported in summary

**Files modified:** `solsys_code/management/commands/sync_gemini_observation_calendar.py`
**Commit:** b1bc798
**Applied fix:** Added two lines after the GN summary write:
```python
unknown_skipped = counters.get('UNKNOWN', {}).get('skipped', 0)
if unknown_skipped:
    self.stdout.write(f'Unknown prefix: skipped: {unknown_skipped}')
```
The conditional keeps the output clean for normal runs (no UNKNOWN line when
zero records were dropped); it only appears when at least one record had an
unrecognised program prefix.

---

_Fixed: 2026-06-27_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
