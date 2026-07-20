---
phase: 05-multi-proposal-multi-facility-selection
reviewed: 2026-06-19T17:25:46Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - solsys_code/management/commands/sync_lco_observation_calendar.py
  - solsys_code/tests/test_sync_lco_observation_calendar.py
  - src/fomo/settings.py
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: issues_found
---

# Phase 05: Code Review Report

**Reviewed:** 2026-06-19T17:25:46Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Reviewed the new `sync_lco_observation_calendar` management command, its test suite, and
the `src/fomo/settings.py` diff (addition of a `SOAR` entry to `FACILITIES`). Verified
behavior directly against the installed `tom_observations`/`tom_calendar` library code
(`LCOFacility.get_failed_observing_states()`, `get_observation_url()`, the
`CalendarEvent` model schema) rather than taking the docstrings' claims at face value,
and ran the full Django test suite for this command (19/19 pass) plus `ruff check`
(clean) against the three reviewed files.

No blocking correctness or security defects were found in the reviewed diff. The logic
for proposal-code parsing, status→title-prefix mapping, and the time-window derivation
all check out against the real library behavior. However, there is one real robustness
gap (unhandled `TypeError` if `parameters` is ever `None`) and a latent data-integrity
risk from the `CalendarEvent.url` field lacking a uniqueness constraint that the
command's `get_or_create` dedup strategy implicitly depends on. Both are flagged below.
`src/fomo/settings.py`'s only change in this diff is the new `SOAR` facilities block,
which is correctly scoped per the D-05 decision recorded in the code comment; the
pre-existing `ALLOWED_HOSTS` trailing-space formatting drift in that file is an
uncommitted local edit outside this phase's diff and is not flagged here.

## Warnings

### WR-01: `parameters[...]` access not guarded against `parameters=None`, crashing the whole sync run

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:139-141, 251-256`
**Issue:** `_build_event_fields()` (and the functions it calls) assume `record.parameters`
is always a `dict`-like object and does `record.parameters['site']`,
`record.parameters['instrument_type']`, `record.parameters['proposal']`. The `handle()`
loop only catches `(KeyError, ValueError)` around this call:
```python
try:
    fields = _build_event_fields(record, facility)
except (KeyError, ValueError) as exc:
    self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')
    counters[record.facility]['skipped'] += 1
    continue
```
If `record.parameters` is ever `None` for some row (the DB CHECK constraint on this
column is `JSON_VALID(parameters) OR parameters IS NULL`, so a `NULL` value is not
prevented at the SQL layer even though the Django field declares `null=False`; legacy
rows, direct SQL edits, or a future code path that does `ObservationRecord.objects.create(..., parameters=None)` could produce this), `record.parameters['site']` raises
`TypeError: 'NoneType' object is not subscriptable`, which is **not** caught by
`except (KeyError, ValueError)`. This propagates out of the `for record in records:` loop
and aborts the entire command for every other matching record in the same run, not just
the offending one — defeating the documented "skip-and-log rather than abort the whole
run" intent (D-07) for this particular failure mode.
**Fix:** Broaden the except clause (or normalize `parameters` to `{}` up front):
```python
try:
    fields = _build_event_fields(record, facility)
except (KeyError, ValueError, TypeError) as exc:
    self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')
    counters[record.facility]['skipped'] += 1
    continue
```

### WR-02: `get_or_create(url=...)` dedup relies on a uniqueness guarantee the schema doesn't enforce

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:258-259`
**Issue:** `CalendarEvent.objects.get_or_create(url=url, defaults=fields)` is the
command's entire create-vs-update keying strategy (SYNC-04 no-churn idempotency
depends on it). However `tom_calendar.models.CalendarEvent.url` is declared with
`unique=False` and no DB index (confirmed via the installed migrations:
`URLField(max_length=200, blank=True)`, no `unique=True`, no `db_index=True`). Two
concurrent invocations of this command (e.g. overlapping cron schedules, or one run
triggered manually while a scheduled one is still in flight) racing on the same new
`url` can each see "no existing row" in their own `get_or_create` call and both insert,
producing duplicate `CalendarEvent` rows for the same observation. This is a real,
not-merely-theoretical risk for a command intended to be run repeatedly/periodically
against a live LCO queue.
**Fix:** Either add a migration giving `CalendarEvent.url` a unique constraint (out of
scope for this command's file, but worth flagging upstream), or wrap the
get-or-create-or-update sequence in `transaction.atomic()` with `select_for_update()`,
or rely on `update_or_create()` plus a periodic dedup pass until the schema can be
fixed. At minimum, document this race as a known limitation since the current code
gives no indication the dedup key is unenforced at the DB level.

### WR-03: `SOAR` facilities entry ships a blank `api_key` for the real (non-test) Django settings

**File:** `src/fomo/settings.py:223-226`
**Issue:** The new `'SOAR': {'portal_url': ..., 'api_key': ''}` block is committed into
the project's primary `settings.py` (not a `local_settings.py` override or an
environment-variable read like the `LASAIR`/`FINK` entries elsewhere in this same file).
An empty `api_key` means any SOAR-facing API call made through this facility config will
either fail outright or — depending on how `tom_observations`'s OCS client treats an
empty bearer token — could send unauthenticated requests to the LCO portal. The comment
correctly explains SOAR reuses LCO's portal/credentials (D-05), but it does not explain
why `api_key` is hardcoded empty here instead of also being sourced the same way the
real `LCO.api_key` value above it is (which is itself a live-looking API key checked
into source — a pre-existing issue not introduced by this diff, but the new `SOAR` block
compounds the same anti-pattern by adding a second hardcoded-secret-shaped entry next to
it).
**Fix:** At minimum, source `SOAR.api_key` from the same value as `LCO.api_key` (or an
env var) rather than leaving it blank, so SOAR observation syncs/lookups actually
authenticate:
```python
'SOAR': {
    'portal_url': 'https://observe.lco.global',
    'api_key': os.getenv('LCO_API_KEY', FACILITIES['LCO']['api_key']),
},
```
(and ideally migrate `LCO.api_key` itself off a hardcoded literal at the same time).

## Info

### IN-01: D-07 "unrecognized facility" defensive branch is unreachable given the query filter

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:233, 238-249`
**Issue:** `records = ObservationRecord.objects.filter(facility__in=['LCO', 'SOAR'])`
guarantees every row already has `facility` exactly equal to `'LCO'` or `'SOAR'`
(confirmed: SQLite's `IN` lookup here is an exact case-sensitive string match, not
case-folded). Since `facilities = {'LCO': LCOFacility(), 'SOAR': SOARFacility()}` is
seeded with exactly those two keys, `facilities.get(record.facility)` can never return
`None` for any row that reached this loop — the `if facility is None:` branch (and its
`counters.setdefault(...)` call) is dead code. This isn't wrong, just permanently
unreachable given the current query; harmless as defensive programming but worth a
comment update or removal if it's confirmed to never trigger.
**Fix:** Either leave as intentional defense-in-depth (acceptable, just note in the
comment that it is currently unreachable by construction) or remove it if the query
filter is considered the sole source of truth for valid facility values.

### IN-02: Trailing bare `return` is redundant

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:281`
**Issue:** `return` with no value at the end of `handle()` is a no-op; Python functions
return `None` implicitly at the end of their body regardless. Purely stylistic, no
behavior difference.
**Fix:** Remove the trailing `return` (or replace with `return None` if making the
implicit-`None` explicit is preferred house style — either way, drop the bare `return`).

---

_Reviewed: 2026-06-19T17:25:46Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
