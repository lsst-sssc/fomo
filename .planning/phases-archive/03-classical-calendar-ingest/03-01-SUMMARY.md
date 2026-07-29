---
phase: 03-classical-calendar-ingest
plan: "01"
subsystem: telescope-runs-calendar
tags: [management-command, calendar-ingest, idempotent-upsert, tdd]
dependency_graph:
  requires:
    - solsys_code/telescope_runs.py (parse_run_line, ParsedRun, get_site, sun_event)
    - tom_calendar.models.CalendarEvent
  provides:
    - solsys_code/management/commands/load_telescope_runs.py
    - solsys_code/tests/test_load_telescope_runs.py
  affects:
    - CalendarEvent rows in the database (created/updated by command)
tech_stack:
  added: []
  patterns:
    - Django BaseCommand with positional file-path argument
    - get_or_create + conditional save (idempotent upsert without modified-timestamp churn)
    - timedelta-based date arithmetic for night-range iteration (month/year safe)
    - astropy Time.to_datetime(timezone=dt_timezone.utc) for CalendarEvent DateTimeField
key_files:
  created:
    - solsys_code/management/commands/load_telescope_runs.py
    - solsys_code/tests/test_load_telescope_runs.py
  modified: []
decisions:
  - "Caught Observatory.DoesNotExist alongside ValueError per D-02's spirit (RESEARCH A1) — both treated as per-line data/setup issues; log+skip rather than abort"
  - "Cross-month runs (day2 < day1) raise ValueError in _iter_run_nights and are reported+skipped — fails loudly per RESEARCH Open Question 1 recommendation"
  - "description format: Dark window ISO datetimes first, then Status, then Source line (D-06 order)"
metrics:
  duration_seconds: 436
  completed_date: "2026-06-16"
  tasks_completed: 2
  files_changed: 2
---

# Phase 03 Plan 01: load_telescope_runs Command Summary

**One-liner:** `load_telescope_runs` management command upserts idempotent `CalendarEvent`s from classical-schedule run lines via `parse_run_line`/`sun_event`, one event per night keyed on `(telescope, instrument, start_time)`.

## What Was Built

### `solsys_code/management/commands/load_telescope_runs.py`

A Django `BaseCommand` that:

1. Accepts a required positional `filepath` argument (a text file of classical run lines).
2. Reads lines with `enumerate(f, start=1)` for human-readable line numbers in error messages.
3. Skips blank lines before processing so they are not counted in totals.
4. Per non-blank line:
   - Calls `parse_run_line(line)` to get a `ParsedRun`.
   - Calls `get_site(parsed.telescope)` to fetch the `Observatory`.
   - Calls `_iter_run_nights(parsed)` to get `E - S + 1` evening dates.
   - For each evening date: calls `sun_event(site, d, 'sun')` for `start_time`/`end_time` and `sun_event(site, d, 'dark')` for dark-window times.
   - Converts all `astropy.time.Time` objects to aware UTC `datetime`s via `.to_datetime(timezone=dt_timezone.utc)`.
   - Builds `title = f'{telescope} {instrument}'` (D-05) and `description` containing dark-window times, status, and source line (D-06).
   - Upserts via `CalendarEvent.objects.get_or_create(telescope=, instrument=, start_time=, defaults={...})`.
   - On the "found" path: compares `end_time`, `title`, `description` using direct equality (not truthiness, per Pitfall 1); only calls `.save()` if any differ (D-04).
   - Catches `(ValueError, Observatory.DoesNotExist)` per line and writes to `self.stderr` with line number and original text (D-02).
5. Writes an end-of-run summary to `self.stdout` reporting `created`/`updated`/`unchanged`/`skipped` counts.

### Module-level helper `_iter_run_nights(parsed: ParsedRun) -> list[date]`

- Guards against cross-month ranges (`day2 < day1` raises `ValueError` — caught per-line by D-02 handler).
- Uses `timedelta` arithmetic from `date(year, month, day1)` for correct month/year rollover.

### `solsys_code/tests/test_load_telescope_runs.py`

Six `django.test.TestCase` tests covering all three INGEST requirements and D-02/D-04:

| Test | Covers |
|------|--------|
| `test_creates_one_event_per_night` | INGEST-01: exactly 5 events for `NTT EFOSC2 allocation 9-13 July` |
| `test_event_durations_within_range` | INGEST-01: `end_time > start_time`, duration 8-15 hours |
| `test_event_fields_set_from_parsed_run` | INGEST-02/D-05/D-06: telescope/instrument/title/description fields |
| `test_idempotent_rerun_no_duplicates` | INGEST-03: second run leaves count at 5 |
| `test_unchanged_rerun_does_not_update_existing_rows` | D-04: no `modified` churn on unchanged re-run; summary reports `updated: 0` |
| `test_unparseable_line_logged_and_skipped` | D-02: ambiguous Magellan line logged with line number, NTT events still created |

`setUpTestData` seeds all 4 `Observatory` rows (268/269/809/E10) verbatim from `test_telescope_runs.py`.

## TDD Execution

- **RED (Task 1, commit `2d80e63`):** Test file written; all 6 tests error with `CommandError: Unknown command: 'load_telescope_runs'`. Confirmed 6 collected, 6 failed.
- **GREEN (Task 2, commit `7134e10`):** Command implemented; all 6 tests pass. Full `./manage.py test solsys_code` runs 95 tests, all OK.

## Deviations from Plan

None — plan executed exactly as written.

The two cross-file ruff warnings in `solsys_code/tests/test_views.py` and `solsys_code/views.py` are pre-existing in files not touched by this plan. They are out-of-scope per the SCOPE BOUNDARY rule and logged below.

## Known Stubs

None. All fields are wired from real `parse_run_line` + `sun_event` computations.

## Threat Flags

No new threat surface beyond what the plan's `<threat_model>` documents. The `load_telescope_runs` command is a CLI-operator tool (same trust level as `./manage.py migrate`). T-03-03 (XSS via description/title) is covered by Django's default auto-escaping; no `|safe` usage introduced.

## Deferred Items

- Pre-existing ruff warnings in `solsys_code/tests/test_views.py` (SIM905) and `solsys_code/views.py` (UP045) — not introduced by this plan, out of scope.

## Self-Check

## Self-Check: PASSED

- `solsys_code/management/commands/load_telescope_runs.py` — FOUND
- `solsys_code/tests/test_load_telescope_runs.py` — FOUND
- `.planning/phases/03-classical-calendar-ingest/03-01-SUMMARY.md` — FOUND
- Commit `2d80e63` (test RED phase) — FOUND
- Commit `7134e10` (feat GREEN phase) — FOUND
