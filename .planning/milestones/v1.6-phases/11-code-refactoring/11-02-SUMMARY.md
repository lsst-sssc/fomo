---
phase: 11-code-refactoring
plan: "02"
subsystem: solsys_code/management/commands
tags: [refactor, calendar, no-new-features, behavior-neutral, docs]
dependency_graph:
  requires: [11-01]
  provides: [REFAC-02-complete]
  affects: [load_telescope_runs, sync_gemini_observation_calendar]
tech_stack:
  added: []
  patterns:
    - "Shared insert_or_create_calendar_event helper now used by all three management commands"
key_files:
  created: []
  modified:
    - solsys_code/management/commands/load_telescope_runs.py
    - solsys_code/management/commands/sync_gemini_observation_calendar.py
    - docs/design/telescope_runs_calendar.rst
    - .planning/MILESTONES.md
decisions:
  - "sync_gemini: used _event (throwaway) for the returned CalendarEvent since the event object is not needed by the caller — cleaner than bare event variable"
  - "sync_gemini: counters[site_key][action] += 1 replaces the explicit if/elif branch since action is already one of the three counter keys ('created'/'updated'/'unchanged')"
  - "load_telescope_runs: CalendarEvent import removed (was only used in the now-deleted get_or_create call); import is no longer needed"
metrics:
  duration_minutes: 15
  completed_date: "2026-06-27"
  tasks_completed: 3
  tasks_total: 3
  files_modified: 4
---

# Phase 11 Plan 02: Remaining Consumer Refactor and Upsert Doc Cleanup Summary

Complete REFAC-02 by delegating `load_telescope_runs` and `sync_gemini_observation_calendar` create-or-update logic to the shared `insert_or_create_calendar_event` helper, and replace three "upsert" occurrences in design docs with plain English.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Refactor load_telescope_runs to use insert_or_create_calendar_event | dca3ffc | solsys_code/management/commands/load_telescope_runs.py |
| 2 | Refactor sync_gemini_observation_calendar to use insert_or_create_calendar_event | 209498a | solsys_code/management/commands/sync_gemini_observation_calendar.py |
| 3 | Replace "upsert" with plain English in design doc and MILESTONES | fa82079 | docs/design/telescope_runs_calendar.rst, .planning/MILESTONES.md |

## What Was Done

**Task 1 — load_telescope_runs.py:**
- Replaced the 22-line `CalendarEvent.objects.get_or_create(telescope=..., instrument=..., start_time=...) + if created / else changed / event.save()` block with a 4-line call to `insert_or_create_calendar_event({'telescope': ..., 'instrument': ..., 'start_time': ...}, {'end_time': ..., 'title': ..., 'description': ...})`.
- Mapped the returned `action` string directly to existing counters (`created_count`, `updated_count`, `unchanged_count`).
- Removed the now-unused `from tom_calendar.models import CalendarEvent` import; added `from solsys_code.calendar_utils import insert_or_create_calendar_event`.
- Net: -20 lines, +8 lines.

**Task 2 — sync_gemini_observation_calendar.py:**
- Replaced the 12-line `CalendarEvent.objects.get_or_create(url=url, defaults=fields) + if created_flag / else changed / update_fields` block with 2 lines: the helper call plus `counters[site_key][action] += 1`.
- The `counters[site_key][action]` increment works directly because `action` is already one of `'created'`/`'updated'`/`'unchanged'` — matching the counter dict keys exactly.
- Removed `from tom_calendar.models import CalendarEvent` import; added `from solsys_code.calendar_utils import insert_or_create_calendar_event`.
- The `safe_params` password-strip (GEM-SECURE-01) line was not touched.
- Net: -15 lines, +6 lines.

**Task 3 — docs:**
- `docs/design/telescope_runs_calendar.rst` line 222: `**upserts**` → `**creates or updates**`.
- `docs/design/telescope_runs_calendar.rst` line 254: `upsert CalendarEvent rows` → `create or update CalendarEvent rows`.
- `.planning/MILESTONES.md` line 60: `upsert via` → `creating or updating via`.
- All three replacements preserve surrounding RST/Markdown formatting exactly.

## Verification

- `./manage.py test solsys_code`: 186 tests pass, 0 failures/errors (behavior-neutral confirmed).
- `ruff check` and `ruff format --check`: clean for both modified command files.
- `grep -ic "upsert" docs/design/telescope_runs_calendar.rst .planning/MILESTONES.md`: both return 0.

## REFAC-02 Completion Status

All three commands now delegate their CalendarEvent create-or-update to `insert_or_create_calendar_event`:
- `sync_lco_observation_calendar.py` — delegated in Plan 11-01 (commit 3fb5ad7)
- `load_telescope_runs.py` — delegated in this plan (commit dca3ffc)
- `sync_gemini_observation_calendar.py` — delegated in this plan (commit 209498a)

REFAC-02 is **complete**.

## Deviations from Plan

None — plan executed exactly as written. The `_event` throwaway variable name in Task 2 (vs plain `event`) is a style choice to make the unused CalendarEvent object explicit; it is strictly equivalent and passes ruff.

## Known Stubs

None. This is a pure behavior-neutral refactor; no data sources were added or wired.

## Threat Flags

None. Pure code relocation within existing trust boundaries. No new network endpoints, auth paths, or schema changes.

## Self-Check: PASSED

Files exist:
- FOUND: solsys_code/management/commands/load_telescope_runs.py
- FOUND: solsys_code/management/commands/sync_gemini_observation_calendar.py
- FOUND: docs/design/telescope_runs_calendar.rst
- FOUND: .planning/MILESTONES.md

Commits exist:
- dca3ffc: refactor(11-02): delegate load_telescope_runs create-or-update to insert_or_create_calendar_event
- 209498a: refactor(11-02): delegate sync_gemini create-or-update to insert_or_create_calendar_event
- fa82079: docs(11-02): replace 'upsert' with plain English in design doc and MILESTONES
