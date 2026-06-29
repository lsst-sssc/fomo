---
plan: 11-01
phase: 11-code-refactoring
status: complete
completed: 2026-06-27
commits:
  - 686f4d9 feat(11-01): create solsys_code/calendar_utils.py with extracted symbols and shared helper
  - 44cb1ab feat(11-01): refactor sync_lco_observation_calendar to use calendar_utils
  - 3fb5ad7 fix(11-01): use event.save() in insert_or_create_calendar_event
key-files:
  created:
    - solsys_code/calendar_utils.py
  modified:
    - solsys_code/management/commands/sync_lco_observation_calendar.py
    - solsys_code/tests/test_sync_lco_observation_calendar.py
---

## Plan 11-01: Create calendar_utils.py and refactor sync_lco_observation_calendar

### What Was Built

Created `solsys_code/calendar_utils.py` — a new shared module holding:

1. **12 relocated symbols** (verbatim from `sync_lco_observation_calendar.py`): constants
   `SITE_TELESCOPE_MAP`, `_API_TIMEOUT_SECONDS`, `_SCIENCE_CONFIGURATION_TYPES`,
   `_MUSCAT_CHANNEL_SUFFIXES`; exception `InstrumentExtractionError`; and functions
   `_aperture_class_from_telescope_code`, `_derive_telescope`, `_resolve_placement_block`,
   `_coarse_telescope_label`, `_extract_instrument`, `_find_science_config`,
   `_find_exposure_signal_config`, `_has_muscat_exposure_signal`.

2. **New shared helper** `insert_or_create_calendar_event(lookup, fields) -> (event, action)`:
   implements the no-churn create-or-update contract. Returns `'created'`, `'updated'`, or
   `'unchanged'`. Uses `event.save()` on update to preserve `auto_now` field behavior.

Refactored `sync_lco_observation_calendar.py` to:
- Import the 6 symbols it still uses from `.calendar_utils`
- Remove all 12 symbol definitions (now in `calendar_utils`)
- Replace the 12-line `get_or_create`/compare/save block with a single
  `insert_or_create_calendar_event({'url': url}, fields)` call
- Keep the `CalendarEventTelescopeLabel` sidecar write as a separate statement (D-03)
- Remove now-unused imports (`requests`, `forms`, `urljoin`, etc.)

Updated test `patch()` targets from `sync_lco_observation_calendar.make_request` →
`solsys_code.calendar_utils.make_request` (19 occurrences).

### Deviations

**`event.save()` vs `event.save(update_fields=changed)`:** The plan described mirroring
the Gemini idiom using `update_fields=changed`, but this silently skips `auto_now` fields
(`CalendarEvent.modified`), breaking 2 tests that assert the timestamp updates after a
write. Fixed to use plain `event.save()` — matches the original LCO sync behavior exactly
and is more behavior-neutral.

**Worktree recovery:** Two executor agents were accidentally spawned in parallel (UI showed
both running). Both were killed; one had committed Task 1 in a git worktree. Orchestrator
recovered the work inline: committed Task 2 changes from the worktree, merged into main,
ran tests, fixed the `update_fields` bug, and cleaned up the orphaned worktree.

### Verification

```
$ PYTHONPATH=src python manage.py test solsys_code
Ran 186 tests in 93.345s
OK
```

All 186 tests pass. `ruff check .` and `ruff format --check .` clean.

## Self-Check: PASSED
