---
created: 2026-07-02T20:16:15.850Z
title: Rename calendar_utils.py private helpers to reflect shared-module API
area: general
files:
  - solsys_code/calendar_utils.py
  - solsys_code/management/commands/sync_lco_observation_calendar.py
  - solsys_code/tests/test_sync_lco_observation_calendar.py
---

## Problem

`solsys_code/calendar_utils.py` is a genuinely shared module now — it has three real
consumers (`load_telescope_runs.py`, `sync_lco_observation_calendar.py`,
`sync_gemini_observation_calendar.py`). This was confirmed while checking off the
2026-06-23 todo (`extract-site-telescope-mapping-and-instrument-extraction-int.md`,
`resolves_phase: 11`) during Phase 14 discussion — that extraction is complete and
correct (see `14-CONTEXT.md`'s "Reviewed Todos (not folded)" section), so this is a
narrower follow-up on top of it, not a re-opening of that todo.

Several symbols in `calendar_utils.py` are imported and used across module boundaries
while still carrying a leading underscore, which conventionally signals "private to
this module" — but they're a de facto public API now:

- `_derive_telescope`, `_extract_instrument`, `_resolve_placement_block`,
  `_coarse_telescope_label` — imported by `sync_lco_observation_calendar.py`
- `_aperture_class_from_telescope_code` — imported directly by
  `solsys_code/tests/test_sync_lco_observation_calendar.py`
- `SITE_TELESCOPE_MAP` (a constant, same issue) — imported directly by that same test
  file

By contrast, the functions that are genuinely internal-only correctly kept the
underscore and should stay as-is: `_find_science_config`, `_find_exposure_signal_config`,
`_has_muscat_exposure_signal` (verified via `grep` — no references outside
`calendar_utils.py`).

Secondary, lower-priority observation: the tests exercising `calendar_utils.py`'s logic
still live in `solsys_code/tests/test_sync_lco_observation_calendar.py` (the old
command-file location) rather than a `test_calendar_utils.py`. Test-file-to-module
ownership doesn't match anymore.

## Solution

TBD. Options to weigh:
- Drop the leading underscore on the cross-module-consumed names
  (`derive_telescope`, `extract_instrument`, `resolve_placement_block`,
  `coarse_telescope_label`, `aperture_class_from_telescope_code`, `SITE_TELESCOPE_MAP`
  already has no underscore) and update the three consumer files' imports accordingly.
- Or add clean public aliases in `calendar_utils.py` re-exporting the underscored
  internals, leaving the originals untouched (lower blast radius, but two names for
  the same thing).
- While touching this, consider moving the `calendar_utils.py`-owned tests out of
  `test_sync_lco_observation_calendar.py` into a new `test_calendar_utils.py`.

Low urgency — style/naming-convention cleanup, not a bug. Good candidate for a quick
task next time `calendar_utils.py` or one of its three consumers is touched anyway.
