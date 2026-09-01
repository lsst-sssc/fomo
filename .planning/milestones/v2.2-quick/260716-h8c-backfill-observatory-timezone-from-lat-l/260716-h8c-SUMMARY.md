---
phase: quick-260716-h8c
plan: 01
subsystem: solsys_code_observatory
tags: [observatory, timezone, mpc, backfill]
status: complete
dependency-graph:
  requires: []
  provides:
    - "Observatory.timezone auto-populated from lat/lon on Tier-2 MPC-resolved records"
  affects:
    - "_project_calendar_event() (consumes obs.timezone; no longer needs a manual admin edit for most new MPC-resolved sites)"
tech-stack:
  added:
    - "timezonefinder>=6.0 (offline IANA timezone-from-coordinates lookup)"
  patterns:
    - "Module-level lazy-cached singleton (_get_timezone_finder()) for an expensive-to-construct third-party object"
key-files:
  created: []
  modified:
    - pyproject.toml
    - solsys_code/solsys_code_observatory/utils.py
    - solsys_code/solsys_code_observatory/tests/test_utils.py
decisions:
  - "timezonefinder chosen per operator pre-approval; no alternative library evaluation performed."
  - "Backfill only fires when obs.timezone is still blank after reading any MPC-supplied value defensively -- never overwrites an existing value, never fabricates for unresolvable coordinates (preserves CR-01 graceful-failure/retry behavior)."
metrics:
  duration: 25min
  completed: 2026-07-16
---

# Phase quick-260716-h8c Plan 01: Backfill Observatory.timezone from lat/lon Summary

Added an offline `timezonefinder` lookup so `MPCObscodeFetcher.to_observatory()` auto-populates
`Observatory.timezone` from the resolved lat/lon at record-creation time, closing the gap where
Tier-2 MPC-resolved observatories always land with a blank timezone and force a manual admin edit
before `_project_calendar_event()`'s calendar projection can succeed.

## What Was Built

1. **`timezonefinder>=6.0` dependency** added to `pyproject.toml` and installed into the active
   environment (resolved to `8.2.5`).
2. **Lazy-cached `TimezoneFinder` singleton** (`_get_timezone_finder()` in
   `solsys_code/solsys_code_observatory/utils.py`) -- constructed once (loading its boundary-polygon
   data on first use) and reused across every `to_observatory()` call. Import is deferred inside the
   function so importing `utils.py` doesn't pay the polygon-data load cost unless a lookup actually
   happens.
3. **Backfill logic in `to_observatory()`**, inserted immediately after `from_parallax_constants()`
   (which resolves `obs.lat`/`obs.lon`) and before the `created_at`/`updated_at` parsing block:
   - Reads any timezone the MPC record already carries (`self.obs_data.get('timezone', '')`,
     defensive -- this key is always absent in live data).
   - Only when still blank and both `obs.lat`/`obs.lon` are resolved, calls
     `_get_timezone_finder().timezone_at(lat=obs.lat, lng=obs.lon)`.
   - A truthy IANA zone name is assigned to `obs.timezone`; a `None` result (coordinate outside any
     timezone polygon, e.g. open ocean) leaves `obs.timezone` as `''` -- no fabricated guess,
     preserving the CR-01 resolve-fails-gracefully / stays-retryable contract.
4. **Three new regression tests** in `TestMPCObscodeFetcher`:
   - `test_to_observatory_backfills_timezone_from_coordinates` -- reuses the existing E10 (Siding
     Spring) fixture and asserts the real `timezonefinder` lookup resolves to `'Australia/Sydney'`.
   - `test_to_observatory_does_not_clobber_existing_timezone` -- injects a deliberately wrong
     `'America/New_York'` into `obs_data['timezone']` and asserts it survives unchanged.
   - `test_to_observatory_leaves_timezone_blank_when_unresolvable` -- patches
     `_get_timezone_finder` to return a `MagicMock` whose `timezone_at()` returns `None`, and asserts
     `obs.timezone == ''`.

## Verification

- `python manage.py test solsys_code.solsys_code_observatory.tests.test_utils` -- 9/9 pass (6
  existing + 3 new).
- `python manage.py test solsys_code` -- full suite: **511/511 pass** (508 previously-passing + 3
  new), no regressions.
- `ruff check` / `ruff format --check` on the three modified files -- clean.
- `ruff check .` / `ruff format --check .` across the whole repo -- 5 pre-existing findings
  surfaced, all in `docs/notebooks/pre_executed/*.ipynb` files not touched by this task (see
  "Deviations" below); none in the files this plan modified.

## Deviations from Plan

### Out-of-scope findings (not fixed, logged only)

**Repo-wide `ruff check .` / `ruff format --check .` surfaced 5 pre-existing findings in three
notebook files** (`load_telescope_runs_demo.ipynb`, `sync_gemini_observation_calendar_demo.ipynb`,
`sync_lco_observation_calendar_demo.ipynb`) -- unsorted imports, a missing docstring, and a
139-character line. None of these notebooks were created or modified by this task, and none of the
three files this plan touches (`pyproject.toml`, `utils.py`, `test_utils.py`) have any lint
findings. Per the SCOPE BOUNDARY rule, these are out-of-scope pre-existing issues in unrelated
files and were left unfixed rather than "helpfully" cleaned up alongside this change.

No other deviations. Plan executed as written.

## Self-Check: PASSED

- FOUND: pyproject.toml (grep confirms `timezonefinder>=6.0` present)
- FOUND: solsys_code/solsys_code_observatory/utils.py (`_get_timezone_finder` and backfill block present)
- FOUND: solsys_code/solsys_code_observatory/tests/test_utils.py (3 new tests present, all passing)
- FOUND commit 421e7c8: chore(260716-h8c): add timezonefinder dependency
- FOUND commit e8b8a12: feat(260716-h8c): backfill Observatory.timezone from lat/lon
- FOUND commit 75962de: test(260716-h8c): add timezone backfill regression tests
