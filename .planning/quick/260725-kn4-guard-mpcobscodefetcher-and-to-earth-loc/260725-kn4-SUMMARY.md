---
phase: quick-260725-kn4
plan: 01
subsystem: observatory
tags: [django, mpc-api, astropy, earthlocation, satellite-obscode, tdd]

# Dependency graph
requires: []
provides:
  - "MPCObscodeFetcher.to_observatory() handles null longitude/rhocosphi/rhosinphi (space-based MPC obscodes) without raising TypeError"
  - "Observatory.to_earth_location() raises an actionable ValueError for a coordinate-less Observatory instead of TypeError"
  - "resolve_site() now resolves a real SATELLITE_OBSTYPE Observatory for a satellite obscode via Tier 2, instead of fabricating a Tier-3 'NEEDS REVIEW:' placeholder"
affects: [campaign-approval, campaign-gap-analysis, telescope-runs-calendar]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Null-coordinate guard: check all related fields (longitude/rhocosphi/rhosinphi) together before conversion, rather than guarding a single field and leaving a second identical TypeError one line below"
    - "None over 0.0 for 'unknown' geodetic fields: altitude=None states 'unknown/not applicable' honestly, vs. altitude=0.0 which falsely claims sea level"
    - "ValueError over None-return for a not-configured-for-this-computation state, chosen specifically because every existing caller already handles ValueError (audited per-caller before choosing)"

key-files:
  created: []
  modified:
    - solsys_code/solsys_code_observatory/utils.py
    - solsys_code/solsys_code_observatory/models.py
    - solsys_code/solsys_code_observatory/tests/test_utils.py
    - solsys_code/solsys_code_observatory/tests/test_models.py
    - solsys_code/tests/test_campaign_approval.py
    - solsys_code/campaign_views.py

key-decisions:
  - "A satellite Observatory gets altitude=None, not the model's 0.0 default -- 0.0 falsely claims sea level; None honestly states unknown, and every existing .altitude reader was audited to confirm this is safe (models.py:133/150 truthiness guards, telescope_runs.py:283 unreachable since to_earth_location() raises first, forms.py:28's altitude__gt=0 filter excludes NULL exactly as it excluded 0.0)."
  - "to_earth_location() raises ValueError (not a custom exception, not None) for a coordinate-less Observatory -- audited all three existing callers of the sole production consumer (telescope_runs.sun_event()) and confirmed each already handles ValueError gracefully, so this guard needed zero caller changes."
  - "Guard on all three of longitude/rhocosphi/rhosinphi together in to_observatory(), not just longitude -- a partially-specified position can't be converted either way, and guarding only longitude would leave float(None) still crashing one line below."
  - "The emergent Tier-2 resolve_site() behavior change for satellite obscodes (no longer falling through to a Tier-3 placeholder) required zero edits to campaign_utils.py -- proven by a new regression test that patches requests.get and exercises the real to_observatory()."

requirements-completed: []

# Metrics
duration: ~11min (measured from first RED commit to final task commit; wall-clock including reads was longer)
completed: 2026-07-25
---

# Quick Task 260725-kn4: Guard MPCObscodeFetcher and to_earth_location() against null/missing coordinates Summary

**Two null-coordinate guards (MPCObscodeFetcher.to_observatory(), Observatory.to_earth_location()) turn a live TypeError crash on every space-based MPC obscode (250 HST, 258 Gaia, C51 NEOWISE) into valid coordinate-less Observatory rows and an actionable ValueError, with the emergent side effect that resolve_site() now resolves real satellite Observatories instead of Tier-3 placeholders.**

## Performance

- **Duration:** ~11 min (commit-span); TDD RED/GREEN cycle for both guards
- **Tasks:** 3/3 completed
- **Files modified:** 6 (+ DEFERRED.md)

## Accomplishments
- `MPCObscodeFetcher.to_observatory()` no longer crashes with `TypeError: float() argument must be a string or a real number, not 'NoneType'` on a space-based MPC obscode's null `longitude`/`rhocosphi`/`rhosinphi` -- it now saves a valid `Observatory` with `lon`/`lat`/`altitude` all `None` and blank `timezone`, every other field populated identically to the ground path.
- `Observatory.to_earth_location()` raises an actionable `ValueError` naming the obscode/short_name for any position-less `Observatory` (all-null or altitude-only-null), instead of `TypeError` from `None * u.deg`. No caller (`telescope_runs.sun_event()` and its three downstream `ValueError` handlers) needed any change.
- Proved, via a new regression test that patches `requests.get` and exercises the real (fixed) `to_observatory()`, that `resolve_site()` now resolves a real, non-placeholder `SATELLITE_OBSTYPE` Observatory for a satellite obscode instead of falling through to a fabricated Tier-3 `NEEDS REVIEW:` placeholder -- an emergent consequence achieved with zero edits to `campaign_utils.py`.
- Corrected the now-false comment in `campaign_views.py` documenting the old Tier-2 `TypeError` fall-through (comment-only diff, mechanically verified).

## Task Commits

Each task followed RED -> GREEN:

1. **Task 1: Guard to_observatory() against null coordinates for space-based obscodes**
   - `958aa8e` (test, RED): failing tests for the live obscode-250 satellite payload
   - `bb77ef0` (fix, GREEN): guard on longitude/rhocosphi/rhosinphi together, set lon/lat/altitude=None
2. **Task 2: Guard Observatory.to_earth_location() against a missing geodetic position**
   - `678bab9` (test, RED): failing raise/return tests
   - `a331068` (fix, GREEN): raise `ValueError` naming obscode/short_name when lon/lat/altitude is None
3. **Task 3: Cover the emergent Tier-2 resolve_site() change, correct the now-false comment, run gates**
   - `4336653` (test): `TestResolveSiteSatelliteObscode` regression test, `campaign_views.py` comment fix, `DEFERRED.md`

## Files Created/Modified
- `solsys_code/solsys_code_observatory/utils.py` - Null-coordinate guard in `MPCObscodeFetcher.to_observatory()`
- `solsys_code/solsys_code_observatory/models.py` - Null-position guard in `Observatory.to_earth_location()`
- `solsys_code/solsys_code_observatory/tests/test_utils.py` - Satellite-record and ground-record regression tests for `to_observatory()`
- `solsys_code/solsys_code_observatory/tests/test_models.py` - `to_earth_location()` raise-and-return tests
- `solsys_code/tests/test_campaign_approval.py` - `TestResolveSiteSatelliteObscode` Tier-2 satellite regression test
- `solsys_code/campaign_views.py` - Corrected now-false comment (comment-only diff)
- `.planning/quick/260725-kn4-guard-mpcobscodefetcher-and-to-earth-loc/DEFERRED.md` - Two out-of-scope follow-up items

## Decisions Made
See `key-decisions` in frontmatter above. Summary: `altitude=None` (not `0.0`) for a satellite record; `ValueError` (not `None`-return, not a custom exception class) from `to_earth_location()`; guard all three coordinate-source fields together in `to_observatory()`, not just longitude.

## Deviations from Plan

None - plan executed exactly as written, including the deliberate narrow comment-only exception in `campaign_views.py` that the plan itself called for.

## Issues Encountered

The worktree's `src/fomo/_version.py` (setuptools_scm-generated, gitignored, not tracked in VCS) was missing, which made `./manage.py`/`python manage.py` fail with `ModuleNotFoundError: No module named 'src.fomo._version'` before any test could run. Copied the file from the main repo checkout (a generated artifact, not source -- harmless to regenerate locally, remains untracked/gitignored in the worktree) so the Django test runner could import settings. No repository files were affected by this workaround.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Both guards are in place and regression-tested; `resolve_site('250')`-style satellite obscodes now resolve to real Observatories.
- `DEFERRED.md` records two follow-ups for a future pass: (1) whether satellite sites should be excluded from campaign coverage-gap analysis entirely (currently degrades safely to "unknown" dates via the existing D-03 `ValueError` skip); (2) closing out the now-stale Open Questions bullet in `docs/design/telescope_runs_calendar.rst`.
- No blockers for the awaiting-next-milestone project state.

---
*Phase: quick-260725-kn4*
*Completed: 2026-07-25*

## Self-Check: PASSED

All 7 declared files found on disk; all 5 task commit hashes (958aa8e, bb77ef0, 678bab9, a331068, 4336653) verified present in git log.
