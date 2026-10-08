---
created: 2026-10-08T16:51:39.659Z
title: Cache telescope_runs.sun_event() and speed up the test suite
area: telescope-runs
severity: minor
files:
  - solsys_code/telescope_runs.py (sun_event, _find_crossing)
  - solsys_code/allocation_projector.py (cross-session IERS-drift reasoning — per-process cache keeps it valid)
  - .github/workflows/testing-and-coverage.yml (CI unit-test step)
  - .pre-commit-config.yaml (django-test hook)
  - src/fomo/settings.py (test-only PASSWORD_HASHERS)
---

## Problem

Profiling by the `fomo_fresh` session (production-deploy / PR #58 work, 2026-10-08) on
`origin/issue37-code-only` @ 846be34, Python 3.12, `manage.py test --exclude-tag functional
--exclude-tag ephemeris_segfault`, no coverage. Write-ups are on PR #43:
findings https://github.com/lsst-sssc/fomo/pull/43#issuecomment-6064460914 and cache-fix
results https://github.com/lsst-sssc/fomo/pull/43#issuecomment-6064524313.

CI unit-test jobs on PR #43 take 12–21 min (vs ~1 min on `main`); the local serial suite is
~8 min and the `django-test` pre-commit hook ~10 min on every code commit. Where the 706 s
(cProfile, 2,174 tests) goes:

- `telescope_runs.sun_event()`: **80%**. 2,268 calls at ~0.17–0.25 s each — every
  `_find_crossing()` is one vectorised AltAz transform over 1,441 one-minute samples (~0.11 s)
  plus ~20 single-time bisection transforms (~0.06 s). The same site/night is recomputed many
  times per process (night minting, the gap analysis' per-date loop, tests).
- Migration tests (`MigrationExecutor.migrate`): 6%.
- Password hashing: 3.7% — 368 PBKDF2 hashes at 1M iterations; no fast test hasher configured.
- Everything else ~10%; no network calls, no sleeps.
- Slowest tests: both `TestGapAnalysisSiteUnknownCount` tests at 17.8 s each
  (`observable_dates()` calls `sun_event(kind='dark')` 91 times for one view), plus dozens of
  ~2.09 s tests in `test_backfill_lco_observations*`, `test_allocation_projector` and
  `test_unattended` (each minted night costs 2 `sun_event` calls).
- Production impact: the gap-analysis view's first load of a ~90-night range costs ~15–20 s
  until `GAP_CACHE_TTL_SECONDS` caching applies.

The suite is parallel-safe: `--parallel 4` runs in 2.6 min today.

## Solution

1. **Memoise the crossing search** (measured: serial 8.0 → 2.3 min; `--parallel 4`
   2.6 min → 51 s; all 2,174 tests pass). Patch (+21/−5 in `solsys_code/telescope_runs.py`,
   git-apply-able against 846be34) at
   `/tmp/claude-10007/-home-tlister-git-fomo-fresh/30f993c6-8bda-4484-a3ae-d0bf380b61f8/scratchpad/sun_event_cache.diff`
   (scratch path — may not survive; the diff is also in the second PR #43 comment):
   - `sun_event()` keeps all validation/error messages and still calls
     `site.to_earth_location()` so coordinate-less sites raise as before.
   - The search moves into an `@lru_cache(maxsize=4096)` helper `_cached_crossings(lon, lat,
     altitude, timezone, date, threshold)` that builds the `EarthLocation`, takes
     `_local_noon_utc(date, timezone)` as anchor and returns
     `tuple(_find_crossing(anchor, location, threshold, search_hours=24))`.
   - `sun_event()` returns `.copy()` of the cached `Time`s so a caller that mutates one
     (e.g. sets `.format`) cannot corrupt the cache.
   - Because it is an inner helper, the 28 `patch(...sun_event...)` call-counting tests (D-13)
     are unaffected; no test changed. Per-process, so `allocation_projector.py`'s cross-session
     IERS-drift reasoning still holds.
   - CLAUDE.md paired docs: `telescope_runs.py` → `docs/notebooks/pre_executed/telescope_runs_demo.ipynb`
     must be re-executed (`jupyter nbconvert --to notebook --execute --inplace`) and committed with output.
2. **Make each call cheaper** (fixes the cache-miss cost — the first gap-analysis test still
   spends 15.9 s filling the cache for 91 dates): a 10-min coarse scan in `_find_crossing()`
   plus a few more bisection steps, estimated ~5× per call. Keep the 2-minute skycalc accuracy
   contract (Stage 1 core value) — verify against the existing sun-event precision tests.
3. **Run tests in parallel**: add `--parallel` to the CI unit-test step and the `django-test`
   pre-commit hook (`.pre-commit-config.yaml`).
4. **Cheaper fixtures**: `PASSWORD_HASHERS = ['django.contrib.auth.hashers.MD5PasswordHasher']`
   under test only; tag the migration tests so pre-commit can skip them.

Related but distinct: `2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md`
(skip the call at one reconciler call site; routed into the Phase 35 allocation projector).
Item 1 subsumes most of that payoff at the function level. Good fit for Phase 41 (TRIAGE-01)
or a `/gsd-quick` for item 1 alone.
