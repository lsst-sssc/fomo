---
phase: 01-site-ephemeris-helper
plan: 01
subsystem: api
tags: [astropy, zoneinfo, django-migrations, ephemeris, observatory-model]

# Dependency graph
requires: []
provides:
  - "solsys_code/telescope_runs.py with SITES dict, get_site(), horizon_dip(), sun_event(), crossing-search helpers"
  - "Observatory.timezone field and Observatory.to_earth_location() method"
  - "Migration 0002 seeding 4 Observatory records (268, 269, 809, E10)"
  - "DB-dependent test suite covering SITE-01..03, EPHEM-01..03"
affects: [01-02]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Coarse 1-min scan + 10-iteration bisection root-finding for solar altitude threshold crossings (astropy get_sun + AltAz, no SPICE/ephem_utils)"
    - "Local-noon-UTC anchor + forward 24h search window for D-08 'local calendar date of sunset' semantics"
    - "get_site() raises Observatory.DoesNotExist (not KeyError) for both unknown SITES names and missing DB records - single error type, no silent fallback"

key-files:
  created:
    - solsys_code/telescope_runs.py
    - solsys_code/tests/test_telescope_runs.py
    - solsys_code/solsys_code_observatory/migrations/0002_observatory_timezone_seed.py
  modified:
    - solsys_code/solsys_code_observatory/models.py

key-decisions:
  - "Renamed django.utils.timezone import to django_timezone in models.py to avoid name collision with the new Observatory.timezone field (Rule 1 - bug fix, required for the field to compile)"
  - "_find_crossing search window changed from the RESEARCH.md sketch's +/-12h-around-local-noon to a forward 0h-to-+24h-from-local-noon window, so crossings[0]/crossings[1] return (sunset of date, sunrise of date+1) matching D-06/D-07/D-08 and the plan's acceptance criteria (21:59/11:25 UTC ordering) - verified numerically against astropy 7.2.0"
  - "get_site(name) catches KeyError from the SITES lookup and re-raises as Observatory.DoesNotExist, since the plan's acceptance criteria require Observatory.DoesNotExist (not KeyError) for get_site('NoSuchTelescope')"

patterns-established:
  - "Pure astropy/zoneinfo computation module with DB lookups confined to get_site(); sun_event() takes a resolved Observatory instance"

requirements-completed: [SITE-01, SITE-02, SITE-03, EPHEM-01, EPHEM-02, EPHEM-03]

# Metrics
duration: 35min
completed: 2026-06-12
---

# Phase 1 Plan 01: Site/Ephemeris Walking Skeleton Summary

**Observatory model gains a timezone field and to_earth_location(), migration 0002 seeds 4 telescope sites (Magellan-Clay/Baade, NTT, FTS), and a new telescope_runs.py computes dip-corrected sunset/sunrise (-(0.833+dip)) and -15deg dark-window UTC crossing times via astropy get_sun/AltAz with coarse-scan + bisection root-finding.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-06-12T20:40:00Z
- **Completed:** 2026-06-12T21:16:15Z
- **Tasks:** 2 completed
- **Files modified:** 4 (1 modified, 3 created)

## Accomplishments
- Observatory model extended with `timezone` CharField and `to_earth_location()` -> `astropy.coordinates.EarthLocation`, mirroring the existing `to_geodetic()`/`to_geocentric()` style with no silent-fallback guard
- Migration `0002_observatory_timezone_seed.py` adds the `timezone` field and seeds/upserts the 4 locked Observatory records (268 Magellan-Clay, 269 Magellan-Baade, 809 NTT/La Silla, E10 FTS/Siding Spring) with reversible `RunPython`
- New `solsys_code/telescope_runs.py`: `SITES` dict (name -> obscode), `get_site()`, `horizon_dip()`, `sun_event()`, and private helpers `_solar_altitude`, `_find_crossing`, `_local_noon_utc` - astropy-only, no `ephem_utils`/SPICE import
- `sun_event(site, date, 'sun')` and `sun_event(site, date, 'dark')` return `(setting, rising)` `astropy.time.Time` (UTC) pairs using the `-(0.833+dip)` and `-15deg` thresholds respectively; `sun_event(..., 'badkind')` raises `ValueError`
- New DB-dependent `solsys_code/tests/test_telescope_runs.py` (Django `TestCase`, 9 tests) covering SITE-01/02/03 and EPHEM-01/02/03, including the June 10 2026 Las Campanas anchor (sunset ~21:59 UTC, sunrise ~11:25 UTC, both within 2 minutes)

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend Observatory model with timezone field + to_earth_location(), and migrate + seed the 4 site records** - `df9bab0` (feat)
2. **Task 2: Create telescope_runs.py (SITES, get_site, horizon_dip, sun_event + crossing search) and DB-dependent tests** - `356e002` (feat)

**Plan metadata:** pending (this commit)

## Files Created/Modified
- `solsys_code/solsys_code_observatory/models.py` - added `timezone` CharField, `to_earth_location()` method; renamed `django.utils.timezone` import alias to `django_timezone` to avoid field-name collision
- `solsys_code/solsys_code_observatory/migrations/0002_observatory_timezone_seed.py` - `AddField(timezone)` + `RunPython` seed/unseed for obscodes 268/269/809/E10
- `solsys_code/telescope_runs.py` - `SITES`, `get_site()`, `horizon_dip()`, `sun_event()`, `_solar_altitude()`, `_find_crossing()`, `_local_noon_utc()`
- `solsys_code/tests/test_telescope_runs.py` - Django `TestCase` with 9 tests covering site resolution, timezones, seeded records, dip, sun/dark events, bad-kind error

## Decisions Made
- Renamed the `django.utils.timezone` import to `django_timezone` in `models.py` (Rule 1 auto-fix) - the new `Observatory.timezone` field would otherwise shadow the module-level `timezone` import used by `created`/`modified` field defaults (`timezone.now`), causing an `F811` redefinition and a real `AttributeError` risk at class-body evaluation time.
- Changed `_find_crossing`'s search window from the RESEARCH.md sketch's `+/-search_hours/2` around `anchor` to a forward `[0, +search_hours]` window starting at `anchor` (local noon of `date`). The `+/-12h` sketch returns `(sunrise_of_date_morning, sunset_of_date_evening)` for Las Campanas - i.e. both events on the morning/evening of the *same* calendar day - which does not match D-08's "(sunset, sunrise)" pairing where sunrise belongs to the *following* morning. The `[0,+24h]` window correctly returns `(sunset @ 21:59 UTC Jun 10, sunrise @ 11:25 UTC Jun 11)`, matching the plan's acceptance criteria exactly (verified numerically against astropy 7.2.0, both within ~25-40s of the 21:59/11:25 anchors).
- `get_site(name)` catches `KeyError` from the `SITES[name]` lookup and re-raises as `Observatory.DoesNotExist` with a descriptive message, so that `get_site('NoSuchTelescope')` raises the single documented exception type per the plan's acceptance criteria (`Observatory.DoesNotExist`, not `KeyError`), preserving "no silent fallback" while giving callers one exception type to catch.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Renamed `django.utils.timezone` import to avoid collision with new `Observatory.timezone` field**
- **Found during:** Task 1
- **Issue:** Adding `timezone = models.CharField(...)` as a class attribute on `Observatory` would shadow the module-level `from django.utils import timezone` import (used by `created`/`modified` field defaults `timezone.now`), causing `ruff` `F811` and an `AttributeError` at class-definition time once the field assignment executes.
- **Fix:** Renamed the import to `from django.utils import timezone as django_timezone` and updated `created`/`modified` defaults to `django_timezone.now`.
- **Files modified:** `solsys_code/solsys_code_observatory/models.py`
- **Verification:** `ruff check`/`ruff format --check` pass clean; `py_compile` succeeds.
- **Committed in:** `df9bab0` (Task 1 commit)

**2. [Rule 1 - Bug] Adjusted `_find_crossing` search window to produce (sunset, next-morning sunrise) ordering**
- **Found during:** Task 2
- **Issue:** The RESEARCH.md `_find_crossing`/`_local_noon_utc` sketch uses a `+/-search_hours/2` window around local-noon-of-`date`, which for Las Campanas June 10 2026 returns `[11:25 UTC Jun 10 (sunrise of Jun 10 morning), 21:59 UTC Jun 10 (sunset of Jun 10 evening)]`. Per D-06/D-07/D-08 and the plan's acceptance criteria, `sun_event(...)[0]` must be the *sunset* (21:59) and `[1]` the *following morning's sunrise* (11:25 UTC Jun 11) - the +/-12h window cannot produce this ordering since both events it finds are on the morning/evening of the same UTC-adjacent day.
- **Fix:** Changed `_find_crossing`'s coarse-scan offsets from `np.arange(-search_hours*30, search_hours*30, ...)` to `np.arange(0, search_hours*60, ...)`, i.e. a forward window `[anchor, anchor+24h]` starting at local noon of `date`. Verified numerically: returns `[21:59:12 UTC Jun 10, 11:25:38 UTC Jun 11]` for Las Campanas (both within ~25-40s of the 21:59/11:25 reference anchors, well inside the 2-min tolerance), and a correctly-ordered `(sunset, sunrise)` pair for FTS (Sydney, UTC+10) as well.
- **Files modified:** `solsys_code/telescope_runs.py`
- **Verification:** Standalone script run against astropy 7.2.0 with `EarthLocation`/`get_sun`/`AltAz` for Las Campanas (268) and FTS (E10) coordinates; `horizon_dip(2402)=1.4376` (within 1.44+/-0.02); `sun`/`dark`/`badkind` paths all exercised.
- **Committed in:** `356e002` (Task 2 commit)

**3. [Rule 1 - Bug] `get_site()` raises `Observatory.DoesNotExist` instead of `KeyError` for unknown telescope names**
- **Found during:** Task 2
- **Issue:** A naive `SITES[name]` lookup raises `KeyError` for an unregistered telescope name, but the plan's acceptance criteria (and `01-PATTERNS.md`'s error-path test pattern) specify `get_site('NoSuchTelescope')` must raise `Observatory.DoesNotExist`.
- **Fix:** Wrapped the `SITES[name]` lookup in `try/except KeyError` and re-raised as `Observatory.DoesNotExist` with a descriptive message (`from exc` chaining preserved).
- **Files modified:** `solsys_code/telescope_runs.py`, `solsys_code/tests/test_telescope_runs.py`
- **Verification:** `test_get_site_unknown` asserts `Observatory.DoesNotExist`.
- **Committed in:** `356e002` (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (all Rule 1 - bug fixes required for correctness/spec compliance)
**Impact on plan:** All three fixes were necessary to make the code compile (deviation 1) and to satisfy the plan's own documented behavior/acceptance criteria exactly (deviations 2 and 3). No scope creep beyond the plan's stated `must_haves`/`acceptance_criteria`.

## Issues Encountered

**Environment blocker - cannot run `./manage.py migrate` / `./manage.py test` in this worktree.**

`./manage.py` fails at Django app-registry population with `ModuleNotFoundError: No module named 'tom_catalogs'`, because the installed `tomtoolkit==3.0.0a9` (an alpha/dev release) no longer ships `tom_catalogs` as a separate package (it existed in the 2.x line that `pyproject.toml`'s `tomtoolkit>=2.31.4` and `src/fomo/settings.py`'s `INSTALLED_APPS`/harvester entries target). This reproduces identically on a clean `HEAD` checkout (confirmed via `git show HEAD:src/fomo/settings.py`), so it is a **pre-existing environment/dependency-version issue unrelated to this plan's files** - out of scope per the executor's scope-boundary rules (fixing it would mean either pinning/reinstalling `tomtoolkit` 2.x, which is a package-manager operation excluded from auto-fix, or restructuring `INSTALLED_APPS`, which is an architectural change outside this plan).

As a result, the plan's `<verify><automated>` commands (`./manage.py migrate solsys_code_observatory`, `./manage.py test solsys_code.tests.test_telescope_runs`) could **not** be executed in this environment. To compensate:
- All non-Django logic in `telescope_runs.py` (`horizon_dip`, `_solar_altitude`, `_find_crossing`, `_local_noon_utc`, `sun_event` with a mocked site object exposing `to_earth_location()`/`timezone`/`altitude`) was exercised in a standalone script directly against the installed `astropy==7.2.0`, confirming: `horizon_dip(2402)=1.4376` (within 1.44+/-0.02 deg, EPHEM-03); `sun_event(..., 'sun')` for Las Campanas June 10 2026 returns sunset 21:59:12 UTC / sunrise (Jun 11) 11:25:38 UTC, both within ~40s of the 21:59/11:25 reference (EPHEM-01); `sun_event(..., 'dark')` returns a later dark-window start than the sun-kind sunset (EPHEM-02); `sun_event(..., 'badkind')` raises `ValueError`.
- `ruff check` and `ruff format --check` pass clean on all 4 changed/created files (CLAUDE.md quality gate).
- `python -m py_compile` succeeds on all 4 files.
- The pre-commit hook's "Run unit tests" step (pytest over `tests/`, `src/`, `docs/` per `pyproject.toml`) passed on both task commits - this does not exercise `solsys_code/tests/test_telescope_runs.py` (Django `TestCase`, run via `./manage.py test`), but confirms no regressions in the existing pytest suite.

**SITE-03 (migration applies + 4 records exist) and the DB-dependent portions of SITE-01/02/EPHEM-01/02 (`./manage.py test solsys_code.tests.test_telescope_runs`) remain unverified by automated test execution in this environment.** The migration was hand-authored following `0001_initial.py`'s structure and `01-PATTERNS.md`/`01-RESEARCH.md` Pattern 4 verbatim (AddField + RunPython with `update_or_create`, reversible via `unseed_observatories`). The verifier/orchestrator should run `./manage.py migrate solsys_code_observatory && ./manage.py test solsys_code.tests.test_telescope_runs -v 2` in an environment where `tom_catalogs` resolves (e.g. a `tomtoolkit` 2.x venv) to confirm SITE-03 and the DB-dependent test suite pass.

## User Setup Required

None - no external service configuration required. (The `tom_catalogs` environment issue above is a pre-existing local-venv/dependency-version mismatch, not new external-service configuration introduced by this plan.)

## Next Phase Readiness

- `telescope_runs.py` (SITES, get_site, horizon_dip, sun_event) and the extended `Observatory` model/migration are ready for Plan 02's validation slice (EPHEM-04/05/06 against LCO skycalc and DST checks).
- **Blocker carried forward:** the `tom_catalogs`/`tomtoolkit==3.0.0a9` environment mismatch prevents `./manage.py migrate`/`./manage.py test` from running in this worktree. Plan 02 (and the phase-gate full-suite run `./manage.py test solsys_code && python -m pytest && ruff check . && ruff format --check .`) will hit the same blocker unless the venv is fixed or a different environment is used for DB-test execution. Recommend flagging this to the user/orchestrator before Plan 02 execution.

---
*Phase: 01-site-ephemeris-helper*
*Completed: 2026-06-12*

## Self-Check: PASSED

All created files and commit hashes verified present on disk / in git history (df9bab0, 356e002, ef7936e).
