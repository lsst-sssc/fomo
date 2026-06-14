# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — Site/Ephemeris Helper

**Shipped:** 2026-06-14
**Phases:** 1 | **Plans:** 2 | **Sessions:** 1

### What Was Built
- `solsys_code/telescope_runs.py`: `SITES` registry, `get_site()`, `horizon_dip()`, and `sun_event()` returning dip-corrected UTC sunset/sunrise and -15° dark-window crossings via astropy `get_sun`/`AltAz` (coarse-scan + bisection root-finding), with no SPICE/`ephem_utils` dependency.
- `Observatory` model gained a `timezone` field and `to_earth_location()`; migration `0002` seeds the 4 telescope sites (Magellan-Clay/Baade, NTT, FTS).
- 12-test DB-dependent suite (`solsys_code/tests/test_telescope_runs.py`) covering site resolution, dip, sun/dark events, 4-date skycalc validation, -18° twilight crosscheck, and Santiago/Sydney DST resolution — all 9 v1 requirements verified 9/9.
- A pre-executed demo notebook (`docs/notebooks/pre_executed/telescope_runs_demo.ipynb`) plus a new "Demo Notebooks" convention in PROJECT.md for future phases.

### What Worked
- This was explicitly a trial of the GSD discuss→plan→execute→verify loop on this repo, and it completed end-to-end without the workflow stumbling — the secondary "experiment" goal of this milestone succeeded.
- Sourcing `SITES` coordinates from the existing `Observatory` model (by MPC obscode) instead of a standalone hardcoded dict avoided duplication and was checked against `tom_observations.facilities.lco` before deciding.
- Coarse-scan + bisection root-finding for solar-altitude threshold crossings, anchored at local noon with a forward 24h search window, produced results within ~10-55s of skycalc references — well inside the 2-minute tolerance.
- The executor's Rule-1 auto-fix path handled three real bugs inline (an import-shadowing collision, an incorrect crossing-search window, and an exception-type mismatch) without derailing the plan.

### What Was Inefficient
- A pre-existing environment issue (`tomtoolkit==3.0.0a9` no longer ships `tom_catalogs`, but `pyproject.toml`/`settings.py` are still 2.x-targeted) blocked `./manage.py migrate`/`./manage.py test` for both plans. All DB-dependent verification had to be reproduced via standalone `astropy`/`zoneinfo` scripts instead of the real Django test runner — extra executor/verifier effort, and the 12-test suite remains formally unconfirmed by `./manage.py test solsys_code`.
- Two follow-up quick tasks (demo notebook + PROJECT.md convention; ESO notebook data-dir cleanup) were needed after the phase to round out the Definition of Done — could have been folded into phase planning.

### Patterns Established
- Pure `astropy`/`zoneinfo` computation modules should avoid importing `solsys_code.ephem_utils` (triggers a ~1.6GB SPICE kernel download) unless genuinely needed.
- DB-dependent tests for `Observatory`-backed lookups go in `solsys_code/tests/` (Django suite); pure-math helpers can live in `tests/` (pytest).
- Each phase should ship a demo notebook under `docs/notebooks/pre_executed/` (or `docs/notebooks/` if dependency-free) per PROJECT.md's Demo Notebooks convention.

### Key Lessons
1. Resolve the `tom_catalogs`/`tomtoolkit==3.0.0a9` environment mismatch before starting v1.1 — it will block `./manage.py test` for every future phase until fixed.
2. When a milestone's goal includes "validate the workflow itself," call that out explicitly in verification — it materially changed how gaps (e.g. the environment blocker) were triaged (pre-existing vs. introduced).
3. Standalone-script re-verification of DB-dependent logic is a workable fallback when `./manage.py test` is blocked, but it's not a substitute for running the real test suite — track it as an open item, not a closed one.

### Cost Observations
- Sessions: 1
- Notable: Two plans (35min + 25min) completed in a single session with no replanning; both Rule-1 auto-fixes and the environment-blocker workaround were absorbed without escalation.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 1 | 1 | First GSD run on this repo; validated discuss→plan→execute→verify loop end-to-end |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1.0 | +12 (Django, unconfirmed by `./manage.py test` due to env blocker) | - | 0 |

### Top Lessons (Verified Across Milestones)

1. Fix environment/dependency blockers before they compound across milestones — the `tomtoolkit`/`tom_catalogs` mismatch surfaced in v1.0 and should be resolved before v1.1.
