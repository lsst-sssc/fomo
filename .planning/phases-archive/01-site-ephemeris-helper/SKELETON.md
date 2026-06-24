# Walking Skeleton — Telescope Runs Calendar (Stage 1: Site/Ephemeris Helper)

**Phase:** 1
**Generated:** 2026-06-12

## Capability Proven End-to-End

Given a telescope name (e.g. "Magellan-Clay") and a local sunset date, the system reads the site's location and timezone from the `Observatory` model (seeded via migration) and computes dip-corrected UTC sunset/sunrise and -15 degree dark-window times accurate to within 2 minutes of the Las Campanas skycalc reference — exercising the full stack: DB write (migration) → DB read (`get_site`) → real astropy computation (`sun_event`) → real validation (dip formula + June 10 2026 -18 degree skycalc anchor).

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Astronomy library | `astropy` (`get_sun`, `AltAz`, `EarthLocation`, `Time`) | Already a project dependency (astropy 7.2.0, verified); design-doc-validated; matches CLAUDE.md constraint; `get_sun` needs no kernel download |
| Sun-position avoidance | Do NOT import `solsys_code.ephem_utils` | That module loads ~1.6GB SPICE kernels at import time (STATE.md blocker); astropy `get_sun` avoids this |
| Site data source | `Observatory` model (MPC obscode lookup), single source of truth | D-02/D-04, CLAUDE.md constraint — no hardcoded coordinate dict; `SITES` is a thin name→obscode map only |
| Timezone handling | stdlib `zoneinfo` + `tzdata` (installed), stored as `Observatory.timezone` CharField | D-01; CLAUDE.md-mandated; correctly handles Southern-Hemisphere DST |
| Root-finding | Hand-rolled coarse 1-min scan + ~10-iteration bisection (pure astropy+numpy) | scipy is only a transitive/undeclared dependency — avoid implicit reliance; bisection reaches sub-second precision, well inside the 2-min tolerance |
| Data seeding | Django `RunPython` data migration (`0002_*`) + `AddField(timezone)` | Project's established pattern for Observatory reference data (D-05) |
| Module layout | New `solsys_code/telescope_runs.py` (SITES, get_site, horizon_dip, sun_event); model extension in `solsys_code_observatory/models.py`; DB tests in `solsys_code/tests/` | Mirrors existing repo structure and CLAUDE.md test-placement rule |

## Stack Touched in Phase 1

- [x] Module scaffold — new `telescope_runs.py`, runs under existing Django app, ruff lint/format gates
- [x] Routing — N/A (helper module, no HTTP endpoint — Stage 1 is intentionally a pure helper per REQUIREMENTS Out of Scope)
- [x] Database — real **write** (migration seeds 4 Observatory records + adds timezone field) AND real **read** (`get_site` → `Observatory.objects.get(obscode=...)`)
- [x] Computation wired to data — `sun_event` consumes `site.to_earth_location()` + `site.timezone` from the DB record (the "interactive element wired to the API" analog for a headless helper)
- [x] Full-stack run command — `./manage.py migrate solsys_code_observatory && ./manage.py test solsys_code.tests.test_telescope_runs` exercises seed → lookup → compute → validate end-to-end

## Out of Scope (Deferred to Later Slices)

- Stage 2 — `load_telescope_runs` classical-run ingest command
- Stage 3 — FTS/MuSCAT4 queue-window banners
- Stage 4 — observation-record sync to calendar
- `tom_calendar` UI / template changes / calendar DB migrations (not needed for a pure helper)
- A `'twilight'` / -18 degree `kind` on `sun_event` (rejected for Stage 1; -18 is a validation cross-check value only, June 10)
- Re-verifying MPC obscode parallax constants against the seeded coordinates (sub-arcsecond differences have negligible time impact at the 2-minute tolerance)

## Subsequent Slice Plan

Each later phase (future GSD milestone, if Stage 1 succeeds) adds one vertical slice on top of this skeleton without altering the architectural decisions above:

- Stage 2: `load_telescope_runs` ingest — consumes `get_site`/`sun_event` to place classical runs on the calendar
- Stage 3: queue-window banners — reuses `sun_event` dark windows for FTS/MuSCAT4
- Stage 4: observation-record sync — joins observation records to the per-night site/sun-event boundaries from this helper
