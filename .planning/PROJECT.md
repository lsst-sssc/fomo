# Telescope Runs Calendar — Stage 1 (Site/Ephemeris Helper)

## What This Is

A small helper module (`solsys_code/telescope_runs.py`) for FOMO that resolves a
telescope name to its observing site (via the existing `Observatory` model,
looked up by MPC obscode) and computes dip-corrected UTC sunset, sunrise, and
-15° dark-window crossing times for a given date. This is Stage 1 of the
"telescope runs on the calendar" feature (issue #37) — the foundation that
Stages 2-4 (classical run ingest, queue window banners, observation-record
sync) will build on.

This GSD run is deliberately scoped to Stage 1 only: a self-contained,
well-specified unit used to trial the GSD discuss→plan→execute→verify loop on
this codebase before deciding whether to scale to the full 4-stage feature.

## Current State

**Shipped:** v1.0 "Site/Ephemeris Helper" — 2026-06-14

- All 9 v1 requirements (SITE-01..03, EPHEM-01..06) validated in Phase 01;
  verifier score 9/9 (`.planning/milestones/1.0-ROADMAP.md`).
- Both halves of the Core Value below held: the sun-event math matches Las
  Campanas skycalc to well within 2 minutes, and the GSD discuss→plan→
  execute→verify loop ran end-to-end on this repo without workflow-level
  blockers — the experiment that motivated this milestone succeeded.
- **Environment blocker resolved (as of 2026-06-13):** the v1.0 note about
  `tomtoolkit==3.0.0a9` no longer shipping `tom_catalogs` is stale. This
  branch's history now includes `main`'s `tomtoolkit3-migration` (PR #38,
  merged 2026-06-11); `tom_catalogs` is no longer referenced anywhere in
  `pyproject.toml`/`settings.py`. Confirmed: `python manage.py test
  solsys_code` runs **79/79** tests OK, including the full 16-test
  `solsys_code/tests/test_telescope_runs.py` suite (EPHEM-04/05/06
  skycalc/twilight/DST checks all pass under the real Django test runner).

## Current Milestone: v1.1 Classical Run Ingest (Stage 2)

**Goal:** Add a `load_telescope_runs` management command that parses
classical-schedule run lines (telescope, instrument, status, date range) and
idempotently creates one `tom_calendar.CalendarEvent` per observing night,
using Stage 1's `telescope_runs.SITES` / `get_site()` / `sun_event()` for
sunset/sunrise times.

**Target features:**
- Parse run lines like `NTT EFOSC2 allocation 9-13 July`,
  `Magellan IMACS 13-19 July (proposed)`,
  `Magellan Proto-Lightspeed Jul 8-12 (proposed)` — month before or after the
  day range, no year given, hyphenated instrument names
- Expand a date range S..E to `E - S + 1` nightly `CalendarEvent`s
  (`start_time = sunset(d)`, `end_time = sunrise(d+1)`)
- Populate `telescope`/`instrument`/`title`/`description` (including the -15°
  dark window and the original run line text)
- Idempotent re-run (no duplicate events on repeated ingest)
- Late-December run lines roll over into the next calendar year

Stages 3-4 (FTS queue banners, observation-record sync) remain deferred
pending this milestone's outcome.

## Core Value

Stage 1 must do two things at once: produce sun-event times accurate to
within 2 minutes of the Las Campanas skycalc reference tool (the feature actually
works), and be built end-to-end through GSD's discuss/plan/execute/verify
loop without the workflow stumbling on this repo's conventions (the
experiment actually validates). Either failing is a meaningful result.

## Requirements

### Validated

- ✓ `Observatory` model (`solsys_code_observatory.models.Observatory`) stores
  MPC-obscode-keyed site `lat`/`lon`/`altitude` with geodetic/geocentric
  conversion helpers — existing
- ✓ `tom_calendar.models.CalendarEvent` (third-party, via TOM Toolkit) has the
  fields needed to represent a telescope run — existing, used by Stages 2-3
- ✓ `tom_observations.models.ObservationRecord` (LCOFacility configured) carries
  `scheduled_start`/`scheduled_end`/status for real observation blocks —
  existing, used by Stage 4
- ✓ A `SITES`-equivalent lookup (`telescope_runs.get_site()`) resolves a
  telescope name (Magellan, NTT, FTS) to an `Observatory` record (by MPC
  obscode), constructing an `astropy.coordinates.EarthLocation` from its
  lat/lon/altitude, plus a timezone (`America/Santiago` for Magellan/NTT,
  `Australia/Sydney` for FTS) — Validated in Phase 01: site-ephemeris-helper
  (SITE-01, SITE-02)
- ✓ `sun_event(site, date, kind)` returns UTC sunset, sunrise, and -15° dark
  crossing times, applying the refraction+semidiameter (-0.833°) and
  altitude-dependent horizon-dip correction (`dip = 1.76' * sqrt(h_metres)`)
  — Validated in Phase 01: site-ephemeris-helper (EPHEM-01, EPHEM-02)
- ✓ Horizon-dip helper returns 1.44° ± 0.02° at 2402 m (Las Campanas
  altitude) — Validated in Phase 01: site-ephemeris-helper (EPHEM-03)
- ✓ Computed Las Campanas sunset/sunrise for June 2026 (sample nights
  1/10/20/30) agree with Las Campanas skycalc to within 2 minutes — Validated in
  Phase 01: site-ephemeris-helper (EPHEM-04)
- ✓ Computed astronomical twilight (-18°) for Las Campanas on 10 June 2026
  agrees with skycalc's twi.end/twi.beg (19:16/06:08 local) to within 2
  minutes — Validated in Phase 01: site-ephemeris-helper (EPHEM-05)
- ✓ `America/Santiago` resolves to UTC-4 in June and UTC-3 in January;
  `Australia/Sydney` resolves to UTC+10 in July and UTC+11 in January —
  Validated in Phase 01: site-ephemeris-helper (EPHEM-06)
- ✓ `Observatory` records exist for Magellan (Las Campanas), NTT (La Silla),
  and FTS (Siding Spring) — created via the existing CreateObservatory form
  — Validated in Phase 01: site-ephemeris-helper (SITE-03)

### Active

_(none — all Stage 1 requirements validated in Phase 01)_

### Out of Scope

- Stage 3 (FTS/MuSCAT4 queue window banners) — deferred; FTS queue input format
  is still an open item from the design doc
- Stage 4 (observation-record sync to calendar) — deferred
- Any `tom_calendar` UI/template changes — not needed for Stage 1 (no DB
  migrations or calendar rendering yet)
- Reworking `tom_observations`' existing astroplan-based visibility/airmass
  plots — confirmed separate (locally computed, different purpose/horizon
  convention from `sun_event()`), not touched by this stage

## Context

- **Codebase**: FOMO (Django + TOM Toolkit), Solar System follow-up
  coordination. Codebase map already exists at `.planning/codebase/`.
- **Design doc**: `docs/design/telescope_runs_calendar.rst` — full feasibility
  study and 4-stage implementation plan for issue #37, validated against the
  Las Campanas skycalc tool for June 2026.
- **Experiment doc**: `docs/design/gsd_experiment.rst` — rationale for using
  this feature as a GSD trial, scoped to Stage 1.
- **Site coordinate sourcing decision**: rather than the design doc's
  standalone hardcoded `SITES` dict, coordinates come from `Observatory`
  records (by MPC obscode). Checked for duplication against
  `tom_observations.facilities.lco` (has partial/slightly different Siding
  Spring coordinates, no Magellan/NTT, no timezone) — `Observatory` is the
  better source. Timezone data is new regardless.
- **Two-test-suite split**: `solsys_code/` Django app tests run via
  `./manage.py test solsys_code`; pure-Python helpers under `tests/` run via
  `python -m pytest`. Because `SITES` now depends on `Observatory` (a Django
  model), tests exercising it need DB access and belong in
  `solsys_code/tests/`. The pure-Python parts of `sun_event()`'s math (e.g.
  the dip-correction formula in isolation) may still be unit-testable under
  `tests/`.
- **SPICE kernel side effect**: importing `solsys_code.ephem_utils` (or
  running `./manage.py test`) triggers a ~1.6GB one-time SPICE kernel
  download/cache (`~/.cache/sorcha/`). `solsys_code/telescope_runs.py` itself
  should avoid this import unless genuinely needed — `astropy` alone covers
  `EarthLocation`/`get_sun`/`AltAz`.

## Constraints

- **Astronomy library**: `astropy` (`get_sun`, `AltAz`, `EarthLocation`) for
  sun-position calculations — matches the design doc's validated approach.
- **Timezones**: `zoneinfo` (stdlib, `tzdata` installed) for
  `America/Santiago` and `Australia/Sydney`.
- **Data source**: Site coordinates come from `Observatory` model records
  (MPC obscode lookup), not hardcoded constants — Observatory records for the
  3 sites must exist (created via CreateObservatory form).
- **Precision**: Sunset/sunrise must match Las Campanas skycalc to <= 2 minutes; horizon
  dip at 2402 m must be 1.44° ± 0.02°.
- **Testing**: DB-dependent tests (Observatory lookups) go in
  `solsys_code/tests/`, run with `./manage.py test solsys_code`. Quality gates:
  `ruff check .` and `ruff format --check .` must stay clean.

## Demo Notebooks

Each phase should produce a demo notebook showing its new feature in action,
to support evaluation of GSD-driven work. Put it under
`docs/notebooks/pre_executed/` when it needs special setup or DB state (DB
records, large data, third-party APIs — excluded from automated
Sphinx/CI/ReadTheDocs builds per `docs/notebooks/README.md`); put it under
`docs/notebooks/` only when it runs cleanly with repo resources in any
environment (in which case it may be added to `notebooks.rst`). See
`docs/notebooks/pre_executed/telescope_runs_demo.ipynb` (Phase 01) for an
example. This is part of each phase's Definition of Done.

### Django setup boilerplate for notebooks

Any notebook that calls `django.setup()` (required before importing
`solsys_code` modules that touch models, e.g. `Observatory`) needs this setup
cell — both fixes were discovered the hard way on the Phase 01 demo notebook:

```python
import os
import sys
from pathlib import Path

import django

# Ensure the repo root is on sys.path so `src.fomo.settings` is importable
# when this notebook is executed from its own directory (e.g.
# docs/notebooks/pre_executed/).
repo_root = str(Path.cwd().resolve().parents[2])  # adjust depth to repo root
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'src.fomo.settings')

# Jupyter's ipykernel runs inside an asyncio event loop, but Django's ORM is
# sync-only by default and refuses to run there; this opts back in.
os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'true')

django.setup()
```

Without the `sys.path` fix, imports fail with `ModuleNotFoundError: No module
named 'src'`. Without `DJANGO_ALLOW_ASYNC_UNSAFE`, any ORM call (e.g.
`Observatory.objects.get(...)`) raises `SynchronousOnlyOperation`. Adjust
`parents[N]` to match the notebook's depth under the repo root.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Source `SITES` coordinates from `Observatory` model by MPC obscode, not a standalone hardcoded dict | Avoids duplicating lat/lon/altitude already modeled in `solsys_code_observatory`; checked `tom_observations.facilities.lco` for overlap and found it incomplete/inconsistent for this purpose | Implemented in Phase 01 via `telescope_runs.get_site()` and `Observatory.to_earth_location()` |
| Scope this GSD run to Stage 1 only | Per `gsd_experiment.rst` recommendation — self-contained, well-specified unit to trial the GSD workflow before committing to the full 4-stage feature | Phase 01 completed end-to-end through GSD discuss/plan/execute/verify loop; 9/9 requirements validated |
| Tests touching `Observatory`-backed `SITES` lookups go in `solsys_code/tests/` (Django suite) | Consistent with existing two-suite split; the pure-Python `tests/` suite has no DB access | Implemented in Phase 01 (`solsys_code/tests/test_telescope_runs.py`) |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-13 — started v1.1 milestone*
