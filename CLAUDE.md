# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FOMO (Follow-up Observations of Moving Objects) is a Django web app built on the
[TOM Toolkit](https://tom-toolkit.readthedocs.io/) for coordinating follow-up observations of
interesting Solar System targets, primarily from the Vera C. Rubin Observatory. It computes
ephemerides for non-sidereal targets and ingests minor-body orbits from JPL.

## Commands

```bash
# One-time dev setup (editable install + dev deps + pre-commit). Use a fresh conda/venv first.
./.setup_dev.sh

# Run the Django dev server / any admin task. Settings module is src.fomo.settings (set by manage.py).
./manage.py runserver
./manage.py migrate
./manage.py createsuperuser

# Custom management command: query JPL SBDB and create Targets from new matches
./manage.py fetch_jplsbdb_objects --orbital_constraints "e>=1.2,q<1.3" --group_name NEOs
./manage.py fetch_jplsbdb_objects --orbit_class IEO

# Tests — use the Django test runner for new tests (see "Testing" below):
./manage.py test                          # Django app tests (solsys_code et al.)
./manage.py test solsys_code.tests.test_views.TestSplitNumberUnitRegex   # single Django test
python -m pytest                          # legacy pytest suite: tests/, src/, docs/ only

# Lint / format (also enforced by pre-commit). Single quotes, 120-col line length.
ruff check . --fix
ruff format .
```

## Layout (non-obvious)

- `src/fomo/` is **only** the Django project config (`settings.py`, `urls.py`, `wsgi.py`). There is no
  app code here. `manage.py` sets `DJANGO_SETTINGS_MODULE=src.fomo.settings`, but `settings.py` itself
  refers to `fomo.urls` / `fomo.wsgi`, so `src/` must be importable as a path root.
- `solsys_code/` (at the repo **root**, not under `src/`) is the main custom Django app and holds nearly
  all the real logic. `solsys_code/solsys_code_observatory/` is a nested sub-app for MPC observatories.
- `src/templates/` and `src/templatetags/` hold the project-level templates and template tags
  (the `solsys_code_extras` / `fomo_extras` tag libraries, registered via `INSTALLED_APPS`).

## Architecture

Two largely separate feature areas, both living in `solsys_code/`:

**1. Ephemeris generation** (`solsys_code/views.py` `Ephemeris`/`MakeEphemerisView`, backed by
`solsys_code/ephem_utils.py`). A `Target`'s orbital elements are converted to a barycentric cartesian
state vector (`convert_target_to_layup`), integrated forward with **REBOUND + ASSIST** under full GR
(`generate_assist_simulations`), and turned into RA/Dec/rates via **sorcha** light-time integration.
Positions are then transformed to observed Alt/Az/HA with **erfa/SOFA** (`build_apco_context`,
`calculate_rates_and_geometry`), and apparent magnitude + sky motion are appended
(`add_magnitude`, `add_sky_motion`). Output is rendered as an ephemeris table.

> **Heavy import side effect:** importing `solsys_code.ephem_utils` (transitively, `solsys_code.views`)
> runs `fomo_furnish_spiceypy()` at module load, which downloads **~1.6 GB of SPICE kernels** to
> `~/.cache/sorcha/` on first use and builds the ASSIST ephemeris. Anything that imports these modules
> (including `manage.py test` collecting `solsys_code/tests/`) pays this cost.

**2. Target ingestion from JPL** (`solsys_code/views.py` `JPLSBDBQuery`, exposed via the
`fetch_jplsbdb_objects` management command). Builds Small-Body Database Query API URLs from
human-readable constraint strings (e.g. `"e>=1.2"`, `"1.0<a<2.0"`, `"H is defined"`; translated to the
API's `field|OP|value` form in `_translate_constraints`), parses results into an astropy `QTable`, and
creates `tom_targets.Target` rows (asteroids vs. comets handled differently for naming and H/G vs M1/K1).

**Observatories** (`solsys_code/solsys_code_observatory/`): the `Observatory` model stores MPC site
codes and geodetic position, with conversions between MPC parallax constants and lat/lon/alt (erfa).
`utils.MPCObscodeFetcher` populates it from the MPC Observatory Codes API.

**TOM plugins & data services** are wired in `settings.py` (`INSTALLED_APPS`, `TOM_FACILITY_CLASSES`,
`TOM_ALERT_CLASSES`, `ALERT_STREAMS`, `DATA_SERVICES`). Notably JPL Scout and Fink are registered as
data services via `SolsysCodeConfig.data_services()` in `solsys_code/apps.py`; the navbar and
target-detail buttons are injected via the app-config integration hooks (`nav_items`,
`target_detail_buttons`).

## Testing

This is a Django project, so **new tests should be written for and run with the Django test runner**
(`./manage.py test`), under the relevant app's `tests/` package (e.g. `solsys_code/tests/`), using
`django.test.TestCase` / `SimpleTestCase`. Prefer this even for pure-Python logic with no DB
dependency — use `SimpleTestCase` and loops with `self.subTest(...)` in place of
`pytest.mark.parametrize`.

A legacy pytest suite still exists under `tests/`, `src/`, and `docs/` (`pyproject.toml`
`testpaths`), which `python -m pytest` collects and **does not** include the Django app tests under
`solsys_code/`. Don't add new tests there; it's kept only for the existing packaging/doctest checks.

## Conventions

- Database is local SQLite (`src/fomo_db.sqlite3`); `DEBUG=True` and the secret key in `settings.py` are
  dev defaults — production overrides belong in a `local_settings.py` (imported at the end of `settings.py`).
- Targets are `NON_SIDEREAL`; default target permission is `OPEN` and `AUTH_STRATEGY='READ_ONLY'`.
- ruff config (`pyproject.toml`) follows Rubin DM style: many `N8xx` naming rules are intentionally
  ignored so astronomical variable names (e.g. `H`, `G`, `RA_deg`) are allowed. Format with single quotes.
- pre-commit blocks direct commits to `main`, clears Jupyter notebook output, runs ruff, builds Sphinx
  docs, and runs the pytest suite. CI (`.github/workflows/`) tests Python 3.10–3.12.
