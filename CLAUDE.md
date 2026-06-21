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

# Tests — there are TWO independent test suites (see "Testing" below):
python -m pytest                          # pytest suite: tests/, src/, docs/ only
./manage.py test                          # Django app tests (solsys_code et al.)
./manage.py test solsys_code.tests.test_views.TestSplitNumberUnitRegex   # single Django test

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

`pyproject.toml` sets `testpaths = ["tests", "src", "docs"]`, so **`python -m pytest` does NOT collect
the Django app tests** under `solsys_code/`. Those use `django.test.TestCase` and run under the Django
test runner (`./manage.py test`). When adding tests, put pure-Python/packaging tests under `tests/` and
Django/DB-dependent tests under the relevant app's `tests/` package.

## Conventions

- Database is local SQLite (`src/fomo_db.sqlite3`); `DEBUG=True` and the secret key in `settings.py` are
  dev defaults — production overrides belong in a `local_settings.py` (imported at the end of `settings.py`).
- Targets are `NON_SIDEREAL`; default target permission is `OPEN` and `AUTH_STRATEGY='READ_ONLY'`.
- **Target test factories:** when fixturing a `Target` in tests or notebooks, always use
  `tom_targets.tests.factories.NonSiderealTargetFactory`, never `SiderealTargetFactory` — FOMO is
  exclusively for Solar System / non-sidereal targets, so a sidereal fixture misrepresents what the
  code actually handles. This applies to every GSD subagent (planner, executor, code-reviewer) that
  writes or reviews test/demo code touching `Target`.
- ruff config (`pyproject.toml`) follows Rubin DM style: many `N8xx` naming rules are intentionally
  ignored so astronomical variable names (e.g. `H`, `G`, `RA_deg`) are allowed. Format with single quotes.
- pre-commit blocks direct commits to `main`, clears Jupyter notebook output, runs ruff, builds Sphinx
  docs, and runs the pytest suite. CI (`.github/workflows/`) tests Python 3.10–3.12.
- **Planning-doc terminology:** in CONTEXT.md/RESEARCH.md/PLAN.md/PATTERNS.md and other
  `.planning/` artifacts, prefer plain English over DB jargon. Write "create or update" /
  "find-or-create" / "create the record if missing, otherwise update it in place" instead of
  "upsert". This applies to every GSD subagent (discuss-phase, researcher, planner, checker) —
  they all read this file before producing planning docs.
- **Demo notebook companions are part of the deliverable**, not optional polish added after the
  fact. Each of `solsys_code/telescope_runs.py`,
  `solsys_code/management/commands/load_telescope_runs.py`, and
  `solsys_code/management/commands/sync_lco_observation_calendar.py` has a paired demo notebook
  under `docs/notebooks/pre_executed/` — `telescope_runs_demo.ipynb`,
  `load_telescope_runs_demo.ipynb`, and `sync_lco_observation_calendar_demo.ipynb` respectively —
  that must stay in sync with the module's behavior. Any plan whose tasks change one of these
  modules' behavior (new extraction logic, new parameters, new fixture shapes — not pure
  refactors or typo fixes) must include its paired notebook in `files_modified` and add or update
  cells exercising the new behavior with real executed output, regenerated via
  `jupyter nbconvert --to notebook --execute --inplace` and committed (pre-commit clears notebook
  output everywhere else, but `pre_executed/` copies are committed with output, per the
  pre-commit convention noted above). When a new module gets its own demo notebook, extend this
  list. This gap was hit twice already — Phase 5 (fixed after the fact via quick task
  `260619-f7u`) and Phase 6 (fixed via quick task `260620-v9x`) — both times because the plan's
  `files_modified` never scoped the notebook in. This applies to every GSD subagent touching
  these modules: the planner (scope the paired notebook into `files_modified` and into a task up
  front, not as a follow-up); the plan-checker (treat this as part of CLAUDE.md Compliance —
  flag any plan that modifies one of the listed modules' behavior without its paired notebook in
  `files_modified`); the executor (update the notebook as part of plan execution, not as an
  afterthought); and the verifier (treat a missing or stale notebook update as a must-have gap,
  not a nice-to-have, whenever the plan touched one of these modules).

<!-- GSD:project-start source:PROJECT.md -->

## Project

**Telescope Runs Calendar — Stage 1 (Site/Ephemeris Helper)**

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

**Core Value:** Stage 1 must do two things at once: produce sun-event times accurate to
within 2 minutes of the LCO skycalc reference tool (the feature actually
works), and be built end-to-end through GSD's discuss/plan/execute/verify
loop without the workflow stumbling on this repo's conventions (the
experiment actually validates). Either failing is a meaningful result.

### Constraints

- **Astronomy library**: `astropy` (`get_sun`, `AltAz`, `EarthLocation`) for
  sun-position calculations — matches the design doc's validated approach.

- **Timezones**: `zoneinfo` (stdlib, `tzdata` installed) for
  `America/Santiago` and `Australia/Sydney`.

- **Data source**: Site coordinates come from `Observatory` model records
  (MPC obscode lookup), not hardcoded constants — Observatory records for the
  3 sites must exist (created via CreateObservatory form).

- **Precision**: Sunset/sunrise must match LCO skycalc to <= 2 minutes; horizon
  dip at 2402 m must be 1.44° ± 0.02°.

- **Testing**: DB-dependent tests (Observatory lookups) go in
  `solsys_code/tests/`, run with `./manage.py test solsys_code`. Quality gates:
  `ruff check .` and `ruff format --check .` must stay clean.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Python 3.10+ - Core application and TOM Toolkit backend (Django-based)
- HTML/CSS/JavaScript - Django templates and frontend components (Bootstrap 4 based)

## Runtime

- Python 3.10, 3.11, 3.12 (tested across versions via GitHub Actions)
- `pip` - Python package management
- Lockfile: `pyproject.toml` (PEP 517/518 compliant)

## Frameworks

- Django 2.1+ (via TOM Toolkit) - Web framework for TOM Toolkit-based TOM application
- TOM Toolkit 2.31.4+ - Target and Observation Manager framework for Solar System object follow-up
- Django REST Framework - REST API support (`rest_framework`, `rest_framework.authtoken`)
- Django Crispy Forms (`crispy_forms`, `crispy_bootstrap4`) - Form rendering with Bootstrap 4
- Bootstrap 4 (`bootstrap4`) - CSS framework
- Plotly (configured in settings, `PLOTLY_THEME = 'plotly_white'`) - Interactive visualization
- Django HTMX (`django_htmx`) - HTMX middleware for AJAX interactions
- Django ORM (via TOM Toolkit) - Database abstraction and models
- SQLite3 (default development) - File-based database backend
- pytest - Test runner
- pytest-cov - Code coverage reporting (`--cov` flags in GitHub workflows)
- setuptools 62+ - Package building
- setuptools_scm 6.2+ - Version management from git tags
- ruff 0.2.1+ - Linting and code formatting (via pre-commit)
- Sphinx - Documentation generation

## Key Dependencies

- tomtoolkit>=2.31.4 - TOM Toolkit framework for observatory management and observations
- tom_fink>=1.0.0 - Fink alert stream integration
- tom_alertstreams - Alert stream handling framework
- sorcha - Solar System object simulation and planning
- tom_eso - ESO (VLT) facility integration
- tom_observations - Core observation facilities (LCO, Gemini, SOAR)
- tom_catalogs - Catalog harvesters (JPL Horizons, MPC, SIMBAD, TNS)
- tom_registration - User registration and management
- django.contrib.auth - Authentication and authorization
- django.contrib.contenttypes - Content type framework
- django.contrib.sessions - Session management
- django.contrib.messages - Messaging framework
- django.contrib.sites - Multi-site framework
- django.contrib.admin - Django admin interface
- django.contrib.staticfiles - Static file serving
- django-extensions - Management commands and utilities
- django-guardian - Object-level permissions
- django-comments - Commenting system
- django-filters - Filtering for querysets
- django-tables2 - Table rendering
- django-gravatar - Gravatar integration
- django_gravatar - Avatar display
- numpy>1.24 - Numerical computing (for photometry/data processing)

## Configuration

- Environment variables via `os.getenv()` (see INTEGRATIONS.md for env var list)
- Django settings module: `src.fomo.settings`
- Local settings override via `local_settings.py` import (fallback: no error on missing)
- `pyproject.toml` - Main configuration (Python 3.10+ required, version dynamic via setuptools_scm)
- `.readthedocs.yml` - ReadTheDocs build configuration (Python 3.10, Sphinx)
- `.pre-commit-config.yaml` - Pre-commit hooks (ruff, pytest, Sphinx, validation)
- Ruff config in `pyproject.toml` - Format style (single quotes), line length 120

## Platform Requirements

- Python 3.10, 3.11, or 3.12
- Git (for setuptools_scm version management)
- SQLite3 support
- Pandoc (optional, for Jupyter notebook rendering in docs)
- Python 3.10+
- SQLite3 or PostgreSQL (configurable via `DATABASES` setting)
- Static file serving setup (via Django `STATIC_URL`, `STATIC_ROOT`, `MEDIA_ROOT`)
- WSGI application server (configured at `src.fomo.wsgi.application`)
- Sphinx 2.1+ - HTML documentation generation
- ReadTheDocs - Hosted documentation platform

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- Snake case for Python files (e.g., `test_ephem_utils.py`, `solsys_code_observatory`)
- Test files follow pattern: `test_*.py` (e.g., `test_models.py`, `test_views.py`, `test_utils.py`)
- Django app directories use descriptive snake_case with nested structures (e.g., `solsys_code/`, `solsys_code_observatory/`)
- Snake case throughout (e.g., `split_number_unit_regex`, `convert_target_to_layup`, `add_magnitude`, `add_sky_motion`)
- Private/internal functions use leading underscore (e.g., `_translate_constraints`)
- Method names follow Django conventions: `get_*`, `form_valid`, `setUp`, `handle`
- Snake case for all variables and parameters (e.g., `target_id`, `start_time`, `obscode`, `test_observatory`)
- Constants use UPPER_CASE (e.g., `AU_KM`, `SEC_PER_DAY`, `PI_OVER_2`, `MJD_TO_JD_CONVERSION`)
- Class attributes and properties follow snake case (e.g., `test_target`, `bary_vec`, `sun_dict`)
- Use modern Python type hints (Python 3.10+): `tuple[float, float]`, `dict[str, Any]`, `Optional[dict[str, Any]]`
- Return type annotations on methods: `def form_valid(self, form: EphemerisForm) -> HttpResponse:`
- Parameter type annotations where helpful: `def query(self, obscode: str, dbg: bool = False)`
- PascalCase for class names (e.g., `Observatory`, `EphemerisForm`, `JPLSBDBQuery`, `FakeSorchaArgs`)
- Inner/nested classes allowed (e.g., `Meta` in Django models)

## Code Style

- Line length: 120 characters (enforced by `ruff` and `black`)
- Quote style: Single quotes preferred by ruff formatter (e.g., `'ephem_form.html'`)
- Target Python version: 3.10+
- Tool: `ruff` for linting and formatting
- Configuration in `pyproject.toml`: `[tool.ruff]`
- Pre-commit hook runs `ruff --fix` and `ruff-format` on all Python files
- Ruff lint rules include: E (pycodestyle), W (warnings), F (Pyflakes), N (pep8-naming), UP (pyupgrade), B (bugbear), SIM (simplify), I (isort)
- Per-file ignores for tests: `D101`, `D102` (missing docstrings)
- Per-file ignores for migrations: `D100`, `D101`, `D102`, `D103`, `E501`, `RUF012`
- Exceptions to naming rules: `N802`, `N803`, `N806`, `N812`, `N813`, `N815`, `N816`, `N999` (allow some variations for scientific/Numpy compatibility)

## Import Organization

- No path aliases defined in this project; relative imports use dot notation (e.g., `from .forms import`, `from .ephem_utils import`)
- Absolute imports from installed packages: `from tom_targets.models import Target`
- Profile: `black`
- Line length: 120

## Error Handling

- Use generic `try/except` blocks for expected failures (e.g., `ValueError` when parsing time strings)
- Custom exceptions not extensively used; rely on built-in exceptions and Django exceptions
- Logging at `debug` level for expected failures: `logger.debug(f'Query failed with status {resp.status_code}')`
- Raise generic `Exception` for invariant violations (e.g., `raise Exception('Must provide target_id')`)

## Logging

- Get logger with `__name__`: `logger = logging.getLogger(__name__)`
- Log at `debug` level for diagnostic info: `logger.debug('No data found in results')`
- Test files can disable logging during test runs: `logging.disable(logging.CRITICAL)`
- Use f-strings for log messages: `logger.debug(f'Query failed with status {resp.status_code}')`

## Comments

- Comment non-obvious algorithmic steps (e.g., "Convert from heliocentric->barycentric using the Sun's position")
- Comment constants and their meaning (e.g., "Speed of light in km/s")
- Comment field meanings in data structures (e.g., chi-square values, degrees of freedom)
- Use comments to explain the "why" not the "what" (code should be readable, comments explain intent)
- Block comments above code sections that need context
- Not used (Python project, not TypeScript)
- Docstrings use Google-style format with `Args:`, `Returns:`, `Raises:` sections

## Docstring Style

- Google-style docstrings (not NumPy style, despite presence of NumPy code)
- Example from `ephem_utils.py`:
- Class docstrings: Simple one-liner (e.g., `"""View for making an ephemeris"""`)
- Method docstrings: Include Parameters and Returns sections
- One-liner functions may skip docstrings if name is self-explanatory

## Function Design

- Methods typically 10-50 lines
- Longer methods acceptable for view handlers (50-100+ lines) due to Django boilerplate
- Extract complex logic into helper functions
- Use keyword arguments for optional form parameters
- Type hints on parameters are encouraged
- Default parameters for optional behavior (e.g., `sun_dict=None`)
- Use type hints for return values
- Return `HttpResponse` from views
- Return `Optional[...]` for nullable types
- Tuples return multiple values with type hints: `-> tuple[float, float, float]`

## Module Design

- Modules export all public functions and classes
- No `__all__` definitions observed; relies on convention (no leading underscore = public)
- Internal/private use indicated by leading underscore
- No barrel files (index-style `__init__.py`) in use
- Package `__init__.py` files are typically empty or minimal

## Code Quality Standards

- `D101`: Missing docstring in public class (enforced except in tests)
- `D102`: Missing docstring in public method (enforced except in tests)
- `D103`: Missing docstring in public function
- Test files (`**/tests/*`) exempt from `D101`, `D102` requirements
- Avoid module-level mutable state
- Exception: `ephem_utils.py` loads and caches SPICE ephemeris kernels at module load time (acceptable for initialization)

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Django App Setup | Entry point, URL routing, WSGI/ASGI | `src/fomo/settings.py`, `urls.py`, `wsgi.py`, `asgi.py` |
| Ephemeris Generation | Form handling and ephemeris request workflow | `solsys_code/views.py:MakeEphemerisView` |
| Ephemeris Display | CSV/HTML rendering of computed ephemeris | `solsys_code/views.py:Ephemeris` |
| Ephemeris Math | Orbital mechanics, coordinate transforms, magnitude calculation | `solsys_code/ephem_utils.py` |
| Observatory Management | CRUD for observatory sites (lat/lon, altitude) | `solsys_code/solsys_code_observatory/models.py`, `views.py` |
| JPL Discovery | Query JPL SBDB for solar system objects | `solsys_code/views.py:JPLSBDBQuery` |
| Form Validation & UI | Form fields and Crispy Forms layout | `solsys_code/forms.py`, `solsys_code_observatory/forms.py` |
| TOM Integration | App config hooks, template tags | `solsys_code/apps.py`, `src/templatetags/` |

## Pattern Overview

- Plugin architecture: FOMO extends TOM Toolkit as an installed app
- Django class-based views for form handling and data display
- Wrapper services (e.g., `FakeSorchaArgs`) abstract external library complexity
- Database-backed registry of observatories queried via Sorcha
- Template tag extensions for TOM integration points

## Layers

- Purpose: Render user-facing forms and results to HTML/CSV
- Location: `src/templates/`
- Contains: Form templates (`ephem_form.html`), result displays (`ephem.html`), observatory CRUD templates
- Depends on: Django template context from views, Crispy Forms layout
- Used by: Django view template rendering
- Purpose: Handle HTTP requests, validate forms, orchestrate business logic
- Location: `solsys_code/views.py`, `solsys_code/solsys_code_observatory/views.py`
- Contains: `MakeEphemerisView` (FormView), `Ephemeris` (View), `CreateObservatory` (CreateView), `ObservatoryList` (ListView), `ObservatoryDetailView` (DetailView)
- Depends on: Forms, models, ephem_utils, external APIs
- Used by: URL dispatcher
- Purpose: Define and validate input data, construct form layout
- Location: `solsys_code/forms.py`, `solsys_code/solsys_code_observatory/forms.py`
- Contains: `EphemerisForm` (date range, observatory selection, output options), `CreateObservatoryForm` (MPC code input)
- Depends on: Models, Crispy Forms helpers
- Used by: Views for initialization and validation
- Purpose: Compute ephemeris, transform coordinates, fetch external data
- Location: `solsys_code/ephem_utils.py`, `solsys_code/solsys_code_observatory/utils.py`
- Contains: Ephemeris computation functions, coordinate transforms (ERFA), magnitude calculation (add_magnitude, add_sky_motion), orbit conversion, n-body integration setup
- Depends on: Sorcha, ASSIST, SPICE, ERFA, Astropy
- Used by: Views, JPLSBDBQuery
- Purpose: Manage persistent storage and query interface
- Location: Django ORM models
- Contains: TOM Toolkit `Target` (external model), `Observatory` model with coordinate transforms
- Depends on: SQLite3, Django ORM
- Used by: Views, forms
- Purpose: Integrate with scientific libraries and remote APIs
- Location: Various dependencies (sorcha, rebound, assist, spiceypy, etc.)
- Contains: Orbital mechanics, coordinate geometry, SPICE kernel management, JPL/MPC API clients
- Depends on: External packages, network connectivity
- Used by: Business logic layer

## Data Flow

### Primary Request Path: Ephemeris Generation

### Secondary Flow: Observatory Discovery & Management

### Tertiary Flow: JPL Discovery

- Request-local state: Form data, computed ephemeris held in request context
- Persistent state: Observatory models, Target models (TOM-managed)
- Module-level state: Sorcha ephemeris object (`ephem`), SPICE kernels (cached in `~/.cache/sorcha/`)

## Key Abstractions

- Purpose: Encapsulates observer position, time, and reference frames needed for coordinate transforms
- Examples: `EphemerisGeometryParameters` (from Sorcha), ERFA context setup in `ephem_utils.py`
- Pattern: Wrapper functions adapt external library interfaces to local use
- Purpose: Represents observing site with coordinate systems (geodetic, geocentric, parallax constants)
- Examples: `Observatory` model with methods `.to_parallax_constants`, `.to_geocentric()`, `.ObservatoryXYZ()`
- Pattern: Domain model with calculated properties and coordinate conversion methods
- Purpose: Intermediate representation of orbital elements in format Sorcha expects
- Examples: NumPy array constructed from Target fields
- Pattern: Adapter converting Django ORM objects to scientific library input
- Purpose: Encapsulates user input validation and context assembly
- Examples: `EphemerisForm` combines target ID, dates, step size, observatory selection
- Pattern: Crispy Forms layout with helper for custom HTML and actions

## Entry Points

- Location: `src/fomo/wsgi.py`
- Triggers: Web server (runserver, gunicorn, etc.)
- Responsibilities: Create Django WSGI application using `get_wsgi_application()`
- Location: `src/fomo/asgi.py`
- Triggers: ASGI server (Daphne, Hypercorn) for async support
- Responsibilities: Create Django ASGI application, configure for async
- Location: `manage.py` (project root)
- Triggers: `python manage.py <command>`
- Responsibilities: Execute management commands (migrate, runserver, etc.)
- Example: `python manage.py fetch_jplsbdb_objects` (custom command at `solsys_code/management/commands/fetch_jplsbdb_objects.py`)
- Location: `src/fomo/urls.py`
- Triggers: Django URL dispatcher
- Routes: `/ephem/<int:pk>/` (Ephemeris view), `/targets/<int:pk>/makeephem/` (MakeEphemerisView), `/observatory/` (solsys_code_observatory app), default TOM urls
- Location: `solsys_code/apps.py:SolsysCodeConfig`
- Method: `target_detail_buttons()` → injects "Make Ephemeris" button into TOM target detail view
- Method: `data_services()` → registers Fink data service for alert integration

## Architectural Constraints

- **Threading:** Django is single-threaded at the request level; ephemeris computation is synchronous. ASSIST and Sorcha operations run in-process and block request handling for large date ranges.
- **Global state:** Module-level Sorcha `ephem` object and SPICE kernels loaded once at startup (`solsys_code/ephem_utils.py:62-69`). This is memory-efficient but prevents kernel updates without restart.
- **Database:** SQLite3 has concurrent write limitations; production deployments should migrate to PostgreSQL.
- **Coordinate frames:** All ephemeris computations assume J2000 equatorial coordinates; celestial latitude/longitude are computed in ecliptic frame and transformed back.
- **Observatory selection:** Form restricts to observatories with `altitude > 0` (no submarine or underground sites).

## Anti-Patterns

### Inline Query URL Construction in JPLSBDBQuery

```python

```

### Form Initialization with Hardcoded Date Defaults

```python

```

### Silent Fallback in MPC Parallax Conversion

```python

```

## Error Handling

- Form validation: `EphemerisForm.clean()` could validate date ranges (currently not implemented)
- View-level: `MakeEphemerisView.form_valid()` wraps ephemeris computation; unhandled exceptions bubble to Django error pages
- Model-level: `Observatory.from_parallax_constants()` silently returns None values (anti-pattern)
- External APIs: `JPLSBDBQuery.run_query()` handles HTTP errors but doesn't log them

## Cross-Cutting Concerns

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
