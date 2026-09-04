# Technology Stack

**Analysis Date:** 2026-09-04

## Languages

**Primary:**
- Python 3.10+ - Core application, TOM Toolkit backend, Django framework, and all business logic (`solsys_code/`, `src/fomo/`)

**Secondary:**
- HTML/CSS/JavaScript - Django templates, Bootstrap 5 frontend components, HTMX interactions (`src/templates/`, `static/`)

## Runtime

**Environment:**
- Python 3.10, 3.11, 3.12 (tested via GitHub Actions workflows)

**Package Manager:**
- pip (configured via `pyproject.toml`)
- Lockfile: `pyproject.toml` (PEP 517/518 compliant, uses setuptools_scm for dynamic versioning)

## Frameworks

**Core:**
- Django 3.2+ (via TOM Toolkit 3.0+) - Web framework foundation (`src/fomo/`)
- TOM Toolkit 3.0+ - Target and Observation Manager framework, multi-facility support (`INSTALLED_APPS` in `settings.py`)

**API & REST:**
- Django REST Framework - REST API support with token authentication (`rest_framework`, `rest_framework.authtoken`)

**Frontend:**
- Django Crispy Forms (`crispy_forms`, `crispy_bootstrap5`) - Bootstrap 5 form rendering
- Bootstrap 5 (`django_bootstrap5`) - CSS framework for UI components
- Django HTMX (`django_htmx`) - HTMX middleware for AJAX interactions

**Observation Management:**
- tom_calendar 3.0+ - Calendar event management and scheduling (`INSTALLED_APPS`)
- tom_observations - Facility integration (LCO, Gemini, SOAR, ESO) (`tom_observations.facilities.*`)
- tom_targets - Target model and management

**Data & Features:**
- django-extensions - Management commands and utilities
- django-guardian - Object-level permissions (`guardian.backends.ObjectPermissionBackend`)
- django-tables2 - Table rendering for observation records
- django-filters - QuerySet filtering
- django-comments - Commenting system
- django-gravatar - User avatar display
- django-tasks + database backend - Async task management (`django_tasks.backends.database`)
- Plotly (via settings `PLOTLY_THEME = 'plotly_white'`) - Interactive data visualization

**Scientific Computing:**
- sorcha - Solar System object simulation, ephemeris generation, orbit conversion
- sbpy 0.6.0+ - Solar System object ephemeris computation (required by sorcha + astropy 7.2.0+)
- numpy >1.24 - Numerical computation (`numpy` for photometry/data processing)
- astropy 7.2.0+ - Astronomical calculations, coordinate transforms, time handling
  - `astropy.coordinates` - Coordinate frame transformations (`AltAz`, `EarthLocation`, `get_sun`)
  - `astropy.time` - Time/date handling with leap-second support
  - `astropy.table` - Table operations with units (`QTable`)
  - `astropy.constants` - Physical constants (GM_sun, c)
- REBOUND + ASSIST - N-body integration for barycentric ephemeris (via sorcha)
- ERFA/SOFA - European standards for astronomical reference frames (`erfa` for coordinate geometry)
- SPICEpy - JPL Horizons ephemeris kernel management (loaded via sorcha's `furnish_spiceypy()`)

**Timezone & Geographic:**
- timezonefinder 6.0+ - Timezone lookup from lat/lon (used for observation site timezone resolution)
- `zoneinfo` (stdlib) - Timezone-aware datetime handling for `America/Santiago`, `Australia/Sydney`

**Testing:**
- pytest - Test runner (configured in `pyproject.toml`)
- pytest-cov - Code coverage reporting
- factory_boy 3.2.1+ - Test data factories for `tom_targets.tests.factories`
- playwright - Headless browser functional tests (solsys_code/tests/test_bootstrap5_rendering.py)

**Development & Quality:**
- ruff 0.2.1 (pinned) - Linting and code formatting; version locked to match `.pre-commit-config.yaml`
- pre-commit - Git hooks for code quality checks
- Sphinx 2.1+ - HTML documentation generation (`docs/conf.py`)
- setuptools 62+ - Package building
- setuptools_scm 6.2+ - Version management from git tags (writes to `src/fomo/_version.py`)
- ipython, jupyter - Interactive development and notebook execution
- graphifyy - Queryable knowledge graph builder (optional, for AI integration)

## Key Dependencies

**Critical (Project-Specific):**
- tomtoolkit >=3.0.0 - TOM Toolkit framework for facility management and observation scheduling
- sorcha - Solar System object simulation, ephemeris calculation with barycentric n-body integration
- sbpy >=0.6.0 - Solar System object properties (transitive, required for sorcha + astropy compat)
- numpy >1.24 - Numerical arrays and matrix operations
- astropy 7.2.0+ - Astronomical calculations (coordinates, time, constants, tables)
- timezonefinder >=6.0 - Timezone resolution for observation sites

**Integrations (Third-Party APIs & Services):**
- tom_fink >=2.0.1 - Fink alert stream broker integration (real-time transient alerts)
- tom_alertstreams >=1.3.0 - Alert stream handling framework, Kafka support for Fink
- tom_eso >=0.3.1 - ESO facility integration (VLT observations)
- tom-registration >=2.0.1 - User registration and management UI
- tom_observations (from tomtoolkit) - LCO, Gemini, SOAR facility clients and schedulers
- tom_catalogs (from tomtoolkit) - JPL Horizons, MPC, SIMBAD, TNS catalog integrations

**Web Framework Stack:**
- Django 3.2+ (via tomtoolkit) - Web framework
- djangorestframework - REST API, token auth
- django-crispy-forms, crispy-bootstrap5 - Form rendering
- django-bootstrap5 - Bootstrap 5 support
- django-htmx - HTMX integration
- django-guardian - Row-level permissions
- django-tables2, django-filters - Data table/filtering
- django-comments - Commenting system
- django-gravatar - Avatar display
- django-extensions - Management commands

**Other Important:**
- rebound - N-body orbital mechanics (used by sorcha)
- ASSIST - Analytical Integration of Satellite and Small body Perturbations (n-body integration, used by sorcha)
- erfa - SOFA reference frame library for coordinate transforms
- spiceypy - Python interface to JPL SPICE ephemeris kernels (~1.6 GB downloaded on first use)

## Configuration

**Environment:**
- Settings via Django `settings.py` (`src/fomo/settings.py`)
- Secret key and DEBUG via settings (dev defaults in repo, production via `local_settings.py` import)
- Environment variables for secrets (LCO_API_KEY, GEMINI_API_KEY, Fink credentials)
- Local settings override: `src/fomo/local_settings.py` (imported at end of settings, not checked in)

**Build:**
- Ruff config in `pyproject.toml`: line length 120, single quotes, target Python 3.10+
- Sphinx config in `docs/conf.py`: ReadTheDocs build configured in `.readthedocs.yml`
- Pre-commit config in `.pre-commit-config.yaml`: ruff v0.2.1, pytest, Sphinx docs validation

**Database:**
- SQLite3 (development default at `src/fomo_db.sqlite3`)
- PostgreSQL ready (production-capable via Django settings)
- Default auto field: `BigAutoField`

## Platform Requirements

**Development:**
- Python 3.10+ with pip or conda
- Git (for setuptools_scm version from tags)
- SQLite3 support (built-in to Python)
- ~1.6 GB disk space for cached SPICE kernels (`~/.cache/sorcha/`)
- Network access for JPL, MPC, Fink APIs (first-run setup)

**Production:**
- Python 3.10, 3.11, or 3.12
- PostgreSQL 12+ (recommended over SQLite)
- Static file serving setup (`STATIC_ROOT`, `STATIC_URL`)
- WSGI application server (gunicorn, uwsgi) at `src.fomo.wsgi.application`
- ASGI support available at `src.fomo.asgi.application` (for async tasks)
- Email backend configuration (console default in dev, SMTP in production)
- ReadTheDocs or equivalent for documentation hosting

---

*Stack analysis: 2026-09-04*
