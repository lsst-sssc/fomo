# Codebase Structure

**Analysis Date:** 2026-06-12

## Directory Layout

```
fomo_devel/
├── src/                          # Main package source (entry point for setuptools)
│   ├── fomo/                      # Django project settings and WSGI/ASGI
│   │   ├── __init__.py            # Package version
│   │   ├── settings.py            # Django settings, INSTALLED_APPS, database config
│   │   ├── urls.py                # URL routing (includes TOM, ephemeris, observatory)
│   │   ├── wsgi.py                # WSGI application for production servers
│   │   ├── asgi.py                # ASGI application for async servers
│   │   └── _version.py            # Auto-generated version from git tags
│   ├── templates/                 # Jinja2/Django templates
│   │   ├── ephem_form.html        # Ephemeris request form template
│   │   ├── ephem.html             # Ephemeris results display (CSV/HTML table)
│   │   ├── solsys_code/           # Solar system code app templates
│   │   │   └── partials/          # Reusable template fragments
│   │   │       └── ephem_button.html  # "Make Ephemeris" button injected into target detail
│   │   ├── solsys_code_observatory/  # Observatory app templates
│   │   │   ├── observatory_create.html
│   │   │   ├── observatory_detail.html
│   │   │   ├── observatory_list.html
│   │   │   └── partials/
│   │   │       ├── navbar.html
│   │   │       └── navbar_list.html
│   │   ├── tom_common/            # Custom overrides of TOM templates
│   │   │   └── index.html         # Home page
│   │   └── tom_targets/
│   │       └── partials/
│   │           └── module_buttons.html  # Integration point for target buttons
│   ├── static/                    # Static CSS, JavaScript, images
│   │   └── tom_common/
│   │       ├── css/
│   │       │   └── custom.css
│   │       └── img/
│   │           └── rubin-logo-color.png
│   ├── templatetags/              # Custom Django template tags
│   │   ├── fomo_extras.py         # FOMO-specific tags
│   │   └── solsys_code_extras.py  # Solar system code tags (e.g., ephem_button context)
│   └── data/                      # Data files (example photometry, notebooks)
│       └── NGC1566/
├── solsys_code/                   # Custom Django app: Solar System object utilities
│   ├── __init__.py
│   ├── admin.py                   # Django admin configuration (empty)
│   ├── apps.py                    # App config with integration hooks
│   ├── models.py                  # Django models (currently empty, user-defined targets commented out)
│   ├── views.py                   # Main views: MakeEphemerisView, Ephemeris, JPLSBDBQuery
│   ├── forms.py                   # EphemerisForm (date range, step, observatory selection)
│   ├── ephem_utils.py             # Core business logic: ephemeris calculations, coordinate transforms
│   ├── migrations/                # Django migration files
│   ├── management/
│   │   └── commands/
│   │       └── fetch_jplsbdb_objects.py  # Management command to bulk-import objects
│   ├── tests/                     # Unit tests for ephem_utils, views
│   │   ├── __init__.py
│   │   ├── test_ephem_utils.py    # Tests for coordinate transforms, magnitude calc
│   │   ├── test_views.py          # Tests for MakeEphemerisView, Ephemeris
│   │   └── data/                  # Test fixtures (CSV, ephemeris samples)
│   ├── solsys_code_observatory/   # Sub-app: Observatory CRUD & MPC integration
│   │   ├── __init__.py
│   │   ├── admin.py               # Django admin for Observatory model
│   │   ├── apps.py                # App config
│   │   ├── models.py              # Observatory model (MPC code, location, coordinate methods)
│   │   ├── views.py               # CreateObservatory, ObservatoryDetailView, ObservatoryList
│   │   ├── forms.py               # CreateObservatoryForm (MPC code input)
│   │   ├── utils.py               # MPCObscodeFetcher (queries MPC Obscodes API)
│   │   ├── urls.py                # URL patterns for observatory views
│   │   ├── migrations/            # Django migrations
│   │   └── tests/                 # Tests for Observatory model, coordinate transforms, views
│   │       ├── __init__.py
│   │       ├── test_models.py     # Tests for Observatory.to_geocentric(), from_parallax_constants()
│   │       ├── test_utils.py      # Tests for MPCObscodeFetcher
│   │       └── test_views.py      # Tests for CRUD views
│   └── etc/                       # Supporting files (deprecated or exploratory)
├── tests/                         # Project-level tests (pytest)
│   └── fomo/                      # Package tests
│       ├── conftest.py            # pytest fixtures for Django integration
│       └── test_packaging.py      # Tests for package metadata
├── docs/                          # Sphinx documentation
│   ├── design/                    # Architecture/design decision documents
│   │   ├── design.rst             # Overview of FOMO design
│   │   ├── gsd_experiment.rst     # GSD spec-driven development experiment notes
│   │   ├── telescope_runs_calendar.rst  # Feature design: telescope runs calendar
│   │   └── (diagrams, PDFs, images)
│   ├── notebooks/                 # Jupyter notebooks for research/validation
│   │   └── (data files, examples)
│   └── pre_executed/              # Pre-rendered notebook outputs
├── .claude/                       # Claude/GSD configuration
│   ├── agents/                    # GSD agent definitions
│   ├── commands/                  # GSD command implementations
│   ├── gsd-core/                  # GSD framework (shared across projects)
│   ├── hooks/                     # Git hooks
│   └── scripts/                   # Utility scripts
├── .github/                       # GitHub workflows, issue templates
│   ├── workflows/                 # CI/CD (smoke tests, coverage)
│   └── ISSUE_TEMPLATE/
├── .planning/                     # Output directory for GSD analysis
│   └── codebase/                  # Codebase documentation (ARCHITECTURE.md, STRUCTURE.md, etc.)
├── pyproject.toml                 # Project metadata, dependencies, tool config
├── manage.py                      # Django management command entry point
├── README.md                       # Project overview
├── LICENSE                         # MIT license
└── .pre-commit-config.yaml        # Pre-commit hooks (ruff, black, isort)
```

## Directory Purposes

**`src/fomo/`:**
- Purpose: Django project configuration and WSGI/ASGI entry points
- Contains: Settings, URL routing, version management
- Key files: `settings.py` (INSTALLED_APPS includes 'solsys_code', TOM modules, Fink), `urls.py` (routes ephemeris and observatory endpoints)

**`src/templates/`:**
- Purpose: HTML templates for rendering views
- Contains: Forms (ephemeris request form), result displays, observatory management UI
- Key files: `ephem_form.html` (input form with Crispy Forms), `ephem.html` (results table), observatory CRUD templates

**`solsys_code/`:**
- Purpose: Core Solar System object management and ephemeris generation
- Contains: Views, forms, business logic, tests
- Key files: `views.py` (MakeEphemerisView, Ephemeris, JPLSBDBQuery), `ephem_utils.py` (coordinate transforms, magnitude calc), `forms.py` (EphemerisForm)

**`solsys_code/solsys_code_observatory/`:**
- Purpose: Observatory registry and MPC API integration
- Contains: Observatory model, CRUD views, MPC fetcher
- Key files: `models.py` (Observatory with coordinate conversion methods), `utils.py` (MPCObscodeFetcher), `views.py` (Create, Detail, List)

**`solsys_code/tests/`:**
- Purpose: Unit tests for ephemeris logic and views
- Contains: Test cases for coordinate transforms, magnitude calculation, view rendering
- Key files: `test_ephem_utils.py`, `test_views.py`

**`docs/design/`:**
- Purpose: Architecture and design decision documentation
- Contains: High-level design docs, component diagrams, feature specifications
- Key files: `design.rst` (overview), `telescope_runs_calendar.rst` (feature spec), `gsd_experiment.rst` (GSD usage notes)

**`tests/fomo/`:**
- Purpose: Package-level tests (not app-specific)
- Contains: Packaging tests, conftest fixtures
- Key files: `conftest.py` (pytest + Django setup), `test_packaging.py` (version, imports)

## Key File Locations

**Entry Points:**
- `manage.py`: Django management command entry point
- `src/fomo/wsgi.py`: Production WSGI application
- `src/fomo/asgi.py`: ASGI application (async support)
- `src/fomo/settings.py`: Django configuration

**Core Logic:**
- `solsys_code/views.py`: MakeEphemerisView (form handling), Ephemeris (result rendering), JPLSBDBQuery (discovery)
- `solsys_code/ephem_utils.py`: Ephemeris math (coordinate transforms, magnitude, sky motion)
- `solsys_code/forms.py`: EphemerisForm (date range, observatory, step size)
- `solsys_code/solsys_code_observatory/models.py`: Observatory model with coordinate methods
- `solsys_code/solsys_code_observatory/utils.py`: MPC API fetcher

**Configuration:**
- `src/fomo/settings.py`: INSTALLED_APPS, middleware, database, TOM config
- `src/fomo/urls.py`: URL routing (includes TOM, ephemeris, observatory)
- `pyproject.toml`: Project metadata, dependencies (tomtoolkit>=2.31.4, sorcha, tom_fink, etc.)
- `.pre-commit-config.yaml`: Linting (ruff), formatting (black)

**Testing:**
- `solsys_code/tests/test_ephem_utils.py`: Coordinate transform, magnitude tests
- `solsys_code/tests/test_views.py`: View rendering, form validation tests
- `solsys_code/solsys_code_observatory/tests/test_models.py`: Observatory model tests
- `solsys_code/solsys_code_observatory/tests/test_utils.py`: MPC fetcher tests
- `tests/fomo/conftest.py`: Django test fixtures

## Naming Conventions

**Files:**
- Django apps: PascalCase folder (`solsys_code`, `solsys_code_observatory`) with snake_case modules (`views.py`, `models.py`, `forms.py`)
- Tests: `test_*.py` (pytest discovery pattern)
- Templates: kebab-case HTML files (`ephem-form.html`, `observatory-list.html`) in app-named folders
- Management commands: kebab-case in `management/commands/` folder

**Directories:**
- Django apps: snake_case (`solsys_code`, `solsys_code_observatory`)
- Sub-packages: snake_case (`templates`, `tests`, `migrations`, `management`)
- Configuration: `.` prefix for dotfiles (`.claude`, `.github`, `.pre-commit-config.yaml`)

**Python Naming:**
- Classes: PascalCase (e.g., `MakeEphemerisView`, `Observatory`, `EphemerisForm`)
- Functions: snake_case (e.g., `convert_target_to_layup()`, `add_magnitude()`)
- Constants: UPPER_CASE (e.g., `AU_KM`, `SEC_PER_DAY`, `PI_OVER_2`)
- Model fields: snake_case (e.g., `obscode`, `rho_cos_phi`, `start_date`)

## Where to Add New Code

**New Feature (e.g., telescope scheduling):**
- Primary code: Create new view in `solsys_code/` or new sub-app alongside `solsys_code_observatory/`
- Tests: `solsys_code/tests/test_feature.py` or new test module in sub-app
- Templates: `src/templates/solsys_code/feature_name.html`
- Forms: Add to `solsys_code/forms.py` if shared, else create `solsys_code/scheduling/forms.py`
- Routes: Add to `src/fomo/urls.py` or create new app `urls.py` and include

**New Observatory-related Feature:**
- Location: `solsys_code/solsys_code_observatory/`
- Models: Add to `models.py`
- Views: Add to `views.py`
- Tests: `solsys_code/solsys_code_observatory/tests/test_*.py`
- Forms: `solsys_code/solsys_code_observatory/forms.py`
- URLs: Update `solsys_code/solsys_code_observatory/urls.py`

**New Utility Functions (coordinate transforms, calculations):**
- Location: `solsys_code/ephem_utils.py` (if ephemeris-related) or new module `solsys_code/utils.py`
- Tests: `solsys_code/tests/test_ephem_utils.py` or `test_utils.py`
- Documentation: Add docstrings with parameter types and return types

**New Template Tags:**
- Location: `src/templatetags/solsys_code_extras.py` (for FOMO-specific) or create app-specific tags in app folder
- Registration: Use `@register.simple_tag`, `@register.filter`, `@register.inclusion_tag`
- Templates: Use in any template with `{% load solsys_code_extras %}`

**Configuration Changes:**
- Local settings: Modify `src/fomo/settings.py` (see INSTALLED_APPS for pattern)
- Dependencies: Add to `pyproject.toml` `dependencies` list and update imports
- Pre-commit rules: Update `.pre-commit-config.yaml`

**Database Schema Changes:**
- Create migration: `python manage.py makemigrations solsys_code`
- File location: `solsys_code/migrations/XXXX_description.py`
- Run: `python manage.py migrate`

## Special Directories

**`src/templates/`:**
- Purpose: Django template rendering for web UI
- Generated: No (hand-written)
- Committed: Yes

**`solsys_code/migrations/`:**
- Purpose: Django database schema version control
- Generated: Yes (via `makemigrations`)
- Committed: Yes (critical for deployment)

**`docs/_build/`:**
- Purpose: Sphinx-generated HTML documentation
- Generated: Yes (from `docs/*.rst`)
- Committed: No (in `.gitignore`)

**`.planning/codebase/`:**
- Purpose: GSD codebase analysis output (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Generated: Yes (by GSD mappers)
- Committed: Yes (guides future phases)

**`.claude/gsd-core/`:**
- Purpose: GSD framework (shared across projects)
- Generated: No (shared dependency)
- Committed: Yes

**`solsys_code/etc/`:**
- Purpose: Exploratory or deprecated code
- Generated: No
- Committed: Conditionally (may be deleted)

---

*Structure analysis: 2026-06-12*
