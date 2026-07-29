<!-- refreshed: 2026-06-12 -->
# Architecture

**Analysis Date:** 2026-06-12

## System Overview

FOMO (Follow-up Observations of Moving Objects) is a Django-based Target and Observation Manager (TOM) extending the TOM Toolkit for coordinating follow-up observations of Solar System targets from Vera C. Rubin Observatory. It combines TOM Toolkit's core functionality with custom Solar System components.

```text
┌──────────────────────────────────────────────────────────────────┐
│                      Request Handling Layer                       │
├──────────────────┬────────────────────────┬──────────────────────┤
│  MakeEphemeris   │      Ephemeris         │   Observatory CRUD   │
│  `views.py`      │  `views.py:Ephemeris`  │   `solsys_code_     │
│                  │                        │    observatory/      │
│                  │                        │    views.py`         │
└────────┬─────────┴────────────┬───────────┴──────────┬───────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Forms & Business Logic Layer                   │
│  `forms.py` (EphemerisForm)                                       │
│  `ephem_utils.py` (ephemeris calculations, coordinate transforms) │
│  `solsys_code_observatory/utils.py` (MPC API fetcher)             │
└──────────────────────────┬───────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Data Access Layer                            │
│  Models: `Target` (TOM Toolkit)                                   │
│  Models: `Observatory` (local Solar System observations)          │
│  Database: SQLite3                                                │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                   External Services & APIs                        │
│  - Sorcha: Orbital mechanics & ephemeris generation               │
│  - SPICE: Coordinate transformations                              │
│  - ASSIST: N-body integration                                     │
│  - JPL SBDB: Object discovery (via JPLSBDBQuery)                  │
│  - MPC ObsCodes API: Observatory data                             │
│  - TOM Fink: Alert stream integration                             │
└──────────────────────────────────────────────────────────────────┘
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

**Overall:** Django MTV (Model-Template-View) with TOM Toolkit extensibility pattern

**Key Characteristics:**
- Plugin architecture: FOMO extends TOM Toolkit as an installed app
- Django class-based views for form handling and data display
- Wrapper services (e.g., `FakeSorchaArgs`) abstract external library complexity
- Database-backed registry of observatories queried via Sorcha
- Template tag extensions for TOM integration points

## Layers

**Presentation (Template Layer):**
- Purpose: Render user-facing forms and results to HTML/CSV
- Location: `src/templates/`
- Contains: Form templates (`ephem_form.html`), result displays (`ephem.html`), observatory CRUD templates
- Depends on: Django template context from views, Crispy Forms layout
- Used by: Django view template rendering

**View/Controller Layer:**
- Purpose: Handle HTTP requests, validate forms, orchestrate business logic
- Location: `solsys_code/views.py`, `solsys_code/solsys_code_observatory/views.py`
- Contains: `MakeEphemerisView` (FormView), `Ephemeris` (View), `CreateObservatory` (CreateView), `ObservatoryList` (ListView), `ObservatoryDetailView` (DetailView)
- Depends on: Forms, models, ephem_utils, external APIs
- Used by: URL dispatcher

**Form Layer:**
- Purpose: Define and validate input data, construct form layout
- Location: `solsys_code/forms.py`, `solsys_code/solsys_code_observatory/forms.py`
- Contains: `EphemerisForm` (date range, observatory selection, output options), `CreateObservatoryForm` (MPC code input)
- Depends on: Models, Crispy Forms helpers
- Used by: Views for initialization and validation

**Business Logic Layer:**
- Purpose: Compute ephemeris, transform coordinates, fetch external data
- Location: `solsys_code/ephem_utils.py`, `solsys_code/solsys_code_observatory/utils.py`
- Contains: Ephemeris computation functions, coordinate transforms (ERFA), magnitude calculation (add_magnitude, add_sky_motion), orbit conversion, n-body integration setup
- Depends on: Sorcha, ASSIST, SPICE, ERFA, Astropy
- Used by: Views, JPLSBDBQuery

**Data Access Layer:**
- Purpose: Manage persistent storage and query interface
- Location: Django ORM models
- Contains: TOM Toolkit `Target` (external model), `Observatory` model with coordinate transforms
- Depends on: SQLite3, Django ORM
- Used by: Views, forms

**External Service Layer:**
- Purpose: Integrate with scientific libraries and remote APIs
- Location: Various dependencies (sorcha, rebound, assist, spiceypy, etc.)
- Contains: Orbital mechanics, coordinate geometry, SPICE kernel management, JPL/MPC API clients
- Depends on: External packages, network connectivity
- Used by: Business logic layer

## Data Flow

### Primary Request Path: Ephemeris Generation

1. User navigates to target detail page and clicks "Make Ephemeris" button (`src/templates/solsys_code/partials/ephem_button.html`)
2. `MakeEphemerisView.get()` initializes form with target ID and default date range (`solsys_code/views.py:88-100`)
3. Form renders with observatory choices via QuerySet from `Observatory.objects.filter(altitude__gt=0)` (`solsys_code/forms.py:28`)
4. User submits form → `MakeEphemerisView.form_valid()` triggered (`solsys_code/views.py:120`)
5. Form extracts: target_id, start_date, end_date, step size, site_code (observatory)
6. View calls ephemeris computation:
   - `convert_target_to_layup(target)` → `solsys_code/ephem_utils.py:72` (convert Django Target to Sorcha format)
   - `build_apco_context()` → sets up ERFA astrometric context
   - `generate_assist_simulations()` → runs ASSIST n-body integrator
   - `calculate_rates_and_geometry()` → computes observer rates and light travel time
   - `add_sky_motion()` + `add_magnitude()` → adds derived columns
7. Results stored in Astropy QTable and rendered:
   - CSV download: `csv.writer` to StringIO
   - HTML table: embedded in `ephem.html` template
8. HTTP response returns CSV or HTML (`solsys_code/views.py:120-162`)

### Secondary Flow: Observatory Discovery & Management

1. User clicks "Observatory" → `ObservatoryList.get_context_data()` retrieves all observatories (`solsys_code/solsys_code_observatory/views.py:78-84`)
2. User clicks "Add Observatory" → `CreateObservatory` view loads form
3. User enters MPC code (e.g., "500") → form validation uppercases code
4. `CreateObservatory.form_valid()` triggers:
   - `MPCObscodeFetcher().query(obscode)` → HTTP GET to MPC Obscodes API (`solsys_code/solsys_code_observatory/utils.py`)
   - Parses JSON response → extracts lon, lat, name
   - `Observatory.from_parallax_constants()` → converts MPC parallax format to geodetic coordinates (lat/lon/alt) using ERFA (`solsys_code/solsys_code_observatory/models.py:70-87`)
   - Saves Observatory instance
5. Redirect to detail view (`ObservatoryDetailView`)

### Tertiary Flow: JPL Discovery

1. User accesses JPL discovery endpoint (not exposed in URLs, but available for future integration)
2. `JPLSBDBQuery` class builds HTTP query URL to JPL Small-Body Database API (`solsys_code/views.py:510-537`)
3. `.run_query()` executes HTTP request, parses JSON (`solsys_code/views.py:538-551`)
4. `.parse_results()` converts table to Astropy QTable (`solsys_code/views.py:552-566`)
5. `.create_targets()` creates TOM `Target` objects in database (`solsys_code/views.py:567-593`)

**State Management:**
- Request-local state: Form data, computed ephemeris held in request context
- Persistent state: Observatory models, Target models (TOM-managed)
- Module-level state: Sorcha ephemeris object (`ephem`), SPICE kernels (cached in `~/.cache/sorcha/`)

## Key Abstractions

**Ephemeris Geometry:**
- Purpose: Encapsulates observer position, time, and reference frames needed for coordinate transforms
- Examples: `EphemerisGeometryParameters` (from Sorcha), ERFA context setup in `ephem_utils.py`
- Pattern: Wrapper functions adapt external library interfaces to local use

**Observatory:**
- Purpose: Represents observing site with coordinate systems (geodetic, geocentric, parallax constants)
- Examples: `Observatory` model with methods `.to_parallax_constants`, `.to_geocentric()`, `.ObservatoryXYZ()`
- Pattern: Domain model with calculated properties and coordinate conversion methods

**Layup Format:**
- Purpose: Intermediate representation of orbital elements in format Sorcha expects
- Examples: NumPy array constructed from Target fields
- Pattern: Adapter converting Django ORM objects to scientific library input

**Form Flow:**
- Purpose: Encapsulates user input validation and context assembly
- Examples: `EphemerisForm` combines target ID, dates, step size, observatory selection
- Pattern: Crispy Forms layout with helper for custom HTML and actions

## Entry Points

**WSGI Application:**
- Location: `src/fomo/wsgi.py`
- Triggers: Web server (runserver, gunicorn, etc.)
- Responsibilities: Create Django WSGI application using `get_wsgi_application()`

**ASGI Application:**
- Location: `src/fomo/asgi.py`
- Triggers: ASGI server (Daphne, Hypercorn) for async support
- Responsibilities: Create Django ASGI application, configure for async

**Django Management Command:**
- Location: `manage.py` (project root)
- Triggers: `python manage.py <command>`
- Responsibilities: Execute management commands (migrate, runserver, etc.)
- Example: `python manage.py fetch_jplsbdb_objects` (custom command at `solsys_code/management/commands/fetch_jplsbdb_objects.py`)

**URL Routing Entry:**
- Location: `src/fomo/urls.py`
- Triggers: Django URL dispatcher
- Routes: `/ephem/<int:pk>/` (Ephemeris view), `/targets/<int:pk>/makeephem/` (MakeEphemerisView), `/observatory/` (solsys_code_observatory app), default TOM urls

**App Integration Hooks:**
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

**What happens:** `build_query_url()` concatenates query parameters manually into URL string (`solsys_code/views.py:510-537`), building parameter dict and then joining keys/values.

**Why it's wrong:** Fragile to parameter ordering, easy to forget URL encoding, difficult to test parameter combinations.

**Do this instead:** Use `requests.models.PreparedRequest` or `urllib.parse.urlencode()` to construct and validate query string. Example:
```python
from urllib.parse import urlencode
params = {'sb-class': 'IEO', 'discovery-date-min': '2020-01-01'}
url = f'{base_url}?{urlencode(params)}'
```

### Form Initialization with Hardcoded Date Defaults

**What happens:** `EphemerisForm.__init__()` sets start_date to now and end_date to now+20 days every time form is instantiated (`solsys_code/forms.py:37-41`).

**Why it's wrong:** Inflexible; prevents users from easily repeating queries. Test fixtures can't override defaults.

**Do this instead:** Move date logic to view layer or accept `initial_start_date` in form constructor:
```python
class EphemerisForm(forms.Form):
    def __init__(self, *args, start_date=None, end_date=None, **kwargs):
        super().__init__(*args, **kwargs)
        if start_date:
            self.fields['start_date'].initial = start_date
```

### Silent Fallback in MPC Parallax Conversion

**What happens:** `Observatory.from_parallax_constants()` and coordinate transforms return tuples with `0.0` or `None` if Observatory data is incomplete (`solsys_code/solsys_code_observatory/models.py:96-102`).

**Why it's wrong:** Silently producing invalid coordinates can lead to garbage ephemeris outputs without clear error messages.

**Do this instead:** Raise `ValidationError` if required fields are missing:
```python
if not all([self.lat, self.lon, self.altitude]):
    raise ValidationError("Observatory location data incomplete")
```

## Error Handling

**Strategy:** Mixed pattern—Django forms validate at submission; ephem_utils has try-except for external library calls; views catch exceptions and render error messages.

**Patterns:**
- Form validation: `EphemerisForm.clean()` could validate date ranges (currently not implemented)
- View-level: `MakeEphemerisView.form_valid()` wraps ephemeris computation; unhandled exceptions bubble to Django error pages
- Model-level: `Observatory.from_parallax_constants()` silently returns None values (anti-pattern)
- External APIs: `JPLSBDBQuery.run_query()` handles HTTP errors but doesn't log them

## Cross-Cutting Concerns

**Logging:** Uses Python `logging` module. `ephem_utils.py` suppresses Sorcha logger warnings (`sorcha_logger.setLevel(logging.WARNING)` at module load). Views use Django messages framework for user feedback.

**Validation:** Multi-layer:
1. Form-level: `EphemerisForm` validators on date fields, QuerySet filter on observatory choices
2. Model-level: Database constraints on Observatory (lon ±180, lat ±90)
3. Business logic: Coordinate transform methods silently handle edge cases (should raise)

**Authentication:** Uses Django's built-in auth + Guardian for object permissions (configured in `settings.py:146-148`). No custom checks in views; relies on TOM Toolkit middleware.

---

*Architecture analysis: 2026-06-12*
