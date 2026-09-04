# External Integrations

**Analysis Date:** 2026-09-04

## APIs & External Services

**Observation Facilities (Scheduling & Queue Management):**
- LCO (Las Cumbres Observatory)
  - Purpose: Schedule observations, query observation queue, retrieve status updates
  - SDK/Client: `tom_observations.facilities.lco.LCOFacility` (class-based facility adapter)
  - API Endpoint: `https://observe.lco.global`
  - Auth: API key via `FACILITIES['LCO']['api_key']` (env var `LCO_API_KEY` in `local_settings.py`)
  - Integration: Registered in `TOM_FACILITY_CLASSES`, `sync_lco_observation_calendar.py` pulls queue to calendar
  - Features: Queue status tracking, SOAR facility shares LCO credentials

- Gemini Observatory
  - Purpose: Schedule observations at Gemini South (GS) and Gemini North (GN)
  - SDK/Client: `tom_observations.facilities.gemini.GEMFacility`
  - API Endpoint: `https://139.229.34.15:8443` (GS), `https://128.171.88.221:8443` (GN)
  - Auth: Separate API keys for GS and GN via `FACILITIES['GEM']['api_key']` dict
  - User Email: `FACILITIES['GEM']['user_email']`
  - Programs: Template dict mapping program codes to descriptive text
  - Integration: Registered in `TOM_FACILITY_CLASSES`, `sync_gemini_observation_calendar.py` pulls status to calendar
  - Features: Dual-site support, program-specific observation templates

- SOAR (Southern Astrophysical Research Telescope)
  - Purpose: Schedule observations at SOAR
  - SDK/Client: `tom_observations.facilities.soar.SOARFacility`
  - API Endpoint: Inherits from LCO (`https://observe.lco.global`)
  - Auth: Same credentials as LCO (both use LCO Observation Portal)
  - Integration: Registered in `TOM_FACILITY_CLASSES`, queue sync included with LCO
  - Note: SOAR entry in `FACILITIES` dict mirrors LCO exactly for key resolution (D-04)

- ESO (European Southern Observatory - VLT)
  - Purpose: Query and schedule observations at ESO facilities
  - SDK/Client: `tom_eso.eso.ESOFacility`
  - Package: `tom_eso>=0.3.1`
  - Integration: Registered in `TOM_FACILITY_CLASSES`
  - Features: VLT observations, optical/infrared facility access

**Catalog & Reference Data Services:**
- JPL Small-Body Database (SBDB)
  - Purpose: Query minor planets, comets, asteroids; retrieve orbital elements
  - API: SBDB Query API (ssd.jpl.nasa.gov/tools/sbdb_query)
  - SDK/Client: HTTP GET requests via `JPLSBDBQuery` class in `solsys_code/views.py`
  - Usage: `fetch_jplsbdb_objects` management command queries by constraints (e.g., `"e>=1.2,q<1.3"`)
  - Constraints: Translated from human-readable form to API's `field|OP|value` format in `_translate_constraints()`
  - Output: astropy `QTable` parsed into `tom_targets.Target` rows (asteroids vs. comets handled separately)

- JPL Horizons API
  - Purpose: Retrieve ephemeris, orbital elements, spacecraft positions
  - API: `ssd.jpl.nasa.gov/api/horizons.api`
  - Integration: Accessed via sorcha's SPICE kernel manager (indirectly via spiceypy)
  - Usage: Observer notation alias table in `solsys_code/observer_codes.py` maps Horizons codes
  - Features: NAIF spacecraft IDs (e.g., `500@<NAIF SPK ID>` for space observatories)
  - Note: ~1.6 GB SPICE ephemeris kernels cached at `~/.cache/sorcha/` on first use

- MPC Observatory Codes API
  - Purpose: Resolve MPC obscode → lat/lon/alt/parallax for observing sites
  - API: `minorplanetcenter.net/iau/lists/ObsCodes.html` (fetched via HTTP)
  - SDK/Client: `MPCObscodeFetcher` in `solsys_code/solsys_code_observatory/utils.py`
  - Integration: Populates `Observatory` model with coordinates and parallax constants
  - Usage: `resolve_site()` and `create_or_update_observatory()` in `campaign_utils.py` query this
  - Storage: Results cached in local `Observatory` table; manual refresh via management commands
  - Conversion: Parallax constants ↔ geodetic coordinates via ERFA functions

- SIMBAD
  - Purpose: Resolve stellar coordinates, magnitudes, spectral types (TOM Toolkit built-in)
  - SDK/Client: Auto-registered in tomtoolkit 3.0
  - Integration: Data services catalog lookup, available via TOM UI

- TNS (Transient Name Server)
  - Purpose: Query and report transient discoveries
  - SDK/Client: `tom_alerts.brokers.tns.TNSBroker`
  - Auth: API key, bot ID, bot name via `BROKERS['TNS']` dict
  - Integration: Registered in `TOM_ALERT_CLASSES`
  - Features: Alert broker for reported transients

- GaiaBroker
  - Purpose: Gaia alert stream for stellar transients and variables
  - SDK/Client: `tom_alerts.brokers.gaia.GaiaBroker`
  - Integration: Registered in `TOM_ALERT_CLASSES`

- ALeRCE Broker
  - Purpose: Astronomical Light Curve Rapid Transient Broker
  - SDK/Client: `tom_alerts.brokers.alerce.ALeRCEBroker`
  - Integration: Registered in `TOM_ALERT_CLASSES`

- LasAir Broker
  - Purpose: Lasair transient alert stream
  - SDK/Client: `tom_alerts.brokers.lasair.LasairBroker`
  - Auth: API key via `BROKERS['LASAIR']['api_key']` (env var `LASAIR_TOKEN`)
  - Integration: Registered in `TOM_ALERT_CLASSES`

**Transient Alert Streams (Real-Time):**
- Fink Alert Stream
  - Purpose: Real-time transient alerts (photometric broker for ZTF/LSST)
  - SDK/Client: `tom_fink.alertstream.FinkAlertStream` (Kafka-based consumer)
  - Package: `tom_fink>=2.0.1`, `tom_alertstreams>=1.3.0`
  - Configuration: `ALERT_STREAMS` list in `settings.py`
  - Auth Vars:
    - `FINK_CREDENTIAL_URL` - Kafka bootstrap URL
    - `FINK_CREDENTIAL_USERNAME` - Kafka username
    - `FINK_CREDENTIAL_GROUP_ID` - Consumer group ID
    - `FINK_TOPIC` - Kafka topic name
  - Optional Vars: `FINK_MAX_POLL_NUMBER` (default 1e10), `FINK_TIMEOUT` (default 10s)
  - Data Service: Registered via `SolsysCodeConfig.data_services()` in `solsys_code/apps.py`
  - Topic Handler: `tom_fink.alertstream.alert_logger` processes incoming alerts
  - Integration: Automatic ingestion and notification system for new candidates

## Data Storage

**Databases:**
- SQLite3 (Development)
  - Connection: Local file `src/fomo_db.sqlite3` via `django.db.backends.sqlite3`
  - Uses: All development and testing data
  - Limitations: Single-writer, not recommended for production

- PostgreSQL (Production-Ready)
  - Connection: Configurable via `DATABASES['default']` in `settings.py` or `local_settings.py`
  - ORM Client: Django ORM (no explicit database client needed)
  - Support: Production deployments should use PostgreSQL for concurrency
  - Version: PostgreSQL 12+ recommended

**File Storage:**
- Local Filesystem Only
  - Media files: `MEDIA_ROOT` (default `src/fomo/data/`)
  - Static files: `STATIC_ROOT` (default `src/fomo/_static/`), `STATICFILES_DIRS` (default `static/`)
  - No cloud storage (S3, GCS, Azure) configured

**Caching:**
- File-Based Cache
  - Backend: `django.core.cache.backends.filebased.FileBasedCache`
  - Location: System temp directory (`tempfile.gettempdir()`)
  - Usage: Template rendering, computed ephemerides caching
  - Session Storage: Via `django.contrib.sessions` (in database)

**Ephemeris Kernel Cache:**
- SPICE Kernels
  - Location: `~/.cache/sorcha/` (user home cache, created on first import of `ephem_utils.py`)
  - Size: ~1.6 GB (downloaded via sorcha's `furnish_spiceypy()`)
  - Scope: Shared across all Django processes that import `ephem_utils`
  - Side Effect: Module-level import of `sorcha` triggers kernel download, paid once at startup

## Authentication & Identity

**Auth Provider:**
- Django Built-In Authentication
  - Implementation: `django.contrib.auth` with `django.contrib.auth.backends.ModelBackend`
  - Sessions: `django.contrib.sessions` (database-backed)
  - Strategy: `AUTH_STRATEGY = 'READ_ONLY'` (read access without login, write requires login)
  - Login URL: `/accounts/login/` (configurable via `LOGIN_URL`)
  - Redirect: `/` on login success, `/` on logout

**Registration:**
- Tom Registration
  - Package: `tom-registration>=2.0.1`
  - Strategy: `REGISTRATION_STRATEGY = 'open'` (self-registration, no approval required)
  - Optional: Approval-required mode with email notification (`SEND_APPROVAL_EMAILS`)
  - Middleware: `tom_registration.middleware.RedirectAuthenticatedUsersFromRegisterMiddleware`

**Object-Level Permissions:**
- Guardian
  - Package: `django-guardian`
  - Backend: `guardian.backends.ObjectPermissionBackend` (added to `AUTHENTICATION_BACKENDS`)
  - Usage: Row-level access control for `ObservationRecord`, `DataProduct`, `ReducedDatum`
  - Setting: `TARGET_PERMISSIONS_ONLY = True` (restricts to Target-level permissions)
  - Default Permission: `TARGET_DEFAULT_PERMISSION = 'OPEN'` (visible to all users)

**API Authentication:**
- Token Authentication
  - Framework: `rest_framework.authtoken` (Django REST Framework)
  - Usage: HTTP Bearer token for API requests
  - Generate: `python manage.py drf_create_token <username>`

**Third-Party API Keys:**
- LCO API Key
  - Env var: `LCO_API_KEY` (set in `local_settings.py`)
  - Storage: `FACILITIES['LCO']['api_key']`

- Gemini API Keys
  - Env vars: `GEMINI_GS_API_KEY`, `GEMINI_GN_API_KEY` (for GS and GN sites)
  - Storage: `FACILITIES['GEM']['api_key']['GS']`, `FACILITIES['GEM']['api_key']['GN']`

- Fink Kafka Credentials
  - Env vars: `FINK_CREDENTIAL_URL`, `FINK_CREDENTIAL_USERNAME`, `FINK_CREDENTIAL_GROUP_ID`, `FINK_TOPIC`
  - Storage: `ALERT_STREAMS[0]['OPTIONS']` dict (Fink FinkAlertStream)

- TNS Credentials
  - Env vars: `TNS_API_KEY`, `TNS_BOT_ID`, `TNS_BOT_NAME`
  - Storage: `BROKERS['TNS']` dict

- Lasair Token
  - Env var: `LASAIR_TOKEN`
  - Storage: `BROKERS['LASAIR']['api_key']`

## Monitoring & Observability

**Error Tracking:**
- None Configured
  - Default: Errors logged to console via `django.contrib.admin`
  - Production: Should configure Sentry or equivalent in `local_settings.py`

**Logs:**
- Console Logging (Development)
  - Backend: `logging.StreamHandler`
  - Level: INFO
  - Output: Standard output (captured by Django runserver or WSGI server)

**Email Notifications:**
- Django Email Backend (Console)
  - Dev Default: `django.core.mail.backends.console.EmailBackend` (prints to console)
  - Production: Must set `EMAIL_BACKEND`, `EMAIL_HOST`, `EMAIL_HOST_USER`, `EMAIL_HOST_PASSWORD`, `DEFAULT_FROM_EMAIL` in `local_settings.py`

## CI/CD & Deployment

**Hosting:**
- Flexible (Not Locked In)
  - WSGI Application: `src.fomo.wsgi.application` (gunicorn, uwsgi compatible)
  - ASGI Application: `src.fomo.asgi.application` (async server support, unused currently)
  - Recommended: Gunicorn + PostgreSQL backend

**CI Pipeline:**
- GitHub Actions
  - Workflow File: `.github/workflows/` (smoke-test.yml visible in git log)
  - Tests: Pytest runs on Python 3.10, 3.11, 3.12
  - Coverage: Reported to Codecov
  - Status Badge: Visible in README.md

**Documentation:**
- ReadTheDocs
  - Config: `.readthedocs.yml` (Python 3.10, Sphinx)
  - Build: Sphinx generates HTML from `docs/conf.py`
  - Excluded: Jupyter notebooks in `docs/notebooks/` excluded from pre-commit but executed in CI

## Calendar & Event Management

**Calendar Integration:**
- tom_calendar
  - Package: Included in TOM Toolkit 3.0+
  - Models: `CalendarEvent`, `CalendarEventMeta` (extended by FOMO)
  - FOMO Extensions: `solsys_code/models.py` defines `CalendarEventMeta` FK relationships
    - `observation_record` - Link to FOMO `ObservationRecord` (nullable, many-to-many via junction)
    - `observation_group` - Link to observation group identifier
    - `run` - Link to `CampaignRun` for campaign decoration
  - Features: Observation event scheduling, site/facility time windows
  - Sync Commands:
    - `sync_lco_observation_calendar.py` - Pull LCO queue → CalendarEvents
    - `sync_gemini_observation_calendar.py` - Pull Gemini schedule → CalendarEvents
    - `load_telescope_runs.py` - Compute sunset/sunrise windows for sites

**Campaign Management:**
- Custom FOMO Models
  - `CampaignRun` - Coordinated campaign identifier and metadata
    - Fields: `name`, `site`, `telescope_class`, `proposal_id`, `start_date`, `end_date`
    - Linked: Multiple `CampaignRunObservation` records, decoration via `CalendarEventMeta.run`
  - `CampaignRunObservation` - Links `CampaignRun` ↔ `ObservationRecord`
    - Fields: Campaign reference, observation reference, attribution metadata
  - Management Commands:
    - `import_campaign_csv.py` - Batch import campaigns from CSV
    - `reconcile_campaign_runs.py` - Reconcile allocated nights vs. actual observations
    - `repair_stale_campaign_run_sites.py` - Fix orphaned campaign sites

**Observation Record Management:**
- Custom Models
  - `ObservationRecord` - Persistent record of scheduled/completed observations
    - Links: Facility, proposal, target, instrument, status
  - `CalendarEventDismissal` - User dismissal of calendar event notifications
  - `ObservationRecordDismissal` - User dismissal of observation record alerts
  - Backfill Commands:
    - `backfill_lco_observations.py` - Import historical LCO queue observations
    - `backfill_lco_observation_records.py` - Create ObservationRecords from LCO API

## Webhooks & Callbacks

**Incoming Webhooks:**
- LCO Queue Status Updates
  - Mechanism: Long-polling via `sync_lco_observation_calendar.py` (pull, not push)
  - Trigger: Manual command invocation or scheduled celery task
  - Payload: LCO facility API response (JSON observation records)

- Gemini Queue Status Updates
  - Mechanism: Long-polling via `sync_gemini_observation_calendar.py` (pull, not push)
  - Trigger: Manual command invocation or scheduled celery task
  - Payload: Gemini facility API response (XML or JSON schedule)

- Fink Alert Stream
  - Mechanism: Kafka consumer (continuous subscription)
  - Trigger: Automatic (configured in `ALERT_STREAMS`)
  - Payload: Kafka topic messages (alert JSON)
  - Handler: `tom_fink.alertstream.alert_logger` processes and stores alerts

**Outgoing Webhooks:**
- None Currently Configured
  - Observation status changes: Logged to console, not pushed externally
  - Future: Could implement via TOM's `HOOKS` system (observation_change_state hook defined in settings)

## Environment Configuration

**Required Environment Variables:**
- Fink Credentials (if Fink stream enabled):
  - `FINK_CREDENTIAL_URL` - Kafka broker URL
  - `FINK_CREDENTIAL_USERNAME` - Kafka username
  - `FINK_CREDENTIAL_GROUP_ID` - Consumer group
  - `FINK_TOPIC` - Topic name
  - `FINK_MAX_POLL_NUMBER` - (optional, default 1e10)
  - `FINK_TIMEOUT` - (optional, default 10s)

- Observation Portal Credentials:
  - `LCO_API_KEY` - LCO portal API key (shared with SOAR)
  - `GEMINI_GS_API_KEY` - Gemini South API key
  - `GEMINI_GN_API_KEY` - Gemini North API key

- Alert Broker Credentials:
  - `LASAIR_TOKEN` - Lasair API token (optional)
  - `TNS_API_KEY`, `TNS_BOT_ID`, `TNS_BOT_NAME` - TNS credentials (optional)

- Django Secrets:
  - `DEBUG` - Set to False in production (default True in repo)
  - `ALLOWED_HOSTS` - Set production domain (default `['tlister-thinkmate.lco.gtn']`)
  - Email config if using SMTP: `EMAIL_BACKEND`, `EMAIL_HOST`, `EMAIL_HOST_USER`, `EMAIL_HOST_PASSWORD`, `DEFAULT_FROM_EMAIL`

**Secrets Location:**
- Configuration: `src/fomo/local_settings.py` (not checked in, imported at end of settings.py)
- Pattern: `from fomo.local_settings import *` with fallback to skip if missing
- Folding: Secret values assigned as flat names in `local_settings.py`, then merged into dict settings (e.g., `LCO_API_KEY` → `FACILITIES['LCO']['api_key']`)

---

*Integration audit: 2026-09-04*
