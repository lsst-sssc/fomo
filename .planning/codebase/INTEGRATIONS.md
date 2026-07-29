# External Integrations

**Analysis Date:** 2026-06-12

## APIs & External Services

**Alert Brokers:**
- ALeRCE - Automatic Learning for the Rapid Classification of Events
  - SDK/Client: `tom_alerts.brokers.alerce.ALeRCEBroker`
  - Status: Configured in `TOM_ALERT_CLASSES` in `src/fomo/settings.py`

- Gaia - ESA Gaia mission alerts
  - SDK/Client: `tom_alerts.brokers.gaia.GaiaBroker`
  - Status: Configured in `TOM_ALERT_CLASSES` in `src/fomo/settings.py`

- LasAIR - Scottish transient survey broker
  - SDK/Client: `tom_alerts.brokers.lasair.LasairBroker`
  - Auth: `LASAIR_TOKEN` (environment variable)
  - Status: Configured in `TOM_ALERT_CLASSES` in `src/fomo/settings.py`

- Transient Name Server (TNS) - IAU transient naming and classification
  - SDK/Client: `tom_alerts.brokers.tns.TNSBroker`
  - Auth: `api_key`, `bot_id`, `bot_name` (configured in `BROKERS` dict in settings)
  - Status: Configured in `TOM_ALERT_CLASSES` in `src/fomo/settings.py`

**Alert Streams:**
- Fink - Real-time alert stream from Vera C. Rubin Observatory (LSST)
  - SDK/Client: `tom_fink.alertstream.FinkAlertStream`
  - Auth: Kafka-based credentials
  - Environment vars required:
    - `FINK_CREDENTIAL_URL` - Kafka broker URL
    - `FINK_CREDENTIAL_USERNAME` - Kafka username
    - `FINK_CREDENTIAL_GROUP_ID` - Kafka consumer group
    - `FINK_TOPIC` - Topic to subscribe to (default: `fink.stream`)
    - `FINK_MAX_POLL_NUMBER` - Max alerts per poll (default: 1e10)
    - `FINK_TIMEOUT` - Poll timeout in seconds (default: 10)
  - Handler: `tom_fink.alertstream.alert_logger`
  - Status: Active alert stream configuration in `ALERT_STREAMS` in `src/fomo/settings.py`

**Catalog Harvesters:**
- JPL Horizons - NASA Solar System small-body ephemeris service
  - SDK/Client: `tom_catalogs.harvesters.jplhorizons.JPLHorizonsHarvester`
  - Status: Configured in `TOM_HARVESTER_CLASSES` in `src/fomo/settings.py`

- Minor Planet Center (MPC) - IAU's central repository for Solar System object data
  - SDK/Client: `tom_catalogs.harvesters.mpc.MPCHarvester`
  - Status: Configured in `TOM_HARVESTER_CLASSES` in `src/fomo/settings.py`

- MPC Explorer - Interactive MPC data browser
  - SDK/Client: `tom_catalogs.harvesters.mpc.MPCExplorerHarvester`
  - Status: Configured in `TOM_HARVESTER_CLASSES` in `src/fomo/settings.py`

- SIMBAD - Astronomical star database (CDS)
  - SDK/Client: `tom_catalogs.harvesters.simbad.SimbadHarvester`
  - Status: Configured in `TOM_HARVESTER_CLASSES` in `src/fomo/settings.py`

- Transient Name Server (TNS) - Harvester for transient data
  - SDK/Client: `tom_catalogs.harvesters.tns.TNSHarvester`
  - Auth: `api_key` (configured in `HARVESTERS` dict in settings)
  - Status: Configured in `TOM_HARVESTER_CLASSES` in `src/fomo/settings.py`

## Data Storage

**Databases:**
- SQLite3 (default)
  - Location: `src/fomo_db.sqlite3`
  - Client: Django ORM
  - Configuration: `DATABASES['default']` in `src/fomo/settings.py`
  - Production override: Configurable via `local_settings.py`

**File Storage:**
- Local filesystem (default)
  - Media root: `src/data/` (configurable via `MEDIA_ROOT`)
  - Static files: `src/_static/` (configurable via `STATIC_ROOT`)
  - Template directories: `src/templates/`

**Caching:**
- File-based cache (default)
  - Backend: `django.core.cache.backends.filebased.FileBasedCache`
  - Location: System temp directory (via `tempfile.gettempdir()`)
  - Configuration: `CACHES` in `src/fomo/settings.py`

## Authentication & Identity

**Auth Provider:**
- Django ModelBackend (custom)
  - Implementation: `django.contrib.auth.backends.ModelBackend`
  - Supports: Username/password authentication
  - Configuration: `AUTHENTICATION_BACKENDS` in `src/fomo/settings.py`

**Authorization:**
- Guardian - Object-level permissions
  - SDK/Client: `guardian.backends.ObjectPermissionBackend`
  - Supports: Row-level data permissions (configurable via `TARGET_PERMISSIONS_ONLY`)

**Registration:**
- tom_registration - Custom user registration system
  - Strategy: `open` (open registration, no approval required)
  - Configuration: `TOM_REGISTRATION` dict in `src/fomo/settings.py`
  - Auth backend: `django.contrib.auth.backends.ModelBackend`

## Observation Facilities

**LCO (Las Cumbres Observatory):**
- SDK/Client: `tom_observations.facilities.lco.LCOFacility`
- Portal: `https://observe.lco.global`
- Auth: `api_key` (configured in `FACILITIES` dict in settings)

- LCO Redirect:
  - SDK/Client: `tom_observations.facilities.lco_redirect.LCORedirectFacility`
  - Status: Alternative redirect facility for LCO

**Gemini (Gemini Observatory):**
- SDK/Client: `tom_observations.facilities.gemini.GEMFacility`
- Portals:
  - Gemini South: `https://139.229.34.15:8443`
  - Gemini North: `https://128.171.88.221:8443`
- Auth: API keys for GS and GN (configured in `FACILITIES['GEM']` in settings)
- Programs: GS and GN program IDs with standard/rapid designations

**SOAR (Southern Astrophysical Research Telescope):**
- SDK/Client: `tom_observations.facilities.soar.SOARFacility`
- Status: Configured in `TOM_FACILITY_CLASSES` in `src/fomo/settings.py`

**ESO (European Southern Observatory):**
- SDK/Client: `tom_eso.eso.ESOFacility`
- Status: Configured in `TOM_FACILITY_CLASSES` in `src/fomo/settings.py`

## Monitoring & Observability

**Error Tracking:**
- None detected - Not currently integrated

**Logs:**
- Console logging (StreamHandler)
  - Configuration: `LOGGING` in `src/fomo/settings.py`
  - Level: INFO for all loggers
  - Handler: Console output

**Code Coverage:**
- Codecov - CI/CD code coverage reporting
  - Integration: GitHub Actions workflow `testing-and-coverage.yml`
  - Token: `CODECOV_TOKEN` (GitHub secret)

## CI/CD & Deployment

**Hosting:**
- ReadTheDocs - Hosted documentation (https://fomo.readthedocs.io/)
  - Python 3.10 runtime
  - Sphinx builder with autoapi and RTD theme
  - Configuration: `.readthedocs.yml`

**CI Pipeline:**
- GitHub Actions
  - Workflows location: `.github/workflows/`
  - Testing workflow: `testing-and-coverage.yml` (runs on push to main, PRs)
  - Smoke test: `smoke-test.yml` (daily at 6:45 UTC, tests Python 3.10, 3.11, 3.12)
  - Pre-commit CI: `pre-commit-ci.yml`
  - Documentation build: `build-documentation.yml`
  - PyPI publishing: `publish-to-pypi.yml`

**Package Distribution:**
- PyPI - Python Package Index
  - Package name: `fomo`
  - Source: https://github.com/lsst-sssc/fomo

## Environment Configuration

**Required env vars (for Fink stream):**
- `FINK_CREDENTIAL_URL` - Kafka broker connection URL
- `FINK_CREDENTIAL_USERNAME` - Kafka authentication username
- `FINK_CREDENTIAL_GROUP_ID` - Kafka consumer group ID
- `FINK_TOPIC` - Kafka topic name
- `FINK_MAX_POLL_NUMBER` - Maximum alerts per poll request
- `FINK_TIMEOUT` - Poll timeout in seconds

**Optional env vars (for alert brokers):**
- `LASAIR_TOKEN` - LasAIR API authentication token

**Secrets location:**
- Environment variables (recommended practice)
- `local_settings.py` - Local Django settings override (not committed)
- `.env` files - Not currently used but supported via `os.getenv()`

**Configuration priority:**
1. Environment variables (via `os.getenv()`)
2. Hardcoded defaults in `src/fomo/settings.py`
3. Local overrides via `local_settings.py` (if exists)

## Security Configuration

**CSRF Protection:**
- Enabled via Django middleware `django.middleware.csrf.CsrfViewMiddleware`
- Configuration: `src/fomo/settings.py` line 77

**HTTPS:**
- X-Frame-Options protection via `django.middleware.clickjacking.XFrameOptionsMiddleware`
- Security middleware: `django.middleware.security.SecurityMiddleware`

**Password Policy:**
- Django built-in validators:
  - User attribute similarity check
  - Minimum length validation
  - Common password check
  - Numeric-only password rejection

**Permissions Model:**
- Target permissions only: `TARGET_PERMISSIONS_ONLY = True` (default)
- Default target visibility: `TARGET_DEFAULT_PERMISSION = 'OPEN'` (visible to all)
- Row-level permissions: Controlled via Django Guardian for observations and data products
- Auth strategy: `AUTH_STRATEGY = 'READ_ONLY'` (read access without login, write requires auth)

## Webhooks & Callbacks

**Incoming:**
- None detected - No webhook endpoints documented

**Outgoing:**
- Django Hooks system (internal)
  - `target_post_save` - Triggered on target save
  - `observation_change_state` - Triggered on observation state change
  - `data_product_post_upload` - Triggered on data product upload
  - `data_product_post_save` - Triggered on data product save
  - `multiple_data_products_post_save` - Triggered on multiple product save
  - Configuration: `HOOKS` dict in `src/fomo/settings.py`

---

*Integration audit: 2026-06-12*
