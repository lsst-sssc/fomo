# Codebase Concerns

**Analysis Date:** 2026-06-12

## Security Issues

**Hardcoded Debug and Secret Key in settings.py:**
- Issue: `DEBUG = True` and exposed `SECRET_KEY` directly in version control
- Files: `src/fomo/settings.py` (lines 27, 24)
- Impact: Enables remote code execution in production; secrets exposed if repo is public or compromised
- Current mitigation: None observed (must rely on `.gitignore` to exclude production settings)
- Recommendations:
  - Move `DEBUG` and `SECRET_KEY` to environment variables (use `os.getenv()` with validation)
  - Create a `local_settings_template.py` showing the pattern
  - Document in onboarding that `local_settings.py` (imported at line 381-383) must be gitignored
  - Consider using `python-decouple` or `django-environ` for configuration management

**Hardcoded API Keys in settings.py:**
- Issue: Empty strings for facility API keys (`LCO`, `GEM`, `TNS`, `LASAIR`) with inline defaults in ALERT_STREAMS
- Files: `src/fomo/settings.py` (lines 215-320)
- Impact: Credentials must be injected post-deployment; inline defaults in FINK_CREDENTIAL_* could leak error messages
- Recommendations:
  - Use `os.getenv()` with descriptive fallback errors instead of "set FINK_CREDENTIAL_URL value in environment"
  - Document all required environment variables in a `.env.example` file
  - Validate environment configuration at startup (e.g., in `apps.py` or a management command)

## Tech Debt

**Heavy SPICE Kernel Import in ephem_utils.py:**
- Issue: `fomo_furnish_spiceypy(cache_dir)` executes at module import (line 62), downloading ~1.6 GB on first run
- Files: `solsys_code/ephem_utils.py` (lines 55-69)
- Impact: Any import of this module (e.g., in Django tests, shell, or views) triggers the download; slows test suite startup; blocks execution if network unavailable
- Workaround documented: Cache is persisted in `~/.cache/sorcha/`
- Recommendations:
  - Defer kernel loading to first use (lazy initialization with a module-level `_kernels_loaded` flag)
  - Add a management command `python manage.py initialize_ephemeris` to pre-warm the cache
  - Document this in project onboarding (see GSD experiment note in `docs/design/gsd_experiment.rst`)

**Test Suite Split (pytest vs Django test):**
- Issue: Two independent test suites that don't collect the same tests
- Files: `tests/fomo/test_packaging.py` (pytest only); `solsys_code/tests/` and `solsys_code/solsys_code_observatory/tests/` (Django ORM tests only)
- Impact: CI runs only `pytest --cov=fomo` (line 35 of `.github/workflows/testing-and-coverage.yml`), misses Django app tests; developers may forget to run `./manage.py test solsys_code`
- Current state: Both test suites are documented in the GSD experiment (`docs/design/gsd_experiment.rst`, lines 63-64), but developers must remember to run both
- Recommendations:
  - Create a unified test runner (e.g., `test.sh` or a tox configuration) that runs both suites
  - Add a pre-commit hook to enforce both (currently relies on manual discipline)
  - Document the split explicitly in project README (not yet found)
  - Consider migrating Django tests to pytest for consistency

**Large Monolithic Views (views.py):**
- Issue: `solsys_code/views.py` is 629 lines; `solsys_code/tests/test_views.py` is 839 lines
- Files: `solsys_code/views.py`, `solsys_code/tests/test_views.py`
- Impact: Difficult to test individual features; high risk when refactoring ephemeris or form logic
- Recommendations:
  - Refactor views into smaller classes (one per form/feature: `EphemerisFormView`, `EphemerisDisplayView`, etc.)
  - Extract ephemeris generation logic into a service module (e.g., `solsys_code/services.py`)
  - Split test file into domain-specific suites (`test_ephemeris_views.py`, `test_ephemeris_forms.py`, etc.)

**Ephemeris Module Complexity (ephem_utils.py):**
- Issue: 458 lines mixing coordinate transforms, SPICE kernel management, Sorcha wrappers, and ASSIST integration
- Files: `solsys_code/ephem_utils.py`
- Impact: High test cost (must mock SPICE/ASSIST imports); unclear separation of concerns; any change risks breaking both observation math and kernel setup
- Recommendations:
  - Extract SPICE/kernel management into `solsys_code/kernel_manager.py` with lazy initialization
  - Create a high-level ephemeris service API in a dedicated module
  - Add type hints throughout (currently minimal)
  - Break coordinate transform helpers into a separate `solsys_code/coordinates.py` module

## Fragile Areas

**Observatory Site Handling:**
- Files: `solsys_code/solsys_code_observatory/models.py`, `solsys_code/solsys_code_observatory/utils.py`
- Why fragile: Observatory model depends on ERFA for coordinate transforms; parallax constant conversions are bidirectional but reversibility is not tested
- Safe modification: Add round-trip tests (lat/lon → parallax → lat/lon) to `solsys_code/solsys_code_observatory/tests/test_models.py`
- Test coverage: `test_models.py` (94 lines) only tests basic object creation; coordinate transform correctness is not verified

**Target to Ephemeris Conversion (convert_target_to_layup):**
- Files: `solsys_code/ephem_utils.py` (lines 72-120), `solsys_code/tests/test_ephem_utils.py` (lines 22-75)
- Why fragile: Mixes optional `sun_dict` parameter for testing vs. in-place Sun determination; special handling for comet nuclear/total magnitudes (documented in commit f75abb9); epoch conversions between MJD and JD
- Safe modification: Ensure all tests pass before changing orbital element conversions; verify against JPL Horizons for sample asteroids/comets
- Test coverage: Tests exist for asteroid H/G magnitudes and comet magnitudes, but only against hardcoded reference values; not parameterized across orbit types

**Form Parsing and Validation (ephem_form_submit):**
- Files: `solsys_code/views.py` (lines 100+), `solsys_code/forms.py`
- Why fragile: Regex parsing of free-text input (e.g., split_number_unit_regex); exception handling with `raise Exception` (line 94 of views.py) instead of proper HTTP responses
- Safe modification: Add input validation in form's `clean()` method; replace generic Exception with `django.core.exceptions.ValidationError`
- Test coverage: Unit tests for `split_number_unit_regex` added in commit 70e0c16, but form validation as a whole is tested at the view level

## Known Bugs / Incomplete Features

**Circular Middleware Issue (settings.py line 85):**
- Issue: `AuthStrategyMiddleware` is listed **twice** in `MIDDLEWARE` (lines 84-85)
- Files: `src/fomo/settings.py`
- Impact: Redundant execution; unlikely to cause functional bugs but adds noise and potential confusion
- Workaround: None necessary (duplicate does not break functionality)
- Fix: Remove one instance

**Django's `local_settings` Import Pattern (settings.py lines 380-383):**
- Issue: Bare `except ImportError: pass` swallows all import errors, masking typos or actual issues in `local_settings.py`
- Files: `src/fomo/settings.py`
- Impact: Mistakes in `local_settings.py` (e.g., syntax errors) are silently ignored; developers see "config not applied" instead of an error
- Recommendations:
  - Change to `except ImportError as e: raise RuntimeError(f"Failed to import local_settings: {e}") from e` to surface issues
  - Or conditionally log a warning if running in development mode

**Missing Docstring in MakeEphemerisView (views.py):**
- Issue: Class-level docstring exists but many methods (`get_initial`, `form_valid`, etc.) lack documentation
- Files: `solsys_code/views.py` (class MakeEphemerisView, lines 65+)
- Impact: Unclear flow; maintainers must read code to understand form initialization and submission path
- Recommendations: Add per-method docstrings following numpy/Django conventions

## Test Coverage Gaps

**Integration Tests for Ephemeris Generation:**
- What's not tested: End-to-end ephemeris workflow (form submission → database save → view render) for realistic Solar System objects
- Files: `solsys_code/tests/test_views.py`
- Risk: Regression in form submission or output formatting would not be caught until manual testing
- Priority: High (this is core functionality)
- Recommendation: Add integration tests for a sample minor planet and comet; mock SPICE/ASSIST to avoid kernel download in tests

**Data Product Upload and Processing:**
- What's not tested: The hook-based pipeline for photometry ingestion (referenced in `src/fomo/settings.py` lines 355-358 via `data_product_post_save`)
- Files: No test files identified for data product processing
- Risk: Broken photometry ingest would go unnoticed until production
- Priority: Medium-High
- Recommendation: Add tests for `tom_dataproducts` processor integration points

**Scout and Fink DataService Integration:**
- What's not tested: Ingest of alerts from Fink or JPL Scout; data transformation and target creation
- Files: `src/fomo/settings.py` (lines 285-298, ALERT_STREAMS; line 24, data_services()); actual code in imported packages
- Risk: Misconfiguration or upstream API changes would only be caught manually
- Priority: Medium
- Recommendation: Add mocked integration tests for FinkAlertStream and FinkDataService

**ASSIST Ephemeris Build:**
- What's not tested: ASSIST Nbody integration setup and execution (documented in GSD experiment as high-cost)
- Files: `solsys_code/ephem_utils.py` (lines 64-68, `create_assist_ephemeris`)
- Risk: Changes to Sorcha versions or ASSIST parameters would not be validated
- Priority: Low (marked as slow in GSD experiment)
- Recommendation: Create a separate optional test suite (`tests/integration/test_assist.py`) marked with pytest decorators to skip in CI by default

## Performance Bottlenecks

**SPICE Kernel Download on First Import:**
- Problem: Any use of ephemeris utilities triggers a ~1.6 GB download and initial setup
- Files: `solsys_code/ephem_utils.py` (module-level initialization, lines 62-69)
- Cause: Eagerly loading kernels at import time rather than on first use
- Improvement path: Implement lazy loading with a module-level initialization guard; add `manage.py initialize_ephemeris` for pre-warming

**Pandas DataFrame Operations in Ephemeris Calculation:**
- Problem: `add_magnitude()` and `add_sky_motion()` operate row-by-row in some cases (commits f75abb9, ced6df6 suggest iterative refinement)
- Files: `solsys_code/ephem_utils.py`, `solsys_code/tests/test_ephem_utils.py`
- Cause: Magnitude calculation for comets requires special handling (nuclear vs. total); vectorization not applied
- Improvement path: Profile `add_magnitude()` and `add_sky_motion()` on large ephemerides (100+ rows); consider numpy ufuncs if CPU-bound

**SQLite for Production:**
- Problem: `settings.py` line 118 defaults to SQLite (`django.db.backends.sqlite3` with `fomo_db.sqlite3`)
- Files: `src/fomo/settings.py` (lines 115-120)
- Impact: Concurrent write contention on production deployments; file locking under load
- Current state: Appropriate for development; must be overridden in production (likely via environment config)
- Recommendations:
  - Document in README that PostgreSQL is required for production
  - Add a check in `apps.py` to warn if DEBUG=True and SQLite is in use

## Scaling Limits

**Calendar Event Sync (Upcoming Feature):**
- Limit: `CalendarEvent` table will grow with each classical night and queue block; no retention policy yet
- Impact: `/calendar/` views may slow down after months of operation
- Scaling path: Add pagination or date filtering to calendar views; implement a management command to archive old events (6+ months)
- Files: Not yet implemented (design in `docs/design/telescope_runs_calendar.rst`); will affect `solsys_code` models and views

**Observatory Registry (solsys_code_observatory):**
- Current capacity: ~1000 MPC observatories (hardcoded list from MPC)
- Scaling: No concern at this scale
- Indexes: Lat/lon are indexed (models.py lines 40, 49); good for spatial queries

## Dependencies at Risk

**tomtoolkit Upcoming Major Version:**
- Risk: Migration to tomtoolkit 3.0 is in-flight (commits eaf75aa, 2b1a74a on tomtoolkit3-migration branch)
- Impact: Large codebase change; pre-release version used in development
- Migration plan: Merge tomtoolkit 3.0 migration PR (#38); bump requirement to `tomtoolkit>=3.0.0` once stable

**tom_fink >= 1.0.0:**
- Risk: Recent requirement (commit 1283557) may have breaking API changes
- Files: `pyproject.toml` line 22
- Impact: Fink dataservice ingest could fail on version mismatch
- Recommendations: Pin to specific minor version (e.g., `tom_fink>=1.0.0,<2.0.0`) once 1.x stabilizes

**sorcha and sbpy Compatibility:**
- Risk: sorcha depends on sbpy >= 0.6.0 (commit 4d8a59d); sbpy requires specific astropy versions
- Files: `pyproject.toml` line 26; Sorcha is pulled in by solsys_code
- Impact: Astropy 7.2.0+ compatibility requires sbpy >= 0.6.0; test matrix covers Python 3.10-3.12
- Recommendations: Monitor sbpy releases; add integration test for latest astropy on Python 3.12

**numpy 2.0 Compatibility:**
- Risk: Handled (atan2 compatibility workaround in ephem_utils.py lines 9-13), but other code may still use deprecated numpy APIs
- Files: `solsys_code/ephem_utils.py` (workaround for numpy<2.0), pyproject.toml (line 25, numpy>1.24)
- Impact: Will need to drop numpy<2.0 support once Python 3.9 is no longer tested
- Recommendations: Run ruff with NPY201 enabled (already in config, pyproject.toml line 99)

## Missing Critical Features

**No Environment Configuration Management:**
- Problem: Configuration scattered across `settings.py` and environment variables; no validation or documentation of required vars
- Impact: Deployment requires institutional knowledge; easy to misconfigure in production
- Recommendations:
  - Create `fomo/settings_base.py` and `fomo/settings_production.py` with explicit environment requirements
  - Add a `manage.py validate_config` command that checks all required env vars at startup
  - Create a `.env.example` with all required variables

**Observation Status Polling:**
- Problem: No automatic polling of LCO/Gemini observation statuses; users must manually trigger updates
- Files: No implementation of background polling (would be in `solsys_code` or a new Django app)
- Impact: Calendar view (#37) will show stale observation blocks
- Recommendations (future): Implement Celery task for periodic `update_observation_status()` calls; integrate with calendar sync

**Calendar Event Lifecycle Management:**
- Problem: Design doc (#37) notes "Terminal-failure policy for #3 (delete vs strike-through) to be confirmed" (line 327)
- Files: `docs/design/telescope_runs_calendar.rst`
- Impact: Feature will be incomplete until policy is decided and implemented
- Recommendations: Implement both behaviors as configurable options; default to deletion for now

## Architectural Concerns

**Tight Coupling Between Views and Ephemeris Utilities:**
- Issue: `views.py` directly imports from `ephem_utils.py`, which drags in heavy dependencies (SPICE, ASSIST, Sorcha)
- Files: `solsys_code/views.py` (lines 35-47), `solsys_code/ephem_utils.py`
- Impact: Testing views in isolation requires mocking the entire ephemeris stack
- Recommendations:
  - Create a high-level ephemeris service API that can be mocked independently
  - Use dependency injection or Django's service locator pattern for cleaner testing

**No Error Handling Standardization:**
- Issue: Mix of bare `Exception`, Django's `ValidationError`, and silent failures
- Examples: `raise Exception('Must provide target_id')` (views.py line 94); silently swallowed ImportError in settings.py line 382
- Impact: Inconsistent user feedback; difficult to surface errors in forms and API responses
- Recommendations:
  - Define a set of custom exceptions in `solsys_code/exceptions.py` (e.g., `EphemerisGenerationError`, `MissingTargetError`)
  - Use these throughout the app; map them to HTTP responses in views

## Minor Issues

**Untracked Ruff Ignores:**
- Issue: N-series (naming convention) rules are broadly ignored (pyproject.toml lines 112-118)
- Impact: Code does not enforce PEP8 naming consistently; OK per Rubin DM style guide, but unusual for a new project
- Recommendations: Document why in a comment (likely copied from Science Pipelines template)

**Missing Pre-commit Docstring Enforcement:**
- Issue: Ruff docstring rules (D101, D102, D103) are enforced but only in certain files; inconsistent coverage
- Files: `pyproject.toml` (lines 123-126)
- Impact: Some modules have full docstrings, others minimal
- Recommendations: Set a consistent standard (either enforce globally or disable globally)

---

*Concerns audit: 2026-06-12*
