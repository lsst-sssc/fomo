# Testing Patterns

**Analysis Date:** 2026-06-12

## Test Framework Overview

This project uses **two separate test suites** that serve different parts of the codebase:

1. **pytest** (for most code) — Run with `python -m pytest`
2. **Django test runner** (for solsys_code Django app) — Run with `./manage.py test`

Each suite has different configuration and serves a specific purpose. See [Test Runners](#test-runners) below.

## Test Runners

### pytest Suite

**Command:** `python -m pytest`

**Configuration file:** `pyproject.toml`

**Test paths (from `[tool.pytest.ini_options]`):**
```
testpaths = ["tests", "src", "docs"]
```

**Coverage command:**
```bash
python -m pytest --cov=./src --cov-report=html
```

**Purpose:**
- Unit tests for non-Django code
- Package/module tests
- Ephemeris utility tests (most numerical code)

**Test files in pytest suite:**
- `tests/fomo/test_packaging.py` — Package version verification
- `solsys_code/solsys_code_observatory/tests/test_models.py` — Observatory model tests
- `solsys_code/solsys_code_observatory/tests/test_utils.py` — Observatory utilities
- `solsys_code/solsys_code_observatory/tests/test_views.py` — Observatory view tests
- `solsys_code/tests/test_ephem_utils.py` — Ephemeris calculations and conversions
- `solsys_code/tests/test_views.py` — Ephemeris view and JPL SBDB query tests

### Django Test Runner Suite

**Command:** `./manage.py test`

**Purpose:**
- Specifically for `solsys_code/` app tests
- Django model and view integration
- **Important:** First run downloads and caches 1.6 GB of SPICE ephemeris kernels to `~/.cache/sorcha/`
- Subsequent runs use cached kernels

**Note on kernel download:**
- Located: `solsys_code/ephem_utils.py` calls `fomo_furnish_spiceypy(cache_dir)` at module import time
- This is why `solsys_code/tests/test_ephem_utils.py` tests marked with `@tag('spiceypy')` may be slow on first run
- The test marked `@tag('spiceypy')` specifically tests sun ephemeris determination using SPICE kernels
- Most other tests use provided test data (`sun_dict` parameter) to avoid kernel dependency

## Test File Organization

**Location & Naming:**
- Test files are co-located with source code (same directory structure)
- Naming pattern: `test_*.py` files contain test classes and functions
- Each app/module has its own `tests/` subdirectory with `__init__.py`

**Structure by location:**
```
solsys_code/
├── tests/
│   ├── __init__.py
│   ├── test_ephem_utils.py       # Ephemeris utility tests
│   ├── test_views.py              # Ephemeris view and query tests
│   └── data/                       # Test data files
├── solsys_code_observatory/
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_models.py         # Observatory model tests
│   │   ├── test_utils.py          # Observatory utility tests
│   │   └── test_views.py          # Observatory view tests
tests/
├── fomo/
│   ├── conftest.py                # pytest fixtures (currently empty)
│   └── test_packaging.py          # Package version test
```

## Test Structure & Organization

### Class-Based Tests

Tests use Django's `TestCase` and `SimpleTestCase` classes:

```python
from django.test import TestCase, SimpleTestCase, Client
from tom_targets.models import Target
from solsys_code.solsys_code_observatory.models import Observatory

class TestEphemeris(TestCase):
    """Test ephemeris generation views"""
    
    def setUp(self):
        """Set up test fixtures before each test"""
        self.test_observatory, created = Observatory.objects.get_or_create(
            obscode='K93',
            name='Sutherland-LCO Dome C',
            lat=-32.380667412,
            lon=+20.81011,
            altitude=1808.33,
        )
        self.test_target, created = Target.objects.get_or_create(
            name='33933',
            type='NON_SIDEREAL',
            permissions='PUBLIC',
            scheme='MPC_MINOR_PLANET',
            # ... orbital elements ...
        )
        self.client = Client()
    
    def test_K93(self):
        """Test ephemeris generation for specific observatory"""
        expected_result = f'Ephemeris for {self.test_target.name} at  ({self.test_observatory.obscode})'
        response = self.client.get(
            reverse('ephem', kwargs={'pk': self.test_target.pk}) + f'?obscode={self.test_observatory.obscode}'
        )
        self.assertInHTML(expected_result, response.content.decode())
```

**Pattern observations:**
- Use `setUp(self) -> None` for test fixtures
- Create test objects using Django ORM: `Model.objects.get_or_create(...)`
- Store fixtures as instance variables (e.g., `self.test_observatory`, `self.test_target`, `self.client`)
- Test method names describe what is being tested: `test_K93`, `test_no_site_given`, `test_int1`
- Use `self.assertEqual()`, `self.assertInHTML()`, `self.assertAlmostEqual()` for assertions

### Simple (Non-Database) Tests

For pure functions without database dependencies, use `SimpleTestCase`:

```python
from django.test import SimpleTestCase
from numpy.testing import assert_almost_equal

class TestAddMagnitude(SimpleTestCase):
    def test_asteroid_no_default_G(self):
        expected_mags = [17.052, 16.838]
        obs_df = pd.DataFrame({
            'epoch_UTC': ['2025-08-21 00:00:00', '2025-09-09 00:00:00'],
            'Range_LTC_au': np.array([2.068039560205, 1.982726563532]),
            'Helio_LTC_au': np.array([3.010816010789, 2.969972029686]),
            'phase_deg': np.array([8.4394, 4.7247]),
        })
        obs_df = add_magnitude(obs_df, 12.79, 0.6)
        self.assertIn('APmag', obs_df.columns)
        assert_almost_equal(expected_mags, obs_df['APmag'], 3)
```

## Mocking

**Framework:** `unittest.mock` (standard library)

**Patterns:**
- Use `@patch()` decorator for method-level mocking
- Use `MagicMock()` and `Mock()` for creating mock objects
- Use `mock_response.json.return_value = {...}` to set mock return values

**Example from `test_utils.py`:**
```python
from unittest.mock import MagicMock, patch

class TestMPCObscodeFetcher(TestCase):
    @patch('requests.get')
    def test_query_failure_invalid_code(self, mock_get):
        """test query of invalid code"""
        mock_response = MagicMock()
        mock_response.status_code = 501
        mock_response.ok = False
        mock_response.json.return_value = {'error': 'input_error', 'message': 'Malformed input: bad obscode'}
        mock_response.content = b'{\n  "error": "input_error",\n  "message": "Malformed input: bad obscode\n}\n'
        mock_get.return_value = mock_response
        
        result = self.fetcher.query('FOO')
        self.assertEqual(self.bad_input_resp, result)
        self.assertIsNone(self.fetcher.obs_data)
```

**What to Mock:**
- External HTTP requests (`requests.get`)
- Database calls (when testing logic independent of database)
- External API responses

**What NOT to Mock:**
- Django ORM queries (use `TestCase` which provides test database)
- Internal functions (test through public API)
- Libraries like numpy/astropy (use real objects)

## Fixtures and Factories

**Test Data Pattern:**
Tests use inline fixtures created in `setUp()` with Django ORM:

```python
def setUp(self):
    self.test_target, created = Target.objects.get_or_create(
        name='33933',
        type='NON_SIDEREAL',
        permissions='PUBLIC',
        scheme='MPC_MINOR_PLANET',
        epoch_of_elements=61000.0,
        mean_anomaly=342.8987983972185,
        arg_of_perihelion=197.2440098291647,
        # ... more fields ...
    )
```

**Test Data Files:**
- `solsys_code/tests/data/` — Test data files for ephemeris tests
- Data loaded using `importlib.resources`: `from importlib.resources import files`

**No factory libraries used** (e.g., factory_boy); data created directly in setUp

## Test Utilities

**Assertion Libraries:**
- Django assertions: `self.assertEqual()`, `self.assertIn()`, `self.assertInHTML()`, `self.assertRaises()`
- NumPy assertions: `assert_almost_equal()`, `assert_array_almost_equal()` (from `numpy.testing`)
- Python unittest: `self.assertTrue()`, `self.assertFalse()`

**Test Tags:**
- `@tag('spiceypy')` — Marks tests requiring SPICE kernel downloads (slow on first run)
- Example: `solsys_code/tests/test_ephem_utils.py:test_sun_ephemeris` is tagged because it needs ephemeris kernels

## Coverage

**Requirements:** Not explicitly enforced (no coverage threshold in config)

**View Coverage:**
```bash
python -m pytest --cov=./src --cov-report=html
```

Generates HTML coverage report in `htmlcov/` directory.

**Omitted from coverage:**
- `src/fomo/_version.py` (generated by setuptools_scm)

## Test Types

### Unit Tests

**Scope:** Individual functions and methods in isolation

**Examples:**
- `test_ephem_utils.py:TestConvertTargetToLayup` — Tests orbital element conversion
- `test_utils.py:TestMPCObscodeFetcher` — Tests observatory code fetching
- `test_views.py:TestSplitNumberUnitRegex` — Tests utility function for parsing

**Approach:** Use `SimpleTestCase` when no database needed; `TestCase` when database interactions required.

### Integration Tests

**Scope:** Multiple components working together

**Examples:**
- `test_views.py:TestEphemeris` — Tests form submission, view routing, and rendering
- `test_views.py:TestJPLSBDBQuery` — Tests JPL query parsing and target creation
- View tests use Django test client: `self.client.get()`, `self.client.post()`

**Database:** Uses test database (TransactionTestCase or TestCase)

### End-to-End Tests

**Status:** Not extensively used in this codebase

**Note:** Some view tests function as E2E tests via Django client

## Logging in Tests

**Pattern:**
```python
import logging

## Silence logging during tests
logging.disable(logging.CRITICAL)
```

Place at module level in test files to suppress verbose output during test runs.

## Common Test Patterns

### Testing Django Views

```python
from django.test import Client, TestCase
from django.urls import reverse

class TestEphemeris(TestCase):
    def test_view(self):
        response = self.client.get(reverse('ephem', kwargs={'pk': self.test_target.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertInHTML('expected string', response.content.decode())
```

### Testing with Provided Data (Avoiding Kernels)

```python
from collections import namedtuple

class TestConvertTargetToLayup(TestCase):
    def setUp(self) -> None:
        # ... create target ...
        # Provide sun position/velocity to avoid SPICE kernel lookup
        Sun = namedtuple('Sun', 'x y z vx vy vz')
        self.sun_dict = {epochJD_TDB: Sun(x=..., y=..., z=..., vx=..., vy=..., vz=...)}
    
    def test_provided_sun_dict(self):
        # Pass sun_dict to avoid kernel download
        converted = convert_target_to_layup(self.target, self.sun_dict)
        for name, j in zip(converted.dtype.names[2:8], range(6), strict=False):
            assert_almost_equal(converted[name], self.bary_vec[j], 8)
```

### Testing Async/Numerical Calculations

```python
from numpy.testing import assert_almost_equal

# Test with precision parameter (decimal places)
assert_almost_equal(expected_value, actual_value, 8)  # 8 decimal places
```

### Testing HTTP Responses

```python
response = self.client.get(url)
self.assertEqual(response.status_code, 200)
self.assertInHTML(expected_html, response.content.decode())
# Or for JSON:
data = response.json()
self.assertEqual(data['key'], expected_value)
```

### Testing Error Conditions

```python
from django.test import TestCase

class TestObservatory(TestCase):
    def test_creation_nocode(self):
        """Test that Observatory requires obscode field"""
        with self.assertRaises(IntegrityError):
            Observatory.objects.create(name='wrong')
```

## Pre-commit Test Execution

Tests are run via pre-commit hook defined in `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: pytest-check
      name: Run unit tests
      description: Run unit tests with pytest.
      entry: bash -c "if python -m pytest --co -qq; then python -m pytest --cov=./src --cov-report=html; fi"
      language: system
      pass_filenames: false
      always_run: true
```

This hook:
1. Checks if tests can be collected (`--co`)
2. If successful, runs tests with coverage reporting

## Notes on SPICE Kernel Caching

**First run behavior:**
- Tests that need SPICE kernels (tagged `@tag('spiceypy')`) will download ~1.6 GB to `~/.cache/sorcha/`
- This happens at module import time in `solsys_code/ephem_utils.py`
- Line 62: `fomo_furnish_spiceypy(cache_dir)` is called at module level

**Subsequent runs:**
- Kernels are cached, so subsequent test runs are fast
- No re-download occurs unless cache is cleared

**Avoiding kernel requirement:**
- Pass test data (like `sun_dict`) to functions instead of letting them compute from kernels
- Most tests do this to avoid the download and speed up test execution

---

*Testing analysis: 2026-06-12*
