# Testing Patterns

**Analysis Date:** 2026-09-04

## Test Framework

**Runner:**
- Django test runner: `python manage.py test` is the **only functioning suite**
- Command: `python manage.py test` (run all tests)
- Command: `python manage.py test solsys_code` (app-specific tests)
- Command: `python manage.py test solsys_code.tests.test_telescope_runs.TestTelescopeRuns` (class-specific)

**Assertion Library:**
- Django's `TestCase` assertion methods: `assertEqual`, `assertIsInstance`, `assertRaises`, `assertLessEqual`, `assertAlmostEqual`, `assertIn`, `assertIsNone`, `assertFalse`, `assertTrue`
- `SimpleTestCase` for tests that don't need database access

**Legacy pytest config (DO NOT USE):**
- Configuration in `pyproject.toml` (`testpaths = ["tests", "src", "docs"]`) is leftover from LINCC project template
- `python -m pytest` does not collect Django app tests
- Do NOT add new tests to `tests/fomo/` — this suite will be removed

## Test File Organization

**Location:**
- Django app tests: `solsys_code/tests/`
- Sub-app tests: `solsys_code/solsys_code_observatory/tests/`

**Naming:**
- Files: `test_*.py` pattern (e.g., `test_telescope_runs.py`, `test_campaign_models.py`, `test_views.py`)
- Classes: `Test*` pattern (e.g., `TestTelescopeRuns`, `TestCampaignRun`, `TestSplitNumberUnitRegex`)
- Methods: `test_*` pattern with descriptive names (e.g., `test_get_site_returns_observatory`, `test_sun_event_sun`)

**File structure:**
```
solsys_code/
├── tests/
│   ├── __init__.py
│   ├── helpers.py                          # Common fixtures/helpers
│   ├── data/                               # Test data files
│   ├── test_telescope_runs.py
│   ├── test_campaign_models.py
│   ├── test_views.py
│   └── ... (one test_*.py per module)
├── telescope_runs.py                       # Module being tested
├── models.py
├── views.py
└── ...
```

## Test Structure

**Class-level setup:**
```python
class TestTelescopeRuns(TestCase):
    @classmethod
    def setUpTestData(cls) -> None:
        """Run once per test class. Ideal for expensive fixtures."""
        cls.user = get_user_model().objects.create_user(username='testuser')
        for obscode, fields in {
            '268': dict(name='Magellan Clay', lat=-29.0146, lon=-70.6926),
        }.items():
            Observatory.objects.update_or_create(obscode=obscode, defaults=fields)
```

**Instance-level setup:**
```python
def setUp(self) -> None:
    """Run before each test method. Use for instance-specific state."""
    self.precision = 6
    return super().setUp()
```

**Test method docstrings:**
```python
def test_sunset_sunrise_validation(self):
    """EPHEM-04: Las Campanas sun-event times for Jun 1/10/20/30 2026 match the skycalc reference within 2 min."""
    # Include requirement/design-doc references in docstrings
```

## Fixtures and Factories

**Django ORM fixtures:**
- Use `setUpTestData` for class-level, read-only fixtures (most efficient)
- Use `setUp` for instance-level fixtures that will be modified
- Use `objects.get_or_create()` or `objects.update_or_create()` for setup

**Factory objects:**
- `tom_targets.tests.factories.NonSiderealTargetFactory` for Target fixtures
- **CRITICAL:** Never use `SiderealTargetFactory` — FOMO is exclusively non-sidereal
- Example:
  ```python
  target = NonSiderealTargetFactory.create(
      name='test-target',
      type='NON_SIDEREAL',
      scheme='MPC_MINOR_PLANET'
  )
  ```

**Test data builders:**
- Extract complex fixture builders into helper functions
- Example from `test_backfill_lco_observations.py`:
  ```python
  def _request(request_id, target_name='Didymos', state='COMPLETED'):
      return {
          'id': request_id,
          'state': state,
          'windows': [{'start': '2026-07-01T00:00:00', 'end': '2026-07-02T00:00:00'}],
          'configurations': [_configuration(target_name=target_name)],
      }
  ```
- Example from `test_sync_gemini_observation_calendar.py`:
  ```python
  def _gem_parameters(prog='GS-2026A-T-999', obsid=None, ready='true'):
      params = {'prog': prog, 'obsid': obsid or ['MM'], 'ready': ready}
      return params
  ```

**Setting up related objects:**
```python
target = NonSiderealTargetFactory.create(name='test-target')
record = ObservationRecord.objects.create(
    observation_id='obs-123',
    target=target,
    user=self.user,
    facility='GEM',
    status='PENDING',
    parameters=_gem_parameters(),
)
```

## Mocking

**Framework:**
- `unittest.mock.patch` for mocking function/method calls
- `unittest.mock.MagicMock` for mocking objects
- `@patch` decorator for method-level patching
- Manual `patcher.start()` / `patcher.stop()` for test-method-level patching

**Pattern - decorator level:**
```python
@patch('solsys_code.management.commands.backfill_lco_observations.make_request')
def test_fetch_records(self, mock_make_request):
    mock_make_request.return_value = {...}
    # Test code
```

**Pattern - method level (in setUp):**
```python
def setUp(self):
    patcher = patch('tom_observations.facilities.lco.LCOFacility.get_observation_status')
    self.mock_get_obs = patcher.start()
    self.addCleanup(patcher.stop)
```

**What to mock:**
- External HTTP calls (LCO API, Gemini API, MPC API)
- Time-dependent behavior (current time, dates)
- Database queries to external services
- File I/O operations

**What NOT to mock:**
- Internal function calls (test the full flow)
- Django ORM queryset operations (use real database with TestCase)
- Observatory model lookups (use real database records)
- Custom ephem_utils functions (test the real implementation)

## Special Considerations

**The ~1.6 GB SPICE kernel download:**
- Importing `solsys_code.ephem_utils` triggers `fomo_furnish_spiceypy()` at module load
- This downloads ~1.6 GB of SPICE kernels to `~/.cache/sorcha/` on first use
- **Impact:** Any test that imports ephem_utils or views (which imports ephem_utils transitively) pays this cost
- **Workaround:** Tests that don't need ephemeris computation should avoid importing ephem_utils; use `unittest.mock.patch` to intercept calls instead

**Test exclusions:**
- `test_views.TestEphemeris` is excluded from CI because it segfaults in native ASSIST
- Management commands that depend on ephem_utils run with the full SPICE kernel cost

**Test configuration:**
- `@override_settings()` decorator to override Django settings per test:
  ```python
  @override_settings(FACILITIES=GEM_SETTINGS)
  class TestSyncGeminiObservationCalendar(TestCase):
  ```

**Database transactions:**
- Use `transaction.atomic()` to test IntegrityError handling:
  ```python
  with self.assertRaises(IntegrityError):
      with transaction.atomic():
          CampaignRun.objects.create(...)  # Duplicate constraint violation
  ```

## Error Testing

**Pattern - assertRaises with context:**
```python
def test_get_site_unknown(self):
    with self.assertRaises(Observatory.DoesNotExist):
        get_site('NoSuchTelescope')
```

**Pattern - check exception message:**
```python
with self.assertRaises(ValueError) as ctx:
    parse_run_line('Magellan IMACS 13-19 July')
self.assertIn('Magellan-Clay', str(ctx.exception))
self.assertIn('Magellan-Baade', str(ctx.exception))
```

**Pattern - CommandError in management commands:**
```python
from django.core.management import CommandError

with self.assertRaises(CommandError):
    call_command('load_telescope_runs', 'nonexistent_file.txt')
```

## Coverage

**Requirements:**
- No hard-coded coverage target enforced
- Coverage reported in GitHub workflows via `pytest-cov`

**Run coverage locally:**
- Via pre-commit: `pytest-check` hook in `.pre-commit-config.yaml` runs `python -m pytest --cov=./src --cov-report=html`
- Generates HTML report (not used in Django workflow)

**GitHub workflow coverage:**
- Runs `python -m pytest --cov=<package> --cov-report=xml`
- Uploads results to Codecov

## Test Types

**Unit tests:**
- Scope: Single function or method
- Example: `test_horizon_dip` tests the `horizon_dip()` function with various altitudes
- Location: Paired with the module being tested
- Database: Optional (use `SimpleTestCase` if not needed)

**Integration tests:**
- Scope: Multiple modules working together
- Example: `test_sun_event_succeeds_for_below_sea_level_site` tests Observatory lookup + sun_event computation
- Location: Paired with the primary module
- Database: Required (use `TestCase`)

**Management command tests:**
- Scope: Full command execution
- Example: `test_load_runs_from_file` in `test_load_telescope_runs.py`
- Pattern: `call_command('command_name', *args, **options)`
- Database: Required
- Mocking: Mock external APIs but not internal models

**E2E/Functional tests:**
- Framework: Django's `Client` for HTTP testing or `playwright` for headless browser testing
- Example: `test_bootstrap5_rendering.py` uses Playwright for form rendering
- Rare in this codebase; most tests are unit or integration

## Common Patterns

**Async testing (datetime/timezone):**
```python
from datetime import date, datetime, timedelta, timezone

def test_timezone_dst_resolution(self):
    """Verify timezone-aware datetime offsets across DST."""
    santiago = ZoneInfo('America/Santiago')
    self.assertEqual(
        datetime(2026, 6, 15, 12, tzinfo=santiago).utcoffset(),
        timedelta(hours=-4)
    )
```

**Time comparison with tolerance:**
```python
def _assert_time_close(self, computed: Time, expected_iso: str, max_seconds: float = 120.0) -> None:
    expected = datetime.fromisoformat(expected_iso).replace(tzinfo=timezone.utc)
    computed_dt = computed.to_datetime(timezone=timezone.utc)
    delta = abs((computed_dt - expected).total_seconds())
    self.assertLessEqual(
        delta, max_seconds,
        f'{computed_dt} not within {max_seconds}s of {expected}'
    )
```

**Nearly-equal numerical comparison:**
```python
self.assertAlmostEqual(clay.lat, -29.0146, self.precision)
self.assertAlmostEqual(horizon_dip(2402).to_value(u.deg), 1.44, delta=0.02)
```

**ORM setup/verification:**
```python
# Setup: use update_or_create for idempotent fixtures
Observatory.objects.update_or_create(
    obscode=obscode,
    defaults=fields
)

# Verify: count and field values
self.assertEqual(Observatory.objects.filter(obscode__in=['268', '269']).count(), 2)
self.assertEqual(clay.timezone, 'America/Santiago')
```

---

*Testing analysis: 2026-09-04*
