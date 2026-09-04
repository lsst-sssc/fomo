# Coding Conventions

**Analysis Date:** 2026-09-04

## Naming Patterns

**Files:**
- Snake case: `test_telescope_runs.py`, `ephem_utils.py`, `sync_lco_observation_calendar.py`
- Test files: `test_*.py` pattern (e.g., `test_models.py`, `test_views.py`)
- Django app directories: Descriptive snake_case with nested structures (e.g., `solsys_code/`, `solsys_code_observatory/`)

**Functions:**
- Snake case throughout (e.g., `split_number_unit_regex`, `convert_target_to_layup`, `add_magnitude`, `sun_event`)
- Private/internal functions use leading underscore (e.g., `_translate_constraints`, `_local_noon_utc`, `_solar_altitude`)
- Method names follow Django conventions: `get_*`, `form_valid`, `setUp`, `handle`, `setUpTestData`

**Variables and Parameters:**
- Snake case for all (e.g., `target_id`, `start_time`, `obscode`, `test_observatory`, `sunset`)
- Astronomical variable names allowed per Rubin DM style (e.g., `H`, `G`, `RA_deg`) due to N8xx naming rule exceptions

**Constants:**
- UPPER_CASE (e.g., `AU_KM`, `SEC_PER_DAY`, `PI_OVER_2`, `MJD_TO_JD_CONVERSION`, `NEEDS_REVIEW_NAME_PREFIX`)

**Classes:**
- PascalCase (e.g., `Observatory`, `EphemerisForm`, `JPLSBDBQuery`, `FakeSorchaArgs`, `ParsedRun`, `TestTelescopeRuns`)
- Inner/nested classes allowed (e.g., `Meta` in Django models, `ApprovalStatus` in CampaignRun)

**Class Attributes:**
- Snake case (e.g., `test_target`, `bary_vec`, `sun_dict`, `obscode`, `is_verified`)

## Code Style

**Formatting:**
- Line length: 120 characters (enforced by `ruff` v0.2.1)
- Quote style: Single quotes (e.g., `'ephem_form.html'`, `'NON_SIDEREAL'`)
- Target Python version: 3.10+
- Ruff version: 0.2.1 (pinned in `.pre-commit-config.yaml`)

**Tool Configuration:**
- Tool: `ruff` for linting and formatting
- Configuration in `pyproject.toml`: `[tool.ruff]` section
- Pre-commit hook runs `ruff --fix` and `ruff-format` on all Python files

**Ruff Rules:**
- Selected: E (pycodestyle), W (warnings), F (Pyflakes), N (pep8-naming), UP (pyupgrade), B (bugbear), SIM (flake8-simplify), I (isort), D101/D102/D103 (docstrings)
- Per-file ignores:
  - Test files: `D101`, `D102` (missing docstrings)
  - Migrations: `D100`, `D101`, `D102`, `D103`, `E501`, `RUF012`
  - Apps files: `D101`, `D102`
- Ignored rules (Rubin DM style): `N802`, `N803`, `N806`, `N812`, `N813`, `N815`, `N816`, `N999` (allow variations for scientific/NumPy compatibility)

## Type Hints

**Style:**
- Modern Python 3.10+ syntax required: `tuple[float, float]`, `dict[str, Any]`, `Optional[dict[str, Any]]`
- No legacy `Tuple[...]`, `Dict[...]`, `Union[...]` syntax
- Return type annotations on all public functions: `def form_valid(self, form: EphemerisForm) -> HttpResponse:`
- Parameter type annotations where helpful: `def query(self, obscode: str, dbg: bool = False)`
- Use `u.Quantity` for astropy units (e.g., `def horizon_dip(altitude_m: float) -> u.Quantity`)
- Tuple returns with type hints: `-> tuple[Time, Time]`, `-> tuple[str, str]`

## Import Organization

**Order:**
1. Standard library (`datetime`, `re`, `logging`, `json`, `pathlib`)
2. Third-party packages (`django`, `astropy`, `numpy`, `rebound`, `spiceypy`)
3. Relative imports from current app (`. models`, `. forms`, `. ephem_utils`)

**Absolute imports from packages:**
- `from django.test import TestCase`
- `from tom_targets.models import Target`
- `from tom_targets.tests.factories import NonSiderealTargetFactory`

**Relative imports within app:**
- `from .forms import EphemerisForm`
- `from .ephem_utils import horizon_dip`
- `from solsys_code.telescope_runs import get_site, sun_event`

**Path aliases:**
- No path aliases defined in this project; absolute imports and relative dot notation used

## Error Handling

**Pattern:**
- Use generic `try/except` blocks for expected failures
- `KeyError` when dictionary lookup fails: wrap in `Observatory.DoesNotExist` or similar
- `ValueError` when constraint parsing fails or invalid arguments provided
- Custom exceptions not extensively used; rely on built-in exceptions and Django exceptions
- `IntegrityError` for database constraint violations (wrapped in `transaction.atomic()`)

**Example:**
```python
try:
    obscode = SITES[name]
except KeyError as exc:
    raise Observatory.DoesNotExist(f'No site registered in SITES for telescope {name!r}') from exc
```

**Raising exceptions:**
- Raise generic `Exception` for invariant violations: `raise Exception('Must provide target_id')`
- Raise `ValueError` for invalid input: `raise ValueError(f'Invalid "is defined" constraint: {c}')`
- Raise `CommandError` in management commands: `from django.core.management.base import CommandError`

## Logging

**Setup:**
- Get logger with `__name__`: `logger = logging.getLogger(__name__)`
- Avoid logging in test runs: `logging.disable(logging.CRITICAL)` at module top for test files

**Level:**
- Log at `debug` level for diagnostic/expected failures: `logger.debug(f'Query failed with status {resp.status_code}')`
- Use f-strings for messages: `logger.debug(f'No data found in results')`

**Sorcha logger:**
- Suppress verbose Sorcha logs in module-level setup:
  ```python
  sorcha_logger = logging.getLogger('sorcha.ephemeris.simulation_setup')
  sorcha_logger.setLevel(logging.WARNING)
  ```

## Comments

**When to comment:**
- Comment non-obvious algorithmic steps (e.g., "Convert from heliocentric→barycentric using the Sun's position")
- Comment constants and their meaning (e.g., "Speed of light in km/s")
- Comment field meanings in data structures
- Use comments to explain the "why" not the "what" (code should be readable)
- Block comments above code sections that need context

**Example:**
```python
# dip = 1.76 arcmin * sqrt(altitude in metres).
# This is the Nautical Almanac dip formula (terrestrial refraction k~1/6
# folded into the spherical-geometry estimate dip ~ sqrt(2h/R), R=6371 km)
def horizon_dip(altitude_m: float) -> u.Quantity:
```

## Docstring Style

**Format:**
- Google-style docstrings (not NumPy style)
- Sections: `Args:`, `Returns:`, `Raises:`

**Class docstrings:**
- Simple one-liner allowed: `"""View for making an ephemeris"""`
- Full docstrings for complex classes explaining purpose and behavior

**Function/Method docstrings:**
- Include Args and Returns sections
- Include Raises section for documented exceptions
- One-liner functions may skip docstrings if name is self-explanatory

**Example:**
```python
def get_site(name: str) -> Observatory:
    """Resolves a telescope name to its Observatory record.

    Args:
        name: Telescope name, a key of SITES (e.g. 'Magellan-Clay').

    Returns:
        Observatory: the observatory record for this telescope's site.

    Raises:
        Observatory.DoesNotExist: if name is not a key in SITES, or no
            Observatory record exists for the resolved MPC obscode.
    """
```

## Function Design

**Size:**
- Typical: 10–50 lines
- Longer methods acceptable for view handlers (50–100+ lines) due to Django boilerplate
- Extract complex logic into helper functions

**Parameters:**
- Type hints on parameters encouraged
- Default parameters for optional behavior (e.g., `sun_dict=None`)
- Use keyword arguments for optional form parameters

**Return values:**
- Use type hints for return values
- Return `HttpResponse` from views
- Return `Optional[...]` for nullable types
- Tuples return multiple values with type hints: `-> tuple[float, float, float]`

## Module Design

**Exports:**
- Modules export all public functions and classes
- No `__all__` definitions observed; relies on convention (no leading underscore = public)
- Internal/private use indicated by leading underscore

**Barrel files:**
- No barrel files (index-style `__init__.py`) in use
- Package `__init__.py` files typically empty or minimal

**Module-level state:**
- Avoid module-level mutable state
- Exception: `ephem_utils.py` loads and caches SPICE ephemeris kernels at module load time (`fomo_furnish_spiceypy()`), acceptable for initialization
- Module-level constants like `SITES` dict documented above their definition

## Code Quality Standards

**Docstring enforcement:**
- `D101`: Missing docstring in public class (enforced except in tests)
- `D102`: Missing docstring in public method (enforced except in tests)
- `D103`: Missing docstring in public function
- Test files (`**/tests/*`) exempt from `D101`, `D102` requirements

**Code organization:**
- One class per file when possible (e.g., `models.py` contains model definitions, `forms.py` contains form classes)
- Related functions grouped with clear purpose comments
- Constants defined at module top before functions

**Validation:**
- Form validation via `clean()` method in Django forms
- Model-level validation via `clean()` method
- Database-level validation via `ValidationError`, `IntegrityError`

---

*Convention analysis: 2026-09-04*
