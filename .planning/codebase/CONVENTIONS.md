# Coding Conventions

**Analysis Date:** 2026-06-12

## Naming Patterns

**Files:**
- Snake case for Python files (e.g., `test_ephem_utils.py`, `solsys_code_observatory`)
- Test files follow pattern: `test_*.py` (e.g., `test_models.py`, `test_views.py`, `test_utils.py`)
- Django app directories use descriptive snake_case with nested structures (e.g., `solsys_code/`, `solsys_code_observatory/`)

**Functions:**
- Snake case throughout (e.g., `split_number_unit_regex`, `convert_target_to_layup`, `add_magnitude`, `add_sky_motion`)
- Private/internal functions use leading underscore (e.g., `_translate_constraints`)
- Method names follow Django conventions: `get_*`, `form_valid`, `setUp`, `handle`

**Variables:**
- Snake case for all variables and parameters (e.g., `target_id`, `start_time`, `obscode`, `test_observatory`)
- Constants use UPPER_CASE (e.g., `AU_KM`, `SEC_PER_DAY`, `PI_OVER_2`, `MJD_TO_JD_CONVERSION`)
- Class attributes and properties follow snake case (e.g., `test_target`, `bary_vec`, `sun_dict`)

**Types:**
- Use modern Python type hints (Python 3.10+): `tuple[float, float]`, `dict[str, Any]`, `Optional[dict[str, Any]]`
- Return type annotations on methods: `def form_valid(self, form: EphemerisForm) -> HttpResponse:`
- Parameter type annotations where helpful: `def query(self, obscode: str, dbg: bool = False)`

**Classes:**
- PascalCase for class names (e.g., `Observatory`, `EphemerisForm`, `JPLSBDBQuery`, `FakeSorchaArgs`)
- Inner/nested classes allowed (e.g., `Meta` in Django models)

## Code Style

**Formatting:**
- Line length: 120 characters (enforced by `ruff` and `black`)
- Quote style: Single quotes preferred by ruff formatter (e.g., `'ephem_form.html'`)
- Target Python version: 3.10+

**Linting:**
- Tool: `ruff` for linting and formatting
- Configuration in `pyproject.toml`: `[tool.ruff]`
- Pre-commit hook runs `ruff --fix` and `ruff-format` on all Python files
- Ruff lint rules include: E (pycodestyle), W (warnings), F (Pyflakes), N (pep8-naming), UP (pyupgrade), B (bugbear), SIM (simplify), I (isort)
- Per-file ignores for tests: `D101`, `D102` (missing docstrings)
- Per-file ignores for migrations: `D100`, `D101`, `D102`, `D103`, `E501`, `RUF012`
- Exceptions to naming rules: `N802`, `N803`, `N806`, `N812`, `N813`, `N815`, `N816`, `N999` (allow some variations for scientific/Numpy compatibility)

## Import Organization

**Order:**
1. Standard library imports (`import logging`, `from datetime import timezone`, `from pathlib import Path`)
2. Third-party imports (`import numpy as np`, `from astropy import units as u`, `from django.test import TestCase`)
3. Local imports (`from solsys_code.views import ...`, `from .forms import EphemerisForm`)

**Path Aliases:**
- No path aliases defined in this project; relative imports use dot notation (e.g., `from .forms import`, `from .ephem_utils import`)
- Absolute imports from installed packages: `from tom_targets.models import Target`

**Isort Configuration:**
- Profile: `black`
- Line length: 120

## Error Handling

**Patterns:**
- Use generic `try/except` blocks for expected failures (e.g., `ValueError` when parsing time strings)
- Custom exceptions not extensively used; rely on built-in exceptions and Django exceptions
- Logging at `debug` level for expected failures: `logger.debug(f'Query failed with status {resp.status_code}')`
- Raise generic `Exception` for invariant violations (e.g., `raise Exception('Must provide target_id')`)

Example from `views.py`:
```python
try:
    start_time = Time(start_time, scale='utc')
except ValueError:
    start_time = Time.now()
    start_time = Time(start_time.datetime.replace(hour=0, minute=0, second=0, microsecond=0), scale='utc')
```

## Logging

**Framework:** `logging` (standard library)

**Patterns:**
- Get logger with `__name__`: `logger = logging.getLogger(__name__)`
- Log at `debug` level for diagnostic info: `logger.debug('No data found in results')`
- Test files can disable logging during test runs: `logging.disable(logging.CRITICAL)`
- Use f-strings for log messages: `logger.debug(f'Query failed with status {resp.status_code}')`

## Comments

**When to Comment:**
- Comment non-obvious algorithmic steps (e.g., "Convert from heliocentric->barycentric using the Sun's position")
- Comment constants and their meaning (e.g., "Speed of light in km/s")
- Comment field meanings in data structures (e.g., chi-square values, degrees of freedom)
- Use comments to explain the "why" not the "what" (code should be readable, comments explain intent)
- Block comments above code sections that need context

**JSDoc/TSDoc:**
- Not used (Python project, not TypeScript)
- Docstrings use Google-style format with `Args:`, `Returns:`, `Raises:` sections

## Docstring Style

**Module and Function Docstrings:**
- Google-style docstrings (not NumPy style, despite presence of NumPy code)
- Example from `ephem_utils.py`:
```python
def convert_target_to_layup(target, sun_dict=None):
    """Converts a `Target` to a numpy array in format needed for 'layup'

    Args:
        target (tom_targets.model.Target): Target
        sun_dict (dict): [Optional] A dict with a key of a JD_TDB pointing
            at a dict of {x,y,z,vx,vy,vz} for position and velocity of the
            Sun to override the internal rebound determination

    Returns:
        output (numpy structured array): Data converted to layup input format
    """
```

- Class docstrings: Simple one-liner (e.g., `"""View for making an ephemeris"""`)
- Method docstrings: Include Parameters and Returns sections
- One-liner functions may skip docstrings if name is self-explanatory

## Function Design

**Size:** 
- Methods typically 10-50 lines
- Longer methods acceptable for view handlers (50-100+ lines) due to Django boilerplate
- Extract complex logic into helper functions

**Parameters:** 
- Use keyword arguments for optional form parameters
- Type hints on parameters are encouraged
- Default parameters for optional behavior (e.g., `sun_dict=None`)

**Return Values:**
- Use type hints for return values
- Return `HttpResponse` from views
- Return `Optional[...]` for nullable types
- Tuples return multiple values with type hints: `-> tuple[float, float, float]`

## Module Design

**Exports:**
- Modules export all public functions and classes
- No `__all__` definitions observed; relies on convention (no leading underscore = public)
- Internal/private use indicated by leading underscore

**Barrel Files:**
- No barrel files (index-style `__init__.py`) in use
- Package `__init__.py` files are typically empty or minimal

## Code Quality Standards

**Docstring Requirements (per ruff):**
- `D101`: Missing docstring in public class (enforced except in tests)
- `D102`: Missing docstring in public method (enforced except in tests)
- `D103`: Missing docstring in public function
- Test files (`**/tests/*`) exempt from `D101`, `D102` requirements

**No Global State:**
- Avoid module-level mutable state
- Exception: `ephem_utils.py` loads and caches SPICE ephemeris kernels at module load time (acceptable for initialization)

---

*Convention analysis: 2026-06-12*
