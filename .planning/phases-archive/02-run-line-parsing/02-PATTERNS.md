# Phase 2: Run Line Parsing - Pattern Map

**Mapped:** 2026-06-13
**Files analyzed:** 2
**Analogs found:** 2 / 2

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|----------------|
| `solsys_code/telescope_runs.py` (add `parse_run_line()`, `ParsedRun`) | utility | transform | `solsys_code/views.py:_translate_constraints` (parsing role) + `telescope_runs.py` itself (style) | role-match |
| `solsys_code/tests/test_telescope_runs.py` (add parser tests) | test | transform | same file, existing test class | exact |

## Pattern Assignments

### `solsys_code/telescope_runs.py::parse_run_line` (utility, transform)

**Analog (style/docstrings):** `solsys_code/telescope_runs.py` — `sun_event()`, `horizon_dip()`, `get_site()`

**Module-level constant pattern** (lines 13-20, `SITES` dict):
```python
# Maps telescope name to MPC observatory code. `Observatory` (looked up via
# get_site()) remains the single source of truth for location and timezone.
SITES = {
    'Magellan-Clay': '268',
    'Magellan-Baade': '269',
    'NTT': '809',
    'FTS': 'E10',
}
```
Follow this convention for the new known-status set, e.g.:
```python
# Known classical-schedule status words/phrases (case-insensitive), per
# docs/design/telescope_runs_calendar.rst "Classical Run Input Format".
KNOWN_STATUSES = {'allocation', 'proposed', 'confirmed', 'cancelled', 'not confirmed'}
```

**Function signature + docstring pattern** (`get_site`, lines 23-40):
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
    try:
        obscode = SITES[name]
    except KeyError as exc:
        raise Observatory.DoesNotExist(f'No site registered in SITES for telescope {name!r}') from exc
    return Observatory.objects.get(obscode=obscode)
```
This is the direct template for `parse_run_line`'s signature/docstring shape:
`Args:` / `Returns:` / `Raises: ValueError: ...` with `f-string` messages that
include the offending value via `!r`.

**Validation/raise pattern** (`horizon_dip`, lines 53-64):
```python
def horizon_dip(altitude_m: float) -> u.Quantity:
    """...
    Raises:
        ValueError: if altitude_m is None or negative.
    """
    if altitude_m is None or altitude_m < 0:
        raise ValueError(f'altitude_m must be a non-negative number, got {altitude_m!r}')
    return 1.76 * sqrt(altitude_m) * u.arcmin
```
Use this `if <bad condition>: raise ValueError(f'...got {value!r}')` shape for
each `parse_run_line` failure mode (bad telescope, bad status, bad date range).

**Multi-branch raise pattern with descriptive context** (`sun_event`, lines
163-189): shows the house style for raising `ValueError` with a multi-line
f-string that lists *all* relevant context (site, date, kind, computed
values) — model the ambiguous-telescope error (D-01, listing candidate SITES
keys) on this:
```python
    if not site.timezone:
        raise ValueError(
            f'Observatory {site.short_name!r} (obscode={site.obscode}) has no timezone set; '
            'set Observatory.timezone (IANA name, e.g. "America/Santiago") before calling sun_event().'
        )
    ...
    if len(crossings) != 2:
        raise ValueError(
            f'Expected 2 sun-event crossings for {site.short_name} on {date} '
            f'(kind={kind!r}), got {len(crossings)}: {crossings}. '
            'This can happen at high latitudes when the sun never sets or never '
            'reaches the requested threshold (e.g. midnight sun or no astronomical darkness).'
        )
```

---

### `solsys_code/telescope_runs.py::ParsedRun` (model/dataclass)

No existing dataclass analog in this module or `solsys_code/`; use a plain
`@dataclass` with typed fields per D-03, following the project's modern
type-hint convention seen throughout `telescope_runs.py` (e.g.
`tuple[Time, Time]`, `list[Time]`):
```python
from dataclasses import dataclass

@dataclass
class ParsedRun:
    """Structured result of parse_run_line().

    Attributes:
        telescope: resolved SITES key (e.g. 'NTT').
        instrument: instrument name as it appears in the run line.
        status: lowercase status word/phrase (defaults to 'allocation').
        year: four-digit year.
        month: month number (1-12).
        day1: first day of the run (inclusive).
        day2: last day of the run (inclusive).
    """
    telescope: str
    instrument: str
    status: str
    year: int
    month: int
    day1: int
    day2: int
```

---

### Free-text parsing precedent: `solsys_code/views.py::JPLSBDBQuery._translate_constraints`

**Analog for:** structuring `parse_run_line`'s internal parsing logic
(regex-driven tokenization, raise `ValueError` on unrecognized formats).

**Regex constant pattern** (`views.py` lines 404-415):
```python
    _CHAIN_PATTERN = re.compile(
        r"""
        ^\s*
        (?P<a>.+?)\s*
        (?P<op1><=|<|>=|>)\s*
        (?P<field>[A-Za-z_][A-Za-z0-9_\.]*)\s*
        (?P<op2><=|<|>=|>)\s*
        (?P<b>.+?)\s*
        $
        """,
        re.VERBOSE,
    )
```
Use `re.compile(..., re.VERBOSE)` with named groups for the run-line regex(es)
(e.g. separate patterns for `9-13 July` vs `Jul 8-12` orderings), defined as
module-level or function-level constants in `telescope_runs.py`.

**Per-branch parse-or-raise pattern** (`views.py` lines 428-450, 506):
```python
    def _translate_constraints(self, constraints):
        translated = []
        for c in constraints:
            s = c.strip()
            lower = s.lower()

            if lower.endswith('is defined'):
                field = s[: -len(' is defined')].strip()
                if field == '':
                    raise ValueError(f'Invalid "is defined" constraint (missing field): {c}')
                translated.append(f'{field}|DF')
                continue

            # ... more elif-style branches via regex match ...

            else:
                raise ValueError(f'Unsupported constraint format: {c}')
```
Mirror this for `parse_run_line`: try telescope/instrument/status/date-range
sub-patterns in sequence, raise `ValueError(f'...: {line!r}')` with the
original input echoed back if nothing matches — and use `s.strip()` /
`lower = s.lower()` for case-insensitive status matching (D-04).

---

## Shared Patterns

### Error handling
**Source:** `solsys_code/telescope_runs.py` (`horizon_dip`, `sun_event`) and
`solsys_code/views.py` (`_translate_constraints`)
**Apply to:** `parse_run_line`

All invariant violations raise plain `ValueError` (no custom exception
classes), with f-string messages that:
- echo back the offending input via `{value!r}`
- state what was expected
- (for `sun_event`-style multi-line messages) include enough context for the
  caller to diagnose without re-reading source

### Docstrings
**Source:** `solsys_code/telescope_runs.py` (all functions)
**Apply to:** `parse_run_line`, `ParsedRun`

Google-style docstrings with `Args:` / `Returns:` / `Raises:` sections; one
sentence summary line; reference `docs/design/telescope_runs_calendar.rst`
for any formula/format derivation (per project convention on formula
citations).

### Type hints
**Source:** `solsys_code/telescope_runs.py` (`-> tuple[Time, Time]`, `-> list[Time]`)
**Apply to:** `parse_run_line(line: str) -> ParsedRun`, `ParsedRun` fields

Modern Python 3.10+ generic syntax (`list[...]`, `tuple[...]`), no `typing.List`/`typing.Tuple`.

## Pattern Assignments — Tests

### `solsys_code/tests/test_telescope_runs.py` (test, transform)

**Analog:** same file, existing `TestTelescopeRuns(TestCase)` class (lines 29-216)

**Import pattern** (line 9):
```python
from solsys_code.telescope_runs import SITES, _find_crossing, _local_noon_utc, get_site, horizon_dip, sun_event
```
Add `ParsedRun, parse_run_line` to this import line.

**Simple-assertion test pattern** (lines 106-115, `test_horizon_dip*`):
```python
    def test_horizon_dip(self):
        self.assertAlmostEqual(horizon_dip(2402).to_value(u.deg), 1.44, delta=0.02)

    def test_horizon_dip_raises_on_negative_altitude(self):
        with self.assertRaises(ValueError):
            horizon_dip(-10)

    def test_horizon_dip_raises_on_none_altitude(self):
        with self.assertRaises(ValueError):
            horizon_dip(None)
```
Use this style for `parse_run_line` success-path tests (assert each
`ParsedRun` field) and `ValueError`-path tests (`with self.assertRaises(ValueError):`).

**Note:** `parse_run_line` is pure-Python (no DB access required for the
function itself), but per CONTEXT.md's "Established Patterns" it should live
in this same `TestTelescopeRuns` class for consistency — new test methods do
not need `setUpTestData`/`Observatory` fixtures, just call `parse_run_line(...)`
directly. The three sample lines from
`docs/design/telescope_runs_calendar.rst` are the acceptance fixtures:
- `'NTT EFOSC2 allocation 9-13 July'` -> success, `ParsedRun(telescope='NTT', instrument='EFOSC2', status='allocation', month=7, day1=9, day2=13, ...)`
- `'Magellan IMACS 13-19 July (proposed)'` -> `ValueError` (ambiguous Magellan-Clay/Magellan-Baade)
- `'Magellan Proto-Lightspeed Jul 8-12 (proposed)'` -> `ValueError` (same ambiguity)

## No Analog Found

None — both files have strong analogs within `telescope_runs.py` itself
(style/docstrings/error-handling) and `views.py` (free-text parsing
precedent).

## Metadata

**Analog search scope:** `solsys_code/telescope_runs.py`, `solsys_code/views.py`, `solsys_code/tests/test_telescope_runs.py`
**Files scanned:** 3
**Pattern extraction date:** 2026-06-13
</content>
</invoke>
