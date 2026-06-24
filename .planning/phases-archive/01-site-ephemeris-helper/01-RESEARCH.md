# Phase 1: Site & Ephemeris Helper - Research

**Researched:** 2026-06-12
**Domain:** Astronomical ephemeris calculation (astropy), Django model extension + data migration, IANA timezone handling (zoneinfo)
**Confidence:** HIGH

## Summary

Phase 1 builds a self-contained module `solsys_code/telescope_runs.py` that
resolves a telescope name to an `Observatory`-backed `astropy.coordinates.EarthLocation`
and computes dip-corrected sunset/sunrise and -15° dark-window crossing times.
All required astronomy primitives (`astropy.coordinates.get_sun`, `AltAz`,
`EarthLocation`, `astropy.time.Time`) are already available in the project's
venv (astropy 7.2.0, `[VERIFIED: pip show astropy]`) and were exercised
directly in this research session against the exact Las Campanas coordinates
from the design doc — results match the design doc's validated reference
values to well within the 2-minute tolerance.

The dip formula (`dip = 1.76' * sqrt(h_metres)`) was verified numerically:
at h=2402 m, dip = 1.4376° ≈ 1.44° ± 0.02° `[VERIFIED: computed in this session]`.
The combined sunset/sunrise threshold is `-(0.833° + dip) ≈ -2.27°`. The -15°
dark-window threshold and the -18° astronomical-twilight cross-check threshold
are applied without the dip correction, per the design doc and D-07.

A coarse-scan + bisection-refinement root-finding approach (no new
dependencies — pure astropy + numpy, both already installed) is sufficient to
hit sub-minute precision, well inside the 2-minute requirement. `zoneinfo`
(stdlib, `tzdata` already installed `[VERIFIED: pip show / python import]`)
correctly resolves `America/Santiago` to UTC-4 in June / UTC-3 in January, and
`Australia/Sydney` to UTC+10 in July / UTC+11 in January — both verified
numerically in this session, matching EPHEM-06 exactly.

For the Django side, the existing single migration (`0001_initial.py`) defines
`Observatory` with no `timezone` field. A second migration must add a
`CharField` for `timezone` (`AddField`) and a `RunPython` data migration to
seed/upsert the 4 Observatory records (D-05). The existing `to_geocentric()` /
`to_geodetic()` methods establish the pattern for the new `to_earth_location()`
method (D-02) — a simple wrapper around `astropy.coordinates.EarthLocation(lon=..., lat=..., height=...)`.

**Primary recommendation:** Use `astropy.coordinates.get_sun` + `AltAz` +
`EarthLocation` directly (no `ephem_utils` import — avoids the SPICE kernel
side effect), compute crossing times via coarse time-array scan (1-min steps)
followed by bisection refinement to sub-minute precision, and seed Observatory
records via a `RunPython` data migration paired with an `AddField` migration
for `timezone`.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Site resolution (telescope name -> Observatory -> EarthLocation/timezone) | Database / Storage (Observatory model + migration) | API/Backend helper module (`telescope_runs.py`) | `Observatory` is the canonical store (ORM); `telescope_runs.py` is a thin lookup/adapter layer, not a new persistence layer |
| Sun-event computation (sunset/sunrise/dark window) | API/Backend (`telescope_runs.py`, pure function) | — | Pure computation module, no DB or HTTP dependency at call time — only `get_site()` touches the DB |
| Timezone resolution | API/Backend (`Observatory.timezone` field + `zoneinfo`) | — | Stored as data on the canonical model (D-01), resolved via stdlib `zoneinfo` at call time |
| Observatory record seeding | Database / Storage (Django migration, `RunPython`) | — | Data migrations are the project's established pattern for seeding reference data |

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SITE-01 | `SITES` resolves telescope name -> `Observatory` -> `EarthLocation` | `to_earth_location()` pattern verified against existing `to_geocentric()`/`to_geodetic()`; `EarthLocation(lon=.., lat=.., height=..)` constructor confirmed working with astropy 7.2.0 |
| SITE-02 | `SITES` provides correct IANA timezone per site | `zoneinfo` DST resolution verified numerically for both `America/Santiago` and `Australia/Sydney` |
| SITE-03 | `Observatory` records exist for Magellan-Clay (268), Magellan-Baade (269), NTT (809), FTS (E10) | Migration pattern (`AddField` + `RunPython`) documented below; existing `0001_initial.py` reviewed for field definitions to mirror |
| EPHEM-01 | `sun_event(..., 'sun')` returns dip+refraction-corrected sunset/sunrise | Threshold `-(0.833° + dip)` verified against Las Campanas June 10 2026 — sunset 21:59 UTC / sunrise 11:25 UTC, both consistent with design doc's "<=1 min" claim |
| EPHEM-02 | `sun_event(..., 'dark')` returns -15° crossings | -15° crossings computed: 23:01:30 UTC / 10:22:30 UTC for June 10 2026 |
| EPHEM-03 | Dip helper returns 1.44° ± 0.02° at 2402 m | Verified: `1.76 * sqrt(2402) / 60 = 1.4376°` |
| EPHEM-04 | Las Campanas sunset/sunrise (dip-corrected) within 2 min of Las Campanas skycalc for Jun 1/10/20/30 2026 | Jun 10 values computed and self-consistent with design doc's -18° reference; Jun 1/20/30 values NOT yet pulled from Las Campanas ephemeris form — see Open Questions |
| EPHEM-05 | -18° twilight for Jun 10 2026 within 2 min of skycalc twi.end/twi.beg (19:16/06:08 local) | Computed -18° crossings: 23:16:00 UTC (= 19:16 local UTC-4) and 10:08:00 UTC (= 06:08 local UTC-4) — **exact match** to design doc reference |
| EPHEM-06 | Timezone DST resolution correctness | Verified: Santiago UTC-4 (Jun) / UTC-3 (Jan); Sydney UTC+10 (Jul) / UTC+11 (Jan) |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| astropy | 7.2.0 `[VERIFIED: pip show astropy]` | `EarthLocation`, `get_sun`, `AltAz`, `Time` for sun-position calculations | Already a project dependency (transitive via sorcha/tom_eso); design-doc-validated approach; matches CLAUDE.md constraint |
| zoneinfo | stdlib (Python 3.10+) | IANA timezone resolution for `America/Santiago` / `Australia/Sydney` | Stdlib, no extra dependency; CLAUDE.md-mandated |
| tzdata | installed `[VERIFIED: python -c "import tzdata"]` | Provides IANA tz database for `zoneinfo` on systems without OS tzdata | Already present in venv |
| numpy | >1.24 (already required) | Vectorized time-array scans for crossing-time search | Already a project dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Django ORM | 2.1+ (via TOM Toolkit, already in use) | `Observatory.objects.get(obscode=...)` lookups, migrations | `get_site()` lookup, data migration |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Coarse-scan + bisection root-finding (hand-rolled, ~20 lines) | `scipy.optimize.brentq` | scipy is installed transitively but NOT a declared dependency (`grep` of pyproject.toml returns nothing) — avoid introducing an implicit dependency on it; hand-rolled bisection is simple and sufficient for 2-min tolerance |
| `astropy.coordinates.get_sun` (low-precision, fast) | `astropy.coordinates.get_body('sun', time, loc)` (higher precision ephemeris, requires downloading JPL kernel via `solar_system_ephemeris`) | `get_sun` is adequate for arcminute-level precision needed here and avoids any kernel download — matches the "avoid `ephem_utils`" constraint in spirit (no SPICE/JPL kernel downloads) |
| Hand-written `EarthLocation.__init__` | `EarthLocation.of_site('lasilla')` / `EarthLocation.of_site('Siding Spring Observatory')` (astropy site database, used in design doc table's "Source" column for NTT/FTS) | D-02 locks the approach to building `EarthLocation` from `Observatory.lon/lat/altitude` fields (single source of truth) — `of_site()` would bypass the `Observatory` model and is explicitly NOT what D-02 specifies |

**Installation:**
No new packages required — astropy, numpy, and tzdata are already installed
in the project venv `[VERIFIED: pip show astropy; python -c "import tzdata"]`.
`zoneinfo` is stdlib in Python 3.10+ (project requires 3.10+).

**Version verification:**
```
$ pip show astropy
Name: astropy
Version: 7.2.0
```
`[VERIFIED: pip show astropy, this session]`. tzdata confirmed importable
from the project venv (`/home/tlister/venv/fomo311_venv/.../tzdata/__init__.py`).

## Package Legitimacy Audit

**No new external packages are introduced by this phase.** All required
libraries (astropy, numpy, tzdata, zoneinfo) are already installed in the
project environment and were exercised directly in this research session.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| astropy | PyPI | 13+ yrs | very high | github.com/astropy/astropy | OK (pre-existing) | No action — already installed |
| tzdata | PyPI | stable, widely used | very high | github.com/python/tzdata | OK (pre-existing) | No action — already installed |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                 +-------------------------+
telescope name → | SITES dict (name->obscode)| → MPC obscode
                 +-------------------------+
                              |
                              v
                 +-------------------------+
                 | get_site(name)           |  ← Observatory.objects.get(obscode=...)
                 | returns Observatory inst. |
                 +-------------------------+
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
       Observatory.to_earth_location()   Observatory.timezone (CharField)
                  |                       |
                  v                       v
         astropy.EarthLocation      zoneinfo.ZoneInfo(tz_name)
                  |                       |
                  +-----------+-----------+
                              |
                              v
        date (local calendar date, D-08) -> local-noon UTC anchor (via zoneinfo)
                              |
                              v
            +---------------------------------------+
            | sun_event(site, date, kind)             |
            |  1. build EarthLocation + tz            |
            |  2. anchor = local noon -> UTC Time     |
            |  3. coarse scan +/-12h, 1-min steps     |
            |     get_sun(times) -> AltAz -> alt[]    |
            |  4. find sign-change vs threshold       |
            |     'sun'  -> -(0.833 + dip(altitude))  |
            |     'dark' -> -15 deg (no dip)          |
            |  5. bisection refine to sub-minute      |
            +---------------------------------------+
                              |
                              v
              (Time_set, Time_rise)  -- astropy.time.Time, UTC scale
```

### Recommended Project Structure
```
solsys_code/
├── telescope_runs.py              # NEW: SITES dict, get_site(), sun_event(), dip helper
├── solsys_code_observatory/
│   ├── models.py                  # MODIFIED: + timezone field, + to_earth_location()
│   └── migrations/
│       ├── 0001_initial.py        # existing
│       └── 0002_<name>.py         # NEW: AddField(timezone) + RunPython seed data
└── tests/
    └── test_telescope_runs.py     # NEW: DB-dependent tests (Observatory lookups, sun_event)
```

### Pattern 1: EarthLocation from Observatory fields (D-02)
**What:** Add `to_earth_location()` alongside existing `to_geocentric()`/`to_geodetic()`.
**When to use:** Whenever `telescope_runs.py` needs an `astropy.coordinates.EarthLocation`.
**Example:**
```python
# Pattern follows existing to_geodetic() style (models.py:120-129)
from astropy.coordinates import EarthLocation
import astropy.units as u

def to_earth_location(self) -> EarthLocation:
    """Returns the observatory location as an astropy EarthLocation.

    Returns:
        EarthLocation: built from this observatory's lon, lat, altitude
    """
    return EarthLocation(lon=self.lon * u.deg, lat=self.lat * u.deg, height=self.altitude * u.m)
```
`[VERIFIED: constructor signature exercised in this session against astropy 7.2.0]`

### Pattern 2: Sun-event crossing search (coarse scan + bisection)
**What:** Find the UTC time(s) where solar altitude crosses a threshold.
**When to use:** `sun_event()` for both `'sun'` and `'dark'` kinds.
**Example:**
```python
# Source: pattern verified in this research session against astropy 7.2.0
import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, get_sun
from astropy.time import Time


def _solar_altitude(times: Time, location) -> np.ndarray:
    """Solar altitude in degrees for an array of Time objects at a location."""
    sun = get_sun(times)
    altaz = sun.transform_to(AltAz(obstime=times, location=location))
    return altaz.alt.deg


def _find_crossing(anchor: Time, location, threshold_deg: float, search_hours: float = 24,
                    coarse_step_min: float = 1.0) -> list[Time]:
    """Find UTC times where solar altitude crosses threshold_deg within +/-search_hours/2 of anchor."""
    offsets = np.arange(-search_hours * 30, search_hours * 30, coarse_step_min) * u.min
    times = anchor + offsets
    alt = _solar_altitude(times, location)
    crossings = []
    for i in range(len(alt) - 1):
        if (alt[i] - threshold_deg) * (alt[i + 1] - threshold_deg) < 0:
            # bisection refine between times[i] and times[i+1]
            lo, hi = times[i], times[i + 1]
            lo_alt = alt[i]
            for _ in range(10):  # ~1/1024 of 1-min step -> sub-second precision
                mid = lo + (hi - lo) / 2
                mid_alt = _solar_altitude(Time([mid]), location)[0]
                if (mid_alt - threshold_deg) * (lo_alt - threshold_deg) < 0:
                    hi = mid
                else:
                    lo, lo_alt = mid, mid_alt
            crossings.append(lo)
    return crossings
```

### Pattern 3: Local-date anchor for D-08
**What:** Convert the "local calendar date of sunset" into a UTC anchor for the search window.
**When to use:** First step of `sun_event(site, date, kind)`.
**Example:**
```python
# Source: pattern verified in this research session
from datetime import date as date_cls, datetime, time
from zoneinfo import ZoneInfo
from astropy.time import Time


def _local_noon_utc(local_date: date_cls, tz_name: str) -> Time:
    """Local noon of local_date, converted to UTC, as an astropy Time."""
    tz = ZoneInfo(tz_name)
    local_noon = datetime.combine(local_date, time(12, 0), tzinfo=tz)
    return Time(local_noon.astimezone(ZoneInfo('UTC')))
```
Searching +/-12h from local noon UTC guarantees both the evening sunset of
`local_date` and the following morning's sunrise/dark-window-end fall within
the window, for all three sites (including FTS at +149° longitude where the
night sits within a single UTC date).

### Pattern 4: Data migration for AddField + seed records (D-05)
**What:** Add `timezone` field, then seed/upsert the 4 Observatory records.
**When to use:** New migration `0002_*.py` in `solsys_code_observatory/migrations/`.
**Example:**
```python
# Source: pattern follows 0001_initial.py structure + standard Django RunPython idiom
from django.db import migrations, models


def seed_observatories(apps, schema_editor):
    Observatory = apps.get_model('solsys_code_observatory', 'Observatory')
    records = [
        dict(obscode='268', name='Magellan Clay Telescope', short_name='Magellan-Clay',
             lat=-29.0146, lon=-70.6926, altitude=2402, timezone='America/Santiago'),
        dict(obscode='269', name='Magellan Baade Telescope', short_name='Magellan-Baade',
             lat=-29.0146, lon=-70.6926, altitude=2402, timezone='America/Santiago'),
        dict(obscode='809', name='ESO, La Silla', short_name='NTT',
             lat=-29.2567, lon=-70.7300, altitude=2347, timezone='America/Santiago'),
        dict(obscode='E10', name='Siding Spring Observatory', short_name='FTS',
             lat=-31.2734, lon=149.0612, altitude=1149, timezone='Australia/Sydney'),
    ]
    for rec in records:
        obscode = rec.pop('obscode')
        Observatory.objects.update_or_create(obscode=obscode, defaults=rec)


def unseed_observatories(apps, schema_editor):
    Observatory = apps.get_model('solsys_code_observatory', 'Observatory')
    Observatory.objects.filter(obscode__in=['268', '269', '809', 'E10']).delete()


class Migration(migrations.Migration):

    dependencies = [
        ('solsys_code_observatory', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='observatory',
            name='timezone',
            field=models.CharField(blank=True, max_length=64, default='', verbose_name='IANA timezone name'),
        ),
        migrations.RunPython(seed_observatories, unseed_observatories),
    ]
```
Note: `name` must be `unique=True` per the existing model field — choose
distinct, descriptive names for Magellan-Clay vs Magellan-Baade (they share
lat/lon/altitude but need distinct `obscode` AND distinct `name`).

### Anti-Patterns to Avoid
- **Importing `ephem_utils` from `telescope_runs.py`:** triggers a ~1.6GB SPICE
  kernel download at module-import time (module-level side effect, per
  STATE.md blockers and ARCHITECTURE.md). Use `astropy.coordinates.get_sun`
  directly — it requires no kernel download.
- **Silent fallback on missing Observatory record:** `get_site(name)` should
  raise (e.g., let `Observatory.DoesNotExist` propagate or re-raise with
  context) rather than returning `None` — matches the project's documented
  anti-pattern warning about `from_parallax_constants()`'s silent failures.
- **Using `EarthLocation.of_site(...)`:** bypasses the `Observatory` model
  (single source of truth per D-02/D-04); only used historically to *derive*
  the coordinates that now live in `Observatory` records.
- **Applying dip correction to the -15° dark window:** per the design doc,
  dip is only applied to the `'sun'` kind threshold; `'dark'` uses -15°
  unmodified.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Sun position at a given time/place | Custom solar ephemeris (VSOP87, etc.) | `astropy.coordinates.get_sun` + `AltAz` | astropy's low-precision solar position is accurate to ~arcminute, sufficient for 2-min time tolerance; already a dependency |
| Timezone offsets / DST transitions | Custom DST tables | stdlib `zoneinfo` (with `tzdata`) | Handles IANA tz database transitions correctly, including Southern Hemisphere DST (reversed relative to Northern Hemisphere) |
| Root-finding for altitude-threshold crossing | Analytic spherical-trig sunrise equation (ignores atmospheric refraction nuance, harder to extend to -15°/-18°) | Coarse numeric scan + bisection over `get_sun`/`AltAz` | Generalizes trivially to any threshold (-2.27°, -15°, -18°) with one function; precision tunable via bisection iteration count |
| Horizon-dip formula | Re-derive from spherical geometry each time | `dip = 1.76' * sqrt(h_metres)` (Nautical Almanac formula, design-doc-specified) | Already verified to match the 1.44° ± 0.02° requirement at 2402 m; re-deriving risks subtle constant errors |

**Key insight:** Every piece of "hard astronomy" needed for this phase
(`get_sun`, `AltAz`, `EarthLocation`, dip formula, threshold values) is either
in astropy or a single documented constant from the design doc — there is no
case in this phase where a custom numerical method outperforms what's already
available.

## Common Pitfalls

### Pitfall 1: Wrong search window for the "local calendar date of sunset" (D-08)
**What goes wrong:** Anchoring the search to UTC midnight of `date` instead of
local noon can put the sunset of interest outside the search window, or
return the wrong night's sunrise (e.g., for Chilean sites where local evening
crosses UTC midnight).
**Why it happens:** `date` is a local-calendar concept but `astropy.time.Time`
has no timezone awareness — naive UTC-midnight anchoring silently shifts the
night by hours.
**How to avoid:** Anchor the search at **local noon of `date`, converted to
UTC via `zoneinfo`** (Pattern 3), then search ±12h. This guarantees the
evening sunset of `date` and the following sunrise both fall inside the
window for all three sites.
**Warning signs:** Sunset/sunrise times off by ~1 day, or off by exactly the
UTC offset (4-11 hours).

### Pitfall 2: Coarse-scan step size too large for -15°/-18° precision near solstice
**What goes wrong:** A 1-minute coarse step combined with skipping bisection
refinement can leave up to ±1 minute of error before refinement — borderline
for the 2-minute EPHEM-04/05 tolerance when combined with astropy's own
~10-30 second accuracy.
**Why it happens:** `get_sun` has inherent ~arcsecond-to-arcminute level error;
solar altitude changes ~0.25°/min near the horizon, so a 1-arcmin error in
solar position corresponds to ~4 seconds of time error — small, but coarse-step
discretization error dominates if not refined.
**How to avoid:** Always run the bisection refinement step (Pattern 2) after
the coarse scan — 8-10 bisection iterations on a 1-minute interval gets to
sub-second precision, comfortably inside the 2-minute tolerance.
**Warning signs:** EPHEM-04/05 test failures with errors of ~30-60 seconds.

### Pitfall 3: `Observatory.name` uniqueness collision for Magellan-Clay/Baade
**What goes wrong:** Both telescopes share the same `lat`/`lon`/`altitude`
(Las Campanas), but `Observatory.name` has `unique=True` — using a generic
name like `"Magellan, Las Campanas"` for both records will raise
`IntegrityError` on the second `update_or_create`/`create`.
**Why it happens:** The model's uniqueness constraint is on `name`, not on
the coordinate tuple.
**How to avoid:** Use distinct, telescope-specific names (e.g., `"Magellan
Clay Telescope"` / `"Magellan Baade Telescope"`) — `obscode` (268 vs 269) is
already distinct per D-05/D-03.
**Warning signs:** `IntegrityError: UNIQUE constraint failed: ...name` during
migration.

### Pitfall 4: Forgetting `astropy.units` on raw floats
**What goes wrong:** `EarthLocation(lon=self.lon, lat=self.lat, height=self.altitude)`
without `* u.deg` / `* u.m` raises a `TypeError` or silently misinterprets
units (astropy's `EarthLocation` requires `Quantity` objects, not bare floats,
for `lon`/`lat`/`height` in the geodetic constructor).
**Why it happens:** `Observatory.lon`/`.lat`/`.altitude` are plain Django
`FloatField`s (degrees / degrees / meters by convention), but astropy
requires explicit unit annotation.
**How to avoid:** Always multiply by `astropy.units.deg` / `astropy.units.m`
when constructing `EarthLocation` (Pattern 1).
**Warning signs:** `TypeError` or `UnitsError` at `EarthLocation(...)`
construction, or `astropy.units.UnitConversionError`.

## Code Examples

### Computing dip
```python
# Source: design doc formula, verified numerically in this session
from math import sqrt

def horizon_dip(altitude_m: float) -> float:
    """Horizon dip in degrees for an observer at altitude_m metres.

    dip = 1.76 arcmin * sqrt(altitude in metres), converted to degrees.

    Returns:
        float: dip angle in degrees (e.g. 1.4376 for 2402 m)
    """
    dip_arcmin = 1.76 * sqrt(altitude_m)
    return dip_arcmin / 60.0
```
Verified: `horizon_dip(2402)` -> `1.4376...` (within 1.44° ± 0.02°).

### Full sun_event sketch
```python
# Source: composition of Patterns 1-3, this research session
from datetime import date as date_cls

def sun_event(site, date: date_cls, kind: str):
    """Compute UTC sun-event crossing times for an observing night.

    Args:
        site: Observatory instance (from get_site())
        date: local calendar date of sunset (D-08)
        kind: 'sun' (dip-corrected sunset/sunrise) or 'dark' (-15 deg window)

    Returns:
        tuple[Time, Time]: (start, end) as astropy.time.Time, UTC scale
    """
    location = site.to_earth_location()
    anchor = _local_noon_utc(date, site.timezone)

    if kind == 'sun':
        dip = horizon_dip(site.altitude)
        threshold = -(0.833 + dip)
    elif kind == 'dark':
        threshold = -15.0
    else:
        raise ValueError(f"kind must be 'sun' or 'dark', got {kind!r}")

    crossings = _find_crossing(anchor, location, threshold, search_hours=24)
    # crossings[0] = setting (evening), crossings[1] = rising (morning)
    return crossings[0], crossings[1]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `pytz` for timezone handling | `zoneinfo` (stdlib since 3.9) | Python 3.9 (2020) | Already specified by CLAUDE.md; no pytz dependency needed |
| `astropy.coordinates.get_sun` low-precision built-in | (no change needed for this phase) | n/a | `get_sun` remains the lightweight, no-kernel-download option in astropy 7.2.0 `[VERIFIED]` |

**Deprecated/outdated:** None identified — astropy 7.2.0 is current and
`get_sun`/`AltAz`/`EarthLocation` APIs used here are stable, non-deprecated.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | NTT (La Silla) coordinates: lat=-29.2567, lon=-70.7300, alt=2347m, obscode=809 | Code Examples / Pattern 4 (migration seed data) | If MPC obscode 809's actual registered coordinates differ from the design doc's `of_site('lasilla')`-derived values, SITE-03's "correct lat/lon/altitude" criterion could fail verification — planner should add a step to cross-check against the MPC obscode API or `EarthLocation.of_site()` during execution |
| A2 | FTS (Siding Spring) coordinates: lat=-31.2734, lon=149.0612, alt=1149m, obscode=E10 | Code Examples / Pattern 4 (migration seed data) | Same as A1 — design-doc-sourced, not independently re-verified against MPC obscode API in this session |
| A3 | Magellan-Clay/Baade `Observatory.name` values (`"Magellan Clay Telescope"` / `"Magellan Baade Telescope"`) are illustrative — exact strings not locked by CONTEXT.md | Pattern 4 | Low risk — `name` uniqueness just needs any two distinct strings; planner/executor can choose final wording |
| A4 | Las Campanas skycalc reference sunset/sunrise values for Jun 1, 20, 30 2026 (EPHEM-04) have not been pulled from https://www.lco.cl/ephemeris-for-lco/ in this session | Phase Requirements (EPHEM-04), Open Questions | If the form is unreachable or values differ significantly from computed values, EPHEM-04 verification may need to rely only on the Jun 10 cross-check (already validated via -18° match) plus internal consistency checks |

**Note:** obscode values 268, 269, 809, E10 themselves are **locked decisions
from CONTEXT.md (D-05)**, not assumptions — only the associated
lat/lon/altitude/timezone *data values* for NTT and FTS in the seed migration
(A1/A2) are flagged, since those were taken from the design doc's table
without independent re-verification against the MPC obscode API in this
session.

## Open Questions (RESOLVED)

1. **Las Campanas skycalc reference values for Jun 1, 20, 30 2026 (EPHEM-04)** — RESOLVED
   - What we know: Jun 10 2026 -18° twilight crossings (19:16/06:08 local)
     match this session's computation exactly; the design doc reports <=1 min
     agreement for Jun 2026 generally.
   - What's unclear: The specific sunset/sunrise times for Jun 1, 20, 30 2026
     could not be fetched from https://www.lco.cl/ephemeris-for-lco/ — the
     ephemeris tool is a dynamic form (`?page_id=299`) with no accessible
     query-string interface for WebFetch, and the page itself returns no
     pre-computed tables.
   - Resolution (user sign-off, 2026-06-12): proceed with the
     internal-consistency fallback documented in Plan 01-02 Task 1. Jun 10
     2026 is validated against the design-doc skycalc values (exact -18°
     twilight match). Jun 1/20/30 2026 are validated via internal
     seasonal-consistency checks (smooth day-to-day progression of
     sunset/sunrise times, correct DST-aware UTC offsets). This is accepted
     as sufficient evidence that the `sun_event('sun', ...)` threshold/dip
     logic generalizes correctly across all four sample dates for EPHEM-04.

2. **MPC obscode 809 / E10 coordinate precision**
   - What we know: Design doc gives lat/lon/altitude for NTT and FTS sourced
     from astropy's `EarthLocation.of_site()`, not directly from the MPC
     obscode API.
   - What's unclear: Whether MPC's registered parallax constants for 809/E10
     convert (via `from_parallax_constants`) to the same lat/lon/altitude as
     the design doc table, to decimal precision.
   - Recommendation: For SITE-03, seed with the design-doc values (D-05 locks
     the obscodes, not necessarily decimal coordinates) — if a planner task
     wants belt-and-suspenders verification, it could call the MPC obscodes
     API and compare, but this is not required for the 2-minute EPHEM
     tolerances (sub-arcsecond coordinate differences have negligible time
     impact).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| astropy | get_sun/AltAz/EarthLocation (EPHEM-01 to 06) | Yes `[VERIFIED]` | 7.2.0 | — |
| tzdata | zoneinfo IANA database (SITE-02, EPHEM-06) | Yes `[VERIFIED]` | (installed, version not checked) | OS tzdata as fallback if absent |
| numpy | vectorized time-array scans | Yes (>1.24, project dependency) | — | — |
| scipy | NOT used — avoided deliberately | n/a (present transitively but undeclared) | — | hand-rolled bisection (Pattern 2) |
| Django ORM / migrations | get_site(), seed migration (SITE-03) | Yes (project framework) | Django 4.2.19 (per 0001_initial.py header) | — |
| Internet access to lco.cl/ephemeris-for-lco | EPHEM-04 reference values (Jun 1/20/30) | Not exercised this session | — | Rely on Jun 10 -18° exact match as validation proxy (see Open Question 1) |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:**
- Las Campanas ephemeris form access for additional reference dates — fallback is
  internal consistency validation against the already-confirmed Jun 10 -18°
  values.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django `TestCase` (DB-dependent) via `./manage.py test solsys_code`; plain `pytest` for pure-math helpers per `[tool.pytest.ini_options] testpaths = ["tests", "src", "docs"]` |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`, `[tool.ruff]`) |
| Quick run command | `./manage.py test solsys_code.tests.test_telescope_runs` |
| Full suite command | `./manage.py test solsys_code && python -m pytest && ruff check . && ruff format --check .` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SITE-01 | `get_site('Magellan-Clay')` returns Observatory w/ `to_earth_location()` | unit (DB) | `./manage.py test solsys_code.tests.test_telescope_runs.TestGetSite -v 2` | ❌ Wave 0 |
| SITE-02 | `get_site(...).timezone` resolves via zoneinfo to correct UTC offset | unit (DB) | `./manage.py test solsys_code.tests.test_telescope_runs.TestSiteTimezone -v 2` | ❌ Wave 0 |
| SITE-03 | Observatory records 268/269/809/E10 exist with correct obscode/lat/lon/altitude/timezone after migration | integration (DB, migration) | `./manage.py test solsys_code_observatory.tests.test_migrations` (or include in `test_telescope_runs`) | ❌ Wave 0 |
| EPHEM-01 | `sun_event(site, date, 'sun')` applies `-(0.833+dip)` threshold | unit (DB for site, pure compute) | `./manage.py test solsys_code.tests.test_telescope_runs.TestSunEventSun -v 2` | ❌ Wave 0 |
| EPHEM-02 | `sun_event(site, date, 'dark')` applies -15° threshold | unit (DB) | `./manage.py test solsys_code.tests.test_telescope_runs.TestSunEventDark -v 2` | ❌ Wave 0 |
| EPHEM-03 | `horizon_dip(2402)` -> 1.44 ± 0.02 | unit (pure, no DB) | `python -m pytest tests/test_telescope_runs_dip.py -x` (or co-located in DB test file as a plain method) | ❌ Wave 0 |
| EPHEM-04 | Las Campanas sunset/sunrise for Jun 1/10/20/30 2026 within 2 min of skycalc | integration (DB) — values from Open Question 1 | `./manage.py test solsys_code.tests.test_telescope_runs.TestSunEventValidation -v 2` | ❌ Wave 0 |
| EPHEM-05 | -18° twilight Jun 10 2026 within 2 min of 19:16/06:08 local | unit (DB) — uses the -18° helper internally for cross-check, computed via `_find_crossing` with threshold=-18 | `./manage.py test solsys_code.tests.test_telescope_runs.TestTwilightCrossCheck -v 2` | ❌ Wave 0 |
| EPHEM-06 | zoneinfo DST resolution for Santiago/Sydney | unit (pure, no DB) | `python -m pytest tests/test_telescope_runs_tz.py -x` (or co-located) | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `./manage.py test solsys_code.tests.test_telescope_runs -v 2` (fast, seconds)
- **Per wave merge:** `./manage.py test solsys_code && python -m pytest && ruff check . && ruff format --check .`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `solsys_code/tests/test_telescope_runs.py` — covers SITE-01, SITE-02, EPHEM-01, EPHEM-02, EPHEM-04, EPHEM-05 (DB-dependent, Django TestCase)
- [ ] Pure-math helper tests for `horizon_dip()` (EPHEM-03) and `zoneinfo` DST checks (EPHEM-06) — can be co-located in the same Django TestCase (simplest) or split into `tests/test_telescope_runs_helpers.py` (plain pytest, per CLAUDE.md's "pure-math helpers may live in `tests/`" note)
- [ ] Migration test for `0002_*` (SITE-03) — verify `AddField` + `RunPython` seed produces the 4 expected records; can be a `TestCase` that queries `Observatory.objects.filter(obscode__in=[...])` after migrations run (Django runs migrations automatically for test DB)
- [ ] No framework install needed — Django `TestCase` and `pytest` both already configured

## Security Domain

Not applicable — this phase is a pure computational helper module with no new
HTTP endpoints, authentication, user input, or external network calls at
runtime (only Django ORM reads of existing `Observatory` records and local
astropy computation). No ASVS categories apply.

## Project Constraints (from CLAUDE.md)

- **Astronomy library:** Use `astropy` (`get_sun`, `AltAz`, `EarthLocation`) —
  satisfied; no alternative considered.
- **Timezones:** Use `zoneinfo` (stdlib, `tzdata` installed) for
  `America/Santiago` and `Australia/Sydney` — satisfied; verified DST
  resolution numerically.
- **Data source:** Site coordinates come from `Observatory` model records
  (MPC obscode lookup), not hardcoded constants — satisfied via D-02/D-04
  (`to_earth_location()` built from `Observatory.lon/lat/altitude`).
- **Precision:** Sunset/sunrise must match Las Campanas skycalc to <= 2 minutes; dip
  at 2402 m must be 1.44° ± 0.02° — both verified achievable in this session.
- **Testing:** DB-dependent tests go in `solsys_code/tests/`, run with
  `./manage.py test solsys_code`. `ruff check .` and `ruff format --check .`
  must stay clean — reflected in Validation Architecture above; new code must
  follow single-quote, 120-char-line, Google-docstring conventions per
  CLAUDE.md's broader conventions section.
- **Module design:** No module-level mutable state (except the documented
  `ephem_utils` exception, which this phase explicitly avoids importing).
- **Naming:** snake_case functions/variables, PascalCase classes — `SITES`
  (module constant, UPPER_CASE per CLAUDE.md's constants convention),
  `get_site`, `sun_event`, `horizon_dip`, `to_earth_location`.

## Sources

### Primary (HIGH confidence)
- `pip show astropy` (this session) — confirmed astropy 7.2.0 installed
- Direct Python execution in project venv (this session) — `EarthLocation`,
  `get_sun`, `AltAz`, `Time`, `zoneinfo` all exercised against real Las
  Campanas / Siding Spring coordinates and produced numerically verified
  results
- `docs/design/telescope_runs_calendar.rst` — dip formula, threshold formulas,
  coordinate table, design-doc validation methodology (read this session)
- `solsys_code/solsys_code_observatory/models.py` — existing `Observatory`
  model, `to_geocentric()`/`to_geodetic()` pattern (read this session)
- `solsys_code/solsys_code_observatory/migrations/0001_initial.py` — existing
  field definitions and migration structure (read this session)
- `.planning/phases/01-site-ephemeris-helper/01-CONTEXT.md` — locked
  decisions D-01 through D-09 (read this session)

### Secondary (MEDIUM confidence)
- None — all astronomy claims were independently verified by direct
  computation in this session rather than relying on external docs/search.

### Tertiary (LOW confidence)
- Las Campanas skycalc reference values for Jun 1, 20, 30 2026 — NOT fetched in this
  session; only Jun 10 (from design doc, cross-checked) is confirmed. See
  Open Questions and Assumptions Log A4.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — astropy/zoneinfo/tzdata already installed and
  exercised directly; no new dependencies
- Architecture: HIGH — patterns directly mirror existing `Observatory` model
  methods and existing migration structure
- Pitfalls: HIGH — each pitfall was either discovered or specifically guarded
  against during hands-on verification in this session

**Research date:** 2026-06-12
**Valid until:** 2026-07-12 (30 days — stable stdlib/astropy APIs, but
calendar-date-specific reference values (Jun 2026 dates) are time-sensitive
only in the sense that they're tied to a specific year already in CONTEXT.md)
