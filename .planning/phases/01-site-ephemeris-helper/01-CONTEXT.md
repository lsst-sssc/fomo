# Phase 1: Site & Ephemeris Helper - Context

**Gathered:** 2026-06-12
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers `solsys_code/telescope_runs.py`: a `SITES` registry that
resolves a telescope name to its `Observatory` record, plus a `sun_event()`
function that computes dip-corrected UTC sunset/sunrise and -15° dark-window
crossing times for a given observing night. It also extends the `Observatory`
model with the data needed to support that registry (timezone field,
`EarthLocation` conversion), and ensures `Observatory` records for the target
telescopes exist via a data migration.

</domain>

<decisions>
## Implementation Decisions

### Observatory model extensions
- **D-01:** Add a `timezone` field to the `Observatory` model (CharField
  storing an IANA timezone name, e.g. `'America/Santiago'`,
  `'Australia/Sydney'`) via a Django migration. Location and timezone live
  together on the canonical model.
- **D-02:** Add `Observatory.to_earth_location()` returning an
  `astropy.coordinates.EarthLocation` built from `self.lon`, `self.lat`,
  `self.altitude` — alongside the existing `to_geocentric()`/`to_geodetic()`
  methods, following the same pattern.

### SITES registry
- **D-03:** `SITES` is a dict mapping telescope name -> MPC obscode:
  ```python
  SITES = {
      'Magellan-Clay': '268',
      'Magellan-Baade': '269',
      'NTT': '809',
      'FTS': 'E10',
  }
  ```
  Magellan-Clay and Magellan-Baade are kept as separate entries (both at Las
  Campanas, same location/timezone) since they are distinct telescopes with
  distinct MPC obscodes — avoids the ambiguity of a single "Magellan" key.
- **D-04:** A lookup function (e.g. `get_site(name)`) resolves `SITES[name]`
  to an `Observatory.objects.get(obscode=...)` instance. `Observatory` is the
  single source of truth for location + timezone (via D-01/D-02); `SITES`
  itself stays a thin name->obscode map.

### Observatory records (SITE-03)
- **D-05:** A data migration in `solsys_code_observatory` creates/ensures
  `Observatory` records for all four telescopes, with these locked MPC
  obscodes:
  - Magellan-Clay: obscode `268` (Las Campanas)
  - Magellan-Baade: obscode `269` (Las Campanas)
  - NTT: obscode `809` (La Silla)
  - FTS: obscode `E10` (Siding Spring)

  Records include lat/lon/altitude and the new `timezone` field
  (`America/Santiago` for the Las Campanas/La Silla sites, `Australia/Sydney`
  for Siding Spring).

### sun_event() signature & return types
- **D-06:** `sun_event(site, date, kind)` returns `astropy.time.Time` objects
  (UTC scale). Callers convert to local datetime as needed.
- **D-07:** `kind` accepts two values:
  - `'sun'` -> `(sunset_time, sunrise_time)` using the dip-corrected
    threshold `-(0.833° + dip)`.
  - `'dark'` -> `(dark_start_time, dark_end_time)` using the -15° threshold
    (not dip-corrected), per the design doc.
- **D-08:** `date` is the **local calendar date of sunset** at the site —
  `sun_event(site, date, 'sun')` returns the sunset on the evening of `date`
  (site-local time) and the following sunrise, i.e. one observing night
  starting on `date`.

### Skycalc validation reference data
- **D-09:** Validation reference values for sunset/sunrise/dark-window
  crossings are obtained from the LCO ephemeris form
  (https://www.lco.cl/ephemeris-for-lco/) for Las Campanas on June 1, 10, 20,
  and 30, 2026. The June 10 2026 -18° astronomical twilight crossings
  (twi.end/twi.beg = 19:16/06:08 local) are already known from the design doc;
  sunset/sunrise values for all four sample dates still need to be pulled from
  the LCO form during planning/execution and used as the comparison targets
  for EPHEM-04/EPHEM-05.

### Claude's Discretion
- Exact return type for `get_site(name)`: returning the `Observatory`
  instance directly (per D-04) vs. wrapping it in a small dataclass is left to
  the planner/executor, as long as `EarthLocation` and `timezone` are
  reachable via D-02's method and D-01's field.
- Whether the `timezone` field uses free-text `CharField` with a default/blank
  policy for existing Observatory rows (migration backfill for the 4 new
  records only; other existing rows can be left blank/null).
- Internal helper structure for the dip-correction and -15°/-18° threshold
  calculations (private helper functions vs. inline in `sun_event`).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Astronomy/design spec
- `docs/design/telescope_runs_calendar.rst` §"Stage 1" — dip formula
  (`dip = 1.76' * sqrt(h_metres)`), sunset/sunrise threshold
  (`-(0.833° + dip)`, dark-window threshold (-15°, not dip-corrected),
  3-site coordinate table, and the validation methodology against the LCO
  ephemeris tool.

### Observatory model
- `solsys_code/solsys_code_observatory/models.py` — `Observatory` model;
  add `timezone` field (D-01) and `to_earth_location()` method (D-02)
  alongside existing `to_geocentric()`/`to_geodetic()`/`ObservatoryXYZ()`
  methods. Note the existing "Silent Fallback in MPC Parallax Conversion"
  anti-pattern in `from_parallax_constants()` (ARCHITECTURE.md) — do not
  replicate that silent-failure style in new methods.

### Requirements
- `.planning/REQUIREMENTS.md` — SITE-01, SITE-02, SITE-03, EPHEM-01 through
  EPHEM-06.
- `.planning/ROADMAP.md` §"Phase 1: Site & Ephemeris Helper" — success
  criteria 1-5, including the exact obscodes/coordinates/timezones table.

### External validation tool
- https://www.lco.cl/ephemeris-for-lco/ — LCO ephemeris form used to obtain
  sunset/sunrise/twilight reference values for Las Campanas on the four
  June 2026 sample dates (D-09).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Observatory.to_geocentric()` / `to_geodetic()` (`solsys_code/solsys_code_observatory/models.py`) — existing ERFA-based coordinate conversion methods; `to_earth_location()` (D-02) should follow the same method style/placement.
- `Observatory.objects.get(obscode=...)` — standard lookup pattern already used by `CreateObservatory`/`ObservatoryDetailView`.

### Established Patterns
- Django data migrations under `solsys_code/solsys_code_observatory/migrations/` for seeding/altering Observatory data (D-05 follows this pattern for adding the `timezone` field + seed records).
- `zoneinfo` (stdlib) for IANA timezone handling, per CLAUDE.md constraints.

### Integration Points
- New module: `solsys_code/telescope_runs.py` (SITES dict, `get_site()`, `sun_event()`).
- Modified: `solsys_code/solsys_code_observatory/models.py` (new field + method).
- New migration: `solsys_code/solsys_code_observatory/migrations/` (timezone field + 4 Observatory records).
- DB-dependent tests go in `solsys_code/tests/`, run via `./manage.py test solsys_code` (per CLAUDE.md).
- **Avoid importing `ephem_utils` from `telescope_runs.py`** — `ephem_utils.py` has a module-level SPICE-kernel-loading side effect (noted in STATE.md blockers).

</code_context>

<specifics>
## Specific Ideas

- Magellan-Clay (268) and Magellan-Baade (269) must be modeled as separate
  `SITES` entries even though they share a location, because they are
  distinct telescopes/instruments with distinct MPC obscodes.
- NTT obscode is 809 (La Silla); FTS obscode is E10 (Siding Spring) — both
  provided directly by the user, locked values.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (A `'twilight'`/-18° `kind` was
considered for `sun_event()` but rejected for Stage 1; -18° twilight is used
only as a validation cross-check value for June 10, not as a third `kind`.)

</deferred>

---

*Phase: 1-site-ephemeris-helper*
*Context gathered: 2026-06-12*
