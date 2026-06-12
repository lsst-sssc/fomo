# Phase 1: Site & Ephemeris Helper - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-12
**Phase:** 1-site-ephemeris-helper
**Areas discussed:** SITES registry structure & lookup, sun_event() signature & return types, Observatory record setup for SITE-03, Skycalc validation reference data

---

## SITES registry structure & lookup

| Option | Description | Selected |
|--------|-------------|----------|
| Use specific names as keys ('Magellan-Clay', 'Magellan-Baade', 'NTT', 'FTS') | Separate entries for both Magellan telescopes, unambiguous about which obscode is requested | ✓ |
| Single 'Magellan' entry, pick one obscode | Map 'Magellan' to just one of Clay/Baade since location is identical | |
| Key by MPC obscode, name is just a label | SITES keyed by obscode, names stored as display field | |

**User's choice:** Use specific names as keys (Magellan-Clay 268, Magellan-Baade 269 kept separate).
**Notes:** User flagged that "Magellan" is ambiguous — could refer to Magellan-Clay (268) or Magellan-Baade (269), two separate telescopes at the same site.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Add `timezone` field to Observatory model | CharField with IANA name via migration; keeps location+tz on canonical model | ✓ |
| Small name->tz dict in telescope_runs.py | Self-contained constant, no model changes | |

**User's choice:** Add `timezone` field to Observatory model.
**Notes:** User observed a separate SITES dict of EarthLocation/tz would duplicate what's effectively a dict of Observatory instances — better to extend Observatory itself.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Add Observatory.to_earth_location() | New method building astropy EarthLocation from lat/lon/altitude, alongside existing to_geocentric()/to_geodetic() | ✓ |
| Build EarthLocation in telescope_runs.py | Construct EarthLocation directly from Observatory fields in the new module | |

**User's choice:** Add `Observatory.to_earth_location()`.
**Notes:** Follows the existing pattern of coordinate-conversion methods on the model.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Dict: telescope name -> obscode | SITES = {'Magellan-Clay': '268', ...}; get_site(name) does Observatory.objects.get(obscode=...) | ✓ |
| get_site(name) function only, no SITES dict | Internal name->obscode mapping inside a single function, no public dict | |

**User's choice:** Dict: telescope name -> obscode.
**Notes:** Observatory is the single source of truth for location+tz (via D-01/D-02); SITES stays a thin name->obscode map.

---

## sun_event() signature & return types

| Option | Description | Selected |
|--------|-------------|----------|
| astropy Time objects | Return astropy.time.Time (UTC scale) | ✓ |
| Python datetime (UTC, tz-aware) | Return tz-aware datetime.datetime in UTC | |

**User's choice:** astropy Time objects.

---

| Option | Description | Selected |
|--------|-------------|----------|
| 'sun' -> (sunset, sunrise); 'dark' -> (dark_start, dark_end) | Two kinds, matching dip-corrected and -15° thresholds | ✓ |
| Single dict result covering all crossings | sun_event(site, date) with no kind, returns dict of all crossings | |
| 'sun', 'dark', and 'twilight' (-18°) all supported | Adds a third kind for -18° astronomical twilight | |

**User's choice:** 'sun' -> (sunset, sunrise); 'dark' -> (dark_start, dark_end).
**Notes:** -18° twilight remains only a validation cross-check value (Jun 10 2026), not a third `kind`.

---

| Option | Description | Selected |
|--------|-------------|----------|
| date = local calendar date of sunset | sun_event computes the observing night starting on the evening of `date` (site-local time) | ✓ |
| date = UTC date, compute crossings within that UTC day | Treats `date` as a UTC calendar day | |

**User's choice:** date = local calendar date of sunset.

---

## Observatory record setup for SITE-03

| Option | Description | Selected |
|--------|-------------|----------|
| Data migration creates the records | Migration in solsys_code_observatory creates the 4 Observatory rows with lat/lon/altitude/timezone | ✓ |
| Test fixtures only (setUp in solsys_code/tests/) | Tests create records as needed, no production migration | |
| Management command + manual run | One-off management command, run manually per environment | |

**User's choice:** Data migration creates the records.

---

| Option | Description | Selected |
|--------|-------------|----------|
| I'll provide/confirm the obscodes now | User supplies MPC obscodes directly | ✓ |
| Look up via MPC Obscodes API during planning/execution | Defer obscode resolution to planner/executor via MPCObscodeFetcher | |

**User's choice:** I'll provide/confirm the obscodes now.
**Notes:** Magellan-Clay 268, Magellan-Baade 269, NTT 809, FTS E10.

---

## Skycalc validation reference data

| Option | Description | Selected |
|--------|-------------|----------|
| I have Jun 10 values; others TBD during planning/execution | Jun 10 2026 twi.end/twi.beg known from design doc; sunset/sunrise for all 4 sample dates TBD | (superseded) |
| I'll provide all reference values now | All 4 sample dates' values provided now | |
| Other: LCO ephemeris form (https://www.lco.cl/ephemeris-for-lco/) | User will source reference values from this form rather than a generic "skycalc" tool | ✓ |

**User's choice:** Use https://www.lco.cl/ephemeris-for-lco/ as the reference source.
**Notes:** Jun 10 2026 -18° twilight values (19:16/06:08 local) already known from the design doc; sunset/sunrise for Jun 1/10/20/30 2026 still need to be pulled from the LCO form during planning/execution.

---

## Claude's Discretion

- Return type/shape of `get_site(name)` (Observatory instance vs. small wrapper dataclass).
- `timezone` field implementation details (CharField choices vs. free IANA string, backfill policy for existing rows).
- Internal structure of dip/-15°/-18° threshold helper functions.

## Deferred Ideas

None — discussion stayed within phase scope. A `'twilight'`/-18° `kind` for `sun_event()` was considered and explicitly not adopted for Stage 1.
