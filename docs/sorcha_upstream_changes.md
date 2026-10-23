# Sorcha upstream changes (review as of 2026-06-24)

This note summarizes how the local `sorcha` checkout(s) used by `fomo`/`fomo_devel` and by the
`NEO_detectability` candle-flame script compare to `sorcha`'s current `origin/main`, and what (if
anything) is likely to break on an update.

At review time, the checkout under `~/git/sorcha` was **153 commits behind** `origin/main`:
local `HEAD` was `2025-07-15` ("saturation-filter bug fix"), upstream `origin/main` was
`2026-05-28` — about 10 months of upstream history.

## What changed upstream, by theme

### 1. New: DES (Dark Energy Survey) support, alongside Rubin
A large share of the upstream commits add **DES as a second supported survey**: a new `DES.py`,
`DES_config_file.ini`, a new `PPVisitsFootprintFilter.py` module (polygon-based footprint
filtering — this pulled in a new core dependency, `shapely`), DES-specific transient-efficiency
and distance/motion-cut filters, and a `hasTracklet` check. This is purely additive; the existing
Rubin code paths are untouched.

### 2. `PPReadPointingDatabase` (the OpSim-pointing-database reader)
- Gained an optional `fading_function_on=0` keyword argument — added with a default, so existing
  calls are unaffected.
- Gained a new branch for DES surveys when deriving `observationStartMJD_TAI`.
- **Now fails loudly** (`sys.exit` with an explicit message) if zero rows match the configured
  `observing_filters`, instead of silently returning an empty dataframe as before.

### 3. `sorchaConfigs.py` refactor
`sorchaConfigs` was split into `basesorchaConfigs` (the plain dataclass holding config fields) and
`sorchaConfigs(basesorchaConfigs)` (adds the file-loading `__init__`). A new `sorchaConfigsNoFile`
path means a config file is no longer strictly required, and a new `return_only` keyword lets
callers skip writing results to disk. `auxiliaryConfigs` (used directly by `fomo`'s ephemeris code)
keeps the same shape, but **its SPICE kernel defaults changed**: the `earth_predict` /
`earth_historical` filenames and NAIF download URLs were bumped to newer kernel versions.

### 4. Dependencies
Core dependencies gained `shapely` (for the new footprint filter). `sbpy` is now pinned to
`>=0.6.0` — `fomo`'s own `pyproject.toml` already anticipates this exact requirement (needed for
astropy 7.2+ compatibility). The rest of the dependency churn is GitHub Actions version bumps
(CI-only, no effect locally).

### 5. Misc
JOSS paper text edits, documentation updates, and removal of an "errant" duplicate
`sorcha_config_demo.ini` that had been accidentally committed at the repo root (the real demo
config under `src/sorcha/data/demo/` is untouched).

## Risk assessment for `fomo` / `fomo_devel`

`fomo`/`fomo_devel` does not use sorcha's OpSim/post-processing (`PP*`) pipeline at all — it
imports sorcha's **ephemeris internals** directly (see `solsys_code/ephem_utils.py`,
`solsys_code/views.py`):

- `sorcha.ephemeris.orbit_conversion_utilities.universal_cartesian`
- `sorcha.ephemeris.simulation_driver.{EphemerisGeometryParameters, get_residual_vectors, get_vec}`
- `sorcha.ephemeris.simulation_geometry.*`
- `sorcha.ephemeris.simulation_parsing.Observatory`
- `sorcha.ephemeris.simulation_setup.{create_assist_ephemeris, furnish_spiceypy}`
- `sorcha.utilities.sorchaConfigs.auxiliaryConfigs`

Diffing each of these against upstream:

- `orbit_conversion_utilities.py`, `simulation_geometry.py`, `simulation_parsing.py`,
  `simulation_setup.py` — **docstring-only changes**, no signature or behavior changes.
- `simulation_driver.py` — **no diff at all**.
- `auxiliaryConfigs` — same shape, but the SPICE kernel filename/URL defaults changed (see above).
  Practical effect: after upgrading, `furnish_spiceypy` / `create_assist_ephemeris` will download
  the *new* kernel files into `~/.cache/sorcha/` on first run. This is a one-time network/cache
  cost, not a code break — but worth knowing if running offline or disk-constrained.

**Bottom line: low risk.** Nothing in `fomo`'s actual call surface was renamed or removed. The one
concrete, observable effect of updating is the SPICE-kernel re-download triggered by the
`auxiliaryConfigs` default change.

## A separate, pre-existing trap: OpSim `filter` vs `band` schema drift

Not introduced by this changelog, but relevant to anyone wiring a *newer* OpSim database into
sorcha's pointing-database reader: comparing sorcha's demo db
(`src/sorcha/data/demo/baseline_v2.0_1yr.db`, an 11-column `observations` table) against a full,
modern `rubin_scheduler` output (`baseline_v5.3.0_10yrs.db`, 45 columns):

- In the v2.0 demo db, `filter` holds the plain band letter (`u,g,r,i,z,y`).
- In the v5.3.0 db, `filter` holds a **composite filter-slot id** instead (e.g. `u_24`, `r_57` —
  distinct physical filter units), and the plain band letter moved to a **new `band` column**.

sorcha's default `pointing_sql_query` (`src/sorcha/data/demo/sorcha_config_demo.ini`) still
selects `filter`, unchanged upstream. Pointed at a v5.3.0-schema database, this query returns zero
matching rows for any of the configured `observing_filters`. As of the current upstream code (see
point 2 above), this now fails with a clear error rather than silently producing empty output —
but the underlying query still needs to be rewritten (e.g. `band as filter`) to work correctly
against newer-schema OpSim databases.
