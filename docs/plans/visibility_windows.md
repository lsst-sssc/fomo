# Plan: multi-site visibility windows and cadence midpoint for non-sidereal targets

Written 2026-09-16 for a fresh Claude Code session (no prior context), from a design discussion held
in the production checkout (`../fomo_fresh`, which stays on `issue29-jpl-scout-ingest` ingesting
Scout objects — do not work there). Everything below was verified on this machine on that date.

## Priority context (read first)

Checked against the "FOMO Quick Requirements" Google Doc
(https://docs.google.com/document/d/1UvdnzIXkKmsiWkHy5EHjJDd9yB1RkrdzQ-aKtzims9E/). This
visibility/cadence tool is **not** on the requirements list (nearest item: "Observing Tools → ETC"
under Full Requirements). Things the doc ranks higher:

1. Scout ingest (`issue29-jpl-scout-ingest`): tom_jpl changes are merged upstream; FOMO
   [PR #50](https://github.com/lsst-sssc/fomo/pull/50) is awaiting review. Nothing to do until then.
2. Brokers (MVP "Add new objects → Brokers"): Tim is adding LSST/TAP support to the ALeRCE broker in
   tom_base, branch `origin/enhancement/alerce-add-tap-and-lsst` (in the tom_base repo, not here).
   FOMO-side wiring (`TOM_ALERT_CLASSES` in `src/fomo/settings.py`) follows once that lands.
3. Hosting/deployment (Tim's action items, overdue since week of 2026-08-17) — not code.
4. MPC observations → `ReducedDatums`: already in flight on `feature/add_mpc_obs`, see
   `docs/plans/ades_processor_port.md` in this directory. **Resume that before starting this plan
   unless Tim says otherwise.**
5. Target matching / metadata storage (provisional ID ↔ final designation, target extras) — should be
   settled as part of (4).

This plan is the self-contained, lower-priority piece to pick up while waiting on reviews.

## Goal

Given a non-sidereal `Target` and a set of sites (initially the three southern LCO sites LSC, CPT,
COJ), compute per-site visibility windows for a night, merge them, and report the observing window
and midpoint to use for a daily (or other period) LCO cadence request. Display it on the target page
as stacked horizontal bars, one per site, plus a combined row and a midpoint marker.

Worked example (UTC): LSC 23:45→08:00, CPT 17:30→02:00, COJ 09:00→17:15.
Merged coverage is 09:00→08:00(+1d) = 23 h; the only real gap is 08:00–09:00 (there is also a
15-min COJ→CPT hole at 17:15–17:30 worth reporting as a sub-gap but not treated as a break).
Result: window 09:00→08:00, **midpoint 20:30 UTC, ±11.5 h**. Use these as the fixture for tests.

## What exists today (verified)

- `solsys_code/views.py` `Ephemeris.get` (~line 170–393) does everything inline: parses request
  (`obscode`, `start`, `stop`, `step`), builds `pointings_df`, runs sorcha/ASSIST `integrate_light_time`
  + `calculate_rates_and_geometry` + erfa `atciqz`/`atioq` per step, and ends with a pandas
  `predictions` DataFrame (one row per time; columns include `epoch_UTC, RA_deg, Dec_deg, Obs_Az_deg,
  Obs_Alt_deg, Obs_HA_deg, APmag, sky_motion, ...`) for a **single** observatory. There is no callable
  that returns that frame — that is the missing internal wiring.
- Importing `solsys_code.ephem_utils` (and hence `solsys_code.views`) runs `fomo_furnish_spiceypy()`
  at import: ~1.6 GB SPICE kernels into `~/.cache/sorcha/`. Keep pure helpers out of those modules so
  their tests stay cheap.
- TOM Toolkit's plotting convention is **Plotly**, embedded via
  `plotly.offline.plot(fig, output_type='div', show_link=False)` in template tags
  (`tom_observations/templatetags/observation_extras.py::observation_plan`,
  `tom_targets/templatetags/targets_extras.py::target_distribution`, `moon_distance`).
  plotly 5.24.1, astroplan 0.10.1 and matplotlib 3.11 are installed; use Plotly.
- `tom_observations.utils.get_sidereal_visibility(target, start, end, interval, airmass_limit,
  facility_name)` returns `{site_name: [datetimes, airmasses]}` and explicitly refuses
  `NON_SIDEREAL` targets. Mirror its output shape so TOM's existing airmass plot also works.
- `Observatory` model (`solsys_code/solsys_code_observatory/`) has MPC codes; LCO sites map as
  `{'LSC': ['W85','W86','W87'], 'CPT': ['K91','K92','K93'], 'COJ': ['Q63','Q64']}` (one code per
  site is enough — inter-dome differences are negligible).
- Project-level template tags live in `src/templatetags/` (`fomo_extras`, `solsys_code_extras`).
- Tests: Django runner only (`python manage.py test`), under `solsys_code/tests/`.

## Design (four pieces, in dependency order)

1. **Ephemeris core extraction** — move the middle of `Ephemeris.get` into
   `ephem_utils.compute_ephemeris(target, observatory, times: astropy.time.Time) -> pd.DataFrame`.
   The view keeps request parsing and rendering and calls it. Pure refactor; existing ephemeris
   output must be byte-identical.

2. **Visibility sampler** — `get_nonsidereal_visibility(target, sites, start, end, interval_min,
   airmass_limit) -> {site: (times, airmass)}`. Runs `compute_ephemeris` per site at a 10–15 min
   step; masks on target altitude/airmass and sun altitude (astroplan `Observer.sun_altaz`, or reuse
   `tom_observations.utils.get_astroplan_sun_and_time`). Build the ASSIST sim once per target.

3. **Pure window helpers** in a new module (e.g. `solsys_code/visibility.py`) that does **not**
   import `ephem_utils`:
   - `visibility_windows(samples) -> {site: [(start, end), ...]}` (contiguous runs of valid samples).
   - `cadence_window(intervals, period=24*u.h) -> CadenceWindow(start, end, midpoint, duration,
     coverage, gaps)`: merge all sites' intervals, fold modulo `period`, find the largest gap on the
     circle; the window is its complement, midpoint its centre. Working on absolute datetimes over a
     ≥2-day horizon and folding makes the midnight wrap disappear and generalises to non-daily
     cadences. `midpoint`/`duration` map directly onto an LCO cadence request
     (`period`, `jitter = duration`, first window anchored at `midpoint`).
   - Unit tests with the worked example above.

4. **Template tag** — `nonsidereal_visibility(target, ...)` in `src/templatetags/fomo_extras.py`
   calling 2→3 and returning a Plotly div: `go.Bar(base=start, x=duration, orientation='h')` per
   site, a "combined" row from `coverage`, a vertical line at `midpoint`, hover text with exact
   times. Include on the target-detail page next to the existing TOM plots.

## Suggested first PR

Steps 1 and 3 together (extraction + pure helper with tests). Neither needs the heavy pipeline to
test, and 1 is the enabler for everything else. Branch off `main` (not `issue29-…`) so the PR stays
independent of the Scout review; rebase after that merges if needed.

## Status (2026-09-18) and follow-ups

Steps 1–4 are implemented on `feature/visibility-windows`. Deviations from the design above: the
template tag is `nonsidereal_target_plan` in `solsys_code/templatetags/visibility_extras.py`
(`src/templatetags/` is not registered with Django), the Plan form carries a checkbox list of all
six LCO sites (all selected by default), the sampling interval is coarsened for long ranges
(`sampling_interval`, ≤300 samples per site) and the ASSIST simulation is rebuilt per site (cheap
compared with the per-sample integration).

To revisit:

- **Fixed colour per site.** Neither TOM's `target_plan`/`observation_plan` nor our figures pin a
  colour to a site; both take Plotly's colorway in trace order, so colours shift when sites are
  deselected or greyed out. Add a colour to each `LCO_SITES` entry and use it in `airmass_figure`
  and `cadence_figure` (plus a fixed colour for the "All sites" row).
- **Window edges.** `visibility_windows` uses the first/last *valid sample*, so each window is
  underestimated by up to one interval and a single-sample run draws as a zero-width bar; pad by
  half an interval at each end if that matters.
- **Rapidly moving NEOs.** No test exercises a close-approaching object (all ephemeris/visibility
  tests use (33933), a main-belt asteroid). Add a fixture near a close approach to check that
  the coarsened sampling interval for long ranges does not miss short windows, and that the
  per-sample light-time/ASSIST integration behaves through the encounter.
- **Render time.** A 7-day, six-site plan takes ~11 s; the per-row `build_apco_context` in
  `compute_ephemeris` is the obvious optimisation target.
