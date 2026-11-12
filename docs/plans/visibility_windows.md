# Plan: multi-site visibility windows and cadence midpoint for non-sidereal targets

Written 2026-09-16 for a fresh Claude Code session (no prior context), from a design discussion held
in the production checkout (`../fomo_fresh`, which stays on `issue29-jpl-scout-ingest` ingesting
Scout objects — do not work there). Everything below was verified on this machine on that date.

## Priority context (read first)

Checked against the "FOMO Quick Requirements" Google Doc
(https://docs.google.com/document/d/1UvdnzIXkKmsiWkHy5EHjJDd9yB1RkrdzQ-aKtzims9E/). This
visibility/cadence tool is **not** on the requirements list (nearest item: "Observing Tools → ETC"
under Full Requirements). Things the doc ranks higher:

1. Brokers (MVP "Add new objects → Brokers"): Tim is adding LSST/TAP support to the ALeRCE broker in
   tom_base, branch `origin/enhancement/alerce-add-tap-and-lsst` (in the tom_base repo, not here).
   FOMO-side wiring (`TOM_ALERT_CLASSES` in `src/fomo/settings.py`) follows once that lands.  **(still) in progress with ALeRCE support; awaiting tom_base PR #1686**.
2. Hosting/deployment (Tim's action items, overdue since week of 2026-08-17): **partly done**.
   PR #58 (2026-10) added the single-host gunicorn + systemd deployment and it is running at
   `tlister-thinkmate.lco.gtn:6729`, but that is only reachable inside the LCO network/VPN. The
   requirement is an externally visible deployment, which is still outstanding.
3. MPC observations → `ReducedDatums`: in flight on `feature/add_mpc_obs` (22 commits ahead of
   `main`, last 2026-10-23; see `docs/plans/ades_processor_port.md` and `ades_reduced_datum.md` on
   that branch). **Blocked on a design decision upstream**, tom_base issue
   [#1685](https://github.com/TOMToolkit/tom_base/issues/1685) (opened 2026-09-23, open):
   - Tim proposed a new `ADESReducedDatum` model with FKs to the `AstrometryReducedDatum` and
     `PhotometryReducedDatum` each ADES row splits into, a `JSONField` for the remaining ADES
     columns, and asked whether `stn`, `astCat`/`rmsCorr`, `band` vs `fltr`, `photCat` and
     `photAp` should become real columns on the existing Datums.
   - phycodurus (2026-09-30) agreed a linking model is needed, since the reverse accessors from
     `DataProduct` don't record which astrometry row pairs with which photometry row.
   - **Joey Chatelain (jchate6, 2026-10-14) pushed back**: make each ADES line its own
     `DataProduct` ("ADESRow"), linked to by one `PhotometryReducedDatum` and one
     `AstrometryReducedDatum`; keep the extra ADES fields in each datum's JSON `value` (duplicated
     where needed); pairing then comes for free via the DataProduct's reverse FK. A bespoke
     `ADESReducedDatum` is only justified if those fields need indexing/filtering. FOMO could still
     build one locally on `ReducedDatumCommon` without a tom_base change.
   - **Not yet answered.** The questions to settle in the reply: whether a DataProduct per ADES
     *row* (not per file) is acceptable for exports with thousands of rows; whether cut/filter
     access to `seeing`, `rmsFit`, `nStars`, `photAp` (comet multi-aperture photometry) is enough
     to justify indexed columns; and whether the per-row approach can round-trip to a PSV export.
     The `feature/add_mpc_obs` plan currently assumes the `ADESReducedDatum` + tom_base
     `DataProcessor.post_process` hook shape, so the answer changes that branch.
4. Target matching / metadata storage (provisional ID ↔ final designation, target extras) — should be
   settled as part of (3).

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

## Status (2026-09-18) and follow-ups

Steps 1–4 are implemented on `feature/visibility-windows`. Deviations from the design above: the
template tag is `nonsidereal_target_plan` in `solsys_code/templatetags/visibility_extras.py`
(`src/templatetags/` is not registered with Django), the Plan form carries a checkbox list of all
six LCO sites (all selected by default), the sampling interval is coarsened for long ranges
(`sampling_interval`, ≤300 samples per site) and the ASSIST simulation is rebuilt per site (cheap
compared with the per-sample integration).

To revisit:

- **Fixed colour per site** (open; flag in the PR for reviewers to prioritise). Neither TOM's `target_plan`/`observation_plan` nor our figures pin a
  colour to a site; both take Plotly's colorway in trace order, so colours shift when sites are
  deselected or greyed out. Add a colour to each `LCO_SITES` entry and use it in `airmass_figure`
  and `cadence_figure` (plus a fixed colour for the "All sites" row).
- **Window edges** (done, `c59d8b9b`): `get_nonsidereal_visibility` returns a third per-site element,
  each sample's margin in degrees to its tightest limit (target altitude or Sun altitude, see
  `observability_margins`), and `visibility_windows` places each edge where that margin, interpolated
  linearly between the last valid and first invalid sample, crosses zero. The earlier midpoint rule
  was wrong by up to half a sampling interval (measured: up to 7.4 min at 15 min, 17.5 min at the
  35-min interval a 7-day plan coarsens to), not the ~20 s / ~2 min this document previously
  claimed. Linear interpolation measures at ≤3.3 s (15 min) and ≤18 s (35 min) against 1-minute
  sampling for two main-belt/NEO targets from LSC, and within 7 s for 2025 FA22 the night after
  close approach. Solving the Sun crossing separately (`_find_crossing` from PR #43) plus a
  quadratic for the target would reach ~1 s but was judged not worth the dependency. The
  two-element TOM shape is still accepted and falls back to the midpoint. Windows shorter than one
  sampling interval can still be missed entirely.
- **Rapidly moving NEOs** (done): `TestCloseApproach2025FA22` in `test_ephem_utils.py` checks the
  geocentric ephemeris of 2025 FA22 against JPL Horizons through its 0.0056 au approach on
  2025-09-18 (agreement < 0.01", 1e-10 au) and that LCO visibility switches on the night after
  closest approach. The IAWN geometry file for the campaign differs from Horizons by up to 218"
  at closest approach (older orbit solution), so it is not used as the reference. Geocentric
  ephemerides need sorcha >= 1.2.1 (dirac-institute/sorcha#1203), which closes FOMO issue #27.
- **Render time** (done, `b22cf751`): `build_apco_context` was ~70% of `compute_ephemeris`, almost
  all of it two IERS table lookups (UT1−UTC and polar motion, ~1.1 ms each) and time scale
  conversions repeated per row; the erfa calls were ~6%. It now accepts the whole pointings frame
  and is called once per site, taking a 7-day six-site plan from 6.4 s to 1.8 s with identical
  output. Considered and not done: feeding TT instead of TDB to `epv00`/`apco` (changes Alt/Az by
  <0.01 µas but saves only ~15 ms per site once vectorised) and disabling the IERS auto-download
  check (~0.5 ms per site, and it would stop the table refreshing for future dates). The remaining
  time is spread over the per-row light-time integration, the pandas `iterrows` loop and the CSV
  round trip, with no single dominant cost.

## Handoff (2026-11-11)

Written so a fresh session on any machine can continue from `docs/plans/visibility_windows.md` alone.

**Branch state.** `feature/visibility-windows` is pushed and merged with `main` (`d367282e`, which
brought in PRs #58, #67, `refresh-obscodes`, #44 and the dependabot bumps #60–#65; one trivial
`pyproject.toml` conflict, resolved keeping `sorcha>=1.2.1`). On top of that: `dac7669f` guards the
`autoapi-skip-member` hook in `docs/conf.py` so esbonio's partial Sphinx can initialise, and
`53eec341` refreshed the priority context above. `b22cf751` (vectorised apco contexts) and `c59d8b9b`
(interpolated window edges) followed on 2026-11-11 and are **not yet pushed**. The full
non-functional test suite (171 tests) passes (pre-commit hook). **No PR has been opened yet.**

**Before opening the PR.**

- Push the branch.
- Flag the one open "To revisit" item, fixed colour per site, in the PR body for reviewers to
  decide its priority. The other two (window edges, render time) are done.
- `src/templates/tom_targets/target_detail.html` is a full copy of tom_targets' template with two
  changes (`{% load visibility_extras %}`, `{% nonsidereal_target_plan %}` beside `{% target_plan %}`).
  Add a comment naming the tomtoolkit version it was copied from (3.1.x) so it gets re-diffed on
  upgrades.
- Closes issue #27 (geocentric ephemerides); say so in the PR body. `issue27-geocenter-ephem` can be
  deleted afterwards.

**Next after this branch.** `feature/add_mpc_obs` (22 commits ahead of `main`, no conflicts
expected) needs `git merge origin/main`, and a reply to tom_base
[#1685](https://github.com/TOMToolkit/tom_base/issues/1685) before its data model can be settled;
see item 3 under "Priority context".

**Local environment (not in git).**

- VS Code over Remote-WSL: esbonio otherwise runs the venv's python with `-S` against its bundled
  Sphinx and cannot import `sphinx-autoapi`/`nbsphinx`. Fix is in the *remote* user settings:
  `"esbonio.sphinx.pythonCommand": ["${command:python.interpreterPath}"]`, and in the gitignored
  `.vscode/settings.json`: `"esbonio.sphinx.configOverrides": {"nbsphinx_execute": "never"}` so
  background rebuilds don't execute the notebooks (one imports `solsys_code.ephem_utils`).
- `pandoc` must be installed for nbsphinx to render the notebooks' markdown cells.
- The venv must be reinstalled (`pip install -e .[dev]`) after the dependabot merges; `whitenoise`
  in particular is now a hard dependency of the test suite.
