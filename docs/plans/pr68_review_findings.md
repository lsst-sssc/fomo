# PR #68 code review findings

Hand-off for a fresh session. These are the findings of a `/code-review max` run on
[PR #68](https://github.com/lsst-sssc/fomo/pull/68) ("Multi-site visibility windows and cadence
midpoint for non-sidereal targets"), branch `feature/visibility-windows` at `b937bd34`, compared
with `origin/main` at `fc9ced0b`. The review ran on 2026-11-12.

**Status: none of these are fixed.** Nothing was changed in the repo by the review. Line numbers
refer to `b937bd34` and will drift as fixes land; tick items off here as they are done.

The companion document is `docs/plans/visibility_windows.md` (the design and hand-off for the
branch itself).

## How to use this

- Work on `feature/visibility-windows` (run `git branch --show-current` first). Do not work in
  `../fomo_fresh`.
- Each finding has a reproduction. Turn it into a failing test before fixing, in
  `solsys_code/tests/test_visibility.py`, `test_views.py` (`TestNonsiderealTargetPlan`) or
  `test_ephem_utils.py`.
- The standard test target is (33933); its elements are in
  `solsys_code/tests/test_plan_panel_functional.py`.
- Run `python manage.py test --exclude-tag functional` and `ruff check . && ruff format --check .`
  before committing.

## Suggested order

1. Findings 2, 4, 5, 6 and 7: small, and on or near the default path.
2. Findings 8 and 3: correctness of the reported window.
3. Finding 1: needs Tim's decision (process-wide lock or single-threaded gunicorn workers).
4. Findings 9 to 12, then 13 to 15.

This order is a suggestion from the session that ran the review, not something Tim has agreed.

## Findings, most severe first

### 1. Unsynchronised SPICE calls from threaded workers — `solsys_code/ephem_utils.py:566`

- [ ] Not fixed

`compute_ephemeris` makes thousands of unsynchronised CSPICE calls (sorcha's
`barycentricObservatoryRates`, `convert_target_to_layup`), and the Plan panel runs it for up to six
sites inside the target-detail request. Two overlapping computations in one multi-threaded gunicorn
worker corrupt SPICE's global state.

- `deploy/gunicorn.conf.py` sets `worker_class = 'gthread'`, `threads = 4`; `runserver` is threaded
  too.
- Two Plan requests, or a Plan and an `/ephem/` request, overlapping in one worker (two users, two
  tabs, a double-click) give one of:
  - spurious `SpiceUNPARSEDTIME` / `SpiceBADTIMESTRING` 500s;
  - a plan with silently wrong samples (one sample off by 15° of margin);
  - CSPICE aborting the whole worker with `SPICE(BADSUBSCRIPT)` (exit 134, every in-flight request
    lost).
- Reproduced by calling `get_nonsidereal_visibility` from 2 and 4 threads; serial runs are
  repeatable.
- The race pre-dates the PR in the old Ephemeris view (about 20 rows per request). The PR raises
  exposure to 600–1800 rows (0.3–2 s) per GET on the target page, reachable anonymously for `OPEN`
  targets.
- Fix: a process-wide lock around the SPICE/ASSIST section, or single-threaded workers.

### 2. Same start and end date gives a 500 — `solsys_code/templatetags/visibility_extras.py:193`

- [ ] Not fixed

Entering the same Start and End date yields one sample. If it is observable, `visibility_windows`
returns a zero-width interval that passes `if intervals:` in the tag, but `cadence_window` filters
it out and raises `ValueError`.

- Reproduce: `GET /targets/<pk>/?start_time=2025-05-10&end_time=2025-05-10&airmass=3` for (33933)
  gives `ValueError: No intervals to build a cadence window from` (`visibility.py:216`).
- The form only rejects end < start. `_run_interval` does not extend a run that is both the first
  and last sample. The tag's guard tests list non-emptiness while the callee tests `end > start`.
- Picking the same date in both boxes is the natural way to ask for one night.
- When that single 00:00 UTC sample is not observable (e.g. `&sites=cpt`, where the window opens at
  17:14 UT) the page instead says "Target is not observable from the selected sites in this date
  range", which is also misleading.

### 3. Long ranges invent windows — `solsys_code/templatetags/visibility_extras.py:126`

- [ ] Not fixed

`sampling_interval` coarsens the step without limit and nothing caps the date range. For
multi-month ranges the step exceeds the nightly window (then a whole day), and `visibility_windows`
joins valid samples from different nights into multi-day "windows". The cadence window is bogus and
there is no warning.

Reproduce with (33933), `sites=lsc`, airmass 2.5, start 2025-05-10:

| Range | Step | Reported window |
|---|---|---|
| 1, 7, 30 days | up to 145 min | 1.7 h |
| 120 days | 580 min | 2.2 h |
| 180 days | | 2.8 h |
| 300 days | 1440 min (every sample at 00:00 UTC) | 24.0 h, "00:00–00:00 UTC, midpoint 12:00 ± 12.0 h" |
| 365 days | | 13.0 h |

`docs/visibility.rst` only says longer ranges "work but are sampled more coarsely" and that short
windows can be missed, not that false ones appear.

### 4. Blank airmass means 10, not 2.5 — `solsys_code/templatetags/visibility_extras.py:168`

- [ ] Not fixed

The 2.5 default applies only when the `airmass` key is absent from the URL. The form renders the
box empty and submits `airmass=`, which cleans to `None` and falls through to the helpers' default
of 10. This contradicts `docs/visibility.rst` ("Defaults to 2.5").

- Reproduce: press Plan with Maximum Airmass left blank (its initial state).
  `?start_time=2025-05-10&end_time=2025-05-11&airmass=&sites=cpt` plots samples up to airmass 9.9
  (about 5.7° altitude) and reports "17:14–20:30 UTC (3.3 h)".
- The same URL with `airmass=2.5`, or without the key, reports "17:14–18:45 UTC (1.5 h)".
- No test sends `airmass=`.

### 5. No error handling around the pipeline — `solsys_code/templatetags/visibility_extras.py:179`

- [ ] Not fixed

The tag calls the ephemeris pipeline with no error handling and no validation beyond date order, so
any failure inside it turns the whole target detail page into a 500. Before the PR the page showed a
static "not supported" note.

- A `NON_SIDEREAL` target without `perihdist` / `epoch_of_perihelion`: pressing Plan raises
  `ValueError: Input values did not match the format class mjd` from `convert_target_to_layup`
  (`ephem_utils.py:102`). TOM requires only `mean_anomaly` and `semimajor_axis` for
  `MPC_MINOR_PLANET`, so manually created, CSV-imported or API-created targets qualify.
- A mistyped year, e.g. `start_time=2205-05-10`, or any date outside roughly 1850 to August 2099:
  `SpiceFRAMEDATANOTFOUND` / `SpiceSPKINSUFFDATA` from `ephem_utils.py:566`.
- The failing parameters stay in the bookmarkable query string.

### 6. Docs give the wrong cadence start — `docs/visibility.rst:71`

- [ ] Not fixed

The docs tell users to set the LCO cadence request's start to `midpoint - jitter/2`. The LCO portal
centres every cadence window on `start + n*period` (± `jitter/2`, clipped at start), so that recipe
shifts each window half its width early. It also contradicts the `CadenceWindow` docstring ("first
window anchored at midpoint").

- Example: single-site window 23:45–08:00 (8.25 h, midpoint 03:52). Following the docs (start 23:45,
  period 24, jitter 8.25) LCO generates windows 19:37–03:52 each night. The second half of every
  real window is never requested and the first half of each requested window is unobservable.
- The start should be the midpoint itself.
- Checked against observation-portal's `expand_cadence_request`
  (`window_start = max(request_window_start - half_jitter, cadence['start'])`, then
  `request_window_start += period`).

### 7. Airmass 0 or negative gives a 500 — `solsys_code/visibility.py:103`

- [ ] Not fixed

`observability_margins` evaluates `asin(min(1.0, 1.0 / airmass_limit))` on an unvalidated user
value; the inherited `DecimalField` has no `min_value`.

- `?start_time=2025-05-10&end_time=2025-05-11&airmass=0` gives `ZeroDivisionError` via
  `ephem_utils.py:686`. A user might type 0 for "no limit".
- `airmass=-0.5` gives `ValueError: math domain error`.
- The same rule is encoded a second time in `airmass_samples`, which accepts these values (all
  `None`). For limits ≤ −1 the two disagree silently (samples `None`, margins positive).
- Fix: one function deriving both outputs, the limit validated once, and `min_value` on the form
  field.

### 8. Edge interpolation is wrong when the binding limit changes — `solsys_code/visibility.py:105`

- [ ] Not fixed

Window edges are found by linearly interpolating `min(target clearance, Sun clearance)` between two
samples. That is wrong whenever the binding limit differs between the two samples: the edge lands
next to the valid sample. Short twilight-limited windows are shortened by up to nearly a whole
sampling interval, not the "few seconds" / "about 20 s" the docstring and docs claim.

- Pure-function case at 35-minute sampling with exactly linear altitudes (target 31.0° → 30.1°
  against a 30° limit, Sun −17.5° → −24.5°): the true window is 36.4 min long; `visibility_windows`
  reports 9.7 min (start 26.7 min late).
- With real ephemerides for (33933):
  - LSC, airmass limit 1.85, 35-minute step: start 29 min late.
  - CPT, limit 1.8, 15-minute step: end 278 s early.
  - 2.3% of edges are off by more than 60 s at 35 minutes.
- The error always shortens the window and hits the short evening/morning windows of
  low-elongation targets.
- Fix: interpolate each clearance separately and take the later (earlier) crossing. Then correct
  the accuracy claims in the docstring and `docs/visibility.rst` if they still do not hold.

### 9. Database writes while rendering a GET — `solsys_code/templatetags/visibility_extras.py:38`

- [ ] Not fixed

`lco_observatories` runs `get_or_create` while rendering a GET. The first Plan request by anyone
(anonymous visitors included, for `OPEN` targets) inserts `Observatory` rows under real MPC codes
with invented names, a blank `short_name` and the LCO facility's rounded coordinates. Those rows
then block the MPC-sourced record.

- After one Plan GET the table holds e.g. `('W85', 'Cerro Tololo-LCO', short_name '', -30.167,
  -70.805, 2198 m)`.
- Observatories → Create for W85 then fails with "already exists" before MPC is queried. The row is
  offered in the Ephemeris site list and titles ephemerides "at  (W85)".
- `name` is unique but only in `defaults`, so an existing row called "Sutherland-LCO" under another
  code makes the request fail with `IntegrityError`.
- An existing row with NULL coordinates raises `TypeError` at `ephem_utils.py:678`.
- `get_service_class('LCO')()` at line 32 runs even when all six rows exist, so removing LCO from
  `TOM_FACILITY_CLASSES` is an `ImportError` 500.
- Unsaved `Observatory` instances work with both functions, so no write is needed.

### 10. One-day plans draw "All sites" and the midpoint a day late — `solsys_code/templatetags/visibility_extras.py:85`

- [ ] Not fixed

For a one-day plan the cadence chart draws the "All sites" row and the dashed midpoint line up to a
day later than the per-site bars they summarise, beyond the end of the sampled range.
`cadence_window` anchors the folded window at `t0 + start_phase` and `cadence_figure` plots it on
the same absolute time axis as the raw per-site windows.

- (33933), 2025-05-10 → 2025-05-11, airmass 3, LSC only: the LSC bars are 10 May 00:00–01:28 and
  10 May 23:22–24:00, but "All sites" is drawn 10 May 23:22 → 11 May 01:28 and the midpoint at
  11 May 00:25, past the last sample. The real 00:00–01:28 bar has nothing under it.
- With all six sites every site bar lies inside 10 May, but "All sites" runs to 11 May 10:43 with
  the midpoint at 11 May 01:59.
- The summary and annotation print HH:MM only, so the day shift is invisible in text.
- Two-day ranges line up. One night is the documented usual choice.

### 11. Sample grid drops the last sample and fails across leap seconds — `solsys_code/ephem_utils.py:671`

- [ ] Not fixed

The sample grid is built with astropy `Time` arithmetic.
`int(((Time(end_time) - start) / (interval * u.min)).decompose()) + 1` truncates a ratio that is
often a hair under the exact integer and drops the last sample. Stepping in SI seconds makes
`times.datetime` (line 684) raise for any range that crosses a leap second.

- `get_nonsidereal_visibility(target, sites, datetime(2025,5,10,12), datetime(2025,5,11,2), 15)`
  returns 56 samples ending 01:45 instead of 57 ending 02:00 (ratio 55.99999999999999).
  18:00 → 19:00 gives 4, not 5.
- Whole-day ranges are unaffected, so the form only hits this with a time of day in the URL, but
  the function is documented for direct use.
- `?start_time=2016-12-31&end_time=2017-01-01` gives `ValueError: Time 2017-01-01 00:00:00 is
  within a leap second but datetime does not support leap seconds` (500).
- Fix: build the grid from datetimes with integer arithmetic.

### 12. Empty airmass plot on coarse plans — `solsys_code/templatetags/visibility_extras.py:63`

- [ ] Not fixed

`airmass_figure` uses `mode='lines'`, which draws nothing for a valid sample with `None` on both
sides. Once the step is coarsened every nightly run can be a single sample, so the airmass plot is
empty although the trace is `visible=True` and a cadence window is reported under it.

- 30-day plan (145-minute step, the case `docs/visibility.rst` describes) for (33933) from LSC at
  airmass 2.5: 18 valid samples in 18 single-sample runs; rendered in headless Chromium the trace
  has zero drawn length and no markers.
- 1- and 7-day plans (runs of 2–5 samples) draw normally.
- `mode='lines+markers'` would show them.

### 13. Defects carried into the public `compute_ephemeris` — `solsys_code/ephem_utils.py:632`

- [ ] Not fixed

These pre-date the PR but are now part of a public function.

- `abs_mag or 22.0` / `slope or 0.15` replace legitimate zeros. A target with G = 0.0 (which
  ingestion deliberately preserves) is computed with G = 0.15, and H = 0.0 becomes 22.0.
- The returned DataFrame holds `sky_motion` as an astropy `Quantity` column: `df.to_csv()` raises
  `TypeError`, `df.copy().iterrows()` raises `UnitConversionError`, and `df['sky_motion'].tolist()`
  raises `NotImplementedError`. The view only works because it iterates the untouched frame.
- `column_types = defaultdict(...).setdefault(float)` (line 548) is `None`, so dtypes are
  re-inferred from the CSV round trip; a target named "33933" comes back with `ObjID` as int64.
- A scalar `Time` raises `IndexError` from `add_sky_motion`.

### 14. Template override can drift from upstream — `src/templates/tom_targets/target_detail.html:1`

- [ ] Not fixed

A 156-line verbatim copy of tomtoolkit 3.1.1's `target_detail.html` is added to change three lines,
while `pyproject.toml` allows any `tomtoolkit>=3.1.0` and nothing detects drift.

- Templates in `DIRS` win over app templates. After any tomtoolkit upgrade the copy silently
  shadows upstream's page for every target, sidereal included. Upstream fixes and new tabs never
  appear, and a renamed tag becomes a `TemplateSyntaxError` on every target page.
- The copy already had to be re-synced once inside this PR (`ff7dd835`).
- Alternatives: mount the panel through TOM's `target_detail_tabs` app hook (works with the
  existing partial and tag function, but TOM calls the context function twice per page so it needs
  memoising), or keep the copy and add a test that diffs it against the installed template.

### 15. Each Plan response inlines two more copies of plotly.js — `solsys_code/templatetags/visibility_extras.py:196`

- [ ] Not fixed

Both `offline.plot(..., output_type='div')` calls use the default `include_plotlyjs=True`.

- The non-sidereal target page is 9.17 MB without a plan (two bundles from upstream tags) and
  18.3 MB with one (four bundles).
- Passing `include_plotlyjs=False` for the cadence figure (the airmass div directly above it has
  already loaded Plotly) saves 4.57 MB. Both figures were checked to still draw.

## Smaller items, cut for the 15-finding cap

- **Geocentre keyed on a string.** `obscode == '500'` in `ephem_utils.py:507` and `forms.py:33`.
  MPC codes 244 and 248 have the same zero parallax constants and get a bogus Az/El by URL.
- **Per-site recomputation.** `get_nonsidereal_visibility` (`ephem_utils.py:680`) redoes
  site-independent work for each site. A reviewer's prototype that shares it ran 5.8× faster with
  identical output (494 ms → 85 ms for seven nights, six sites).
- **Rotation figure in the `compute_ephemeris` docstring.** It quotes 11.7′ and points to the
  test's recipe. That recipe pushes the pole through TETE to ICRS and so includes annual
  aberration. The pure precession-nutation rotation is 12.1′ and the docstring's own formula gives
  12.05′. The roughly 4% seasonal bias is hidden by the test's 0.002″/h tolerance. This bears on
  the ICRF versus of-date rate question the PR raises for reviewers.
- **Docs wording in `docs/visibility.rst`.**
  - "Short holes" understates the gaps: the six-site example has 4.5 h of holes, and its midpoint
    01:59 falls inside a 01:28–03:10 gap.
  - The `:class:` reference cannot resolve.
  - The code example raises `DoesNotExist` on a fresh database.
- **Test weaknesses.**
  - `assertAnglesWithin(..., scale=cos_alt)` skips NaN rows because the scale is a pandas Series.
  - `test_visibility_switches_on_after_close_approach` does not assert what its name says; LSC
    already has a window on 18 Sep 07:11–09:17, starting before closest approach.
  - The Playwright legend count passes with no line drawn.
  - The (33933) Target fixture is pasted four times.
- **`docs/plans/visibility_windows.md`.** It is a session hand-off containing an internal hostname
  and stale statements (175 tests; "not checked in a browser").
- **`CLAUDE.md`.** Its Layout note that the `src/templatetags/` libraries are "registered via
  INSTALLED_APPS" is false. That pre-dates this PR, which works around it without updating the
  note.
- **Minor.**
  - `visibility_windows` does not length-check `margins`.
  - `sampling_interval` can give 301 samples.
  - `cadence_figure` hard-codes the `qualitative.Plotly` colours.
  - Unticking every site selects all six.
  - `PlaywrightTestCase.setUpClass` leaks the env var if the browser launch fails.

## Checked and found fine

- The `compute_ephemeris` extraction is output-identical to the old view body.
- The vectorised `build_apco_context` matches the per-row path.
- The template override differs from tomtoolkit 3.1.1 only in the intended lines.
- Nothing still imports names removed from `solsys_code.views`.
- The `docs/conf.py` guard is correct in a real build.
- `ruff check` and `ruff format --check` pass.
- Both browser test classes are tagged `functional`.
- The six LCO MPC codes are right.

## How far to trust these

- Ten finder agents and a final gap sweep ran. The reviewer did **not** run the separate
  per-finding verifier pass the recipe calls for, after a rate-limit interruption.
- Findings 2, 3, 4, 5, 7, 8, 10, 11, 12, 13 and 15 were reproduced by the coordinating reviewer's
  own experiments (Django test client on an in-memory database, or the pure functions).
- Finding 1 was reproduced by three finder agents but not rerun by the coordinator.
- Finding 6 was checked against the public LCO observation-portal source, not against a live
  cadence request.
- Finding 9 was reproduced by finder agents; the coordinator confirmed the code path by reading.
- Finding 14 is a maintenance cost, confirmed by diffing the copy against the installed template.
- The experiment scripts were written to a session scratch directory under `/tmp`
  (`/tmp/claude-10007/-home-tlister-git-fomo/f503f9cb-c7db-4391-b132-57eb48c6ab6a/scratchpad/`,
  subdirectories `exp1`–`exp5`, `angle_*`, `sweep`). They are not in the repo and may be gone;
  rebuild reproductions from the descriptions above.
