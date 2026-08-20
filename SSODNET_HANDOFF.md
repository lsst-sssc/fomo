# SsODNet DataService — handoff notes

This branch (`ssodnet`) has a scaffold for pulling a Target's ssoCard from IMCCE's
[SsODNet](https://ssp.imcce.fr/webservices/ssodnet/) service and showing it on the
Target detail page. Nothing is functional yet — every real piece is a `TODO`.

## What's here

- `solsys_code/ssodnet.py` — `SsODNetDataService(DataService)` class stub. The two
  methods to actually implement are `build_query_parameters_from_target()` and
  `query_service()`.
- `src/templates/solsys_code/partials/ssodnet_card.html` — placeholder template for
  rendering the card on the target detail page.
- `src/templatetags/solsys_code_extras.py` — `ssodnet_card` inclusion tag (feeds the
  template above).
- `solsys_code/apps.py` — registers the service in `data_services()` and the card in
  `target_detail_buttons()`, same mechanism the Ephemeris button uses.
- `src/fomo/settings.py` — placeholder `DATA_SERVICES['SsODNet']` config comment.
- `solsys_code/tests/test_ssodnet.py` — trivial scaffold tests; needs real coverage.
- `pyproject.toml` — added `space-rocks` (import name `rocks`) as a dependency. Run
  `pip install -e .` (or your usual dev install) to pick it up before working on this.

## Why this shape

- **Not like `tom_jpl` / our own `JPLSBDBQuery`.** Those *search* an external
  catalog and create new Targets from matches. This is the opposite: enrich an
  *existing* Target with data, similar to how `tom_fink.FinkDataService` looks up
  ZTF alerts for a target via `build_query_parameters_from_target()` →
  `query_service()`. See https://github.com/TOMToolkit/tom_fink for reference.
- **`rocks` for the actual fetch.** The Fink portal's own SsODNet card (different
  framework, same data) uses `rocks.Rock(name)` rather than hand-rolling the
  quaero + ssoCard HTTP calls:
  https://github.com/astrolabsoftware/ztf.fink-portal.org/blob/master/apps/sso/cards.py
  (`get_sso_data()` / `card_sso_rocks_params()`) — worth using as a reference for
  both the fetch logic and what fields are worth surfacing.
- **Rendering doesn't fit the DataService UI's built-in assumptions.** ssoCard data
  is nested properties + references, not a photometry/spectroscopy time series
  (`to_reduced_datums`) and not a new-target search result. The plan is to call
  `query_service()` directly from the `ssodnet_card` template tag and let
  `target_detail_buttons()` inject the rendered partial — the same pattern already
  used for the Ephemeris button — rather than relying on the DataService's own
  query-form/results UI.

## Before writing the real logic

Confirm the exact `DataService` method signatures against what's actually installed
(this scaffold was written from TOM Toolkit docs + `tom_fink` as reference, not from
reading the installed source directly):

    python -c "import tom_dataservices.dataservices as m; help(m.DataService)"

## Status

All three original open questions are resolved:

- No `DATA_SERVICES['SsODNet']` config needed (public API).
- `target.name` resolves fine via quaero as-is.
- No-card case: the card shows "No SsODNet data available for this target."

`query_service()` and the rendering path (`build_card_context()` in
`solsys_code/ssodnet.py`, wired through the `ssodnet_card` template tag and
partial) are implemented, matching the fields the Fink portal's own SsODNet card
shows: name/number, orbital class, parent body, dynamical system, then physical
parameters -- taxonomy, absolute magnitude (H), slope parameter (G), diameter,
albedo -- each with its SsODNet reference(s), linked out to ADS by bibcode.

**Confirmed working end to end (2026-08-20):** `pip install -e .` to pick up
`space-rocks`, `manage.py runserver`, open a Target detail page -- the card
renders alongside the Ephemeris button via `target_detail_buttons()`. Note that
spot is really a button toolbar, not a content section, so a full card sitting
there may look visually squeezed -- worth revisiting the layout once there's more
than one physical-parameters card, but functionally it works.

**Watch out for:** `{% if %}` on these values must use `!= None`, not plain
truthiness -- `G` (slope parameter) and `albedo` can legitimately be `0.0`, which
is falsy in both Python and Django templates but is NOT the same as "missing"
(only `NaN`, cleaned to `None` by `_clean_float()`, means missing). Covered by
`test_zero_slope_parameter_is_not_treated_as_missing` in `test_ssodnet.py`.

## Deliberately deferred to a follow-up pass

Only orbital-class/physical "at a glance" fields are shown for v1 -- dynamical
properties (proper elements, MOID, Yarkovsky, family membership, ...) were
excluded by design (not wanted for this card), and mass/density weren't
requested. Spin is deferred because it's structurally more complex than a single
value+error+bibref -- it's a *list* of possibly-multiple pole/period solutions,
each with its own references, and colors (B-V, g-i, ...) are a whole dict of band
pairs rather than a single field -- both deserve their own pass rather than
bolting onto the current simple-field pattern.

If these get added, extend `build_card_context()` in `solsys_code/ssodnet.py` and
the corresponding block in `ssodnet_card.html`.
