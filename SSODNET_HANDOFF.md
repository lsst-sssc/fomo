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

## Open questions worth resolving early

- Does `DATA_SERVICES['SsODNet']` need any config at all (contact email for SsODNet's
  usage policy, timeouts), or is the public API enough as-is?
- Does `target.name` reliably resolve via SsODNet's quaero resolver, or does it need
  the same prefix handling `JPLSBDBQuery.create_targets()` applies on ingest
  (`solsys_code/views.py`)? Should a Target alias be tried as a fallback?
- What should the target detail page show when SsODNet has no card for an object
  (very new discoveries, etc.)?
