# Stack Research

**Domain:** Multi-facility (LCO+SOAR) observation-record sync, per-record API enrichment, robust HTTP mocking
**Researched:** 2026-06-18
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

No new core technologies are needed. v1.3 is additive to the existing v1.2 stack — every new capability is satisfied by libraries already vendored transitively through `tomtoolkit`/`tom_observations`, already installed in this environment, and already used elsewhere in this codebase.

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `tomtoolkit` | `3.0.0a9` (pinned `>=3.0.0a9` in `pyproject.toml`) | Supplies `tom_observations.facilities.lco.LCOFacility` and `tom_observations.facilities.soar.SOARFacility` | Already the project's pinned TOM Toolkit alpha (required by `tom_jpl`/`tom_fink`, see `pyproject.toml` comment). `SOARFacility(LCOFacility)` is a direct subclass — same `OCSFacility.get_observation_status()`, same `_portal_headers()`, same OCS `/api/requests/<id>/observations/` endpoint shape. No version bump needed. |
| `requests` | `2.32.3` (transitive, via `tomtoolkit`) | HTTP client for the per-record LCO Observation Portal call | This is **already the library `OCSFacility`/`LCOFacility` use internally** (`tom_observations/facilities/ocs.py`'s module-level `make_request()` wraps `requests.request`). Reuse the facility's own method rather than hand-rolling a second `requests.get` call — see Integration Points below. |

### Supporting Libraries

None required. Do not add `httpx`, `aiohttp`, or any async HTTP client — the sync command is a synchronous, single-threaded `BaseCommand`, consistent with the rest of `solsys_code/management/commands/`.

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `unittest.mock` (stdlib) | builtin | Mock the per-record LCO API call in tests | Always, for the new `get_observation_status`/`_portal_headers`/`requests.request` call path — see Mocking Approach below. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `ruff check .` / `ruff format --check .` | Lint/format gate (existing) | No new rules needed; multi-config-scanning loops and dict-mapping code are ordinary Python, nothing exotic for ruff to flag. |
| `./manage.py test solsys_code` | Django test runner (existing) | New tests for proposal-list/`ALL` parsing, multi-facility selection, `c_1..c_5` scanning, and the per-record API call (mocked) all belong here — DB-dependent (`ObservationRecord`, `CalendarEvent`). |

## Installation

No installation step required — `requests` and `tomtoolkit` (and therefore `LCOFacility`/`SOARFacility`) are already installed in this environment and already imported by `sync_lco_observation_calendar.py`.

```bash
# Nothing to add. Confirm versions already present:
pip show tomtoolkit requests
# tomtoolkit  3.0.0a9
# requests    2.32.3
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|--------------------------|
| Call `LCOFacility().get_observation_status(observation_id)` (or `SOARFacility()`'s inherited version) for the per-record enrichment call | Hand-roll a new `requests.get(f'{portal_url}/api/requests/{id}/observations/', headers=...)` directly in the management command | Only if you need a response shape `get_observation_status()` doesn't already return (it does — `state`, `scheduled_start`, `scheduled_end` come straight from the same `/observations/` blocks endpoint this milestone needs site/enclosure/telescope from). Since the existing method only *extracts* a subset of fields and discards `site`/`observatory`/`telescope`, the practical answer is: call the facility's own authenticated request helper (`_portal_headers()` + `make_request()`) but parse the **raw JSON blocks yourself** to pull the discarded fields, rather than introducing a parallel HTTP call path. See Integration Points. |
| `unittest.mock.patch` against `requests.request`/`requests.get` | `responses` or `requests-mock` PyPI packages | Only if test setups become complex enough (e.g. many sequential calls with different URLs needing route-based matching) that manual `Mock()`/`side_effect` wiring gets unwieldy. Not needed here — this project already mocks `requests.get` directly in `solsys_code/tests/test_views.py` (`JPLSBDBQuery` tests), and the new call is a single GET per record. Introducing a new test-mocking dependency for one call site is not justified. |
| Filtering `ObservationRecord.objects.filter(facility__in=['LCO', 'SOAR'])` plus a `proposal_list`/`ALL` Q-object branch | Django `JSONField` lookups (`parameters__proposal__in=[...]`) | `parameters` is a `TextField`, not `JSONField` (confirmed in milestone context) — `parameters__proposal` lookups only work today because SQLite's JSON1-via-text-cast happens to support simple key equality through Django's JSON path lookup shim; do not assume `__in` or nested config-key (`c_1_instrument_type`) lookups work the same way at the DB level. Filter on `facility` (a real column) at the DB level, then parse `parameters` (JSON-decode) and apply proposal/`c_1..c_5` logic in Python — exactly the path this milestone's context already specifies ("no DB-level JSON filtering, must filter/parse in Python"). |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|--------------|
| A new HTTP client library (`httpx`, `aiohttp`, `urllib3` directly) | `requests` is already the project- and TOM-Toolkit-wide standard; introducing a second HTTP stack for one call site adds a dependency with zero benefit and breaks the "reuse the facility's request pattern" goal. | `requests` via `LCOFacility`'s own `_portal_headers()` + the shared `make_request()` helper in `tom_observations.facilities.ocs`/`lco`. |
| `responses` / `requests-mock` / `vcrpy` for the new tests | Not currently a project dependency; this codebase's existing convention (`test_views.py`) is plain `@patch('requests.get')` + `Mock()`. Adding a new mocking library for a single new call site is unjustified dependency growth. | `unittest.mock.patch` targeting either `requests.request` (what `make_request()` calls) or, more precisely, `tom_observations.facilities.ocs.make_request` itself (patch where it's *looked up*, i.e. in the `ocs` module, since `lco.py`/`soar.py` don't redefine `get_observation_status`). |
| A flat `instrument_type`/`site` key read from `ObservationRecord.parameters` (the v1.2 approach) | Confirmed broken against real DB data — real LCO submissions are multi-configuration cadence requests (`c_1_instrument_type`..`c_5_instrument_type`); a flat read raises `KeyError` on every real record. | Scan `c_1` through `c_{max_configurations}` (`OCSSettings.default_settings['max_configurations'] == 5`), and for each configuration that exists in `parameters`, check whether any `c_N_ic_1..5_exposure_time` is populated (truthy/non-zero); use the first configuration whose exposure time is populated as the "meaningful" one. |
| Hardcoding `max_configurations=5`/`max_instrument_configs=5` as magic numbers scattered in the sync command | `OCSSettings.default_settings` already defines these as the canonical OCS limits (`'max_instrument_configs': 5, 'max_configurations': 5`), and `LCOFacility().facility_settings.get_setting('max_configurations')` / `get_setting('max_instrument_configs')` reads them live. | Read the bounds from `facility.facility_settings.get_setting(...)` instead of inlining `range(1, 6)` — keeps the scan correct if LCO/OCS ever changes the limit, and self-documents *why* it's 5. |
| Building a second `requests` session / connection pool for the per-record call | The command already constructs one `LCOFacility()` (and will now also need one `SOARFacility()`) per run; each call to `get_observation_status()`/`make_request()` opens its own short-lived connection via `requests.request(...)`, which is the existing TOM Toolkit pattern (no session reuse there either). Introducing a `requests.Session()` for "performance" would diverge from upstream's pattern for no measured benefit at this record volume. | Call the facility instance's existing per-call `make_request` plumbing as-is; if N+1 HTTP calls per sync run become a real latency problem at higher record volumes, that is a separate, measured optimization for a future milestone — not a v1.3 concern. |

## Stack Patterns by Variant

**If extracting only state/scheduled_start/scheduled_end (what `get_observation_status()` already returns):**
- Call `facility.get_observation_status(record.observation_id)` directly — it already does the two-request dance (`/api/requests/<id>` for state, `/api/requests/<id>/observations/` for the block) and returns a clean `{'state', 'scheduled_start', 'scheduled_end'}` dict.
- Because this is the *intended*, already-tested upstream entry point; don't reinvent it for fields it already exposes.

**If extracting site/enclosure/telescope (what this milestone needs, and what `get_observation_status()` discards):**
- Do **not** call `get_observation_status()` — it does not return site/telescope, since it explicitly only extracts `block['start']`/`block['end']` from each block dict (`tom_observations/facilities/ocs.py:1561-1573`).
- Instead, replicate its *second* request only: `make_request('GET', urljoin(portal_url, f'/api/requests/{observation_id}/observations/'), headers=facility._portal_headers())`, and read the same block-selection logic (prefer a `'COMPLETED'` block, else the first `'PENDING'` one) but additionally pull whichever of `site`/`observatory`/`telescope`/`enclosure` keys the real OCS response actually contains. The exact field names should be confirmed against a live response when an API key becomes available (the OCS scheduler block schema commonly includes `site`, `observatory` (enclosure), and `telescope` per-block, paralleling the `siteid-enclid-telid` fully-qualified code used elsewhere in this milestone).
- Because: this milestone explicitly needs fields `OCSFacility.get_observation_status()` discards, while the auth/URL/error-handling plumbing it uses is exactly what should be reused — write a small project-local helper (e.g. `_fetch_block_location(facility, observation_id)` in `sync_lco_observation_calendar.py`) that imports and calls `make_request`/`_portal_headers` from the `tom_observations.facilities.ocs` module rather than duplicating header-building or status-code logic.

**If the LCO API call fails (network error, 401/403 `ImproperCredentialsException`, 4xx `ValidationError`, blank `api_key` in this dev environment):**
- Catch `(requests.exceptions.RequestException, ImproperCredentialsException, forms.ValidationError)` around the helper call (these are exactly the three failure modes `make_request()` itself can raise — `ImproperCredentialsException` for 401-403, `django.forms.ValidationError` for 400, and `response.raise_for_status()` → `requests.exceptions.HTTPError`/`ConnectionError`/`Timeout` for everything else).
- On any of those, fall back to the coarse instrument-class label (`1m0`/`0m4`/`2m0`) parsed from whichever `c_N_instrument_type` was selected as "meaningful" (e.g. regex `r'(\d[mM]\d)'` against strings like `1M0-SCICAM-SINISTRO` or `0M4-SCICAM-SBIG`).
- Because: the dev environment's blank `api_key` means `_portal_headers()` returns `{}` (no `Authorization` header) and the live LCO portal will reject the request with 401/403 — this is the **expected, untestable-live path in this environment**; the fallback must be exercised by unit tests via mocking, not by a live call, and the command must never raise — it should log/skip-degrade per record, consistent with the existing `(KeyError, ValueError)` skip-and-continue pattern already in `_build_event_fields`'s caller.

**If supporting `--proposal ALL`:**
- Treat `ALL` as "no proposal filter" — i.e., skip the `parameters__proposal` filter clause entirely (do not literally search for a proposal named `"ALL"`).
- Because: there's no project codes named `ALL` in MPC/LCO proposal-naming conventions, and a magic sentinel string is the simplest, most explicit way to express "sync everything" without inventing a separate `--all` boolean flag that interacts awkwardly with `--proposal`.

**If supporting a comma-separated `--proposal` list:**
- Split on `,`, strip whitespace, and use `parameters__proposal__in=[...]` (this is a flat top-level key lookup, which Django's SQLite JSON path shim already supports per the v1.2-shipped `parameters__proposal=proposal` exact-match filter) — `__in` is the same kind of single-level JSON-path lookup, just with an `IN` instead of `=`, so it stays inside what's already proven to work for this field.
- Because: avoids a Python-side `OR` loop over the queryset when the database can do it in one query; only need to drop to Python-side filtering for the *nested* `c_1..c_5` instrument-config scan, which the JSON1 path shim cannot do.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|------------------|-------|
| `tomtoolkit==3.0.0a9` | `tom_jpl`, `tom_fink` (per existing `pyproject.toml` comment) | Pre-existing constraint from PR #38 (merged 2026-06-11); do not bump within this milestone — `SOARFacility`/`LCOFacility`'s relevant methods (`get_observation_status`, `_portal_headers`, `get_observation_url`, `get_failed_observing_states`) are stable across the 2.31.x → 3.0.0a9 line referenced in this codebase; no breaking signature changes observed for any method this milestone touches. |
| `tom_observations.facilities.soar.SOARFacility` | `tom_observations.facilities.lco.LCOFacility` | `SOARFacility(LCOFacility)` — direct subclass; only overrides `__init__` (default `facility_settings=SOARSettings('SOAR')`) and `get_form()`. Crucially, it does **not** override `get_observation_status()`, `_portal_headers()`, or `get_observation_url()` — those are inherited unchanged from `OCSFacility`/`LCOFacility`, confirming the milestone's premise that LCO and SOAR share the same OCS API/parameter shape for sync purposes. |
| `FACILITIES['SOAR']` in `src/fomo/settings.py` | `FACILITIES['LCO']` | SOAR's docstring states it "uses the LCO API key", and `SOARSettings(LCOSettings)` inherits the same `default_settings` (`portal_url='https://observe.lco.global'`, same `api_key` setting key). **Action needed in settings, not code**: confirm/add a `'SOAR'` entry to `FACILITIES` in `src/fomo/settings.py` (can point at the same `portal_url`/`api_key` as `'LCO'`) — `SOARSettings.get_setting()` looks up `settings.FACILITIES['SOAR']`, falling back to `OCSSettings.default_settings` only for keys missing from that dict, so a missing `'SOAR'` entry entirely would silently use the *built-in* `portal_url`/blank `api_key` defaults rather than erroring — functionally harmless here since both point at the same portal, but worth an explicit settings entry for clarity. |
| `requests==2.32.3` | `tomtoolkit==3.0.0a9` | `requests` is a transitive dependency, not directly pinned in this project's `pyproject.toml`; no compatibility risk — same version already in use for `JPLSBDBQuery` in `solsys_code/views.py`. |

## Integration Points (read this before writing code)

- **Reuse, don't reimplement, `LCOFacility`'s request plumbing.** `tom_observations.facilities.ocs.OCSFacility._portal_headers()` builds the `Authorization: Token <api_key>` header (or `{}` if blank), and the module-level `make_request()` function in the same file wraps `requests.request(...)`, translating 401-403 → `tom_common.exceptions.ImproperCredentialsException` and 400 → `django.forms.ValidationError`, then calls `response.raise_for_status()` for everything else. Import and call these directly (`from tom_observations.facilities.ocs import make_request`) from the new per-record enrichment helper in `sync_lco_observation_calendar.py`, passing `facility._portal_headers()` for headers — this is a private (`_`-prefixed) method but is the established intra-package access pattern other facility modules use too (`lco_redirect.py` calls `self.lco_facility._portal_headers()` directly).
- **`facility.facility_settings.get_setting('portal_url')`** is the correct source for the portal base URL — do not hardcode `'https://observe.lco.global'` a second time in the sync command; both `LCOFacility()` and `SOARFacility()` resolve to the same portal URL through their respective settings classes, but going through `get_setting()` keeps the command correct if `FACILITIES['SOAR']['portal_url']` is ever pointed elsewhere.
- **Per-facility instance, not per-record.** Construct one `LCOFacility()` and one `SOARFacility()` per command invocation (mirroring the existing single `facility = LCOFacility()` in `handle()`), and dispatch to the matching instance based on `record.facility` (`'LCO'` vs `'SOAR'`) when building event fields and making the enrichment call — both share `get_observation_url()`/`get_failed_observing_states()` behavior, so the existing `_failure_prefix`/`_title_for` helpers need no changes beyond accepting either facility instance (their type hints can stay `LCOFacility` since `SOARFacility` *is* an `LCOFacility`).
- **Mocking target for tests:** patch `tom_observations.facilities.ocs.make_request` (the function actually invoked, looked up in the module where the new helper code calls it from), or patch `requests.request` directly (one level lower, matching the existing `@patch('requests.get')` convention in `test_views.py` — note the existing code uses `requests.get`, but `OCSFacility.make_request()` calls the more general `requests.request`, so patch `requests.request`, not `requests.get`). Build the mock with `Mock(status_code=200, json=Mock(return_value=[{...block...}]))` matching the real `/observations/` endpoint's list-of-blocks JSON shape (a list of dicts, each with at least `state`, `start`, `end`, and — needs confirming live — `site`/`observatory`/`telescope` keys). For the fallback-path tests, set `mock_request.side_effect = requests.exceptions.ConnectionError` (or return a `Mock(status_code=403)` to drive `ImproperCredentialsException`) and assert the coarse `1m0`/`0m4`/`2m0` label is produced instead.
- **No `responses`/`requests-mock` dependency to add** — confirmed by the existing `test_views.py` convention; stay consistent.

## Sources

- Installed package inspection (`pip show tomtoolkit requests`) — `tomtoolkit==3.0.0a9`, `requests==2.32.3`, confidence HIGH (direct inspection of this environment, not web-search-only).
- `tom_observations/facilities/ocs.py` (installed package source, read directly) — `OCSFacility.get_observation_status`, `make_request`, `_portal_headers`, `OCSSettings.default_settings` (`max_configurations`/`max_instrument_configs` = 5), confidence HIGH.
- `tom_observations/facilities/lco.py` (installed package source, read directly) — `LCOSettings.default_settings` (`portal_url`, `archive_url`, `api_key`), confidence HIGH.
- `tom_observations/facilities/soar.py` (installed package source, read directly) — `SOARFacility(LCOFacility)`, `SOARSettings(LCOSettings)`, confirms no override of `get_observation_status`/`_portal_headers`/`get_observation_url`, confidence HIGH.
- `solsys_code/tests/test_views.py` (this repo, read directly) — existing `@patch('requests.get')` + `Mock()` HTTP-mocking convention for `JPLSBDBQuery`, confidence HIGH.
- `src/fomo/settings.py` (this repo, read directly) — current `FACILITIES` dict has `'LCO'` and `'GEM'` entries but no `'SOAR'` entry yet, confidence HIGH.
- `pyproject.toml` (this repo, read directly) — `tomtoolkit>=3.0.0a9` pin with explanatory comment, confidence HIGH.
- WebSearch ("tomtoolkit PyPI release history 2026") — confirms a parallel stable `2.31.x`/`2.32.x` release line exists alongside the `3.0.0a*` alpha series this project deliberately tracks; not relevant to change the existing pin, confidence MEDIUM (web search, used only to corroborate the alpha-vs-stable timeline, not to drive any recommendation).

---
*Stack research for: multi-facility (LCO+SOAR) observation sync, multi-proposal filtering, per-record API enrichment with fallback*
*Researched: 2026-06-18*
