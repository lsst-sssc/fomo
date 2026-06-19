# Architecture Research

**Domain:** Django management command generalization — multi-facility, multi-proposal sync of robotic-telescope queue observations into a local calendar model, backed by an existing third-party OCS/LCO REST client library (`tom_observations.facilities.{ocs,lco,soar}`).
**Researched:** 2026-06-18
**Confidence:** HIGH — every claim below is grounded in direct reads of the installed `tom_observations` package source (`ocs.py`, `lco.py`, `soar.py`), the project's own `sync_lco_observation_calendar.py` and its test suite, `src/fomo/settings.py`'s `FACILITIES` dict, and a live Django ORM query confirming `facility__in` + `parameters__proposal__in` JSON-field filtering. No speculative library behavior is asserted.

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                  Management Command Layer                            │
│  solsys_code/management/commands/sync_lco_observation_calendar.py    │
│  ┌────────────────────┐   ┌──────────────────────────────────────┐   │
│  │ Command.handle()    │   │ _build_event_fields() / _time_window()│   │
│  │ (arg parse, query,  │──▶│ _title_for() / _failure_prefix()      │   │
│  │  create-or-update   │   │ _derive_telescope() [NEW: + API path] │   │
│  │  loop, per-record   │   └──────────────────┬───────────────────┘   │
│  │  try/except)        │                      │                       │
│  └─────────┬────────────┘                      │                       │
│            │ ObservationRecord queryset         │ facility instance    │
├────────────┴────────────────────────────────────┴───────────────────┤
│                  Facility Adapter Layer (third-party, reused)        │
│  tom_observations.facilities.lco.LCOFacility                         │
│  tom_observations.facilities.soar.SOARFacility(LCOFacility)          │
│  tom_observations.facilities.ocs.OCSFacility / OCSSettings           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ facility.get_observation_url(id)   (existing, already reused)  │  │
│  │ facility.facility_settings.get_setting('portal_url'/'api_key') │  │
│  │ facility._portal_headers()         (existing, NOT yet reused)  │  │
│  │ ocs.make_request(method, url, ...) (existing, NOT yet reused)  │  │
│  └────────────────────────────────────────┬───────────────────────┘  │
├──────────────────────────────────────────┴───────────────────────────┤
│                  External Service / Persistence Layer                │
│  ┌──────────────────┐         ┌────────────────────────────────┐    │
│  │ LCO Observation   │ HTTPS  │ tom_calendar.models.CalendarEvent│    │
│  │ Portal REST API   │◀──────▶│ (idempotency key: url)           │    │
│  │ /api/requests/... │        │ tom_observations.models.         │    │
│  │ (shared by LCO+   │        │ ObservationRecord (read-only,    │    │
│  │  SOAR — same      │        │ facility/parameters JSON field)  │    │
│  │  portal_url)      │        └────────────────────────────────┘    │
│  └──────────────────┘                                                │
└────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Current State (v1.2) | v1.3 Change |
|-----------|----------------|------------------------|-------------|
| `Command.handle()` | Parse `--proposal`, build the `ObservationRecord` queryset, loop + create-or-update | Single facility (`'LCO'`), single proposal (`parameters__proposal=proposal`) | Modify: parse comma-list/`ALL`, `facility__in=['LCO','SOAR']`, `parameters__proposal__in=[...]` (no filter at all for `ALL`) |
| `_derive_telescope()` | Map a site identifier to a telescope label | Maps bare 3-letter `site_code` via 2-entry dict (`coj`/`ogg`) | **Replace its lookup data + add a new caller path**: still the single function other code calls, but now receives a fully-qualified `siteid-enclid-telid` code (from the new per-record API helper) or a fallback class label, looked up in an 8-site verified dict |
| `_build_event_fields()` | Assemble all `CalendarEvent` kwargs from one record | Calls `record.parameters['site']` and `record.parameters['instrument_type']` directly (both confirmed broken) | Modify: call new `_extract_instrument_type(record)` instead of flat key; call new `_resolve_telescope_label(record, facility)` instead of `_derive_telescope(record.parameters['site'])` directly |
| `_extract_instrument_type()` | **NEW** — scan `c_1..c_5_instrument_type` configs, return the one with a populated `exposure_time` | N/A | New pure function, no I/O, easy to unit test in isolation |
| `_resolve_telescope_label()` | **NEW** — orchestrate the per-record LCO API call + fallback + label lookup | N/A | New function; this is the seam between `_build_event_fields` and the network call — keeps `_build_event_fields`'s call shape unchanged (still one call, still returns a label string or raises) |
| `_fetch_request_site_codes()` | **NEW** — the actual authenticated HTTP call to `/api/requests/<id>/observations/`, returns the fully-qualified site code or `None` | N/A | New function; thin wrapper around `ocs.make_request` + facility's own headers/portal_url, isolated so it's the only function that needs HTTP mocking in tests |
| `LCOFacility` / `SOARFacility` | Provide `get_observation_url()`, `get_failed_observing_states()`, `facility_settings` (portal_url/api_key), `_portal_headers()` | Already instantiated once per run, reused for `get_observation_url()` and `get_failed_observing_states()` | **Reuse further**: same shared instance now also supplies `facility_settings.get_setting('portal_url')` and `_portal_headers()` for the new API call — no new facility instantiation needed, but now you need **two** instances (one `LCOFacility()`, one `SOARFacility()`) since a SOAR record must hit `https://observe.lco.global` through `SOARSettings` (same portal, but correct `name`/API key resolution path) |

## Recommended Project Structure

No new files needed — this is a generalization of one existing command module, not a new module. Keep everything in:

```
solsys_code/
├── management/
│   └── commands/
│       └── sync_lco_observation_calendar.py   # all new + modified functions live here
└── tests/
    └── test_sync_lco_observation_calendar.py  # extend with new test classes/cases
```

### Structure Rationale

- **Single-file command, no new module:** the command is ~225 lines today; even with the new helper functions it stays well under a size where splitting into a package is worth the indirection. Django management commands in this codebase (see `fetch_jplsbdb_objects`) are consistently single-file. Introducing `solsys_code/lco_site_map.py` or similar would create a second source of truth to keep in sync with the command's docstrings/comments — not worth it for one static dict plus two small functions.
- **Static mapping dict stays inline, at module level**, exactly where `SITE_TELESCOPE_MAP` and `_FAILURE_PREFIX_BY_STATUS` already live — consistent with the existing pattern of "verified reference data as a module constant with a comment citing its source."

## Architectural Patterns

### Pattern 1: Reuse the facility object's existing authenticated-request machinery, do not hand-roll `requests` calls

**What:** `tom_observations.facilities.ocs.make_request(method, url, **kwargs)` is a thin wrapper around `requests.request()` that already handles LCO/OCS-specific HTTP error semantics (401-403 → `ImproperCredentialsException`, 400 → `forms.ValidationError`, else `raise_for_status()`). `OCSFacility._portal_headers()` already builds the `Authorization: Token <api_key>` header from `facility_settings.get_setting('api_key')`. `OCSFacility.get_observation_status()` (in `ocs.py:1548-1575`) is the existing, in-library precedent for exactly this kind of per-record call to `/api/requests/<id>/observations/` — it already does `make_request('GET', urljoin(portal_url, f'/api/requests/{id}/observations/'), headers=self._portal_headers())` and parses the JSON array of "blocks."

**When to use:** Any time the command needs to hit the LCO/SOAR portal API per-record. Always prefer this over a raw `requests.get(...)` call.

**Trade-offs:**
- Pro: zero new dependency surface (no new `requests` import needed in the command — `ocs.make_request` is already a transitive dependency); inherits the library's error-status handling (`ImproperCredentialsException`/`ValidationError`/`HTTPError`) for free, which the command can catch alongside the existing `(KeyError, ValueError)` skip-and-log path; matches the codebase's existing pattern of "wrapper services abstract external library complexity" (per CLAUDE.md's Architecture section).
- Con: `ocs.make_request` is a **module-level function**, not a method — it must be imported directly (`from tom_observations.facilities.ocs import make_request`) rather than called as `facility.make_request(...)`. This is a minor naming surprise but not a blocker.
- Con: `make_request` raises on 4xx/5xx rather than returning `None` — the new helper function must catch `requests.exceptions.RequestException` (covers `HTTPError`, `ConnectionError`, `Timeout`) plus the library's own `ImproperCredentialsException`/`forms.ValidationError` and treat all of them as "fall back to coarse label," consistent with the milestone's stated fallback requirement.

**Example:**
```python
from tom_observations.facilities.ocs import make_request

def _fetch_request_site_codes(record: ObservationRecord, facility: LCOFacility) -> str | None:
    """Call the LCO API for one record's placed observation block site/enclosure/telescope.

    Returns the fully-qualified 'siteid-enclid-telid' code (e.g. 'coj-clma-2m0a'),
    or None if the call fails or no block has a usable site/observatory/telescope triplet.
    """
    portal_url = facility.facility_settings.get_setting('portal_url')
    try:
        response = make_request(
            'GET',
            urljoin(portal_url, f'/api/requests/{record.observation_id}/observations/'),
            headers=facility._portal_headers(),
        )
    except Exception as exc:  # network error, 4xx/5xx, credentials issue -- all fall back
        logger.debug(f'Site-code lookup failed for observation_id={record.observation_id!r}: {exc}')
        return None
    for block in response.json():
        site, enclosure, telescope = block.get('site'), block.get('observatory'), block.get('telescope')
        if site and enclosure and telescope:
            return f'{site}-{enclosure}-{telescope}'
    return None
```

### Pattern 2: Two-tier resolution function (precise lookup with a coarse fallback), kept as a thin orchestrator so `_build_event_fields`'s call shape doesn't change

**What:** Introduce `_resolve_telescope_label(record, facility) -> str` as the single new call site inside `_build_event_fields`, replacing `_derive_telescope(record.parameters['site'])`. Internally it: (1) calls `_fetch_request_site_codes()`, (2) if that returns a fully-qualified code, looks it up in the new verified dict, (3) if either step fails/misses, falls back to a coarse instrument-class label derived from the (now correctly extracted) instrument type string.

**When to use:** Whenever a precise external lookup might fail and the system has a name for "good enough" already-available data to fall back to — this is the same shape as `Observatory.from_parallax_constants()`'s fallback pattern already flagged as an anti-pattern in this codebase (silent `None`s), so the new version must **not** silently return `None` — it must always return *something* usable for the calendar title, even if it's the coarse `'1m0'`/`'0m4'`/`'2m0'` label.

**Trade-offs:**
- Pro: keeps `_build_event_fields`'s shape — it still calls one function and gets one string back, same as today's `telescope = _derive_telescope(record.parameters['site'])`. No restructuring of `_build_event_fields` itself, satisfying the quality gate directly.
- Pro: isolates all network-call error handling in one place, away from the pure-function `_build_event_fields`, which (per its existing docstring contract) only raises `KeyError`/`ValueError` for genuinely-skip-worthy data problems — a flaky network call should never cause a record to be skipped when a fallback label is available.
- Con: this function now does I/O where the rest of `_build_event_fields`'s helpers (`_title_for`, `_time_window`) are pure — but this is unavoidable given the milestone's requirement, and is the smallest possible surface for that I/O (one function, one new test double needed: mock `_fetch_request_site_codes`, not the full `make_request`/`requests` chain, in most `_build_event_fields`-level tests).

**Example:**
```python
def _resolve_telescope_label(record: ObservationRecord, facility: LCOFacility, instrument: str) -> str:
    """Resolve a telescope label for a record: precise API lookup, else coarse fallback.

    Args:
        record: the ObservationRecord being synced.
        facility: a shared LCOFacility or SOARFacility instance for this record's facility.
        instrument: the already-extracted instrument_type string (used for the fallback class).

    Returns:
        str: a telescope label -- either the precise SITE_TELESCOPE_MAP value for the
            fully-qualified site-enclosure-telescope code, or a coarse instrument-class
            label ('1m0'/'0m4'/'2m0') if the API call or lookup fails.
    """
    fq_code = _fetch_request_site_codes(record, facility)
    if fq_code is not None and fq_code in SITE_TELESCOPE_MAP:
        return SITE_TELESCOPE_MAP[fq_code]
    return _coarse_instrument_class_label(instrument)
```

### Pattern 3: Per-facility dispatch via a small lookup dict, not branching logic threaded through the query/loop

**What:** Rather than `if record.facility == 'LCO': ... elif record.facility == 'SOAR': ...` scattered through `handle()`, build a `{'LCO': LCOFacility(), 'SOAR': SOARFacility()}` dict once at the top of `handle()` and select `facility = facility_by_name[record.facility]` inside the loop. Both classes share the exact same `get_observation_url`, `get_failed_observing_states`, `_portal_headers`, `facility_settings.get_setting('portal_url')` interface (SOAR subclasses LCO and only overrides site-listing/weather methods, confirmed by reading `soar.py`), so every downstream helper function (`_title_for`, `_failure_prefix`, `_resolve_telescope_label`) keeps taking a single `facility` parameter — no new parameter, no new branch needed in any helper.

**When to use:** Any time a loop needs to select between a small fixed set of structurally-identical adapter objects keyed by a string already present on the row being processed (`record.facility`).

**Trade-offs:**
- Pro: zero changes to every helper function's signature; only `handle()` changes, by building the dict and indexing into it instead of constructing one `LCOFacility()`.
- Pro: trivially extensible if Gemini/ESO are added later (explicitly out of scope for v1.3, per `PROJECT.md`, but the dict shape costs nothing to leave open).
- Con: none significant — SOAR's `facility_settings` correctly resolves `portal_url`/`api_key` from `FACILITIES['SOAR']` if present, or its own `SOARSettings.default_settings` (same `https://observe.lco.global` portal, same env-var-sourced API key) if absent, confirmed by reading `soar.py:234-262` — no new `settings.py` entry is strictly required, though adding an explicit `FACILITIES['SOAR']` entry mirroring `FACILITIES['LCO']` is cheap insurance against future drift.

**Example:**
```python
FACILITY_CLASSES = {'LCO': LCOFacility, 'SOAR': SOARFacility}

def handle(self, *args, **options):
    proposals = _parse_proposal_option(options['proposal'])  # list[str] | None (None == ALL)
    facilities = {name: cls() for name, cls in FACILITY_CLASSES.items()}

    qs = ObservationRecord.objects.filter(facility__in=FACILITY_CLASSES.keys())
    if proposals is not None:
        qs = qs.filter(parameters__proposal__in=proposals)

    for record in qs:
        facility = facilities[record.facility]
        ...
```

## Data Flow

### Request Flow (per-record sync)

```
Command.handle()
    ↓ build queryset: facility__in=[...], parameters__proposal__in=[...] or unfiltered for ALL
ObservationRecord queryset (DB read, unchanged ORM pattern)
    ↓ per record
_build_event_fields(record, facility)
    ↓                              ↓                            ↓
_extract_instrument_type(record)  _resolve_telescope_label(...)  _time_window(record)  [unchanged]
    ↓ pure, scans                  ↓ orchestrates:                ↓ unchanged
  c_1..c_5_instrument_type        _fetch_request_site_codes()
  for populated exposure_time      ↓ HTTP GET via ocs.make_request
                                   /api/requests/<id>/observations/
                                    ↓ on success: fully-qualified code → SITE_TELESCOPE_MAP lookup
                                    ↓ on failure/miss: _coarse_instrument_class_label(instrument)
    ↓ all three feed into dict returned by _build_event_fields (shape unchanged: same 8 keys)
CalendarEvent.objects.get_or_create(url=..., defaults=fields)  [unchanged create-or-update logic]
```

### Key Data Flows

1. **Filtering generalization:** today's flow is `proposal: str` → single `=` filter on one facility string. The v1.3 flow is `--proposal` (comma-list or `ALL`) parsed into `list[str] | None` → `facility__in=['LCO','SOAR']` always applied, `parameters__proposal__in=proposals` applied only when not `ALL`. Confirmed live against this project's SQLite/JSON1 backend that both `__in` lookups compose correctly with the JSON-field extraction Django already generates for `parameters__proposal`.
2. **Instrument extraction:** today's flow reads one flat key that does not exist in real data (`record.parameters['instrument_type']`, confirmed via live DB rows to be absent). The v1.3 flow scans `c_1_instrument_type` through `c_5_instrument_type`, and for whichever index has a populated `c_N_ic_1_exposure_time` (or equivalent — exact sub-key needs confirming against the two known real records' full parameter dumps before finalizing this helper, see Open Questions below), returns that config's `instrument_type` value.
3. **Telescope-label resolution:** today's flow is a synchronous, zero-I/O dict lookup on a key that's never present (`record.parameters['site']` — confirmed absent in both real DB rows). The v1.3 flow adds one **new network round-trip per record** (a GET to the LCO portal), which is then mapped through a verified 8-site dict, with a no-network-required fallback path. This is the only part of the sync that introduces new I/O latency/failure modes — worth flagging for the roadmap as the part needing the most defensive test coverage (mocked success, mocked failure, mocked empty-blocks-list, mocked malformed-JSON).

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Tens of records (current proposal-scoped run) | Current design (one HTTP call per record, no caching, run synchronously in the management command) is entirely adequate. |
| Hundreds of records (`--proposal ALL` across many proposals) | The new per-record API call is the only scaling risk — N records means N additional HTTP round-trips to the LCO portal, on top of the existing per-record `get_observation_url` (free, no I/O) work. Consider a simple in-process cache keyed on `observation_id` only if the same run ever re-resolves the same id twice (it won't, since the queryset is row-per-record) — not needed initially. |
| Thousands of records / scheduled/cron-driven frequent runs | If this command starts running on a schedule (e.g. cron, not in v1.3 scope) against `ALL` proposals, consider batching the network calls or adding a short-TTL cache so a record whose `CalendarEvent` is unchanged doesn't re-trigger the API call on every run — but only the *site/telescope* lookup needs this; everything else is already no-churn (SYNC-04). This is explicitly a v1.4+ concern, not v1.3. |

### Scaling Priorities

1. **First bottleneck:** the new per-record LCO API call, once `--proposal ALL` is used against a proposal set with many records. Mitigate by making the fallback path cheap (no network) and by **not** re-fetching site codes for records whose `CalendarEvent` already exists and is unchanged in every other field — though note this optimization is incompatible with re-detecting an LCO scheduler *site reassignment* after the fact, so it should be a deliberate, documented trade-off if added, not silent.
2. **Second bottleneck:** none expected at this scale — the SQLite-backed JSON queries and create-or-update loop are unchanged in shape from v1.2, which already passed all idempotency/no-churn requirements.

## Anti-Patterns

### Anti-Pattern 1: Hand-rolling a new `requests.get(...)` call for the site/telescope lookup instead of reusing `ocs.make_request` + `facility._portal_headers()`

**What people do:** Add `import requests` to the command and write a fresh authenticated GET, duplicating the `Authorization: Token ...` header construction and status-code handling that `ocs.py` already implements.
**Why it's wrong:** Duplicates error-handling logic that's already correct and tested in the upstream library (`ImproperCredentialsException`/`ValidationError`/`raise_for_status`), creates a second source of truth for "how do I authenticate to the LCO portal," and silently diverges if the upstream library's auth scheme ever changes (e.g. a future OCS API key rotation mechanism).
**Do this instead:** Import `make_request` from `tom_observations.facilities.ocs` and call `facility._portal_headers()` / `facility.facility_settings.get_setting('portal_url')` exactly as `OCSFacility.get_observation_status()` already does internally. (Note: `_portal_headers` is a "private" leading-underscore method on the facility instance — calling it from outside the class is a minor layering violation but is the established precedent in this exact codebase area, since `get_observation_status()` itself is a public method built the same way; treat it as "package-internal API," acceptable for a sibling module within the same Django project.)

### Anti-Pattern 2: Restructuring `_build_event_fields`'s call shape to thread the facility-keyed dict or proposal list through every helper

**What people do:** When adding multi-facility support, change `_build_event_fields(record, facility)` to `_build_event_fields(record, facilities_by_name)` and have it do the `facilities_by_name[record.facility]` lookup itself, or similarly push the proposal-list-vs-`ALL` distinction down into the per-record helpers.
**Why it's wrong:** The quality gate explicitly calls for *not* restructuring `_build_event_fields`'s existing call shape. Pushing facility-selection logic down into it also means every existing test for `_build_event_fields` (which currently passes a bare `LCOFacility()` instance) would need to change its fixture shape, multiplying the blast radius of this change for no benefit — facility selection is a `handle()`-level concern (per-record dispatch, Pattern 3 above), not a `_build_event_fields`-level concern.
**Do this instead:** Resolve `facility = facilities[record.facility]` once per record inside the loop in `handle()`, and pass the **already-resolved single facility instance** into `_build_event_fields(record, facility)` exactly as today. Same for proposals: the comma-list/`ALL` parsing happens once in `handle()` (or a new small `_parse_proposal_option()` helper called from `handle()`), producing a `list[str] | None` that's consumed directly in the queryset filter — never passed into `_build_event_fields` at all, since it was never a parameter there to begin with.

### Anti-Pattern 3: Treating a failed/empty site-code API response as a skip-worthy error (raising `KeyError`/`ValueError` from `_resolve_telescope_label`)

**What people do:** Let `_fetch_request_site_codes`'s exception or "no usable block" case propagate up as a `KeyError` so the existing `except (KeyError, ValueError)` skip-and-log path in `handle()` catches it, on the theory that "no telescope label" is the same kind of problem as "no proposal key."
**Why it's wrong:** The milestone explicitly requires a **fallback label**, not a skip — a transient network blip to the LCO portal should never cause a real, validly-scheduled observation to vanish from the calendar. Skipping also silently regresses SYNC-04 (no-churn) semantics if the record's event already exists: a flaky API call on a re-run could cause the command to skip updating (or even appear to "lose") an event that was previously created successfully.
**Do this instead:** `_resolve_telescope_label` must always return a string (never raise for network reasons) — catch and log-at-debug inside `_fetch_request_site_codes`, return `None` on any failure, and let `_resolve_telescope_label`'s fallback branch handle it. Reserve the `KeyError`/`ValueError`-skip path for genuine, permanent data problems on the record itself (missing `proposal` key, unparseable `start`/`end`), consistent with the existing docstring contract on `_build_event_fields`.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| LCO Observation Portal REST API (`https://observe.lco.global`, shared by both `LCOFacility` and `SOARFacility`) | New per-record `GET /api/requests/<observation_id>/observations/` call, via `ocs.make_request(...)` + `facility._portal_headers()` + `facility.facility_settings.get_setting('portal_url')`. Mirrors the existing in-library `OCSFacility.get_observation_status()` pattern exactly (same endpoint, same auth). | Response is a JSON array of "blocks"; each block is expected to carry `site`/`observatory`/`telescope` keys (the fully-qualified `siteid-enclid-telid` triplet) based on the library's own internal precedent (`get_facility_status()`'s telescope-state keys use the same `site.enclosure.telescope` dotted/hyphenated convention) — **confirm the exact block JSON key names against a real API response or the LCO API docs before finalizing `_fetch_request_site_codes`'s parsing**, this is inferred from sibling code, not yet directly observed against this endpoint's actual payload. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `Command.handle()` ↔ facility adapter instances | Direct method calls (`get_observation_url`, `get_failed_observing_states`, `_portal_headers`, `facility_settings.get_setting`) on a `{name: instance}` dict built once per run | No new boundary type — same pattern as v1.2's single shared `LCOFacility()` instance, just keyed by `record.facility` now |
| `_build_event_fields()` ↔ new helpers (`_extract_instrument_type`, `_resolve_telescope_label`) | Direct function calls, same module, same file | `_build_event_fields`'s external call shape (parameters: `record`, `facility`; return: 8-key dict) is **unchanged** — only its internal body changes which functions it calls for `instrument` and `telescope` |
| `_resolve_telescope_label()` ↔ `_fetch_request_site_codes()` | Direct function call; the only function boundary in this whole change that crosses into network I/O | Isolating the network call behind this one function is what makes `_build_event_fields`-level tests stay simple (mock one function, not `requests`/`make_request` directly) while still letting a dedicated test suite mock `make_request`/`requests.request` itself for `_fetch_request_site_codes`'s own unit tests |

## Recommended Build Order

The quality gate specifically calls out the dependency between "fix instrument extraction" and "add telescope-label API+fallback." Here is the recommended order and why:

1. **`--proposal` comma-list/`ALL` parsing + `facility__in` query generalization** (pure `handle()`/queryset change, zero new I/O, no dependency on anything else). Land first because it's independently testable, lowest-risk, and unblocks testing every subsequent change against both LCO and SOAR fixture records in the same test run.

2. **`_extract_instrument_type()` (the `c_1..c_5_instrument_type` scan) — before the telescope-label work.** This must land second, not third, for two concrete reasons:
   - **It's a pure function with no I/O** — fully unit-testable against fixture `parameters` dicts (matching the real shape already confirmed against the two known DB rows) with zero mocking required. Landing the cheap, fully-deterministic fix first means the test suite has a stable, correct `instrument` value to depend on before the harder API-integration work begins.
   - **The telescope-label fallback path needs a correct instrument string.** Per `PROJECT.md`'s Active requirements, the fallback label is a *coarse instrument-class label* (`1m0`/`0m4`/`2m0`) derived from the instrument type. If `_extract_instrument_type` is still broken when `_resolve_telescope_label`'s fallback path is built, the fallback itself would be built and tested against wrong/absent data — forcing rework. Fixing extraction first means the fallback's only remaining job is "what's the coarse class for this *correct* instrument string," a much narrower and independently-verifiable task.

3. **`_fetch_request_site_codes()` (new HTTP call) + verified `SITE_TELESCOPE_MAP` (8-site dict, replacing the 2-entry `[ASSUMED]` one) + `_resolve_telescope_label()` orchestrator — built and tested together as one unit, last.** This is the highest-risk, highest-complexity piece (new network I/O, new error/fallback paths, a new static reference dict needing verification against the LCO sites-to-MPC-codes table already captured in `PROJECT.md`). Landing it last means:
   - It can be developed and tested with a known-good `instrument` value already available (depends on step 2).
   - It can be developed and tested against an already-generalized multi-facility queryset (depends on step 1), so SOAR-specific fixtures (`sor-...` site codes) are exercised from the start rather than retrofitted.
   - Its test suite is the most involved (mocked success / mocked HTTP failure / mocked empty-blocks / mocked malformed-JSON / fallback-label-matches-instrument-class) and benefits from not being entangled with simultaneous changes to filtering or instrument extraction — easier to review, easier to bisect if something breaks.

4. **`Command.handle()` per-facility dispatch dict (`{'LCO': LCOFacility(), 'SOAR': SOARFacility()}`)** — technically needed by step 1 (to build the `facility__in` query target list) but the *instance dict* itself (as opposed to just the filter's name list) is only load-bearing once step 3's per-record `facility` parameter needs to vary by record. Build the dict in step 1 (cheap, needed for the query anyway) but don't worry about exercising SOAR's `_portal_headers()`/`facility_settings` path end-to-end until step 3's tests.

**Net order:** filtering generalization → instrument extraction fix → telescope-label API+fallback (dict + HTTP helper + orchestrator, together). This matches the natural dependency chain (fallback needs correct instrument data; both depend on the queryset already covering both facilities) and front-loads the cheapest, most deterministic fixes before the riskiest new I/O.

## Open Questions / Items Needing Confirmation Before Finalizing the Plan

- **Exact JSON key names in `/api/requests/<id>/observations/` block responses.** This research infers `site`/`observatory`/`telescope` keys by analogy with `OCSFacility.get_facility_status()`'s telescope-state key format (`site.enclosure.telescope` dotted) and the `siteid-enclid-telid` reference already captured in `PROJECT.md`, but no live response from this exact endpoint was captured during this research pass. Recommend confirming against one real `observation_id` from this project's dev DB (e.g. obs_id=3781325, the COMPLETED record) before finalizing `_fetch_request_site_codes`'s parsing.
- **Exact `c_N_ic_M_exposure_time`-style sub-key** that indicates "this configuration is the populated one," per `PROJECT.md`'s note that "only the configuration(s) with a populated `exposure_time`" are meaningful — the precise key path (e.g. `c_1_ic_1_exposure_time` vs a simpler `c_1_exposure_time`) should be confirmed against the two known real records' full parameter dumps, not assumed from the `c_N_instrument_type` naming pattern alone.
- **Whether `FACILITIES['SOAR']` should be added explicitly to `src/fomo/settings.py`.** Not strictly required (verified `SOARSettings.default_settings` provides a working fallback), but worth a deliberate decision rather than relying on the fallback silently — especially since `LCO_API_KEY` env var is shared between `FACILITIES['LCO']['api_key']` (hardcoded empty string today) and `SOARSettings.default_settings['api_key']` (sourced from `os.getenv('LCO_API_KEY')`) — these two facilities currently resolve their API key from *different* sources in this project's settings, which is worth flagging explicitly during implementation.

## Sources

- `solsys_code/management/commands/sync_lco_observation_calendar.py` (this repo, v1.2 shipped code) — direct read, HIGH confidence
- `solsys_code/tests/test_sync_lco_observation_calendar.py` (this repo) — direct read, HIGH confidence
- `tom_observations/facilities/ocs.py` (installed package, `tomtoolkit` dependency, `/home/tlister/venv/fomo311_venv/lib/python3.11/site-packages/tom_observations/facilities/ocs.py`) — direct read of `make_request`, `OCSSettings.get_setting`, `OCSFacility.get_observation_status`, `OCSFacility._portal_headers`, `OCSFacility.get_facility_status` — HIGH confidence (primary library source, current installed version)
- `tom_observations/facilities/lco.py` (same install) — direct read of `LCOSettings`/`LCOFacility` class definitions and `default_settings` — HIGH confidence
- `tom_observations/facilities/soar.py` (same install) — direct read confirming `SOARFacility(LCOFacility)` subclassing and `SOARSettings(LCOSettings)` — HIGH confidence
- `src/fomo/settings.py` (this repo) — direct read of `FACILITIES` dict (`LCO` entry present, no `SOAR` entry) — HIGH confidence
- `.planning/PROJECT.md` (this repo) — milestone context, real-DB-data findings (obs_id=3780553/3781325, `c_1..c_5` cadence config shape), LCO site→MPC code reference table — HIGH confidence (already-verified project artifact)
- Live Django ORM query against this project's dev SQLite DB confirming `facility__in` + `parameters__proposal__in` JSON1 query composition — HIGH confidence (directly executed, not inferred)
- Web search for LCO `/api/requests/{id}/observations/` response schema — LOW confidence / inconclusive; no official field-level documentation surfaced beyond the in-library precedent already cited above; flagged as an open question rather than asserted as fact

---
*Architecture research for: FOMO v1.3 "Full LCO Facility Sync" milestone*
*Researched: 2026-06-18*
