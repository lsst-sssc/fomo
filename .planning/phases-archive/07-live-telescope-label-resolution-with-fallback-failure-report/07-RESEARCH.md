# Phase 7: Live Telescope-Label Resolution with Fallback & Failure Reporting - Research

**Researched:** 2026-06-21
**Domain:** Per-record external API resolution with timeout/fallback discipline, in a Django management command (LCO Observation Portal API integration)
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Resolution timing (when the API call happens)**
- **D-01:** The per-record LCO API call is only attempted for **placed** records — i.e. `scheduled_start`/`scheduled_end` are populated. A queue-banner record (not yet scheduled) gets the coarse fallback label immediately, with **no API call attempted** — there is nothing to look up yet.
- **D-02:** The new SYNC-06 fallback/API-failure counter only increments on an actual failed/timed-out/unmapped-code API call for a **placed** record. A banner-stage record's coarse label does **not** increment this counter — it isn't a real failure, just "not yet resolvable." Conflating the two would make the new counter noisy and useless for spotting genuine API degradation.

**Verified static dict (TELESCOPE-01)**
- **D-03:** Collapse by **(site, aperture class)** pair, not by individual `enclid`/`telid`. A site with only one telescope class gets exactly one label; a site with multiple classes present gets one label **per class** — but multiple domes of the same class at the same site still collapse to one label.
- **D-04:** Label format is `SITECODE-CLASS`, hyphenated, site code uppercased, class token drawn from the exact same vocabulary as the coarse fallback (`1m0`/`0m4`/`2m0`) — e.g. `'LSC-1m0'`, `'LSC-0m4'`, `'CPT-1m0'`.
- **D-05:** Migrate the 3 existing entries too: `'coj'` (currently `'FTS'`) becomes `'COJ-2m0'`, `'ogg'` (currently `'FTN'`) becomes `'OGG-2m0'`, `'sor'` (currently `'SOAR'`) becomes `'SOR-<class>'` (confirm SOAR's real aperture class during research — likely `4m0`). This is an explicit, accepted one-time visible label change on already-synced historical `CalendarEvent`s at these sites. This dict (`SITE_TELESCOPE_MAP` in `sync_lco_observation_calendar.py`) is separate from Stage 1/2's `telescope_runs.py:SITES` dict — this migration does not touch that other dict or feature.
- All 8 real LCO-network sites per PROJECT.md's MPC-code reference table (`ogg`, `elp`, `lsc`, `cpt`, `coj`, `tfn`, `tlv`, `sor`) must be covered; exact per-site aperture classes present need confirming during research/planning.

**Fallback visibility (TELESCOPE-04)**
- **D-06:** Add a new title prefix, mirroring the existing `[QUEUED]`/`[EXPIRED]`/`[CANCELLED]`/`[FAILED]` convention — the calendar's day view only shows the truncated title at a glance.
- **D-07:** The new prefix applies **only** to a placed record whose API call genuinely failed/timed out/returned an unmapped code (same scope as D-02's counter). Banner-stage records keep their existing `[QUEUED]` prefix unchanged and do **not** get the new prefix.
- **D-08:** Prefix text is `[UNVERIFIED]`.
- **D-09 (open, Claude's discretion):** How `[UNVERIFIED]` combines with the existing terminal-state prefixes (`[EXPIRED]`/`[CANCELLED]`/`[FAILED]`, per Phase 4's D-04 priority rule that terminal prefixes beat `[QUEUED]`) if a record reaches a terminal state after an API-failure fallback was already applied. Likely the terminal prefix takes priority/replaces `[UNVERIFIED]` too, but the planner should confirm and document the exact combination rule explicitly.

**API call discipline (SYNC-08/09 — already locked by ROADMAP)**
- **D-10:** Explicit timeout of **10 seconds**, single attempt, no retry/backoff loop. There is no existing HTTP-timeout precedent anywhere else in this codebase (`JPLSBDBQuery.run_query()` in `views.py:543` calls `requests.get(url)` with no timeout at all — a known anti-pattern, not a convention to follow) — this is the first explicit timeout introduced in `solsys_code/`.
- **D-11:** No raw response body or credential/API-key content may appear in any logged error/exception message for a failed API call (SYNC-09). `tom_observations.facilities.ocs.make_request()` raises exceptions whose messages embed `response.content` directly — any `except`/log path that stringifies and logs that exception verbatim would violate SYNC-09. The catch site must construct its own fixed, generic message rather than logging the caught exception's `str()` directly.

### Claude's Discretion
- Exact label string for SOAR's real aperture class (D-05) — confirm against `tom_observations.facilities.soar` or real data rather than guessing.
- Exact per-site aperture-class inventory for the 5 newly-added sites (`elp`, `lsc`, `cpt`, `tfn`, `tlv`) — confirm during research, not assumed from the bare MPC-code table alone.
- D-09's exact prefix-combination/priority rule when `[UNVERIFIED]` and a terminal-state prefix could both apply to the same record.
- Whether the new SYNC-06 fallback counter is reported per-facility in the run summary (mirroring the existing per-facility breakdown established in Phase 5's D-08) — planner should confirm this is the obvious extension and not a fresh decision point.
- Exact helper/method structure for the new per-record API call (extend `OCSFacility.get_observation_status()` vs. add a new dedicated call) — this is a HOW/implementation question for research, not a user-vision decision. **Resolved below: add a new dedicated method/function; do not extend `get_observation_status()`.**

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope. Retry/backoff and caching/memoizing the per-record API result were already explicitly placed in REQUIREMENTS.md's "Out of Scope"/v2 sections before this discussion and were not re-raised.

Reviewed-not-folded: `2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md` — visual/UI coloring change requiring a `tom_calendar` template override, distinct from this phase's scope. Left pending/deferred unchanged.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TELESCOPE-01 | Verified static site/telescope mapping dict, keyed on fully-qualified LCO code, covers all real LCO-network sites | "Standard Stack" / "Code Examples" sections give confirmed `SITE_TELESCOPE_MAP` structure and per-site aperture-class findings below; `tlv` confirmed absent from installed library — see Open Questions #1 |
| TELESCOPE-02 | For a placed record, sync calls LCO Observation Portal API to resolve actual site/enclosure/telescope, maps via verified dict | "Architecture Patterns" Pattern 1 gives the confirmed `/api/requests/{id}/observations/` response shape (`site`/`enclosure`/`telescope` flat siblings of `state`) and the new dedicated-method recommendation |
| TELESCOPE-03 | Per-record API call failure/timeout/unmapped-code falls back to coarse instrument-class label instead of skipping | "Architecture Patterns" Pattern 2 + "Common Pitfalls" give the fallback decision tree and exception-handling design |
| TELESCOPE-04 | Fallback-labeled event distinguishable from verified-label event (title prefix, description note, no silent churn) | D-06..D-09 in User Constraints + "Architecture Patterns" Pattern 3 (title-prefix integration) |
| SYNC-06 | Per-record telescope-API failures tracked as a distinct counter from `skipped_count`, reported in summary | "Architecture Patterns" Pattern 4 (counters dict extension) |
| SYNC-07 | Per-record API failure does not abort the run; record still gets a CalendarEvent with fallback label, batch continues | "Architecture Patterns" Pattern 2 (try/except wrapping the new API call, never raised past the per-record loop) |
| SYNC-08 | Per-record API call uses explicit timeout, single attempt, no retry/backoff loop | "Code Examples" — `requests`/`make_request()` timeout kwarg passthrough confirmed; "Common Pitfalls" Pitfall 2 |
| SYNC-09 | Error/exception output from failed API call never includes raw response body or credential content | "Common Pitfalls" Pitfall 1 — `ImproperCredentialsException`/`ValidationError` embed `response.content`; catch site must never log `str(exc)` directly |
</phase_requirements>

## Summary

This phase replaces a single bad assumption (`record.parameters['site']` reliably exists) with a live API call that resolves the real, fully-qualified placement (`site`/`enclosure`/`telescope`) for **placed** records only, and a coarse, always-available fallback for everything else (banner-stage records, and placed records whose API call fails). The installed `tom_observations` library (TOM Toolkit's OCS facility backend) already has all the pieces needed — a `make_request()` helper that passes `**kwargs` straight to `requests.request()` (so `timeout=10` works with zero library modification), and an existing `get_observation_status()` method that hits the exact right endpoint (`/api/requests/{id}/observations/`) but only extracts 3 of the fields the response actually contains.

The single most important finding from this research: **the installed library's `get_observation_status()` does NOT surface `site`/`enclosure`/`telescope`, but the underlying API response it parses already contains them as flat sibling keys to `state`/`start`/`end`** — confirmed against the open-source `observatorycontrolsystem/observation-portal` Django backend (the actual upstream of `observe.lco.global`)'s `ObservationSerializer`, whose declared `Meta.fields` are exactly `('site', 'enclosure', 'telescope', 'start', 'end', 'priority', 'configuration_statuses', 'request', 'state', 'modified', 'created')`. This phase should add a **new, dedicated function** (not extend `get_observation_status()`) that re-issues the same two GETs and additionally extracts `site`/`enclosure`/`telescope` from the matched block — extending the existing method risks changing its return shape for Phase 4-6 callers that only destructure `state`/`scheduled_start`/`scheduled_end`.

Also confirmed locally: `tlv` (Wise Observatory) is **NOT** present in installed `LCOSettings.get_sites()` or `SOARSettings.get_sites()` — it exists only in PROJECT.md's external MPC-code reference table, sourced from a website, not from this codebase's actual facility configuration. The verified dict must still include `tlv` per the locked decision (TELESCOPE-01 requires covering "all real LCO-network sites"), but the planner should flag this explicitly: the live API call will simply never need to resolve `tlv` through `get_observing_sites()` (the dict lookup doesn't depend on that method), but if a real placed record ever does return `tlv-...`, no installed-library default exists to corroborate the aperture class — this is an [ASSUMED] entry until a real `tlv` API response or LCO's own site documentation is checked.

**Primary recommendation:** Add a new function `_resolve_telescope_via_api(record, facility)` (returns `tuple[str | None, bool]` — resolved label or `None`, and whether an API attempt was even made) that: (1) returns immediately with no API call if `record.scheduled_start is None` (D-01); (2) otherwise calls the LCO API with `timeout=10`, single attempt, catching `requests.exceptions.RequestException` and the library's `ImproperCredentialsException`/`forms.ValidationError` together as one fallback branch that never logs `str(exc)` verbatim (SYNC-09); (3) on success, maps the returned `f'{site}-{enclosure}-{telescope}'`-style fully-qualified code through the new `SITE_TELESCOPE_MAP` (keyed by bare site code, see D-03 collapse) plus the already-extracted aperture class from `telescope` (e.g. `'0m4a'` → `'0m4'`); (4) on any failure path, falls back to the coarse instrument-class label already derivable from Phase 6's `_extract_instrument()` output.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Per-record LCO API call (site/enclosure/telescope resolution) | API / Backend (management command, server-side) | — | Synchronous, in-process Django management command; no browser/client tier exists for this batch job |
| Static site→label mapping dict | API / Backend | — | Pure in-memory lookup table inside the same command module |
| Fallback coarse-label derivation | API / Backend | — | Reuses Phase 6's already-extracted `instrument_type` string, no new I/O |
| Failure/fallback counters + summary reporting | API / Backend | — | Extends existing `counters` dict in `Command.handle()`; printed to `self.stdout` |
| CalendarEvent title/description/telescope field write | API / Backend | Database / Storage | Backend computes the values; Database/Storage (Django ORM, SQLite) persists them via existing `get_or_create`/`.save()` |
| Calendar display of `[UNVERIFIED]` prefix | Browser / Client (rendering only) | — | No code change in this tier — `tom_calendar`'s existing unmodified templates render whatever title string is stored; this phase's only output is data, not template logic |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `requests` | 2.32.5 (installed, confirmed via `pip`) `[VERIFIED: installed environment]` | HTTP client for the per-record LCO API call | Already the transitive dependency `tom_observations.facilities.ocs.make_request()` uses internally; no new dependency needed |
| `tom_observations.facilities.lco.LCOFacility` / `.soar.SOARFacility` | Installed with `tomtoolkit` (already a project dependency) `[VERIFIED: installed environment]` | Provides `facility_settings.get_setting('portal_url')`/`'api_key'` and `_portal_headers()`-equivalent auth header construction | Already used throughout Phases 4-6 of this command; no alternative client should be introduced |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `requests.exceptions` (stdlib of `requests`) | 2.32.5 | Catch `Timeout`/`ConnectionError`/`HTTPError`/base `RequestException` | Single `except (requests.exceptions.RequestException, ImproperCredentialsException, forms.ValidationError)` clause around the new API call |
| `tom_common.exceptions.ImproperCredentialsException` | Installed with `tomtoolkit` `[VERIFIED: installed environment]` | Library's own exception type raised by `make_request()` on 401-403 | Must be caught alongside `requests.exceptions.RequestException` — it is NOT a subclass of `RequestException` |
| `django.forms.ValidationError` | Bundled with Django 5.2.14 (installed) `[VERIFIED: installed environment]` | Library's own exception type raised by `make_request()` on exactly 400 | Must be caught alongside the above — also NOT a `RequestException` subclass |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| New dedicated function for site/enclosure/telescope resolution | Extending `OCSFacility.get_observation_status()` to also return these 3 fields | Rejected — `get_observation_status()` is called elsewhere in this codebase's Phase 4-6 work only by destructuring `state`/`scheduled_start`/`scheduled_end` (3-key dict); widening its return contract risks silently breaking any future caller that does `result['state']`-style access without expecting new keys, and conflates "TOM Toolkit's generic facility status check" with "this command's site-label-specific need." A new, command-local function keeps the blast radius contained to `sync_lco_observation_calendar.py`. |
| Catching `requests.exceptions.RequestException` only | Catching `Exception` broadly | Rejected — too broad; would also catch programming errors (e.g. `KeyError` on a malformed response) and silently misreport them as "API failure," obscuring real bugs. The library's two extra exception types (`ImproperCredentialsException`, `forms.ValidationError`) must be added explicitly since they are not `RequestException` subclasses. |

**Installation:** No new packages required — `requests` and `tomtoolkit` are already installed project dependencies.

**Version verification:**
```bash
python3 -c "import requests; print(requests.__version__)"   # -> 2.32.5
python3 -c "import django; print(django.VERSION)"            # -> (5, 2, 14, 'final', 0)
```
Both confirmed installed in the project's venv (`/home/tlister/venvs/fomo312_venv`) during this research session.

## Package Legitimacy Audit

This phase installs **no new external packages**. All libraries used (`requests`, `tomtoolkit`/`tom_observations`, Django's `forms.ValidationError`) are already-installed project dependencies used by the existing `sync_lco_observation_calendar.py` and its Phase 4-6 predecessors. The Package Legitimacy Gate is not applicable — no `gsd-tools query package-legitimacy check` run was performed because there is nothing new to audit.

**Packages removed due to [SLOP] verdict:** none (no new packages considered)
**Packages flagged as suspicious [SUS]:** none (no new packages considered)

## Architecture Patterns

### System Architecture Diagram

```
Command.handle()  [existing loop, per ObservationRecord]
        |
        v
_build_event_fields(record, facility)  [existing function, modified]
        |
        +-- record.scheduled_start is None? ---------------------> YES --> coarse fallback label
        |                                                                  (from Phase 6 instrument_type)
        |   NO (placed record)
        v
_resolve_telescope_via_api(record, facility)  [NEW function]
        |
        +-- GET /api/requests/{id}/observations/  (timeout=10, single attempt)
        |        |
        |        +-- requests.exceptions.RequestException
        |        |   or ImproperCredentialsException
        |        |   or forms.ValidationError  --------------> caught, generic log message only
        |        |                                              (no response body / API key in log)
        |        |                                                  |
        |        |                                                  v
        |        |                                          coarse fallback label
        |        |                                          + fallback/API-failure counter += 1
        |        |                                          + '[UNVERIFIED]' title prefix
        |        |
        |        v  (success)
        |   response.json() -> matched block's site/enclosure/telescope
        |        |
        |        v
        +-- code in SITE_TELESCOPE_MAP[site]? ---------------> NO (unmapped) --> same fallback path as above
        |
        |   YES
        v
   verified label (e.g. 'LSC-1m0')  --> clean title, no prefix, telescope field = verified label
        |
        v
   CalendarEvent.objects.get_or_create(...) / .save()  [existing, unmodified upsert logic]
        |
        v
   counters[facility][...] += 1  [existing dict, new 'telescope_api_failed' key added]
        |
        v
   Command summary line (self.stdout.write)  [existing, extended with new counter]
```

### Recommended Project Structure
No new files. All changes are within the existing single-file management command:
```
solsys_code/management/commands/
└── sync_lco_observation_calendar.py    # SITE_TELESCOPE_MAP extended/migrated, _derive_telescope()
                                          # replaced/extended, new _resolve_telescope_via_api()
                                          # and _coarse_telescope_label() helpers, counters dict
                                          # extended, _title_for() extended for D-08/D-09
```
Paired demo notebook that must be updated alongside (per CLAUDE.md convention):
```
docs/notebooks/pre_executed/
└── sync_lco_observation_calendar_demo.ipynb   # must gain cells exercising: a placed record with
                                                 # successful API resolution, a placed record with a
                                                 # mocked API failure (fallback path), and the new
                                                 # summary-line counter — regenerated via
                                                 # `jupyter nbconvert --to notebook --execute --inplace`
```

### Pattern 1: Dedicated per-record resolution call (not extending `get_observation_status()`)
**What:** A new function/method that re-issues the same `GET /api/requests/{id}/observations/` call `get_observation_status()` already makes, but extracts `site`/`enclosure`/`telescope` from the matched block instead of (or in addition to, if reuse is preferred) `state`/`start`/`end`.
**When to use:** Always — TELESCOPE-02's resolution logic.
**Confirmed response shape** (via the open-source `observatorycontrolsystem/observation-portal` backend's `ObservationSerializer`, which is the actual Django backend `observe.lco.global` runs — `[VERIFIED: github.com/observatorycontrolsystem/observation-portal observation_portal/observations/serializers.py + models.py]`):
```python
# Each item in the GET /api/requests/{id}/observations/ array (this is what
# response.json() returns — a list of "block" dicts):
{
    "id": 73533469,
    "request": 2089633,
    "site": "coj",            # 3-char site code (CharField max_length=10), e.g. 'coj', 'lsc'
    "enclosure": "clma",      # 4-char enclosure code (CharField max_length=10), e.g. 'clma', 'doma'
    "telescope": "0m4b",      # 4-char telescope code (CharField max_length=10), e.g. '0m4b', '1m0a', '2m0a'
    "start": "2020-03-18T17:25:27Z",
    "end": "2020-03-18T17:30:33Z",
    "priority": 10,
    "state": "COMPLETED",     # matches the existing get_observation_status() 'state'/'start'/'end' extraction
    "configuration_statuses": [ ... ]   # not needed by this phase
}
```
**Existing extraction (`ocs.py:1548-1575`) for comparison — confirms `site`/`enclosure`/`telescope` are untouched siblings:**
```python
# Source: installed tom_observations/facilities/ocs.py:1548-1575 (OCSFacility.get_observation_status)
def get_observation_status(self, observation_id):
    response = make_request(
        'GET',
        urljoin(self.facility_settings.get_setting('portal_url'), f'/api/requests/{observation_id}'),
        headers=self._portal_headers()
    )
    state = response.json()['state']

    response = make_request(
        'GET',
        urljoin(self.facility_settings.get_setting('portal_url'), f'/api/requests/{observation_id}/observations/'),
        headers=self._portal_headers()
    )
    blocks = response.json()
    current_block = None
    for block in blocks:
        if block['state'] == 'COMPLETED':
            current_block = block
            break
        elif block['state'] == 'PENDING':
            current_block = block
    if current_block:
        scheduled_start = current_block['start']
        scheduled_end = current_block['end']
    else:
        scheduled_start, scheduled_end = None, None

    return {'state': state, 'scheduled_start': scheduled_start, 'scheduled_end': scheduled_end}
```
The new function should reuse the **exact same** `current_block`-selection logic (COMPLETED-first, else PENDING) so site/enclosure/telescope come from the same block `scheduled_start`/`scheduled_end` are sourced from — picking a different block for telescope vs. timing would be internally inconsistent. Recommend calling only the second endpoint (`/observations/`) since this command does not need the separate `state` value from `/api/requests/{id}` — the record's own `record.status` field (already populated by TOM Toolkit's existing observation-status-sync background job) serves that purpose.

### Pattern 2: Single-attempt, explicit-timeout call with library-aware exception handling
**What:** The per-record API call must time out at 10s and never retry (D-10), and must never leak `response.content` or the API key into logs (D-11/SYNC-09).
**When to use:** Always — this is the core SYNC-08/SYNC-09 implementation.
**Example:**
```python
# New function in sync_lco_observation_calendar.py
import requests
from django import forms
from django.urls import NoReverseMatch  # not used; example only
from tom_common.exceptions import ImproperCredentialsException
from tom_observations.facilities.ocs import make_request
from urllib.parse import urljoin

_API_TIMEOUT_SECONDS = 10  # D-10: explicit timeout, single attempt, no retry loop.

def _resolve_placement_block(observation_id: str, facility) -> dict[str, Any] | None:
    """Call the LCO Observation Portal API once to resolve a placed record's site/enclosure/telescope.

    Args:
        observation_id: the record's LCO observation_id.
        facility: a shared LCOFacility/SOARFacility instance (for portal_url/api_key/headers).

    Returns:
        dict | None: the matched block dict (with 'site'/'enclosure'/'telescope' keys) on
            success, or None if the API call failed, timed out, or returned no usable block.
            Never raises -- all failure modes are caught and converted to None so the caller
            always falls through to the coarse fallback (SYNC-07: never abort the run).
    """
    try:
        response = make_request(
            'GET',
            urljoin(facility.facility_settings.get_setting('portal_url'), f'/api/requests/{observation_id}/observations/'),
            headers=facility._portal_headers(),
            timeout=_API_TIMEOUT_SECONDS,  # SYNC-08: explicit timeout, single attempt (no retry).
        )
    except (requests.exceptions.RequestException, ImproperCredentialsException, forms.ValidationError):
        # SYNC-09: never log str(exc) -- ImproperCredentialsException/ValidationError embed
        # response.content directly (see ocs.py make_request()), and could contain the API key
        # echoed back in an error body. Caller logs ONE fixed, generic message instead.
        return None

    blocks = response.json()
    current_block = None
    for block in blocks:
        if block['state'] == 'COMPLETED':
            current_block = block
            break
        elif block['state'] == 'PENDING':
            current_block = block
    return current_block  # None if no COMPLETED/PENDING block found -- also routes to fallback.
```
**Caller's logging discipline (SYNC-09 compliance):**
```python
# In the per-record loop / _build_event_fields, NEVER do this:
#     except Exception as exc:
#         self.stderr.write(f'API call failed: {exc}')          # VIOLATES SYNC-09 -- exc may embed
#                                                                  # response.content / API key
# Instead:
block = _resolve_placement_block(record.observation_id, facility)
if block is None:
    self.stderr.write(
        f'Telescope API lookup failed or timed out for observation_id={record.observation_id!r}; '
        'using fallback label.'
    )  # SYNC-09: fixed, generic message -- never derived from the caught exception's str().
```

### Pattern 3: Title-prefix integration with existing terminal-state priority (D-09)
**What:** `[UNVERIFIED]` must combine correctly with the existing `_FAILURE_PREFIX_BY_STATUS`/`_failure_prefix()` convention.
**When to use:** Inside `_title_for()`.
**Recommendation (resolving D-09):** Follow the same precedent Phase 4's D-04 already established — terminal-state prefixes win. A record can only be in ONE terminal state at sync time (`record.status` is a single field), and a terminal-state record by definition already has a final outcome that supersedes "was this label resolved or guessed" as the more important piece of at-a-glance information. Recommended priority order, highest first:
1. Terminal-failure prefix (`[EXPIRED]`/`[CANCELLED]`/`[FAILED]`) — existing, unchanged.
2. `[UNVERIFIED]` — new, only reached if not terminal.
3. `[QUEUED]` — existing, only reached if not terminal and not placed (mutually exclusive with `[UNVERIFIED]` per D-07, since `[UNVERIFIED]` only applies to placed records).
4. Clean (no prefix) — placed, not terminal, label resolved successfully via the API.
```python
# Source: extends existing sync_lco_observation_calendar.py:_title_for() (lines 166-184)
def _title_for(record, telescope, instrument, facility, label_was_fallback: bool) -> str:
    prefix = _failure_prefix(record.status, facility)
    if prefix is not None:
        return f'{prefix} {telescope} {instrument}'          # terminal wins, even over [UNVERIFIED]
    if record.scheduled_start is None:
        return f'[QUEUED] {telescope} {instrument}'            # banner stage, D-07: never [UNVERIFIED]
    if label_was_fallback:
        return f'[UNVERIFIED] {telescope} {instrument}'        # D-08, placed + API failure/unmapped
    return f'{telescope} {instrument}'                          # placed + verified label
```

### Pattern 4: Counter dict extension (mirrors Phase 6's `extraction_failed` precedent)
**What:** A new counter key tracking telescope-API fallbacks, kept distinct from `skipped`.
**When to use:** `Command.handle()`'s per-facility `counters` dict.
**Example:**
```python
# Source: extends existing sync_lco_observation_calendar.py:Command.handle() (lines 339-342)
counters = {
    'LCO': {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0,
             'extraction_failed': 0, 'telescope_api_failed': 0},   # NEW key, D-02 scope
    'SOAR': {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0,
              'extraction_failed': 0, 'telescope_api_failed': 0},
}
# ... increment only when D-02's exact condition is met: a PLACED record whose API call
# failed/timed out/returned an unmapped code. NOT incremented for banner-stage records.
```
The summary line f-string (line 396-401) extends with `telescope_api_failed: {counts["telescope_api_failed"]}` following the exact same `key: N` phrasing already used for `extraction_failed`.

### Anti-Patterns to Avoid
- **Logging the caught exception's `str()` directly:** `ImproperCredentialsException('OCS: ' + str(response.content))` and `forms.ValidationError(f'OCS: {str(response.content)}')` (both in installed `ocs.py:182-185`) embed the raw response body. A `except Exception as exc: log(str(exc))` pattern anywhere in the new code path is an automatic SYNC-09 violation — always construct a fixed, generic message instead.
- **Adding a retry loop "just in case":** D-10/SYNC-08 explicitly forbid this. A single `requests`/`make_request()` call with `timeout=10` and one `except` clause is the complete implementation — no `for attempt in range(...)`, no `tenacity`/`backoff` library, no manual `time.sleep()` retry.
- **Reusing/widening `get_observation_status()`'s return contract:** risks silently changing behavior for any future caller (inside or outside this command) expecting exactly the 3-key `{'state', 'scheduled_start', 'scheduled_end'}` shape.
- **Treating `tlv` as confirmed:** the bare presence of `tlv` in PROJECT.md's MPC-code table (sourced from a website, not this codebase) does NOT mean the installed `tom_observations` library or this project's real `ObservationRecord` data has ever seen a `tlv`-prefixed fully-qualified code. Tag the `tlv` dict entry `[ASSUMED]` and flag for a `checkpoint:human-verify` before relying on it operationally.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP timeout/retry semantics | A custom retry-with-backoff wrapper | `requests`'s native `timeout=` kwarg, single call | `requests` already implements correct timeout (connect + read) semantics; D-10 explicitly forbids retry logic anyway |
| Auth header construction | A hand-rolled `Authorization: Token ...` string builder | `facility._portal_headers()` (existing private method on `OCSFacility`, already used by `get_observation_status()`) | Already handles the empty-api-key case (`{}` vs `{'Authorization': ...}`) correctly; duplicating this logic risks a subtly different (and untested) auth-header bug |
| URL construction for the API endpoint | String concatenation (`portal_url + '/api/requests/' + id + '/observations/'`) | `urljoin(facility.facility_settings.get_setting('portal_url'), f'/api/requests/{id}/observations/')` | Matches the exact pattern every other call site in `ocs.py` uses; `urljoin` correctly handles trailing-slash edge cases that string concatenation gets wrong |

**Key insight:** Every piece of infrastructure this phase needs (auth headers, URL joining, the underlying HTTP call, the response-block-selection logic) already exists verbatim in the installed `tom_observations` library or this command's own Phase 4-6 code. The only genuinely new logic is: (1) extracting 3 additional keys from a response shape the library already fetches, (2) the timeout kwarg, (3) the fallback decision tree, and (4) the new counter/prefix. Resist the temptation to build a more "robust" HTTP client wrapper — it would duplicate untested logic the library already provides correctly.

## Common Pitfalls

### Pitfall 1: Logging an exception whose `str()` embeds the response body or API key
**What goes wrong:** `except Exception as exc: self.stderr.write(f'Lookup failed: {exc}')` looks like reasonable error logging, but if `exc` is `ImproperCredentialsException` or `forms.ValidationError` raised by `make_request()`, `str(exc)` is literally `'OCS: ' + str(response.content)` — which on a 401/403 could echo back diagnostic text that includes the submitted `Authorization` header context in some OCS deployments, or simply leak unrelated user data in the response body.
**Why it happens:** Python's natural `except ... as exc: log(exc)` idiom is exactly the failure mode here — the exception message itself is the leak vector, not the catching code.
**How to avoid:** The catch clause must construct its own fixed string (e.g. `'Telescope API lookup failed or timed out for observation_id={...}'`) and must never interpolate `exc`, `str(exc)`, or `repr(exc)` into any logged/written output.
**Warning signs:** Any `f'{exc}'`, `f'{e}'`, `str(exception_variable)`, or bare `except Exception as exc: ...write(exc)` pattern in the diff touching the new API call path.

### Pitfall 2: Confusing `requests.exceptions.RequestException` coverage with the library's own exception types
**What goes wrong:** `except requests.exceptions.RequestException:` alone does NOT catch `ImproperCredentialsException` or `forms.ValidationError` — both are raised by `make_request()` itself (in `ocs.py:182-185`) and are plain `Exception` subclasses, not `requests.exceptions.RequestException` subclasses. A handler that only catches `RequestException` will let a 400/401/403 response from a malformed/expired API key propagate as an unhandled exception, aborting the whole sync run — directly violating SYNC-07 ("a per-record API failure does not abort the run").
**Why it happens:** It's easy to assume "the requests library raised it" implies "it's a requests exception" — but `make_request()` is the library's own wrapper that translates certain HTTP status codes into its own exception types before `requests` itself would raise anything (`raise_for_status()` only fires for status codes `make_request()` hasn't already intercepted).
**How to avoid:** Catch all three explicitly: `except (requests.exceptions.RequestException, ImproperCredentialsException, forms.ValidationError):`.
**Warning signs:** A test that mocks a 401/403/400 response and asserts the run completes without raising — if this test fails with an uncaught exception, the catch clause is incomplete.

### Pitfall 3: Picking a different "current block" for telescope resolution than for timing
**What goes wrong:** `get_observation_status()`'s existing block-selection logic picks the first `COMPLETED` block, else the first `PENDING` block. If the new telescope-resolution code re-implements this selection independently (e.g. "just take `blocks[0]`"), it could select a different block than the one whose `start`/`end` populate `scheduled_start`/`scheduled_end` elsewhere — producing a telescope label that doesn't match the time window actually reported for the event.
**Why it happens:** Copy-paste drift when extracting similar-but-not-identical logic into a new function instead of reusing the existing selection helper.
**How to avoid:** Either call (or duplicate verbatim) the exact `COMPLETED`-first-else-`PENDING` selection logic from `get_observation_status()`, and extract `site`/`enclosure`/`telescope` from that *same* matched block object — not a separately-selected one.
**Warning signs:** A multi-block test fixture (one `PENDING` block, one `WINDOW_EXPIRED`/`CANCELED` block) where the telescope label comes from the wrong block.

### Pitfall 4: Treating an "unmapped code" the same as a parse/network failure for the title prefix, but differently for clarity in the description
**What goes wrong:** TELESCOPE-03 explicitly groups "fails, times out, or returns a code not in the verified dict" into one fallback path — but a developer might be tempted to give "API succeeded but the code just isn't in our dict yet" a different, less alarming prefix than "the API call itself failed." The locked decisions (D-02, D-07) deliberately treat both as the same fallback/failure bucket for counting and prefixing purposes.
**Why it happens:** Intuitively these feel like different failure classes (one is "our dict is incomplete," the other is "the network/API is unreliable").
**How to avoid:** Follow the locked scope exactly — both increment the same `telescope_api_failed` counter and get the same `[UNVERIFIED]` prefix. The `description` field is the place to surface the distinction in detail (e.g. "API lookup failed: timeout" vs. "API returned unmapped code 'tlv-meckering-1m0a'") since TELESCOPE-04 requires the description to state that the lookup failed/was unverified, but the *unmapped-code* case is the one place a non-leaking, fully-detailed description is safe to write (the unmapped site code itself is not a credential or response body).
**Warning signs:** Two different title prefixes or two different counters for "API failed" vs. "API succeeded but code unmapped."

### Pitfall 5: Assuming `tlv`'s aperture class without a real data point
**What goes wrong:** Defaulting `tlv` to `1m0` (the most common LCO aperture class) "to be safe" silently introduces an unverified guess into a dict the phase's own locked decisions (TELESCOPE-01) demand be "verified."
**Why it happens:** All 8 sites need *some* entry to satisfy the "covers all real LCO-network sites" requirement, and `1m0` is statistically the most common class across the network, making it a tempting default.
**How to avoid:** Tag the `tlv` entry `[ASSUMED]` in code comments (mirroring the existing `[ASSUMED]` comment style already in `SITE_TELESCOPE_MAP`), and route it through a `checkpoint:human-verify` task before the planner considers the dict "verified" per TELESCOPE-01's own wording. See Open Questions #1 below for the concrete confirmation path.
**Warning signs:** A `tlv` dict entry with no comment explaining its provenance, or a test asserting `tlv` resolves correctly without a citation for where that aperture class came from.

## Code Examples

### Confirmed `/api/requests/{id}/observations/` response shape (resolves Research Gap #1)
```json
// Source: observatorycontrolsystem/observation-portal (open-source upstream of observe.lco.global)
// observation_portal/observations/serializers.py ObservationSerializer.Meta.fields +
// observation_portal/observations/models.py Observation field definitions.
// [VERIFIED: github.com/observatorycontrolsystem/observation-portal]
[
  {
    "id": 73533469,
    "request": 2089633,
    "site": "coj",
    "enclosure": "clma",
    "telescope": "0m4b",
    "start": "2020-03-18T17:25:27Z",
    "end": "2020-03-18T17:30:33Z",
    "priority": 10,
    "state": "COMPLETED",
    "configuration_statuses": [ /* not needed by this phase */ ]
  }
]
```
`site`, `enclosure`, `telescope` are `CharField(max_length=10)` on the `Observation` model with help text "3 character site code" / "4 character enclosure code" / "4 character telescope code" respectively — confirming `site` values like `'coj'`/`'lsc'` and `telescope` values like `'0m4b'`/`'1m0a'`/`'2m0a'` (aperture class + a/b dome-instance letter suffix).

### Mapping a resolved block to the verified dict
```python
# New helper in sync_lco_observation_calendar.py
def _aperture_class_from_telescope_code(telescope_code: str) -> str | None:
    """Extract the aperture-class token (D-04 vocabulary) from a 4-char telescope code.

    Args:
        telescope_code: e.g. '0m4b', '1m0a', '2m0a' (from the API response's 'telescope' key).

    Returns:
        str | None: '0m4'/'1m0'/'2m0' (strips the trailing dome-instance letter), or None if
            the code doesn't match the expected 3-char-class + 1-char-suffix shape.
    """
    if len(telescope_code) >= 4 and telescope_code[:3] in {'0m4', '1m0', '2m0'}:
        return telescope_code[:3]
    return None  # unrecognized shape -- routes to fallback per TELESCOPE-03.


def _derive_telescope(site: str, telescope_code: str) -> str | None:
    """Map a resolved (site, telescope_code) pair to a verified label via SITE_TELESCOPE_MAP.

    Args:
        site: 3-letter site code from the API response (e.g. 'lsc').
        telescope_code: 4-char telescope code from the API response (e.g. '1m0a').

    Returns:
        str | None: the verified label (e.g. 'LSC-1m0'), or None if the (site, class) pair
            isn't in SITE_TELESCOPE_MAP -- caller falls back to the coarse label (TELESCOPE-03).
    """
    aperture_class = _aperture_class_from_telescope_code(telescope_code)
    if aperture_class is None:
        return None
    return SITE_TELESCOPE_MAP.get((site, aperture_class))
```

### Existing fallback-label precedent (D-03/D-04 reuse target for the coarse label)
The fallback label vocabulary (`1m0`/`0m4`/`2m0`) must come from Phase 6's already-extracted instrument type, not be guessed independently. Confirm the instrument-type-to-aperture-class mapping convention against the installed library's own internal pattern at `ocs.py:1317-1319` (`self._get_instruments()[instrument_type]['class']`) — this confirms LCO's own data model already treats "instrument type implies aperture class" as the canonical relationship, validating Stage 1's design-doc approach of deriving the coarse label from `instrument_type` directly (e.g. instrument codes prefixed `2M0-`/`1M0-`/`0M4-` as seen in `valid_instruments = ['1M0-SCICAM-SINISTRO', '0M4-SCICAM-SBIG', '2M0-SPECTRAL-AG']`, `lco.py:792`).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| `_derive_telescope(record.parameters['site'])` — flat key, raises `KeyError` if absent/unmapped | `_resolve_placement_block()` + `_derive_telescope(site, telescope_code)` — live API call for placed records, graceful fallback otherwise | This phase | Real `ObservationRecord.parameters` data doesn't reliably have a flat `site` key (the v1.2 bug this phase exists to fix); the new approach is the only one that works against real data |
| 2-site `[ASSUMED]` `SITE_TELESCOPE_MAP` (`coj`→`FTS`, `ogg`→`FTN`, `sor`→`SOAR`) | 8-site, (site, aperture-class)-keyed verified dict, `SITECODE-CLASS` label format | This phase | Existing 3 entries migrate (D-05); 5 new sites added; format changes uniformly across all entries |

**Deprecated/outdated:**
- The flat `SITE_TELESCOPE_MAP[site_code] -> label` 1-to-1 dict structure: replaced by a `(site, aperture_class) -> label` keyed structure (D-03) since some sites host multiple aperture classes.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `tlv` (Wise Observatory) aperture class is `1m0` (or whichever class the planner chooses as a placeholder) | Standard Stack / Code Examples (the `SITE_TELESCOPE_MAP` dict the planner will design) | If wrong, a real `tlv`-placed record gets an incorrect verified label silently (no error, just a wrong-but-confident-looking label) until a human notices, since the dict claims to be "verified" |
| A2 | SOAR's real aperture class is `4m0` | User Constraints D-05 (carried from CONTEXT.md, not newly introduced by this research) | SOAR is well-known publicly as a 4.1m telescope (general astronomy knowledge, not confirmed against installed `tom_observations.facilities.soar`, which has no aperture-class field at all in its `get_sites()` dict) — low risk since this is widely-documented fact, but technically unverified against this codebase's installed library |
| A3 | All 5 newly-added LCO sites (`elp`, `lsc`, `cpt`, `tfn`, `tlv`) host the aperture classes implied by PROJECT.md's MPC-code table groupings (e.g. `lsc` having both `1m0` and `0m4` based on its 5 listed MPC codes W85-89) | Open Questions #2 below | The MPC-code-count heuristic is suggestive, not authoritative — actual per-site telescope inventory could differ (e.g. a code could be retired/added since the table was compiled); a wrong inventory means the verified dict has either missing classes (records that should resolve cleanly instead fall back) or phantom classes (dead code, no functional risk) |

**If this table is empty:** N/A — see entries above; all three require user/operator confirmation before the dict can be called fully "verified" per TELESCOPE-01's wording.

## Open Questions

1. **Exact aperture-class inventory per site, especially `tlv`**
   - What we know: PROJECT.md's MPC-code table lists `tlv`/Wise Observatory as having a single MPC code (`097`), unlike the other sites' multiple codes — suggesting (not confirming) a single aperture class at that site. The installed `tom_observations` library does not include `tlv` in `LCOSettings.get_sites()` at all (confirmed by reading the installed source), meaning this project's facility configuration has never actually talked to a `tlv` telescope through this library.
   - What's unclear: Whether `tlv` is even currently an active part of the LCO network this FOMO instance's API key has access to, and if so, what aperture class it runs.
   - Recommendation: Add a `checkpoint:human-verify` task before finalizing the `tlv` dict entry. Concrete confirmation paths, in order of preference: (a) if a real LCO API key with network access is available, query `GET /api/instruments/` or `GET /api/telescope_states/` against `https://observe.lco.global` and look for any `tlv.*` key; (b) check LCO's own public telescope status/network page (https://lco.global/observatory/sites/, https://lco.global/observatory/status/) for current `tlv` instrumentation; (c) if neither is feasible during planning, ship the dict with `tlv` mapped to the most defensible single guess (cite the chosen aperture class explicitly as `[ASSUMED]` in the code comment) and accept the fallback path will silently handle it correctly even if the verified entry is wrong (TELESCOPE-03's fallback exists precisely for this kind of gap).

2. **Per-site aperture-class inventory for `elp`, `lsc`, `cpt`, `tfn` (beyond `tlv`)**
   - What we know: PROJECT.md's MPC-code table groups suggest multi-class sites (e.g. `lsc` has 5 codes, `elp` has 5, `cpt` has 4, `tfn` has 4) versus single/near-single-class sites (`coj` has 5 codes but is already confirmed `2m0`-only for the *currently mapped* entry — though this codebase's only confirmed `coj` evidence is `FTS`/`2m0`, raising the question of whether `coj` *also* needs a `1m0`/`0m4` entry that the current 1-entry dict has never needed because no real synced record happened to place there).
   - What's unclear: The MPC-code-count-to-aperture-class-count relationship is not strictly 1-to-1 (a single physical telescope can have multiple historical MPC codes from instrument/optics changes over time) — this table alone cannot be mechanically translated into a verified aperture-class inventory.
   - Recommendation: Cross-reference LCO's public telescope specifications page (https://lco.global/observatory/telescopes/) during planning, which documents each site's actual telescope classes directly (1-meter network sites, 0.4-meter network sites, 2-meter sites are each listed by name). This is a more direct authoritative source than back-deriving from MPC codes. If time-constrained, ship with the most conservative coverage (only add a (site, class) entry the planner can cite a specific source for) and let TELESCOPE-03's fallback absorb any gap.

3. **Whether `record.status` (already populated by TOM Toolkit's own background observation-status sync) is sufficiently fresh/authoritative to use instead of re-fetching `state` from `/api/requests/{id}`**
   - What we know: `get_observation_status()` makes 2 separate GETs (`/api/requests/{id}` for `state`, then `/observations/` for the block). This phase's new function only strictly needs the second GET.
   - What's unclear: Whether `record.status` in the DB could be stale relative to a live API call at sync time (e.g. updated by a different periodic job on a different cadence).
   - Recommendation: Skip re-fetching `state` from `/api/requests/{id}` — use the already-available `record.status` field (as `_failure_prefix()` and `_title_for()` already do) and only call `/observations/` for the new site/enclosure/telescope data. This halves the new API traffic per record (1 GET instead of 2) without changing any existing behavior, since nothing in this phase needs a second, independently-fetched `state` value.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `requests` | Per-record API call (SYNC-08) | ✓ | 2.32.5 | — |
| `tomtoolkit` / `tom_observations` | `LCOFacility`/`SOARFacility`, `make_request()`, `ImproperCredentialsException` | ✓ | installed in project venv | — |
| Django | `forms.ValidationError` exception type, ORM/`CalendarEvent` | ✓ | 5.2.14 | — |
| Live `https://observe.lco.global` API + valid `LCO_APIKEY` | Confirming exact field values against a real placed record (Open Questions #1/#2) | Not verified during this research session (no network call attempted against the live, authenticated portal; all confirmation done via the open-source upstream's public GitHub source instead) | — | Use the open-source `observation-portal` serializer/model definitions as the schema source of truth (done in this research); defer live-data confirmation of `tlv`/site inventory to a `checkpoint:human-verify` task per Open Questions #1/#2 |

**Missing dependencies with no fallback:** none — all code-level dependencies are already installed.

**Missing dependencies with fallback:** Live, authenticated confirmation against `observe.lco.global` for the exact `tlv`/multi-site aperture-class inventory — fallback is the open-source upstream schema (already obtained, HIGH confidence for field *names*) plus `checkpoint:human-verify` for the specific *site inventory facts* that only a live API call or LCO's own public telescope-specs page can fully confirm.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django's `django.test.TestCase` test runner, via `./manage.py test solsys_code` (confirmed — this is a DB-dependent test file, not pytest-collected per `pyproject.toml`'s `testpaths`) |
| Config file | none — Django test discovery via `manage.py test`; no separate pytest config applies to this file |
| Quick run command | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` |
| Full suite command | `./manage.py test solsys_code` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TELESCOPE-01 | Verified dict covers all 8 sites with correct (site, class) → label mapping | unit | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_telescope_01_verified_dict_covers_all_sites` (new) | ❌ Wave 0 |
| TELESCOPE-02 | Placed record + successful mocked API response resolves to verified label | integration (mocked HTTP) | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_telescope_02_placed_record_resolves_via_api` (new) | ❌ Wave 0 |
| TELESCOPE-03 | Placed record + mocked API failure/timeout/unmapped code falls back to coarse label, record still synced | integration (mocked HTTP) | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_telescope_03_api_failure_falls_back_not_skipped` (new) | ❌ Wave 0 |
| TELESCOPE-04 | Fallback event has coarse telescope token + description failure note + `[UNVERIFIED]` prefix; re-run with success flips label visibly (updates, not silently unchanged) | unit + integration | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_telescope_04_fallback_label_visibly_distinguishable` (new) | ❌ Wave 0 |
| SYNC-06 | `telescope_api_failed` counter increments only for placed+failed records, reported in summary, distinct from `skipped` | integration | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_sync_06_fallback_counter_distinct_from_skipped` (new) | ❌ Wave 0 |
| SYNC-07 | A per-record API failure does not abort the run; subsequent records still process | integration | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_sync_07_api_failure_does_not_abort_run` (new) | ❌ Wave 0 |
| SYNC-08 | Mocked slow/failing response — assert no second call attempted (single-attempt, no retry) | unit (mocked HTTP, call-count assertion) | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_sync_08_single_attempt_no_retry` (new) | ❌ Wave 0 |
| SYNC-09 | Mocked failure whose exception embeds fake response content/API key — assert logged output is the fixed generic message, never the raw content/key | unit (mocked HTTP, log-content assertion) | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_sync_09_no_credential_or_body_leak_in_logs` (new) | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar`
- **Per wave merge:** `./manage.py test solsys_code` (full Django suite) + `python -m pytest` (pytest suite, unaffected by this phase but cheap to run for regression safety) + `ruff check .` + `ruff format --check .`
- **Phase gate:** Full Django suite green, plus the paired demo notebook (`docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`) regenerated via `jupyter nbconvert --to notebook --execute --inplace` and committed with output, before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] All 8 new test methods listed in the table above — none exist yet; the existing 19 tests in `test_sync_lco_observation_calendar.py` cover Phases 4-6 only.
- [ ] Mocking strategy for the new API call: `unittest.mock.patch` on `tom_observations.facilities.ocs.make_request` (as imported in `sync_lco_observation_calendar.py`, if imported directly) or on `requests.request`/`requests.get` at the lowest level the new function actually calls — the planner must decide exactly which call site to patch based on the final function signature, and add a shared mock-response-builder helper (mirroring the existing `_parameters()` fixture helper pattern) for "successful block," "failed/timeout," and "unmapped code" response shapes.
- [ ] A timeout-specific test double: `requests.exceptions.Timeout` raised via `side_effect` on the mocked call, to drive the SYNC-08 "single attempt" assertion (`mock.assert_called_once()`).

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | This phase consumes an existing API key (`LCO_APIKEY` env var), doesn't implement authentication itself |
| V3 Session Management | no | Management command, no session/request-cycle concept |
| V4 Access Control | no | No new access-control surface — this is a backend batch job, not a request handler |
| V5 Input Validation | yes | The API response itself is untrusted input — `response.json()` parsing must tolerate missing/malformed `site`/`enclosure`/`telescope` keys (`.get()` not `[]` direct indexing) without raising, since SYNC-07 requires the run to continue regardless |
| V6 Cryptography | no | No cryptographic operations in this phase; credential handling is "don't log it," not "encrypt/hash it" |
| V7 Error Handling and Logging (ASVS 7.4) | yes | SYNC-09's core requirement — error messages must not disclose sensitive data (API keys, raw response bodies) per ASVS 7.4.1 ("verify that a generic message is shown when an unexpected or security sensitive error occurs, potentially with a unique ID") |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Credential leakage via verbose exception logging | Information Disclosure | Never log `str(exc)`/`repr(exc)` for exceptions originating from `make_request()` (`ImproperCredentialsException`, `forms.ValidationError`) since they embed `response.content` directly; construct a fixed, generic message instead (this phase's core SYNC-09 control) |
| Response-body-derived denial via malformed JSON | Denial of Service (minor — single record, not the whole run) | `response.json()` inside a broad enough `try`/`except` that a malformed/non-JSON response (e.g. an HTML error page from a misconfigured reverse proxy) is caught as a generic failure and routed to the fallback path, not allowed to raise `json.JSONDecodeError` uncaught and abort the run (SYNC-07) |
| Slow/hanging upstream API blocking the whole batch job | Denial of Service | `timeout=10` (SYNC-08) bounds the worst-case per-record delay; with no retry, a consistently slow API degrades the run's total duration linearly with record count, but never blocks indefinitely on any single record |

## Sources

### Primary (HIGH confidence)
- `solsys_code/management/commands/sync_lco_observation_calendar.py` (this repo, installed) — existing `SITE_TELESCOPE_MAP`, `_derive_telescope()`, `_build_event_fields()`, `_FAILURE_PREFIX_BY_STATUS`/`_failure_prefix()`, `Command.handle()` counters/summary
- `solsys_code/tests/test_sync_lco_observation_calendar.py` (this repo, installed) — existing test fixtures/conventions
- `/home/tlister/venvs/fomo312_venv/lib/python3.12/site-packages/tom_observations/facilities/ocs.py` (installed `tomtoolkit` dependency) — `OCSFacility.get_observation_status()` (lines 1548-1575), `make_request()` (lines 180-187), `_portal_headers()` (lines 1591-1595), `_build_location()` (lines 1315-1319)
- `/home/tlister/venvs/fomo312_venv/lib/python3.12/site-packages/tom_observations/facilities/lco.py` (installed) — `LCOSettings.get_sites()` (lines 99-137, confirms 6 sites, no `tlv`), `valid_instruments` (line 792), `_build_location()` (lines 1027-1032)
- `/home/tlister/venvs/fomo312_venv/lib/python3.12/site-packages/tom_observations/facilities/soar.py` (installed) — `SOARSettings.get_sites()` (lines 31-39, confirms only `sor`, no aperture-class field)
- `/home/tlister/venvs/fomo312_venv/lib/python3.12/site-packages/tom_common/exceptions.py` (installed) — `ImproperCredentialsException` base class confirmation
- `src/fomo/settings.py` (this repo) — `FACILITIES['LCO']`/`FACILITIES['SOAR']` confirmed, `LCO_APIKEY` env var sourcing already present in the uncommitted diff (benign, matches expected pattern)
- `github.com/observatorycontrolsystem/observation-portal` `observation_portal/observations/serializers.py` + `observation_portal/observations/models.py` (open-source upstream of `observe.lco.global`) `[VERIFIED: github.com/observatorycontrolsystem/observation-portal]` — confirms `ObservationSerializer.Meta.fields` includes `site`/`enclosure`/`telescope` as flat siblings of `state`/`start`/`end`/`request`, and the `Observation` model's field definitions (`CharField(max_length=10)`, "3 character site code" / "4 character enclosure code" / "4 character telescope code")

### Secondary (MEDIUM confidence)
- `https://observe.lco.global/api/requests/2089633/` (WebFetch of a real, publicly-viewable request-detail page) `[CITED: observe.lco.global]` — confirmed the request-level `location` block contains only `telescope_class`, not site/enclosure/telescope (the actual placement only appears at the `/observations/` sub-resource level, matching the open-source serializer)
- `.planning/PROJECT.md` (this repo) — LCO site → MPC code reference table, sourced from `https://lco.global/observatory/sites/mpccodes/` per its own citation; used as a starting point for site enumeration, not as an authoritative aperture-class source

### Tertiary (LOW confidence)
- General astronomical knowledge that SOAR is a 4.1m-class telescope (training data, not verified against any installed source or live API in this session — see Assumptions Log A2)
- The exact per-site aperture-class inventory for `elp`/`lsc`/`cpt`/`tfn`/`tlv` beyond what's directly confirmed in installed `tom_observations` source (only `coj`/`ogg`/`sor` have a directly-observed aperture class in this codebase's existing dict/tests) — see Open Questions #1/#2

## Metadata

**Confidence breakdown:**
- Standard Stack: HIGH — no new packages, all already installed and verified against the running venv
- Architecture (API response shape, Research Gap #1): HIGH — confirmed against the open-source upstream Django backend's actual serializer/model source code, not inference or guesswork
- Architecture (per-site aperture-class inventory, Research Gaps #2-4): LOW-MEDIUM — `coj`/`ogg`/`sor` confirmed via this codebase's existing tests/dict; the 5 new sites' exact class inventories and `tlv`'s presence/class are not confirmable from any source available in this session (live API access would be needed) — flagged explicitly in Open Questions and Assumptions Log for planner/operator follow-up
- Pitfalls: HIGH — directly derived from reading the installed library's actual exception-raising code (`make_request()`), not speculation

**Research date:** 2026-06-21
**Valid until:** 30 days (stable, no fast-moving dependencies — `tom_observations`/`requests`/Django are pinned project dependencies; the LCO Observation Portal API schema itself is the only externally-controlled variable, and the open-source upstream repo is the most stable possible reference for it)
