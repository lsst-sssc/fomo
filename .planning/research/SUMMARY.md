# Project Research Summary

**Project:** FOMO v1.3 — "Full LCO Facility Sync" milestone (generalize `sync_lco_observation_calendar` to multi-facility, multi-proposal sync with live telescope-placement enrichment)
**Domain:** Django management command generalization — astronomical follow-up scheduling, queue-network (LCO/SOAR OCS-family) observation sync into a shared calendar model
**Researched:** 2026-06-18/19
**Confidence:** HIGH

## Executive Summary

FOMO v1.3 takes the v1.2 `sync_lco_observation_calendar` command — which today handles exactly one proposal and one facility (LCO) and has a confirmed correctness bug (it reads `parameters['instrument_type']`/`parameters['site']` keys that don't exist in any real LCO submission) — and generalizes it along three axes at once: multi-proposal/`ALL` selection, multi-facility scope (LCO + SOAR, which share the same `OCSFacility`/`LCOFacility` code path), and a new per-record live API call to resolve the *actual* placed site/enclosure/telescope rather than guessing from submission-time parameters. All four research passes agree this is an additive change requiring **zero new dependencies** — `requests`, `tomtoolkit`'s `LCOFacility`/`SOARFacility`, and `unittest.mock` already cover everything needed, and the right pattern throughout is to reuse the existing `tom_observations.facilities.ocs.make_request`/`_portal_headers` plumbing rather than hand-rolling new HTTP calls.

The recommended approach is a single-file generalization of the existing command (no new modules), built in a specific dependency order: (1) `--proposal` comma-list/`ALL` parsing + `facility__in=['LCO','SOAR']` queryset generalization first (pure, zero I/O, independently testable); (2) the `c_1..c_5_instrument_type` multi-configuration scan second, because the telescope-label fallback needs a *correct* instrument string before it's worth building; (3) the per-record live API call + verified 8-site mapping dict + coarse-instrument-class fallback last, as one cohesive unit, since it is the highest-risk, highest-complexity piece (new network I/O, new failure/fallback semantics) and benefits from already having correct inputs from steps 1-2.

The dominant risk across all four research files is **silent failure modes that only show up against real data**, mirroring exactly the bug that motivated this milestone: (a) comma-separated `--proposal` values silently iterate as characters if not split explicitly; (b) SOAR records will silently go unauthenticated if `FACILITIES['SOAR']` is absent from `settings.py` (confirmed absent today) and a shared `LCOFacility()` instance is reused for SOAR records; (c) a naive "first populated `c_N` config" scan can pick a calibration block (Arc/Lamp Flat) over the science spectrum for SOAR's 3-config submissions, or silently fail for MUSCAT's per-channel exposure keys; (d) the new per-record HTTP call has no timeout in the underlying library and must be wrapped with one, with its own narrow exception handling (not merged into the existing `KeyError`/`ValueError` skip path) so a network blip degrades gracefully to a fallback label instead of skipping or crashing; (e) raw exception/response-body logging from the new auth-aware call risks leaking the LCO API token into stderr/CI logs. Every one of these has a concrete, low-cost prevention (explicit parsing, per-facility-name instance construction, ordered/multi-shape test fixtures, `timeout=` + dedicated exception class + dedicated counter, redacted/generic error messages) and should be a same-phase deliverable, not deferred hardening.

## Key Findings

### Recommended Stack

No new core technologies are needed; v1.3 is purely additive on top of what's already installed and used elsewhere in this codebase (`STACK.md`, confidence HIGH).

**Core technologies:**
- `tomtoolkit` (`3.0.0a9`, already pinned) — supplies `LCOFacility`/`SOARFacility`/`OCSFacility`; `SOARFacility(LCOFacility)` is a direct subclass sharing the same request/observation API shape, so no version bump or new library is needed.
- `requests` (`2.32.3`, transitive via `tomtoolkit`) — already the HTTP client `OCSFacility`'s internal `make_request()` wraps; reuse it via the facility's own `_portal_headers()`/`make_request()` rather than a second hand-rolled call.
- `unittest.mock` (stdlib) — the existing, sufficient mocking convention in this codebase (`@patch('requests.get')`/`@patch('requests.request')` style, as already used in `test_views.py`); no new mocking dependency (`responses`, `requests-mock`) is justified for a single new call site.

### Expected Features

Summary from `FEATURES.md` (confidence MEDIUM — codebase/API facts HIGH, general UX best-practice claims LOW/generic websearch).

**Must have (table stakes):**
- Multi-proposal selection (`--proposal a,b,c` and `--proposal ALL`)
- Facility scope covering LCO + SOAR (verify SOAR's `parameters` shape against a real record before shipping, don't assume)
- Correct `c_1..c_5_instrument_type` extraction (fixes v1.2's confirmed 100%-failure bug against real data)
- Verified static 8-site mapping dict (already supplied by `PROJECT.md`'s MPC-code table)
- Run summary counts (created/updated/unchanged/skipped) extended with new failure-mode counters
- Per-record skip-and-continue on data errors (existing pattern, must extend cleanly to the new API failure axis)
- No-churn idempotent create-or-update preserved even as a new field (telescope label) can vary run-to-run

**Should have (differentiators):**
- Per-record live API enrichment for the *actual* placed site/enclosure/telescope, not the submission-time guess — the single most valuable new capability in v1.3
- Graceful coarse instrument-class (`1m0`/`0m4`/`2m0`) fallback label when the API call fails, handled as a distinct non-fatal degrade path (own counter, own log line, visually distinguishable label/description text) — not a skip
- `ALL` proposal mode as an operator/admin "whole network" view distinct from per-PI proposal runs

**Defer (v2+):**
- Status-aware `CalendarEvent` coloring (already deferred per `PROJECT.md`)
- A dedicated boolean/field distinguishing fallback-vs-verified telescope labels beyond title/description text (only if operators report confusion)
- Retry/backoff for the per-record API call (anti-feature unless evidence emerges; the next scheduled run is the natural retry)
- Gemini facility support (explicitly out of scope; different base class)

### Architecture Approach

Summary from `ARCHITECTURE.md` (confidence HIGH — grounded in direct reads of installed `tom_observations` source, this repo's command/tests, settings, and a live ORM query). This is a generalization of one existing ~225-line management command, not a new module — keep everything in `solsys_code/management/commands/sync_lco_observation_calendar.py` plus its existing test file. Three architectural patterns recur throughout the research: (1) reuse the facility object's existing authenticated-request machinery (`ocs.make_request` + `facility._portal_headers()`) rather than hand-rolling `requests` calls; (2) a two-tier resolution function (`_resolve_telescope_label`) that tries the precise API lookup first and falls back to a coarse label, kept as a thin orchestrator so `_build_event_fields`'s existing call shape is unchanged; (3) per-facility dispatch via a small `{name: instance}` lookup dict built once in `handle()`, rather than branching logic threaded through every helper.

**Major components:**
1. `Command.handle()` — parses `--proposal`/builds the per-facility instance dict/builds the generalized queryset (`facility__in`, `parameters__proposal__in` or unfiltered for `ALL`), loops with per-record try/except
2. `_extract_instrument_type()` (new) — pure, no-I/O scan of `c_1..c_5_instrument_type` for the populated config; must land before the telescope-label work since the fallback depends on a correct instrument string
3. `_resolve_telescope_label()` (new) — orchestrates `_fetch_request_site_codes()` (new HTTP call) → `SITE_TELESCOPE_MAP` lookup → coarse-class fallback; the only function in this change that touches network I/O, isolated specifically so `_build_event_fields`-level tests don't need to mock the full HTTP chain
4. `LCOFacility`/`SOARFacility` instances — instantiated once per facility name (not shared across LCO/SOAR records), supplying `get_observation_url()`, `_portal_headers()`, `facility_settings.get_setting('portal_url'/'api_key')`

### Critical Pitfalls

Top findings from `PITFALLS.md` (confidence HIGH — grounded in this repo's source, installed library source, and this dev environment's actual settings/DB rows, not generic web research).

1. **`--proposal` comma-list/`ALL` collapses to an opaque string under Django's `CommandParser`** — `type=str` plus a raw `,`-separated value passed straight into `__in=` will iterate over *characters*, matching nothing or matching wrong single-character codes, with no error raised. Avoid by explicitly splitting/checking for the `ALL` sentinel in `handle()` before building any filter, and test with 2+ comma-separated codes plus a single-character false-positive check.
2. **SOAR records silently go unauthenticated** — `FACILITIES['SOAR']` is confirmed absent from this repo's `settings.py` today; `SOARSettings.get_setting('api_key')` falls through to a blank default with no warning, and reusing one shared `LCOFacility()` instance for SOAR records calls the wrong portal/key entirely. Avoid by instantiating per-facility-name inside the loop and auditing `SOARSettings('SOAR').get_unconfigured_settings()` as a precondition.
3. **Naively scanning `c_1..c_5_instrument_type` for "the first populated one" picks the wrong config** for SOAR's real 3-config submissions (Spectrum/Arc/Lamp Flat, all populated) or silently finds nothing for MUSCAT's per-channel (`_g/_r/_i/_z`) exposure keys. Avoid with explicit `c_1`-first-wins ordering plus dedicated SOAR-3-config and MUSCAT-per-channel test fixtures (neither shape exists in today's tests).
4. **The new per-record HTTP call has no timeout, no batching, and an error path that cannot be exercised live in this dev environment** (blank `LCO_API_KEY`). Must add `timeout=` explicitly (the installed `make_request` passes none), use a separate exception class/counter from the existing `KeyError`/`ValueError` skip path, and add an HTTP-mocked failure-path test — this is the only way to validate the fallback in CI.
5. **Exception/response-body logging from the new auth-aware call risks leaking the API token** — `ImproperCredentialsException` embeds raw response body text; reusing the existing `f'...: {exc}'` log pattern verbatim for this new exception class could leak credentials into stderr/CI logs. Use a fixed, generic message for this specific failure class instead.

## Implications for Roadmap

Based on combined research, the natural dependency chain and risk profile suggest the following phase structure:

### Phase 1: Multi-proposal & multi-facility query generalization
**Rationale:** Pure `handle()`/queryset change, zero new I/O, lowest risk, and unblocks testing every subsequent phase against both LCO and SOAR fixtures in the same test run. Architecture research explicitly recommends this land first.
**Delivers:** `--proposal` comma-list/`ALL` parsing, `facility__in=['LCO','SOAR']` filtering, per-facility instance dict in `handle()`, `FACILITIES['SOAR']` settings audit.
**Addresses:** Multi-proposal selection, facility scope LCO+SOAR (FEATURES.md table stakes).
**Avoids:** Pitfall 1 (comma-list/`ALL` parsed as opaque string), Pitfall 2 (SOAR silent unauthenticated fallback).

### Phase 2: Correct instrument-type extraction
**Rationale:** Must land before telescope-label work — the fallback label is derived from the instrument string, so building it against still-broken extraction forces rework. Pure function, no I/O, fully unit-testable in isolation.
**Delivers:** `_extract_instrument_type()` scanning `c_1..c_5_instrument_type` in order, handling both SOAR's 3-config shape and MUSCAT's per-channel exposure keys.
**Uses:** No new stack elements — pure Python over existing `ObservationRecord.parameters` JSON.
**Implements:** `_extract_instrument_type()` component from ARCHITECTURE.md; avoids Pitfall 3 (wrong config picked).

### Phase 3: Live telescope-placement enrichment with fallback
**Rationale:** Highest-risk, highest-complexity piece (new network I/O, new error/fallback semantics, a new static reference dict needing verification). Sequenced last so it can be built and tested against already-correct instrument data (Phase 2) and an already-generalized multi-facility queryset (Phase 1).
**Delivers:** `_fetch_request_site_codes()` (new HTTP call via `ocs.make_request`/`_portal_headers`), verified 8-site `SITE_TELESCOPE_MAP` (replacing the 2-entry `[ASSUMED]` dict, with dated provenance comment), `_resolve_telescope_label()` orchestrator, coarse instrument-class fallback, distinct fallback/unmapped-site counters in the summary line, redacted error logging.
**Addresses:** Per-record API enrichment + fallback (FEATURES.md differentiator).
**Avoids:** Pitfall 4 (no timeout / unexercisable error path / partial-failure containment), Pitfall 5 (credential leakage in logs), Pitfall 6 (stale/unverifiable site mapping).

### Phase Ordering Rationale

- Dependency chain discovered in research: fallback label needs correct instrument data (Phase 2 before Phase 3); both depend on the queryset already covering both facilities (Phase 1 before Phase 2/3) so SOAR fixtures are exercised throughout rather than retrofitted.
- This ordering front-loads the cheapest, most deterministic fixes (pure functions, no I/O) before the riskiest new I/O and failure-mode surface (Phase 3), matching ARCHITECTURE.md's explicit "Recommended Build Order" section.
- Grouping avoids the anti-pattern (flagged in ARCHITECTURE.md) of restructuring `_build_event_fields`'s call shape to thread facility/proposal state through every helper — facility/proposal selection stays a `handle()`-level concern in Phase 1.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** Needs research/confirmation — the exact JSON key names in `/api/requests/<id>/observations/` block responses (`site`/`observatory`/`telescope`) are inferred by analogy with sibling library code, not confirmed against a live response; confirm against a real `observation_id` before finalizing parsing. Also needs explicit design discussion on fallback-vs-verified-label churn interaction with no-churn (SYNC-04) semantics.

Phases with standard patterns (skip research-phase):
- **Phase 1:** Well-documented Django ORM/argparse patterns; queryset composition already confirmed live against this project's SQLite/JSON1 backend.
- **Phase 2:** Pure-function scan over a known dict shape; SOAR/MUSCAT shapes are documented directly from installed library source.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Direct package inspection (`pip show`), direct reads of installed library source; no new dependencies, no speculation |
| Features | MEDIUM | Codebase/API facts HIGH (direct inspection); general batch/UX best-practice claims LOW (generic websearch, no domain-specific authoritative source for this niche) |
| Architecture | HIGH | Every claim grounded in direct reads of installed `tom_observations` source, this project's command/tests/settings, and a live ORM query — no speculative library behavior asserted |
| Pitfalls | HIGH | Grounded directly in this repo's source, installed library source, and this dev environment's actual settings/DB rows — not generic web research |

**Overall confidence:** HIGH

### Gaps to Address

- Exact JSON key names in the `/api/requests/<id>/observations/` block response (`site`/`observatory`/`telescope`) — inferred by analogy, not yet confirmed against a live response; confirm during Phase 3 planning/execution against a real `observation_id` (e.g. the COMPLETED record already identified in `PROJECT.md`).
- Exact `c_N_ic_M_exposure_time`-style sub-key indicating "this configuration is populated" — should be confirmed against the two known real records' full parameter dumps during Phase 2, not assumed from naming pattern alone.
- Whether `FACILITIES['SOAR']` should be added explicitly to `src/fomo/settings.py` vs. relying on `SOARSettings` defaults — flagged as a deliberate decision needed during Phase 1, not a code-only change.
- No domain-specific authoritative convention exists for "unknown specific site, known instrument class" fallback-label UX — FEATURES.md recommends extending the existing `[QUEUED]`/`[EXPIRED]` bracket-prefix convention, but this is a 5-minute product decision to make explicitly during Phase 3 planning, not further research.
- `tlv` (Wise Observatory) appears in the webpage-sourced 8-site table but not in either `LCOSettings.get_sites()`/`SOARSettings.get_sites()` in the installed library version — verify against a real `ObservationRecord` before shipping the static mapping dict in Phase 3.

## Sources

### Primary (HIGH confidence)
- `tom_observations/facilities/{ocs,lco,soar}.py` (installed `tomtoolkit` package source, read directly) — `make_request`, `_portal_headers`, `get_observation_status`, `OCSSettings.default_settings`, `SOARFacility(LCOFacility)` subclassing, `LCOMuscatImagingObservationForm`, `SOARSimpleGoodmanSpectroscopyObservationForm`
- `solsys_code/management/commands/sync_lco_observation_calendar.py` and `solsys_code/tests/test_sync_lco_observation_calendar.py` (this repo, read directly) — v1.2 shipped behavior and test blind spots
- `src/fomo/settings.py` (this repo, read directly) — confirmed `FACILITIES['SOAR']` absent, `FACILITIES['LCO']['api_key']` blank
- `.planning/PROJECT.md` (this repo) — v1.3 milestone scope, verified MPC-code/site table, v1.2 real-data bug findings
- Live Django ORM query against this project's dev SQLite DB confirming `facility__in` + `parameters__proposal__in` JSON1 query composition
- Direct empirical test confirming `CommandParser` passes `--proposal A,B,C` through as a raw unsplit string

### Secondary (MEDIUM confidence)
- WebSearch corroborating the `tomtoolkit` alpha-vs-stable release timeline (not used to change any pin)

### Tertiary (LOW confidence)
- General batch/ETL partial-failure-handling guidance (AWS, MuleSoft, generic ETL blog) — corroborates skip-and-continue pattern but not domain-specific
- General UX fallback-state guidance — no astronomy-domain-specific authoritative convention found; recommendation leans on this project's own existing bracket-prefix convention instead
- Web search for LCO `/api/requests/{id}/observations/` response schema — inconclusive, flagged as an open question rather than asserted fact

---
*Research completed: 2026-06-19*
*Ready for roadmap: yes*
