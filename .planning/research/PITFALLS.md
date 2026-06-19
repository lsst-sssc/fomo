# Pitfalls Research

**Domain:** Generalizing a single-facility/single-proposal Django management command into a multi-facility/multi-proposal sync that adds a new outbound per-record API call (LCO/OCS-style queue network, TOM Toolkit)
**Researched:** 2026-06-18
**Confidence:** HIGH (grounded directly in this repo's `sync_lco_observation_calendar.py` source, the installed `tom_observations` library source for `ocs.py`/`lco.py`/`soar.py`, and this dev environment's actual `FACILITIES` setting and real `ObservationRecord` rows — not generic web research)

## Critical Pitfalls

### Pitfall 1: `ALL` and comma-lists collapse into Django's `CommandParser` as a single opaque string

**What goes wrong:**
`parser.add_argument('--proposal', type=str, required=True)` (the current v1.2 shape) hands `handle()` whatever raw string the user typed — `--proposal LTP2025A-004,LTP2025A-005` arrives as the **single string** `'LTP2025A-004,LTP2025A-005'`, not a list. If the generalized code forgets to split on `,` before building the filter, `ObservationRecord.objects.filter(parameters__proposal__in=proposal_str)` will iterate over the *characters* of the string (Django's `__in` accepts any iterable), silently matching nothing or matching on single-character "codes" — no error raised. Verified empirically in this environment: `CommandParser` does no implicit splitting; `Namespace(proposal='A,B,C')` is the raw value.

**Why it happens:**
Argparse/Django's `CommandParser` has no built-in "comma-separated list" type; using `type=str` and then directly passing the result to `__in=` looks like it "just works" if you only test with a single code (as v1.2's existing test suite does — every fixture uses one proposal).

**How to avoid:**
- Parse `--proposal` once in `handle()`: `if proposal_arg.strip().upper() == 'ALL': proposal_filter = None` else `proposal_codes = [p.strip() for p in proposal_arg.split(',') if p.strip()]`.
- Build the queryset conditionally: omit the `parameters__proposal` filter entirely for `ALL` (don't pass `parameters__proposal__in=[]`, which matches nothing — an empty `__in` list is *not* the same as "no filter"); use `parameters__proposal__in=proposal_codes` for the list case.
- Add a dedicated unit test asserting `--proposal A,B` matches records with proposal `A` or `B` but not `AB` or a single-character code, and a separate test asserting `--proposal ALL` matches every facility-scoped record regardless of proposal.

**Warning signs:**
- A test suite that only ever exercises a single proposal code (true of v1.2's `test_sync_lco_observation_calendar.py` today) is a hard blind spot for this bug — it will pass even with the broken character-iteration version.
- `grep -n "proposal__in" solsys_code/management/commands/sync_lco_observation_calendar.py` followed by manually confirming the value being filtered is always a `list`, never a raw string, before the queryset is built.

**Phase to address:**
The phase that introduces the `--proposal` generalization itself — this is not a follow-up concern, it is the central correctness risk of that one change. Verification (the two tests above) belongs in the same phase, not deferred to an "integration test" phase.

---

### Pitfall 2: Treating SOAR identically to LCO hides a silent missing-API-key fallback, not a loud one

**What goes wrong:**
`SOARSettings` extends `LCOSettings`/`OCSSettings`, and `OCSSettings.get_setting(key)` is implemented as `settings.FACILITIES.get(self.facility_name, self.default_settings).get(key, self.default_settings[key])`. In **this dev environment's actual `src/fomo/settings.py`**, `FACILITIES` only has `'LCO'` and `'GEM'` keys — there is no `'SOAR'` entry at all. That means `SOARFacility()._portal_headers()` → `get_setting('api_key')` falls through to `default_settings['api_key']` = `''` (falsy) → `_portal_headers()` returns `{}` (no `Authorization` header) with **no exception, no log line, no warning**. A SOAR-scoped per-record API call in this environment will issue an unauthenticated GET, which OCS's `get_observation_status`/`/api/requests/<id>/observations/` may still answer for public-visibility-scoped or low-privilege fields, masking the misconfiguration — or it returns 401/403 with no indication this is a *configuration* problem versus a normal "API call failed" path.
Separately, **SOAR shares the LCO portal/API key namespace by design** (per SOAR's own docstring: "It also uses the LCO API key... the LCO dictionary in FACILITIES will need to be completed") — so even a *correct* SOAR config most likely needs to literally reuse the LCO `api_key`/`portal_url`, which is non-obvious from the facility name alone and easy to get backwards (e.g. assuming SOAR needs its own separate `FACILITIES['SOAR']` block with a different key, when the real risk is *not adding one at all* and getting silent unauthenticated calls instead).

**Why it happens:**
The two facilities share `LCOFacility`/`OCSFacility` machinery so completely (`SOARFacility(LCOFacility)`) that it's natural to assume `facility__in=['LCO', 'SOAR']` is purely a queryset change with no settings-layer consequences. The actual divergence is in `settings.FACILITIES` configuration, which lives outside the Python code path being touched.

**How to avoid:**
- Before writing the SOAR-scope code, confirm in this repo's `settings.py`/`local_settings.py` whether a `'SOAR'` key exists in `FACILITIES`. As of milestone start it does not — flag this as a required environment-config addition (likely reusing the same `LCO_API_KEY` env var under a `'SOAR'` key, mirroring the SOAR docstring's documented pattern), not a code-only change.
- When the per-record LCO API call (`/api/requests/<id>/observations/`) is added, instantiate the **facility-specific** class per record (`LCOFacility()` for `facility='LCO'` records, `SOARFacility()` for `facility='SOAR'` records) rather than a single shared `LCOFacility()` instance for everything — `_portal_headers()`/`portal_url` are resolved from `self.facility_settings.facility_name`, and reusing one `LCOFacility()` instance for SOAR records would silently call the *LCO* portal URL with the *LCO* key for a SOAR observation_id, which may not even resolve (different ID namespace on the same OCS instance is plausible, but unverified — see Gap below).
- Add an explicit unit test that asserts a SOAR-facility record's API call uses SOAR's `facility_settings.get_setting('portal_url')`/`api_key`, not LCO's, even though today they may be configured identically.
- Treat "is `FACILITIES['SOAR']` actually present and non-empty" as a precondition the command checks and warns about loudly (e.g. via `get_unconfigured_settings()`, which already exists on `OCSSettings` for exactly this) rather than discovering it via a silent no-op API call.

**Warning signs:**
- Any code path that does `facility = LCOFacility()` once outside the per-record loop and reuses it for both LCO and SOAR records.
- `SOARSettings('SOAR').get_unconfigured_settings()` returning `['api_key']` or `['portal_url']` in this dev environment (this is the loud version of the check the command should perform before silently proceeding).
- A SOAR per-record API call that "succeeds" with a 200 in dev despite a blank API key — that is itself a red flag the call is unauthenticated, not a sign it's working.

**Phase to address:**
The phase that generalizes facility scope to `facility__in=['LCO', 'SOAR']` must include the `FACILITIES['SOAR']` settings audit and the per-record facility-class selection as first-class success criteria — do not let this slip into the phase that adds the API call, since by then the wrong facility instance may already be baked into the loop structure.

---

### Pitfall 3: Scanning `c_1..c_5_instrument_type` and picking the wrong one when more than one config is populated

**What goes wrong:**
v1.2's bug (documented in PROJECT.md) was assuming a flat `parameters['instrument_type']` key that doesn't exist on real multi-configuration cadence requests; the v1.3 fix scans `c_1_instrument_type`..`c_5_instrument_type` for the config whose `c_N_ic_1..5_exposure_time` is populated. The two real records checked in this dev DB happened to have only `c_1` populated — but **SOAR's own `SOARSimpleGoodmanSpectroscopyObservationForm`** (in the actual installed `tom_observations/facilities/soar.py`) deliberately builds *three* populated configs per submission: `c_1` = Spectrum, `c_2` = Arc (hardcoded `exposure_time = 0.5`), `c_3` = Lamp Flat (hardcoded `exposure_time = 0.5`). A naive "first config with any populated `exposure_time`" scan over a SOAR-submitted record could pick `c_1` (correct, the science spectrum) on a lucky ordering, or could pick `c_2`/`c_3` (the calibration Arc/Flat) if the scan logic doesn't preserve `c_1`-before-`c_2`-before-`c_3` ordering, or — worse — could pick *whichever* config is non-zero/non-falsy and treat a `0.5`s calibration exposure as "the meaningful one" for display purposes, mislabeling a Goodman science block as an Arc/Flat run on the calendar.
Additionally, MUSCAT submissions use `c_N_ic_M_exposure_time_g/_r/_i/_z` (per-channel fields), **not** a flat `c_N_ic_M_exposure_time` key at all (confirmed in `LCOMuscatImagingObservationForm._build_instrument_config`) — a scan that only checks `exposure_time` (no per-channel suffix) will find zero populated configs for a MUSCAT record and either `KeyError`/skip it, or (if the code falls back to "first config regardless") silently report stale/wrong instrument data for every MUSCAT row.

**Why it happens:**
The "scan for the populated one" heuristic was designed against the two real rows actually present in this dev DB (both single-config, both `c_1`-only) — it generalizes from a sample of two, neither SOAR nor MUSCAT, which together cover a meaningful fraction of the in-scope facility set (SOAR is explicitly in scope for this milestone; MUSCAT4 syncing is already shipped per PROJECT.md "Shipped"/SYNC-05 history).

**How to avoid:**
- Scan configs **in `c_1`→`c_5` order** and take the *first* one with a populated exposure-time-equivalent field, explicitly documenting "first populated wins, calibration configs (Arc/Lamp Flat) are expected to be `c_2`+/secondary and are not picked over `c_1`" as a code comment and a test assertion (mirrors v1.2's own commenting style for `_FAILURE_PREFIX_BY_STATUS`).
- Treat "populated" as facility/instrument-aware: check `c_N_ic_1_exposure_time` for non-MUSCAT instrument types, but `c_N_ic_1_exposure_time_g/_r/_i/_z` (any one of the four) for MUSCAT — or, more robustly, derive "populated" from whichever of the known exposure-time key variants exists and is truthy, rather than hardcoding one key name.
- Write a dedicated regression test using a 3-config SOAR-shaped `parameters` dict (Spectrum/Arc/Lamp Flat, matching `SOARSimpleGoodmanSpectroscopyObservationForm`'s real shape) and a MUSCAT-shaped `parameters` dict (per-channel exposure keys), asserting the scan picks `c_1`'s instrument type in both cases — these are exactly the two real-library config shapes most likely to break a naive scan, and neither is currently represented in `solsys_code/tests/test_sync_lco_observation_calendar.py`'s fixtures (which all use a single flat `instrument_type` — itself the bug being fixed).

**Warning signs:**
- The scan logic only checks one exposure-time key name (`exposure_time`) without considering per-channel (`_g/_r/_i/_z`) or per-instrument variants.
- No test fixture in the updated test file has more than one populated `c_N` config.
- `_build_event_fields` reporting an "Arc" or "Lamp Flat" instrument label for what was submitted as a science spectroscopy block.

**Phase to address:**
The phase that replaces the flat `instrument_type` lookup with the `c_1..c_5` scan. Both the SOAR-3-config and MUSCAT-per-channel shapes must be in that phase's test fixtures as explicit success criteria, not added later as a bug-fix follow-up.

---

### Pitfall 4: New per-record outbound API call has no batching, no timeout, no partial-failure containment, and an unexercisable error path in this environment

**What goes wrong:**
Adding a per-record `GET /api/requests/<id>/observations/` call (the same endpoint `OCSFacility.get_observation_status()` already uses internally) inside the existing `for record in records:` loop turns what was a pure-DB-read command into one outbound HTTP request per `ObservationRecord` row. For a large proposal (or `--proposal ALL`, per Pitfall 1) this is N sequential synchronous `requests` calls with **no timeout argument** — `make_request`/`requests.request` in the installed library is called with no `timeout=` kwarg anywhere in `ocs.py`, so a single hung connection blocks the entire command indefinitely. There is also no batching: OCS exposes `/api/requestgroups?request_id=<id>` style list-with-filter endpoints (used internally by `_get_requestgroup_id`), suggesting a single list-fetch-and-filter approach could replace N round trips, but a naive per-record implementation will not discover this.
Separately, this dev environment's `FACILITIES['LCO']['api_key']` is the **empty string** (confirmed directly in `settings.py`) — meaning `_portal_headers()` returns `{}` and any real call to `observe.lco.global` will get a 401/403 (or possibly succeed read-only against public data, masking the missing-auth issue, similar to Pitfall 2). **The failure path literally cannot be exercised against the live LCO API in this dev environment** — testing it requires `responses`/`requests-mock`-style HTTP mocking, not a live call, and that mocking must be added as test infrastructure in the same phase, not assumed to be "tested via manual UAT" (manual UAT here cannot reach the failure branch at all).
Finally, partial-failure semantics: the existing loop already does per-record try/except around `_build_event_fields` (catching `KeyError`/`ValueError`) and continues to the next record on failure (skip + count), which is the right pattern — but a new per-record API call introduces a *third* failure class (`requests.exceptions.RequestException`/`ImproperCredentialsException` from `make_request`) that must be caught with the same "skip this record, count it, continue" discipline. If the API call's exception type isn't added to the existing `except (KeyError, ValueError)` clause (or a parallel one), an API failure on record #3 of 500 will crash the whole command, leaving records #1-2 already `.save()`d/created and #4-500 never attempted — an inconsistent partial sync with no transaction boundary, and worse, no record of *which* records were left unsynced beyond reading stdout/stderr scroll-back.

**Why it happens:**
The existing command's error handling pattern (try/except around the pure-computation `_build_event_fields`, masking out `KeyError`/`ValueError`) is easy to extend syntactically (just add the new exception types to the tuple) but easy to get *wrong* semantically if the API call is added inside `_build_event_fields` itself versus as a separate step — mixing "data shape is wrong" errors with "network call failed" errors in one except clause loses the ability to report them differently (a `KeyError` on `parameters['site']` is a data problem worth flagging to a human immediately; a transient API 503 might be worth retrying, or at least worth a different log message/count than "skipped: bad data").

**How to avoid:**
- Add `timeout=` to any new outbound call this command makes directly (the per-record telescope-lookup call), even though the installed `tom_observations.facilities.ocs.make_request` itself has no timeout — wrap the call site, don't rely on the library default (there is no library default; `requests` with no `timeout=` blocks forever on a hung TCP connection).
- Keep the per-record API call's exception handling **separate** from the existing `_build_event_fields` data-shape exceptions — use a distinct except clause (or a distinct counter: `api_skipped_count` vs `data_skipped_count`) so the final summary line (`created: N, updated: N, unchanged: N, skipped: N`) can tell a human *why* records were skipped, matching the existing summary-line convention in the v1.2 code.
- Catch `requests.exceptions.RequestException` (covers `ConnectionError`, `Timeout`, `HTTPError` — `make_request` already raises `HTTPError`/`ImproperCredentialsException` for 4xx) around the per-record API call specifically, fall back to the coarse instrument-class label (`1m0`/`0m4`/`2m0`) per the PROJECT.md spec, and continue the loop — never let one record's API failure abort the whole sync run.
- Since this dev environment cannot exercise a real API failure (blank key), add an HTTP-mocking-based unit test (e.g. `responses` or `unittest.mock.patch('requests.request', ...)`) that simulates a non-200/timeout response and asserts: (a) the record still gets a `CalendarEvent` with the coarse fallback label, (b) the command does not crash, (c) the skip/fallback is counted and reported in the summary line. This test is the only way to validate the fallback path in CI given the blank `LCO_API_KEY`.
- Consider (not necessarily implement, but flag for the roadmap) whether the per-record telescope-lookup call can be cached per `(site, enclosure, telescope)` triple across the loop rather than per record — many records in the same proposal sync will share the same site/telescope, so caching avoids true N+1 blowup once N is large. This is a performance phase decision, not a correctness one, but should be raised in the same phase that adds the call so it isn't "discovered" only after a slow production run.

**Warning signs:**
- No `timeout=` on the new request call.
- A single `except (KeyError, ValueError)` clause that also silently swallows `requests.exceptions.RequestException` without distinguishing it in the summary counts.
- Test suite has zero tests that simulate an API failure for the new call (only the happy path is tested) — given the blank dev API key, this gap is easy to miss because "I ran it locally and it didn't crash" is not evidence the fallback path works, only that the call probably never got a real response to fall back from.
- Running the command against a large proposal noticeably slower than v1.2's pure-DB version with no progress indication.

**Phase to address:**
The phase that adds the per-record LCO API call must include: the timeout, the separate exception handling/counting, and the HTTP-mocked failure-path test, as in-phase deliverables — not deferred to a "hardening" or "productionization" phase. The caching/N+1 mitigation can be flagged as a fast-follow but the *correctness* of the fallback (it must trigger on failure, not just on success) cannot be deferred.

---

### Pitfall 5: API key/portal URL/stack trace leakage in error output from the new per-record call

**What goes wrong:**
`OCSFacility._portal_headers()` builds `{'Authorization': f'Token {api_key}'}`. If the new per-record API call's exception handling logs the raw exception object (`str(exc)`) from a `requests` failure, and that exception is a `requests.exceptions.HTTPError` raised by `response.raise_for_status()`, the exception's `.response.request.headers` (and therefore the `Authorization: Token <key>` value) can end up in `repr()`/`str()` of certain requests-library exception chains, or in a traceback if the command is run with `--traceback`/under Django's default uncaught-exception handling. The existing v1.2 code's `self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')` pattern is safe today only because the current exceptions are local `KeyError`/`ValueError` with no credential material in them — extending that exact f-string pattern to wrap the new API exception without first confirming what `str(exc)` actually renders for `requests`/`ImproperCredentialsException` could leak the token into command output, which may be captured in shell history, CI logs, or a cron job's emailed stdout/stderr.

**Why it happens:**
`ImproperCredentialsException('SOAR: ' + str(response.content))` (the actual exception SOAR's `make_request` raises on 4xx, per the installed `soar.py` source) embeds the **raw response body** in the exception message — if the LCO/SOAR API ever echoes back request metadata in an error body (common for auth-failure responses, e.g. "Invalid token <token>" style messages from some API gateways), that body text flows straight into `str(exc)` and then straight into stderr/logs via the existing skip-and-log pattern.

**How to avoid:**
- Before reusing the existing `self.stderr.write(f'...: {exc}')` pattern for the new API-call exception type, inspect what `ImproperCredentialsException`/`requests.exceptions.HTTPError` actually stringify to in this library version, and explicitly truncate/redact if response body content is included — at minimum, never log `response.content` or `response.text` verbatim for a failed auth request.
- Never log the `Authorization` header or the configured `api_key` setting value directly, even at `debug` level (per this repo's logging convention of using `debug` for expected failures — a debug log is still a log).
- Prefer logging a fixed, generic message ("API lookup failed for observation_id=X, status=<exc.__class__.__name__>") over interpolating the full exception text for this specific failure class, even though the existing code interpolates the full exception for `KeyError`/`ValueError` (those are safe; this one is not).

**Warning signs:**
- Any `f'{exc}'` or `str(exc)` applied to an exception raised from the new API call path without first checking what that exception actually contains.
- Grepping log output / stderr capture in tests for the literal dev API key value (even though it's blank in this environment, a test using a fake non-blank key should confirm it never appears in command output).

**Phase to address:**
The phase that adds the per-record API call — the redaction discipline must be designed alongside the exception handling from Pitfall 4, not bolted on after a leak is noticed. A test asserting a fake-but-nonblank API key never appears in `stdout`/`stderr` capture during a simulated auth-failure is the concrete verification.

---

### Pitfall 6: The 8-site LCO MPC-code mapping table is hand-built from a marketing webpage, not a verifiable API, and will silently go stale

**What goes wrong:**
PROJECT.md documents the verified mapping table's source as `https://lco.global/observatory/sites/mpccodes/` — a public webpage, not a versioned API endpoint or a library constant. `LCOSettings.get_sites()`/`SOARSettings.get_sites()` in the **installed** `tom_observations` library are the actual authoritative-ish source for site existence (used for the visibility/weather tooling), and as of this library version `LCOSettings.get_sites()` lists only 6 sites (Siding Spring/coj, Sutherland/cpt, Teide/tfn, Cerro Tololo/lsc, McDonald/elp, Haleakala/ogg) — **not 8**. The two sites in the v1.2 `SITE_TELESCOPE_MAP` (`coj`, `ogg`) match; the v1.3 target of "8 real LCO sites" includes `tlv` (Wise Observatory) and `sor` (SOAR Cerro Pachón, from `SOARSettings.get_sites()`), neither of which appears in `LCOSettings.get_sites()`. If the new static mapping dict is built purely from the webpage table without cross-checking against what the *installed library version actually returns* from `get_sites()`, a future `tom_observations`/`tom_eso` upgrade that adds/removes a site code will silently desynchronize the FOMO-local dict from the library's notion of valid sites, with no test or import-time check to catch the drift.

**Why it happens:**
There is no single authoritative source: the webpage table is human-curated (and was last fetched as of milestone start, June 2026 — it can change at any time without notice), the library's `get_sites()` only covers visibility/weather metadata (not the fully-qualified `siteid-enclid-telid`→MPC-code mapping this milestone actually needs), and the real per-record API response (`/api/requests/<id>/observations/`) is the only way to get the actual `site`/`enclosure`/`telescope` triple used for a given observation — none of these three sources is both complete and machine-verifiable today.

**How to avoid:**
- Treat the static mapping dict as explicitly versioned/dated in a code comment (mirroring the existing `[ASSUMED]` convention already used in `SITE_TELESCOPE_MAP`), citing both the webpage URL and the date it was checked, so a future maintainer knows to re-verify rather than trust it blindly.
- Add a runtime sanity check (not necessarily blocking, but at least a warning) when the per-record API call returns a `siteid-enclid-telid` combination not present in the static dict — fall back to the coarse instrument-class label per the milestone spec, but also surface a distinct "unmapped site" count in the summary line, distinct from "API call failed" — these are different failure modes (Pitfall 4 is "couldn't reach the API"; this is "reached the API, got an answer, but don't recognize it") and conflating them in one fallback-and-silently-continue path will hide real site-table drift from whoever runs the command.
- Cross-check the static dict against `LCOFacility().facility_settings.get_sites()` / `SOARFacility().facility_settings.get_sites()` at test-write time (not runtime — these don't return MPC codes) to at least confirm the *3-letter site codes* the static dict claims to cover are a superset of what the installed library itself considers valid LCO+SOAR sites; investigate `tlv` (Wise) since it doesn't appear in either `get_sites()` method in this library version, which may mean it is a real site not yet surfaced in OCS's settings metadata, or that the webpage table includes a site that isn't actually live in this network — confirm before shipping.

**Warning signs:**
- The static mapping dict has no comment indicating when/how it was verified.
- An "unmapped site" event is silently treated identically to an "API call failed" event in the summary counts.
- `tlv` appears in the static dict's site list but was never confirmed against a real `ObservationRecord` or a library `get_sites()` call.

**Phase to address:**
The phase that builds the verified static site/telescope mapping dict. The distinct "unmapped site" counter (separate from "API failed") should be a success criterion of that same phase, since it's the mechanism that will actually catch future drift in production use, long after this milestone closes.

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Reuse a single shared `LCOFacility()` instance for both LCO and SOAR records in the sync loop | Less code, matches v1.2's existing `facility = LCOFacility()` pattern | Silently calls the wrong portal URL/API key for SOAR records (Pitfall 2); breaks the moment SOAR gets distinct settings | Never — instantiate per-facility-name inside the loop or memoize by facility name |
| Hand-typed `_FAILURE_PREFIX_BY_STATUS`-style snapshot dicts (already used for failure states) extended to site/telescope mapping | Avoids depending on an unstable/nonexistent upstream API for the mapping | Drifts silently from upstream library/site reality (Pitfall 6) with no automatic detection | Acceptable only with an explicit dated-comment provenance note and a distinct "unmapped" runtime counter, never as a fire-and-forget constant |
| Catching the new API-call exception in the same `except (KeyError, ValueError)` clause as the existing data-shape errors | Minimal code change, looks like "just adding a type to the tuple" | Loses the ability to distinguish "bad data" skips from "API unreachable" skips in the summary line; can also accidentally swallow auth-leak-prone exception messages with the same `f'{exc}'` logging (Pitfall 5) | Never for this milestone — use a separate except clause/counter |
| Treating an `ALL` proposal value as "just don't pass `parameters__proposal__in`" without also reconsidering whether `facility` should default to `LCO+SOAR` simultaneously | Simpler mental model: "ALL means no filter at all" | If `--proposal ALL` is combined with a facility scope that isn't also explicit, a typo in the facility default could resync far more records than intended for an "ALL proposals" run — the blast radius of a mistake is much larger once both axes are unrestricted | Acceptable if the command prints (and ideally requires `--yes`/confirmation for) the record count it's about to touch before processing when `proposal=ALL` is combined with a broad facility scope |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `tom_observations.facilities.soar.SOARFacility` / `SOARSettings` | Assuming SOAR needs (or already has) its own independent `FACILITIES['SOAR']` entry with separate credentials | Confirm `FACILITIES['SOAR']` exists in `settings.py`/`local_settings.py` before shipping; per SOAR's own docstring it deliberately reuses the LCO API key/portal — the real risk is the entry being *absent* (this dev environment confirms it currently is), causing silent unauthenticated calls, not a credentials mismatch |
| `OCSFacility.get_observation_status()` / `/api/requests/<id>/observations/` endpoint | Calling this once per record with no caching or batching, assuming the existing pure-DB command's performance characteristics still hold once HTTP calls are added | Add a per-`(site,enclosure,telescope)` or per-`observation_id` cache within a single command invocation; add an explicit `timeout=` since the installed `make_request` helper passes none |
| `ObservationRecord.parameters` JSON shape across instrument types (flat `instrument_type` vs `c_1..c_5_instrument_type` vs MUSCAT's `_exposure_time_g/_r/_i/_z`) | Assuming all `ObservationRecord` rows for "LCO-family" facilities share one parameters shape, since they share one Django model/field | Branch the "find the meaningful instrument config" logic on whichever exposure-time key variant is actually present, not on a single hardcoded key name; test against SOAR-3-config and MUSCAT-per-channel shapes explicitly (Pitfall 3) |
| Django `CommandParser` `--proposal` argument | Assuming `type=str` plus a comma in the user's input means Django gives you a list | Split the raw string manually inside `handle()`, after checking for the `ALL` sentinel, before building any queryset filter (Pitfall 1) |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-record synchronous HTTP call with no timeout, no batching, inside the existing `for record in records:` loop | Command runtime scales linearly with proposal size once the API call is added; a single slow/hung connection blocks the entire run | Add `timeout=`; cache per-`(site,enclosure,telescope)` triple within one invocation; consider whether OCS's list/filter endpoints (`/api/requestgroups?request_id=`) could batch-fetch rather than fetch-per-record | Becomes noticeable once a proposal/`ALL` run spans more than roughly a few dozen records, or if any single LCO/SOAR API call is slow/hung — both plausible the first time this runs against a large real proposal or `--proposal ALL` |
| `--proposal ALL` combined with `facility__in=['LCO','SOAR']` and the new per-record API call, run with no progress indication | A full unattended sync becomes the slowest, most failure-prone path in the whole command, exactly when it's also touching the most records, with no feedback while it runs | Print incremental progress (e.g. every N records) to stdout for `ALL`-scoped runs; treat `ALL` as the worst-case path to load-test before shipping, not an edge case | Breaks first against `ALL` specifically, since that's the path with the largest N and the least amount of existing test coverage (v1.2's tests are all single-proposal, small-N) |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Logging the raw exception/response body from a failed per-record API call via the same `f'...: {exc}'` pattern used for the existing `KeyError`/`ValueError` skip messages | `Authorization: Token <key>` or response-body-echoed credential material ends up in stderr/logs/CI output (Pitfall 5) | Use a fixed, generic message for the new exception class; never interpolate `str(exc)` for `requests`/`ImproperCredentialsException` without first confirming what it actually renders |
| Assuming a blank dev `LCO_API_KEY` means the auth-failure code path is "safe to skip testing" | The failure/fallback path ships untested and the redaction discipline above is never actually exercised before real users hit it in production | Use HTTP mocking (`responses`/`unittest.mock.patch`) to simulate both a 401/403 and a network timeout against a **non-blank fake key**, and assert the fake key never appears in captured stdout/stderr |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| `--proposal ALL` silently touches every LCO+SOAR record with no confirmation or dry-run option | An operator typo or misunderstanding of scope causes thousands of `CalendarEvent` create/update operations with no warning | Print the matched record count before processing for `ALL`/large multi-proposal runs; consider a `--dry-run` flag that reports counts without writing |
| Summary line (`created/updated/unchanged/skipped`) doesn't distinguish *why* records were skipped once there are 3 distinct skip reasons (bad data shape, API unreachable, unmapped site) | An operator sees "skipped: 12" and has no way to tell if it's a real outage worth re-running later, a permanent data problem, or a stale site table | Break the single `skipped_count` into named sub-counts (`skipped_data`, `skipped_api`, `skipped_unmapped_site`) in the final summary line, consistent with the existing single-line summary convention |

## "Looks Done But Isn't" Checklist

- [ ] **`--proposal` comma-list/`ALL` support:** Often missing a test with 2+ comma-separated codes plus a single-character-code false-positive check — verify a fixture with `--proposal A,B` does not also match a hypothetical proposal literally named `A` *and* `B` combined as one string, and that `ALL` is tested against records spanning more than one proposal.
- [ ] **Facility generalization to LCO+SOAR:** Often missing confirmation that `FACILITIES['SOAR']` actually exists and is non-blank in `settings.py`/`local_settings.py` — verify via `SOARSettings('SOAR').get_unconfigured_settings()` returning `[]`, not just via code review of the Python diff.
- [ ] **`c_1..c_5` instrument-config scan:** Often missing SOAR's 3-config (Spectrum/Arc/Lamp Flat) and MUSCAT's per-channel (`_g/_r/_i/_z`) shapes in test fixtures — verify both are present as dedicated tests, not just the single-config shape that motivated the fix.
- [ ] **Per-record LCO API call with fallback:** Often missing an actual *failure-path* test (vs. happy-path-only) — verify via HTTP mocking that a simulated timeout/4xx/5xx triggers the coarse-label fallback, increments a distinct counter, does not crash the command, and does not leak the (fake, non-blank) API key into stdout/stderr.
- [ ] **Verified static site/telescope mapping dict:** Often missing a distinct "unmapped site code" code path separate from "API call failed" — verify the summary line reports them differently, and that `tlv`'s presence in the dict was actually confirmed against a real record or library `get_sites()` call, not just the webpage table.
- [ ] **Partial-failure/atomicity for a multi-hundred-record sync:** Often missing any statement (in docs or code comments) of what state the calendar is left in if the command is killed/crashes partway through — verify re-running after an interrupted run produces the same end state as one uninterrupted run (idempotency under interruption, not just under repeated full runs).

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|-----------------|
| Pitfall 1 (comma-list/`ALL` parsed wrong) | LOW | Fix the split logic, re-run the command — `CalendarEvent` upsert is keyed on `url`, so a corrected re-run naturally reconciles any previously-missed or wrongly-scoped records without manual cleanup |
| Pitfall 2 (SOAR silently unauthenticated) | MEDIUM | Add `FACILITIES['SOAR']` config, re-run; any `CalendarEvent`s created from unauthenticated/partial API responses during the misconfigured window need a manual audit (diff against a corrected re-run) since the upsert won't know which prior records were built from bad data unless a marker/version field is added |
| Pitfall 3 (wrong instrument config picked) | LOW–MEDIUM | Fix the scan logic, re-run — the no-churn comparison (already implemented in v1.2 per `_build_event_fields`) will detect the now-different `instrument` field value and update the event in place; no manual cleanup needed beyond the re-run itself |
| Pitfall 4 (partial failure mid-run) | MEDIUM | Because each record is processed independently with its own try/except and `get_or_create`/save, a crashed run can simply be re-run from scratch — already-correct events are no-ops (no-churn), records not yet reached get processed normally; the cost is mainly diagnosing *why* it crashed (no timeout, unhandled exception type) before re-running |
| Pitfall 5 (credential leak in logs) | HIGH | Rotate the leaked API key immediately if any real (non-blank) key was ever exposed in logs/CI output; scrub log retention/CI artifact history if feasible; this is the one pitfall here with cost beyond "just re-run the command" |
| Pitfall 6 (stale site mapping) | LOW | Add the missing site code to the static dict once discovered (via the distinct "unmapped site" counter from the prevention strategy), re-run — no data corruption, just delayed/missing calendar coverage for the unmapped site until fixed |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1: comma-list/`ALL` parsing | Phase generalizing `--proposal` | Test with 2+ comma-separated codes and a single-char-code false-positive check; test `ALL` against multi-proposal fixtures |
| 2: SOAR silent unauthenticated fallback | Phase generalizing facility scope to LCO+SOAR | `SOARSettings('SOAR').get_unconfigured_settings() == []` check (or equivalent settings audit) plus a test asserting per-record facility-class selection uses the record's own `facility` field, not a hardcoded `LCOFacility()` |
| 3: wrong `c_N` config picked | Phase replacing flat `instrument_type` with the `c_1..c_5` scan | Dedicated SOAR-3-config and MUSCAT-per-channel test fixtures, both asserting `c_1`'s instrument type wins |
| 4: per-record API call N+1/timeout/partial-failure | Phase adding the per-record LCO API call + fallback | `timeout=` present in the call; HTTP-mocked failure-path test asserting fallback label + distinct counter + no crash |
| 5: credential/stack-trace leakage | Same phase as Pitfall 4 (API call) | Test asserting a fake non-blank API key never appears in captured stdout/stderr during a simulated auth failure |
| 6: stale/unverifiable site mapping | Phase building the verified 8-site static mapping dict | Distinct "unmapped site" counter in the summary line; dated provenance comment; cross-check against `get_sites()` for the 6 sites it does cover |

## Sources

- `solsys_code/management/commands/sync_lco_observation_calendar.py` (this repo, v1.2 shipped code) — read directly, basis for Pitfalls 1, 3, 4, 5.
- `/home/tlister/venv/fomo311_venv/lib/python3.11/site-packages/tom_observations/facilities/ocs.py` (installed `tom_observations` library source, `OCSSettings`/`OCSFacility`/`make_request`/`_portal_headers`/`get_observation_status`) — read directly, basis for Pitfalls 2, 4, 5.
- `/home/tlister/venv/fomo311_venv/lib/python3.11/site-packages/tom_observations/facilities/lco.py` (installed library source, `LCOSettings.get_sites()`, `LCOMuscatImagingObservationForm._build_instrument_config`) — read directly, basis for Pitfalls 3, 6.
- `/home/tlister/venv/fomo311_venv/lib/python3.11/site-packages/tom_observations/facilities/soar.py` (installed library source, `SOARSettings`, `SOARFacility`, `SOARSimpleGoodmanSpectroscopyObservationForm`) — read directly, basis for Pitfalls 2, 3, 6.
- `/home/tlister/git/fomo_devel/src/fomo/settings.py` (this repo's actual `FACILITIES` dict, confirmed `'SOAR'` key absent, `LCO`'s `api_key` blank) — read directly, basis for Pitfalls 2, 4.
- `/home/tlister/git/fomo_devel/solsys_code/tests/test_sync_lco_observation_calendar.py` (this repo's existing test fixtures, confirmed single-proposal/single-config blind spots) — read directly, basis for Pitfalls 1, 3.
- `.planning/PROJECT.md` (this repo's milestone context: real-data correctness bug, 8-site mapping table provenance, target features/out-of-scope) — read directly, basis for milestone framing and Pitfall 6.
- Direct empirical test in this environment: `django.core.management.base.CommandParser` confirmed to pass `--proposal A,B,C` through as the raw string `'A,B,C'`, no implicit list-splitting — basis for Pitfall 1's HIGH confidence.

---
*Pitfalls research for: generalized LCO-family facility sync (FOMO v1.3 milestone)*
*Researched: 2026-06-18*
