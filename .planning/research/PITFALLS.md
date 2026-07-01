# Pitfalls Research

**Domain:** Adding ESO/VLT `ObservationRecord` calendar sync (`sync_eso_observation_calendar`) to an existing TOM Toolkit-based FOMO sync architecture (v1.7)
**Researched:** 2026-07-01
**Confidence:** HIGH for facts read directly from the installed `tom_eso==0.2.4` / `p2api` source and this repo's DB/settings (primary source); MEDIUM for the ESO Phase 2 OB-status vocabulary (single official ESO doc, not cross-verified against a second source)

**Method:** Read `tom_eso/eso.py`, `tom_eso/eso_api.py`, `tom_eso/models.py`, and `tom_common/session_utils.py` from the environment's installed `tom-eso==0.2.4` package (the exact version pinned in this repo's venv and already registered in `settings.py`'s `TOM_FACILITY_CLASSES`). Cross-checked `settings.py` for an `'ESO'` `FACILITIES` entry and queried `src/fomo_db.sqlite3` for existing `ObservationRecord(facility='ESO')` rows. Fetched ESO's official Phase 2 status-code documentation for the real OB status vocabulary.

## Critical Pitfalls

### Pitfall 1: Assuming `ESOFacility.get_observation_status()` returns OB execution/completion state

**What goes wrong:**
Code written by analogy to LCO (`record.status` populated from a working `get_observation_status()`/polling flow) silently assumes ESO records will likewise carry a live, sync-able status. In the actually-installed `tom_eso==0.2.4`, `ESOFacility.get_observation_status(observation_id)` is:

```python
def get_observation_status(self, observation_id):
    raise NotImplementedError
```

There is no working implementation to call, and no periodic status-refresh path exists anywhere in `tom_eso` today. This directly resolves the open question raised in the milestone: **OB execution/completion status is *not* knowable through the `tom_eso` facility API as currently shipped.** The only place real per-OB status data appears at all is the raw ESO P2 API response's `obStatus` key (visible only in a commented-out line in `tom_eso/eso_api.py::folder_item_choices`/`folder_ob_choices`, e.g. `f"{item['name']} : {item['itemType']} : {item['obStatus']}"` — deliberately stripped out of the returned choices tuples).

**Why it happens:**
Pattern-matching against LCO/Gemini's working `get_observation_status()`-backed sync loop, without first checking whether the *specific* method actually exists for ESO. This is the same class of mistake as v1.2's flat `instrument_type` assumption — trusting the facility abstraction's shape instead of the concrete installed version's behavior.

**How to avoid:**
Before writing any status-mapping code, grep the installed `tom_eso` package (`python -c "import tom_eso; print(tom_eso.__file__)"` then read `eso.py`) for `get_observation_status`. Confirm it raises `NotImplementedError`. Treat "sync OB status" as **out of scope for v1.7** unless a separate, explicit design decision is made to bypass `ESOFacility` entirely and call `ESOAPI.getOB(ob_id)` directly (raw p2api `getOB()` response, which does carry `obStatus`) — this requires its own credentialed API round trip per record, its own vocabulary mapping (see Pitfall 5), and its own error handling, not a reuse of the existing `record.status` field pattern.

**Warning signs:**
Any plan/task that writes `record.status` derivation logic for ESO records, or that calls `facility.get_observation_status(...)` in the new command, without a preceding research task that confirmed the method's real (non-stub) behavior.

**Phase to address:**
Phase 1 (scoping/research spike) — must resolve this *before* committing to a design that assumes status sync is possible. If ESO OB status turns out to require a second per-record P2 API call, that should be its own explicitly-scoped phase, not folded silently into the initial sync command.

---

### Pitfall 2: Assuming `ESOFacility.get_observation_url()` gives a portal URL to key idempotency on

**What goes wrong:**
LCO's sync is keyed on `CalendarEvent.url = LCOFacility().get_observation_url(observation_id)` (a real, working method). Gemini has no portal URL either, and v1.5 worked around it with a hand-built `GEM:{prog}/{observation_id}` key. For ESO, `get_observation_url()` is also a stub:

```python
def get_observation_url(self, observation_id):
    raise NotImplementedError
```

Calling it directly (as the LCO command does) will raise at runtime, not silently return an empty/wrong string — so this fails loudly in dev/test, but only if a record with `facility='ESO'` actually gets exercised (see Pitfall 6: none currently exist in this DB, so a naive implementation could pass all existing tests and CI while still being broken against the first real record).

**Why it happens:**
Copy-adapting `_build_event_fields`'s `url = facility.get_observation_url(record.observation_id)` line from `sync_lco_observation_calendar.py` without checking that the method is implemented for the target facility.

**How to avoid:**
Build the idempotency key by hand from data that reliably exists on ESO `ObservationRecord.parameters`/fields, the same way Gemini's `GEM:{prog}/{observation_id}` was built — e.g. something like `ESO:{p2_environment}/{ob_id}` (see Pitfall 4 for why `p2_environment` needs to be part of the key, not just `ob_id`). Confirm the exact field names actually present in `ObservationRecord.parameters` for an ESO record before finalizing the key shape (see Pitfall 6).

**Warning signs:**
Any call to `facility.get_observation_url(...)` or `facility.get_observation_status(...)` in the new ESO sync module without a `try/except NotImplementedError` or a preceding confirmation the method works.

**Phase to address:**
Phase 1 (design) — the idempotency-key scheme must be settled before any create-or-update logic is written, exactly as it was for GEM-KEY-01 in Phase 10.

---

### Pitfall 3: Reusing LCO's/Gemini's static-credential pattern for ESO's per-user encrypted session-bound password

**What goes wrong:**
LCO/SOAR use a single static `FACILITIES['LCO']['api_key']` string. Gemini uses a static per-site `FACILITIES['GEM']['api_key']` dict. ESO's credential model is fundamentally different and more fragile for a headless management command:

- Primary path: a per-*Django-user* `ESOProfile` model (`tom_eso/models.py`) storing `p2_username`, `p2_environment`, and an **encrypted** `p2_password` (`EncryptedProperty` backed by a `BinaryField`).
- Decryption (`tom_common.session_utils.get_encrypted_field`) requires an **active Django session** for that user: it looks up `UserSession.objects.filter(user=user).first()`, pulls the Fernet cipher key that was derived from the user's *login password* and stashed in their session at login time, and decrypts with it. If no active session exists for that user (near-certain for a `./manage.py sync_eso_observation_calendar` cron invocation with nobody logged in), `get_encrypted_field` **returns `None` silently** (logged only as a `warning`, no exception raised) — `ESOAPI(environment, username, None)` would then be constructed with a `None` password.
- Fallback path: `ESOFacility._configure_credentials()` falls back to `self._get_setting_credentials('ESO', ['p2_username', 'p2_password', 'p2_environment'])` reading plaintext values from `settings.FACILITIES['ESO']` — but **this repo's `settings.py` currently has no `'ESO'` key in `FACILITIES` at all** (confirmed by grep; only `LCO`, `SOAR`, `GEM` exist). Out of the box, a headless run has neither a usable session-decrypted password nor a settings fallback, and lands in `CredentialStatus.NOT_INITIALIZED`.

**Why it happens:**
Assuming "ESO differs from LCO's simple API-key header" means "a different header format," when the actual difference is architectural: ESO credentials are designed around an interactive, logged-in web session (P2 Tool iframe), not a background job. This is a materially bigger gap than the LCO-vs-Gemini credential difference the project has already handled.

**How to avoid:**
1. Add a `'ESO': {'p2_username': ..., 'p2_password': ..., 'p2_environment': ...}` entry to `FACILITIES` in `settings.py` (mirroring the `LCO`/`SOAR`/`GEM` pattern) and use `ESOFacility._get_setting_credentials(...)`/the settings-fallback path explicitly in the management command — never rely on `ESOProfile` + session decryption for a cron-style command.
2. If a real per-user `ESOProfile` must be used instead (e.g. because credentials are user-scoped for compliance reasons), the command needs to call `facility.set_user(user)` with a real logged-in-adjacent session available, or accept that decryption will return `None` and fail closed — and that failure path must be surfaced loudly (raise/log-and-abort), never silently proceed with a `None` password.
3. Apply the same credential-scrubbing discipline used for Gemini (GEM-SECURE-01: strip `password` before any logging) to `p2_password`/`p2_username` here — the encrypted-field machinery adds new places (exception messages from `ESOAPI.__init__`, `p2api` connection errors) where a plaintext password could leak into logs if not scrubbed first.

**Warning signs:**
Command runs green in a test using a `--settings`-injected fake facility/API but errors or silently produces zero synced records the first time it's run for real outside an interactive session; `ESOFacility.credential_status == CredentialStatus.NOT_INITIALIZED` in production logs; a `FACILITIES['ESO']` entry absent from `settings.py` at merge time.

**Phase to address:**
Phase 1 (design) for the credential-sourcing decision; a dedicated task/phase for the `FACILITIES['ESO']` settings entry + a `SYNC-09`-equivalent credential-scrubbing requirement, verified the same way GEM-SECURE-01 was (a test asserting the password/username never appear in stdout/stderr/logger output).

---

### Pitfall 4: Assuming a single fixed ESO site/telescope, missing the VLT's 4 Unit Telescopes and the per-environment site split

**What goes wrong:**
`ESOFacility.get_observing_sites()` is hardcoded to exactly two entries:

```python
return {
    'PARANAL':  {'sitecode': 'paranal', 'latitude': -24.62733, 'longitude': -70.40417, 'elevation': 2635.43},
    'LA_SILLA': {'sitecode': 'lasilla', 'latitude': -29.25667, 'longitude': -70.73194, 'elevation': 2400.0},
}
```
with a `# TODO: get data for all the ESO sites for production` comment left in place — it does not model APEX, VISTA, VST, La Silla's other telescopes, or (critically) VLT's 4 separate Unit Telescopes (UT1-4) at all. There is no `telescope` field anywhere in `get_observing_sites()`.

The *only* place a telescope-level string surfaces at all is `run['telescope']` inside the raw p2api `getRuns()` payload, consumed in `ESOAPI.observing_run_choices()` (`f"{run['progId']} - {run['telescope']} - {run['instrument']}"`) — an uncontrolled, un-enumerated string (could be `'UT1'`, `'UT2'`, `'UT3'`, `'UT4'`, `'VISTA'`, `'VST'`, `'NTT'`, etc., depending on the actual observing run).

Additionally, ESO's account model splits Paranal vs. La Silla by **environment/credential set**, not by a field on the OB itself: `ESOP2Environment` has three values — `demo`, `production` (Paranal), `production_lasilla` — so a single set of P2 credentials is scoped to *one* site's environment already. A design that reuses LCO's `SITE_TELESCOPE_MAP` pattern (one static dict, one shared credential set, dispatch purely on data returned per-record) will not work for ESO the same way: syncing both Paranal and La Silla requires two separate `ESOAPI`/credential configurations, not one shared lookup table keyed on a site code found in the record.

**Why it happens:**
Direct analogy to `sync_lco_observation_calendar`'s `SITE_TELESCOPE_MAP` (a single static dict covering every LCO/SOAR site under one shared credential) without noticing that ESO's site/telescope granularity (UT1-4) and its account/credential scoping (per-environment) are structurally different axes, both unaddressed by the installed library.

**How to avoid:**
1. Do not build a `calendar_utils`-style static `SITE_TELESCOPE_MAP` for ESO from `get_observing_sites()` alone — it is known-incomplete (`# TODO`) and has no telescope granularity.
2. Extract the actual telescope/UT identifier from the real per-record data (`run['telescope']` from the raw P2 payload, or whatever equivalent field ends up on `ObservationRecord.parameters` once real ESO records are examined — see Pitfall 6) rather than assuming a bare `'VLT'` label the way `SITES`'s `'Magellan'` bare-ambiguity was accepted for Baade/Clay.
3. Decide explicitly (as a D-0x decision, mirroring the SOAR-mirrors-LCO credential decision from Phase 5) whether v1.7 syncs Paranal only, La Silla only, or both — and if both, whether that means two credential configurations dispatched by facility/environment, analogous to the LCO+SOAR eager dispatch dict, but split by `p2_environment` instead of by TOM facility name.

**Warning signs:**
A `SITE_TELESCOPE_MAP`-equivalent for ESO with only 1-2 entries and no UT granularity; any code that hardcodes `telescope = 'VLT'` for every ESO record; a plan that treats "sync ESO" as single-site without an explicit decision about La Silla/Paranal scope.

**Phase to address:**
Same phase as Pitfall 1/2 research spike — telescope/site extraction design should be informed by real sample OB/`ObservationRecord` data (Pitfall 6), verified the same way the v1.3 `SITE_TELESCOPE_MAP` was operator-confirmed rather than shipped `[ASSUMED]`.

---

### Pitfall 5: Assuming ESO's terminal/failure-state vocabulary matches LCO's or Gemini's

**What goes wrong:**
`ESOFacility.get_terminal_observing_states()` returns `[]` (empty list) — there is no built-in terminal-state vocabulary to borrow, unlike LCO's `get_terminal_observing_states()`/`get_failed_observing_states()` (5-vs-4 states, the basis of `_FAILURE_PREFIX_BY_STATUS` and the D-06 research correction in Phase 4) or Gemini's simple boolean `ready` flag (GEM-STATUS-01).

The real ESO Phase 2 OB status vocabulary (from ESO's official Phase 2 documentation) is a 12-value, single-letter/symbol code set entirely different in shape from both existing facilities:

| Code | Name | Meaning |
|------|------|---------|
| P | Partially Defined | just created, fully editable |
| D | Defined | passed certification, limited editing |
| – | Rejected | needs user attention, re-editable |
| R | Review | under revision by support astronomer |
| + | Accepted | ready to be observed |
| C | Completed | executed successfully, will not repeat (terminal/success) |
| X | Executed | executed successfully, can repeat |
| M | Must Repeat | executed outside constraints, re-queued |
| A | Aborted | aborted during execution, re-queued |
| F | Failed | absolute time window expired (irreversible — terminal/failure) |
| K | Kancelled | support-astronomer-cancelled (terminal/failure) |
| T | Terminated | run itself was terminated (terminal/failure) |

Reusing `_FAILURE_PREFIX_BY_STATUS`'s LCO string keys (`'WINDOW_EXPIRED'`, `'CANCELED'`, etc.) against this vocabulary would silently never match anything — a fail-*open* bug (every ESO record gets a clean, non-prefixed title even when genuinely terminal/failed), which is worse than a loud crash because it looks correct in casual testing.

**Why it happens:**
Pattern-matching Phase 4's `_FAILURE_PREFIX_BY_STATUS` dict/D-06 approach onto a facility whose status vocabulary was never actually looked up, assuming "terminal states" is a universal TOM Toolkit concept with consistent string values across facilities (it explicitly is not — LCO, Gemini, and ESO each define it completely differently, or, for ESO, don't define it via the library at all).

**How to avoid:**
Build a dedicated ESO-specific status→prefix mapping keyed on the real single-letter/symbol codes above (e.g. `C`/`X` → clean title; `F`/`K`/`T`/`A`/`M` → some `[..._]` prefix scheme distinct from LCO's), sourced from ESO's own P2 documentation, not derived by calling `get_terminal_observing_states()` (which is unimplemented/empty) or by copying LCO's/Gemini's constants. Treat this as entirely new research, gated on Pitfall 1 first (whether OB status is obtainable at all through this sync path).

**Warning signs:**
Any status-prefix dict for ESO containing LCO string keys (`WINDOW_EXPIRED`, `CANCELED`, `FAILURE_LIMIT_REACHED`, `NOT_ATTEMPTED`) or a Gemini-style boolean `ready` check; `get_terminal_observing_states()` called and trusted without checking it returns `[]` for ESO.

**Phase to address:**
Same phase as Pitfall 1 (status is knowable at all) — this pitfall only matters once Pitfall 1 is resolved in favor of "yes, we bypass the facility API and pull raw `obStatus`." If Pitfall 1 resolves to "status sync is out of scope for v1.7," this pitfall becomes moot for this milestone and should be documented as deferred, not silently dropped.

---

### Pitfall 6: Fixture/test data not matching the real shape of ESO `ObservationRecord`s — compounded by zero currently existing

**What goes wrong:**
v1.2 shipped assuming a flat `instrument_type` key that doesn't exist in real LCO data; v1.3 found `SITE_TELESCOPE_MAP` had unconfirmed entries. For ESO this risk is worse, not just repeated: **this dev database currently has zero `ObservationRecord` rows with `facility='ESO'`** (confirmed by direct query against `src/fomo_db.sqlite3`), and `ESOFacility.submit_observation()`/`submit_new_observation_block()` **always returns an empty `created_observation_ids` list**:

```python
def submit_observation(self, observation_payload):
    self.submit_new_observation_block(observation_payload)
    created_observation_ids = []
    return created_observation_ids
```

This means the *normal* TOM Toolkit "submit an observation through the UI" flow that would populate `tom_observations.ObservationRecord` rows for LCO/Gemini **does not do so for ESO as currently implemented** — OBs are created via the P2 Tool iframe/API directly, outside TOM's standard submission bookkeeping. There is, right now, no confirmed mechanism by which a `ObservationRecord(facility='ESO')` row would ever exist in this app at all, let alone what shape its `.parameters` JSON would take (P2 OBs are JSON documents keyed on things like `obId`, `target`, `constraints`, `obStatus` — nothing resembling LCO's `c_1_instrument_type`/cadence-request shape or Gemini's `windowDate`/`obsid`/`prog` shape).

**Why it happens:**
Assuming "sync ObservationRecords for facility X" is a well-posed task by analogy to the two prior facilities, without first confirming (a) that ESO `ObservationRecord`s get created by any path in this app, and (b) if so, by what path and in what shape. Building extraction logic and test fixtures purely from imagination/LCO-shape-copying (exactly what caused the v1.2 rewrite) is far riskier here because there is no real row to check against and no working creation path to inspect.

**How to avoid:**
1. Before writing any extraction/sync logic, explicitly establish (with the operator, as was done for `SITE_TELESCOPE_MAP` in Phase 07 and the `tlv` site removal) *how* an `ObservationRecord(facility='ESO')` is expected to come to exist in this app — e.g., is it created by a different, not-yet-written import/polling path that reads P2 OBs directly and creates `ObservationRecord` rows manually (bypassing `submit_observation()`), is it out of scope until `tom_eso` matures, or is "sync" actually meant to operate directly against the P2 API rather than against `ObservationRecord` at all?
2. If real records genuinely don't exist yet, do not invent a plausible-looking `parameters` fixture shape and build against it uncritically — get at least one operator-confirmed real (or realistic, ESO-documentation-sourced) sample OB/record shape, the same discipline that caught the v1.2/v1.3 bugs when it was finally applied.
3. Explicitly flag in the roadmap/`PROJECT.md` that this milestone's "ObservationRecord sync" scope may need to be redefined (e.g. to "sync OBs from P2 API directly to CalendarEvent," skipping `ObservationRecord` as an intermediate model) once this is confirmed, rather than silently forcing ESO into the LCO/Gemini `ObservationRecord`-centric shape it may not actually use.

**Warning signs:**
A plan phase with fixture-only tests (factory-built `ObservationRecord(facility='ESO', parameters={...})` with hand-invented keys) and no task that checks/confirms this shape against anything real or ESO-documented; zero mention in the plan of *how* the ESO ObservationRecord rows the sync is meant to consume actually get created.

**Phase to address:**
Phase 1 (scoping/research spike) — this is the single highest-leverage question to resolve before any implementation phase, since it can invalidate the entire "sync_eso_observation_calendar operates on ObservationRecord like sync_lco/sync_gemini" premise.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|-----------------|------------------|
| Building `sync_eso_observation_calendar` against hand-invented `parameters` fixtures without operator/documentation confirmation | Unblocks writing tests/code immediately | Repeats the exact v1.2 rewrite pattern — likely `KeyError`s or silent skips against the first real record | Never — get at least a documentation-sourced or operator-confirmed shape first (Pitfall 6) |
| Copying `_FAILURE_PREFIX_BY_STATUS`'s LCO keys as a starting point "to adapt later" | Fast first draft | Fails open (never matches, no failures ever flagged) rather than fails loud | Only as a throwaway spike, never merged without replacing keys with ESO's real P2 codes |
| Relying on `ESOFacility.get_observing_sites()`'s 2-entry hardcoded dict as the full site list | Avoids extra research | Misses La Silla's other telescopes, APEX, VISTA, VST, and all 4 VLT UTs; the library's own `# TODO` comment flags it incomplete | Never for anything beyond a throwaway spike |
| Skipping the `FACILITIES['ESO']` settings entry and relying on `ESOProfile` + session decryption for a management command | No settings.py change needed short-term | Command silently gets `p2_password=None` outside an active user session — near-guaranteed for any cron/headless run | Never for a management command; fine only for the interactive web UI flow |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|-----------------|-------------------|
| `tom_eso.eso.ESOFacility` | Calling `get_observation_status()`/`get_observation_url()` expecting LCO-like behavior | Both raise `NotImplementedError` in the installed `0.2.4`; confirm with a direct source read before depending on either, and build a hand-rolled idempotency key (Pitfall 2) instead |
| `tom_eso.models.ESOProfile` / `tom_common.get_encrypted_field` | Assuming decryption works the same for a background job as it does mid-request | Decryption needs an active `UserSession`; a headless command should use the `settings.FACILITIES['ESO']` plaintext-credentials fallback path instead (Pitfall 3) |
| ESO P2 API (`p2api`) via `ESOAPI` | Treating `p2api.p2api.P2Error` the same as an LCO HTTP error | `eso_api.py` already catches `p2api.p2api.P2Error` in a few call sites (`folder_item_choices`, `folder_ob_choices`) — reuse that exception type, not a generic `requests`/HTTP exception class, when wrapping calls in the new sync command |
| ESO Phase 2 environments (`demo`/`production`/`production_lasilla`) | Assuming one credential set covers all ESO sites, like LCO's single portal covers all LCO/SOAR sites | Each `p2_environment` is a separate account/API connection scoped to one site; syncing multiple ESO sites needs multiple credential configurations, not one shared static dict (Pitfall 4) |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|-----------------|
| Per-record live P2 API call for status/telescope resolution (if Pitfall 1/5 end up requiring it) | Slow sync runs, timeouts under load, mirrors Phase 7's per-record LCO API resolution but against an even less mature client library | Apply the same SYNC-08-style discipline already proven for LCO (explicit timeout, single attempt, no retry loop) and the same dedicated failure counter pattern (SYNC-06) rather than inventing new retry logic | Any run syncing more than a handful of OBs, given `p2api`'s connection-per-call overhead is unverified/unbenchmarked in this codebase |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Logging/echoing `p2_password`, `p2_username`, or `ESOAPI` construction/connection exceptions verbatim | ESO P2 credentials (and the derived Fernet session key) leak into logs/stderr, worse than a leaked static LCO API key since it's tied to the user's real login password via key derivation | Scrub `p2_password`/`p2_username` before any logging, mirroring GEM-SECURE-01's `safe_params` pattern; never interpolate raw exception objects from `ESOAPI.__init__`/`p2api` connection errors (which may embed credentials) into stdout/stderr, mirroring the LCO sync's SYNC-09 fixed-message discipline |
| Assuming `get_encrypted_field`'s silent `None`-on-failure return means "no credentials configured" rather than "decryption failed" | A real, correctly-configured `ESOProfile` could still silently yield `p2_password=None` in a headless context, masking a configuration problem as an unconfigured-facility state | Distinguish "no `ESOProfile` exists" from "an `ESOProfile` exists but session decryption failed" in logging, so operators can tell the difference between "not set up" and "can't run headless" |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-------------------|
| Silently producing zero synced ESO events with no diagnostic output, because credentials never resolved (`CredentialStatus.NOT_INITIALIZED`) | Operator believes the sync command ran successfully and there's simply nothing to sync, when actually credentials were never usable | Have the command check and report `facility.credential_status` explicitly (loud, at the top of `handle()`), the same way LCO/Gemini report per-facility counters at the end |
| Treating an ESO OB in an intermediate P2 status (`P`/`D`/`–`/`R`) the same as a queued/placed LCO record | Calendar shows a banner/block implying the OB is scheduled/executing when it may still be in draft/review and could be rejected before ever running | Map only the P2-genuinely-scheduled-or-later statuses (`+`/`X`/`C`/`M`/`A`/`F`/`K`/`T`) to a CalendarEvent at all, and clearly label draft-stage statuses (`P`/`D`/`–`/`R`) differently or exclude them, once/if status sync is in scope |

## "Looks Done But Isn't" Checklist

- [ ] **Idempotency key:** Confirm the command never calls `facility.get_observation_url()`/`get_observation_status()` directly — both raise `NotImplementedError` in the installed `tom_eso==0.2.4`; verify via a test that exercises the real installed `ESOFacility`, not a mock that happens to implement the method.
- [ ] **Real fixture shape:** Verify test fixtures for `ObservationRecord(facility='ESO', parameters={...})` are backed by an operator-confirmed or ESO-documentation-sourced shape, not an LCO/Gemini-shape guess — check `git log`/plan docs for an explicit confirmation step, the same way `SITE_TELESCOPE_MAP` needed operator sign-off in Phase 07.
- [ ] **Credential path:** Verify the command uses the `settings.FACILITIES['ESO']` plaintext-credential fallback (or an explicitly designed headless-safe path), not a bare `ESOProfile` + session-decryption call that will silently yield `None` outside an interactive session.
- [ ] **Site/telescope granularity:** Verify the sync distinguishes at least Paranal vs. La Silla (and ideally VLT UT1-4) rather than a single hardcoded `'VLT'`/`'ESO'` telescope label for every record.
- [ ] **Status vocabulary:** Verify any status-to-title-prefix mapping is keyed on ESO's real P2 codes (`P`/`D`/`–`/`R`/`+`/`C`/`X`/`M`/`A`/`F`/`K`/`T`), not reused LCO/Gemini constants — check for a dict literal containing any of `WINDOW_EXPIRED`/`CANCELED`/`FAILURE_LIMIT_REACHED`/`NOT_ATTEMPTED` in the new ESO module (that would indicate a copy-paste error).
- [ ] **Credential scrubbing:** Verify a test asserts `p2_password`/`p2_username` never appear in stdout/stderr/logger output, mirroring the existing `GEM-SECURE-01` test for Gemini.
- [ ] **Paired demo notebook:** Per this repo's CLAUDE.md convention, any new `solsys_code/management/commands/sync_eso_observation_calendar.py` needs a paired `docs/notebooks/pre_executed/sync_eso_observation_calendar_demo.ipynb`, scoped into `files_modified` from the first plan, not bolted on after the fact (as happened twice already for other modules per CLAUDE.md).

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|----------------|-----------------|
| Shipped a status-prefix dict keyed on wrong (LCO/Gemini) vocabulary | LOW | Swap the dict's keys for ESO's real P2 codes; add a regression test asserting the old LCO/Gemini keys are absent from the new module |
| Shipped assuming `get_observation_url()`/`get_observation_status()` work, discovered `NotImplementedError` in later testing/production | MEDIUM | Same shape as the v1.2→v1.3 recovery: replace the direct facility-method call with a hand-built key/derivation, add a live-call smoke test the way `LCOFacility().get_observation_url('12345')` was confirmed live in Phase 4 |
| Command runs headless and silently produces `p2_password=None`/no synced records | MEDIUM | Add `settings.FACILITIES['ESO']` plaintext fallback; add a startup check that aborts loudly (not silently) if `facility.credential_status != USING_DEFAULTS/USING_USER_CREDS` |
| Discovered ESO `ObservationRecord`s never actually get created by any path in this app | HIGH | Requires re-scoping the milestone itself — may need a separate ingest/import command (P2 API → `ObservationRecord` or directly to `CalendarEvent`) before "sync" is even a well-posed operation; treat as a milestone-level scope revision, not a quick task |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|-------------------|----------------|
| OB status not exposed via `ESOFacility` (Pitfall 1) | Phase 1 (research spike) | A task explicitly reads `tom_eso.eso.ESOFacility.get_observation_status` source and documents its real (stub) behavior before any status-mapping code is written |
| No portal URL for idempotency key (Pitfall 2) | Phase 1 (design) | A test that a hand-built key (not `get_observation_url()`) is used, mirroring GEM-KEY-01's dedicated test |
| Session-bound encrypted credentials break headless runs (Pitfall 3) | Phase 1 (design) + a credential-handling task | A test running the command with no active `UserSession` for the relevant user, asserting it still authenticates via `settings.FACILITIES['ESO']` (or fails loudly, never silently with `None`) |
| Single fixed site/telescope, missing VLT UTs (Pitfall 4) | Phase 1 (research spike), informed by real sample data | Operator confirmation of the site/telescope scope for v1.7 (Paranal only? both? UT granularity?), mirroring the Phase 07 `SITE_TELESCOPE_MAP` sign-off |
| Wrong terminal-state vocabulary (Pitfall 5) | Same phase as Pitfall 1, once status sync is confirmed in scope | A dedicated ESO status-prefix dict test using the real P2 codes, with an explicit regression test asserting no LCO/Gemini status strings appear |
| Fixture shape doesn't match real ESO data; zero real records exist today (Pitfall 6) | Phase 1 (research spike) — the gating question for the whole milestone | An explicit checkpoint (like the Phase 07 `tlv`-site removal checkpoint) where the operator confirms how/whether `ObservationRecord(facility='ESO')` rows get created, before any extraction-logic phase begins |

## Sources

- `tom_eso/eso.py`, `tom_eso/eso_api.py`, `tom_eso/models.py` — read directly from this repo's installed `tom-eso==0.2.4` package at `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_eso/` (primary source, HIGH confidence)
- `tom_common/session_utils.py` — read directly from the installed `tom-common` package, same venv (primary source, HIGH confidence)
- `src/fomo/settings.py` — this repo's `FACILITIES`/`TOM_FACILITY_CLASSES` config, confirming no `'ESO'` `FACILITIES` entry exists (HIGH confidence)
- `src/fomo_db.sqlite3` — direct query confirming zero `ObservationRecord(facility='ESO')` rows exist in this dev DB (HIGH confidence)
- `.planning/PROJECT.md` — Key Decisions table documenting the v1.2 flat-`instrument_type` bug, the v1.3 `[ASSUMED]` `SITE_TELESCOPE_MAP` gaps, and the Gemini credential-scrubbing precedent (HIGH confidence, project history)
- ESO official Phase 2 status documentation, https://www.eso.org/sci/observing/phase2_p114/p2intro/phase-2-status.html — OB status code table (MEDIUM confidence, single official source not cross-verified against a second independent source)
- ESO Phase 2 API docs, https://www.eso.org/sci/observing/phase2_p115/p2intro/Phase2API.html and https://www.eso.org/sci/observing/phase2/p2intro/Phase2API/api--python-programming-tutorial.html — general P2 API background (MEDIUM confidence, referenced but not deeply read)

---
*Pitfalls research for: ESO/VLT ObservationRecord calendar sync (v1.7)*
*Researched: 2026-07-01*
