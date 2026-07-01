# Project Research Summary

**Project:** FOMO v1.7 — ESO/VLT ObservationRecord Calendar Sync (Stage 4)
**Domain:** Django/TOM Toolkit calendar integration with an incomplete ESO facility plugin
**Researched:** 2026-07-01
**Confidence:** HIGH for verified facts (code inspection, DB queries); MEDIUM for feasibility (depends on unresolved blockers)

## Executive Summary

ESO/VLT calendar sync appears straightforward on the surface — add `sync_eso_observation_calendar` following the proven LCO/Gemini pattern — but research has uncovered a critical blocker that reshapes the entire milestone: **the installed `tom_eso==0.2.4` plugin does not implement observation creation, status lookups, or URL generation at all, and there is currently no working path for `ObservationRecord(facility='ESO')` rows to exist in this codebase.** This is not a "not yet implemented" gap; it is a hardcoded limitation in the library (`submit_observation()` unconditionally returns an empty ID list, both `get_observation_status()` and `get_observation_url()` raise `NotImplementedError`, and `get_terminal_observing_states()` returns empty). Direct inspection of `tom_eso/eso.py`, queries against this repo's current database (zero ESO records), and verification of `settings.py` (no `FACILITIES['ESO']` entry) confirm this is not an assumption.

**Recommended approach:** Restructure this as a two-phase discovery:
1. **Phase 0 (Spike):** Confirm how/whether ESO `ObservationRecord` rows can be created at all, gather real P2 API data shapes, and decide between "Bridge" (create records from P2 API) vs. "Bypass" (read directly from P2 API, skip `ObservationRecord`). This is prerequisite to meaningful Phase 1 planning.
2. **Phase 1+ (Implementation):** Build the sync command and supporting infrastructure based on Phase 0's findings about data sources and credential handling.

**Key risks:** Shipping code against imagined fixture shapes that don't match real ESO P2 data (repeating the v1.2 flat-`instrument_type` and v1.3 `SITE_TELESCOPE_MAP` gaps). Assuming credential patterns work the same for headless management commands as for interactive web sessions (they do not — ESO's are session-bound encrypted keys). Assuming terminal/failure status vocabularies transfer across facilities (they do not — ESO's 12-letter codes are completely different from LCO's/Gemini's).

## Key Findings

### Recommended Stack

No new dependencies are required — `tom-eso==0.2.4`, `p2api==1.0.10`, and `tom_observations` are already installed and configured in `INSTALLED_APPS` / `TOM_FACILITY_CLASSES` (`src/fomo/settings.py`). This milestone is pure application code (a new management command and supporting helpers).

**Core technologies:**
- **`tom-eso==0.2.4` and `p2api==1.0.10`** — ESO Phase 2 facility plugin (already installed). `tom_eso.eso_api.ESOAPI` wraps `p2api.ApiConnection` and is the only documented entry point to OB metadata. Note: does NOT implement the standard TOM Toolkit `get_observation_status()` / `get_observation_url()` methods; if OB status/URLs are needed, they must be extracted from raw `p2api` responses directly.
- **`tom_observations.models.ObservationRecord`** — the model this sync reads from (or should read from, once the creation gap is resolved). Facilities-agnostic; no ESO-specific fields exist.
- **`solsys_code.calendar_utils.insert_or_create_calendar_event()`** — idempotent no-churn helper, reusable unchanged. Already proven to generalize across LCO and Gemini; Gemini's synthetic key approach (`GEM:{prog}/{observation_id}`) is the precedent for ESO (since `get_observation_url()` raises `NotImplementedError`).
- **`Observatory` model (this repo)** — site-coordinate source, following this project's established convention. Current database has La Silla (`809`) but **no Cerro Paranal record** (obscode `309` missing). Must be created for site resolution.

### Expected Features

**Must have for v1.7 launch (table stakes):**
- `sync_eso_observation_calendar` management command — one `CalendarEvent` per `ObservationRecord(facility='ESO')`, mirroring LCO/Gemini pattern.
- Idempotent, no-churn create-or-update keyed on a **synthetic string** (e.g., `ESO:{p2_environment}/{obId}`), since `ESOFacility.get_observation_url()` raises `NotImplementedError` and cannot be used as a key.
- Submission-time window banner (single state, Gemini-style) derived from OB's observing-run validity dates or PI-set absolute-time constraints. ESO Service Mode has no advance per-OB schedule (unlike LCO), so the honest model is "OB exists and is scheduled for this run period," not LCO's "queued→placed→terminal" state machine.
- **Resolution of the `ObservationRecord(facility='ESO')` creation gap** — explicitly scoped as a Phase 0 spike/decision, not assumed to be auto-solvable by the command itself.

**Should have (differentiators, v1.x after validation):**
- Deep link to ESO's Run Progress page (no FOMO re-implementation; just a hyperlink since Run Progress is web-only and requires ESO login).
- Absolute-time-constraint ingestion for the rare OB that has one (tighter window than run-period fallback).
- Telescope/instrument derived from run metadata (already exposed by `ESOAPI.observing_run_choices()`).

**Defer (v2+, blocked by unresolved infrastructure):**
- Real per-night execution status (`[EXECUTED]`/`[MUST REPEAT]`/`[FAILED]` prefixes, mirroring LCO's terminal-state behavior) — requires (a) resolving which night(s) to poll per OB, (b) a service-account credential story for headless commands (doesn't exist for ESO today, only per-user `ESOProfile`), and (c) mapping ESO's 12-letter status codes to a meaningful prefix scheme.
- VLT UT1-4 disambiguation — unknown whether the P2 API exposes per-unit-telescope assignment at all outside of ESO's internal short-term scheduler (unverified; Phase 0 spike required).

### Architecture Approach

The existing LCO/Gemini sync-command architecture is fundamentally built on the assumption that `ObservationRecord` rows already exist (created by the normal TOM submission flow) and that the facility provides `get_observation_status()` / `get_observation_url()` as a uniform interface. **For ESO, neither assumption holds.** `ESOFacility.submit_observation()` returns an empty ID list (so no records are ever created by the UI submission flow), and both status/URL methods raise `NotImplementedError`. This forces a choice between two structurally different approaches, both requiring Phase 0 investigation:

1. **Bridge option:** A separate command or initial step that reads from the ESO P2 API directly, creates `ObservationRecord` rows by hand (working around the `submit_observation()` limitation), then runs the standard LCO/Gemini sync pattern on top of those rows. Keeps the downstream pattern identical; adds a new responsibility (record creation from external API) neither existing command has.
2. **Bypass option:** `sync_eso_observation_calendar` reads directly from the P2 API (via `ESOAPI`/`p2api`, walking `getRuns()` → `getItems()` → `getOB()`), builds `CalendarEvent`s straight from OB data, keyed on P2 identifiers (e.g., `ESO:{p2_environment}/{obId}`), never touching `ObservationRecord` at all. Bigger philosophical break from LCO/Gemini precedent but matches what the plugin actually supports today.

**Reusable components (unchanged):**
- `insert_or_create_calendar_event()` — facility-agnostic, handles the idempotent create-or-update; no modifications needed.
- `CalendarEventTelescopeLabel` sidecar model — structure is reusable; wiring depends on whether Phase 0 finds a verified-vs-fallback telescope resolution for ESO.
- Counter/summary-line pattern from `sync_lco_observation_calendar.handle()`.

### Critical Pitfalls

1. **`ESOFacility.get_observation_status()` and `get_observation_url()` both raise `NotImplementedError`** — code written assuming these methods work (copying from LCO) will crash at runtime. Avoid by: building idempotency keys by hand (Gemini's `GEM:...` pattern is precedent), confirming via source inspection before relying on either method, never attempting to populate `record.status` from a facility method for ESO.

2. **There is currently no working path for `ObservationRecord(facility='ESO')` rows to exist in this codebase** — `ESOFacility.submit_observation()` unconditionally returns an empty list, so the standard "submit observation via TOM UI" flow never creates a record. This dev DB has zero ESO records. Avoid by: explicitly resolving how/whether ESO records are meant to be created (Bridge, Bypass, or external ingest) before Phase 1 begins, obtaining operator confirmation or ESO-documentation-sourced sample P2 OB shapes before finalizing field-extraction code.

3. **ESO credentials are per-user session-bound encrypted keys, not static facility API keys** — `ESOProfile` models require an active Django session for decryption. For headless management commands, decryption fails silently (`get_encrypted_field` returns `None`; no exception raised), and FOMO's `settings.py` currently has no `FACILITIES['ESO']` fallback. Avoid by: adding `FACILITIES['ESO']` plaintext-credential entry to `settings.py`, never relying on `ESOProfile` + session decryption for background jobs, applying credential-scrubbing discipline to prevent password leakage into logs.

4. **ESO's terminal/failure-state vocabulary is completely different from LCO's and Gemini's** — ESO Phase 2 OB status is a 12-letter code set (`P`/`D`/`–`/`R`/`+`/`C`/`X`/`M`/`A`/`F`/`K`/`T`). Copying LCO's `_FAILURE_PREFIX_BY_STATUS` dict produces a fail-open bug (never matches). Avoid by: building a dedicated ESO status→prefix mapping from real P2 codes, adding regression tests asserting no LCO/Gemini constants appear in ESO code, treating this as entirely new design.

5. **Single fixed site assumption misses ESO's Paranal/La Silla split and VLT UT1-4 granularity** — `ESOFacility.get_observing_sites()` hardcodes exactly two sites with `# TODO` comment. ESO's credentials also split by environment (`production` for Paranal, `production_lasilla` for La Silla). Avoid by: confirming with operator whether v1.7 syncs Paranal only, La Silla only, or both, extracting actual telescope/UT from real per-record data, not reusing `SITE_TELESCOPE_MAP`'s LCO pattern.

## Implications for Roadmap

Research indicates this milestone requires a **three-phase structure** (not the two-phase "plan then execute" normally suggested):

### Phase 0: Spike — Data Source & Credential Investigation
**Rationale:** The fundamental question "how do ESO `ObservationRecord` rows come to exist in this codebase, and in what shape?" cannot be answered from documentation alone. The installed plugin actively prevents the normal TOM submission path from creating records.

**Delivers:**
- Confirmed mechanism for ESO `ObservationRecord` creation (or explicit scope decision for Bridge/Bypass).
- Real sample ESO P2 API response shapes (OB JSON, run JSON) — not guessed patterns.
- Confirmation of which ESO sites (Paranal, La Silla, both?) are in v1.7 scope.
- Clarity on whether OB-level telescope/UT assignment is exposed by the P2 API.

**Avoids pitfalls:** 1 (status access), 2 (idempotency key), 3 (credentials), 6 (fixture shape).

### Phase 1: Design & Setup — Bridge/Bypass Decision, Credential Config, Observatory/Helpers
**Rationale:** Once Phase 0 answers "what data shape and what credential approach," Phase 1 makes the design-decision tasks and infrastructure changes that Phase 2's implementation depends on.

**Delivers:**
- `FACILITIES['ESO']` entry in `settings.py` with headless-safe credentials.
- Cerro Paranal `Observatory` record (obscode `309`).
- ESO-specific helper functions designed against real P2 data from Phase 0.
- SPEC.md design document clarifying Bridge vs. Bypass choice.
- Credential-scrubbing test setup.

**Avoids pitfalls:** 3 (credentials), 4 (site/telescope), 5 (status codes).

### Phase 2: Implementation — `sync_eso_observation_calendar` Command & Tests
**Rationale:** With Phase 0's data shapes and Phase 1's infrastructure in place, the command is a straightforward adaptation of LCO/Gemini pattern — constrained by real ESO data shapes.

**Delivers:**
- `sync_eso_observation_calendar.py` management command.
- Unit/integration tests against fixture ESO data from Phase 0.
- Paired demo notebook (`docs/notebooks/pre_executed/sync_eso_observation_calendar_demo.ipynb`), scoped into `files_modified` from the start.
- `CalendarEventTelescopeLabel` wiring (if Phase 0/1 found verified-vs-fallback telescope resolution).

**Avoids pitfalls:** 1 (status access), 2 (idempotency), 6 (fixture shape).

### Phase Ordering Rationale

- **Phase 0 (spike) is mandatory, not optional:** Unlike Gemini's Phase 10 (records already existed), ESO's Phase 0 must answer whether records can be created at all. Installed plugin actively prevents normal creation flow.
- **Phase 1 (design) before Phase 2 (command):** Bridge vs. Bypass choice, credential strategy, and site/telescope sourcing reshape Phase 2's implementation significantly. Scope separately so they don't block command coding.
- **Why Phase 2 is straightforward after Phase 0 and Phase 1:** Data shapes confirmed, infrastructure in place, tests grounded in real data (avoids v1.2/v1.3 fixture-shape pitfalls).

### Research Flags

**Phases requiring deeper research during planning:**
- **Phase 0 (Spike):** MANDATORY. Resolves `ObservationRecord` creation/data-sourcing question through operator confirmation and P2 API exploration. Gating dependency.
- **Phase 1 (Design):** Moderate research — credential sourcing, site/telescope scope, Bridge vs. Bypass implications. Builds on Phase 0 findings.

**Phases with standard patterns (minimal additional research):**
- **Phase 2 (Implementation):** Standard pattern once Phase 0/1 prerequisites met. Command closely parallels LCO/Gemini; pitfalls well-documented; minimal new research.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| **Stack** | **HIGH** | `tom-eso==0.2.4`, `p2api==1.0.10` read directly from installed packages. Limitation verified by source inspection. |
| **Features** | **MEDIUM** | Table-stakes features clear (sync command, idempotent key, single-state banner). Blocking dependency (ObservationRecord creation) confirmed but unresolved. Differentiators viable but deferred. |
| **Architecture** | **MEDIUM-HIGH** | Existing LCO/Gemini pattern is HIGH confidence. ESO integration is MEDIUM because data source (Bridge vs. Bypass) is unresolved. Pitfalls well-documented. |
| **Pitfalls** | **HIGH** | Five critical pitfalls identified with root-cause analysis and avoidance strategies. All verified by source inspection or documented ESO API behavior. |

**Overall confidence: MEDIUM**

Research has successfully identified fundamental blockers and provided clear avoidance strategies. However, Phase 0 (spike) must precede Phase 1 planning. Once Phase 0 confirms data sourcing and credential strategy, roadmap can confidently structure Phase 1 and Phase 2.

### Gaps to Address

1. **`ObservationRecord(facility='ESO')` creation path:** Currently unresolved. Phase 0 must confirm whether records are created by planned-but-unwritten import, via Bridge/Bypass option, or deferred entirely.
2. **Real P2 API response shapes:** Research could not inspect live P2 API (no active ESO credentials in dev environment). Phase 0 must obtain real samples.
3. **Telescope/UT assignment in P2 API:** Unknown whether `getOB()` includes which UT an OB is scheduled on, or if visible only in ESO's internal scheduler. Phase 0 must inspect real response.
4. **Absolute-time-constraint ingestion:** Rare feature mentioned but not analyzed in depth. Phase 0 should note whether it's worth implementing for v1.7 or v1.x follow-up.
5. **Multi-site credential sourcing:** If Phase 0 confirms both Paranal + La Silla should be synced, Phase 1 must design two-credential-configuration dispatch (split by `p2_environment`).

## Sources

### Primary (HIGH confidence — read directly from source)
- `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_eso/eso.py` (installed `tom-eso==0.2.4`)
- `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/p2api/p2api.py` (installed `p2api==1.0.10`)
- `src/fomo/settings.py` (this repo) — confirmed `FACILITIES` lacks `'ESO'` key
- Direct dev DB query — `ObservationRecord.objects.count()` = 0; no Paranal `Observatory` record
- Existing sync commands (`sync_lco_observation_calendar.py`, `sync_gemini_observation_calendar.py`) — shipped reference implementation

### Secondary (MEDIUM confidence — official docs)
- [ESO Phase 2 Status](https://www.eso.org/sci/observing/phase2/p2intro/phase-2-status.html) — OB status code definitions
- [ESO Phase 2 API Documentation](https://www.eso.org/sci/observing/phase2/p2intro/Phase2API.html) — schema reference
- `.planning/PROJECT.md`, `.planning/codebase/INTEGRATIONS.md`, `CLAUDE.md` (this repo) — project history and conventions

---

*Research completed: 2026-07-01*
*Researched by: 4-agent parallel research (STACK, FEATURES, ARCHITECTURE, PITFALLS)*
*Ready for roadmap: Conditional — Phase 0 (spike) must precede Phase 1 planning. Once Phase 0 confirms data sourcing and credential strategy, roadmap can confidently structure Phase 1 (design) and Phase 2 (implementation).*
