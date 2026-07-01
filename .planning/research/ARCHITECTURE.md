# Architecture Research

**Domain:** ESO/VLT `ObservationRecord` calendar sync — integration into an existing Django/TOM Toolkit calendar-sync architecture (FOMO v1.7)
**Researched:** 2026-07-01
**Confidence:** HIGH for the "what exists today" findings (verified directly against the installed `tom_eso==0.2.4` source at `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_eso/` and against this repo's live dev DB via `./manage.py shell`) / MEDIUM-LOW for "what to build," because the verified findings materially constrain what's buildable this milestone (see Critical Finding below).

This file supersedes the previous contents (dated 2026-06-24, about v1.4's calendar visual-treatment work) — that topic is now shipped and documented elsewhere; this is a full rewrite for the v1.7 ESO milestone.

## Critical Finding (reshapes the whole question)

**There is currently no path by which an ESO `ObservationRecord` gets created in this codebase, and the installed `tom_eso` plugin does not implement observation status/URL lookups at all.** This was not assumed — it was verified by reading `tom_eso/eso.py` and by querying the dev DB directly:

- `ESOFacility.submit_observation()` (`tom_eso/eso.py:659-672`) is hard-coded to `return created_observation_ids = []` unconditionally, even after successfully creating a P2 Observation Block. TOM Toolkit's `ObservationCreateView.form_valid()` (`tom_observations/views.py`) only creates an `ObservationRecord` by iterating the `observation_id`s returned from `submit_observation()` — so for ESO, **that loop is always empty**. No `ObservationRecord(facility='ESO')` row can be produced by the normal TOM submission flow, in this dev environment or in a production deployment running this same plugin version.
- `ESOFacility.get_observation_status(observation_id)` → `raise NotImplementedError` (`eso.py:601-602`).
- `ESOFacility.get_observation_url(observation_id)` → `raise NotImplementedError` (`eso.py:604-605`). LCO's sync command keys `CalendarEvent.url` on exactly this method (`facility.get_observation_url(...)`); it does not exist for ESO.
- `ESOFacility.get_terminal_observing_states()` → `return []` (`eso.py:626-627`). There is no terminal-status vocabulary to drive a TERM-01-style title prefix.
- Confirmed live: `ObservationRecord.objects.filter(facility='ESO').count()` → `0`; in fact `ObservationRecord.objects.count()` → `0` for **every** facility in this dev DB right now (the dev DB has been reset/reseeded since the v1.3 LCO records referenced in `.planning/PROJECT.md` were inspected — do not assume those rows still exist).
- `ESOFacility.get_observing_sites()` (`eso.py:607-624`) hardcodes exactly two sites — `PARANAL` (lat -24.62733, lon -70.40417, elev 2635.43 m) and `LA_SILLA` (lat -29.25667, lon -70.73194, elev 2400.0 m) — as a plain dict literal, **not** looked up via MPC obscode or the `Observatory` model. There is no telescope-level (UT1-4/AT) breakdown anywhere in the installed plugin.
- Existing `Observatory` records in this DB: `268` (Magellan Clay), `269` (Magellan Baade), `809` (ESO La Silla — already used by Stage 1/2's `NTT` classical-schedule key), `E10` (Siding Spring). **No Cerro Paranal record exists** (MPC obscode for Paranal is `309`; it is not in the table).
- `ESOAPI`/`eso_api.py` wraps ESO's real Phase-2 API (`p2api`) and does carry an OB-level status concept in commented-out code (`item['obStatus']`), and `folder_ob_choices()`/`folder_item_choices()` do walk `api2.getItems()` results — but nothing in the installed plugin surfaces this into `ObservationRecord`, `get_observation_status()`, or any URL. Reaching it would mean calling `ESOAPI`/`p2api` directly, which requires a valid `ESOProfile` (per-user P2 credentials) or `FACILITIES['ESO']` settings defaults — **neither is configured in this repo's `settings.py`** (`FACILITIES` currently only has `LCO`, `SOAR`, `GEM`; no `ESO` key).

This is a genuine **MAYBE, not a YES**, on the milestone's premise of "a `sync_eso_observation_calendar` management command following the LCO/Gemini pattern." The LCO/Gemini pattern assumes `ObservationRecord` rows already exist with a real `facility.get_observation_url()`/`get_observation_status()`/terminal-state vocabulary to key off. None of that exists for ESO in the installed plugin version. Question 4 (build order) is answered accordingly: **this must start as a research/spike phase**, not an implementation phase.

## Standard Architecture (of the existing 3-command pattern, for reference)

```
┌───────────────────────────────────────────────────────────────────────┐
│  management commands (one per facility)                               │
│  sync_lco_observation_calendar / sync_gemini_observation_calendar /   │
│  sync_eso_observation_calendar (NEW)                                  │
│  - query ObservationRecord.objects.filter(facility=...)               │
│  - per record: build a `fields` dict (title/desc/times/telescope/...) │
│  - per-facility counters (created/updated/unchanged/skipped/...)      │
├───────────────────────────────────────────────────────────────────────┤
│  solsys_code/calendar_utils.py (shared, facility-agnostic)             │
│  - insert_or_create_calendar_event(lookup, fields) -- no-churn         │
│    create-or-update, keyed on an arbitrary `lookup` dict               │
│  - SITE_TELESCOPE_MAP / _extract_instrument / _coarse_telescope_label  │
│    (LCO/SOAR-specific vocabulary -- NOT reusable for ESO as-is)        │
├───────────────────────────────────────────────────────────────────────┤
│  tom_calendar.models.CalendarEvent (third-party, unmodified)           │
│  solsys_code/models.py: CalendarEventTelescopeLabel (sidecar,          │
│    OneToOneField PK, currently LCO/SOAR-only)                          │
└───────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Reused as-is for ESO? |
|-----------|-----------------|------------------------|
| `insert_or_create_calendar_event(lookup, fields)` (`calendar_utils.py:296-332`) | Facility-agnostic no-churn create-or-update on `CalendarEvent`, keyed on caller-supplied `lookup` dict | **Yes, unchanged.** It has no facility knowledge — Gemini already proves this generalizes (`lookup={'url': ...}` there too, different derivation). |
| `SITE_TELESCOPE_MAP`, `_derive_telescope`, `_resolve_placement_block`, `_coarse_telescope_label`, `_extract_instrument` | LCO/SOAR-specific: site-code+aperture-class table, per-record LCO Observation Portal API call, LCO/SOAR `instrument_type` string parsing | **No.** These are all shaped around LCO's `c_N_*` multi-configuration parameters and LCO Observation Portal REST responses. ESO OBs (if ever reachable) come from `p2api`'s JSON shape, not LCO's — none of this vocabulary transfers. |
| `CalendarEventTelescopeLabel` sidecar (`solsys_code/models.py`) | Verified-vs-fallback telescope label provenance | **Conceptually reusable, mechanically not yet wired.** The model itself is facility-agnostic (`OneToOneField` to any `CalendarEvent`), but ESO has no live per-record API resolution today to be "verified" against — see below. |
| Per-command `handle()` (`sync_lco_observation_calendar.py`) | Query `ObservationRecord`, build fields, call `insert_or_create_calendar_event`, track counters | **Pattern reusable; data source is not.** ESO has no `ObservationRecord` rows to query under the installed plugin. |

## New vs. Reused Components (answering Question 1)

**Reusable as-is (genuinely facility-agnostic):**
- `insert_or_create_calendar_event()` — no changes needed. Its `lookup`/`fields` contract already generalizes across LCO's `{'url': ...}` key and would work identically for ESO's key (whatever that ends up being — see Question 3).
- The counters/summary-line pattern, and the "per-record try/except → dedicated counter, never abort the run" discipline from `sync_lco_observation_calendar.handle()`.
- `CalendarEventTelescopeLabel` as a *model* — no schema change needed to attach it to ESO-sourced events, if ESO ever gets a live/verified vs. fallback distinction worth recording.

**New components needed:**
- `solsys_code/management/commands/sync_eso_observation_calendar.py` — new `BaseCommand`, but its data source is fundamentally different from LCO/Gemini (see Question 2/3). It cannot be "LCO/Gemini pattern, ESO field names" — the input side (where records come from) is a different shape of problem, not just different field names.
- An ESO-specific helper section in `calendar_utils.py` (or a new `_eso_helpers` module if the ESO-specific logic is large — see Structure Rationale below) analogous to `SITE_TELESCOPE_MAP`/`_extract_instrument`, but built from whatever the spike (build-order step 1) discovers about real OB/P2 JSON shape — this cannot be designed from documentation alone because the installed plugin doesn't expose it.
- Almost certainly: a `FACILITIES['ESO']` entry in `settings.py` (currently absent) and, if going the direct-P2-API route, either an `ESOProfile` fixture/service-account credential or a documented manual precondition for the command to run at all.
- Possibly: a `Command`-level Observatory record for Cerro Paranal (obscode `309`), if the command is meant to resolve site coordinates the same way Stage 1/2 do (`get_site()` by MPC obscode) rather than hardcoding lat/lon like `ESOFacility.get_observing_sites()` does. This repo already has a "coordinates come from `Observatory`, not hardcoded constants" convention (Stage 1 constraint) — recommend following it here too rather than copying `ESOFacility`'s hardcoded dict.
- Almost certainly **not** a new field on `CalendarEvent` itself (see Question 3) — the existing fields (`title`, `description`, `start_time`, `end_time`, `telescope`, `instrument`, `proposal`, `url`) already cover what ESO would need to display, once a value for each is decided.

## Where ESO-specific data comes from (answering Question 2)

**Site:** Effectively fixed — VLT is at Cerro Paranal, one site, unlike LCO's multi-site network. But there is **no existing source of truth for it in this codebase**: `ESOFacility.get_observing_sites()` hardcodes it inline (not obscode-based); the `Observatory` table has no Paranal row. Recommendation: create a Paranal `Observatory` record (obscode `309`) and resolve it the same way Stage 1's `get_site()` does — reuse the existing convention rather than importing `ESOFacility.get_observing_sites()`'s hardcoded dict, since the latter conflicts with this repo's "Site coordinates come from `Observatory` model records" constraint (`CLAUDE.md`, `.planning/PROJECT.md`).

**Telescope (UT1-4 / auxiliary telescopes):** **Not exposed anywhere in the installed `tom_eso` plugin.** Unlike LCO where the Observation Portal API returns a `telescope` code per placed block (`_resolve_placement_block`), nothing in `eso.py`/`eso_api.py` surfaces which UT or AT an OB is scheduled on — `get_observing_sites()` stops at the site level, and OB scheduling to a specific unit telescope happens inside ESO's own short-term scheduling system, which this plugin doesn't query. This is a genuine unknown requiring the Question-4 spike: does the real ESO P2 API (`p2api.ApiConnection`) expose a telescope/instrument assignment per OB anywhere in `getOB()`'s response, or only after the night's execution (in a different ESO system entirely, e.g. the Short-Term Schedule or Night Log, which this plugin has no client for at all)?

**Instrument:** Partially reachable. `ESOAPI.observing_run_choices()` (`eso_api.py:57-77`) parses `run['instrument']` off `api2.getRuns()` — so instrument-per-observing-run is at least visible through the existing `p2api` wrapper, unlike telescope. Per-OB instrument would need `getOB()`'s response inspected directly (not currently done anywhere in the plugin).

**Bottom line for Question 2:** this is neither "simpler, single fixed site" (site alone is simple; telescope is unresolved) nor "already in `ObservationRecord.parameters`" (there is no `ObservationRecord` at all) nor "requires an API call like LCO's" in the same shape — it requires a wholly different API client (`p2api`/`ESOAPI`, credentialed per-user or via service-account defaults) queried in a fundamentally different way, because the object model on ESO's side (P2 Observation Blocks in folders under observing runs) doesn't map 1:1 onto TOM's `ObservationRecord` the way LCO's request/observation model does.

## Data flow changes (answering Question 3)

`insert_or_create_calendar_event()` and its 7-field no-churn comparison **do not need new fields** — they're already facility-agnostic, and Gemini already proved the pattern generalizes to a facility with a different key derivation (`GEM:{prog}/{observation_id}` vs. LCO's portal URL) and different terminal-state vocabulary (ready/on-hold vs. WINDOW_EXPIRED/CANCELED/...). Whatever ESO's per-event data turns out to be, it plugs into the same `fields` dict shape (`title`, `description`, `start_time`, `end_time`, `telescope`, `instrument`, `proposal`) and the same `lookup` dict contract.

What **does** need to change is upstream of that helper — the part LCO/Gemini both take for granted (a queryable, populated `ObservationRecord` table) does not exist for ESO. Two structurally different options, to be decided by the Question-4 spike, not assumed now:

1. **Bridge option:** teach the command (or a separate small sync step) to call the ESO P2 API directly and *create* `ObservationRecord` rows itself (working around `submit_observation()`'s empty-list bug/limitation), then apply the LCO/Gemini pattern on top of those newly-created rows. This keeps the downstream shape identical to LCO/Gemini but adds a new responsibility (record creation from an external system) that neither existing command has — LCO/Gemini both assume `ObservationRecord` creation already happened elsewhere (the TOM submission flow).
2. **Bypass option:** have `sync_eso_observation_calendar` read directly from the ESO P2 API (via `ESOAPI`/`p2api`, walking `getRuns()` → `getItems()` → `getOB()`) and build `CalendarEvent`s straight from OB/folder-item data, keyed on `p2_environment` + `obId` (or similar) as the `lookup` dict, never touching `ObservationRecord` for ESO at all. This is a bigger philosophical break from the LCO/Gemini precedent (those two are explicitly "sync `ObservationRecord`s to the calendar"; this would be "sync ESO P2 OBs to the calendar" with no `ObservationRecord` involved) but matches what the installed plugin actually supports today.

Either way, no new `CalendarEvent` fields and no new sidecar-model fields are needed — `CalendarEventTelescopeLabel.is_verified` already models exactly the "resolved live vs. fell back" boolean this integration would want, if the per-OB telescope-assignment lookup ever gets built; until then, the sidecar simply isn't written for ESO events (same "no row = verified by documented default" rule the template already uses for `load_telescope_runs` events).

## Suggested build order (answering Question 4)

**Phase 0 — Spike/research (do this before writing any command code):**
1. Confirm whether this milestone actually has (or can get) valid ESO P2 credentials (`ESOProfile` or `FACILITIES['ESO']` defaults) in a reachable environment (production credentials, ESO demo/sandbox environment, or a recorded fixture from a prior real session) — without this, nothing past "count is currently 0" can be verified live, mirroring exactly how v1.3 was driven by inspecting two real LCO `ObservationRecord` rows before writing extraction logic.
2. With those credentials (or a captured fixture), call `ESOAPI.observing_run_choices()` → `folder_name_choices()` → `folder_ob_choices()` → `getOB(ob_id)` for at least one real observing run, and record the **actual JSON shape** of a P2 OB: does it carry a schedulable date/night, a telescope/instrument assignment, an execution/observed status distinct from `obStatus` at the P2-authoring level (P, S, D states in ESO's phase-2 status vocabulary) that would map to something like LCO's placed-vs-banner distinction?
3. Decide, based on step 2's real findings — not assumption — whether "synced" for ESO can mean anything beyond "an OB exists in P2 with a target/window" (i.e., whether true execution/night-of status is reachable at all through this plugin, or only through some other ESO system this plugin doesn't touch). This directly determines whether Stage 4's ESO scope is a full LCO-style placed/banner/terminal-status sync, or a much narrower "OB exists and is scheduled for a run" banner-only sync.
4. Only after 1-3: decide Bridge vs. Bypass (Question 3) and write the actual `SPEC.md`/`PLAN.md` for the implementation phase(s).

**Phase 1 — Implementation (scope depends entirely on Phase 0's findings):**
- Create the Cerro Paranal `Observatory` record (obscode `309`) if site resolution is to follow the `Observatory`-model convention.
- Add `FACILITIES['ESO']` to `settings.py` if the command needs its own service-account-style credentials (mirroring how `GEM`'s per-site credential dict was added in v1.5, not LCO/SOAR's single flat dict).
- Build the new `sync_eso_observation_calendar.py` command and its `calendar_utils.py` (or dedicated `_eso_helpers`) additions, following whichever of the two Question-3 options Phase 0 justified.
- Reuse `insert_or_create_calendar_event()` unchanged.
- Only wire `CalendarEventTelescopeLabel` for ESO events if Phase 0 found a genuine live-vs-fallback telescope resolution to record — do not force-fit it if there's nothing to verify against.
- Paired demo notebook (`docs/notebooks/pre_executed/sync_eso_observation_calendar_demo.ipynb`) per this repo's standing convention — scope it into `files_modified` from the start of the plan, per the CLAUDE.md note about this gap being hit twice before (Phase 5's `260619-f7u`, Phase 6's `260620-v9x`).

**Why research-first here is not optional (unlike, say, Gemini's phase in v1.5):** Gemini's `ObservationRecord`s already existed and were queryable when that milestone started — the open questions there were about window-derivation rules, not about whether the data existed at all. Here, the verified absence of any working `submit_observation()`→`ObservationRecord` path, plus two `NotImplementedError` stubs on the exact two methods LCO's command depends on, means the "confirm real ESO `ObservationRecord` shape" step from the milestone context cannot be satisfied by inspecting the dev DB (it's empty and will stay empty under this plugin version) — it has to be satisfied by exercising the real ESO P2 API directly, which is a materially bigger and riskier first step than any prior phase in this project took.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| ESO Phase 2 API (`p2api`, wrapped by `tom_eso.eso_api.ESOAPI`) | Direct API client call, credentialed per-user (`ESOProfile`) or via `FACILITIES['ESO']` settings defaults | No timeout/retry discipline exists in `ESOAPI` today, unlike LCO's `_resolve_placement_block` (`_API_TIMEOUT_SECONDS = 10`, single attempt) — any new ESO sync code should add the same explicit-timeout, no-retry, never-log-response-body discipline (SYNC-08/SYNC-09 precedent) since `p2api.p2api.P2Error` messages can carry response content, same risk class as LCO's `ImproperCredentialsException`/`forms.ValidationError`. |
| `Observatory` model (this repo) | ORM lookup by MPC obscode, mirroring `telescope_runs.get_site()` | Needs a new Paranal (`309`) record; La Silla (`809`) already exists and is already used by Stage 1/2's `NTT` classical-schedule path — do not conflate ESO *queue* sync (this milestone) with ESO/NTT *classical* scheduling (`load_telescope_runs`, already out of scope per `.planning/PROJECT.md`). |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `sync_eso_observation_calendar.py` ↔ `calendar_utils.insert_or_create_calendar_event()` | Direct function call, `(lookup, fields)` dicts | Unchanged contract — no modification needed to this function for ESO. |
| `sync_eso_observation_calendar.py` ↔ ESO P2 API or (Bridge option) `ObservationRecord` | Either a direct `ESOAPI` call per run/OB, or an ORM query, depending on Phase 0's outcome | This is the boundary that doesn't exist yet in a working form anywhere in the codebase — it's the actual unknown this milestone must resolve, not a known integration to wire up. |
| `sync_eso_observation_calendar.py` ↔ `CalendarEventTelescopeLabel` | `update_or_create`, same call shape as LCO's | Only wire this up if Phase 0 finds a genuine verified-vs-fallback telescope resolution for ESO; otherwise leave ESO events with no sidecar row (falls back to "verified" by the template's documented default, same as `load_telescope_runs` events). |

## Anti-Patterns to avoid in this integration

### Anti-Pattern 1: Assuming ESO's `ObservationRecord` shape mirrors LCO's

**What people would do:** copy `sync_lco_observation_calendar.py`, swap `facility='LCO'` for `facility='ESO'`, and assume `record.parameters` has some ESO-flavored equivalent of `c_N_instrument_type`/`proposal`/`start`/`end`.
**Why it's wrong:** verified directly against the installed plugin — no code path in this codebase or in `tom_eso` ever populates an ESO `ObservationRecord`. There is nothing to copy the pattern onto.
**Do this instead:** run the Phase-0 spike against the real ESO P2 API first; let its actual JSON shape (not LCO's) drive the field-extraction design, the same way Phase 6 (v1.3) let real LCO `c_N_*` shapes replace the v1.2 flat-key assumption.

### Anti-Pattern 2: Reusing `_coarse_telescope_label`/`SITE_TELESCOPE_MAP` for ESO by adding a third facility branch

**What people would do:** extend `_coarse_telescope_label(instrument_type, facility_name)` with an `if facility_name.upper() == 'ESO': ...` branch, parallel to the existing SOAR special-case.
**Why it's wrong:** that function's whole design is keyed on LCO's aperture-class-prefixed `instrument_type` strings (`'1M0-SCICAM-SINISTRO'`) and SOAR's single-site/single-class shortcut. ESO instrument codes (UVES/X-shooter/FORS2/etc.) carry no aperture-class prefix and VLT is not a single-aperture-class site (four UTs are the same aperture class, but the plugin doesn't expose which UT anyway) — forcing ESO through this function would produce meaningless labels, not a genuine reuse.
**Do this instead:** if ESO needs a coarse/fallback label at all, give it its own small ESO-specific helper once Phase 0 clarifies what data is actually available, rather than overloading a function whose docstring and tests are explicitly scoped to LCO/SOAR vocabulary.

## Sources

- `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_eso/eso.py` (installed `tom-eso==0.2.4`) — read directly, HIGH confidence (primary source, this exact installed version).
- `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_eso/eso_api.py` — read directly, HIGH confidence.
- `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_observations/views.py` (`ObservationCreateView.form_valid`) — read directly via `./manage.py shell`, HIGH confidence.
- Live dev DB queries via `./manage.py shell` (`ObservationRecord.objects.count()`, `Observatory.objects.all()`) — HIGH confidence, but a point-in-time snapshot of this specific dev DB (2026-07-01); do not assume it matches production.
- `solsys_code/calendar_utils.py`, `solsys_code/management/commands/sync_lco_observation_calendar.py`, `solsys_code/models.py`, `solsys_code/telescope_runs.py` — this repo's existing, shipped implementation (v1.0-v1.6), HIGH confidence.
- `.planning/PROJECT.md` — project history/decision log, HIGH confidence as a record of this project's own past decisions (note: its "real LCO records pk=1/pk=2" reference no longer matches the current, reset dev DB — flagged above).

---
*Architecture research for: ESO/VLT ObservationRecord calendar sync integration (FOMO v1.7)*
*Researched: 2026-07-01*
