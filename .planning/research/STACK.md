# Stack Research

**Domain:** ESO/VLT `ObservationRecord` calendar sync (extending FOMO's `sync_lco_observation_calendar` / `sync_gemini_observation_calendar` pattern to `tom_eso`)
**Researched:** 2026-07-01
**Confidence:** HIGH — every claim below is verified by reading the actual installed source (`tom_eso` 0.2.4, `tom_observations` (via `tomtoolkit`), and `p2api` 1.0.10) on disk at `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/`, not from PyPI README or memory.

## Headline Finding (read this before scoping the sync command)

**`tom_eso.eso.ESOFacility` (v0.2.4) does not implement the TOM Toolkit status/URL/data-product interface at all, and its `submit_observation()` never returns an observation ID — meaning the standard TOM UI submission flow cannot create an `ObservationRecord(facility='ESO')` row in the first place.** Concretely, in `tom_eso/eso.py`:

```python
def data_products(self, observation_id, product_id=None):
    raise NotImplementedError

def get_observation_status(self, observation_id):
    raise NotImplementedError

def get_observation_url(self, observation_id):
    raise NotImplementedError

def get_terminal_observing_states(self):
    return []

def submit_observation(self, observation_payload):
    self.submit_new_observation_block(observation_payload)
    created_observation_ids = []
    return created_observation_ids
```

`ObservationCreateView.form_valid()` (`tom_observations/views.py`) does:

```python
observation_ids = self.facility_instance.submit_observation(form.observation_payload())
for observation_id in observation_ids:
    record = ObservationRecord.objects.create(..., observation_id=observation_id)
```

Since `ESOFacility.submit_observation()` hardcodes `created_observation_ids = []` and never appends to it, this loop never executes for ESO — the standard TOM submission UI creates zero `ObservationRecord` rows for ESO. `ESOObservationForm.button_layout()` also deliberately removes the standard "Submit"/"Validate" buttons, confirming ESO observations are meant to be finished in the external ESO P2 Tool (an iframe), not through TOM's normal create-observation-record flow.

**Implication for the sync command:** before building `sync_eso_observation_calendar`, confirm with the operator how/whether `ObservationRecord(facility='ESO')` rows get created at all in this FOMO instance (e.g. a separate ingestion path, manual creation, or a planned addition). If none exists yet, the "sync" command's job may need to be paired with (or preceded by) a decision about where ESO `ObservationRecord`s come from — this is a scope question for the milestone, not something `tom_eso` answers.

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `tom-eso` | 0.2.4 (already installed & in `TOM_FACILITY_CLASSES`) | TOM Toolkit facility plugin exposing ESO Phase 1/2 access (`ESOFacility`, `ESOAPI`, `ESOProfile`) | Already the project's chosen integration point; no alternative ESO TOM plugin exists in the TOMToolkit ecosystem. But treat it as a thin, incomplete wrapper (see Gaps below), not a drop-in replacement for `LCOFacility`/`GEMFacility` |
| `p2api` | 1.0.10 (transitive dep of `tom-eso`, already installed) | Raw ESO Phase 2 REST client (`getOB`, `getOBExecutions`, `getItems`, `getRuns`, ...) | This is the *actual* source of any OB status/execution data. `tom_eso.eso_api.ESOAPI` only wraps a small slice of it (`observing_run_choices`, `folder_name_choices`, `folder_item_choices`, `folder_ob_choices`, `getOB`, `create_observation_block`) — nothing that surfaces execution/completion status through a documented method |
| `p1api` | (transitive dep, installed) | ESO Phase 1 proposal-time API client | Irrelevant to Phase 2 OB status/sync; not used by anything in scope here |
| `tom_observations.models.ObservationRecord` | existing (Django model, unchanged) | Source model for the new sync command, exactly as for LCO/Gemini | No ESO-specific fields exist or are needed on the model itself; everything ESO-specific must come from `record.parameters` and `record.facility == 'ESO'` (`ESOFacility.name = 'ESO'`, confirmed `eso.py:357`) |
| `solsys_code.calendar_utils.insert_or_create_calendar_event()` | existing (this repo) | Idempotent create-or-update helper shared by all three prior sync commands | Reuse verbatim — same no-churn contract (`created`/`updated`/`unchanged`) the LCO and Gemini commands already rely on; no ESO-specific reason to diverge |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `requests` | already a transitive dep (used inside `p2api`/`calendar_utils`) | HTTP calls, if the sync command needs to call `p2api` directly rather than through `tom_eso.eso_api.ESOAPI` | Only if the milestone decides to bypass `tom_eso`'s thin wrapper to reach `getOB()`/`getOBExecutions()` for OB status — mirrors the existing `_resolve_placement_block()` pattern in `calendar_utils.py` that already calls the LCO API directly for data `LCOFacility` doesn't expose |
| `tom_eso.eso_api.ESOAPI` | bundled with `tom-eso` 0.2.4 | Thin, credentialed passthrough to `p2api.ApiConnection` (`getOB(ob_id)` returns the raw OB dict including `obStatus`) | If the sync command needs per-OB status, this is the only *supported* entry point already wired to user credentials (`ESOProfile`) — call `ESOAPI(environment, username, password).getOB(ob_id)` and read `ob['obStatus']` yourself; there is no higher-level method that decodes it |

## Installation

No installation is needed — `tom-eso==0.2.4`, `p2api==1.0.10`, and `p1api` are already installed in the project venv and `tom_eso` is already in `INSTALLED_APPS` / `TOM_FACILITY_CLASSES` (`src/fomo/settings.py`). This milestone is pure application code (a new management command), not a new dependency.

## Answers to the Research Questions

### 1. Does `tom_eso` expose per-OB execution/completion status, or only submission-time metadata?

**Neither, cleanly — and effectively "not even submission-time metadata" given the Headline Finding above.** `ESOFacility.get_observation_status()` (the TOM-standard per-record status method every other facility implements, e.g. `LCOFacility`/`GEMFacility`) is a bare `raise NotImplementedError`. There is no method anywhere in `tom_eso` that calls `p2api`'s `getOBExecutions(obId, night)` (execution-event history) or decodes `obStatus`.

The *only* place raw OB status data is reachable is `ESOAPI.getOB(ob_id)` → `self.api2.getOB(ob_id)` → returns the OB dict as delivered by the real ESO Phase 2 API, which includes an `obStatus` field (confirmed via `p2api/p2api.py` docstrings at `getOB`/`saveOB`/`deleteOB`, e.g. "A CB does not have a target. The properties itemType, obId, obStatus, ipVersion, exposureTime and executionTime cannot be changed."). `tom_eso.eso_api.ESOAPI.folder_item_choices()` even has this in a comment (currently unused): `# folder_item_choices = [(item['obId'], f"{item['name']} : {item['itemType']} : {item['obStatus']}") ...]` — i.e. the author was aware `obStatus` exists per-item but never wired it into a facility-level status method.

The official ESO Phase 2 status codes (from ESO's public documentation, cross-checked against the `p2api` docstrings' status-gated preconditions) are single letters:

| Code | Meaning | Terminal? |
|------|---------|-----------|
| `P` | Partially defined (just created) | No |
| `D` | Defined (passed certification, ready for review) | No |
| `-` | Rejected (needs user attention) | No |
| `R` | Review (under revision by support astronomer) | No |
| `+` | Accepted (ready to be observed) | No |
| `C` | Completed (executed successfully, will not repeat) | **Yes** |
| `X` | Executed (successfully completed, can repeat — e.g. visitor mode) | **Yes** (per-execution) |
| `M` | Must repeat (executed outside constraints, will be requeued) | No (requeues) |
| `A` | Aborted during execution (will be requeued) | No (requeues) |
| `F` | Failed (absolute time window expired; read-only, irreversible) | **Yes** |
| `K` | Kancelled (support-astronomer set, irreversible) | **Yes** |
| `T` | Terminated (run terminated, irreversible) | **Yes** |

`tom_eso` does not surface, map, or use any of this. If the sync command needs OB status, it must fetch `obStatus` itself via `ESOAPI.getOB(ob_id)['obStatus']` and interpret the single-letter code using a hand-built mapping like the one above — there is no library-provided constant or helper to lean on (contrast with LCO, where `LCOFacility().get_failed_observing_states()`/`get_terminal_observing_states()` already return the vocabulary).

### 2. What does `ESOFacility` provide that's analogous to `LCOFacility.get_observation_url()`, `scheduled_start`/`scheduled_end` population, and status-checking (`update_all_observation_statuses`)?

Verified against `tom_observations/facility.py` (the base class) and `tom_eso/eso.py`:

| LCO/Gemini capability | ESO equivalent | Status |
|---|---|---|
| `LCOFacility.get_observation_url(observation_id)` | `ESOFacility.get_observation_url(observation_id)` | Overridden, but `raise NotImplementedError` — no ESO portal/P2-tool deep link comes back from this method. (`ESOFacility.get_p2_tool_url(observation_block_id=...)` exists and *does* build a real `https://www.eso.org/p2[demo]/home/ob/<obId>` URL, but it is a differently-named, differently-shaped method — not the `get_observation_url()` the base class/other sync commands call, and it needs a live, credentialed `ESOProfile` lookup, not just an ID) |
| `scheduled_start`/`scheduled_end` population via `update_observation_status()` | `BaseRoboticObservationFacility.update_observation_status()` (inherited, not overridden) calls `self.get_observation_status(observation_id)` and expects a dict with `state`/`scheduled_start`/`scheduled_end` keys | Inherited method exists on `ESOFacility`, but since `get_observation_status()` raises `NotImplementedError`, calling `update_observation_status()` on an ESO facility instance always raises. `scheduled_start`/`scheduled_end` can never be populated by any ESO-facility code path today |
| `LCOFacility.update_all_observation_statuses(target=None)` (from `BaseRoboticObservationFacility`) | Same inherited method | Present (inherited, not overridden), but per above it calls `update_observation_status()` per record, which will raise for every non-terminal ESO record and get caught into `failed_records` — i.e. it "runs" but produces zero real status updates, only a list of failures |
| `LCOFacility.get_failed_observing_states()` | — | **Does not exist anywhere on `ESOFacility` or its base class.** `get_failed_observing_states()` is defined only in `tom_observations/facilities/ocs.py` (`OCSFacility`, the shared base for LCO/SOAR) — `ESOFacility` inherits directly from `BaseRoboticObservationFacility`, which has no such method. Any `_FAILURE_PREFIX_BY_STATUS`-style dict for ESO must be hand-built from the `obStatus` table above, mirroring `sync_lco_observation_calendar.py`'s existing hand-typed-snapshot approach — there is no library method to call |
| `LCOFacility().get_observing_sites()` | `ESOFacility.get_observing_sites()` | Implemented, but hardcoded to two dicts (`PARANAL`, `LA_SILLA`) with a comment `"I don't see an API for this info, so it's hardcoded"` — do not use this as the site-coordinate source; per this project's existing convention (`Observatory` model, MPC-obscode lookup, used throughout `telescope_runs.py`/all three prior sync commands) and per PROJECT.md's Out-of-Scope note, ESO classical scheduling already goes through `Observatory`/`SITES`, not through `tom_eso` |
| `record.parameters['start']`/`['end']` (LCO queue-banner window) | `get_start_end_keywords()` (inherited default `'start'`, `'end'`, not overridden by `ESOFacility`) | The *keys* are inherited, but nothing in `ESOObservationForm`/`submit_new_observation_block()` ever populates `'start'`/`'end'` in the parameters dict — the ESO form only collects `p2_observing_run`, `p2_folder_name`, `observation_blocks`, `observation_block_name`. There is no window-banner data source analogous to LCO's `parameters['start']`/`['end']` or Gemini's `windowDate`/`windowTime`/`windowDuration` |

**Net: every capability the LCO/Gemini sync commands lean on is either unimplemented (`NotImplementedError`), silently absent (no failure-state vocabulary), or structurally different (site coords, URL builder) in `ESOFacility` 0.2.4.**

### 3. ESO-specific terminal/failure states analogous to LCO's `WINDOW_EXPIRED`/`CANCELED`/`FAILURE_LIMIT_REACHED`?

`ESOFacility.get_terminal_observing_states()` is overridden and unconditionally `return []` — i.e. as far as the TOM Toolkit interface is concerned, **ESO has zero terminal states**, which is itself informative: it confirms the library authors never wired OB status into the TOM abstraction at all (an honest "empty" implementation, not a placeholder bug to work around).

The *real* ESO Phase 2 terminal/failure vocabulary (from the official ESO Phase 2 status documentation, not from `tom_eso`) is the `C`/`X`/`F`/`K`/`T` subset of the `obStatus` table in Question 1 above:
- `C` (Completed) — success, will not repeat — the ESO analogue of LCO's clean/`COMPLETED` title
- `F` (Failed, absolute time window expired) — direct analogue of LCO's `WINDOW_EXPIRED`
- `K` (Kancelled) — direct analogue of LCO's `CANCELED`
- `T` (Terminated, run terminated) — no direct LCO analogue; closest to `FAILURE_LIMIT_REACHED`'s "give up" semantics
- `X` (Executed, can repeat — visitor mode) — no LCO analogue at all; this is an ESO-only "success but re-runnable" state that doesn't fit the LCO terminal/non-terminal binary cleanly
- `M` (Must repeat) / `A` (Aborted) — both requeue, so neither is terminal, unlike anything explicit in LCO's failure set

None of this is derivable from `tom_eso` itself — any `_FAILURE_PREFIX_BY_STATUS`-equivalent dict for ESO must be hand-typed against the official ESO documentation (as `sync_lco_observation_calendar.py` already does for LCO's 4 failure states), and the data to populate it (`obStatus`) can only be fetched via `ESOAPI.getOB(ob_id)`, not any status-checking method on `ESOFacility`.

### 4. Prior FOMO/solsys_code investigation of `tom_eso`?

None found beyond incidental references. Searched `docs/design/`, `solsys_code/`, and `.planning/`:
- `docs/design/telescope_runs_calendar.rst` mentions "NTT / EFOSC2 at ESO La Silla Observatory" only in the context of Stage 2 *classical* scheduling (`load_telescope_runs`), which is explicitly out of scope for `ObservationRecord`/queue sync per PROJECT.md.
- `.planning/codebase/INTEGRATIONS.md` lists `tom_eso.eso.ESOFacility` as "Configured in `TOM_FACILITY_CLASSES`" with no further detail (a codebase-mapping artifact, not a design decision).
- The only ESO-named quick task (`260613-f7d-modify-docs-notebooks-eso-how-to-downloa`) is about redirecting a *data-download* demo notebook (`ESO_How_to_download_data.ipynb`) to write FITS files into `data/` — unrelated to Phase 2 OB status or `ObservationRecord` sync; it doesn't touch `tom_eso` or `ESOFacility` at all.
- No prior `.planning/` phase, design doc, or decision log addresses ESO queue/OB sync, `ESOFacility`, or `p2api` capabilities. This milestone's research is the first investigation of this surface in the repo.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|--------------------------|
| Build `sync_eso_observation_calendar` as a **banner-only** sync (mirrors Gemini's stub-`get_observation_url()` precedent, which PROJECT.md already accepted as Out-of-Scope reasoning for Gemini being a full facility sync) using whatever fields *are* reliably present on `ObservationRecord.parameters` for ESO (created time, folder/OB identifiers) | Attempt full status/placement sync via direct `p2api` calls (`ESOAPI.getOB()` for `obStatus`, or raw `p2api.ApiConnection.getOBExecutions()` for execution history) | Only if the milestone explicitly wants OB-level completion tracking; requires bypassing `tom_eso`'s facility abstraction, hand-building a status-code table (Question 1/3 above), and handling per-record ESO credentials (`ESOProfile`) the same way `_resolve_placement_block()` handles LCO/SOAR credentials in `calendar_utils.py` |
| Use `Observatory`/`SITES` for La Silla/Paranal site data (existing project convention) | `ESOFacility.get_observing_sites()`'s hardcoded `PARANAL`/`LA_SILLA` dict | Never — this project already sources all site coordinates from the `Observatory` model by MPC obscode (documented convention in PROJECT.md and CLAUDE.md); `tom_eso`'s hardcoded dict has no MPC-obscode linkage and would introduce a second, inconsistent site-data source |
| Key idempotency (`CalendarEvent.url`) on something derived from the ESO OB/folder identifiers already in `record.parameters` | Key on `ESOFacility.get_observation_url()` | Not usable — that method raises `NotImplementedError`; there is no ESO analogue to `LCOFacility().get_observation_url()` to build a lookup key from |

## What NOT to Use / What NOT to Assume

| Don't assume | Why | Do instead |
|---------------|-----|------------|
| `ESOFacility.get_observation_status(observation_id)` returns a dict like LCO/Gemini | It unconditionally `raise NotImplementedError` in installed 0.2.4 | Do not call it from the sync command; if OB status is needed, call `ESOAPI(...).getOB(ob_id)` directly and read `ob['obStatus']` yourself |
| `ESOFacility.get_observation_url(observation_id)` returns a usable portal URL, the way `LCOFacility`'s does (used as the LCO sync command's idempotency key) | Also unconditionally `raise NotImplementedError` | Derive the `CalendarEvent.url` idempotency key from something else already present in `record.parameters` (e.g. folder/OB identifiers), or from `ESOFacility.get_p2_tool_url(observation_block_id=...)` if the OB id is separately available and a live `ESOProfile` is in scope |
| `record.scheduled_start`/`scheduled_end` will ever be populated for ESO records by any existing code path | `update_observation_status()` (which sets those fields) is only ever reachable via `get_observation_status()`, which raises | Treat every ESO `ObservationRecord` as banner-only (no placed-block stage) unless/until a future `tom_eso` version or custom code populates these fields directly |
| `get_terminal_observing_states()`/`get_failed_observing_states()` give you an ESO failure vocabulary the way they do for LCO | `get_terminal_observing_states()` returns `[]`; `get_failed_observing_states()` doesn't exist on `ESOFacility` at all (it's OCS-only) | Hand-build an ESO status-code table from the official ESO Phase 2 documentation (see Question 1/3 tables above), the same "hand-typed snapshot, not auto-derived" approach `sync_lco_observation_calendar.py` already uses and documents in its own comments |
| `ObservationRecord.objects.filter(facility='ESO')` will return real, populated data under normal FOMO usage | `ESOFacility.submit_observation()` always returns an empty ID list, so the standard TOM submission UI never creates one of these rows (Headline Finding) | Confirm with the operator how ESO `ObservationRecord`s actually get created in this instance before writing sync logic against assumed real data — this may need to be resolved as a phase-0/scope question, not discovered mid-implementation the way v1.3's `SITE_TELESCOPE_MAP` gap was (see PROJECT.md's v1.2→v1.3 correctness-bug entry for the cautionary precedent) |
| ESO's site coordinates should come from `ESOFacility.get_observing_sites()` | Hardcoded, not MPC-obscode-linked, explicitly marked `# TODO: get data for all the ESO sites for production` in the library source | Use the existing `Observatory` model (by MPC obscode), consistent with every other part of this codebase |

## Version Compatibility

| Package | Version (installed) | Notes |
|---------|----------------------|-------|
| `tom-eso` | 0.2.4 | Depends on `p1api`/`p2api` (ESO's own client libraries) and `tom_common.session_utils.get_encrypted_field`/`tom_common.models.EncryptableModelMixin` for credential storage (`ESOProfile`) — same encrypted-credential pattern as other per-user facility profiles in this TOM Toolkit version |
| `p2api` | 1.0.10 | The version actually installed; confirmed via `getOB`/`getOBExecutions`/`getOBSchedulingInfo` method presence. No compatibility issue found with `tomtoolkit`/Django versions already pinned in this project |
| `tomtoolkit` | project-pinned (per CLAUDE.md, `tomtoolkit>=2.31.4`) | `BaseRoboticObservationFacility`/`BaseObservationFacility` (in `tom_observations/facility.py`) define the abstract contract `ESOFacility` only partially implements — this is a `tom_eso` completeness gap, not a version-mismatch issue |

## Sources

- `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_eso/eso.py` (installed v0.2.4) — primary source for all `ESOFacility`/`ESOSettings`/`ESOObservationForm` claims
- `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_eso/eso_api.py` — `ESOAPI` wrapper methods and their `p2api`/`p1api` passthroughs
- `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_eso/models.py` — `ESOProfile`/`ESOP2Environment` credential model
- `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_eso/views.py` — confirms no dedicated ESO `ObservationCreateView` override; standard TOM flow is used
- `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_observations/facility.py` — `BaseObservationFacility`/`BaseRoboticObservationFacility` abstract contract, `update_observation_status()`/`update_all_observation_statuses()` implementations
- `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_observations/views.py` (`ObservationCreateView.form_valid`) — confirms the empty-`observation_ids`-list → zero-`ObservationRecord`-created behavior
- `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_observations/facilities/ocs.py` — confirms `get_failed_observing_states()` is OCS(LCO/SOAR)-only, absent from `BaseRoboticObservationFacility`
- `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/p2api/p2api.py` (installed v1.0.10) — `getOB`/`saveOB`/`deleteOB`/`getOBExecutions` docstrings, confirms `obStatus` field and status-gated preconditions
- https://www.eso.org/sci/observing/phase2/p2intro/phase-2-status.html — official ESO Phase 2 OB status code definitions (used to build the status tables in Questions 1/3), HIGH confidence (primary/official source)
- `/home/tlister/git/fomo_devel/solsys_code/calendar_utils.py`, `solsys_code/management/commands/sync_lco_observation_calendar.py` — this repo's existing pattern being extended
- `/home/tlister/git/fomo_devel/.planning/codebase/INTEGRATIONS.md`, `.planning/PROJECT.md` — confirmed no prior ESO-sync-specific investigation or decision exists

---
*Stack research for: ESO/VLT ObservationRecord calendar sync (v1.7 milestone)*
*Researched: 2026-07-01*
