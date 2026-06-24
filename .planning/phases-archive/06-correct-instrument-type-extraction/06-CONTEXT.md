# Phase 6: Correct Instrument-Type Extraction - Context

**Gathered:** 2026-06-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix `sync_lco_observation_calendar.py`'s instrument-type extraction so it
always identifies the scientifically meaningful instrument configuration for
a record, regardless of which LCO-family facility (LCO or SOAR) submitted
it. Today's code reads a flat `record.parameters['instrument_type']` key
that doesn't exist on real data; this phase replaces that single line with
logic that scans the real `c_1..c_5`-prefixed multi-configuration shape,
correctly distinguishing a science config from SOAR's arc/lamp-flat
calibration configs and correctly detecting LCO MUSCAT's per-channel
exposure shape. Telescope/site label derivation (`_derive_telescope`,
`SITE_TELESCOPE_MAP`) and facility/proposal selection are untouched — those
are Phase 7 and Phase 5 (already shipped) respectively.

</domain>

<decisions>
## Implementation Decisions

### Science vs. calibration disambiguation
- **D-01:** Use the per-config `c_N_configuration_type` field (confirmed
  flat key name in installed `tom_observations/facilities/ocs.py:1029,1213`)
  as the primary signal. Scan `c_1..c_5` in order and pick the **first**
  config whose `c_N_configuration_type` is in the recognized science set:
  `EXPOSE`, `REPEAT_EXPOSE`, `SPECTRUM`, `REPEAT_SPECTRUM`, `STANDARD`.
  Never recognize `ARC`/`LAMP_FLAT` (SOAR's calibration configs) or
  `NRES_SPECTRUM`/`REPEAT_NRES_SPECTRUM` (NRES is never in scope).
  `STANDARD` is included now for vocabulary forward-compatibility with a
  future Blanco phase — see "Claude's Discretion"/deferred note below;
  Blanco **facility** scope itself stays deferred.
- **D-02:** If no config has a recognized `configuration_type` (key missing
  entirely — e.g. a legacy record — or all present types are
  unrecognized/calibration), fall back to the original EXTRACT-01 heuristic:
  the first config with a populated exposure signal — a truthy flat
  `c_N_exposure_time`, OR (for MUSCAT) any of
  `c_N_ic_M_exposure_time_g`/`_r`/`_i`/`_z` being truthy.
- **D-03:** The returned `instrument` value is always that selected config's
  `c_N_instrument_type` raw value, unchanged in format from today's
  single-config case — no per-channel annotation is added for MUSCAT.

### MUSCAT per-channel shape
- **D-04:** MUSCAT records have no flat `c_N_exposure_time`; only
  `c_N_ic_M_exposure_time_g/_r/_i/_z` (confirmed in installed
  `tom_observations/facilities/lco.py:585-596`,
  `LCOMuscatImagingObservationForm`). Detect population by **any** of the 4
  channel keys being truthy — more lenient than the real submission form's
  own all-4-required validation (`lco.py:651-654`), so sparse/legacy data
  still extracts correctly.
- **D-05:** `parameters['observation_type']` (hidden field on every OCS
  submission form, `tom_observations/facility.py:87`) records which form
  built the request (`MUSCAT_IMAGING`/`SPECTRA`/`IMAGING`/
  `PHOTOMETRIC_SEQUENCE`/`SPECTROSCOPIC_SEQUENCE` for LCO;
  `IMAGING`/`Goodman_BLUE_Spectra`/`Goodman_RED_Spectra`/`SPECTRA_Advanced`
  for SOAR) and confirms the three real shapes named in EXTRACT-02 are
  exhaustive for the forms actually used. The extraction algorithm does
  **not** need to branch on it, though — `configuration_type` plus the
  exposure-signal fallback (D-01/D-02/D-04) disambiguates shape per-config
  without it. Noted as confirmed code context / a free correctness
  cross-check, not a required dispatch mechanism.

### Extraction-failure fallback
- **D-06:** If neither the `configuration_type` whitelist (D-01) nor the
  exposure-signal fallback (D-02) finds any candidate config at all (fully
  malformed/empty record), skip the record, log it (with record-identifying
  info), and count it in its **own dedicated counter** — distinct from the
  existing `skipped_count` — so it's visible separately in the run summary.
  Matches the established per-record catch-log-continue convention (Phase 3
  D-02, Phase 5 D-07) but kept distinguishable rather than merged.

### Claude's Discretion
- Exact log message wording for the D-06 skip case.
- Whether the science-type whitelist / fallback logic is implemented as one
  helper function or several — any approach matching D-01..D-06 is
  acceptable.
- Exact variable/counter naming for the new dedicated extraction-failure
  counter (D-06), as long as it's visible and distinct from `skipped_count`
  in the run summary.

### Explicitly out of scope (raised and deferred during discussion)
- Adding `BLANCOFacility`/`'BLANCO'` to `TOM_FACILITY_CLASSES` or this
  command's `facility__in` queryset — confirmed by the user as a later
  phase. Only Blanco's `configuration_type` vocabulary value (`STANDARD`,
  D-01) is adopted now, purely for forward compatibility; no Blanco records
  can appear in this DB under current facility registration regardless.
- Telescope/site label derivation (`_derive_telescope`,
  `SITE_TELESCOPE_MAP`) — unchanged; Phase 7 scope.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design & requirements
- `.planning/REQUIREMENTS.md` — EXTRACT-01, EXTRACT-02 (this phase's locked
  acceptance criteria)
- `.planning/milestones/v1.3-ROADMAP.md` §"Phase 6: Correct Instrument-Type
  Extraction" — success criteria 1-3 and dependency on Phase 5
- `.planning/PROJECT.md` — "Current Milestone: v1.3 Full LCO Facility Sync"
  section, including the v1.2 real-data bug note about the real `c_1..c_5`
  multi-configuration shape that drives this phase
- `.planning/STATE.md` — Phase 5 "Key Technical Notes" / blockers section

### Existing code (Phase 4/5)
- `solsys_code/management/commands/sync_lco_observation_calendar.py` —
  lines 18-22 (`SITE_TELESCOPE_MAP`, untouched by this phase); lines 142-144
  (`telescope = _derive_telescope(record.parameters['site'])` stays as-is;
  `instrument = record.parameters['instrument_type']` is the exact line this
  phase replaces with the new extraction logic)
- `solsys_code/tests/test_sync_lco_observation_calendar.py` — `_parameters()`
  fixture helper (lines 16-43), currently flat-`instrument_type`-only; must
  be extended with `c_N_configuration_type`/`c_N_instrument_type`/MUSCAT
  channel-key variants for the new test cases. Per CLAUDE.md, any new
  `Target` fixtures must use `NonSiderealTargetFactory`, never
  `SiderealTargetFactory`.
- `.planning/phases/05-multi-proposal-multi-facility-selection/05-CONTEXT.md`
  — Phase 5's decisions this phase builds on (per-record catch-log-continue
  convention D-07, per-facility dispatch dict; no scope overlap with
  instrument extraction)
- `.planning/phases/04-lco-queue-sync-command/04-CONTEXT.md` — D-02's
  original (now-superseded) telescope/instrument derivation note; the
  no-churn upsert and per-record error-handling conventions this phase
  continues

### Third-party library (not in this repo) — confirmed against installed `tomtoolkit`
- `tom_observations/facilities/ocs.py:1025-1030` — `_add_config_fields`
  defines `c_{N}_instrument_type` and `c_{N}_configuration_type` as the flat
  per-config field names serialized into `ObservationRecord.parameters`
- `tom_observations/facilities/ocs.py:1213` — `_build_configuration` maps
  `configuration['type']` directly from
  `cleaned_data[f'c_{id}_configuration_type']`, confirming the nested OCS
  API `type` field and the flat DB key hold the same value
- `tom_observations/facilities/soar.py:103,118` —
  `SOARSpectroscopyObservationForm.configuration_type_choices()` returns
  exactly `SPECTRUM`/`ARC`/`LAMP_FLAT`; `c_3_configuration_type` defaults to
  `LAMP_FLAT` — confirms the real 3-config spectrum+arc+lamp-flat shape
- `tom_observations/facilities/lco.py:740-743,757-760,998` — full
  `configuration_type` vocabulary across LCO forms: `EXPOSE`/
  `REPEAT_EXPOSE` (imaging), `SPECTRUM`/`REPEAT_SPECTRUM` (spectroscopy,
  becomes `NRES_SPECTRUM`/`REPEAT_NRES_SPECTRUM` only for the
  NRES-specific spectroscopic-sequence form — never in scope here)
- `tom_observations/facilities/blanco.py:177` —
  `BLANCOImagingObservationForm.configuration_type_choices()` returns
  `EXPOSE`/`STANDARD`; confirms `STANDARD` as the vocabulary value to
  recognize now even though Blanco facility scope itself is deferred
- `tom_observations/facilities/lco.py:569-678` —
  `LCOMuscatImagingObservationForm`/`_build_instrument_config`: confirms
  `c_{N}_ic_{M}_exposure_time_g/_r/_i/_z` as the real MUSCAT per-channel keys
  (no flat `exposure_time`), and that the real form requires all 4 truthy
  (this phase is intentionally more lenient, D-04)
- `tom_observations/facility.py:87` —
  `BaseRoboticObservationForm.observation_type` hidden field, confirms
  `parameters['observation_type']` records which form/shape built the
  record
- `tom_observations/facilities/lco.py:1123-1129`
  (`LCOFacility.observation_forms`), `soar.py:254-259`
  (`SOARFacility.observation_forms`) — the real `observation_type` →
  form-class dicts confirming the shape vocabulary above

### User-provided reference
- User-shared real LCO Observation Portal API response (local file,
  `observe.lco.global.json`, outside this repo) — a 6-configuration
  `SOAR_GHTS_REDCAM_IMAGER` mosaic request, all configs `"type": "EXPOSE"`.
  Used during discussion to confirm the nested OCS API
  `configurations[].type`/`instrument_type`/`instrument_configs[].exposure_time`
  shape that the flat `c_N_*` parameters keys are built from.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `sync_lco_observation_calendar.py`'s existing per-record error handling
  (catch, log, increment counter, continue) — extend with the new D-06
  dedicated counter, same pattern.
- `test_sync_lco_observation_calendar.py`'s `_parameters()`/`_create_record()`
  fixture helpers — extend (don't replace) for the new
  `c_N_configuration_type`/MUSCAT-channel/SOAR-multi-config test cases.

### Established Patterns
- No-churn idempotency, per-record catch-log-continue, DB-dependent tests in
  `solsys_code/tests/` run via `./manage.py test solsys_code` — all
  unchanged, all apply to this phase's new extraction code path.

### Integration Points
- The single line `instrument = record.parameters['instrument_type']`
  (`sync_lco_observation_calendar.py:143`) is replaced by a new extraction
  helper implementing D-01..D-06; everything else in the record-processing
  loop (telescope derivation, proposal, title/description building) is
  untouched.

</code_context>

<specifics>
## Specific Ideas

User provided a real LCO Observation Portal API JSON response
(`observe.lco.global.json`) and used it to confirm field shapes against the
installed library source during discussion (see canonical_refs). User also
confirmed Blanco's `configuration_type` vocabulary value (`STANDARD`)
should be recognized now even though Blanco facility scope is explicitly
deferred to a later phase.

</specifics>

<deferred>
## Deferred Ideas

- Adding `BLANCOFacility`/`'BLANCO'` to `TOM_FACILITY_CLASSES` and this
  command's `facility__in` queryset — explicitly deferred to a later phase
  by the user during discussion. Only the `STANDARD` configuration-type
  vocabulary value is adopted now (D-01), not facility scope.

### Reviewed Todos (not folded)
- `2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md`
  — reviewed (`todo.match-phase` score 0.3, area=ui); confirmed unrelated to
  instrument-type extraction during discussion; left pending/deferred
  unchanged.

</deferred>

---

*Phase: 6-Correct Instrument-Type Extraction*
*Context gathered: 2026-06-20*
