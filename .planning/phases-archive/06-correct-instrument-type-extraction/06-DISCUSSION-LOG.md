# Phase 6: Correct Instrument-Type Extraction - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-20
**Phase:** 6-Correct Instrument-Type Extraction
**Areas discussed:** Science vs calibration rule, MUSCAT shape & label, Extraction-failure fallback

---

## Todo cross-reference

A `todo.match-phase` query surfaced one weak match: the pending
status-aware `CalendarEvent` coloring todo (score 0.3, area=ui).

| Option | Description | Selected |
|--------|-------------|----------|
| Leave deferred | Not related to instrument-type extraction — keep as a separate future todo, unchanged | ✓ |
| Fold into Phase 6 | Pull it into this phase's scope anyway | |

**User's choice:** Leave deferred.

---

## Science vs calibration rule

Initial framing: when a SOAR record has multiple configs (spectrum + arc +
lamp-flat) all with populated exposure times, what marks the spectrum one
as "the" science config?

| Option | Description | Selected |
|--------|-------------|----------|
| Config 'type' field | Each c_N config has its own type key; pick SPECTRUM/EXPOSE, never ARC/LAMP_FLAT/BIAS/DARK | ✓ |
| instrument_type code pattern | Disambiguate via the instrument code itself | |
| Not sure / research it | Let the researcher confirm the real field before planning locks it in | |

**User's choice:** Config 'type' field.

**Notes:** User then shared a real LCO Observation Portal API JSON response
(`observe.lco.global.json`, local file, a 6-config `SOAR_GHTS_REDCAM_IMAGER`
mosaic request, all configs `"type":"EXPOSE"`) — this is the nested OCS API
shape, not the flat `ObservationRecord.parameters` shape directly. Claude
cross-checked the installed `tom_observations` library source and confirmed
the flat key is `c_N_configuration_type` (`ocs.py:1029,1213`), and that
`soar.py:103,118` hardcodes SOAR's 3-config spectrum/arc/lamp-flat form with
exactly those `configuration_type` values.

User clarified that Blanco's `configuration_type` vocabulary (`EXPOSE`/
`STANDARD`, confirmed `blanco.py:177`) should be included in the recognized
science set now (vocabulary only), while NRES's spectroscopic-sequence
types (`NRES_SPECTRUM`/`REPEAT_NRES_SPECTRUM`, confirmed `lco.py:757-760`)
should never be recognized. User confirmed Blanco **facility** scope itself
(adding `BLANCOFacility`/`'BLANCO'` to `TOM_FACILITY_CLASSES`/the queryset)
is explicitly deferred to a later phase — only the vocabulary value is
adopted now.

User also pointed out that facility classes' `get_form(observation_type)`
(e.g. `MUSCAT_IMAGING`, `IMAGING`, `SPECTRA`) route to per-shape form
classes with their own `get_instruments()`/`configuration_type_choices()`.
Claude confirmed `parameters['observation_type']` (`facility.py:87`) records
which form built the request, and that the real `observation_forms` dicts
(`lco.py:1123-1129`, `soar.py:254-259`) confirm the three shapes named in
EXTRACT-02 are exhaustive — but the chosen algorithm doesn't need to branch
on it; `configuration_type` plus the exposure-signal fallback already
disambiguates shape per-config.

Final algorithm locked in: scan `c_1..c_5` for the first config whose
`c_N_configuration_type` is in `{EXPOSE, REPEAT_EXPOSE, SPECTRUM,
REPEAT_SPECTRUM, STANDARD}`; never `ARC`/`LAMP_FLAT`/`NRES_*`. If none match,
fall back to the first config with a populated exposure signal (flat
`c_N_exposure_time`, or any MUSCAT channel key). Return that config's
`c_N_instrument_type` as the instrument value.

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, lock it in | configuration_type whitelist primary, exposure-populated fallback secondary, c_N_instrument_type as returned value | ✓ |
| Adjust something | Describe what should change | |

**User's choice:** Yes, lock it in.

---

## MUSCAT shape & label

Initial framing: where do real MUSCAT per-channel (g/r/i/z) exposure values
live in parameters, and what should the resulting `instrument` text say?

**Notes:** User clarified MUSCAT is a simultaneous g/r/i/z imager, so
exposure times must always be specified for all 4 filters in a real
submission. Claude confirmed via `lco.py:585-678`
(`LCOMuscatImagingObservationForm`/`_build_instrument_config`) that the real
keys are `c_N_ic_M_exposure_time_g/_r/_i/_z` (no flat `exposure_time`), and
that the real form's own validation requires all 4 truthy to build the
instrument config at all.

| Option | Description | Selected |
|--------|-------------|----------|
| Any of g/r/i/z populated; label unchanged | More lenient than the form's all-4 requirement; instrument stays the raw c_N_instrument_type value | ✓ (folded into the combined "Algorithm" lock-in above) |
| Require all 4 populated, matching form validation exactly | Mirror the real form's strict behavior | |

**User's choice:** Any of g/r/i/z populated (the more lenient detection), confirmed as part of the combined algorithm lock-in. Instrument label stays unchanged (raw `c_N_instrument_type`) — channels don't change the instrument code.

---

## Extraction-failure fallback

| Option | Description | Selected |
|--------|-------------|----------|
| Skip + log + own counter | Matches the established per-record catch-log-continue convention, reported as a distinct count, not merged into skipped_count | ✓ |
| Skip + log, merge into existing skipped_count | Same skip/log/continue behavior, counted together with today's skipped_count | |

**User's choice:** Skip + log + own counter.

---

## Claude's Discretion

- Exact log message wording for the extraction-failure skip case.
- Whether the science-type whitelist / fallback logic is implemented as one helper function or several.
- Exact variable/counter naming for the new dedicated extraction-failure counter, as long as it's visible and distinct from `skipped_count` in the run summary.

## Deferred Ideas

- Adding `BLANCOFacility`/`'BLANCO'` to `TOM_FACILITY_CLASSES` and this command's `facility__in` queryset — explicitly deferred to a later phase by the user. Only the `STANDARD` configuration-type vocabulary value is adopted now.
- Status-aware `CalendarEvent` coloring todo (`2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md`) — reviewed via todo cross-reference, confirmed unrelated, left pending.
