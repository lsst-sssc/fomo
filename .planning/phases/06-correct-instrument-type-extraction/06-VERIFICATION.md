---
phase: 06-correct-instrument-type-extraction
verified: 2026-06-21T01:17:26Z
status: passed
score: 7/7 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 6: Correct Instrument-Type Extraction Verification Report

**Phase Goal:** The command always identifies the scientifically meaningful instrument configuration for a record, regardless of which LCO-family facility submitted it
**Verified:** 2026-06-21T01:17:26Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

Merged from ROADMAP.md Success Criteria (roadmap contract) and PLAN frontmatter `must_haves.truths` (plan-specific detail) — no contraction of scope; both sources agree on the same four behaviors.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Single-populated-config record (today's flat LCO shape) still extracts the same instrument value as before — all 19 existing tests pass unmodified | VERIFIED | Re-ran the 19 pre-existing tests at the GREEN commit (`de063db`); confirmed 0 failures. Verified the source: `_extract_instrument` (sync_lco_observation_calendar.py:137-163) falls through `_find_science_config`→`_find_exposure_signal_config`→flat `parameters.get('instrument_type')` when no `c_N_*` keys exist at all (the legacy fixture shape), preserving original behavior. |
| 2 | SOAR multi-config record (SPECTRUM + ARC + LAMP_FLAT) extracts the SPECTRUM config's instrument_type, never the ARC or LAMP_FLAT config | VERIFIED | `test_extract_02_soar_multi_config_picks_spectrum_not_calibration` (test file lines 481-515) passes; independently re-ran `_extract_instrument` against the test's exact parameter dict and confirmed it returns `'SOAR_GHTS_REDCAM'` (the SPECTRUM config), not `'SOAR_GHTS_REDCAM_ARC'`/`'SOAR_GHTS_REDCAM_LAMPFLAT'`. `_SCIENCE_CONFIGURATION_TYPES` (line 46) excludes `ARC`/`LAMP_FLAT`. |
| 3 | LCO MUSCAT record with only per-channel exposure keys (no flat `c_N_exposure_time`) extracts its instrument_type without raising or returning empty | VERIFIED | `test_extract_02_muscat_per_channel_exposure_extracts_instrument` (lines 517-559) passes for both the all-4-channel and the single-channel (D-04 leniency) case. `_has_muscat_exposure_signal` (lines 89-99) uses `any(...)` over the 4 suffixes, matching D-04. |
| 4 | A fully-malformed record (no recognized configuration_type and no exposure signal anywhere) is skipped, logged with its observation_id, and counted in a dedicated counter distinct from 'skipped' — visible in the run summary | VERIFIED | `test_d06_no_extractable_config_logged_and_counted_separately` (lines 561-591) passes. Independently re-ran `_extract_instrument` against the exact malformed-record parameter dict and confirmed it returns `None`. Verified the `InstrumentExtractionError` catch block (handle(), lines 367-372) routes to `counters[...]['extraction_failed']`, never `'skipped'`. Counter wired into all 4 required locations (see Key Link Verification). |

**Score:** 4/4 observable truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/management/commands/sync_lco_observation_calendar.py` | Instrument extraction helper(s) implementing D-01..D-06 + dedicated extraction-failure counter wiring; contains `_extract_instrument` | VERIFIED | `_extract_instrument` defined at line 137, calling `_find_science_config` (D-01) then `_find_exposure_signal_config` (D-02/D-04) then flat-key fallback. Module-level `_SCIENCE_CONFIGURATION_TYPES` (line 46) and `_MUSCAT_CHANNEL_SUFFIXES` (line 52) present. `InstrumentExtractionError` exception class (line 218) implements the D-06 sentinel-vs-exception contract. |
| `solsys_code/tests/test_sync_lco_observation_calendar.py` | `extra_params` fixture extension + SOAR multi-config, MUSCAT per-channel, and D-06 malformed-record tests; contains `extra_params` | VERIFIED | `_parameters()` signature includes `extra_params: dict | None = None` (line 22), merged via `params.update(extra_params or {})` (line 48) immediately before return — additive, does not disturb the 5 existing named params. Three new test methods present and passing (lines 481, 517, 561). |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `_build_event_fields` | `_extract_instrument` | `instrument = _extract_instrument(record.parameters)` replaces the flat `record.parameters['instrument_type']` read on the old line 143 | WIRED | Confirmed at line 244. `grep -c "parameters\['instrument_type'\]"` on the source returns 0 — old flat-key read fully removed. |
| `handle()` | the new dedicated extraction-failure counter (`extraction_failed`) | per-record sentinel/exception check increments the new counter instead of 'skipped', and the summary join displays it | WIRED | All 4 required locations confirmed by direct grep: (1) eager dict-literal init for both LCO/SOAR (lines 340-341), (2) `setdefault` defensive default (line 360), (3) the `except InstrumentExtractionError` block increments `counters[record.facility]['extraction_failed']` (line 371), distinct from the separate `except (KeyError, ValueError)` block that increments `'skipped'` (lines 373-376), (4) summary `' | '.join(...)` f-string includes `extraction_failed: {counts["extraction_failed"]}` (line 399). |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|---------------------|--------|
| `_extract_instrument(parameters)` | `instrument` (config index `n` → `c_{n}_instrument_type`) | Directly invoked against the exact parameter dicts used in the SOAR, MUSCAT, and malformed tests (independent of the test harness/Django test runner) | Yes — `_extract_instrument` returned `'SOAR_GHTS_REDCAM'` for the SOAR case (not the ARC/LAMP_FLAT sentinel values), `'2M0-SCICAM-MUSCAT'` for the MUSCAT case, and `None` for the malformed case, matching the asserted `event.instrument` values exactly | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 19 pre-existing tests pass unmodified at GREEN commit | `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar` | 22 tests, 0 failures/errors | PASS |
| RED state existed before Task 2 implementation (TDD discipline) | Checked out `aaf4d1f` test file + pre-fix (`f14c6b7`) source, re-ran test file, then restored to `de063db` | 2 failures + 1 error confined to the 3 new tests only; all pre-existing tests passed at that point too | PASS |
| `_extract_instrument` returns the SPECTRUM config, not ARC/LAMP_FLAT | Direct Python invocation against the SOAR test's exact parameter dict | Returned `'SOAR_GHTS_REDCAM'` | PASS |
| `_extract_instrument` returns `None` for the fully-malformed shape (D-06) | Direct Python invocation against the malformed test's exact parameter dict | Returned `None` | PASS |
| Counter wiring complete in all 4 locations | `grep -n 'extraction_failed' sync_lco_observation_calendar.py` | 4 functional occurrences (dict init x2 facilities counted as 1 literal pattern, setdefault, increment, summary join) plus 3 comment references | PASS |
| Full app test suite green | `python manage.py test solsys_code` | 117 tests, 0 failures | PASS |
| Lint/format clean | `ruff check` + `ruff format --check` on both modified files | All checks passed; 2 files already formatted | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EXTRACT-01 | 06-01-PLAN.md | Instrument type is extracted by scanning `c_1_instrument_type`..`c_5_instrument_type` for the configuration with a populated exposure time, replacing the v1.2 flat-key assumption | SATISFIED | `_find_exposure_signal_config` (D-02) scans `c_1..c_5` for the first config with a truthy `c_N_exposure_time` or MUSCAT per-channel signal; `_find_science_config` (D-01) takes priority when a recognized `configuration_type` exists. Old flat-key-only read removed from `_build_event_fields`. |
| EXTRACT-02 | 06-01-PLAN.md | Extraction is verified against SOAR's multi-configuration shape and LCO MUSCAT's per-channel exposure-key shape, not just the single-populated-config shape, so a calibration/non-science config is never mistaken for the meaningful one | SATISFIED | `test_extract_02_soar_multi_config_picks_spectrum_not_calibration` and `test_extract_02_muscat_per_channel_exposure_extracts_instrument` both pass; independently confirmed via direct helper invocation (Data-Flow Trace above). `_SCIENCE_CONFIGURATION_TYPES` whitelist explicitly excludes ARC/LAMP_FLAT. |

No orphaned requirements — REQUIREMENTS.md maps only EXTRACT-01 and EXTRACT-02 to Phase 6, both declared in the plan's `requirements:` frontmatter field.

### Anti-Patterns Found

None. Scanned both modified files for `TBD`/`FIXME`/`XXX`, `TODO`/`HACK`/`PLACEHOLDER`, "not yet implemented"/"placeholder"/"coming soon" phrasing, and empty-return stubs — zero matches.

### Human Verification Required

None. All four observable truths are verifiable via existing automated tests plus independent direct-invocation spot-checks against the same parameter shapes; no visual, real-time, or external-service behavior is involved in this phase's scope.

### Gaps Summary

No gaps. All ROADMAP success criteria and PLAN must-haves are verified against the actual codebase (not just SUMMARY.md claims):

- The flat-key read (`record.parameters['instrument_type']` at the old line 143) is fully removed and replaced with `_extract_instrument(record.parameters)`.
- The SOAR SPECTRUM-vs-ARC/LAMP_FLAT distinction is implemented via an explicit whitelist (`_SCIENCE_CONFIGURATION_TYPES`) and independently confirmed to select the science config in a SPECTRUM+ARC+LAMP_FLAT scenario.
- The MUSCAT per-channel shape is detected leniently (`any()` over 4 channel suffixes) and independently confirmed to extract correctly with both 4 and 1 populated channels.
- A dedicated `extraction_failed` counter, distinct from `skipped`, is wired into all 4 required locations and independently confirmed to fire (not `skipped`) for a genuinely malformed record.
- TDD discipline confirmed via direct commit checkout: the RED commit (`aaf4d1f`) produced exactly 3 failing/erroring tests against the pre-fix source, and the GREEN commit (`5e1489c`) made all 22 pass.
- Full app suite (117 tests) green; `ruff check`/`ruff format --check` clean for both modified files.
- One plan deviation (documented in 06-01-SUMMARY.md): an additional flat-`instrument_type` fallback tier beyond D-01/D-02 was added to keep the 19 legacy-shape regression tests passing. This is consistent with — not contradictory to — Success Criterion 1 ("unchanged from today's correct cases") and was independently verified above, not just taken on the SUMMARY's word.

---

_Verified: 2026-06-21T01:17:26Z_
_Verifier: Claude (gsd-verifier)_
