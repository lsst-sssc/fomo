---
status: complete
phase: 06-correct-instrument-type-extraction
source: [.planning/phases/06-correct-instrument-type-extraction/06-01-SUMMARY.md]
started: 2026-06-21T18:28:23Z
updated: 2026-06-21T21:06:24Z
---

## Current Test

[testing complete]

## Tests

### 1. SOAR multi-config record extracts the science instrument, not calibration
expected: |
  Run `sync_lco_observation_calendar` (or open
  `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` and re-run the
  "Phase 6" SOAR cell) against a SOAR record with SPECTRUM + ARC + LAMP_FLAT configs.
  The resulting CalendarEvent's `instrument` field is the SPECTRUM config's value
  (`SOAR_GHTS_REDCAM` in the demo fixture) — never the ARC or LAMP_FLAT calibration
  instrument value.
result: pass

### 2. LCO MUSCAT per-channel record extracts correctly
expected: |
  Run the command against an LCO MUSCAT record that has only per-channel exposure keys
  (`c_1_ic_1_exposure_time_{g,r,i,z}`, no flat `c_1_exposure_time`). The resulting
  CalendarEvent's `instrument` field is `2M0-SCICAM-MUSCAT` — no error, no empty value.
result: pass

### 3. Fully malformed record is skipped and counted separately
expected: |
  Run the command against a record with no recognized `configuration_type` and no
  exposure signal anywhere. That record is skipped (no CalendarEvent created), its
  `observation_id` is logged, and the run's printed summary line shows it under a
  distinct `extraction_failed` count — not folded into the existing `skipped` count.
result: pass

### 4. Legacy single-config records still sync exactly as before
expected: |
  Run the command against an old-style record with a flat `instrument_type` key (today's
  production shape, no `c_N_*` keys at all). It still produces the same `instrument`
  value it would have before this phase — no regression for existing real data.
result: pass

## Summary

total: 4
passed: 4
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
