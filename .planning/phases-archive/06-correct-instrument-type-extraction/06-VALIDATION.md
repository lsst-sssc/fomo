---
phase: 6
slug: correct-instrument-type-extraction
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-20
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django test runner (`django.test.TestCase`), not pytest — `pyproject.toml` `testpaths = ["tests", "src", "docs"]` excludes `solsys_code/` |
| **Config file** | none — Django test discovery via `./manage.py test` |
| **Quick run command** | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` |
| **Full suite command** | `./manage.py test solsys_code` |
| **Estimated runtime** | <1s quick (19 existing tests measured at 0.15s during research); ~10 seconds full suite |

---

## Sampling Rate

- **After every task commit:** Run `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar`
- **After every plan wave:** Run `./manage.py test solsys_code` + `ruff check .` + `ruff format --check .`
- **Before `/gsd-verify-work`:** Full Django suite green, plus `python -m pytest` (separate pytest suite, unaffected by this phase but must remain green)
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 06-0x-0x | TBD | TBD | EXTRACT-01 | V5 / — | Single populated config (today's real LCO shape) extracts unchanged | unit | existing `test_sync_05_telescope_instrument_proposal_populated` must keep passing | ✅ (existing) | ⬜ pending |
| 06-0x-0x | TBD | TBD | EXTRACT-02 | V5 / T-6-01 | SOAR multi-config (spectrum/arc/lamp-flat): extracted instrument is the science config, never `ARC`/`LAMP_FLAT` | unit | new `test_extract_02_soar_multi_config_picks_spectrum_not_calibration` | ❌ W0 | ⬜ pending |
| 06-0x-0x | TBD | TBD | EXTRACT-02 | V5 / T-6-01 | MUSCAT per-channel: extraction reflects populated channel(s), no crash/empty result | unit | new `test_extract_02_muscat_per_channel_exposure_extracts_instrument` | ❌ W0 | ⬜ pending |
| 06-0x-0x | TBD | TBD | D-02 (CONTEXT.md) | — | Legacy/flat shape with no `configuration_type` key falls back to exposure-signal heuristic | unit | covered by existing flat-shape tests (regression); explicit fallback-path test optional | ✅ (existing, partial) | ⬜ pending |
| 06-0x-0x | TBD | TBD | D-06 (CONTEXT.md) | V5 / T-6-01 | Fully malformed/empty record (no config found by either signal) is skipped, logged, counted in dedicated counter | unit | new `test_d06_no_extractable_config_logged_and_counted_separately` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*Task IDs/plan/wave are placeholders — the planner fills these in once plans are split into waves.*

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_sync_lco_observation_calendar.py::_parameters()` — add an additive `extra_params: dict | None = None` parameter (the existing five fixed kwargs cannot express `c_N_configuration_type`/MUSCAT-channel keys today; `_create_record`'s `**parameter_overrides` already forwards through, so `TypeError` is the current failure mode) — required before any new test in this phase can be written.
- [ ] New test: SOAR multi-config (`c_1_configuration_type='SPECTRUM'`, `c_2_configuration_type='ARC'`, `c_3_configuration_type='LAMP_FLAT'`, all with populated `c_N_instrument_type`/exposure fields) — asserts extracted instrument matches `c_1`'s instrument type, never `c_2`/`c_3`'s.
- [ ] New test: LCO MUSCAT per-channel (`c_1_ic_1_exposure_time_g/_r/_i/_z` populated, no flat `c_1_exposure_time`) — asserts extraction succeeds and reflects the populated channel config; include one case proving the "any of 4 truthy" leniency (D-04).
- [ ] New test: D-06 fully malformed record (no `configuration_type` anywhere, no exposure signal anywhere) — asserts the record is skipped, logged (observation_id visible in stderr), and counted in the new dedicated counter (distinct from the existing per-facility `'skipped'` key), visible in the stdout summary line.
- No new test framework/config needed — existing Django `TestCase` infrastructure in this file fully covers the new requirements once `_parameters()` is extended.

---

## Manual-Only Verifications

*None — all phase behaviors have automated verification via Django TestCase.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
