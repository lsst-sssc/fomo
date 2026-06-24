---
phase: 5
slug: multi-proposal-multi-facility-selection
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-19
updated: 2026-06-19
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django test runner (`django.test.TestCase`), not pytest — `pyproject.toml` `testpaths = ["tests", "src", "docs"]` excludes `solsys_code/` |
| **Config file** | none — Django test discovery via `./manage.py test` |
| **Quick run command** | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` |
| **Full suite command** | `./manage.py test solsys_code` |
| **Estimated runtime** | ~10 seconds |

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
| 05-01-01 | 01 | 1 | — | — | N/A | fixture | `_create_record(facility: str = 'LCO')` param added | ✅ | ✅ green |
| 05-01-03 | 01 | 1 | SELECT-02 | T-05-01 | `--proposal A,B,C` matches any of 3 codes, no substring leakage | unit | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_select_02_comma_list_matches_any_no_substring_leakage` | ✅ | ✅ green |
| 05-01-03 | 01 | 1 | SELECT-03 | T-05-01 | `--proposal ALL` (case-insensitive) syncs every LCO-family record | unit | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_select_03_all_token_case_insensitive_syncs_everything` | ✅ | ✅ green |
| 05-01-03 | 01 | 1 | SELECT-04 | — | Single run produces correct events for both `facility='LCO'` and `facility='SOAR'` records together | unit | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_select_04_single_run_covers_both_facilities` | ✅ | ✅ green |
| 05-01-03 | 01 | 1 | SELECT-05 | — | SOAR record dispatched via SOAR-credentialed instance, never reused `LCOFacility()` | unit (spy/patch) | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_select_05_soar_record_uses_soar_facility_instance` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*Verified 2026-06-19: all 4 SELECT tests + the `_create_record` fixture param confirmed present in `solsys_code/tests/test_sync_lco_observation_calendar.py` and re-run individually (`./manage.py test ...test_select_02...05`) — 4/4 pass.*

---

## Wave 0 Requirements

- [x] `solsys_code/tests/test_sync_lco_observation_calendar.py::_create_record` — add `facility: str = 'LCO'` parameter (Pitfall 4) — done in Task 1 (`adc5a61`)
- [x] New test cases for SELECT-02/03/04/05 listed above — done in Task 3 (`61a1c80`), 4/4 pass
- [x] `src/fomo/settings.py` — add `FACILITIES['SOAR']` entry (D-04), resolving Open Question 1 for `api_key`'s exact value/expression — done in Task 1 (`adc5a61`); mirrors `FACILITIES['LCO']` literally
- No new pytest fixtures/conftest needed — existing `setUpTestData()` class-level `user`/`target` fixtures are reused as-is

---

## Manual-Only Verifications

*None — all phase behaviors have automated verification via Django TestCase.*

---

## Validation Audit 2026-06-19

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

All 4 requirement tests (SELECT-02/03/04/05) and the Wave-0 fixture/settings prerequisites were present and green at audit time — no Nyquist auditor spawn needed.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 10s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** verified 2026-06-19
