---
phase: 7
slug: live-telescope-label-resolution-with-fallback-failure-report
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-21
updated: 2026-06-24
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django's `django.test.TestCase` test runner, via `./manage.py test solsys_code` (this is a DB-dependent test file, not pytest-collected per `pyproject.toml`'s `testpaths`) |
| **Config file** | none — Django test discovery via `manage.py test`; no separate pytest config applies to this file |
| **Quick run command** | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` |
| **Full suite command** | `./manage.py test solsys_code` |
| **Estimated runtime** | ~30-60 seconds (in-process Django test runner; all HTTP calls mocked, no real network I/O, no SPICE/ASSIST kernel loading — this module doesn't import `ephem_utils`) |

---

## Sampling Rate

- **After every task commit:** Run `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar`
- **After every plan wave:** Run `./manage.py test solsys_code` (full Django suite) + `python -m pytest` (pytest suite, unaffected by this phase but cheap regression safety) + `ruff check .` + `ruff format --check .`
- **Before `/gsd-verify-work`:** Full suite must be green, plus the paired demo notebook (`docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`) regenerated via `jupyter nbconvert --to notebook --execute --inplace` and committed with output (CLAUDE.md demo-notebook-companion convention).
- **Max feedback latency:** ~60 seconds

---

## Per-Task Verification Map

> Task ID / Plan / Wave columns are assigned by the planner once PLAN.md exists for this phase; rows below are keyed by requirement ID from RESEARCH.md's "Phase Requirements → Test Map" until then.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 07-01 T2/T3 | 07-01 | 1 | TELESCOPE-01 | T-07-01/02 | Verified dict covers all 7 real sites (tlv dropped, operator-confirmed) with correct (site, class) → label mapping | unit | `test_telescope_01_verified_dict_covers_all_sites`, `test_telescope_01_aperture_class_from_telescope_code` | ✅ | ✅ green |
| 07-02 T1/T2 | 07-02 | 2 | TELESCOPE-02 | — | Placed record + successful mocked API response resolves to verified label | integration (mocked HTTP) | `test_telescope_02_placed_record_resolves_via_api` | ✅ | ✅ green |
| 07-02 T1/T2 | 07-02 | 2 | TELESCOPE-03 | T-07-03 | Placed record + mocked API failure/timeout/unmapped code falls back to coarse label, record still synced; malformed block missing site/telescope also falls back, not skipped | integration (mocked HTTP) | `test_telescope_03_api_failure_falls_back_not_skipped`, `test_telescope_03_block_missing_site_or_telescope_falls_back_not_skipped` (added by quick task 260623-ocs closing T-07-03) | ✅ | ✅ green |
| 07-02 T1/T2 | 07-02 | 2 | TELESCOPE-04 | T-07-04/05 | Fallback event has coarse telescope token + description failure note + `[UNVERIFIED]` prefix; re-run with success flips label visibly | unit + integration | `test_telescope_04_fallback_label_visibly_distinguishable` | ✅ | ✅ green |
| 07-02 T1/T2 | 07-02 | 2 | SYNC-06 | — | `telescope_api_failed` counter increments only for placed+failed records, reported in summary, distinct from `skipped` | integration | `test_sync_06_fallback_counter_distinct_from_skipped` | ✅ | ✅ green |
| 07-02 T1/T2 | 07-02 | 2 | SYNC-07 | T-07-06 | A per-record API failure does not abort the run; subsequent records still process | integration | `test_sync_07_api_failure_does_not_abort_run` | ✅ | ✅ green |
| 07-01 T2/T3 | 07-01 | 1 | SYNC-08 | T-07-02 | Mocked slow/failing response — assert no second call attempted (single-attempt, no retry) | unit (mocked HTTP, call-count assertion) | `test_sync_08_single_attempt_no_retry` | ✅ | ✅ green |
| 07-01/07-02 | both | 1+2 | SYNC-09 | T-07-01/04 | Mocked failure whose exception embeds fake response content/API key — assert logged output is the fixed generic message, never the raw content/key | unit (mocked HTTP, log-content assertion) | `test_sync_09_no_credential_or_body_leak_in_logs`, `test_sync_09_log_line_is_fixed_generic_message` | ✅ | ✅ green |
| (D-01 supplemental) | 07-02 | 2 | TELESCOPE-02/03 (D-01) | — | Banner-stage (unscheduled) record gets coarse fallback with NO API call and no `[UNVERIFIED]` prefix | integration | `test_d01_banner_record_no_api_call_no_unverified_prefix` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Verified 2026-06-24:** `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar` — 34/34 tests pass. All 8 originally-planned test methods exist (3 added beyond the original Wave 0 plan: `test_telescope_01_aperture_class_from_telescope_code`, `test_d01_banner_record_no_api_call_no_unverified_prefix`, and `test_telescope_03_block_missing_site_or_telescope_falls_back_not_skipped` — the last closing the T-07-03 security gap found during `/gsd-secure-phase 7` and fixed via quick task `260623-ocs`). No gaps found — Nyquist auditor was not needed.

---

## Wave 0 Requirements

- [x] `solsys_code/tests/test_sync_lco_observation_calendar.py` — all 8 planned test methods added (plus 3 supplemental), 19 pre-existing tests still pass unmodified
- [x] Shared mock-response-builder helper (`_observations_block_response()`) for "successful block" shapes; "failed/timeout" and "unmapped code"/"missing key" shapes built inline per-test with `MagicMock`/`side_effect`
- [x] Timeout-specific test double: `requests.exceptions.Timeout` via `side_effect`, driving `test_sync_08_single_attempt_no_retry`'s `assert_called_once()`
- [x] Mock patch target resolved: `solsys_code.management.commands.sync_lco_observation_calendar.make_request` (import-site patching, matching the file's existing convention)

---

## Manual-Only Verifications

None remaining. The original entry (per-site aperture-class assignment for `tlv`/`elp`/`lsc`/`cpt`/`tfn`) was resolved during Plan 07-01's Task 1 human-verify checkpoint: `tlv` was dropped entirely (confirmed absent from installed `LCOSettings`/`SOARSettings`, not merely unverified), and `elp`/`lsc`/`cpt`/`tfn` were confirmed by the operator (Tim Lister, LCO staff) as standard 1m-network sites — see `07-01-SUMMARY.md` "Deviations from Plan" for the full decision record. No `[ASSUMED]` entries remain in `SITE_TELESCOPE_MAP`.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-06-24 — 34/34 tests green, no gaps found.

---

## Validation Audit 2026-06-24

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 (none needed) |
| Escalated | 0 |

All 8 originally-planned Nyquist test requirements are COVERED by passing tests, plus 3 supplemental tests added during execution (D-01 banner-record behavior, aperture-class parsing unit test, and the T-07-03 security-gap regression test from quick task `260623-ocs`). Full suite (`./manage.py test solsys_code.tests.test_sync_lco_observation_calendar`) verified green: 34/34 tests pass. No auditor dispatch was needed — gap analysis found zero MISSING/PARTIAL requirements.
