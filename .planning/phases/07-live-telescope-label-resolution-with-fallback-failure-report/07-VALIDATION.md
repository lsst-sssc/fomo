---
phase: 7
slug: live-telescope-label-resolution-with-fallback-failure-report
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-21
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
| TBD | TBD | TBD | TELESCOPE-01 | — | Verified dict covers all 8 sites with correct (site, class) → label mapping | unit | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_telescope_01_verified_dict_covers_all_sites` (new) | ❌ Wave 0 | ⬜ pending |
| TBD | TBD | TBD | TELESCOPE-02 | — | Placed record + successful mocked API response resolves to verified label | integration (mocked HTTP) | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_telescope_02_placed_record_resolves_via_api` (new) | ❌ Wave 0 | ⬜ pending |
| TBD | TBD | TBD | TELESCOPE-03 | — | Placed record + mocked API failure/timeout/unmapped code falls back to coarse label, record still synced | integration (mocked HTTP) | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_telescope_03_api_failure_falls_back_not_skipped` (new) | ❌ Wave 0 | ⬜ pending |
| TBD | TBD | TBD | TELESCOPE-04 | — | Fallback event has coarse telescope token + description failure note + `[UNVERIFIED]` prefix; re-run with success flips label visibly | unit + integration | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_telescope_04_fallback_label_visibly_distinguishable` (new) | ❌ Wave 0 | ⬜ pending |
| TBD | TBD | TBD | SYNC-06 | — | `telescope_api_failed` counter increments only for placed+failed records, reported in summary, distinct from `skipped` | integration | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_sync_06_fallback_counter_distinct_from_skipped` (new) | ❌ Wave 0 | ⬜ pending |
| TBD | TBD | TBD | SYNC-07 | — | A per-record API failure does not abort the run; subsequent records still process | integration | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_sync_07_api_failure_does_not_abort_run` (new) | ❌ Wave 0 | ⬜ pending |
| TBD | TBD | TBD | SYNC-08 | T-7-01 | Mocked slow/failing response — assert no second call attempted (single-attempt, no retry) | unit (mocked HTTP, call-count assertion) | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_sync_08_single_attempt_no_retry` (new) | ❌ Wave 0 | ⬜ pending |
| TBD | TBD | TBD | SYNC-09 | T-7-02 | Mocked failure whose exception embeds fake response content/API key — assert logged output is the fixed generic message, never the raw content/key | unit (mocked HTTP, log-content assertion) | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_sync_09_no_credential_or_body_leak_in_logs` (new) | ❌ Wave 0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_sync_lco_observation_calendar.py` — add the 8 new test methods listed in the Per-Task Verification Map above; the existing 19 tests cover Phases 4-6 only and must keep passing unmodified
- [ ] Shared mock-response-builder helper for "successful block", "failed/timeout", and "unmapped code" response shapes (extends the existing `_parameters()`/`_create_record()` fixture pattern already used in this test file)
- [ ] Timeout-specific test double: `requests.exceptions.Timeout` raised via `side_effect` on the mocked call, to drive the SYNC-08 "single attempt" assertion (`mock.assert_called_once()`)
- [ ] Decide and document the exact mock patch target (`tom_observations.facilities.ocs.make_request` vs. `requests.request`/`requests.get` at the lowest level the new resolver function calls) once its signature is finalized

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Per-site aperture-class assignment for `tlv`, `elp`, `lsc`, `cpt`, `tfn` in the verified `SITE_TELESCOPE_MAP` is factually correct | TELESCOPE-01 | No installed-library default or repo-local data confirms the real aperture-class inventory per site (RESEARCH.md Open Questions #1-2); requires checking LCO's public telescope-specs documentation or a live authenticated API response | Cross-check each site's class assignment against LCO's current site/telescope documentation (or a live `/api/instruments/` response) before treating the dict as final; mark unconfirmed entries `[ASSUMED]` in code comments until verified |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
