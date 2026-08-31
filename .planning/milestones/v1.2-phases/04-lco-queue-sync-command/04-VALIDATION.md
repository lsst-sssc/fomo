---
phase: 04
slug: lco-queue-sync-command
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-17
---

# Phase 04 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django test runner (`django.test.TestCase`), not pytest — per CLAUDE.md, `pytest` only collects `tests/`, `src/`, `docs/`; Django app tests under `solsys_code/` run via `./manage.py test` |
| **Config file** | none dedicated — driven by `src/fomo/settings.py` (`DJANGO_SETTINGS_MODULE`); `pyproject.toml` configures `ruff`/pytest scope only |
| **Quick run command** | `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar` |
| **Full suite command** | `python manage.py test solsys_code && ruff check . && ruff format --check .` |
| **Estimated runtime** | ~0.04s for this module's 15 tests; full `solsys_code` suite (109 tests) is slower on a cold cache — importing `solsys_code.views`/`ephem_utils` triggers a one-time ~1.6 GB SPICE kernel + ASSIST ephemeris load (CLAUDE.md "Heavy import side effect") |

---

## Sampling Rate

- **After every task commit:** Run `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar`
- **After every plan wave:** Run `python manage.py test solsys_code && ruff check . && ruff format --check .`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~1s (module-scoped run, warm SPICE cache)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-T1/T2 | 01 | 1 | SELECT-01 | T-04-01 (accept) | Non-matching proposal produces no event | unit (Django TestCase) | `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_select_01_only_matching_proposal_creates_events` | ✅ | ✅ green |
| 04-01-T1/T2 | 01 | 1 | SYNC-01 (D-01) | T-04-03 (mitigate) | `url` keyed via `get_observation_url`, never `requestgroups` | unit | `...test_sync_01_d01_url_uses_requests_path_not_requestgroups` | ✅ | ✅ green |
| 04-01-T1/T2 | 01 | 1 | SYNC-02 (D-03) | T-04-02 (mitigate) | Unscheduled record → `parameters` times + `[QUEUED]` title | unit | `...test_sync_02_d03_unscheduled_uses_parameters_times_and_queued_title` | ✅ | ✅ green |
| 04-01-T1/T2 | 01 | 1 | SYNC-03 (D-03) | T-04-02 (mitigate) | Placed record → `scheduled_start/end`, clean title | unit | `...test_sync_03_d03_placed_uses_scheduled_times_and_clean_title` | ✅ | ✅ green |
| 04-01-T1/T2 | 01 | 1 | SYNC-04 | — | Re-run updates in place, no duplicate, no `modified` churn on unchanged | unit | `...test_sync_04_rerun_updates_in_place_no_churn_on_unchanged` | ✅ | ✅ green |
| 04-01-T1/T2 | 01 | 1 | SYNC-05 (D-05) | — | `telescope`/`instrument`/`proposal` populated; description contains proposal/status/window | unit | `...test_sync_05_telescope_instrument_proposal_populated` + `...test_sync_05_d05_description_contains_proposal_status_and_window` | ✅ | ✅ green |
| 04-01-T1/T2 | 01 | 1 | TERM-01 (D-04, D-06) | — | All 4 failure states get correct prefix and are retained; `COMPLETED` gets clean title | unit | `...test_term_01_d04_window_expired_gets_expired_prefix` + 3 sibling tests + `...test_d06_completed_gets_clean_title_no_prefix` | ✅ | ✅ green |
| 04-01-T3 | 01 | 1 | — (quality gate) | — | Full `solsys_code` suite + `ruff check`/`format` clean, no regressions | integration/lint | `python manage.py test solsys_code && ruff check . && ruff format --check .` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

Bonus coverage beyond the 7 phase requirements (not gaps, found during discovery): `test_skip_path_missing_site_logged_and_skipped`, `test_skip_path_inconsistent_scheduled_times_logged_and_skipped` (T-04-02 mitigation evidence), `test_zero_match_reports_created_zero_no_command_error` (T-04-04 mitigation evidence).

---

## Wave 0 Requirements

*None: existing Django test infrastructure (`./manage.py test`) covers all phase requirements — no new framework or fixture scaffolding was needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| End-to-end spot-check against a real LCO proposal | PLAN.md `<verification>` item 3 | Optional human verify per the plan; requires a real fixture `ObservationRecord` and live shell run | Create a fixture record, run `./manage.py sync_lco_observation_calendar --proposal <code>`, confirm one `CalendarEvent` with the `/requests/<id>` url, correct title transition, and a sensible stdout summary |

*Marked optional in PLAN.md — automated suite already covers the equivalent behavior end-to-end (15/15 tests, independently re-run during this audit).*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (none — no gaps found)
- [x] No watch-mode flags
- [x] Feedback latency < 5s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-06-17

---

## Validation Audit 2026-06-17

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

All 7 requirement IDs (SELECT-01, SYNC-01..05, TERM-01) cross-referenced against `solsys_code/tests/test_sync_lco_observation_calendar.py` by test method name; independently re-ran `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar -v 1` during this audit — 15/15 tests pass (`OK`), confirming COVERED status for every requirement rather than trusting SUMMARY.md's claim alone.
