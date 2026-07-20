---
phase: 17
slug: coverage-gap-analysis-deferrable-to-v2-1
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-04
---

# Phase 17 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django `TestCase` (`django.test.TestCase`), per this codebase's established two-suite split |
| **Config file** | None dedicated — governed by `manage.py`'s `DJANGO_SETTINGS_MODULE=src.fomo.settings` |
| **Quick run command** | `./manage.py test solsys_code.tests.test_campaign_gap` |
| **Full suite command** | `./manage.py test solsys_code` |
| **Estimated runtime** | ~30-60s quick; full suite per existing solsys_code baseline |

---

## Sampling Rate

- **After every task commit:** Run `./manage.py test solsys_code.tests.test_campaign_gap`
- **After every plan wave:** Run `./manage.py test solsys_code`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|--------------------|-------------|--------|
| 17-01 T1 | 17-01 | 1 | GAP-01 | — | `17-GAP-01-DECISION.md` documents the dark-window-only decision (D-02) | manual-only | N/A — verified by document review | ✅ | ✅ green |
| 17-01 T3 | 17-01 | 1 | GAP-02 | — | `observable_dates()` returns dates with a non-zero dark window, skipping `ValueError` dates (D-03/D-04) | unit | `./manage.py test solsys_code.tests.test_campaign_gap.TestObservableDates -v 2` | ✅ | ✅ green (2/2) |
| 17-01 T3 | 17-01 | 1 | GAP-02 | — | `claimed_dates()` derives dates per D-05/D-06/D-07 and excludes cancelled/not-awarded/weather-failure runs | unit | `./manage.py test solsys_code.tests.test_campaign_gap.TestClaimedDates -v 2` | ✅ | ✅ green (6/6) |
| 17-01 T3 | 17-01 | 1 | GAP-02 | — | Undated `CampaignRun`s (D-08) are flagged, not silently dropped or counted as claiming a date | unit | `./manage.py test solsys_code.tests.test_campaign_gap.TestClaimedDates::test_undated_runs_flagged -v 2` | ✅ | ✅ green |
| 17-02 T3 | 17-02 | 2 | GAP-02 | — | Gap view never computes inline on the table-view GET; is a separate endpoint (D-09) | integration | `./manage.py test solsys_code.tests.test_campaign_gap.TestGapAnalysisView::test_table_view_does_not_trigger_computation -v 2` | ✅ | ✅ green |
| 17-02 T3 | 17-02 | 2 | GAP-02 | — | Cache hit avoids recomputation; TTL and "last computed at" behave correctly (D-10) | integration | `./manage.py test solsys_code.tests.test_campaign_gap.TestGapAnalysisView::test_cache_hit_skips_recomputation -v 2` | ✅ | ✅ green |
| 17-01 T3 | 17-01 | 1 | GAP-02 | — | Date range clamps to 180-day max regardless of client input (D-11) | unit | `./manage.py test solsys_code.tests.test_campaign_gap.TestClampDateRange -v 2` | ✅ | ✅ green (3/3) |
| 17-02 T3 | 17-02 | 2 | GAP-02 | T-17-01 | Submitted `target_pk`/`site_pk` outside this campaign's scope are rejected (IDOR) | integration | `./manage.py test solsys_code.tests.test_campaign_gap.TestGapAnalysisView::test_rejects_out_of_scope_target_and_site -v 2` | ✅ | ✅ green |
| 17-03 T2 | 17-03 | 3 | GAP-02 | — | Gap-analysis button hidden/disabled when D-14 applies (no targets, or no resolved site) | integration | `./manage.py test solsys_code.tests.test_campaign_gap.TestGapAnalysisButton -v 2` | ✅ | ✅ green (3/3) |
| 17-01 T3 | 17-01 | 1 | GAP-01 (transitively) | — | No module in this phase imports the heavy SPICE-loading ephemeris/`solsys_code.views` module at module scope | static check | `grep -rnE "^[[:space:]]*(from\|import)[[:space:]].*ephem_utils\|^[[:space:]]*from solsys_code.views import" solsys_code/campaign_gap.py solsys_code/campaign_views.py solsys_code/tests/test_campaign_gap.py` (expect zero output) | N/A | ✅ green (zero matches) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*Plan/wave/task IDs assigned by gsd-planner (2026-07-04): 17-01 (Wave 1, decision doc + campaign_gap.py + unit tests), 17-02 (Wave 2, form/view/url + integration tests), 17-03 (Wave 3, templates + button gating test + human-verify). The import guard is scoped to real import lines so a conceptual docstring mention does not false-positive.*

---

## Wave 0 Requirements

- [x] `solsys_code/tests/test_campaign_gap.py` — created in 17-01, extended in 17-02/17-03; covers all GAP-02 rows above (23 test methods total across `TestClampDateRange`, `TestObservableDates`, `TestClaimedDates`, `TestGapAnalysisView`, `TestGapAnalysisButton`)
- [x] No new shared fixtures/conftest needed — existing `Observatory`/`CampaignRun`/`TargetList` factories (`NonSiderealTargetFactory` per CLAUDE.md) covered this phase's fixture needs
- [x] Framework install: none — Django `TestCase` was already the established framework for this codebase's DB-dependent tests

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|--------------------|
| `17-GAP-01-DECISION.md` written and documents the dark-window-only decision with rationale | GAP-01 | It's a written decision artifact (D-02), not executable code | Confirm the file exists in the phase directory, mirrors `13-DECISION.md`'s shape, and states the dark-window-only decision with rationale referencing the pre-milestone research |

---

## Validation Audit 2026-07-05

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

All 10 rows in the Per-Task Verification Map re-run against the post-execution, post-code-review-fix
codebase (commits `8d93714`, `2b7a7e8`) and confirmed green: `TestClampDateRange` (3/3),
`TestObservableDates` (2/2), `TestClaimedDates` (6/6, incl. `test_undated_runs_flagged`),
`TestGapAnalysisView` (4/4, incl. `test_rejects_out_of_scope_target_and_site` and
`test_cache_hit_skips_recomputation`), `TestGapAnalysisButton` (3/3), and the static import-guard
grep (zero matches, as required). The `17-GAP-01-DECISION.md` manual-only row was confirmed present
and on-topic. No gaps to fill — auditor spawn was not needed (Step 3 short-circuit).

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies (GAP-01 decision doc is manual-only per D-02; the human-verify checkpoint is the only non-automated implementation gate)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (test_campaign_gap.py created in 17-01 T3, extended in 17-02 T3 and 17-03 T2)
- [x] No watch-mode flags
- [x] Feedback latency < 60s (unit + small-range integration tests; full-window sun_event runs avoided in tests)
- [x] `nyquist_compliant: true` set in frontmatter
- [x] All 10 rows re-verified green post-execution (2026-07-05)

**Approval:** verified 2026-07-05 (all rows green, zero gaps, phase Nyquist-compliant)
