---
phase: 17
slug: coverage-gap-analysis-deferrable-to-v2-1
status: draft
nyquist_compliant: false
wave_0_complete: false
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
| TBD | TBD | TBD | GAP-01 | — | `17-GAP-01-DECISION.md` documents the dark-window-only decision (D-02) | manual-only | N/A — verified by document review | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | GAP-02 | — | `observable_dates()` returns dates with a non-zero dark window, skipping `ValueError` dates (D-03/D-04) | unit | `./manage.py test solsys_code.tests.test_campaign_gap.TestObservableDates -v 2` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | GAP-02 | — | `claimed_dates()` derives dates per D-05/D-06/D-07 and excludes cancelled/not-awarded/weather-failure runs | unit | `./manage.py test solsys_code.tests.test_campaign_gap.TestClaimedDates -v 2` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | GAP-02 | — | Undated `CampaignRun`s (D-08) are flagged, not silently dropped or counted as claiming a date | unit | `./manage.py test solsys_code.tests.test_campaign_gap.TestClaimedDates::test_undated_runs_flagged -v 2` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | GAP-02 | — | Gap view never computes inline on the table-view GET; is a separate endpoint (D-09) | integration | `./manage.py test solsys_code.tests.test_campaign_gap.TestGapAnalysisView::test_table_view_does_not_trigger_computation -v 2` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | GAP-02 | — | Cache hit avoids recomputation; TTL and "last computed at" behave correctly (D-10) | integration | `./manage.py test solsys_code.tests.test_campaign_gap.TestGapAnalysisView::test_cache_hit_skips_recomputation -v 2` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | GAP-02 | — | Date range clamps to 180-day max regardless of client input (D-11) | unit | `./manage.py test solsys_code.tests.test_campaign_gap.TestClampDateRange -v 2` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | GAP-02 | T-17-01 | Submitted `target_pk`/`site_pk` outside this campaign's scope are rejected (IDOR) | integration | `./manage.py test solsys_code.tests.test_campaign_gap.TestGapAnalysisView::test_rejects_out_of_scope_target_and_site -v 2` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | GAP-02 | — | Gap-analysis button hidden/disabled when D-14 applies (no targets, or no resolved site) | integration | `./manage.py test solsys_code.tests.test_campaign_gap.TestGapAnalysisButton -v 2` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | GAP-01 (transitively) | — | No module in this phase imports `ephem_utils`/`solsys_code.views` at module scope | static check | `grep -rn "import.*ephem_utils\|from solsys_code.views import" solsys_code/campaign_gap.py solsys_code/tests/test_campaign_gap.py` (expect zero output) | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*Plan/wave/task IDs are `TBD` until gsd-planner assigns them; gsd-planner should fill these in as plans are created.*

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_campaign_gap.py` — new file, covers all GAP-02 rows above
- [ ] No new shared fixtures/conftest needed — existing `Observatory`/`CampaignRun`/`TargetList` factories (`NonSiderealTargetFactory` per CLAUDE.md) already cover this phase's fixture needs
- [ ] Framework install: none — Django `TestCase` is already the established framework for this codebase's DB-dependent tests

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|--------------------|
| `17-GAP-01-DECISION.md` written and documents the dark-window-only decision with rationale | GAP-01 | It's a written decision artifact (D-02), not executable code | Confirm the file exists in the phase directory, mirrors `13-DECISION.md`'s shape, and states the dark-window-only decision with rationale referencing the pre-milestone research |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
