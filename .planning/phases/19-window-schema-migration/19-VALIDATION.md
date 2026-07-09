---
phase: 19
slug: window-schema-migration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-09
---

# Phase 19 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django `TestCase` (`unittest`-style), run via `./manage.py test` |
| **Config file** | none — `pyproject.toml`'s `testpaths` deliberately excludes `solsys_code/` per CLAUDE.md; Django's own test runner config lives in `src/fomo/settings.py` |
| **Quick run command** | `./manage.py test solsys_code.tests.test_campaign_models` |
| **Full suite command** | `./manage.py test solsys_code` |
| **Estimated runtime** | seconds (no SPICE-importing module is touched by this phase's files) |

---

## Sampling Rate

- **After every task commit:** Run `./manage.py test solsys_code.tests.test_campaign_models`
- **After every plan wave:** Run `./manage.py test solsys_code` (full app suite) plus `ruff check .` / `ruff format --check .`
- **Before `/gsd-verify-work`:** Full `./manage.py test solsys_code` suite green, plus a manual `manage.py migrate` dry run against a copy of the real dev DB to exercise the D-07/D-08 dedup against the two known real duplicate-row pairs
- **Max feedback latency:** seconds — no long-running suite; this phase's modules deliberately avoid importing `solsys_code.views`/`ephem_utils` (the ~1.6GB SPICE-kernel-downloading module) per CLAUDE.md

---

## Per-Task Verification Map

Task IDs are not yet assigned (VALIDATION.md is created before PLAN.md in the
plan-phase workflow). The planner/executor should map each row below to the
concrete task(s) that implement it as plans are created; the requirement-level
mapping itself is fixed by 19-RESEARCH.md's Validation Architecture section.

| Requirement | Behavior | Test Type | Automated Command | File Exists | Status |
|-------------|----------|-----------|---------------------|-------------|--------|
| SCHED-02 | `CampaignRun` savable with `window_start == window_end` (single night) | unit | `./manage.py test solsys_code.tests.test_campaign_models` | ✅ (rewrite `TestCampaignRunFieldInventory`) | ⬜ pending |
| SCHED-03 | `CampaignRun` savable fully TBD (both window fields null) | unit | `./manage.py test solsys_code.tests.test_campaign_models` | ❌ W0 — new test case needed | ⬜ pending |
| SCHED-04 | Two TBD rows for same campaign+telescope+contact_person collide (`IntegrityError`); two TBD rows differing only in `contact_person` both save; resolved-window collisions key on all 4 fields | unit | `./manage.py test solsys_code.tests.test_campaign_models` | ❌ W0 — new test cases needed for both constraints | ⬜ pending |
| SCHED-05 | Migration backfill: every pre-existing row survives with `window_start == window_end == former obs_date`; de-dup removes only the known duplicate pairs | migration / data | `TestCase` against post-migration schema, plus manual `manage.py migrate` dry run against a dev-DB copy (see Manual-Only Verifications) | ❌ W0 — no migration-testing convention exists yet in this repo | ⬜ pending |
| Table/queue rendering (D-03/D-04/D-05) | TBD badge (or plain-text fallback), `->` range display, nulls-last default sort | unit + view | `./manage.py test solsys_code.tests.test_campaign_views` | ✅ (rewrite `test_default_sort_is_obs_date_descending`; add TBD-badge and range-display cases) | ⬜ pending |
| Calendar projection (D-06) | Ground vs. space branch produces correct `CalendarEvent` window | unit | `./manage.py test solsys_code.tests.test_campaign_approval` | ✅ (existing calendar-projection tests need window-field rewrite + new space-branch case) | ⬜ pending |
| Coverage-gap `claimed_dates()` rewrite | Every date in `[window_start, window_end]` claimed; TBD → undated bucket | unit | `./manage.py test solsys_code.tests.test_campaign_gap` | ✅ (rewrite `TestClaimedDates`; delete `_observing_night_date()`-dependent case per Pitfall 5) | ⬜ pending |
| CSV import lookup key | Natural-key lookup uses `window_start` not `ut_start` | unit | `./manage.py test solsys_code.tests.test_import_campaign_csv` | ✅ (rewrite `test_duplicate_unparseable_ut_time_rows_do_not_merge`, `test_idempotent_rerun_no_duplicates`) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_campaign_models.py` — add a test case asserting `CampaignRun` saves successfully with both `window_start`/`window_end` null (SCHED-03)
- [ ] `solsys_code/tests/test_campaign_models.py` — add test cases exercising both new partial `UniqueConstraint`s directly: same-key TBD collision raises `IntegrityError`; same-key resolved-window collision raises `IntegrityError`; two TBD rows differing only in `contact_person` both save (SCHED-04)
- [ ] No new test framework install needed — `./manage.py test` already covers this app

*Migration-replay testing (SCHED-05) has no existing convention in this repo — see Manual-Only Verifications below rather than introducing a new `TransactionTestCase`-based migration-testing framework for a single phase, per 19-RESEARCH.md's recommendation.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|--------------------|
| Migration backfill + de-dup against real pre-migration data | SCHED-05 | A `TestCase` can only assert the model's shape *after* migrating — it cannot recreate the old three-field shape to feed through the real `RunPython` backfill/dedup logic; the two known real duplicate pairs (pks 15/17, 16/18) only exist in the actual dev DB, not in test fixtures | Take a copy of the dev DB, run `./manage.py migrate`, confirm all pre-existing rows now have `window_start == window_end == former obs_date`, confirm the two known duplicate pairs were removed (one row per pk pair survives) and the removal was logged, per D-07/D-08 |

*Automated coverage for the same requirement (asserting current field set / constraint shape post-migration) is listed separately in the Per-Task Verification Map above — the manual step exists to validate the backfill/dedup transition itself, not the end-state schema.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s (no long-running suite expected)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
