---
phase: 27
slug: the-canonical-run-record
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-29
---

# Phase 27 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django `TestCase` / `TransactionTestCase` via `./manage.py test` (app tests under `solsys_code/tests/`). The separate `pytest` suite (`testpaths = tests, src, docs`) is unaffected by this phase — no app-level test lives there. |
| **Config file** | None dedicated. `pyproject.toml` `[tool.pytest.ini_options]` scopes the pytest suite; Django test discovery is app-default. |
| **Quick run command** | `./manage.py test solsys_code.tests.test_admin solsys_code.tests.test_campaign_models solsys_code.tests.test_campaign_approval solsys_code.tests.test_calendar_template solsys_code.tests.test_calendar_utils solsys_code.tests.test_sync_lco_observation_calendar solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_import_campaign_csv` |
| **Full suite command** | `./manage.py test solsys_code` (pays the ~1.6 GB SPICE kernel cost once, via `test_ephem_utils.py` / `test_views.py`), plus `python -m pytest` for the unrelated suite |
| **Estimated runtime** | Quick run ~15–30 s (no SPICE cost). Full suite: minutes, dominated by the SPICE import side effect. |

**Why the quick run excludes two modules:** importing `solsys_code.ephem_utils` (transitively
`solsys_code.views`) runs `fomo_furnish_spiceypy()` at module load, downloading ~1.6 GB of SPICE
kernels on first use. The quick-run selection mirrors Phase 26's own narrow selection and
deliberately omits `test_ephem_utils.py` and `test_views.py` so per-task feedback stays fast.

---

## Sampling Rate

- **After every task commit:** Run the quick run command (fast, no SPICE cost)
- **After every plan wave:** Run `./manage.py test solsys_code` (full suite, once per wave)
- **Before `/gsd-verify-work`:** Full suite green AND `ruff check .` / `ruff format --check .` clean
- **Max feedback latency:** ~30 seconds (quick run)

---

## Per-Task Verification Map

> Task IDs are filled in by the planner. This table maps each phase requirement to its test
> entry point; the planner MUST attach an automated command to each task it writes.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 27-04..06 | 27 | 3 | CANON-01 | — | `source` is staff-only; not exposed via `ALLOWED_FIELDS_FOR_NON_STAFF` | unit/DB | `./manage.py test solsys_code.tests.test_import_campaign_csv solsys_code.tests.test_campaign_submission` | ✅ both exist | ✅ green |
| 27-01,04,06 | 27 | 1,3,4 | CANON-02 | — | N/A | unit/DB | `./manage.py test solsys_code.tests.test_campaign_models solsys_code.tests.test_calendar_utils` | ✅ exists | ✅ green |
| 27-04 | 27 | 3 | CANON-02 (backfill) | — | N/A | migration | `./manage.py test solsys_code.tests.test_canonical_record_migration` | ✅ exists (9 tests) | ✅ green |
| 27-01,03,05 | 27 | 1,2,5 | CANON-03 | — | N/A | unit/DB | `./manage.py test solsys_code.tests.test_admin solsys_code.tests.test_sync_lco_observation_calendar solsys_code.tests.test_calendar_template solsys_code.tests.test_load_telescope_runs` | ✅ all four exist | ✅ green |
| 27-03 | 27 | 2 | CANON-03 (rename) | — | N/A | migration | `./manage.py test solsys_code.tests.test_canonical_record_migration` | ✅ exists (same module) | ✅ green |
| 27-04 | 27 | 3 | CANON-04 | T-27-01 | Attribution rows only exist once a human confirms (D-01); `confirmed_by` never self-assigned | unit/DB | `./manage.py test solsys_code.tests.test_campaign_run_observation` | ✅ exists (7 tests) | ✅ green |
| 27-05,07 | 27 | 4,6 | CANON-05 | T-27-02 | `pending_review` runs stay hidden from non-staff in the modal (D-09/D-10) | unit/DB (admin client) | `./manage.py test solsys_code.tests.test_admin` (extended) | ✅ extend existing | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Source:** `27-VERIFICATION.md` (2026-08-06 re-verification, 5/5 truths, 420-test targeted
regression sweep — `python manage.py test solsys_code.tests.test_canonical_record_migration
solsys_code.tests.test_campaign_run_observation solsys_code.tests.test_repair_stale_campaign_run_sites
solsys_code.tests.test_admin solsys_code.tests.test_calendar_template solsys_code.tests.test_calendar_utils
solsys_code.tests.test_campaign_views solsys_code.tests.test_campaign_submission
solsys_code.tests.test_campaign_approval solsys_code.solsys_code_observatory.tests.test_timezone_backfill_migration
solsys_code.tests.test_import_campaign_csv -v 0` → OK). All Wave 0 test modules this file
originally flagged missing were created during execution and independently re-run by the
verifier, not merely claimed by the SUMMARYs.

---

- [x] New migration-testing module (`solsys_code/tests/test_canonical_record_migration.py`,
      9 tests), following `test_window_schema_migration.py`'s existing `MigrationExecutor` shape.
      Covers: the rename preserves companion rows' `is_verified` history; `source` takes the
      chosen legacy value for pre-existing rows; the `telescope_class` backfill produces the
      correct value for each D-16 row model.
- [x] New tests for the observation-link model (`solsys_code/tests/test_campaign_run_observation.py`,
      7 tests): the named `UniqueConstraint` fires on a genuine duplicate; `CASCADE` on run delete
      removes the link row but leaves the `ObservationRecord` untouched; `confirmed_by`/`confirmed_at`
      are set by `save_formset` and left blank by any other write path.
- [x] `test_admin.py` extended with inline-formset submission tests proving D-07's stamping.
- [x] Mock fixture for the D-16a repair task's HST tier-2 resolution — present in
      `test_repair_stale_campaign_run_sites.py`, part of the 420-test regression sweep.

**All Wave 0 items closed during execution (Plan 27-03/27-04/27-07); re-confirmed present and
green by the independent 2026-08-06 re-verification pass (420 tests, 0 failures).**

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| The one-time D-16 live repair run against the real dev DB, resolving HST (pk 8, 12) through a genuine tier-2 MPC Obscodes API hit | D-16 / D-16a (no CANON requirement) | D-16a explicitly accepts that this result is not reproducible offline or in CI. The automated suite proves the repair task's *code path* with a mocked API; the live run against real data is by definition a single execution against a reachable network. | Run the repair management command / migration against the dev DB with network available. Confirm pk 8 and 12 gain a real (non-placeholder) `Observatory`, and that `site_needs_review` clears through the normal resolution path (D-17 — no special-case flag clearing). |
| The calendar event modal visibly shows a link back to its run, and shows nothing for a `pending_review` run | CANON-05 / D-08, D-09 | The modal is rendered by an upstream `tom_calendar` view through a FOMO template override; the visual result in a browser is not asserted by the Django test client. | Open the calendar page, click an event whose companion record has a `run` set, confirm the run link renders and resolves. Repeat with a `pending_review` run as a non-staff user and confirm no link appears. |

---

## Validation Sign-Off

- [x] All tasks have an automated verify command or a declared Wave 0 dependency
- [x] Sampling continuity: no 3 consecutive tasks without an automated verify
- [x] Wave 0 covers all ❌ MISSING references in the verification map above
- [x] No watch-mode flags in any test command
- [x] Feedback latency < 30 s for the quick run
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated (Phase 30 plan 30-04, D-08 reconciliation)

## Validation Audit 2026-08-31

Reconciled by Phase 30 plan 30-04 (D-08). This file was seeded by plan-phase with `TBD` task
IDs and 4 Wave 0 gaps (❌) before execution. Cross-referenced against `27-VERIFICATION.md`
(2026-08-06 re-verification): all 4 Wave 0 test modules were created during execution
(`test_canonical_record_migration.py`, `test_campaign_run_observation.py`, extended
`test_admin.py`, `test_repair_stale_campaign_run_sites.py`) and were independently re-run by
the verifier in a 420-test targeted regression sweep — not merely claimed by the plan
SUMMARYs. No new test was written by this reconciliation; the gap was that the file was never
promoted after the gaps were closed.

| Metric | Count |
|--------|-------|
| Gaps found | 4 (all pre-existing Wave 0 markers, already closed by execution) |
| Resolved | 4 (confirmed closed via 27-VERIFICATION.md cross-reference, no new work needed) |
| Escalated | 0 |
