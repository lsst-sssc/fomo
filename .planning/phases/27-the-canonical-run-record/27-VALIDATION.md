---
phase: 27
slug: the-canonical-run-record
status: draft
nyquist_compliant: false
wave_0_complete: false
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
| TBD | TBD | TBD | CANON-01 | — | `source` is staff-only; not exposed via `ALLOWED_FIELDS_FOR_NON_STAFF` | unit/DB | `./manage.py test solsys_code.tests.test_import_campaign_csv solsys_code.tests.test_campaign_submission` | ✅ both exist | ⬜ pending |
| TBD | TBD | TBD | CANON-02 | — | N/A | unit/DB | `./manage.py test solsys_code.tests.test_campaign_models solsys_code.tests.test_calendar_utils` | ✅ exists | ⬜ pending |
| TBD | TBD | TBD | CANON-02 (backfill) | — | N/A | migration | new `TransactionTestCase` module | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | CANON-03 | — | N/A | unit/DB | `./manage.py test solsys_code.tests.test_admin solsys_code.tests.test_sync_lco_observation_calendar solsys_code.tests.test_calendar_template solsys_code.tests.test_load_telescope_runs` | ✅ all four exist | ⬜ pending |
| TBD | TBD | TBD | CANON-03 (rename) | — | N/A | migration | new rename migration test | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | CANON-04 | T-27-01 | Attribution rows only exist once a human confirms (D-01); `confirmed_by` never self-assigned | unit/DB | new tests for the observation-link model | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | CANON-05 | T-27-02 | `pending_review` runs stay hidden from non-staff in the modal (D-09/D-10) | unit/DB (admin client) | `./manage.py test solsys_code.tests.test_admin` (extended) | ✅ extend existing | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] New migration-testing module (e.g. `solsys_code/tests/test_canonical_record_migration.py`),
      following `test_window_schema_migration.py`'s existing `MigrationExecutor` shape. Must cover:
      the rename preserves companion rows' `is_verified` history; `source` takes the chosen legacy
      value for pre-existing rows; the `telescope_class` backfill produces the correct value for
      each D-16 row model (JWST-alias, HST tier-2 mocked, Swift-empty, JUICE-empty, class-wide-empty).
- [ ] New tests for the observation-link model: the named `UniqueConstraint` fires on a genuine
      duplicate; `CASCADE` on run delete removes the link row but leaves the `ObservationRecord`
      untouched; `confirmed_by` / `confirmed_at` are set by `save_formset` and left blank by any
      other write path.
- [ ] Extend `test_admin.py` with inline-formset submission tests proving D-07's stamping: a new
      row created via the admin gets `confirmed_by=request.user`; an existing row edited via the
      admin is NOT re-stamped.
- [ ] A mock fixture for the D-16a repair task's HST tier-2 resolution, mirroring
      `test_import_campaign_csv.py`'s existing `_MPC_OBS_DATA_E10` / `@patch('requests.get')` pattern.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| The one-time D-16 live repair run against the real dev DB, resolving HST (pk 8, 12) through a genuine tier-2 MPC Obscodes API hit | D-16 / D-16a (no CANON requirement) | D-16a explicitly accepts that this result is not reproducible offline or in CI. The automated suite proves the repair task's *code path* with a mocked API; the live run against real data is by definition a single execution against a reachable network. | Run the repair management command / migration against the dev DB with network available. Confirm pk 8 and 12 gain a real (non-placeholder) `Observatory`, and that `site_needs_review` clears through the normal resolution path (D-17 — no special-case flag clearing). |
| The calendar event modal visibly shows a link back to its run, and shows nothing for a `pending_review` run | CANON-05 / D-08, D-09 | The modal is rendered by an upstream `tom_calendar` view through a FOMO template override; the visual result in a browser is not asserted by the Django test client. | Open the calendar page, click an event whose companion record has a `run` set, confirm the run link renders and resolves. Repeat with a `pending_review` run as a non-staff user and confirm no link appears. |

---

## Validation Sign-Off

- [ ] All tasks have an automated verify command or a declared Wave 0 dependency
- [ ] Sampling continuity: no 3 consecutive tasks without an automated verify
- [ ] Wave 0 covers all ❌ MISSING references in the verification map above
- [ ] No watch-mode flags in any test command
- [ ] Feedback latency < 30 s for the quick run
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
