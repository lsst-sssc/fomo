---
phase: "35"
slug: "allocation-layer-classical-cutover"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: true
wave_0_complete: false
created: "2026-09-12"
---

# Phase 35 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django built-in test runner (`django.test.TestCase` via `python manage.py test`) — the project's only functioning suite |
| **Config file** | none — Django test discovery via `manage.py test <labels>`; `pyproject.toml`'s pytest config is legacy/unused for this app |
| **Quick run command** | `python manage.py test solsys_code.tests.test_allocation_projector` (new file) or `python manage.py test solsys_code.tests.test_load_telescope_runs` for ingest changes |
| **Full suite command** | `LABELS=$(ls solsys_code/tests/test_*.py solsys_code/solsys_code_observatory/tests/test_*.py \| grep -v "tests/test_views\.py$" \| sed "s\|/\|.\|g; s\|\.py$\|\|" \| tr "\n" " "); python manage.py test $LABELS && python manage.py test solsys_code.tests.test_views.TestSplitNumberUnitRegex solsys_code.tests.test_views.TestJPLSBDBQuery` |
| **Estimated runtime** | ~60 seconds per module (SPICE kernel load dominates); full label-list suite several minutes |

---

## Sampling Rate

- **After every task commit:** Run the relevant single test module/class (e.g. `python manage.py test solsys_code.tests.test_allocation_projector`)
- **After every plan wave:** Run the full label-list suite command above. This is a wave-boundary sampling step owned by the execution loop, not any single plan's acceptance gate — in waves 2 and 3 two plans run side by side, so a whole-suite result there is not attributable to one of them. Each plan in those waves gates on its own modules and files instead; the phase's repo-wide suite and lint gates live in **35-06 Task 3** (wave 4, the first wave with a single plan in it).
- **Before `/gsd-verify-work`:** Full suite must be green, plus `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files`
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 35-01-01 | 35-01 | 1 | ALLOC-01 | T-35-06 | per-night events only for resolved-site awarded windows; queue/class-wide/satellite runs keep one container | unit | `python manage.py test solsys_code.tests.test_allocation_projector` | ❌ W0 (created by this task) | ⬜ pending |
| 35-01-02 | 35-01 | 1 | ALLOC-03 | T-35-03 | link retires the night and attributes the record's own event; unlink restores the night and clears that attribution unless a staff member confirmed it; observation event byte-unchanged; foreign human-confirmed attribution refused | unit | `python manage.py test solsys_code.tests.test_allocation_projector` | ❌ W0 (created by 35-01-01) | ⬜ pending |
| 35-01-03 | 35-01 | 1 | ALLOC-02 | — | site-local observing-night keying for a Chilean and an Australian site, plus the no-`sun_event()`-recompute regression | unit | `python manage.py test solsys_code.tests.test_allocation_projector` | ❌ W0 (created by 35-01-01) | ⬜ pending |
| 35-02-01 · 35-02-02 · 35-02-03 | 35-02 | 2 | ALLOC-01, ALLOC-03 | T-35-09 | the existing suite states the new dispatch and the new key families; no retired test is dropped without a destination or a reason | unit + integration | `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs solsys_code.tests.test_campaign_approval solsys_code.tests.test_calendar_utils solsys_code.tests.test_observation_projector solsys_code.tests.test_observation_projector_signals solsys_code.tests.test_project_observation_calendar` (this plan's own seven modules; the repo-wide label-list run is 35-06 Task 3's) | ✅ | ⬜ pending |
| 35-03-01 · 35-03-02 | 35-03 | 2 | ALLOC-01, ALLOC-04 | T-35-10 / T-35-11 | sub-night window fields honoured per night; a changed span re-mints rather than rewriting; no astropy call on an unchanged night | unit | `python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_campaign_models` | ✅ | ⬜ pending |
| 35-04-01 · 35-04-02 · 35-04-03 | 35-04 | 3 | ALLOC-03 | T-35-04 / T-35-12 | link/unlink and record-save triggers are immediate, never raise, never call out, never recurse | unit + integration | `python manage.py test solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_observation_projector_signals` | ❌ W0 (created by 35-04-01) / ✅ | ⬜ pending |
| 35-05-02 · 35-05-03 | 35-05 | 3 | ALLOC-04 | T-35-01 / T-35-02 / T-35-13 | `load_telescope_runs` writes a campaign-less `CampaignRun` with a collision-safe `source_identifier`; re-import is a no-op; the per-night calendar is unchanged field by field | integration | `python manage.py test solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_telescope_runs` | ✅ | ⬜ pending |
| 35-06-01 · 35-06-02 · 35-06-03 | 35-06 | 4 | ALLOC-05 | T-35-05 / T-35-14 / T-35-15 | cutover never leaves a duplicate or orphan; an unexplained event is left byte-identical, reported, and exits non-zero; no `CalendarEvent` is ever deleted by the command | integration + notebook | `python manage.py test solsys_code.tests.test_cutover_classical_allocations` | ❌ W0 (created by 35-06-02) | ⬜ pending |
| 35-07-01 · 35-07-02 · 35-07-03 | 35-07 | 5 | ALLOC-04, ALLOC-05 | T-35-16 / T-35-17 | paired docs regenerated with real executed output; the notebook never writes to the developer database; the runbook states the four cutover steps in order | docs + notebook | `pre-commit run sphinx-build --all-files` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Task IDs are `{phase}-{plan}-{task}`. Rows marked `❌ W0 (created by …)` name the task that creates the missing test module — every Wave 0 gap in this phase is closed by the first task that needs it, so no task ships with a `MISSING` verify sentinel.*

---

## Wave 0 Requirements

Each gap below is closed by the first task that needs it, inside the plan named — there is no
separate Wave 0 plan and no task ships with a `MISSING` verify sentinel.

- [ ] `solsys_code/tests/test_allocation_projector.py` — new file covering ALLOC-01/02/03, created by **35-01 Task 1**; carries `AllocationProjectorTestBase` with a campaign-less (`campaign=None`) `CampaignRun` helper
- [ ] A Chilean `Observatory` test fixture (`timezone='America/Santiago'`, obscode `809`) alongside an Australian one (`E10`, `Australia/Sydney`) on `AllocationProjectorTestBase` — created by **35-01 Task 1**; both rows are created by the fixture rather than read from the developer database
- [ ] `solsys_code/tests/test_allocation_projector_signals.py` — new file covering the trigger contract, created by **35-04 Task 1**
- [ ] `solsys_code/tests/test_cutover_classical_allocations.py` — new file covering the cutover command, created by **35-06 Task 2**
- [ ] A Chilean `Observatory` fixture on `CampaignReconcilerTestBase` (the existing base fixtures only `F65`) — added by **35-02 Task 1** so the migrated boundary cases can pick either hemisphere
- [ ] Framework install: none — `django.test.TestCase` and `tom_targets.tests.factories.NonSiderealTargetFactory` are already available project-wide

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Cutover against the real dev DB leaves one event per night (241 events / 56 `RUN:{pk}:{date}` / 0 `ALLOC:` baseline) | ALLOC-05 | Real-data before/after diff; the unit test proves the mechanism on fixtures, not the operator's actual calendar | **35-06 Task 3**: copy `src/fomo_db.sqlite3` to a scratch path, point `FOMO_DATABASE_PATH` at the copy, run the four stated steps in order (`migrate`; deploy is a no-op in-tree; `cutover_classical_allocations --dry-run` then for real; `reconcile_campaign_runs --dry-run` then for real), capture before/after counts by key family and reconcile the three night groups. **35-07 Task 3** then reproduces the same diff as executed output in `reconcile_campaign_runs_demo.ipynb` with four `assert` statements, so a later re-execution fails loudly rather than printing a stale claim. Never run any of it un-routed. |
| The final calendar reads correctly to an operator (one entry per night, the unexplained list contains only recognisable rows) | ALLOC-05 | A judgement about what the calendar *means*, not a property a test can assert | **35-06 Task 3** `<human-check>` and **35-07 Task 3** `<human-check>`: read the before/after table and the rendered notebook, confirm the reconciliation sums and the unexplained list is recognisable |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies — every task in all seven plans carries at least one runnable `<automated>` command with a `<fails_when>` sibling; `check verify-failure-directions 35` reports 66/66 ok, 0 blockers
- [x] Sampling continuity: no 3 consecutive tasks without automated verify — no task in this phase lacks one
- [x] Wave 0 covers all MISSING references — all four new test modules and both new fixtures are created by the first task that needs them (see Wave 0 Requirements); no `MISSING — Wave 0 …` sentinel appears in any plan
- [x] No watch-mode flags — every command is a single-shot `python manage.py test`, `pre-commit run`, `python -c` or `jupyter nbconvert` invocation
- [x] Feedback latency < 120s — a single test module runs in roughly 60 seconds (SPICE kernel load dominates); the full label-list suite is a per-wave gate, not a per-task one
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending (set by `/gsd-validate-phase`)
