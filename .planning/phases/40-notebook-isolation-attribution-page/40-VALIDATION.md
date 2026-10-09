---
phase: "40"
slug: "notebook-isolation-attribution-page"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-10-09"
---

# Phase 40 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django test runner (`python manage.py test`), the project's only runner |
| **Config file** | none (settings in `src/fomo/settings.py`; test-only overrides under `sys.argv[1] == 'test'`) |
| **Quick run command** | `python manage.py test solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_notebook_helper solsys_code.tests.test_projector_demo_notebook solsys_code.tests.test_attribution_template solsys_code.tests.test_campaign_attribution_views solsys_code.tests.test_bootstrap5_rendering.TestTemplatesUseBootstrap5DataAttributes` |
| **Full suite command** | `python manage.py test --exclude-tag functional --exclude-tag ephemeris_segfault --exclude-tag migration --parallel` |
| **Browser tests** | `python manage.py test solsys_code.tests.test_bootstrap5_rendering --tag functional` |
| **Estimated runtime** | ~15 seconds (quick), ~80 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run the quick run command
- **After every plan wave:** Run the full suite command
- **Before `/gsd-verify-work`:** Full suite green, plus `--tag functional` for the Bootstrap file, plus a fresh `jupyter nbconvert --to notebook --execute --inplace` of every converted notebook (execution counts 1..N, no error output)
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

Filled in by the planner and executor; requirement-level map from RESEARCH.md:

| Requirement | Behavior | Test Type | Automated Command | File Exists | Status |
|-------------|----------|-----------|-------------------|-------------|--------|
| WARN-02 | Guard flags own `FOMO_DATABASE_PATH` assignment with stale output, aliased `django.setup`, setup inside `def`/`if False:`, missing/late helper call, `shutil.copy*`, non-contiguous execution counts; all eight real notebooks pass; not vacuous | unit (synthetic notebooks) | `python manage.py test solsys_code.tests.test_pre_executed_notebooks` | rewrite existing | ⬜ pending |
| WARN-03 | Helper raises `RuntimeError` (not `AssertionError`) when a `DATABASES` override wins, under `python -O` / `PYTHONOPTIMIZE=1`, before any migrate; helper has no `ast.Assert` | subprocess + static | `python manage.py test solsys_code.tests.test_notebook_helper` | ❌ W0 | ⬜ pending |
| WARN-05 | Fresh setup routes and migrates; snapshot uses `mode=ro` backup and leaves the source unchanged; missing source raises `RuntimeError` first | in-process + subprocess | `python manage.py test solsys_code.tests.test_notebook_helper` | ❌ W0 | ⬜ pending |
| WARN-06 | Committed `telescope_runs_demo` dip output in degrees; no committed output contains `arcmin deg` | unit | `python manage.py test solsys_code.tests.test_pre_executed_notebooks` | ❌ W0 add | ⬜ pending |
| WARN-04 | No removed Bootstrap 4 class in any repo-owned template; scans cover every `DIRS` entry | unit | `python manage.py test solsys_code.tests.test_bootstrap5_rendering.TestTemplatesUseBootstrap5DataAttributes` (+ new class) | extend | ⬜ pending |
| WARN-04 | High-band rows render `border-0 border-start border-4 border-success`, other rows do not | GET render test | `python manage.py test solsys_code.tests.test_attribution_template` | ❌ W0 add | ⬜ pending |
| WARN-04 | High-band row computed left border 4px green, other sides 0 | functional (Playwright) | `python manage.py test solsys_code.tests.test_bootstrap5_rendering --tag functional` | ❌ W0 add | ⬜ pending |
| WARN-07 | Undo carrying `dismissed-page` reopens Dismissed on the same page; confirmed Undo lands on Confirmed; clamp; all five actions keep band/page/sort; tampered values dropped | `TestCase` (POST + follow) | `python manage.py test solsys_code.tests.test_campaign_attribution_views` | ❌ W0 add | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

### Task-level map (filled by the planner, 2026-10-09)

| Task | Requirement | Automated command (log under `$HOME/tmp`) | Wave 0 file | Status |
|------|-------------|--------------------------------------------|-------------|--------|
| 40-01 T1 (tracer) | WARN-05, WARN-03 | `python manage.py test --noinput solsys_code.tests.test_notebook_helper solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_projector_demo_notebook` + seed/notebook AST checks | `docs/notebooks/fomo_nb_utils.py`, `solsys_code/tests/test_notebook_helper.py` (created here) | ⬜ pending |
| 40-01 T2 | WARN-05 | notebook AST/output check for backfill, lifecycle, reconcile + the same three test modules | — | ⬜ pending |
| 40-02 T1 (tracer) | WARN-02, WARN-03, WARN-06 | `python manage.py test --noinput solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_notebook_helper solsys_code.tests.test_projector_demo_notebook solsys_code.tests.test_import_campaign_csv` + WR-13 bypass replay via `manage.py shell` | guard rewrite in place | ⬜ pending |
| 40-02 T2 | WARN-05, WARN-03 | `python manage.py test --noinput solsys_code.tests.test_pre_executed_notebooks` (README tests) + rule-sentence / assert-audit / ledger checks | — | ⬜ pending |
| 40-03 T1 (tracer) | WARN-04 | `python manage.py test --noinput solsys_code.tests.test_attribution_template solsys_code.tests.test_campaign_attribution_views.TestEvidenceColumns` + the two named functional tests | — | ⬜ pending |
| 40-03 T2 | WARN-04 | in-file class scan + `python manage.py test --noinput --parallel solsys_code.tests.test_campaign_views solsys_code.tests.test_campaign_approval solsys_code.tests.test_campaign_gap solsys_code.tests.test_campaign_attribution_views solsys_code.tests.test_attribution_template ...DataAttributes` | — | ⬜ pending |
| 40-03 T3 | WARN-04 (IN-12) | `python manage.py test --noinput solsys_code.tests.test_bootstrap5_rendering.TestTemplatesUseBootstrap5ClassNames solsys_code.tests.test_bootstrap5_rendering.TestTemplatesUseBootstrap5DataAttributes solsys_code.tests.test_campaign_approval solsys_code.tests.test_campaign_submission` | — | ⬜ pending |
| 40-04 T1 (tracer) | WARN-07 | `python manage.py test --noinput solsys_code.tests.test_campaign_attribution_views solsys_code.tests.test_attribution_template` + template boolean check | — | ⬜ pending |
| 40-04 T2 | WARN-07 | same two modules (all five actions, clamp, tampering, IN-15) + include check | `src/templates/campaigns/partials/attribution_return_state.html` (created here) | ⬜ pending |
| 40-04 T3 | WARN-07, WARN-04 | named functional Undo test + `solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_notebook_helper` + lifecycle/runbook/ledger checks | — | ⬜ pending |

The README-vs-notebook-modes item formerly listed under Manual-Only is automated by 40-02 T2
(`test_readme_lists_each_notebook_under_its_mode`).

---

## Wave 0 Requirements

- [ ] `docs/notebooks/fomo_nb_utils.py` — must exist before the helper tests can load it
- [ ] `solsys_code/tests/test_notebook_helper.py` — covers WARN-03, WARN-05 (new file)
- [ ] Guard rewrite in `solsys_code/tests/test_pre_executed_notebooks.py` — covers WARN-02, WARN-06
- [ ] Marker, class-guard and return-state tests — cover WARN-04, WARN-07

No framework install needed.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `docs/notebooks/README.md` names every snapshot/fresh notebook in the D-04 wording | WARN-05 | Prose review (unless the planner adds a README-vs-guard-list test) | Compare the README's lists with the notebooks' helper calls |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
