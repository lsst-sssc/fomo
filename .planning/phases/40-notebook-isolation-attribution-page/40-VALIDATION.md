---
phase: "40"
slug: "notebook-isolation-attribution-page"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: validated
nyquist_compliant: true
wave_0_complete: true
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
| WARN-02 | Guard flags own `FOMO_DATABASE_PATH` assignment with stale output, aliased `django.setup`, setup inside `def`/`if False:`, missing/late helper call, `shutil.copy*`, non-contiguous execution counts; all eight real notebooks pass; not vacuous | unit (synthetic notebooks) | `python manage.py test solsys_code.tests.test_pre_executed_notebooks` | ✅ | ✅ green |
| WARN-03 | Helper raises `RuntimeError` (not `AssertionError`) when a `DATABASES` override wins, under `python -O` / `PYTHONOPTIMIZE=1`, before any migrate; helper has no `ast.Assert` | subprocess + static | `python manage.py test solsys_code.tests.test_notebook_helper` | ✅ | ✅ green |
| WARN-05 | Fresh setup routes and migrates; snapshot uses `mode=ro` backup and leaves the source unchanged; missing source raises `RuntimeError` first | in-process + subprocess | `python manage.py test solsys_code.tests.test_notebook_helper` | ✅ | ✅ green |
| WARN-06 | Committed `telescope_runs_demo` dip output in degrees; no committed output contains `arcmin deg` | unit | `python manage.py test solsys_code.tests.test_pre_executed_notebooks` | ✅ | ✅ green |
| WARN-04 | No removed Bootstrap 4 class in any repo-owned template; scans cover every `DIRS` entry | unit | `python manage.py test solsys_code.tests.test_bootstrap5_rendering.TestTemplatesUseBootstrap5DataAttributes` (+ new class) | ✅ | ✅ green |
| WARN-04 | High-band rows render `border-0 border-start border-4 border-success`, other rows do not | GET render test | `python manage.py test solsys_code.tests.test_attribution_template` | ✅ | ✅ green |
| WARN-04 | High-band row computed left border 4px green, other sides 0 | functional (Playwright) | `python manage.py test solsys_code.tests.test_bootstrap5_rendering --tag functional` | ✅ | ✅ green |
| WARN-07 | Undo carrying `dismissed-page` reopens Dismissed on the same page; confirmed Undo lands on Confirmed; clamp; all five actions keep band/page/sort; tampered values dropped | `TestCase` (POST + follow) | `python manage.py test solsys_code.tests.test_campaign_attribution_views` | ✅ | ✅ green |
| WARN-03 | (gap round) Settings checked before `django.setup()`; a refusal or any later failure leaves `apps.ready` False and every alias on the dummy backend (raw cursor and ORM raise `ImproperlyConfigured`) | subprocess | `python manage.py test solsys_code.tests.test_notebook_helper` | ✅ | ✅ green |
| WARN-05 | (gap round) Second alias / `DATABASE_ROUTERS` refused before setup; `Django ready:` line carries no checkout path; teardown removes only the directory setup created and restores the routing environment | subprocess + in-process | `python manage.py test solsys_code.tests.test_notebook_helper` | ✅ | ✅ green |
| WARN-02 | (gap round) R12 reports any `DATABASES`/`settings_dict`/`override_settings`/`modify_settings`/router/connection-handler/helper-internal/sqlite3 use in any cell; all eight real notebooks pass | unit (synthetic notebooks) | `python manage.py test solsys_code.tests.test_pre_executed_notebooks` | ✅ | ✅ green |
| WARN-02 | (gap round 2) R12 bans whole capabilities -- process start, environment change, connection replacement, code from a string or file, raw SQL ATTACH -- and R0 reports shell escapes and `manage.py` / `env -i` / `env -u` text; the four re-review forms are reported on a real-notebook copy; all eight real notebooks pass; residual accepted by override | unit (synthetic + real-notebook copies) | `python manage.py test solsys_code.tests.test_pre_executed_notebooks` | ✅ | ✅ green |
| WARN-03 | (gap round 2) Fail-closed step switches every alias to the dummy backend before a best-effort close and keeps the helper's own `RuntimeError` (no default alias, failing close, failing disable step) | subprocess | `python manage.py test solsys_code.tests.test_notebook_helper` | ✅ | ✅ green |
| WARN-02 | (gap round 4) R12 matches every listed name at every identifier, in every '.'/':' part of a string or bytes constant and in string code; R6 covers copytree/copyfileobj/move/write_bytes in any spelling; R0 is an allow-list (Python IPython runs unchanged apart from `%matplotlib inline` and a first-line `%%time`/`%%capture`) plus an automagic rule for one-line cells; every third-review and verifier-probe form is reported on a real-notebook copy; all eight real notebooks pass | unit (synthetic + real-notebook copies) | `python manage.py test solsys_code.tests.test_pre_executed_notebooks` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

### Task-level map (filled by the planner, 2026-10-09)

| Task | Requirement | Automated command (log under `$HOME/tmp`) | Wave 0 file | Status |
|------|-------------|--------------------------------------------|-------------|--------|
| 40-01 T1 (tracer) | WARN-05, WARN-03 | `python manage.py test --noinput solsys_code.tests.test_notebook_helper solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_projector_demo_notebook` + seed/notebook AST checks | `docs/notebooks/fomo_nb_utils.py`, `solsys_code/tests/test_notebook_helper.py` (created here) | ✅ green |
| 40-01 T2 | WARN-05 | notebook AST/output check for backfill, lifecycle, reconcile + the same three test modules | — | ✅ green |
| 40-02 T1 (tracer) | WARN-02, WARN-03, WARN-06 | `python manage.py test --noinput solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_notebook_helper solsys_code.tests.test_projector_demo_notebook solsys_code.tests.test_import_campaign_csv` + WR-13 bypass replay via `manage.py shell` | guard rewrite in place | ✅ green |
| 40-02 T2 | WARN-05, WARN-03 | `python manage.py test --noinput solsys_code.tests.test_pre_executed_notebooks` (README tests) + rule-sentence / assert-audit / ledger checks | — | ✅ green |
| 40-03 T1 (tracer) | WARN-04 | `python manage.py test --noinput solsys_code.tests.test_attribution_template solsys_code.tests.test_campaign_attribution_views.TestEvidenceColumns` + the two named functional tests | — | ✅ green |
| 40-03 T2 | WARN-04 | in-file class scan + `python manage.py test --noinput --parallel solsys_code.tests.test_campaign_views solsys_code.tests.test_campaign_approval solsys_code.tests.test_campaign_gap solsys_code.tests.test_campaign_attribution_views solsys_code.tests.test_attribution_template ...DataAttributes` | — | ✅ green |
| 40-03 T3 | WARN-04 (IN-12) | `python manage.py test --noinput solsys_code.tests.test_bootstrap5_rendering.TestTemplatesUseBootstrap5ClassNames solsys_code.tests.test_bootstrap5_rendering.TestTemplatesUseBootstrap5DataAttributes solsys_code.tests.test_campaign_approval solsys_code.tests.test_campaign_submission` | — | ✅ green |
| 40-04 T1 (tracer) | WARN-07 | `python manage.py test --noinput solsys_code.tests.test_campaign_attribution_views solsys_code.tests.test_attribution_template` + template boolean check | — | ✅ green |
| 40-04 T2 | WARN-07 | same two modules (all five actions, clamp, tampering, IN-15) + include check | `src/templates/campaigns/partials/attribution_return_state.html` (created here) | ✅ green |
| 40-04 T3 | WARN-07, WARN-04 | named functional Undo test + `solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_notebook_helper` + lifecycle/runbook/ledger checks | — | ✅ green |
| 40-05 T1 (tracer, gap) | WARN-03, WARN-05 (CR-01) | `solsys_code.tests.test_notebook_helper solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_projector_demo_notebook` (child-process refusal and forced-migrate-failure probes) | — | ✅ green |
| 40-05 T2 (gap) | WARN-05 (WR-02, IN-02, IN-07) | `solsys_code.tests.test_notebook_helper solsys_code.tests.test_pre_executed_notebooks` (alias/router refusal, evidence line, snapshot + failed snapshot) | — | ✅ green |
| 40-05 T3 (gap) | WARN-05 (IN-06, IN-01) | `solsys_code.tests.test_notebook_helper solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_projector_demo_notebook` (teardown scope, environment restore) + README check | — | ✅ green |
| 40-06 T1 (tracer, gap) | WARN-02 (WR-01) | `solsys_code.tests.test_pre_executed_notebooks` (settings_dict / override_settings re-route replayed on a real notebook copy) | — | ✅ green |
| 40-06 T2 (gap) | WARN-02 (WR-01) | `solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_projector_demo_notebook` (one synthetic test per R12 form) | — | ✅ green |
| 40-07 T1 (tracer, gap) | WARN-05 (IN-05, IN-02) | `solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_notebook_helper solsys_code.tests.test_projector_demo_notebook` + load_telescope_runs_demo output checks | — | ✅ green |
| 40-07 T2 (gap) | WARN-05, WARN-03 | `solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_notebook_helper` + five fresh notebooks' output/prose checks | — | ✅ green |
| 40-07 T3 (gap) | WARN-05, WARN-03, WARN-02 | `solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_notebook_helper solsys_code.tests.test_projector_demo_notebook` + snapshot notebooks, home-path inventory, ledger checks | — | ✅ green |
| 40-08 T1 (tracer, gap) | WARN-02 (WR-01) | `solsys_code.tests.test_pre_executed_notebooks` (four re-review forms replayed on a real-notebook copy; RED log `phase40-08-t1-red.log`) | — | ✅ green |
| 40-08 T2 (gap) | WARN-02 (WR-01) | `solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_projector_demo_notebook` (one synthetic test per capability form; RED log `phase40-08-t2-red.log`) | — | ✅ green |
| 40-08 T3 (gap) | WARN-03, WARN-05 (WR-02, IN-01) | `solsys_code.tests.test_notebook_helper solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_projector_demo_notebook` (exception-safe fail-closed child tests; RED log `phase40-08-t3-red.log`) + README check | — | ✅ green |
| 40-09 T1 (tracer, gap) | WARN-05, WARN-02, WARN-03 | `solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_notebook_helper` + telescope_runs_demo output/prose checks | — | ✅ green |
| 40-09 T2 (gap) | WARN-05, WARN-03 | `solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_notebook_helper` + five fresh notebooks' output/prose checks | — | ✅ green |
| 40-09 T3 (gap) | WARN-05, WARN-02, WARN-03 (IN-02) | `solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_notebook_helper solsys_code.tests.test_projector_demo_notebook` + snapshot notebooks, home-path inventory, ledger and override checks | — | ✅ green |
| 40-10 T1 (tracer, gap) | WARN-02 (CR-01 a) | `solsys_code.tests.test_pre_executed_notebooks` (R12 at every identifier; `from webbrowser import subprocess` on a real-notebook copy; RED log `phase40-10-t1-red.log`) | — | ✅ green |
| 40-10 T2 (gap) | WARN-02, WARN-05 (CR-01 b-e, WR-01) | `solsys_code.tests.test_pre_executed_notebooks` (R12 string/bytes parts and string code, new lookup/process/connection names, ATTACH (; R6 copytree/copyfileobj/move/write_bytes; RED log `phase40-10-t2-red.log`) | — | ✅ green |
| 40-10 T3 (gap) | WARN-02 (CR-02) | `solsys_code.tests.test_pre_executed_notebooks` (R0 allow-list over IPython's TransformerManager + automagic rule; alias/magic forms on a real-notebook copy; RED log `phase40-10-t3-red.log`) + ledger check | — | ✅ green |
| 40-11 T1 (gap) | WARN-02, WARN-03 (CR-01 docs, IN-01) | `solsys_code.tests.test_pre_executed_notebooks` `solsys_code.tests.test_notebook_helper solsys_code.tests.test_projector_demo_notebook` + docstring/README/helper-docstring checks (code unchanged with docstrings stripped) | — | ✅ green |
| 40-11 T2 (gap) | WARN-05 (WR-02) | `solsys_code.tests.test_pre_executed_notebooks` + nbformat cell-id check (six markdown sentences; code cells and outputs identical to 1930f31; widened 'never opened' check clean in all eight notebooks) + ledger check | — | ✅ green |
| 40-12 T1-T3 (gap, checkpoint) | WARN-02 (CR-01, CR-02 override) | draft check (every reason claim backed by a named guard test or docstring phrase), YAML load of the edited front matter, file-set check (only 40-VERIFICATION.md changed); developer approved 2026-10-10T19:14:57Z | — | ✅ green |
| 40-13 T1 (tracer, gap) | WARN-02 (CR-01) | `solsys_code.tests.test_pre_executed_notebooks` (R0 automagic rule reports a one-line magic or alias cell whatever an earlier cell binds; seven bindings of rm on a real-notebook copy; RED log `phase40-13-t1-red.log`) | — | ✅ green |
| 40-13 T2 (gap) | WARN-02, WARN-05 (WR-01, WR-02) | `solsys_code.tests.test_pre_executed_notebooks` (R6 distinctive copy names at every identifier, shutil names followed across cells and string code, look-alikes clean; R5/R7 bytes constants; RED log `phase40-13-t2-red.log`) | — | ✅ green |
| 40-13 T3 (gap) | WARN-05 (IN-02) | pyproject check (exactly one added line `ipykernel>=7` in the dev extra; validate-pyproject hook) + workflow/CLAUDE.md dev-extra greps | — | ✅ green |
| 40-14 T1-T3 (gap, checkpoint) | WARN-02, WARN-05 (CR-01, WR-01, WR-02 docs; IN-02 ledger) | draft check (every docstring/README/reason claim proven against 40-13's committed guard by inert calls; named R6 boundary unreported), docstring splice keeps the AST identical, README outside the guard paragraph equal to e08f8eb, ledger 10-line check, YAML load + file-set check of the override edit; developer approved 2026-10-10T23:20:47Z | — | ✅ green |
| 40-15 T1 (tracer, gap, tdd) | WARN-02 (round-5 CR-01, CR-02) | `solsys_code.tests.test_pre_executed_notebooks` (R6 reports shutil, _shutil and write_bytes at every identifier and string part; six shutil hand-on forms and the class-pattern write_bytes form on a real-notebook copy; RED log `phase40-15-t1-red.log`) | — | ✅ green |
| 40-15 T2 (gap, tdd) | WARN-02 (round-5 WR-01, WR-02) | `solsys_code.tests.test_pre_executed_notebooks` (IPyAutocall is an R12 name; autocall of exit and quit and the raw-source Django-backed gate pinned; RED log `phase40-15-t2-red.log`) | — | ✅ green |
| 40-16 T1-T3 (gap, checkpoint) | WARN-02, WARN-05 (round-5 CR-01, CR-02, WR-01, WR-02 docs; ledger; override) | draft check (8-sentence reason, 77 evidence rows proven against 40-15's committed guard by inert calls), docstring/README check, ledger check, override key comparison; developer approval 2026-10-11T03:35:54Z | — | ✅ green |

The README-vs-notebook-modes item formerly listed under Manual-Only is automated by 40-02 T2
(`test_readme_lists_each_notebook_under_its_mode`).

---

## Wave 0 Requirements

- [x] `docs/notebooks/fomo_nb_utils.py` — must exist before the helper tests can load it
- [x] `solsys_code/tests/test_notebook_helper.py` — covers WARN-03, WARN-05 (new file)
- [x] Guard rewrite in `solsys_code/tests/test_pre_executed_notebooks.py` — covers WARN-02, WARN-06
- [x] Marker, class-guard and return-state tests — cover WARN-04, WARN-07

No framework install needed.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| None remaining — the README-vs-notebook-modes check is automated by `test_readme_lists_each_notebook_under_its_mode` (40-02 T2) | — | — | — |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 90s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-10-09 (validate-phase State A audit: 0 gaps; 552 tests across the phase-40 modules OK incl. functional; configured full suite 2280 + 40 OK); re-approved 2026-10-10 after gap plans 40-05..40-07 (State A: 8 task rows added, 0 gaps; configured full suite 2305 + 40 OK at faacbfe); re-approved 2026-10-10 after gap plans 40-08..40-09 (State A: 6 task rows and 2 requirement rows added, 0 test gaps; configured full suite 2321 + 40 OK at 7e1fdad; the guard-coverage defects in 40-REVIEW.md CR-01/CR-02 are verification gaps, not missing tests); re-approved 2026-10-10 after gap plans 40-10..40-12 (State A: 6 task rows and 1 requirement row added, 0 test gaps; guard module 63 -> 82 tests; configured full suite 2340 + 40 OK at d688504); re-approved 2026-10-10 after gap plans 40-13..40-14 (State A: 4 task rows added, 0 test gaps; guard module 82 -> 86 tests; configured full suite 2344 + 40 OK at 51da348; the R6 routes in round-5 40-REVIEW.md CR-01/CR-02 are verification gaps, not missing tests); re-approved 2026-10-11 after gap plans 40-15..40-16 (State A: 3 task rows added, all green; guard module 91 tests OK; configured full suite 2349 + 40 OK on ce82278 with one load-induced Playwright timeout in test_bootstrap5_rendering that passed 22/22 on re-run, no surface changed)
## Validation Audit 2026-10-10

| Metric | Count |
|---|---|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

## Validation Audit 2026-10-10 (gap round 40-05..40-07)

| Metric | Count |
|---|---|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
