---
phase: 40-notebook-isolation-attribution-page
verified: 2026-10-10T05:51:25Z
status: gaps_found
score: 79/80 must-haves verified (prohibitions 21/21)
covered_files:
  - ".planning/phases/40-notebook-isolation-attribution-page/40-01-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-01-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-02-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-02-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-03-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-03-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-04-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-04-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-05-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-05-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-06-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-06-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-07-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-07-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-UI-SPEC.md"
  - "CLAUDE.md"
  - "docs/notebooks/README.md"
  - "docs/notebooks/fomo_nb_utils.py"
  - "docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb"
  - "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb"
  - "docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb"
  - "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb"
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb"
  - "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb"
  - "docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb"
  - "docs/notebooks/pre_executed/telescope_runs_demo.ipynb"
  - "docs/runbooks/telescope_runs_calendar.rst"
  - "solsys_code/campaign_tables.py"
  - "solsys_code/campaign_views.py"
  - "solsys_code/tests/test_attribution_template.py"
  - "solsys_code/tests/test_bootstrap5_rendering.py"
  - "solsys_code/tests/test_campaign_attribution_views.py"
  - "solsys_code/tests/test_notebook_helper.py"
  - "solsys_code/tests/test_pre_executed_notebooks.py"
  - "solsys_code/tests/test_projector_demo_notebook.py"
  - "src/fomo/settings.py"
  - "src/templates/campaigns/attribution_queue.html"
  - "src/templates/campaigns/partials/attribution_return_state.html"
covered_digest: "v3:sha256:fcbe19ccff572c899820bf90bd63472a7209e9c3d93e00a6a514a124826b8fde"
behavior_unverified: 0
overrides_applied: 0
overrides:
  - must_have: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale"
    reason: >-
      Rule R12, extended in 40-08 [301bd93, f1b4bb6, c45edbc], reports in every notebook any reference to
      Django's database settings and connection objects (DATABASES, DATABASE_ROUTERS, settings_dict,
      connections, ConnectionHandler, create_connection, get_connection_params, get_new_connection,
      override_settings, modify_settings, DatabaseWrapper, load_backend, sqlite3, raw SQL ATTACH), to
      starting a process (subprocess, multiprocessing, pty, pexpect, ptyprocess, psutil, Jupyter kernel
      clients, os.system, os.popen, os.exec*, os.spawn*, os.posix_spawn*, os.fork, asyncio's
      create_subprocess_*, Popen, ProcessPoolExecutor, get_ipython and IPython's shell helpers), to the
      environment (os.environ, environb, putenv, unsetenv) and to running code from a string or another file
      (exec, eval, compile, __import__, __builtins__, importlib, runpy, builtins, ctypes, cffi, call_command
      of shell, shell_plus or runscript, execute_from_command_line), wherever the name appears, so binding
      one to another name is reported too; R0 reports shell escapes, process magics and manage.py, env -i or
      env -u in an unparseable cell. All four re-review forms and both round-1 probes are reported on a copy
      of a real notebook (40-08 must_haves). What stays out of reach is deliberate obfuscation that no
      static check of a notebook's own source can enumerate, exactly as the guard docstring names it: values
      assembled or looked up at runtime (a name, module, SQL statement or file path built from pieces, or
      fetched with getattr, globals(), vars() or sys.modules), code decoded at runtime, and code moved
      outside the notebook into a module it imports or a library function that starts a process or writes a
      file under a name these lists do not contain.
    accepted_by: "Tim Lister"
    accepted_at: "2026-10-10T13:46:17Z"
re_verification:
  previous_status: gaps_found
  previous_score: 49/50 (prohibitions 7/8)
  gaps_closed:
    - "40-01 prohibition (test tier) / review CR-01: the helper now checks the settings before django.setup(); a refusal or a later setup failure leaves no database reachable (TestHelperRefusalLeavesNoDatabase, 3 child-process tests, green)"
  gaps_remaining:
    - "ROADMAP SC 2 (first half) / WARN-02: narrowed but still open. Both round-1 probes are now caught by rule R12; four further plain-source re-route forms (re-review WR-01) are not"
  regressions: []
gaps:
  - truth: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale (re-review WR-01)"
    status: partial
    reason: >-
      Rule R12 (40-06) catches both round-1 probes and every spelling it lists, but it is a deny-list of names.
      The verifier appended each of the re-review's forms to the setup cell of a copy of the real
      import_campaign_csv_demo.ipynb (committed output kept) and isolation_problems() returned [] for all four:
      (a) connections['default'] = ConnectionHandler({...'NAME': dev})['default'], with the developer path built
      by string concatenation; (b) subprocess.run([sys.executable, 'manage.py', 'migrate'], env={'PATH': ...});
      (c) os.environ.clear() then subprocess.run(... 'manage.py', 'flush', '--noinput'); (d) a magic cell
      `!cd ../.. && env -i PATH=/usr/bin python manage.py flush --noinput` (R0 has no manage.py / env -i text).
      Controls (a literal 'fomo_db.sqlite3', a settings_dict store) are caught. The verifier also ran form (a)
      for real after a successful setup with NAME pointed at a scratchpad file: Group.objects.create() wrote its
      row there, and the teardown still printed "The developer database (src/fomo_db.sqlite3) was never opened
      for writing." Forms (b)-(d) send a child Django to the developer database because FOMO_DATABASE_PATH is no
      longer inherited -- the exact harm the phase exists to prevent. The module docstring's "what the guard does
      not see" list names none of these forms, so the 40-06 claim that R12 "covers every database re-routing
      spelling" is too strong. A static deny-list cannot be made complete; this is the second round in which
      readable source has got through, so see the escalation note in the Gaps Summary.
    artifacts:
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "R12 (lines 58-74, 361-393) has no entry for ConnectionHandler, a store into connections[...], or starting a process; R0's UNPARSEABLE_SUSPECT_TEXT (lines 118-125) lacks 'manage.py' and 'env -i'; the docstring's blind-spot list (lines 32-34) omits these forms"
      - path: "docs/notebooks/fomo_nb_utils.py"
        issue: "line 364 prints an unmeasured 'never opened for writing' claim that the verifier observed after a real re-route (kept by developer decision on round-1 WR-02)"
    missing:
      - "Either extend R12/R0 (ConnectionHandler and create_connection names; any Assign/AugAssign/Delete whose target is a Subscript of `connections`; any import or call of subprocess / os.system / os.popen / os.exec* / os.spawn* / Popen, and os.environ.clear / os.environ.pop / del os.environ[...]; 'manage.py', 'env -i', 'env -u' in UNPARSEABLE_SUSPECT_TEXT) with one synthetic test per form, and re-run test_every_real_pre_executed_notebook_is_isolated -- or accept the residual risk through the override below"
      - "In either case, make the guard docstring's 'does not see' list name what is still out of reach, and consider rewording the teardown's second line to what the helper actually checked (e.g. that Django's only configured database was the scratch file)"
      - "(non-blocking, same files, recommended in the same plan) re-review WR-02: in _disable_django_databases() switch every alias to the dummy backend before connections.close_all(), make the close best-effort, and re-raise the original error; add a child test with a settings module that has no 'default' alias asserting RuntimeError"
human_verification:
  - test: "Read the notebook prose against the committed output in all eight pre-executed notebooks, now including the re-review IN-02 specifics: reconcile_campaign_runs_demo code cells 2c723427 and 48b9f602 (and their sibling) print 'expected -- pk=334 plus the duplicate-identity group' while the output beside them reports only duplicate_identity=3 and the BEFORE table shows blank_url: 0; project_observation_calendar_demo cell 35debc54's requirement table cites headings 'Per-corpus reconciliation' and 'Per-marker tally' that do not exist (the real section is 'The whole real corpus, not a sample')"
    expected: "Each claim matches the output beside it; decide whether the NF-14 label and the requirement table are fixed now or sent to Phase 41 triage (both predate Phase 40, and 40-07 barred code-cell edits)"
    why_human: "Prose accuracy is a reading judgment no test asserts (40-07 D3, 40-05 D7 flag human_judgment: true)"
  - test: "Read docs/notebooks/README.md's helper section (lines ~60-80) and the guard module docstring (solsys_code/tests/test_pre_executed_notebooks.py lines 1-35)"
    expected: "Clear and honest. Note re-review IN-01: 'after a refusal, or any later failure in the setup, no cell can reach a database' and 'where a setup already failed, raises RuntimeError' overstate the case for failures before the settings are read (missing snapshot, unknown mode), where nothing was routed and a retry is allowed. The docstring's blind-spot list depends on the WR-01 decision"
    why_human: "Documentation wording; tests pin sentences, not meaning (40-05 D7, 40-06 D5 human_judgment: true)"
  - test: "Read docs/runbooks/telescope_runs_calendar.rst lines ~935-990 (attribution pass: green bar, 'The page keeps your place', last-page landing, Confirmed landing)"
    expected: "Accurate and clear; 'each table's page and sort' should not read as covering the two worklists, whose event_page/record_page are not kept (round-1 review IN-03, still open in the ledger)"
    why_human: "Documentation wording; carried forward unchanged from the previous report (no file in this area changed)"
---

# Phase 40: Notebook Isolation & Attribution Page Verification Report

**Phase Goal:** Every pre-executed demo notebook builds and uses its own scratch database, guarded by a check that catches one that does not, and the staff attribution page renders and pages correctly under Bootstrap 5.3 -- closing the remaining six Phase 37.1 review warnings (WR-13 to WR-18).
**Verified:** 2026-10-10T05:51:25Z (HEAD b6be648; last source change faacbfe)
**Status:** gaps_found
**Re-verification:** Yes -- after gap-closure plans 40-05, 40-06 and 40-07

## Summary

Gap 1 (CR-01) is closed. The helper now reads and checks Django's settings before `django.setup()`. After a
refusal, `apps.ready` is False and every alias is on the dummy backend. After a later failure (a forced
migrate error), a raw cursor and the ORM both raise `ImproperlyConfigured`. Three child-process tests prove
this, and they pass in the verifier's run. The 40-01 test-tier prohibition is therefore VERIFIED.

Gap 2 (WR-01) is narrower but not closed. Rule R12 catches both of the previous round's probes. The verifier
reproduced all four forms from the re-review (a new `ConnectionHandler` wrapper stored into `connections[...]`,
and three ways of starting `manage.py` in a child process without the scratch `FOMO_DATABASE_PATH`), and the
guard returned `[]` for each one. One of them was also run for real: the ORM wrote to the redirected file, and
the teardown still printed "never opened for writing". ROADMAP SC 2's first half is still only partly met.
Because this is the second round in which readable source has got past a deny-list guard, the gap is
escalated: the developer should decide between one more extension and an override (see the Gaps Summary).

The attribution-page half (WARN-04, WARN-07) is unchanged since the previous report and still passes.
`git diff 0ad6991 HEAD` touches no attribution file, and the 111 non-functional tests in its three modules
are OK.

## Goal Achievement

### Observable Truths -- ROADMAP success criteria (the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | No notebook byte-copies or writes the dev DB; snapshot via read-only online backup, others fresh; three byte-copiers on the helper; all re-executed; README rule and mode lists | ✓ VERIFIED | Verifier audit of all eight at HEAD: counts 1..N, 0 error outputs, exact `Django ready: settings module='src.fomo.settings'` line (only the 40-05 helper prints it, which proves the regeneration ran on it), Resolved lines in `/tmp/fomo-notebook-db-*` with the right mode text (6 fresh, 2 snapshot), teardown last with output, no `/home/` in setup or teardown output. The refusal path is now fail-closed too (gap 1 closed) |
| SC2a | The guard fails a notebook whose current source routes Django to the dev DB, even with stale output | ✗ FAILED (partial) | Round-1 probes now caught (R12; `TestRerouteRuleOnARealNotebook` green). Four re-review forms return `[]` on a copy of the real import_campaign_csv_demo (verifier probe). See gap |
| SC2b | Every notebook's DB-path check raises an ordinary exception that fires under `python -O` | ✓ VERIFIED | `check_database_settings` / `check_resolved_database` raise RuntimeError; `test_module_has_no_assert_statements`; `test_a_databases_override_is_refused_before_migrate` covers plain, `-O` and `PYTHONOPTIMIZE=1` (green) |
| SC3 | telescope_runs_demo reports the dip in one unit | ✓ VERIFIED | Unchanged; dip pin tests within the 82 OK |
| SC4 | High-band marker, template test, BS 5.3 names only | ✓ VERIFIED | No attribution file changed since the previous report; 111 tests OK (template, views, BS5 scans) |
| SC5 | Undo on a Dismissed row keeps the section open on the same page | ✓ VERIFIED | As above; the browser test passed in the previous run and in the orchestrator's full-suite gate at faacbfe (2305 + 40 OK) |

### Observable Truths -- gap-plan must_haves (40-05, 40-06, 40-07)

| # | Truth (abridged) | Status | Evidence |
|---|------------------|--------|----------|
| 5.1 | CR-01: override refused before setup; apps.ready False; every alias ImproperlyConfigured; other file never created; env restored; second setup refused | ✓ VERIFIED | `test_a_refused_setup_leaves_django_unable_to_reach_any_database` (green); code: check at fomo_nb_utils.py:295 before `django.setup()` at :300 |
| 5.2 | Failure after setup (forced migrate error) leaves cursor and ORM refused, no scratch dir | ✓ VERIFIED | `test_a_failure_after_django_setup_leaves_no_usable_database` |
| 5.3 | Up-front refusal never disables a running Django | ✓ VERIFIED | `test_setup_refuses_when_django_is_already_set_up`; the check sits outside the `try` (:275) |
| 5.4 | WR-02 (round 1): extra alias, routers, missing or empty default refused before setup | ✓ VERIFIED | `TestDatabaseSettingsCheck` (5) + `test_a_second_alias_or_a_router_is_refused_before_django_setup`. Note: a missing `default` in a real child raises ImproperlyConfigured, not RuntimeError (re-review WR-02, reproduced; still refused, nothing reachable) |
| 5.5 | IN-02: exact Django ready line, no repo path; `re.escape(tmp)` | ✓ VERIFIED | `test_fresh_setup_routes_migrates_and_tears_down`; all eight notebooks show the exact line |
| 5.6 | IN-07 helper parts: real snapshot subprocess test; failed snapshot removes dir | ✓ VERIFIED | `test_snapshot_setup_copies_the_developer_rows_migrates_and_tears_down`, `test_a_failed_snapshot_removes_the_scratch_directory` |
| 5.7 | IN-06: teardown refuses outside-temp and not-ours directories | ✓ VERIFIED | Two refusal tests; checks at :340-352 run before Django is imported |
| 5.8 | IN-01: env recorded and restored on failure and teardown | ✓ VERIFIED | `test_teardown_restores_the_routing_environment`; refusal probe asserts both vars None |
| 5.9 | README carries the setup-check sentence, D-04 sentence and mode lists | ✓ VERIFIED | Verifier whitespace-normalized match True; `TestReadmeStatesTheIsolationRule` green (wording: human item 2) |
| 5.10-5.12 | Edge adjacency / empty / ordering | ✓ VERIFIED | `test_refuses_a_second_alias_even_when_it_names_the_scratch_file`, `test_refuses_a_missing_or_empty_default`, `test_names_the_extra_alias_before_the_routers`; handler order disable, rmtree, restore, raise (:306-314) |
| 5.13 | Gate: 14 old helper tests kept; ruff clean; 3 full-hook commits | ✓ VERIFIED | No `def test_` removed (`git diff 0ad6991`); 27 helper tests; commits 222a25f, 7cfd78e, 819321e exist |
| 6.1 | settings_dict probe on a real notebook copy reported under R12 | ✓ VERIFIED | Test green; verifier control probe also caught |
| 6.2 | override_settings probe reported | ✓ VERIFIED | Test green |
| 6.3 | R12 runs on every notebook and matches the listed names, suffixes and sqlite3 imports | ✓ VERIFIED | test_pre_executed_notebooks.py:361-393; called before `is_django_backed` (:425-427) |
| 6.4 | Synthetic test per spelling | ✓ VERIFIED | `TestRerouteRuleOnSyntheticNotebooks` (9) green |
| 6.5 | All eight real notebooks pass with R12 | ✓ VERIFIED | `test_every_real_pre_executed_notebook_is_isolated` green |
| 6.6 | Docstring states what R0-R12 check and the four named blind spots | ✓ VERIFIED (literal) | Lines 22-34 match the must-have's wording; incomplete against the re-review forms (see gap) |
| 6.7 | REFERENCE_SETUP_OUTPUT updated; R0 list includes R12 names | ✓ VERIFIED | Lines 118-125 |
| 6.8 | Gate: no test removed, no skip/tag, EXPECTED_NOTEBOOKS unchanged | ✓ VERIFIED | `git diff 0ad6991` shows only added tests, no skip, tag or EXPECTED_NOTEBOOKS change |
| 7.1 | IN-05: eight NamedTemporaryFile calls pass `dir=scratch_db_dir` | ✓ VERIFIED | Verifier count 8/8 |
| 7.2 | All eight re-executed on the 40-05 helper (counts, Django ready line, /tmp Resolved, teardown last, no /home in helper output) | ✓ VERIFIED | Verifier audit table above |
| 7.3 | Each setup markdown carries the README setup-check sentence | ✓ VERIFIED | 8/8 normalized match |
| 7.4 | No code cell changed except load_telescope_runs' seven | ✓ VERIFIED | Verifier diff against 0ad6991: only a0c4592f, 448d1568, aa3de367, cc1cb4e1, 096760c2, b9c3b23d, 61e7eb42 |
| 7.5 | Snapshot notebooks: cell ids kept, projector test passes, cutover prose matches | ✓ VERIFIED | `test_projector_demo_notebook` in the 82 OK; the four listed markdown cells match (the NF-14 label mismatch is in code cells the plan barred editing: human item 1) |
| 7.6 | Home-path inventory | ✓ VERIFIED | Only the spiceypy stderr line in campaign_lifecycle_demo, as listed |
| 7.7 | Guard, helper and projector tests pass on the final tree | ✓ VERIFIED | Verifier: 82 tests OK |
| 7.8 | Ledger: seven fixed, IN-07 deferred in part | ✓ VERIFIED (historical) | True at 49a0530. The re-review gate (0d018aa, b6be648) then reused ids WR-01, WR-02 and IN-01..03 for new findings; the round-1 decisions are kept in the "ID reuse across rounds" table |
| 7.9 | Gate: notebook commits through the full hook, ledger with SKIP | ✓ VERIFIED | Commits 29490c3, 8beeb5c, 304022a, 49a0530 exist |

Previous plan truths 1.1-4.13 and UI F1/F3 (45 rows, all VERIFIED before) passed a regression check. Their
files are either unchanged since 0ad6991 (attribution) or superseded by the rows above (helper and guard).
None regressed.

**Score:** 79/80 truths verified (0 present but behavior-unverified). Every behavior-dependent truth (refusal
state, post-setup failure, environment restore, teardown refusal, redirect state) has a named child-process or
view test, and the verifier ran each one.

### Prohibitions

| Plan | Prohibition | Tier | Status | Evidence |
|------|-------------|------|--------|----------|
| 40-01 | MUST NOT write/byte-copy/move the dev DB from the helper, a notebook, a test or a step | test | ✓ VERIFIED (was the gap) | Refusal and failure paths are fail-closed (5.1, 5.2); `git status src/` is clean after every verifier probe |
| 40-01 | MUST NOT fall back silently without a dev DB in snapshot mode | test | ✓ VERIFIED | Unchanged test green |
| 40-02 | Guard not weakened | judgment | ✓ VERIFIED (judgment) | Only additions since 0ad6991 |
| 40-02 | No credential-shaped text | test | ✓ VERIFIED | Test green |
| 40-02 / 40-04 / 40-07 | No dev DB write while re-executing | test | ✓ VERIFIED | All Resolved lines in `/tmp/fomo-notebook-db-*`; snapshot through `mode=ro` |
| 40-03 / 40-04 | BS4 compat, HTMX/session state | judgment | ✓ VERIFIED (judgment) | Unchanged |
| 40-05 | Never disable a Django that was already running | test | ✓ VERIFIED | 5.3 |
| 40-05 | Public call shape and teardown lines unchanged | test | ✓ VERIFIED | Signatures and :363-364 unchanged; guard R10 green |
| 40-05 | Out-of-scope files untouched | test | ✓ VERIFIED | The 40-05 commits touch only the helper, its tests and the README |
| 40-05 | No existing helper test loosened | judgment | ✓ VERIFIED (judgment) | No removal; the two changed tests got stricter |
| 40-06 | Guard not weakened; no notebook/helper/README edit; R10 and teardown constants unchanged | test | ✓ VERIFIED | `git show --stat 50ffb59 26940ce`: guard module only |
| 40-07 | No `--allow-errors`, no error output, counts 1..N; no other code-cell edit; no credentials; IN-07 not plainly fixed | test | ✓ VERIFIED | Verifier audit; ledger ID-reuse table |

### Code review findings (40-REVIEW.md, gap round) -- explicit classification

| Finding | Reproduced? | Classification | Reasoning |
|---------|-------------|----------------|-----------|
| WR-01 R12 misses the ConnectionHandler and cleared-environment child-process forms | **Yes, all four** (guard returns `[]`; form (a) also run for real: ORM row written to the redirected scratchpad file, teardown printed "never opened for writing") | **Gap**, escalated for a developer decision | The ROADMAP SC 2 wording is the contract, and this is plain readable source doing what SC 2 forbids. Same class and same classification as round-1 WR-01. Not covered by a later phase. A deny-list cannot be completed, so an override narrowing SC 2 is a legitimate choice (template below) |
| WR-02 `_disable_django_databases()` closes before switching; no `default` alias masks RuntimeError | **Yes** (settings module with only a `primary` alias: `ImproperlyConfigured` raised with RuntimeError as `__context__`; `primary` left on sqlite3) | **Warning, not a gap** (fix with the gap plan) | The fail-closed outcome still holds in the reproduction: `apps.ready` False, both aliases refused, `django.setup()` and the ORM afterwards raise `ImproperlyConfigured`, env restored, scratch dir removed, `primary.sqlite3` never created. That is Django's own refusal, not the helper's, so the docstring and README describe the mechanism wrongly. The other trigger (`SynchronousOnlyOperation` in close) can only follow a passed check, so the alias left behind is the scratch file, never the developer database. No must-have is false |
| IN-01 README/notebook prose overstates "cannot re-run" for failures before settings are read | By reading (fomo_nb_utils.py:281 runs before the `try`) | **Acceptable / human item 2** | Harmless direction (nothing was routed, a retry is safe), and 40-05/40-07 required this sentence word for word. Wording fix optional |
| IN-02 reconcile NF-14 label "pk=334" contradicts output; projector requirement table cites missing headings | **Yes** (cells 2c723427, 48b9f602; 35debc54 vs the actual headings) | **Human decision** (human item 1) | Predates Phase 40, sits in code cells that 40-07 barred editing, and touches no isolation or attribution must-have. It is still published output that contradicts itself, so under the CLAUDE.md paired-docs rule the developer decides: fix now or Phase 41 triage |
| IN-03 two fail-closed branches untested | Partly: the verifier's first WR-02 probe hit the settings-import-failure branch by accident (`No module named 'src'`) and saw `apps.ready` False, `default` on dummy, env restored, no scratch dir | **Out of scope** (test hardening) | Behavior is correct when observed; no must-have names these branches |
| TDD RED evidence only as logs | Logs exist (`$HOME/tmp/phase40-0{5,6}-t*-red.log`) | **Acceptable** | The plans require one full-hook commit per task, and a failing RED commit cannot pass the django-test hook. No effect on the goal |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `docs/notebooks/fomo_nb_utils.py` | fail-closed setup, `check_database_settings`, teardown checks | ✓ VERIFIED | 364 lines; imported by 8 notebooks and both test modules (WR-02 warning) |
| `solsys_code/tests/test_notebook_helper.py` | refusal, post-setup failure, alias/router, snapshot, environment tests | ✓ VERIFIED | 27 tests, green |
| `solsys_code/tests/test_pre_executed_notebooks.py` | R12, real-notebook replay, synthetic tests, docstring | ✓ VERIFIED (coverage gap: WR-01) | 50 tests, green |
| `docs/notebooks/README.md` | check-before-setup description | ✓ VERIFIED | Sentence present |
| 8 notebooks | re-executed on the 40-05 helper | ✓ VERIFIED | Audit above |
| `40-REVIEW-DISPOSITION.md` | round recorded | ✓ VERIFIED | ID-reuse table preserves the round-1 decisions |
| Attribution artifacts (templates, views, tables, tests, runbook) | unchanged | ✓ VERIFIED | Regression run, 111 OK |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| `setup_scratch_django` | `settings.DATABASES` / `DATABASE_ROUTERS` | `check_database_settings(...)` at :295, before `django.setup()` at :300 | ✓ WIRED |
| setup `except` path | connections / DATABASES | `_disable_django_databases()` sets `django.db.backends.dummy` (:206) | ✓ WIRED (ordering weakness: WR-02) |
| `teardown_scratch_django` | `_active_setup` | recorded dir and environment (:347, :361) | ✓ WIRED |
| `isolation_problems` | `_database_rerouting_problems` | `problems.extend(...)` at :425, before the Django-backed return | ✓ WIRED |
| `TestRerouteRuleOnARealNotebook` | import_campaign_csv_demo.ipynb | `_append_to_setup_cell` deep copy | ✓ WIRED |
| notebook setup output | helper | exact `Django ready:` line in 8/8 | ✓ WIRED |
| load_telescope_runs schedule cells | teardown | `dir=scratch_db_dir` 8/8 | ✓ WIRED |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data | Status |
|----------|------|--------|-----------|--------|
| Notebook Resolved line | resolved NAME | lazy `settings.DATABASES['default']['NAME']`, checked before setup | Yes | ✓ FLOWING |
| Teardown second line | "never opened for writing" | hard-coded string, not measured | No | ⚠️ STATIC (developer-kept wording; shown false by the WR-01 reproduction) |
| Attribution page state | `dismissed_section_open`, return state | unchanged | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Helper, guard, projector-notebook modules | `python manage.py test --noinput ...test_notebook_helper ...test_pre_executed_notebooks ...test_projector_demo_notebook --parallel auto` | Ran 82 tests, OK | ✓ PASS |
| Attribution regression | `python manage.py test --noinput ...test_attribution_template ...test_campaign_attribution_views ...test_bootstrap5_rendering --exclude-tag functional --parallel auto` | Ran 111 tests, OK | ✓ PASS |
| Guard vs re-review forms (a)-(d) | `isolation_problems()` on copies of import_campaign_csv_demo with each form appended or inserted | `[]` x4; controls caught | ✗ FAIL (gap) |
| Re-route at runtime after setup | child: setup, `connections['default'] = ConnectionHandler({...scratchpad file})['default']`, migrate, `Group.objects.create` | Row written to redirected file; teardown printed "never opened for writing" | ✗ FAIL (gap evidence) |
| WR-02 missing `default` | child with `DATABASES = {'primary': ...}` | `ImproperlyConfigured` (RuntimeError as context); nothing reachable; env restored; no scratch dir | ⚠️ PASS on safety, wrong exception |
| Developer DB untouched | `git status --short src/` after all probes | clean | ✓ PASS |

All probes ran in the session scratchpad with `TMPDIR` pointed there. None set `FOMO_DATABASE_PATH` in the
shell or opened `src/fomo_db.sqlite3`.

### Probe Execution

Step 7c: SKIPPED. No `scripts/*/tests/probe-*.sh` exists, and no plan or summary declares one.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| WARN-02 | 40-02, 40-06, 40-07 | Guard fails a notebook whose current source routes Django to the dev DB | ✗ BLOCKED (partial) | R12 catches the round-1 forms; four re-review forms are not caught (gap) |
| WARN-03 | 40-01, 40-02, 40-05, 40-07 | No `assert` guards migrate; ordinary exception under `-O` | ✓ SATISFIED | SC2b; fail-closed refusal (gap 1 closed) |
| WARN-04 | 40-03, 40-04 | BS 5.3 only; High-band marker; template test | ✓ SATISFIED | SC4 |
| WARN-05 | 40-01, 40-02, 40-05, 40-07 | Audit, shared helper, no byte copy or write, README rule and lists | ✓ SATISFIED | SC1, 5.1-5.12, 7.1-7.6 |
| WARN-06 | 40-02 | Dip in one unit | ✓ SATISFIED | SC3 |
| WARN-07 | 40-04 | Undo keeps Dismissed open on the same page | ✓ SATISFIED | SC5 |

All six IDs appear in plan frontmatter and in REQUIREMENTS.md (rows 102-107, all mapped to Phase 40, all
currently "Gaps Found"). No requirement is orphaned.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| solsys_code/tests/test_pre_executed_notebooks.py | 32-34, 58-74, 118-125 | Deny-list with an incomplete stated blind-spot list | ⚠️ Warning (backs the gap) | WR-01 |
| docs/notebooks/fomo_nb_utils.py | 199-207 | Clean-up step that can throw before the fail-closed switch | ⚠️ Warning | WR-02 (no DB reachable in the observed cases) |
| docs/notebooks/fomo_nb_utils.py | 364 | Unmeasured "never opened for writing" claim | ⚠️ Warning | Kept by developer decision; shown false by the WR-01 runtime probe |
| reconcile_campaign_runs_demo.ipynb | 2c723427, 48b9f602 | Hard-coded "pk=334" label that contradicts its output | ℹ️ Info | Human item 1 |

No TBD, FIXME, XXX, TODO or HACK markers in lines added since 0ad6991.

### Human Verification Required

These do not change the gaps_found status. They are carried forward and updated for the developer.

1. **Notebook prose against output.** Read all eight notebooks, especially re-review IN-02:
   reconcile_campaign_runs_demo's NF-14 labels ("expected -- pk=334 ...") against `duplicate_identity=3`, and
   project_observation_calendar_demo cell 35debc54's two non-existent heading citations. Decide whether to fix
   them now or send them to Phase 41. Why human: prose accuracy.
2. **README helper section and guard docstring.** Check that they are clear and honest. Re-review IN-01
   overstates the "cannot re-run" rule for failures before the settings are read, and the docstring's
   blind-spot list depends on the WR-01 decision. Why human: wording.
3. **Runbook attribution-pass wording** (carried forward, unchanged). Lines ~935-990: "each table's page and
   sort" should not imply that the worklists' `event_page` and `record_page` are kept (round-1 IN-03). Why
   human: wording.

(The previous report's item 3, the round-1 WR-02 decision, is resolved. The developer chose to refuse extra
aliases and routers and to keep the teardown wording, and 40-05 implemented that.)

### Gaps Summary

One gap remains: the "guarded by a check that catches one that does not" clause of the goal, ROADMAP SC 2's
first half and WARN-02.

**Escalation (developer decision requested).** The static guard has now been extended once (R12), and a
second review found four more plain-source ways to reach the developer database. Two of them are worth noting
in particular. A child `manage.py` with a cleared environment needs no database name in the notebook at all.
A string-concatenated path already gets past R6. Each further extension is cheap, but none can make a
deny-list complete. The two honest options are:

1. **Extend once more**, accepting that the result is still not complete. Add the R12/R0 entries listed under
   the gap's `missing` (ConnectionHandler, stores into `connections[...]`, process-starting and
   environment-clearing calls, `manage.py` / `env -i` text), with a synthetic test for each. Make the docstring
   name what stays out of reach. Optionally fix WR-02 in the same plan, since it is the same file pair and a
   small change.
2. **Accept the residual risk with an override.** The reasoning would be: SC 2 is read as "the WR-13 bypass
   class plus every Django database-settings surface is caught", and a deliberate re-route through a new
   connection wrapper or a child process is out of reach for any static check. The docstring still has to be
   corrected so that it names these forms. To accept, add to this file's frontmatter:

```yaml
overrides:
  - must_have: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale"
    reason: "R12 catches the WR-13 class and every Django database-settings surface (both round-1 probes tested); re-routes through a new ConnectionHandler or a child process with a cleared environment are deliberate acts no static deny-list can enumerate; guard docstring updated to name them"
    accepted_by: "<name>"
    accepted_at: "<ISO timestamp>"
```

No later phase covers this. Phase 41 triages todos and Phase 42 re-verifies v2.4, so nothing is deferred.

---

_Verified: 2026-10-10T05:51:25Z_
_Verifier: Claude (gsd-verifier)_
