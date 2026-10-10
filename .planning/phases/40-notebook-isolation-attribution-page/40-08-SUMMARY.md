---
phase: 40-notebook-isolation-attribution-page
plan: 08
subsystem: testing
tags: [notebooks, isolation-guard, ast, scratch-database, fail-closed, gap-closure]
gap_closure: true
gap_ids: [WR-01, WR-02, IN-01]

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-05/40-06/40-07 helper, guard rule R12 and README that the 2026-10-10 re-review and the verification gap reopened"
provides:
  - "Guard rule R12 extended from database names to whole capabilities (database objects, process start, environment, code from a string or file, raw SQL ATTACH) matched wherever a name appears"
  - "Guard rule R0 reports IPython shell escapes, process magics, manage.py, env -i and env -u in an unparseable cell"
  - "Real-notebook replay of all four re-review forms plus a synthetic test per capability form"
  - "Exception-safe fail-closed step in the notebook helper: switch every alias to the dummy backend, then best-effort close, original error always re-raised"
  - "Teardown second line reworded to what the setup checked; README refusal wording narrowed to failures once Django's settings are read"
affects: [40-09, phase-41]

tech-stack:
  added: []
  patterns:
    - "Ban references, not calls: each capability name is matched as a name, attribute, keyword, imported name or exact string, so aliasing is reported too"
    - "Fail-closed step orders the irreversible protection (switch aliases to dummy) before the fallible clean-up (close) and logs, never raises, clean-up errors"

key-files:
  created: []
  modified:
    - solsys_code/tests/test_pre_executed_notebooks.py
    - docs/notebooks/fomo_nb_utils.py
    - solsys_code/tests/test_notebook_helper.py
    - docs/notebooks/README.md
    - .planning/phases/40-notebook-isolation-attribution-page/40-REVIEW-DISPOSITION.md

key-decisions:
  - "D-G1: extend the guard to whole capabilities, then override only the residual (the override is recorded by 40-09)"
  - "D-G2: switch every alias to the dummy backend before a best-effort connections.close_all(); re-raise the setup's own error; teardown line says what setup checked (supersedes the 2026-10-09 decision and the 40-05 prohibition)"
  - "D-G3: the 'no database reachable / cannot be re-run' wording covers only failures once the helper has started reading Django's settings"
  - "The 40-06 claim that R12 covered every re-routing spelling is withdrawn; the guard docstring makes no completeness claim and names the residual"

patterns-established:
  - "R12 name lists are pairwise disjoint and R12_ALL_NAMES feeds R0's whole-word search of unparseable cells, so the two rules cannot drift"

requirements-completed: [WARN-02, WARN-03, WARN-05]

actuals:
  tokens: 12000   # chars/4 over the added and removed lines of the plan's four commits (48223 chars)
  tasks: 3
  commits: 4
plan_head_before: ca593bfa7c6d07d38861e33c7c80b0dfac34baa4
plan_head_after: 7ba970e1b4f55872c376b2d88c5d8eb61abfe9b9

coverage:
  - id: D1
    description: "The four re-review forms (ConnectionHandler store, child manage.py with a bare environment, cleared environment plus child manage.py, shell escape running manage.py) are reported on a copy of a real notebook"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestReReviewFormsOnARealNotebook"
        status: pass
    human_judgment: false
  - id: D2
    description: "R12 bans whole capabilities (process start, environment, connection replacement, code from a string or file, raw SQL ATTACH) with a synthetic test per form; look-alikes stay allowed; all eight real notebooks stay clean"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestCapabilityRulesOnSyntheticNotebooks"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestPreExecutedNotebooksUseTheHelper.test_every_real_pre_executed_notebook_is_isolated"
        status: pass
    human_judgment: false
  - id: D3
    description: "The helper's fail-closed step is exception-safe: no default alias gives RuntimeError, a failing close or disable step never replaces the original error and every alias ends on the dummy backend"
    requirement: WARN-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestHelperFailClosedIsExceptionSafe"
        status: pass
    human_judgment: false
  - id: D4
    description: "Teardown second line states what setup checked; README states S1, S2 and S3 and the new guard capabilities"
    requirement: WARN-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestHelperRunsForReal.test_fresh_setup_routes_migrates_and_tears_down"
        status: pass
    human_judgment: true
    rationale: "README and guard-docstring wording is read at /gsd-verify-work (Task 3 human-check)"

duration: 12 min
completed: 2026-10-10
status: complete
---

# Phase 40 Plan 08: Guard capabilities, exception-safe helper, README wording Summary

**The notebook guard now bans whole capabilities (a process, the environment, a new connection, code from a string, an attached database) instead of a deny-list of database names, and the helper's fail-closed step switches every alias to the dummy backend before a best-effort close so its own error is always the one raised.**

## Commits

| Task | Commit | Closes | What |
|------|--------|--------|------|
| 1 (tracer) | `301bd93` | WR-01 | R12 learns connections, ConnectionHandler, create_connection, subprocess, environ; R0 learns shell escapes and manage.py / env -i / env -u; `TestReReviewFormsOnARealNotebook` replays the four re-review forms |
| 2 | `f1b4bb6` | WR-01 | Seven disjoint name lists, call_command('shell'...), raw SQL ATTACH, `TestCapabilityRulesOnSyntheticNotebooks` (nine tests), docstring naming the residual |
| 3 | `c45edbc` | WR-02, IN-01 | Exception-safe `_disable_django_databases` and failure handler, new teardown line, `TestHelperFailClosedIsExceptionSafe`, README S1-S3 |
| ledger | `7ba970e` | WR-01, WR-02 | 40-REVIEW-DISPOSITION.md records both fixed, `open: 5` (SKIP=django-test, `.planning`-only) |

The three code commits went through the full pre-commit hook (ruff, ruff-format, django-test). The hook ran to completion each time on a host with load average about 33; wall time was not measured individually. Nothing was pushed.

## What was built

- **Guard (`test_pre_executed_notebooks.py`):** `_database_rerouting_problems` collects name -> capability per parseable cell and reports one `references <name>; <reason> (R12: no re-routing after setup)` problem per (cell, name). Matching covers names, attributes, keywords, imported names and exact string constants; modules are matched by import, attribute or string; builtins (`exec`, `eval`, `compile`, `__import__`, `__builtins__`) never as attributes, so `re.compile` and a DataFrame's `eval` stay allowed; `call_command` of `shell`, `shell_plus` or `runscript`; string constants matching `ATTACH_SQL`. R0 additionally reports `SHELL_ESCAPE` matches, any R12 name as a whole word (`UNPARSEABLE_SUSPECT_NAMES`) and the text `manage.py`, `env -i`, `env -u`. The database problem text is byte-identical to before.
- **Helper (`fomo_nb_utils.py`):** `_disable_django_databases` = configure-if-needed, switch every alias's ENGINE and NAME in place, `connections.close_all()` inside `try/except Exception` that logs a warning, then drop the cached wrappers. `setup_scratch_django`'s failure handler wraps the disable step in `try/except Exception` (logged), keeps `finally: rmtree; _restore_environ`, and ends with a bare `raise`. The already-configured refusal now says a setup "already ran, or failed after reading Django's settings". Teardown's second line: `Before django.setup() ran, the helper checked that Django's only configured database was the scratch file in that directory.` The call shape, first teardown line and `TEARDOWN_LINE_PREFIX` are unchanged.
- **README:** S1, S2, S3 verbatim; the dummy-before-close sentence; the teardown-line sentence; the guard paragraph lists starting a process, changing the environment and running code built from a string and points to the module docstring for what a static check cannot see. The D-04 sentence and both mode lists are unchanged.

## TDD evidence

| Task | RED log | RED result | GREEN |
|------|---------|------------|-------|
| 1 | `$HOME/tmp/phase40-08-t1-red.log` | `Ran 4 tests`, `FAILED (failures=5)`: `FAIL:` for each of the four tests (form d twice, one per subTest) | guard module `Ran 54 tests ... OK` |
| 2 | `$HOME/tmp/phase40-08-t2-red.log` | `Ran 9 tests`, `FAILED (failures=59, errors=1)`: FAIL for starting-a-process (26), code-from-a-string (17), ipython (7), environment (3), ATTACH (3), connection replacement (2), shell escapes (1); ERROR for `test_r12_name_lists_are_disjoint_and_reach_r0` (name lists missing) | guard + projector `Ran 68 tests ... OK` |
| 3 | `$HOME/tmp/phase40-08-t3-red.log` | `Ran 30 tests`, `FAILED (failures=5)`: close-failure test (`RAISED OSError`, `WRITE_SUCCEEDED default` -- the real hazard), disable-failure test (`RAISED ValueError`), no-default test twice (`RAISED ImproperlyConfigured`, primary left on sqlite3), `test_fresh_setup_routes_migrates_and_tears_down` (old teardown line) | helper + guard + projector `Ran 98 tests ... OK` |

Independent checks (all printed their OK line): the importlib replay with no Django settings (forms a-c under R12 with the expected names, form d by R0 inserted and appended, eight real notebooks clean); the Task 2 structural check (`OK: five capabilities, seven disjoint name lists feeding R0, docstring names the residual, 63 tests (none removed or skipped), eight real notebooks clean`); the Task 3 structural check (`OK: switch-then-close order, logged best-effort close, guarded disable with a bare raise, new teardown line everywhere, README carries S1-S3 and unchanged rule and lists`); the ledger check (`OK: WR-01 and WR-02 fixed with real 40-08 commits touching their files, open 7 -> 5, exactly 5 lines changed`).

Final counts: guard module 63 tests, helper module 30 tests, projector evidence 5 tests.

## The 40-06 completeness claim is withdrawn

The 40-06 commit message and SUMMARY claimed R12 covered every re-routing spelling; those stay as history and are superseded. The guard docstring now makes no completeness claim and names exactly what stays out of reach:

> What the guard does not see: it reads only the notebook's own source and matches literal names and exact strings, so deliberate obfuscation is out of its reach. (1) Values assembled or looked up at runtime: a name, module, SQL statement or file path built from pieces (string concatenation, f-strings, joined path parts) or fetched with ``getattr``, ``globals()``, ``vars()`` or ``sys.modules`` using such a string, and code decoded at runtime (pickle, marshal, codecs). (2) Code outside the notebook: a module it imports, or a library function that starts a process or writes a file under a name these lists do not contain.

The old "object first bound to another name" blind spot is gone: binding a listed capability to another name is now reported. 40-09 records the developer's override of this residual as its last step.

## IN-03

The new child tests do not execute either branch review IN-03 names (the `if not django_settings.configured: configure(dummy)` branch inside `_disable_django_databases`, or the configured-but-not-ready refusal in `setup_scratch_django`): settings are configured in all three, and none calls a second setup. IN-03 stays open for Phase 41 triage, as do WR-03, IN-04 and IN-07's `campaign_views.py` tidy-up; round-1 IN-03's runbook text was not touched.

## Paired docs

`docs/notebooks/fomo_nb_utils.py` pairs with every notebook under `docs/notebooks/pre_executed/`. Its changed teardown line and refusal message reach the notebooks' committed output only when they are re-executed, which 40-09 does (wave 8, in its `files_modified`). No notebook and no `40-VERIFICATION.md` changed in this plan (checked against 488a7bb). No `docs/runbooks/` page documents the helper, its teardown line or its failure behavior, so no runbook changes.

## Deviations from Plan

None - plan executed exactly as written. Two small implementation notes, neither a deviation from behavior: (1) the Task 3 commit-file-set check in the plan sorts file names in the C locale (`README.md` before `fomo_nb_utils.py`); under this shell's default locale the order differs, and the check passes with `LC_ALL=C sort`. (2) `ruff` required wrapping two over-long lines in the new tests and the `dynamic code` reason; no behavior changed.

## Threat Flags

None. No new network endpoint, auth path or trust-boundary file access; the changes tighten T-40-28 to T-40-31, and T-40-32 (deliberate obfuscation) remains accepted and is named in the docstring.

## Known Stubs

None.

## Self-Check: PASSED

All modified files, the three RED logs and commits `301bd93`, `f1b4bb6`, `c45edbc` and `7ba970e` exist and are ancestors of HEAD.
