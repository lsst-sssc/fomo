---
phase: 40-notebook-isolation-attribution-page
plan: 10
subsystem: testing
tags: [notebooks, guard, ipython, automagic, gap-closure, scratch-database]
gap_closure: true
gap_ids: [CR-01, CR-02, WR-01]
status: complete

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-08/40-09 extended guard (R0 deny-list, R12 capability lists, R6 shutil.copy*) and the round-3 review (CR-01, CR-02, WR-01)"
provides:
  - "R12 matched at every identifier the parser records, in every '.'/':' part of a str or bytes constant, and in code inside strings (two levels)"
  - "New R12 names: connection, _connections, Database; getoutputerror, get_output_error_code, process_handler; import_string, resolve_name, locate, Macro; ATTACH_SQL also matches `ATTACH (`"
  - "R6 reports copytree, copyfileobj, move and write_bytes as calls, references, imports and string lookups"
  - "R0 is an allow-list over IPython's own TransformerManager plus an automagic rule built from IPython's own magic registry"
  - "Review ledger rows for CR-02 and WR-01 fixed (open 6 of 12)"
affects: [40-11, 40-12, verify-work]

tech-stack:
  added: []
  patterns:
    - "Guard reads a cell the way IPython does: static transform via TransformerManager, automagic via split_user_input and the live magic registry, so the guard and the kernel cannot disagree on what runs"
    - "Python views: every rule after R0 reads the cell with its harmless magic lines removed, so code under %%time/%%capture is checked like any other"

key-files:
  created: []
  modified:
    - solsys_code/tests/test_pre_executed_notebooks.py
    - .planning/phases/40-notebook-isolation-attribution-page/40-REVIEW-DISPOSITION.md

key-decisions:
  - "D-G5 (1): R12 matches every table at every identifier (names, attributes, import path parts, imported names and aliases, keywords, parameters, def/class/except names, global/nonlocal, match captures, class-pattern attributes, type parameters); attributes never against the builtins"
  - "A string counts as code only when it parses and holds a statement or a call; a bare expression such as 'subprocess-free' runs nothing and is matched by its dotted parts only (needed so the plan's own look-alike stays allowed)"
  - "D-G5 (2): harmless magic set is exactly `%matplotlib inline` and a first-line `%%time` (alone) or `%%capture` (--no-stderr/--no-stdout/--no-display and an output name); %config and other %matplotlib backends are not harmless"
  - "Automagic rule: a one-line cell is reported when its first word is an IPython line magic or alias not bound earlier in the notebook, not a builtin or keyword, and not followed by = or ,"
  - "R6's from-shutil import message now names the imported function(s), so `from shutil import copytree` is reported by name"

requirements-completed: [WARN-02, WARN-05]

duration: 16 min
completed: 2026-10-10

actuals:
  tokens: 13400   # chars/4 over the plan's four commits' diff (53633 chars incl. diff headers)
  tasks: 3
  commits: 4
plan_head_before: ba48dcf31365ccd68f6a1697044bab1fcfd9fa44
plan_head_after: f4f2cd1b7deb9f5ae1ff63a1a5ce632678d786b1

coverage:
  - id: D1
    description: "R12 matches every listed module and name at every identifier; `from webbrowser import subprocess` and `from django.utils.autoreload import subprocess as sp` reported on a real notebook (CR-01 a)"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestThirdReviewFormsOnARealNotebook.test_subprocess_imported_from_another_module_is_reported_on_a_real_notebook"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR12MatchesEveryIdentifierAndString.test_a_listed_name_is_reported_wherever_a_name_appears"
        status: pass
      - kind: command
        ref: "Task 1 independent replay"
        status: pass
    human_judgment: false
  - id: D2
    description: "R12 reads strings, bytes and string code; lookup functions, IPython helpers, the connection proxy and ATTACH ( reported on a real notebook (CR-01 b-e)"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestThirdReviewFormsOnARealNotebook.test_a_process_looked_up_by_a_literal_string_is_reported_on_a_real_notebook"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestThirdReviewFormsOnARealNotebook.test_the_connection_proxy_database_connect_and_attach_are_reported_on_a_real_notebook"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR12MatchesEveryIdentifierAndString.test_code_inside_a_string_is_checked_like_code"
        status: pass
      - kind: command
        ref: "Task 2 independent replay (name lists exactly base plus decided additions)"
        status: pass
    human_judgment: false
  - id: D3
    description: "R6 reports copytree, copyfileobj, move and write_bytes in every spelling; both WR-01 forms on a real notebook"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR6CopyAndWriteFunctions.test_every_copy_and_write_function_in_any_spelling_is_a_problem"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestThirdReviewFormsOnARealNotebook.test_copytree_and_write_bytes_are_reported_on_a_real_notebook"
        status: pass
    human_judgment: false
  - id: D4
    description: "R0 allow-list plus IPython rewrite check plus automagic rule; every CR-02 form and the planning-time automagic and prompt forms reported on a real notebook; harmless magic lines clean"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR0AllowList"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestThirdReviewFormsOnARealNotebook.test_ipython_aliases_and_shell_magics_are_reported_on_a_real_notebook"
        status: pass
      - kind: command
        ref: "Task 3 independent replay and no-weakening check"
        status: pass
    human_judgment: false
  - id: D5
    description: "Look-alikes stay allowed (WARN-05 adjacency) and all eight real notebooks pass the whole guard"
    requirement: WARN-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR12MatchesEveryIdentifierAndString.test_names_that_only_look_like_listed_ones_are_not_a_problem"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestPreExecutedNotebooksUseTheHelper.test_every_real_pre_executed_notebook_is_isolated"
        status: pass
    human_judgment: false
  - id: D6
    description: "Ledger: CR-02 and WR-01 fixed with real 40-10 hashes, open 6, exactly 5 lines changed"
    verification:
      - kind: command
        ref: "Task 3 ledger verify (fifth automated block)"
        status: pass
    human_judgment: false
---

# Phase 40 Plan 10: Guard Reads Every Identifier, String and IPython Rewrite Summary

**R12 now matches its lists at every identifier, in every dotted string part and in string code; R6 covers copytree, copyfileobj, move and write_bytes; R0 is an allow-list checked with IPython's own TransformerManager, plus an automagic rule built from IPython's live magic registry. Every literal route from the third review and the verifier's probe is reported on a copy of a real notebook, and all eight real notebooks stay clean.**

## Performance

- **Duration:** 16 min (2026-10-10T17:41:44Z to 2026-10-10T17:57:41Z)
- **Tasks:** 3 of 3 (Task 1 tracer, Tasks 2-3 auto, all TDD)
- **Files modified:** 2 (guard module: 811 lines added, 48 removed in total, now 2263 lines; ledger: 5 lines)
- **Guard module final Ran count:** 82 tests OK (63 at 1930f31)
- **django-test hook duration (full hook per code commit):** 79 s, 82 s, 81 s. The last code commit, c6c6322, took 81 s.

## Commits

| Commit | Task | Closes |
|--------|------|--------|
| 482606b | Task 1 (tracer): R12 matches every listed module and name at every identifier | CR-01 (a), D-G5 (1) |
| 3232341 | Task 2: R12 reads strings, bytes and string code, adds the new names and ATTACH (; R6 adds copytree, copyfileobj, move, write_bytes | CR-01 (b)-(e), WR-01, D-G5 (1), D-G6 (a)-(b) |
| c6c6322 | Task 3: R0 allow-list over IPython's input handling plus the automagic rule | CR-02, D-G5 (2)-(3) |
| f4f2cd1 | Task 3 ledger (SKIP=django-test, ledger only) | records CR-02 and WR-01 fixed, open 8 to 6 |

## RED / GREEN evidence

RED logs (each new test failed before its implementation; one GREEN full-hook commit per task):

- `$HOME/tmp/phase40-10-t1-red.log`: `FAILED (failures=20)`. 18 subTest FAILs in test_a_listed_name_is_reported_wherever_a_name_appears and 2 in test_subprocess_imported_from_another_module_is_reported_on_a_real_notebook (webbrowser, autoreload).
- `$HOME/tmp/phase40-10-t2-red.log`: `FAILED (failures=43)`. test_a_listed_name_inside_a_string_or_bytes_constant_is_a_problem 7, test_a_process_looked_up_by_a_literal_string_is_reported_on_a_real_notebook 2, test_code_inside_a_string_is_checked_like_code 5, test_copytree_and_write_bytes_are_reported_on_a_real_notebook 2, test_every_copy_and_write_function_in_any_spelling_is_a_problem 12, test_ipython_process_helpers_are_reported_on_a_real_notebook 3, test_the_connection_proxy_and_attach_with_a_parenthesis_are_a_problem 9, test_the_connection_proxy_database_connect_and_attach_are_reported_on_a_real_notebook 3.
- `$HOME/tmp/phase40-10-t3-red.log`: `FAILED (failures=32, errors=4)`. ERROR in test_a_cell_is_a_problem_unless_it_is_python_that_ipython_runs_unchanged, test_a_one_line_alias_or_magic_that_is_valid_python_is_reported_on_a_real_notebook, test_a_one_line_magic_or_alias_is_a_problem_unless_the_notebook_defines_the_name and test_the_automagic_rule_knows_every_line_magic_a_fresh_ipython_shell_has (the markers and the name-set helper did not exist yet). FAIL in test_a_cell_is_a_problem_unless... 9, test_a_one_line_magic_or_alias... 10, test_a_shell_escape_after_a_line_magic... 1, test_code_under_a_harmless_magic_is_checked_by_every_rule 3, test_harmless_magic_lines_leave_a_real_notebook_clean 2, test_ipython_aliases_and_shell_magics... 7.

GREEN:
- Task 1: `Ran 66 tests` / `OK` (guard module).
- Task 2: `Ran 79 tests` / `OK` (guard module plus test_projector_demo_notebook).
- Task 3: `Ran 117 tests` / `OK` (guard module, test_notebook_helper, test_projector_demo_notebook).

Replay and check OK lines:
- `OK: subprocess imported from another module is reported on a real notebook (RED recorded); every identifier position matched; real notebooks clean`
- `OK: independent replay -- subprocess from webbrowser and autoreload reported on a real notebook; every identifier position matched; look-alikes and the eight real notebooks clean`
- `OK: ruff clean, one commit after 1930f31 holding exactly the guard module (482606b...)`
- `OK: strings, string code, lookup functions, connection proxy, ATTACH ( and the copy and write functions reported (RED recorded); real notebooks and projector evidence green`
- `OK: new names exactly as decided (none removed, lists disjoint); ATTACH ( matched; CR-01 b-e, IPython helper and WR-01 forms reported on a real notebook; string code checked; look-alikes and eight real notebooks clean`
- `OK: ruff clean, the strings-and-names commit holds only the guard module (3232341...)`
- `OK: R0 allow-list, IPython rewrite check and automagic rule proven (RED recorded); guard, helper and projector tests green`
- `OK: independent replay -- every CR-02 form and the planning-time automagic forms reported on a real notebook; harmless magic lines clean; code under %%capture checked; 102 IPython line magic and alias names; eight real notebooks clean`
- `OK: no weakening -- 82 guard tests (none removed or skipped), frozen constants unchanged, every list member kept, deny-list and rule calls kept, eight real notebooks clean`
- `OK: ruff clean; the R0 commit holds only the guard module (c6c6322...); three guard commits after 1930f31`
- `OK: CR-02 and WR-01 fixed with real 40-10 commits touching the guard module, open 8 -> 6, exactly 5 lines changed`
- `OK: the ledger commit follows the last guard commit and holds only the ledger (f4f2cd1...); nothing under docs, the helper tests, the review or the verification report changed`

## Harmless magic set (R0 allow-list)

Exactly these are removed before the cell is parsed:
- a whole line `%matplotlib inline`, with optional trailing spaces (HARMLESS_LINE_MAGIC);
- a first line `%%time` on its own, or `%%capture` followed only by `--no-stderr`, `--no-stdout`, `--no-display` and an output name (HARMLESS_CELL_MAGIC, anchored at the start of the cell).

Not harmless, with the reasons:
- `%config`: `%config AliasManager.user_aliases = [...]` defines a shell alias that a later one-line cell runs, and `%config` reconfigures other magics.
- `%matplotlib` with any other argument: a backend such as `module://name` imports a module by name.
- A cell magic with leading whitespace, a cell magic that is not on the first line, or `%%time` with an argument. IPython does not strip these the same way (or refuses them), so they are reported as not parseable.

## IPython name set

- `_ipython_line_magic_names()` has **102** names: 78 line magics from the IPython.core.magics classes, `store` from storemagic, 9 from ipykernel's KernelMagics (autosave, clear, connect_info, edit, less, man, more, qtconsole, subshell; `edit` is also an IPython magic), the 12 default aliases, and the start-up aliases ed, hist and rep.
- A fresh `InteractiveShell` with storemagic loaded has 94 line magics. Every one of them is in the set. Beyond the static classes and the default aliases, the fresh shell printed only **ed, hist, rep**, which are the start-up aliases that IPYTHON_STARTUP_MAGIC_ALIASES lists. The set's extra names compared with the fresh shell are the KernelMagics names, which only a kernel registers.

## Every literal form now reported (with its test)

R12, identifiers (TestR12MatchesEveryIdentifierAndString.test_a_listed_name_is_reported_wherever_a_name_appears, unless noted):
- `from webbrowser import subprocess` + `subprocess.run([... 'manage.py', 'flush' ...], env={'PATH': '/usr/bin'})` on a real notebook: test_subprocess_imported_from_another_module_is_reported_on_a_real_notebook
- `from django.utils.autoreload import subprocess as sp` + `sp.run(...)` on a real notebook: same test
- `x = subprocess`, `x = runpy`, `from webbrowser import importlib`, `import json as subprocess`, `import pkg.subprocess.thing`, `from pkg.subprocess import thing`
- `f(subprocess=1)`, `f(environ=1)`, `def f(subprocess)`, `def importlib()`, `class Popen`, `except OSError as environ`, `global environ`
- `case object(subprocess=sp)`, `case [*environ]`, `case {**environ}`, `case _ as subprocess`, `f = exec`

R12, strings and string code:
- `import_string('subprocess.run')(...)` on a real notebook: test_a_process_looked_up_by_a_literal_string_is_reported_on_a_real_notebook
- `pkgutil.resolve_name('subprocess:run')(...)` on a real notebook: same test
- `import_string('os.system')`, `pydoc.locate('os.system')`, `operator.attrgetter('environ.clear')(os)()`, `sys.modules[b'subprocess'.decode()]`, `x = 'django.db:connection'`, `Macro('!ls')`: test_a_listed_name_inside_a_string_or_bytes_constant_is_a_problem
- `timeit.timeit('from webbrowser import subprocess as sp; ...')`, `code.InteractiveInterpreter().runsource('import os; os.system(c)')`, `cProfile.run('x = __import__(m)')`, `timeit.timeit(setup='import multiprocessing')`, a string holding a string holding `import subprocess`: test_code_inside_a_string_is_checked_like_code

R12, new names:
- `getoutputerror(...)`, `get_output_error_code(...)`, `process_handler(...)` imported from IPython.utils, on a real notebook: test_ipython_process_helpers_are_reported_on_a_real_notebook
- `connection._connections.settings[...] = ...` (CR-01 b), `connection.connection = connection.Database.connect(...)` (CR-01 c), `cursor.execute('ATTACH (%s) AS dev', ...)` (CR-01 d), on a real notebook: test_the_connection_proxy_database_connect_and_attach_are_reported_on_a_real_notebook
- `from django.db import connection`, `django.db.connection.close()`, `c = connection`, `import_string('django.db.connection')`, `type(connection.connection)(p)`, `from django.db.backends.sqlite3.base import Database`, `ATTACH('/x') AS dev`, `attach  (?) as dev`: test_the_connection_proxy_and_attach_with_a_parenthesis_are_a_problem

R6:
- `shutil.copytree(scratch_db_dir, repo_root_path / 'src', dirs_exist_ok=True)` and the `write_bytes` loop on a real notebook: test_copytree_and_write_bytes_are_reported_on_a_real_notebook
- `shutil.copytree`, `shutil.copyfileobj`, `shutil.move`, `sh.move` (an alias), `from shutil import copytree`, `cp = shutil.copytree`, `.write_bytes(data)`, `w = p.write_bytes`, `getattr(shutil, 'copytree')`, `getattr(shutil, 'move')`, `methodcaller('write_bytes', ...)`, `timeit.timeit('import shutil; shutil.copytree(a, b)')`: test_every_copy_and_write_function_in_any_spelling_is_a_problem

R0:
- `%time !cd ../.. && env --ignore-environment ... python -m django flush ...` on a real notebook: test_a_shell_escape_after_a_line_magic_is_reported_on_a_real_notebook
- `cp /tmp/fomo-notebook-db-*/* ../../src/`, `%cp ...`, `rm -r ../../src`, `%time !rm -r ../../src`, `%alias wipe rm -r ../../src`, `%rehashx`, `%pip install -e ../..` on a real notebook: test_ipython_aliases_and_shell_magics_are_reported_on_a_real_notebook
- `rm -r -- '../../src'`, `rehashx`, `store -r`, `reset -f`, `sx('rm' ' -r' ' ../../src')`, `%%time` + `rm -r -- '../../src'` (automagic), and `In [1]: rm -r -- '../../src'` (prompt rewrite), on a real notebook: test_a_one_line_alias_or_magic_that_is_valid_python_is_reported_on_a_real_notebook
- `%config InlineBackend.figure_format = 'svg'` and `%matplotlib module://...` on a real notebook: test_harmless_magic_lines_leave_a_real_notebook_clean
- `%ls`, `%time !ls`, `%%time` + `!ls`, `%%capture` + `%cd ..`, `%%writefile`, `ls ?`, `%matplotlib qt`, ` %%time`, a `%%time` that is not the first line, `%%time --flag`, `In [1]: x = 1`: test_a_cell_is_a_problem_unless_it_is_python_that_ipython_runs_unchanged
- code under a harmless magic (`%%capture` + `from shutil import copytree`, `%%time` + `import sqlite3`, `%%capture` + `open('src/fomo_db.sqlite3')`): test_code_under_a_harmless_magic_is_checked_by_every_rule
- `cp`, `time -x`, `rm += 3`, `man -k`, `ed`, an alias with a trailing comment or a trailing blank line: test_a_one_line_magic_or_alias_is_a_problem_unless_the_notebook_defines_the_name

## Accomplishments

- R12 matches every table at every identifier, in every dotted string part and in string code, so importing a listed module from another module or looking it up by a literal string is reported.
- The connection proxy, Database.connect, `ATTACH (`, IPython's process helpers and the lookup-by-string functions are listed. The seven name lists stay pairwise disjoint, and R12_ALL_NAMES still feeds R0's deny-list.
- R6 reports every shutil copy or move function and any byte write, in every spelling.
- R0 reports any cell that IPython would run as something other than the Python the guard reads. Every later rule reads each cell's Python view.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] A string counts as code only when it holds a statement or a call**
- **Found during:** Task 2
- **Issue:** The plan says to parse and check "every str or bytes constant whose text parses as Python". The plan's own look-alike `print('subprocess-free')` (Task 1 and truth 7) parses as `subprocess - free`, so under that literal reading it is reported under R12. A replay with the literal reading confirmed this.
- **Fix:** `_parse_string_code` treats a parsed string as code only when it holds a statement other than a bare expression, or a call. A bare expression with no call, such as prose, a dotted path or a date, runs nothing even under a library runner, and R12 still matches its dotted parts. Every string-code case the plan lists still parses as code, because each is an import, an assignment or a call.
- **Files modified:** solsys_code/tests/test_pre_executed_notebooks.py
- **Commit:** 3232341

**2. [Rule 1 - Bug] R6's from-shutil import message names the function**
- **Found during:** Task 2
- **Issue:** The plan requires `from shutil import copytree` to give "a problem naming copytree" (Task 2 behavior, and Task 3's `%%capture` case). The existing import message, `imports a shutil copy function`, named no function.
- **Fix:** The message keeps its old text and adds the names, for example `imports a shutil copy function (copytree); a notebook must never copy`. The check itself is unchanged, and test_byte_copy_is_a_problem still passes.
- **Commit:** 3232341

**3. [Rule 3 - Blocking] `from keyword import iskeyword` instead of `import keyword`**
- **Found during:** Task 3
- **Issue:** The plan said to import the module `keyword`, but its name collides with existing loop and local variables called `keyword` (ruff F402 in `_call_command_target`).
- **Fix:** The guard uses `from keyword import iskeyword` and leaves the existing functions untouched.
- **Commit:** c6c6322

**4. [Rule 1 - Lint] SIM102 in the sqlite3 ImportFrom check**
- **Found during:** Task 1
- **Issue:** Removing the old first-component `note(...)` line left a nested `if`, which ruff reported.
- **Fix:** The two conditions are joined with `and`. The behavior is the same.
- **Commit:** 482606b

**Total deviations:** 4 auto-fixed (2 Rule 1 behavior, 1 Rule 3, 1 lint). **Impact:** none of them weakens a rule. Deviation 1 narrows only what counts as string code, and the dotted-part matching still covers those strings.

## Scope statement

- No notebook, docs/notebooks/README.md, docs/notebooks/fomo_nb_utils.py, solsys_code/tests/test_notebook_helper.py, 40-REVIEW.md or 40-VERIFICATION.md changed since 1930f31 (checked by the verify commands).
- 40-11 rewrites the guard's module docstring and the README to match this behavior (D-G5 (4)). The module docstring was left unchanged here on purpose. Only the docstrings of `_database_rerouting_problems`, `_byte_copy_problems` and `isolation_problems` changed. 40-12 puts the corrected override to the developer.
- Round-2 IN-03, WR-03, IN-04, the IN-07 remainder and round-1 IN-03 (runbook text) were not touched. No attribution file was touched.
- Nothing was pushed.

## Issues Encountered

None.

## Known Stubs

None. The plan changed only a test module and the ledger.

## Threat Flags

None. The guard's only new surface is a test that starts one child Python process. It runs with `-I`, a temporary IPYTHONDIR and history off, and reads IPython's line-magic names. No network, no file outside the temporary directory, and no database is involved.

## Next Phase Readiness

Ready for 40-11 (docstring, README and notebook prose for the new behavior, the IN-01 docstring, and the ledger rows for CR-01, WR-02 and IN-01), then 40-12 (the override checkpoint).

## Self-Check: PASSED

- FOUND: solsys_code/tests/test_pre_executed_notebooks.py
- FOUND: .planning/phases/40-notebook-isolation-attribution-page/40-REVIEW-DISPOSITION.md
- FOUND commits: 482606b, 3232341, c6c6322, f4f2cd1 (all ancestors of HEAD; `git rev-list --count ba48dcf..HEAD` = 4)
