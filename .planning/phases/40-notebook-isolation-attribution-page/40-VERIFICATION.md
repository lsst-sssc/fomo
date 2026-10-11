---
phase: 40-notebook-isolation-attribution-page
verified: 2026-10-11T03:58:27Z
status: gaps_found
score: 185/190 must-haves verified (prohibitions 58/58, one superseded by D-G2)
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
  - ".planning/phases/40-notebook-isolation-attribution-page/40-08-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-08-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-09-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-09-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-10-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-10-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-11-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-11-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-12-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-12-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-13-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-13-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-14-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-14-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-15-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-15-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-16-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-16-SUMMARY.md"
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
  - "pyproject.toml"
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
covered_digest: "v3:sha256:1ca5aa76c63e013f6c3dd35f2abbf5e166a8171fb60c958a5d82e21fb0f86322"
behavior_unverified: 0
overrides_applied: 0
overrides:
  - must_have: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale"
    reason: >-
      Rule R12, extended in 40-08 [301bd93, f1b4bb6], 40-10 [482606b, 3232341, c6c6322] and 40-15 [1c830b1],
      reports in every notebook any name it lists for five capabilities: Django's database settings and
      connection objects (DATABASES, DATABASE_ROUTERS, settings_dict, override_settings, modify_settings, the
      helper's internals, DatabaseWrapper, load_backend, connections, connection, _connections,
      ConnectionHandler, create_connection, get_connection_params, get_new_connection, Database, sqlite3);
      attaching another database through raw SQL (a string holding ATTACH followed by DATABASE, a quote, a
      placeholder or an opening parenthesis); starting a process (subprocess, multiprocessing, pty, pexpect,
      ptyprocess, psutil, jupyter_client, nbclient, nbconvert, system, popen, fork, forkpty, startfile,
      posix_spawn, posix_spawnp, every exec* and spawn* function of os, Popen, ProcessPoolExecutor,
      create_subprocess_exec, create_subprocess_shell, getoutput, getstatusoutput, IPython's get_ipython,
      system_raw, run_line_magic and run_cell_magic, and the IPython.utils helpers getoutputerror,
      get_output_error_code and process_handler); the environment (environ, environb, putenv, unsetenv); and
      running code from a string or another file (exec, eval, compile, __import__, __builtins__, importlib,
      runpy, builtins, ctypes, cffi, execute_from_command_line, ManagementUtility, import_string,
      resolve_name, locate, IPython's Macro and IPyAutocall, and call_command of shell, shell_plus or
      runscript). Each name is matched at every identifier the parser records, as the guard docstring
      enumerates them (attributes are never matched against the builtins exec, eval, compile, __import__ and
      __builtins__), and against every '.'- or ':'-separated part of a string or bytes constant; a string that
      parses and holds a statement or a call is read like the cell's own code, to two levels. So a listed name
      imported from another module (from webbrowser import subprocess), bound to another name first or looked
      up by a literal string (import_string('subprocess.run')) is reported. R0 reports any code cell that is
      not Python IPython runs unchanged, checked with IPython's own TransformerManager, once %matplotlib
      inline lines and a first-line %%time or %%capture (with only its --no-stderr, --no-stdout and
      --no-display flags and an output name) are removed, apart from IPython's autocall, which runs a one-line
      cell that starts with an autocall object as a call (exit -1 runs as exit(-1)): the kernel's only
      autocall objects, exit and quit, end the kernel, R12 reports the name IPyAutocall (40-15 [1c830b1]), and
      an autocall object made without that name (a subclass of type(exit)) runs a call method that is notebook
      code the rules read; and, since 40-13 [905408a], R0 reports any one-line cell whose first word is an
      IPython line magic or alias, because IPython's automagic runs it unless the name is bound when the cell
      runs, whatever an earlier cell assigns, imports or defines; only a word that is a builtin or a keyword,
      or one followed by = or a comma, is exempt. R5 and R7 report a str or bytes constant that names
      FOMO_DATABASE_PATH or fomo_db.sqlite3 (40-13 [4f02628]). R6 reports the module names shutil and _shutil,
      shutil's copy2, copyfile, copytree and copyfileobj, and write_bytes at every identifier the parser
      records and as a string part (40-13 [4f02628], 40-15 [bc99b0e]), so code that names shutil is reported
      however it then hands the module on, and copy and move read from an expression that mentions shutil,
      _shutil or a name the notebook binds to one in any cell through an import, an assignment, a loop, a
      with, a comprehension or a parameter default; copy and move read from anything that mentions no shutil
      name (another object's method of that name, such as a path's copy and move on Python 3.14 and later), a
      shutil module, shutil function or write_bytes method obtained at runtime without writing its name (from
      inspect.getmodule, a function's __module__ or a search over sys.modules or a module's vars()), and code
      in a string nested more than two strings deep, which R6 and R12 match only by its parts, are not
      reported. Every literal form in the third review (CR-01, CR-02, WR-01), the fourth review's CR-01 and
      WR-01 forms, the fifth review's CR-01 and CR-02 forms, the re-review's four forms and both round-1
      probes is reported on a copy of a real notebook, the fourth review's WR-02 bytes literals on the
      reference notebook, and every real pre-executed notebook passes the guard. The guard docstring makes no
      claim of completeness and names what stays out of reach: the guard reads only the notebook's own source,
      so values assembled or looked up at runtime (a name, module, SQL statement or file path built from
      pieces, or fetched with getattr, globals(), vars() or sys.modules using such a string), code decoded at
      runtime (pickle, marshal, codecs, base64), code outside the notebook (a module it imports, IPython's
      profile and start-up files with the extensions, aliases and magics they define, or a library function
      that starts a process or writes a file under a name these lists do not contain) and, left out on
      purpose, JavaScript or HTML in a cell's output, which a notebook front end may run when the notebook is
      opened, are not seen.
    accepted_by: "Tim Lister"
    accepted_at: "2026-10-11T03:35:54Z"
re_verification:
  previous_status: gaps_found
  previous_score: 163/167 (prohibitions 49/49)
  gaps_closed:
    - "Round-5 CR-01 (shutil handed on through a call argument, a lambda, map, a keyword-only parameter, a match capture or a class pattern): all six forms are reported on a copy of import_campaign_csv_demo (verifier re-ran 40-15's Task 1 replay: OK line printed)"
    - "Round-5 CR-02 (write_bytes as a class-pattern attribute, a parameter or a keyword): all three reported on the same copy (same replay)"
    - "Round-5 WR-01 (docstring said the code under a harmless magic is checked by every later rule): the sentence is gone; the docstring and isolation_problems docstring describe the raw-source gate, pinned by test_the_django_backed_gate_reads_each_cell_as_written"
    - "Round-5 advisory WR-02 (autocall): IPyAutocall is an R12 name; the exit/quit exception is named and pinned"
    - "The round-5 R6 parts of gap 2 ('any write_bytes', the three-item R6 boundary, 'binds to it', README 'says exactly'): all removed; the R6 boundary the docstring now names matches the code (seven boundary probes give [], three old boundary forms are reported)"
  gaps_remaining:
    - "ROADMAP SC 2 (first half) / WARN-02: now narrowed to the round-6 review's CR-01 (IPython's shell reached through InteractiveShell.instance(): a notebook-defined alias run by a later one-line cell, run_cell of a shell escape, autocall mode switched on) and CR-02 (R3, R4 and is_django_backed do not read code inside strings). Both are outside the override's residual, and the reason itself claims R0 reports any one-line cell that starts with an alias, so the override does not apply"
    - "40-08 must_have 8.14, 40-11 must_have 11.1 and 40-14 must_have 14.2: the docstring, README and override reason still claim more than the code checks, now for the round-6 reasons (R0 alias sentence, 'one rewrite', README 'never sets Django up ... in code inside a string', _string_code_trees 'the rules read it')"
  regressions: []
gaps:
  - truth: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale (round-6 review CR-01 and CR-02)"
    status: partial
    reason: >-
      Two families of literal notebook source still reach the developer database with isolation_problems() == [].
      The verifier reproduced both with the guard's own functions, loaded from a scratch copy of the module, on a
      deep copy of import_campaign_csv_demo.ipynb. (1) CR-01: `from IPython.core.interactiveshell import
      InteractiveShell` then `InteractiveShell.instance().alias_manager.define_alias('wipe', 'cd ../.. && env -i
      PATH=/usr/bin python manage.py flush --noinput')` appended to the setup cell, plus a later one-line cell
      `wipe`, gives []; `_automagic_word('wipe')` is None because the guard knows only IPython's default magic
      names. `InteractiveShell.instance().run_cell('!...')` gives [], `.autocall = 2` gives [] and `.ex('print(1)')`
      gives []. The controls `get_ipython().alias_manager...` and a `%config AliasManager.user_aliases` cell are
      each reported. The verifier confirmed the runtime binding from source: ipykernel builds its shell with
      `self.shell_class.instance(...)` (ipkernel.py:140), and traitlets' `SingletonConfigurable.instance` stores
      that instance on every class in the MRO (configurable.py:586-590), so `InteractiveShell.instance()` returns
      the kernel's shell. None of IPython, InteractiveShell, instance, alias_manager, define_alias, run_cell or
      autocall is on any list. `env -i` clears FOMO_DATABASE_PATH for the child manage.py, which then falls back to
      src/fomo_db.sqlite3 (settings.py:100) and flushes it. (2) CR-02: R3 (`_django_imports(statement)`), R4
      (`_django_setup_problems`) and `is_django_backed` read only the cell tree, not `_string_code_trees`. The
      review's probe (a timeit string with `import django; django.setup(); ... call_command('flush',
      interactive=False)` in a cell before the setup cell) gives []; the plain-code control gives two problems.
      The review's literal probe also needs DJANGO_SETTINGS_MODULE in the kernel's environment (the helper sets
      it), so the verifier checked a variant that needs nothing outside the notebook: inside the setup cell, after
      its sys.path lines and before the helper call, `timeit.timeit("from django.conf import settings; import
      src.fomo.settings as s; settings.configure(s); import django; django.setup(); ... call_command('flush',
      interactive=False)", number=1)` gives []. The control (the same code as plain statements) gives four
      problems. Django's `LazySettings.configure(default_settings=...)` reads every setting it is not given from
      the module it is passed (conf/__init__.py:111-124, 224-235), so DATABASES is the developer database while
      FOMO_DATABASE_PATH is unset. The helper's later RuntimeError comes after the flush. Against the override's
      exact words: the alias form is not a value assembled or looked up at runtime (the alias name and command are
      literals); it is not code outside the notebook (the reason's residual names aliases defined by "IPython's
      profile and start-up files", and this one is defined by notebook code); and the reason itself says R0 reports
      any one-line cell whose first word is an alias. Autocall mode 2 falls outside the reason's autocall
      exception, which is limited to "a one-line cell that starts with an autocall object". CR-02 is literal source
      one string level deep, in the notebook, and settings.configure, settings, src and fomo are on no list. Both
      families are outside the accepted residual, so the override is carried forward verbatim and not applied.
      For the developer's weighing: as in rounds 4-5, residual (2) already accepts process-starting library calls
      under unlisted names (a webbrowser controller), and the run_cell form could be read as one of those. But the
      alias and string-code forms contradict sentences the reason, docstring and README state outright.
    artifacts:
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "PROCESS_MODULES / PROCESS_NAMES (lines 162-203) list get_ipython but not IPython, ipykernel, traitlets, InteractiveShell, ZMQInteractiveShell, alias_manager, define_alias, run_cell, autocall; _ipython_line_magic_names (430-448) is IPython's default set only, so a runtime alias is not seen by _automagic_word"
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "_django_setup_problems (679-693), R3 in isolation_problems (`_django_imports(statement)`, ~line 1190) and is_django_backed (605-607) walk only the cell tree; only R6 and R12 call _string_code_trees"
    missing:
      - "R12: report IPython's shell object however it is reached, as the review sketches: add IPython, ipykernel and traitlets to PROCESS_MODULES and InteractiveShell, ZMQInteractiveShell, IPKernelApp, alias_manager, define_alias, soft_define_alias, run_cell, run_cell_async, autocall, define_macro, magics_manager and register_magic_function to PROCESS_NAMES. No real notebook names any of them in code (verifier grep over all eight: none of IPython, ipykernel, traitlets, InteractiveShell, alias_manager, define_alias, run_cell, autocall, timeit)"
      - "R3, R4 and is_django_backed: read string code as R6 and R12 do (`(tree, *_string_code_trees(tree))`). No real notebook holds a Django import or django.setup in a string"
      - "Add the four CR-01 forms (alias plus a one-line cell, run_cell, autocall = 2, ex) and both CR-02 probes (the review's and the settings.configure variant before the helper call) as tests on a copy of a real notebook, with controls"
      - "Keep all eight real notebooks green"
      - "Or, if the developer prefers: keep the code and approve a reason, docstring and README that name these routes as residual (that needs the developer's approval)"
  - truth: "40-08 must_have 8.14, 40-11 must_have 11.1 and 40-14 must_have 14.2: the guard docstring (and the README and override reason that repeat it) says what the guard reports and names exactly what stays out of reach"
    status: failed
    reason: >-
      Same root cause as gap 1. The module docstring says a one-line cell that starts with an IPython line magic or
      alias is reported, and "R0 does not see one rewrite, IPython's autocall: a one-line cell that starts with an
      autocall object". A notebook-defined alias and autocall mode 2 make both false (CR-01). The README says the
      guard checks that a notebook "never sets Django up ... in code inside a string (read to two levels)" and "has
      no one-line cell that starts with an IPython magic or alias name". CR-02 and CR-01 make both false. The
      `_string_code_trees` docstring says "the rules read it like the cell's own code", but R3 and R4 do not. The
      approved override reason repeats the R0 alias and autocall sentences. Two smaller wording points from the
      same review are folded in here: the README, docstring and reason say exit and quit "end the kernel" and give
      `exit -1` as the example, but ipykernel's ZMQExitAutocall.__call__(keep_kernel) with -1 keeps the kernel
      alive (IN-02, verifier read autocall.py:68-70 and zmqshell.py:652-654); and the `_byte_copy_problems`
      docstring says a name bound to `_shutil` by an import counts, which is false for `from tempfile import
      _shutil as sh` then `sh.move` (only `references _shutil` is reported; IN-01, detection holds).
    artifacts:
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "module docstring lines 28-40 (alias sentence, 'one rewrite', 'end the kernel'); _string_code_trees docstring ('the rules read it'); isolation_problems docstring ('a one-line cell may not start with an IPython line magic or alias name'); _byte_copy_problems docstring ('a name bound to either by an import')"
      - path: "docs/notebooks/README.md"
        issue: "lines 92-101 ('never sets Django up ... in code inside a string', 'has no one-line cell that starts with an IPython magic or alias name', 'only end the kernel')"
    missing:
      - "After the gap-1 fix, make the docstring, README and override reason say what the code checks; rewording the override reason needs the developer's approval again"
      - "Reword 'end the kernel' (IN-02) and the _byte_copy_problems import clause (IN-01), or use `alias.name in SHUTIL_MODULE_NAMES` in _shutil_names"
  - truth: "40-15 must_have 15.10 (Edge WARN-05 ordering): the identifier pass reports write_bytes with exactly the first pass's text, so a cell has at most one write_bytes problem"
    status: failed
    reason: >-
      A cell with `p.write_bytes(b'')` and `getattr(p, 'write_bytes')` gives two write_bytes problems ('references
      write_bytes; ...' and 'names write_bytes in a string; ...'), verifier inert run on the reference notebook. The
      identifier and reference passes share one text, but the string-part pass uses another. This is not a
      detection gap: both are reported. It is a stated invariant in a must-have, in the `_byte_copy_problems`
      docstring and in a code comment that is false. test_write_bytes_at_any_identifier_is_reported_on_a_real_notebook
      asserts `== 1` only for forms with no string part (review WR-01).
    artifacts:
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "_byte_copy_problems docstring ('a cell has at most one write_bytes problem') and the comment 'so a cell never has two write_bytes problems'"
    missing:
      - "Either say 'at most one write_bytes problem from identifiers, and one more for a string part', or key `found` by name and keep the first text; add a method-plus-string case that asserts the documented count"
advisory:
  - finding: "Round-6 IN-03: the README and docstring still leave out R7's keyword check, R10, R8's error-output check and R11's setup-cell assert, and the rewritten README paragraph dropped copy and move"
    category: other
    reason: "An understatement, not a false claim; out of scope for the gap round by developer decision D-G14 and kept for Phase 41 triage"
    evidence_status: "code and README read"
  - finding: "Round-5 IN-02 (carried): the magic-name pin test uses a plain InteractiveShell, not ipykernel's ZMQInteractiveShell"
    category: other
    reason: "The sets matched in the round-5 child check; D-G14 out of scope; kept for Phase 41"
    evidence_status: "unchanged since the previous report"
  - finding: "Round-4 IN-01 (carried): the helper's logged warning wording; a non-dict DATABASES alias"
    category: other
    reason: "Out of scope by developer decision 2026-10-10; recorded for Phase 41"
    evidence_status: "code read; helper unchanged"
human_verification:
  - test: "Decide how to close the round-6 routes (CR-01, CR-02) and approve the wording again"
    expected: "Either (A) apply the review's fixes (report IPython's shell, its alias, cell and autocall entry points as R12 names; have R3, R4 and is_django_backed read string code), then reword the docstring, README and override reason and approve them, or (B) keep the code and approve a reason, docstring and README that name a notebook-defined alias, run_cell, autocall mode and Django set up from string code as residual, and drop the R0 alias and 'one rewrite' claims"
    why_human: "Accepting residual risk and wording the override are the developer's decisions. The verifier cannot widen an override. After six rounds, each new review has found a route the latest wording rules out, so option (B), or a broader residual class, is a real choice"
  - test: "Read docs/notebooks/README.md lines 91-104 and the guard module docstring after the fix"
    expected: "Clear and honest about what the guard reports and what it cannot see"
    why_human: "Documentation wording (40-11 D1/D2 human_judgment: true)"
  - test: "Read both snapshot notebooks' prose against this regeneration's output (reconcile cells 8b703ea8, ce1fd748, e1edbaa7, d2adacf8, 3616c012, the NF-14 labels, projector cell 35debc54)"
    expected: "Each claim matches the output beside it"
    why_human: "Prose accuracy is a reading judgment (40-09 D2/D3), carried forward"
  - test: "Read docs/runbooks/telescope_runs_calendar.rst lines ~935-990 (attribution pass)"
    expected: "'each table's page and sort' does not read as covering the worklists' event_page/record_page (round-1 IN-03, kept for Phase 41)"
    why_human: "Documentation wording, carried forward unchanged"
---

# Phase 40: Notebook Isolation & Attribution Page Verification Report

**Phase Goal:** Every pre-executed demo notebook builds and uses its own scratch database, guarded by a check that catches one that does not, and the staff attribution page renders and pages correctly under Bootstrap 5.3, closing the remaining six Phase 37.1 review warnings (WR-13 to WR-18).
**Verified:** 2026-10-11T03:58:27Z (HEAD 414cad5; last source change b04f09f docstring/README, last code change 1c830b1)
**Status:** gaps_found
**Re-verification:** Yes, after gap-closure plans 40-15 and 40-16 (gap round 6)

## Summary

Gap round 6 closed everything the round-5 report asked for. R6 now reports `shutil`, `_shutil` and `write_bytes`
wherever they are named, so all six round-5 CR-01 forms and the CR-02 class-pattern form are reported on a copy of a
real notebook. `IPyAutocall` is an R12 name. The docstring, README and override reason are exactly the approved
texts. The R6 boundary they name matches the code: all seven boundary probes give [], and the three old boundary
forms are now reported. No test was removed, and the gate and R0 code are identical to 1dbdb6d. Guard, helper and
projector modules: 126 tests OK in the verifier's run. The orchestrator's full suite on ce82278 ran 2349 tests with
one load-induced Playwright timeout that re-ran 22/22 OK, plus 40/40 functional tests OK.

SC 2's first half is still open, now on two routes the round-6 review found. The verifier reproduced both:

- **CR-01:** `InteractiveShell.instance()` is the kernel's shell under a name no list holds. A notebook can define a
  shell alias that a later one-line cell runs, run a shell escape through `run_cell`, or switch autocall on.
- **CR-02:** R3, R4 and the Django-backed gate do not read code inside strings. So
  `timeit.timeit("...settings.configure(s); django.setup(); call_command('flush', ...)")` before the helper call
  flushes the developer database unreported.

Both routes are outside the override's residual, and the reason claims the opposite for the alias form. So the
override is **carried forward verbatim and not applied**. The fixes are small and touch no real notebook: none of the
eight names IPython, its shell or `timeit` in code.

To stay safe, the verifier made only inert calls of the guard's own functions, loaded from a scratch copy of the
module, on placeholder source or a deep copy of a real notebook. It read IPython, ipykernel, traitlets and Django
source without running it. It ran no notebook, never started an IPython shell, never called `django.setup()`, never
set `FOMO_DATABASE_PATH`, and never opened `src/fomo_db.sqlite3` (`git status --short src/` is clean).

## Goal Achievement

### Observable Truths: ROADMAP success criteria (the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | No notebook byte-copies or writes the dev DB; snapshot via read-only backup, others fresh; README rule and lists | ✓ VERIFIED | `git diff 5e03565 HEAD` touches only the guard module and README; all eight notebooks `isolation_problems()` = [] and Django-backed (verifier script); TestReadmeStatesTheIsolationRule green |
| SC2a | The guard fails a notebook whose current source routes Django to the dev DB, even with stale output | ✗ FAILED (partial). Override checked and **not applicable** | Round-5 routes closed; round-6 CR-01 and CR-02 reproduced (gap 1) |
| SC2b | DB-path check raises an ordinary exception under `python -O` | ✓ VERIFIED | Helper unchanged; TestHelperFailClosedIsExceptionSafe among the 126 OK |
| SC3 | telescope_runs_demo reports the dip in one unit | ✓ VERIFIED | Notebook unchanged; dip pin test green |
| SC4 | High-band marker, template test, BS 5.3 only | ✓ VERIFIED | No attribution file changed since 5e03565; orchestrator's full run (the one Playwright timeout was load-induced and re-ran 22/22 OK) |
| SC5 | Undo on a Dismissed row keeps the section open on the same page | ✓ VERIFIED | Unchanged files; 40/40 functional tests OK in the orchestrator's run |

### Override decision (ROADMAP SC 2 first half / WARN-02)

The override (Tim Lister, accepted_at 2026-10-11T03:35:54Z, recorded in 88fc09e) is carried forward verbatim. The
verifier checked: 88fc09e changed only the reason block and `accepted_at`, and the reason equals
`$HOME/tmp/phase40-16-approved-reason.txt`. It then tested each review route against the reason's exact words.

| Route (all reproduced: guard gives []) | Inside the reason's residual words? | Verdict |
|------|------|---------|
| CR-01 alias: `InteractiveShell.instance().alias_manager.define_alias('wipe', 'cd ../.. && env -i ... manage.py flush --noinput')`, then a one-line cell `wipe` | No. The residual names aliases defined by "IPython's profile and start-up files", and this one is notebook code with literal values. The reason says "R0 reports any one-line cell whose first word is an IPython line magic or alias" | **Outside**, and it falsifies a sentence of the reason |
| CR-01 `InteractiveShell.instance().run_cell('!cd ../.. && env -i ... flush')` | Arguable. Residual (2) names "a library function that starts a process ... under a name these lists do not contain", but R12 lists `get_ipython` precisely to keep this object out of reach | Human decision (contested) |
| CR-01 `InteractiveShell.instance().autocall = 2` | No. The reason's exception is "a one-line cell that starts with an autocall object"; with mode 2 any callable is called | **Outside** (wording); no new database capability shown on its own |
| CR-01 `InteractiveShell.instance().ex('print(1)')` | n/a: the string is read by R6/R12 like any string code; not by R3/R4 (that is CR-02) | Covered by CR-02 |
| CR-02 `timeit.timeit("... settings.configure(s); django.setup(); ... call_command('flush', ...)")` before the helper call | No: literal source, one string level deep, in the notebook | **Outside** |
| For comparison, accepted: `f.replace(...)`, `open(..., 'wb')`, a `webbrowser` controller | Yes, residual (2) | Inside |

At least two route families are clearly outside the residual, and the reason contains a sentence the alias route
falsifies. So the override does not apply (`overrides_applied: 0`).

### Observable Truths: 40-15 must_haves

| # | Truth (abridged) | Status | Evidence |
|---|------------------|--------|----------|
| 15.1 | Six CR-01 forms reported on a real-notebook copy; parameter-default control | ✓ VERIFIED | Verifier re-ran 40-15 Task 1 replay: OK line; test green |
| 15.2 | CR-02 class pattern, parameter, keyword reported; method control | ✓ VERIFIED | Same replay; test green |
| 15.3 | SHUTIL_MODULE_NAMES; reported at every identifier and string part; listed cases | ✓ VERIFIED | Line 280; replay asserts; test green |
| 15.4 | shutil.which superseded; other look-alikes and three new ones clean | ✓ VERIFIED | Replay; test green |
| 15.5 | IPyAutocall in DYNAMIC_CODE_NAMES; autocall exception pinned | ✓ VERIFIED | Line 225; Task 2 replay OK line; test green |
| 15.6 | Gate code unchanged; raw-source pin | ✓ VERIFIED | ast comparison with 1dbdb6d (is_django_backed, _django_imports, isolation_problems, _automagic_word, _ipython_rewrites, _python_view, _django_setup_problems all identical); pin test green |
| 15.7 | Function/class docstrings contain the required statements | ✓ VERIFIED (literal) | Replay needles present; two clauses are inaccurate (IN-01 import clause, gap 2; "at most one", gap 3) |
| 15.8 | Adjacency edge | ✓ VERIFIED | Look-alikes clean; exact names reported |
| 15.9 | Empty edge | ✓ VERIFIED | Eight real notebooks [] |
| 15.10 | Ordering edge, including "a cell has at most one write_bytes problem" | ✗ FAILED (partial) | Method plus string part gives two write_bytes problems (gap 3) |
| 15.11 | No weakening | ✓ VERIFIED | 86 tests at 1dbdb6d all present (91 now, five added); only decorator is `staticmethod`; module docstring unchanged by 40-15 (changed later by 40-16, as planned) |
| 15.12 | Gate: two full-hook commits, guard module only | ✓ VERIFIED | `git show --name-only bc99b0e 1c830b1`: only the guard module each |

### Observable Truths: 40-16 must_haves

| # | Truth (abridged) | Status | Evidence |
|---|------------------|--------|----------|
| 16.1 | Draft with five sections before any change | ✓ VERIFIED (historical) | `phase40-16-draft.md` 20:28 local, approval files 20:36, docs commit log 20:38 |
| 16.2 | D-G11 R6 paragraph and boundary true of the code | ✓ VERIFIED | Seven boundary probes []; old forms reported; all R6 claims hold |
| 16.3 | D-G12 gate sentence | ✓ VERIFIED | "checked by every later rule" gone; gate pinned |
| 16.4 | D-G13 autocall named; probes hold | ✓ VERIFIED (literal) | Probes hold; the "one rewrite" and "end the kernel" wording is inaccurate (gap 2) |
| 16.5 | README phrases, no completeness claim | ✓ VERIFIED (literal) | Required phrases present; "exactly" absent from the guard paragraph; two README clauses are false (gap 2) |
| 16.6 | Reason structure, hashes, evidence rows | ✓ VERIFIED (literal) | Reason equals the approved text; 77 evidence rows in 40-16-SUMMARY. Being approved does not make its R0 alias sentence true (Override decision) |
| 16.7 | Blocking-human checkpoint; reply saved | ✓ VERIFIED | `phase40-16-decision.txt`: approve; `phase40-16-approved-at.txt`: 2026-10-11T03:35:54Z |
| 16.8 | Docstring and README equal approved; code identical to 1c830b1; README otherwise unchanged | ✓ VERIFIED | Verifier script: all four checks True |
| 16.9 | Ledger | ✓ VERIFIED (historical) | 930d2df: 11 lines, `open: 5`; since regenerated for round 6 (7962851, f972dd1) |
| 16.10 | Override: only reason and accepted_at | ✓ VERIFIED | `git show 88fc09e`: only the reason block and accepted_at; current entry equals 88fc09e's |
| 16.11 | Paired docs | ✓ VERIFIED | README changed; helper and notebooks unchanged, so no re-execution |

### Carried truths (40-01 to 40-14)

The helper, all eight notebooks and every attribution file are unchanged since the previous report. SC2a, 8.14,
11.1 and 14.2 stay failed, now for the round-6 reasons. 14.2's R6 and gate parts are now true, but its "says exactly
what [the guard] checks" fails on the R0 alias sentence, which has been there since 40-13.

**Score:** 185/190 truths verified, 0 present but behavior-unverified. The 190 are 167 carried (163 verified; SC2a,
8.14, 11.1 and 14.2 failed), 12 from 40-15 (15.10 failed) and 11 from 40-16.

### Prohibitions

| Plan | Prohibition | Status | Evidence |
|------|-------------|--------|----------|
| 40-01..40-14 | 49 carried | ✓ VERIFIED (one superseded by D-G2) | No regression |
| 40-15 | Guard not weakened | ✓ VERIFIED | No test removed or renamed; only the superseded look-alike left the list; no skip/tag |
| 40-15 | Gate and R0 code unchanged | ✓ VERIFIED | ast comparison with 1dbdb6d |
| 40-15 | No notebook run, no dev DB, no shell, no install | ✓ VERIFIED | `git status --short src/` clean |
| 40-15 | No module docstring, README, helper, ledger edit | ✓ VERIFIED | Commit file sets; module docstring at 1c830b1 equals 1dbdb6d's |
| 40-15 | D-G14 items untouched | ✓ VERIFIED | `git diff 5e03565 HEAD` touches only the guard module and README |
| 40-16 | Nothing written before the reply; only approved text | ✓ VERIFIED | Timestamps; texts equal the approved files |
| 40-16 | No code change; only reason and accepted_at; no push | ✓ VERIFIED | ast check; 88fc09e diff |
| 40-16 | No dev DB access | ✓ VERIFIED | Inert checks |
| 40-16 | No completeness claim; D-G14 untouched | ✓ VERIFIED | No completeness phrase in the docstring or the guard paragraph |

### Round-6 review findings (40-REVIEW.md): explicit classification

| Finding | Reproduced? | Classification | Reasoning |
|---------|-------------|----------------|-----------|
| CR-01 IPython's shell through `InteractiveShell.instance()` | **Yes**: alias define plus a `wipe` cell, run_cell, autocall = 2 and ex all give []; `get_ipython` and `%config` controls are reported; runtime binding confirmed from ipykernel/traitlets source | **Gap** (gap 1, gap 2); the run_cell form alone is contestable under residual (2), so it is a **human decision** | The alias form is outside the residual and contradicts the reason, docstring and README |
| CR-02 R3/R4/gate do not read string code | **Yes**: the review's probe and a self-contained `settings.configure(s)` variant both give []; plain-code controls are reported | **Gap** (gap 1, gap 2) | Literal source, one string level deep; contradicts the README and the `_string_code_trees` docstring |
| WR-01 "at most one write_bytes problem" | **Yes** (two problems) | **Gap** (gap 3, low weight) | False invariant in a must-have; no detection gap |
| IN-01 `from tempfile import _shutil as sh; sh.move` | **Yes** (only `references _shutil`) | Wording only (folded into gap 2) | Detection holds; docstring clause false |
| IN-02 `exit -1` keeps the kernel | Source read: `ZMQExitAutocall.__call__(keep_kernel)` sets keepkernel_on_exit = -1, and `ask_exit` sets exit_now = not -1 | Wording only (folded into gap 2) | No database route |
| IN-03 README/docstring omissions | Read | **Accepted residual** (advisory, D-G14) | Understatement, not a false claim |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/tests/test_pre_executed_notebooks.py` | SHUTIL_MODULE_NAMES, IPyAutocall, pin tests, approved docstring | ⚠️ VERIFIED with coverage gap | 91 tests green; CR-01/CR-02 (gap 1); wording (gaps 2, 3) |
| `docs/notebooks/README.md` | Approved guard paragraph | ⚠️ VERIFIED (text) | Two clauses false (gap 2) |
| `40-REVIEW-DISPOSITION.md` | Round-5 rows recorded | ✓ VERIFIED | Round-6 rows open; round-5 decisions kept in "ID reuse (round 6)" |
| `40-VERIFICATION.md` override | Approved reason recorded | ✓ VERIFIED (recorded, not applicable) | Carried verbatim |
| Helper, notebooks, attribution artifacts | Unchanged | ✓ VERIFIED | git diff empty |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| `_byte_copy_problems` identifier pass | `_identifiers` | SHUTIL_MODULE_NAMES, DISTINCTIVE_COPY_NAMES, BYTE_WRITE_NAMES | ✓ WIRED |
| `_byte_copy_problems` string parts | `_string_parts` | `part in SHUTIL_MODULE_NAMES` | ✓ WIRED |
| `_shutil_names` | SHUTIL_MODULE_NAMES | `set(SHUTIL_MODULE_NAMES)` start set | ✓ WIRED (imports follow only `alias.name == 'shutil'`, IN-01) |
| DYNAMIC_CODE_NAMES | R12_ALL_NAMES, UNPARSEABLE_SUSPECT_NAMES | `'IPyAutocall'` | ✓ WIRED |
| R3, R4, `is_django_backed` | `_string_code_trees` | none | ✗ NOT WIRED (gap 1, CR-02) |
| R12 process names | IPython's shell object | `get_ipython` only | ⚠️ PARTIAL (gap 1, CR-01) |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data | Status |
|----------|------|--------|-----------|--------|
| Notebook Resolved line | resolved NAME | lazy settings, checked before setup | Yes | ✓ FLOWING (unchanged) |
| Teardown second line | what setup checked | `check_database_settings` before `django.setup()` | Yes | ✓ FLOWING (unchanged) |
| Attribution page state | return state | unchanged | Yes | ✓ FLOWING (unchanged) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Guard, helper, projector | `python manage.py test --noinput --parallel auto solsys_code.tests.test_pre_executed_notebooks ...test_notebook_helper ...test_projector_demo_notebook` | Ran 126 tests, OK | ✓ PASS |
| 40-15 Task 1 replay | `$HOME/tmp/verify40r6/t1_replay.py` | OK line; seven boundary lines "not reported" | ✓ PASS |
| 40-15 Task 2 replay | `$HOME/tmp/verify40r6/t2_replay.py` | OK line | ✓ PASS |
| 40-16 records | inline script | docstring and README equal the approved texts; code equals 1c830b1; reason equals the approved text | ✓ PASS |
| CR-01 / CR-02 / WR-01 / IN-01 | `$HOME/tmp/verify40r6/repro.py`, `cr02b.py` (scratch copy of the guard) | all [] or as described; controls reported | ✗ FAIL (gaps 1-3) |
| Dev DB untouched | `git status --short src/` | clean | ✓ PASS |

### Probe Execution

Step 7c: SKIPPED. No `scripts/*/tests/probe-*.sh` exists, and no plan declares one.

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|--------------|-------------|--------|----------|
| WARN-02 | 40-02, 40-06..40-16 | Guard fails a notebook whose current source routes Django to the dev DB | ✗ BLOCKED (partial) | Gap 1; override not applicable |
| WARN-03 | 40-01, 40-02, 40-05, 40-07..40-09, 40-11 | Ordinary exception under `-O` | ✓ SATISFIED | SC2b |
| WARN-04 | 40-03, 40-04 | BS 5.3 only; High-band marker; template test | ✓ SATISFIED | SC4 |
| WARN-05 | 40-01, 40-02, 40-05, 40-07..40-11, 40-13, 40-15, 40-16 | Audit, shared helper, no byte copy or write, README rule and lists | ✓ SATISFIED | SC1; the real notebooks are clean; guard coverage gaps are counted under WARN-02 |
| WARN-06 | 40-02 | Dip in one unit | ✓ SATISFIED | SC3 |
| WARN-07 | 40-04 | Undo keeps Dismissed open on the same page | ✓ SATISFIED | SC5 |

All six IDs appear in plan front matter and in REQUIREMENTS.md (Phase 40). No requirement is orphaned.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| solsys_code/tests/test_pre_executed_notebooks.py | 162-203 | Shell reached by `get_ipython` only; other accessors unlisted | 🛑 Blocker | Gap 1 (CR-01); carried-forward gap, file modified since the prior verification |
| solsys_code/tests/test_pre_executed_notebooks.py | 605-607, 679-693, ~1190 | R3/R4/gate skip string code | 🛑 Blocker | Gap 1 (CR-02) |
| solsys_code/tests/test_pre_executed_notebooks.py, docs/notebooks/README.md | docstring 28-40; README 92-101 | Wording claims more than the code checks | ⚠️ Warning | Gap 2 |
| solsys_code/tests/test_pre_executed_notebooks.py | `_byte_copy_problems` docstring and comment | "at most one write_bytes problem" | ⚠️ Warning | Gap 3 |

No TBD, FIXME, XXX, TODO or HACK in the lines added since 1dbdb6d.

### Human Verification Required

These items do not change the gaps_found status.

1. **Close CR-01 and CR-02 and approve the wording again.** Option (A): add IPython's shell and its alias, cell and
   autocall entry points to R12, and have R3, R4 and the gate read string code. Option (B): keep the code and name
   these routes as residual. The run_cell form is contestable under residual (2), so the developer should rule on
   it either way. Why human: accepting residual risk is the developer's decision. Note that each of six reviews has
   found a route the latest wording rules out.
2. **README guard paragraph and guard docstring** after the fix. Why human: wording.
3. **Snapshot notebook prose** (carried). Why human: prose accuracy.
4. **Runbook attribution-pass wording** (round-1 IN-03, kept for Phase 41). Why human: wording.

### Gaps Summary

There is one root cause again: **the guard's code is narrower than what its docstring, README and approved override
reason say.** Round 6 fixed R6 completely. But the round-6 review found two older routes that the rewritten texts
now rule out explicitly:

- **The shell object.** `get_ipython` is listed, but `InteractiveShell.instance()` returns the same shell, so a
  notebook can define an alias that a later one-line cell runs.
- **String code.** R3 and R4 never read string code, so Django set up inside a `timeit` string before the helper
  writes the developer database.

The review's fixes are a few names in R12 and a string-code pass in R3, R4 and the gate. No real notebook names
any of these, so all eight stay green. Fix the "at most one write_bytes problem" invariant, the `_shutil` import
clause and "end the kernel" in the same pass. Rewording the override reason needs the developer's approval.

Nothing here is deferred: Phases 41 and 42 do not cover the notebook guard.

---

_Verified: 2026-10-11T03:58:27Z_
_Verifier: Claude (gsd-verifier)_
