---
phase: 40-notebook-isolation-attribution-page
verified: 2026-10-11T14:04:58Z
status: gaps_found
score: 210/214 must-haves verified (prohibitions 67/67; one superseded by D-G2, two 40-15 items superseded by D-G16)
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
  - ".planning/phases/40-notebook-isolation-attribution-page/40-17-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-17-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-18-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-18-SUMMARY.md"
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
covered_digest: "v3:sha256:4482ce31540d91d33856f9514e36c3edfcb7c4f017262e9151040fef3466636c"
behavior_unverified: 0
overrides_applied: 0
overrides:
  - must_have: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale"
    reason: >-
      The guard reports only the names and patterns on its lists, found in a notebook's literal source, and
      makes no claim of completeness; the guard docstring lists every name. Rule R12, extended in 40-08
      [301bd93, f1b4bb6], 40-10 [482606b, 3232341, c6c6322], 40-15 [1c830b1] and 40-17 [c78e517], reports in
      every notebook the names on its five lists: Django's database settings and connection objects
      (DATABASES, DATABASE_ROUTERS, settings_dict, override_settings, modify_settings, the helper's internals
      developer_database_path, prepare_scratch_database and snapshot_database, DatabaseWrapper, load_backend,
      connections, connection, _connections, ConnectionHandler, create_connection, get_connection_params,
      get_new_connection, Database and sqlite3, the connections.settings and connections.databases attribute
      chains and any sqlite3 import); attaching another database through raw SQL (a string holding ATTACH
      followed by DATABASE, a quote, a placeholder or an opening parenthesis); starting a process or reaching
      IPython's shell (subprocess, multiprocessing, pty, pexpect, ptyprocess, psutil, jupyter_client,
      nbclient, nbconvert, IPython, ipykernel, traitlets, system, popen, fork, forkpty, startfile,
      posix_spawn, posix_spawnp, os's exec* and spawn* functions, Popen, ProcessPoolExecutor,
      create_subprocess_exec, create_subprocess_shell, getoutput, getstatusoutput, get_ipython, system_raw,
      run_line_magic, run_cell_magic, getoutputerror, get_output_error_code, process_handler,
      InteractiveShell, ZMQInteractiveShell, IPKernelApp, alias_manager, define_alias, soft_define_alias,
      run_cell, run_cell_async, autocall, define_macro, magics_manager and register_magic_function); the
      environment (environ, environb, putenv, unsetenv); and running code from a string or another file (exec,
      eval, compile, __import__, __builtins__, importlib, runpy, builtins, ctypes, cffi,
      execute_from_command_line, ManagementUtility, import_string, resolve_name, locate, Macro, IPyAutocall,
      and call_command of shell, shell_plus or runscript). Each name is matched at the identifiers the guard
      docstring lists (attributes never against the builtins exec, eval, compile, __import__ and __builtins__)
      and against every '.'- or ':'-separated part of a str or bytes constant, and a string that parses and
      holds a statement or a call is read like the cell's own code, to two levels, so reaching IPython's shell
      as InteractiveShell.instance() or a process by a literal dotted string (import_string('subprocess.run'))
      is reported. R3 and R4 (40-17 [fdd8b3e]) report a Django import before the helper call and a call
      written django.setup(), an import of setup from django or django imported under an alias, in a cell's
      code and in code inside its strings to two levels, and the Django-backed gate reads string code the same
      way. R0 reports a cell that does not parse once %matplotlib inline lines and a first-line %%time or
      %%capture (with only its --no-stderr, --no-stdout and --no-display flags and an output name) are
      removed, a cell whose meaning IPython's static TransformerManager changes, and, since 40-13 [905408a], a
      one-line cell whose first word is in IPython's default line-magic and alias set, whatever an earlier
      cell assigns, imports or defines, unless the word is a builtin or keyword or is followed by = or a
      comma. R5 and R7 report a str or bytes constant that contains FOMO_DATABASE_PATH or fomo_db.sqlite3 and
      a keyword argument named FOMO_DATABASE_PATH (40-13 [4f02628]). R6 reports shutil, _shutil, copy2,
      copyfile, copytree, copyfileobj and write_bytes at the identifiers the docstring lists and as string
      parts (40-13 [4f02628], 40-15 [bc99b0e]), and copy and move read from an expression that mentions
      shutil, _shutil or a name the notebook binds to one through an import, an assignment, a loop, a with, a
      comprehension or a parameter default (40-17 [ca1cb7d]); a cell has at most one write_bytes problem
      (40-17 [ca1cb7d]). Every literal form in the third review (CR-01, CR-02, WR-01), the fourth review's
      CR-01 and WR-01 forms, the fifth review's CR-01 and CR-02 forms, the sixth review's CR-01 and CR-02
      forms, the re-review's four forms and both round-1 probes is reported on a copy of a real notebook, the
      fourth review's WR-02 bytes literals on the reference notebook, and every real pre-executed notebook
      passes the guard and is Django-backed. Any route to a process, IPython's shell, Django's settings or the
      database through a name or form the lists do not contain is not seen, including values assembled or
      looked up at runtime (a name, module, SQL statement or file path built from pieces, or fetched with
      getattr, globals(), vars() or sys.modules using such a string, and a module, function or method found at
      runtime without its name, through inspect.getmodule, a function's __module__ or a search over
      sys.modules or a module's vars()), code decoded at runtime (pickle, marshal, codecs, base64), code
      nested more than two strings deep, rewrites IPython decides at run time (autocall, so exit -1 runs as
      exit(-1), and a one-line cell that runs an alias or magic the notebook defined while it ran), code
      outside the notebook (a module it imports, IPython's profile and start-up files with the extensions,
      aliases and magics they define, or a library function that starts a process, sets Django up or writes a
      file under a name these lists do not contain, such as a path's copy and move on Python 3.14 and later)
      and, left out on purpose, JavaScript or HTML in a cell's output, which a notebook front end may run when
      the notebook is opened.
    accepted_by: "Tim Lister"
    accepted_at: "2026-10-11T13:37:50Z"
re_verification:
  previous_status: gaps_found
  previous_score: 185/190 (prohibitions 58/58)
  gaps_closed:
    - "Round-6 CR-01 (IPython's shell through InteractiveShell.instance()): the alias-plus-wipe, run_cell, autocall = 2 and ex forms are reported on a copy of import_campaign_csv_demo (verifier repro: alias form names IPython, InteractiveShell, alias_manager, define_alias); test green"
    - "Round-6 CR-02 (R3, R4 and the gate did not read string code): the review's timeit probe before the setup cell and the settings.configure variant before the helper call each give 'calls django.setup()' and 'imports Django before the setup_scratch_django call' (verifier repro); test green"
    - "Gap 3 / 40-15 must_have 15.10 (two write_bytes problems in one cell): method plus getattr now gives exactly one (verifier repro); test green"
    - "40-14 must_have 14.2 (R0 alias sentence): the docstring, README and reason now say IPython's default line-magic and alias set, which matches _ipython_line_magic_names"
    - "Round-6 IN-01 (_shutil import not followed): from tempfile import _shutil as sh / sh.move reports move and _shutil"
    - "Round-6 IN-02 ('end the kernel'): gone from the docstring, README and reason"
    - "Round-6 IN-03 (README and docstring omitted rules): R7's keyword check, R8's error output, R9, R10, R11's setup-cell assert and copy/move are now listed (D-G18)"
  gaps_remaining:
    - "ROADMAP SC 2 (first half) / WARN-02: now narrowed to the round-7 review's CR-01, Django set up from string code inside the setup_scratch_django call's own argument, which Python runs before the helper and R3 does not read"
    - "40-08 must_have 8.14 and 40-11 must_have 11.1: the docstring's account of what R0-R12 check is still not accurate, now for the round-7 reasons (R2 'no other call of the setup function', 'R12 reports the IPython shell names a notebook would write to set those up', 'a comment ... cannot ... trip a check')"
  regressions: []
gaps:
  - truth: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale (round-7 review CR-01)"
    status: partial
    reason: >-
      Reproduced by the verifier with the guard's own functions, loaded from a scratchpad copy of the module, on a deep
      copy of import_campaign_csv_demo.ipynb: insert `import timeit` before the helper call and replace the call line
      with `scratch_db_dir = setup_scratch_django(timeit.timeit("from django.conf import settings; import
      src.fomo.settings as s; settings.configure(s); from django.apps import apps;
      apps.populate(settings.INSTALLED_APPS); from django.core.management import call_command; call_command('flush',
      interactive=False)", number=1) and repo_root_path, mode='fresh')`. setup_call() still finds the call,
      notebook_mode() is 'fresh', and isolation_problems() is []. Control: the same timeit string as a statement before
      the call gives 'code cell 1 imports Django before the setup_scratch_django call'. Cause: R3 stops at
      `(number, index) >= (setup_number, setup_index)` (test_pre_executed_notebooks.py, isolation_problems R3 loop), so
      the setup statement itself is never read; R2 checks only that there is one positional argument and a constant
      mode, never what the argument is; R4 reports only its three written forms, and apps.populate is not one of them
      (with a literal django.setup() in the argument R4 does report it). At run time Python evaluates the argument
      before the helper: the setup cell has already put the repo root on sys.path, FOMO_DATABASE_PATH is unset, so
      settings.py:100 falls back to src/fomo_db.sqlite3 and the flush empties it; the helper's RuntimeError ("already
      set up or configured", fomo_nb_utils.py:293-299) comes after. Against the override's exact words: the reason
      says "R3 and R4 (40-17 [fdd8b3e]) report a Django import before the helper call ... in a cell's code and in code
      inside its strings to two levels". This route is a literal Django import, one string level deep, in the
      notebook, that runs before the helper call, and it is not reported. The approved README says the same ("imports
      Django before that call ... also in code inside a string"). So the route contradicts a specific sentence of the
      reason and the override cannot be applied. The broad residual ("a name or form the lists do not contain") does
      not rescue it: the route needs a Django import in string code, which is exactly the form the R3 sentence claims.
      Only the module docstring scopes R3 to "a top-level statement before that call", and the reason does not.
    artifacts:
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "isolation_problems: R2 (`if len(call.args) != 1 or keyword is None ...`) never checks that the argument is a plain name, though its own message says 'with exactly those arguments'; the R3 loop breaks at the setup statement and never reads the call's arguments"
      - path: "docs/notebooks/README.md"
        issue: "line 95-96: 'imports Django before that call ..., also in code inside a string' is false for the call's own argument"
    missing:
      - "R2: report a setup call whose positional argument is not a plain ast.Name (all eight real notebooks pass `repo_root_path`, a Name; verifier checked each)"
      - "R3: also read the setup call's own arguments and keyword values through _with_string_code, so the rule does not depend on R2"
      - "A real-notebook test with the review's probe (argument holds the timeit string) and the statement-before-the-call control"
      - "Keep all eight real notebooks clean and Django-backed"
      - "Or, if the developer prefers: approve a reason and README that scope R3 to 'a top-level statement before the call' and name the call's own arguments as residual (needs the developer's approval)"
  - truth: "40-08 must_have 8.14, 40-11 must_have 11.1 and 40-18 must_have 6 (every form the texts claim is reported is reported): the guard docstring and README state what R0-R12 check"
    status: failed
    reason: >-
      Four sentences approved in 40-18 (aff2547) or kept from 40-02 are contradicted by the code or by the module's own
      tests, each reproduced by the verifier on a deep copy of import_campaign_csv_demo.ipynb or a synthetic notebook.
      (1) CR-01 above: the README's "imports Django before that call ..., also in code inside a string". (2) WR-01: the
      R2 docstring's "and no other call of the setup function" and the README's "call the helper first and once":
      `fomo_nb_utils.setup_scratch_django(repo_root_path, mode='snapshot')`, `s = setup_scratch_django; s(...)` and a
      call inside timeit string code each give [] (control: a second bare call is reported). total_calls counts only
      calls whose func is the bare Name. No database route: the helper refuses a second setup. (3) WR-02: "R12 reports
      the IPython shell names a notebook would write to set those up" (module docstring line 40, residual item (3) at
      line 126, isolation_problems docstring): `exit._ip.find_magic('alias')('wipe cd ../.. && env -i ... flush
      --noinput')` plus a later one-line `wipe` cell gives []; `exit._ip.find_line_magic('sx')(...)` and
      `sys.displayhook.shell.find_line_magic('sx')(...)` give []; the module's own autocall test asserts [] for an
      autocall object built from `type(exit)`. IPython's autocall.py stores the shell as `self._ip` (lines 37-48, read,
      not run). The routes are inside the override's residual ("through a name or form the lists do not contain"), so
      the override is unaffected, but the sentence is false as written. (4) WR-03: "Code is inspected with ``ast`` so
      a comment that merely mentions a name cannot satisfy or trip a check" (since 40-02 f24b04f): `%%time` / `# the
      solar system objects` / `x = 1` is reported as 'not parseable Python but mentions Django, a database, a process
      ...' while the same comment in a plain cell gives []; the comment above UNPARSEABLE_DJANGO_IMPORT ("tolerated
      unless it is about Django or a database") is stale too. This one errs toward reporting, so it is a wording fault
      only, made more likely by 40-17's new R12 words (system, IPython).
    artifacts:
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "module docstring lines 21-22 (comment cannot trip), 40 and 126 (R12 reports the shell names a definition writes), 56 (no other call of the setup function); isolation_problems docstring (same R12 sentence); comment at line 357"
      - path: "docs/notebooks/README.md"
        issue: "line 95 'first and once'; lines 95-96 'imports Django before that call ... also in code inside a string'"
    missing:
      - "After the gap-1 fix, reword or back each sentence: scope R2 to 'no other call written setup_scratch_django(...) in a cell's code' (or count every call whose _call_name is SETUP_FUNCTION across _with_string_code, plus other references); say R12 reports the listed shell names when written and a definition made through an unlisted name (an autocall object built from type(exit), the shell reached as exit._ip) is not seen (or add _ip, find_magic, find_line_magic and find_cell_magic to PROCESS_NAMES; none appears in the eight notebooks' code per the review); say a cell that does not parse as written is searched as text, comments included, and fix the line-357 comment"
      - "Fold in the same pass (not failing truths on their own): WR-04, README lines 59 and 90 say /tmp/fomo-notebook-db-* but prepare_scratch_database uses tempfile.mkdtemp (fomo_nb_utils.py:249), which honours TMPDIR, and only the nbconvert route pins TMPDIR=/tmp; IN-01, README implies R5/R7 read string code (an escaped FOMO_DATABASE_PATH inside timeit string code gives []); IN-02, 'as the not-vacuous test requires' (is_django_backed now accepts a string-only import; all eight real notebooks do hold a real import in a cell tree, verifier checked)"
      - "Rewording the README paragraph, docstring and override reason needs the developer's approval again (D-G17 procedure)"
advisory:
  - finding: "Round-7 IN-03: patching tempfile.mkdtemp before the helper call makes the developer database the 'scratch' file; the guard gives [] (verifier repro)"
    category: security
    reason: "Inside the approved residual class (tempfile and mkdtemp are on no list; 'any route to ... the database through a name or form the lists do not contain'). Optional helper hardening (refuse a scratch path that is not a fomo-notebook-db-* directory directly under tempfile.gettempdir(), or that resolves to the developer database) is a Phase 41 triage candidate"
    evidence_status: "reproduced: isolation_problems() == [] on a deep copy; not executed"
  - finding: "Round-7 IN-04: three spellings of 'the tree plus its string code' (_with_string_code vs inline tuples in _byte_copy_problems and _database_rerouting_problems)"
    category: other
    reason: "Maintainability only; no must-have depends on it"
    evidence_status: "code read"
  - finding: "Round-5 IN-02 (carried): the magic-name pin test uses a plain InteractiveShell, not ipykernel's ZMQInteractiveShell"
    category: other
    reason: "D-G14 out of scope; kept for Phase 41"
    evidence_status: "unchanged since the previous report"
  - finding: "Round-4 IN-01 (carried): the helper's logged warning wording; a non-dict DATABASES alias"
    category: other
    reason: "Out of scope by developer decision 2026-10-10; recorded for Phase 41"
    evidence_status: "helper unchanged"
human_verification:
  - test: "Decide how to close round-7 CR-01 and the four false sentences, and approve the wording again"
    expected: "Either (A) make R2 require a plain-name argument and have R3 read the call's own arguments, then fix or scope the R2, R12-shell and comment sentences (adding _ip, find_magic, find_line_magic, find_cell_magic is cheap), or (B) keep the code and approve a reason and README that scope R3 to a top-level statement before the call and name the call's own arguments as residual"
    why_human: "Accepting residual risk and wording the override are the developer's decisions; the verifier cannot widen an override. The reason's R3 sentence can be read by source position; the verifier read it by execution order, as the README does"
  - test: "Read both snapshot notebooks' prose against this regeneration's output (reconcile cells 8b703ea8, ce1fd748, e1edbaa7, d2adacf8, 3616c012, the NF-14 labels, projector cell 35debc54)"
    expected: "Each claim matches the output beside it"
    why_human: "Prose accuracy is a reading judgment (40-09 D2/D3), carried forward"
  - test: "Read docs/runbooks/telescope_runs_calendar.rst lines ~935-990 (attribution pass)"
    expected: "'each table's page and sort' does not read as covering the worklists' event_page/record_page (round-1 IN-03, kept for Phase 41)"
    why_human: "Documentation wording, carried forward unchanged"
---

# Phase 40: Notebook Isolation & Attribution Page Verification Report

**Phase Goal:** Every pre-executed demo notebook builds and uses its own scratch database, guarded by a check that
catches one that does not, and the staff attribution page renders and pages correctly under Bootstrap 5.3, closing the
remaining six Phase 37.1 review warnings (WR-13 to WR-18).
**Verified:** 2026-10-11T14:04:58Z (HEAD d5770ce; last source change aff2547, last code change ca1cb7d)
**Status:** gaps_found
**Re-verification:** Yes, after gap-closure plans 40-17 and 40-18 (gap round 7)

## Summary

Gap round 7 closed everything the round-6 report asked for:

- Code (40-17): R12 lists IPython's shell modules, classes and entry points. R3, R4 and the Django-backed gate read
  code inside strings. A cell gets one write_bytes problem. An import of `_shutil` is followed.
- Wording (40-18): the docstrings, README and override reason were reworded and approved by the developer.

All four round-6 shell forms and both round-6 string-code probes are reported in the verifier's own reproduction. The
code is unchanged since ca1cb7d apart from docstrings and comments (AST check). The module equals the approved
candidate byte for byte. No test was removed. The only test that changed is the IPyAutocall set D-G16 superseded.

Test runs:
- Verifier: guard, helper, projector and both attribution template modules, 174 tests, OK.
- Orchestrator, full suite on ab72542: 2354 + 40 tests OK. No source has changed since.

SC 2's first half is still open on one route, the round-7 review's CR-01. String code in the `setup_scratch_django(...)`
call's **own argument** runs before the helper, and R3 never reads it. The verifier reproduced it: it gives [].
That contradicts the override reason's R3/R4 sentence and the approved README, so the override is **carried forward
verbatim and not applied**. The fix is two small checks. All eight real notebooks pass the plain name `repo_root_path`,
so they stay green.

Separately, four documentation sentences claim more than the code does (round-7 WR-01, WR-02, WR-03 and CR-01's README
clause), so 8.14 and 11.1 stay failed on wording.

How the verifier stayed safe: it made only inert calls of the guard's own functions, loaded from a scratchpad copy of
the module, on placeholder source or a deep copy of a real notebook. It read IPython source without running it. It ran
no notebook, started no IPython shell, never set `FOMO_DATABASE_PATH`, and never opened `src/fomo_db.sqlite3`
(`git status --short src/` is clean).

## Goal Achievement

### Observable Truths: ROADMAP success criteria (the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | No notebook byte-copies or writes the dev DB; snapshot via read-only backup, others fresh; README rule and lists | ✓ VERIFIED | Helper and notebooks unchanged since cd883a6; all eight `isolation_problems()` = [] and Django-backed, each with a real Django import in a cell tree; TestReadmeStatesTheIsolationRule green |
| SC2a | The guard fails a notebook whose current source routes Django to the dev DB, even with stale output | ✗ FAILED (partial). Override checked and **not applicable** | Round-6 routes closed; round-7 CR-01 reproduced (gap 1) |
| SC2b | DB-path check raises an ordinary exception under `python -O` | ✓ VERIFIED | Helper unchanged; test_notebook_helper among the 174 OK |
| SC3 | telescope_runs_demo reports the dip in one unit | ✓ VERIFIED | Notebook unchanged; dip pin test green |
| SC4 | High-band marker, template test, BS 5.3 only | ✓ VERIFIED | No attribution file changed since cd883a6; test_attribution_template and test_bootstrap5_rendering OK; full suite on ab72542 OK |
| SC5 | Undo on a Dismissed row keeps the section open on the same page | ✓ VERIFIED | Unchanged files; 40/40 functional tests OK in the orchestrator's run on ab72542 |

### Override decision (ROADMAP SC 2 first half / WARN-02)

The override (Tim Lister, accepted_at 2026-10-11T13:37:50Z, recorded in 0f5b540) is carried forward verbatim. The
verifier checked the record:

- The reason equals `$HOME/tmp/phase40-18-approved-reason.txt` (whitespace-normalized).
- accepted_at equals `phase40-18-approved-at.txt`, and must_have and accepted_by are unchanged.
- At 0f5b540, every other front-matter key and the whole body equalled cd883a6.

The verifier then tested every round-7 review route against the reason's exact words:

| Finding | Reproduced? | Inside the reason's residual? | Classification |
|---------|-------------|-------------------------------|----------------|
| CR-01: Django set up from string code in the setup call's own argument (`timeit.timeit("...settings.configure(s); apps.populate(...); call_command('flush', ...)") and repo_root_path`) | **Yes**, []; control (same string as a statement before the call) reported; a literal `django.setup()` in the argument *is* reported by R4 | **No.** It contradicts "R3 and R4 ... report a Django import before the helper call ... in a cell's code and in code inside its strings to two levels". The import is literal, one string deep, in the notebook, and runs before the helper. It also contradicts the README ("imports Django before that call ... also in code inside a string") and R2's own message ("with exactly those arguments") | **Gap** (gap 1, gap 2). Contestable only if "before the helper call" is read as source position, which the docstring says ("top-level statement before that call") and the reason does not: **human decision** on wording |
| WR-01: "no other call of the setup function", README "first and once" | **Yes**: attribute call, call through another name and call in string code each give []; second bare call reported | Not addressed by the reason (it says nothing about R2). No database route: the helper refuses a second setup (fomo_nb_utils.py:293-299) | **Gap** (wording, gap 2) |
| WR-02: "R12 reports the IPython shell names a notebook would write to set those up" | **Yes**: `exit._ip.find_magic('alias')(...)` plus a `wipe` cell, `exit._ip.find_line_magic('sx')`, `sys.displayhook.shell.find_line_magic('sx')` all give []; `_ip` confirmed in IPython's autocall.py source | **Yes**: "through a name or form the lists do not contain". The reason does not repeat the sentence | Override unaffected; **gap** for the docstring wording (gap 2) |
| WR-03: "a comment ... cannot satisfy or trip a check" | **Yes**: a comment under `%%time` is reported; the same comment in a plain cell gives [] | Not a route (the guard reports more, not less) | **Gap** (wording, gap 2; low weight, pre-existing since 40-02) |
| WR-04: README `/tmp` vs `$TMPDIR` | **Yes**: README lines 59/90 say `/tmp`; helper uses `tempfile.mkdtemp` (line 249) | Not a guard route | Warning; fix in the gap-2 pass (40-02's must-have "README tells operators to delete leftover /tmp/... directories" is literally met) |
| IN-01: README implies R5/R7 read string code | **Yes**: an escaped `FOMO_DATABASE_PATH` in timeit string code gives [] | Not a capability on its own (using the variable needs a listed environment or process name) | Info; fold into gap 2 |
| IN-02: "as the not-vacuous test requires" | **Yes**: `x = 'import django'` makes a notebook Django-backed | Not a route; every real notebook has a real import in a cell tree | Info; fold into gap 2 |
| IN-03: patching `tempfile.mkdtemp` before the helper | **Yes**: [] | **Yes**: an unlisted name reaching the database | **Accepted residual** (advisory; helper hardening for Phase 41) |
| IN-04: three spellings of tree-plus-string-code | Code read | n/a | Not relevant to the must-haves (advisory) |

One route (CR-01) contradicts a specific sentence of the reason, so the override does not apply (`overrides_applied:
0`).

### Observable Truths: 40-17 must_haves

| # | Truth (abridged) | Status | Evidence |
|---|------------------|--------|----------|
| 17.1 | Four round-6 CR-01 forms reported with controls | ✓ VERIFIED | Test green; verifier repro of the alias form: IPython, InteractiveShell, alias_manager, define_alias |
| 17.2 | Exactly fifteen names added; lists disjoint | ✓ VERIFIED | Constant diff vs cd883a6: only PROCESS_MODULES (+3) and PROCESS_NAMES (+12), nothing removed |
| 17.3 | R3/R4/gate read string code; both probes report | ✓ VERIFIED | Verifier repro: both give 'calls django.setup()' and 'imports Django before ...' |
| 17.4 | Gate on string-only Django import; `%%capture` pin holds | ✓ VERIFIED | Tests green |
| 17.5 | One write_bytes problem per cell | ✓ VERIFIED | Repro: count 1; test green |
| 17.6 | `_shutil` import followed | ✓ VERIFIED | Repro: move and _shutil reported |
| 17.7 | Superseded expectations only | ✓ VERIFIED | Only test_autocall_..._reported changed (its one set) |
| 17.8 | Adjacency edge | ✓ VERIFIED | Look-alike subtests green |
| 17.9 | Empty edge | ✓ VERIFIED | Eight notebooks [] and Django-backed; `wipe` alone and `exit -1` give [] |
| 17.10 | Ordering edge | ✓ VERIFIED | Pinned by the new tests |
| 17.11 | No weakening | ✓ VERIFIED | All 91 cd883a6 test methods present; only the superseded one differs; docstrings and comments at ca1cb7d identical to cd883a6; only decorator `staticmethod` |
| 17.12 | Three full-hook commits, guard module only | ✓ VERIFIED | `git show --name-only c78e517 fdd8b3e ca1cb7d` |

### Observable Truths: 40-18 must_haves

| # | Truth (abridged) | Status | Evidence |
|---|------------------|--------|----------|
| 18.1 | Draft and checks before any change | ✓ VERIFIED (historical) | Draft files 02:41-02:45 local; approval 06:37:50 local; docs commit 06:39:31 |
| 18.2 | Module docstring names every list entry and the residual class | ✓ VERIFIED | Script: no entry of the 14 lists missing; residual sentence present |
| 18.3 | Sweep free of BANNED_ALL | ✓ VERIFIED | Script: no hit in any docstring or comment |
| 18.4 | D-G18 rules listed | ✓ VERIFIED | Keyword check, error output, R9-R11, copy/move present |
| 18.5 | IN-02 wording | ✓ VERIFIED | No "end the kernel" in the docstring, README or reason |
| 18.6 | Boundary both directions: named not-seen forms give [], and every form the texts claim is reported is reported | ✗ FAILED (partial) | The named boundary probes give [] and the enumerated forms are reported. But the README and reason claim a Django import in string code before the helper call is reported, and CR-01 is not (gap 1/2) |
| 18.7 | Reason structure, citations, Evidence rows | ✓ VERIFIED (literal) | 70 rows; all cited commits exist. The R3/R4 sentence's evidence row cites the docstring's narrower "top-level statement" wording (Override decision) |
| 18.8 | Blocking-human checkpoint; reply saved | ✓ VERIFIED | `phase40-18-decision.txt`: approve; approved_at 2026-10-11T13:37:50Z |
| 18.9 | Module equals approved; code identical to ca1cb7d; README | ✓ VERIFIED | Byte-for-byte equal; stripped-AST equal; approved paragraph present; README diff limited to the paragraph |
| 18.10 | Ledger at 226acc0 | ✓ VERIFIED (historical) | 13/13 lines changed, `open: 2`, `total: 12`; ledger since regenerated for round 7 (b0f33f2, a254611) |
| 18.11 | Override: only reason and accepted_at | ✓ VERIFIED | Checked against cd883a6 at 0f5b540 |
| 18.12 | Paired docs | ✓ VERIFIED | README changed; helper and notebooks unchanged, so no re-execution |

### Carried truths (40-01 to 40-16)

The helper, the eight notebooks and the attribution files are unchanged since the previous report, and so are the
earlier PLAN and SUMMARY files. Status changes:

| Truth | Before | Now |
|-------|--------|-----|
| 14.2 | Failed on the R0 alias sentence | **Verified**: it now says "IPython's default set", which matches `_ipython_line_magic_names` |
| 15.10 | Failed | **Verified** (one write_bytes problem per cell) |
| 15.5, 15.6 | Verified | **Superseded on purpose** by developer decision D-G16 (the IPyAutocall-only set; "gate code unchanged"), recorded in 40-17. Counted as verified, like the D-G2 supersession |
| 8.14, 11.1 | Failed | **Still failed**, now on the round-7 wording findings (gap 2) |

**Score:** 210/214 truths verified, 0 present but behavior-unverified. The 214 are 190 carried and 24 from 40-17 and
40-18. The four failed truths are SC2a, 8.14, 11.1 and 18.6.

### Prohibitions

| Plan | Prohibition | Status | Evidence |
|------|-------------|--------|----------|
| 40-01..40-16 | 58 carried | ✓ VERIFIED (one superseded by D-G2; 40-15's no-gate-change superseded by D-G16) | No regression |
| 40-17 | Guard not weakened | ✓ VERIFIED | AST comparison; nothing removed |
| 40-17 | No existing docstring/comment edited | ✓ VERIFIED | Identical at ca1cb7d |
| 40-17 | `_django_imports`, R0, R12 matching, string-code helpers unchanged | ✓ VERIFIED | Only the planned functions differ |
| 40-17 | No notebook run, dev DB, shell, install | ✓ VERIFIED | `git status --short src/` clean |
| 40-17 | No other file touched | ✓ VERIFIED | Commit file sets |
| 40-18 | Nothing before the reply; only approved text | ✓ VERIFIED | Timestamps; byte-for-byte equality |
| 40-18 | No code change; only reason and accepted_at; no push | ✓ VERIFIED | Stripped-AST check; 0f5b540 checks |
| 40-18 | No dev DB access | ✓ VERIFIED | Inert checks |
| 40-18 | No completeness claim; D-G14 untouched | ✓ VERIFIED | No banned phrase; attribution/helper files unchanged |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/tests/test_pre_executed_notebooks.py` | 15 shell names, string-code R3/R4/gate, write_bytes key, approved docstrings | ⚠️ VERIFIED with coverage gap | 96 guard tests green; CR-01 (gap 1); wording (gap 2) |
| `docs/notebooks/README.md` | Approved guard paragraph | ⚠️ VERIFIED (text) | Two clauses false (gap 2); `/tmp` wording (WR-04) |
| `40-REVIEW-DISPOSITION.md` | Round-6 rows fixed | ✓ VERIFIED | At 226acc0; round-7 rows now open (b0f33f2, a254611) |
| `40-VERIFICATION.md` override | Approved reason recorded | ✓ VERIFIED (recorded, not applicable) | Carried verbatim |
| Helper, notebooks, attribution artifacts | Unchanged | ✓ VERIFIED | `git diff cd883a6 HEAD` empty for them |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| PROCESS_MODULES / PROCESS_NAMES | R12_ALL_NAMES, UNPARSEABLE_SUSPECT_NAMES | built from the lists | ✓ WIRED |
| is_django_backed, `_django_setup_problems`, R3 loop | `_string_code_trees` | `_with_string_code` | ✓ WIRED |
| `_byte_copy_problems` | per-cell `found` | `setdefault` on `('write', name)` | ✓ WIRED |
| `_shutil_names` | SHUTIL_MODULE_NAMES | `alias.name in SHUTIL_MODULE_NAMES` | ✓ WIRED |
| R3 | the setup call's own arguments | none | ✗ NOT WIRED (gap 1, round-7 CR-01) |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data | Status |
|----------|------|--------|-----------|--------|
| Notebook Resolved line | resolved NAME | lazy settings, checked before setup | Yes | ✓ FLOWING (unchanged) |
| Teardown second line | what setup checked | `check_database_settings` | Yes | ✓ FLOWING (unchanged) |
| Attribution page state | return state | unchanged | Yes | ✓ FLOWING (unchanged) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Guard, helper, projector, attribution template modules | `python manage.py test --noinput --parallel auto solsys_code.tests.test_pre_executed_notebooks ...test_notebook_helper ...test_projector_demo_notebook ...test_attribution_template ...test_bootstrap5_rendering` | Ran 174 tests, OK | ✓ PASS |
| Round-6 forms (regression) | scratchpad `extra.py` | all reported as 40-17 claims | ✓ PASS |
| Real notebooks | scratchpad `extra.py` | eight [] and Django-backed; each passes `repo_root_path` (a Name) | ✓ PASS |
| Round-7 CR-01 | scratchpad `repro.py` | [] (control reported) | ✗ FAIL (gap 1) |
| Round-7 WR-01/WR-02/WR-03/IN-01/IN-03 | scratchpad `repro.py` | as the review describes | ✗ FAIL (gap 2) / residual |
| Dev DB untouched | `git status --short src/` | clean | ✓ PASS |

### Probe Execution

Step 7c: SKIPPED. No `scripts/*/tests/probe-*.sh` exists, and no plan declares one.

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|--------------|-------------|--------|----------|
| WARN-02 | 40-02, 40-06..40-18 | Guard fails a notebook whose current source routes Django to the dev DB | ✗ BLOCKED (partial) | Gap 1; override not applicable. REQUIREMENTS.md marks it `[x]` / Complete, which this report does not support |
| WARN-03 | 40-01, 40-02, 40-05, 40-07..40-09, 40-11 | Ordinary exception under `-O` | ✓ SATISFIED | SC2b |
| WARN-04 | 40-03, 40-04 | BS 5.3 only; High-band marker; template test | ✓ SATISFIED | SC4 |
| WARN-05 | 40-01, 40-02, 40-05, 40-07..40-11, 40-13..40-18 | Audit, shared helper, no byte copy or write, README rule and lists | ✓ SATISFIED | SC1; real notebooks clean; guard coverage counted under WARN-02 |
| WARN-06 | 40-02 | Dip in one unit | ✓ SATISFIED | SC3 |
| WARN-07 | 40-04 | Undo keeps Dismissed open on the same page | ✓ SATISFIED | SC5 |

All six IDs appear in plan front matter and in REQUIREMENTS.md (Phase 40). No requirement is orphaned.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| solsys_code/tests/test_pre_executed_notebooks.py | isolation_problems R2/R3 | Setup call argument neither shape-checked nor read | 🛑 Blocker | Gap 1 (carried-forward gap; file modified since the prior verification) |
| solsys_code/tests/test_pre_executed_notebooks.py | 21-22, 40, 56, 126, 357 | Docstring/comment claims more than the code | ⚠️ Warning | Gap 2 |
| docs/notebooks/README.md | 59, 90, 95-96 | `/tmp` vs `$TMPDIR`; "first and once"; string-code import claim | ⚠️ Warning | Gap 2 |

No TBD, FIXME, XXX, TODO or HACK in the lines added since cd883a6.

### Human Verification Required

These items do not change the gaps_found status.

1. **Close round-7 CR-01 and approve the wording again.** Option (A): make R2 require a plain-name argument and have
   R3 read the call's arguments, then fix or scope the R2, R12-shell and comment sentences. Option (B): approve a
   reason and README that scope R3 to a top-level statement before the call. Why human: residual risk and override
   wording are the developer's decisions.
2. **Snapshot notebook prose** (carried). Why human: prose accuracy.
3. **Runbook attribution-pass wording** (round-1 IN-03, kept for Phase 41). Why human: wording.

### Gaps Summary

The root cause is the same as in earlier rounds, now much narrower: **the guard's code is still slightly narrower than
what its approved texts say.** Round 7 closed both round-6 families completely. One literal route is left: string code
inside the setup call's own argument runs before the helper and is not read by R3. Two checks close it, an R2 shape
check (plain-name argument) and an R3 pass over the call's arguments. All eight real notebooks pass the plain name
`repo_root_path`, so none changes.

Fix these wording points in the same pass, with the developer's approval:
- R2's "no other call of the setup function" and the README's "once".
- "R12 reports the IPython shell names a notebook would write". It is false for `exit._ip`. Listing `_ip` and
  `find_*magic` would also be cheap.
- "a comment cannot trip a check".
- The README's `/tmp`, its string-code scope and "as the not-vacuous test requires".

Nothing here is deferred: Phases 41 and 42 do not cover the notebook guard.

---

_Verified: 2026-10-11T14:04:58Z_
_Verifier: Claude (gsd-verifier)_
