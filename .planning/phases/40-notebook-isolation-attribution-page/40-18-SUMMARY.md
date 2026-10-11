---
phase: 40-notebook-isolation-attribution-page
plan: 18
subsystem: testing
tags: [notebook-isolation, static-guard, docstrings, readme, override, review-ledger, gap-closure]
gap_closure: true
gap_ids: [CR-01, CR-02, WR-01, IN-01, IN-02, IN-03]

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-17's guard code (R12 IPython names, string-code reading in R3/R4/gate, one write_bytes problem per cell, _shutil import followed) at c78e517, fdd8b3e, ca1cb7d"
provides:
  - "Guard module docstring, nine function docstrings, four class docstrings and fourteen comments that describe each rule by what its lists contain and name unlisted routes as one residual class (developer-approved, D-G17)"
  - "README guard paragraph that lists every rule (keyword check, R8 error output, R10 teardown, R11 setup-cell assert, copy and move from a shutil name; D-G18) and claims no completeness"
  - "Review ledger rows for the sixth review's CR-01, CR-02, WR-01, IN-01, IN-02 and IN-03 recorded fixed (open 8 -> 2)"
  - "The developer-approved reworded WARN-02 override reason in 40-VERIFICATION.md (reason and accepted_at only)"
affects: [40-verification, phase-41-triage]

requirements-completed: [WARN-02, WARN-05]

actuals:
  tokens: 17154          # chars/4 over `git diff 2b14e27 0f5b540` (guard docstrings and comments, README, ledger, override)
  tasks: 3
  commits: 3             # MEASURED: git rev-list --count 2b14e27..HEAD before this SUMMARY commit
plan_head_before: 2b14e27
plan_head_after: 0f5b540

tech-stack:
  added: []
  patterns:
    - "Docstrings and comments describe a check by the lists it holds and end with one named residual class, never a completeness claim"
    - "Draft in $HOME/tmp, blocking-human decision, record only the saved approved text and the orchestrator-supplied approval time"

key-files:
  created: []
  modified:
    - solsys_code/tests/test_pre_executed_notebooks.py
    - docs/notebooks/README.md
    - .planning/phases/40-notebook-isolation-attribution-page/40-REVIEW-DISPOSITION.md
    - .planning/phases/40-notebook-isolation-attribution-page/40-VERIFICATION.md

key-decisions:
  - "Developer (Tim Lister) approved all drafted wording and the reworded override reason as drafted (decision approve, approved_at 2026-10-11T13:37:50Z)"
  - "The residual is one class: any route to a process, IPython's shell, Django's settings or the database through a name or form the lists do not contain"
  - "Autocall is described as a rewrite IPython decides at run time that R0 does not read (exit -1 runs as exit(-1)); the exit and quit end-the-kernel sentence is dropped (IN-02)"

patterns-established:
  - "Evidence table: every override-reason sentence has a row citing a committed test or an exact phrase of the module docstring"

duration: 12 min (resume segment; Task 1 drafting ran earlier, the checkpoint wait is not counted)
completed: 2026-10-11
status: complete
---

# Phase 40 Plan 18: Guard Documentation, Ledger and Override Summary

**The guard's module docstring, function and class docstrings, comments and README paragraph now describe each rule by the names its lists contain and name one residual class, in wording the developer approved; the ledger records the sixth review's six findings fixed and the approved override reason is on file.**

## Performance

- **Duration:** 12 min for this resume segment (Task 1 drafting and the Task 2 wait are not counted)
- **Started (resume):** 2026-10-11T13:38Z
- **Completed:** 2026-10-11T13:50Z
- **Tasks:** 3 (1 tracer, 1 blocking-human decision, 1 auto)
- **Files modified:** 4 (plus this SUMMARY)

## Accomplishments

- Task 1 (tracer): drafted the candidate guard module, the README paragraph, the reason and an Evidence table under `$HOME/tmp`, with nothing in the repository changed; both draft checks printed their OK lines (the candidate differs from 40-17's guard only in docstrings and comments, diff 504 lines, passes the pinned ruff 0.16.9 check and format and the 120-column limit; the reason has 9 sentences and 70 Evidence rows, names every R12 entry and cites the three 40-17 commits; every claimed form is reported and every named not-seen form gives `isolation_problems() == []`).
- Task 2 (blocking-human decision): the developer answered `approve`, approved_at `2026-10-11T13:37:50Z`. The reply is saved at `$HOME/tmp/phase40-18-decision.txt`, the time at `$HOME/tmp/phase40-18-approved-at.txt`, and the approved texts at `$HOME/tmp/phase40-18-approved-guard.py`, `-approved-readme.txt` and `-approved-reason.txt`.
- Task 3: recorded the approved docstrings, comments and README (docs commit `aff2547`, full hook green), the ledger rows (`226acc0`) and, last, the approved override reason and approval time (`0f5b540`).
- No code changed in the guard module (its code is identical to 40-17's last guard commit `ca1cb7d`); no notebook was re-executed because neither the helper nor any notebook changed.

## Task Commits

1. **Task 1 (tracer): draft the reworded texts and prove them against 40-17's guard** - none (scratch under `$HOME/tmp` by design)
2. **Task 2 (checkpoint:decision, blocking-human): developer approval** - none (decision files under `$HOME/tmp`)
3. **Task 3: record docs, ledger, override**
   - docs: `aff2547` - guard module docstrings and comments plus the README guard paragraph (full pre-commit hook, including the Django test hook)
   - ledger: `226acc0` - `40-REVIEW-DISPOSITION.md` only (`SKIP=django-test`)
   - override: `0f5b540` - `40-VERIFICATION.md` only (`SKIP=django-test`); the plan's last commit before this metadata commit

**Plan metadata:** the commit that carries this SUMMARY, STATE.md, ROADMAP.md (docs: complete plan)

## Decision and approval

- Decision: `approve` (all as drafted, override accepted). Approval time: `2026-10-11T13:37:50Z`, supplied by the orchestrator, not taken by the executor.
- Recording a reason does not apply the override: the verifier applies or rejects it at re-verification (`overrides_applied` stays as the verifier left it).

## Recorded module docstring (verbatim)

~~~text
Guard tests: every Django-backed pre-executed notebook runs on a scratch database through the shared helper.

Why this module exists (UAT G-37.1-3, then review findings WR-13, WR-14 and WR-17 of the 37.1 review,
WR-01 of the Phase 40 review, WR-01 of the 2026-10-10 re-review, CR-01 and CR-02 of the third review, CR-01, WR-01 and
WR-02 of the fourth review, CR-01, CR-02, WR-01 and WR-02 of the fifth review, and CR-01, CR-02, WR-01 and IN-01 to
IN-03 of the sixth review):
notebooks under ``docs/notebooks/pre_executed/`` once opened the developer database
``src/fomo_db.sqlite3`` and wrote demo rows into it, and a guard that only matched the setup
cell's text was easy to get around (an own ``FOMO_DATABASE_PATH`` assignment with stale good-looking
output, ``from django import setup as _s``, a setup call hidden in a ``def``). Regenerating such a
notebook is routine, so each regeneration would have written to the developer database again.

The rule the notebooks follow: every Django-backed notebook calls the shared helper
(``docs/notebooks/fomo_nb_utils.py``) first -- one top-level ``scratch_db_dir = setup_scratch_django(repo_root_path,
mode='fresh' | 'snapshot')`` before any Django import -- does not set Django up or route the database itself, does not
byte-copy a database, ends with ``teardown_scratch_django(scratch_db_dir)``, and shows output from one fresh run
(execution counts 1..N, the ``Resolved database:`` line inside a ``fomo-notebook-db-*`` directory).
``isolation_problems()`` reports the names and patterns the rules below list, read from the notebook's literal source.
The helper's own behaviour is proven by running it in ``test_notebook_helper.py``.

These tests read notebook JSON only and never execute a notebook. Code is inspected with ``ast`` so a
comment that merely mentions a name cannot satisfy or trip a check.

What the guard checks. ``isolation_problems()`` reports only the names and patterns listed here, found in a notebook's
literal source. R0, R5, R6, R7 and R12 apply to every notebook; R2-R4 and R8-R11 apply to a Django-backed notebook. R0:
once ``%matplotlib inline`` lines and a first-line ``%%time`` or ``%%capture`` (with only its ``--no-stderr``,
``--no-stdout`` and ``--no-display`` flags and an output name) are removed, a code cell must parse and IPython's static
input transformer (``TransformerManager``) must leave its meaning unchanged, so an escape on any line (also after a line
magic, as in ``%time !ls``), any other magic, an escaped alias and a prompt prefix (``In [1]:``) are reported.
``%config`` is not harmless, because ``%config AliasManager.user_aliases = [...]`` defines a shell alias;
``%matplotlib`` with any other argument is not harmless either, because a backend such as ``module://name`` imports a
module by name. A one-line cell whose first word, read with IPython's ``split_user_input``, is in IPython's default set
of line magics and aliases (``_ipython_line_magic_names``: the line magics of IPython's own magics classes, of
storemagic and of ipykernel's ``KernelMagics``, the default shell aliases and the magic aliases ``ed``, ``hist`` and
``rep`` a shell registers at start-up) is reported whatever an earlier cell assigns, imports or defines, because
IPython's automagic runs it unless that name is bound when the cell runs; only a first word that is a Python builtin or
keyword, or one followed by ``=`` or ``,``, is not reported. R0 reads each cell as text: it does not read a rewrite
IPython decides at run time from the namespace or its configuration, such as autocall (a one-line cell that starts with
an autocall object runs as a call, so ``exit -1`` runs as ``exit(-1)``) or an alias or magic a notebook defines while it
runs; R12 reports the IPython shell names a notebook would write to set those up. A cell that does not parse as written
and mentions Django, a database, a process, a shell command or the environment (an ``import`` or ``from`` of ``django``,
``solsys_code`` or ``tom_*`` at the start of a line, an IPython shell escape, a process cell magic such as ``%%bash``, a
process or environment line magic such as ``%system`` or ``%env``, ``django.setup``, ``FOMO_DATABASE_PATH``,
``shutil.copy``, ``fomo_db.sqlite3``, ``connections.settings``, ``connections.databases``, ``manage.py``, ``env -i``,
``env -u`` or any name R12 lists) is reported by its own message. Every later rule reads each cell's Python view, so the
code under a harmless magic is read like any other code.

Django-backed (``is_django_backed``, D-13): a notebook is Django-backed when a code cell that parses as written (before
any magic line is removed), or code inside one of its strings, imports Django, the project app or a TOM plugin; a
notebook whose only such import sits in a cell under a harmless magic is not, and gets only the rules every notebook
gets. Every real notebook makes such an import in a cell that parses as written, as the not-vacuous test requires.

R2: exactly one top-level assignment of one name to a call of ``setup_scratch_django`` (the shape ``scratch_db_dir =
setup_scratch_django(repo_root_path, mode=...)``) after a top-level ``from fomo_nb_utils import`` of it (a star import
or an aliased helper import is reported), with one positional argument and a ``mode`` keyword whose value is the
constant ``'fresh'`` or ``'snapshot'``, and no other call of the setup function. R3: no top-level statement before that
call imports Django, the project app or a TOM plugin, nested in a block or in code inside its strings. R4: no call
written ``django.setup()``, no ``from django import setup`` and no ``import django as`` an alias, in a cell's code or in
code inside its strings. R5 and R7: no str or bytes constant in a cell's code (a bytes constant is read as latin-1)
contains ``FOMO_DATABASE_PATH`` or ``fomo_db.sqlite3``, and no keyword argument is named ``FOMO_DATABASE_PATH``. R8: the
code cells' execution counts are 1..N and no cell has an error output. R9: the setup cell's committed output has a
``Resolved database:`` line naming ``fomo_db.sqlite3`` in a ``fomo-notebook-db-*`` directory. R10: the last code cell
ends with a call of ``teardown_scratch_django``, imported from the helper, on the name the setup call assigned
(``teardown_scratch_django(scratch_db_dir)``), and its committed output contains ``Removed scratch database
directory:``. R11: no ``assert`` in the setup cell, and no ``assert`` whose test calls a write function
(WRITE_CALL_NAMES: call_command, save, create, delete, update, bulk_create, bulk_update, get_or_create,
update_or_create, post, put, patch, execute, executemany, write, write_text, unlink, rmtree), because ``python -O``
drops it.

R6, no byte copy or byte write, reads a cell's code and the code inside its strings, to two levels (STRING_CODE_DEPTH)
as R12 does. It reports the module names ``shutil`` and ``_shutil`` (tempfile's name for shutil), shutil's ``copy2``,
``copyfile``, ``copytree`` and ``copyfileobj``, and ``write_bytes`` at the identifiers ``_identifiers`` reads (listed
under R12 below) and as a '.'- or ':'-separated part of a str or bytes constant, so shutil handed to a parameter, a
lambda, a ``match`` capture or a class pattern is reported by its name; a cell has at most one write_bytes problem. It
also reports ``copy`` and ``move``, which are also common method names (a DataFrame's ``copy``), when read as an
attribute of an expression that mentions a shutil name, called or not, when imported with ``from shutil import``, and as
a string part (``getattr(shutil, 'move')``). A shutil name is ``shutil``, ``_shutil`` or a name the notebook binds to
one in any cell through ``import shutil as X``, ``import _shutil as X`` or ``from <any module> import shutil`` or
``_shutil`` (``[as X]``), or as the target of an assignment, a loop, a ``with``, a comprehension or a parameter default
whose value mentions a shutil name, where a mention is a name, an attribute or a string part, so ``zipfile.shutil`` and
``sys.modules['shutil']`` count.

R12 reports, in every notebook, the names on five lists of capabilities that can route around the scratch database
(WR-01, D-G1): Django's database settings and connection objects (``REROUTE_NAMES``: DATABASES, DATABASE_ROUTERS,
settings_dict, override_settings, modify_settings, the helper's internals developer_database_path,
prepare_scratch_database and snapshot_database, DatabaseWrapper, load_backend, ``connections``, ``connection``,
``_connections``, ConnectionHandler, create_connection, get_connection_params, get_new_connection, ``Database`` and
``sqlite3``, plus the attribute chains ``connections.settings`` and ``connections.databases`` and any ``sqlite3``
import); attaching another database through raw SQL (a string holding ATTACH followed by DATABASE, a quote or a
placeholder, or by an opening parenthesis, ``ATTACH (``); starting a process or reaching IPython's shell
(``PROCESS_MODULES``: subprocess, multiprocessing, pty, pexpect, ptyprocess, psutil, Jupyter's kernel clients
jupyter_client, nbclient and nbconvert, and IPython, ipykernel and traitlets; ``PROCESS_NAMES``: ``system``, ``popen``,
``fork``, ``forkpty``, ``startfile``, ``posix_spawn``, ``posix_spawnp``, os's ``exec*`` and ``spawn*`` functions, Popen,
ProcessPoolExecutor, asyncio's ``create_subprocess_exec`` and ``create_subprocess_shell``, ``getoutput``,
``getstatusoutput``, IPython's ``get_ipython``, ``system_raw``, ``run_line_magic`` and ``run_cell_magic``, the
IPython.utils helpers ``getoutputerror``, ``get_output_error_code`` and ``process_handler``, and IPython's shell classes
and entry points ``InteractiveShell``, ``ZMQInteractiveShell``, ``IPKernelApp``, ``alias_manager``, ``define_alias``,
``soft_define_alias``, ``run_cell``, ``run_cell_async``, ``autocall``, ``define_macro``, ``magics_manager`` and
``register_magic_function``); the environment (``ENVIRON_NAMES``: ``os.environ``, ``os.environb``, ``os.putenv``,
``os.unsetenv``; ``os.getenv`` reads are allowed); and running code from a string or another file
(``DYNAMIC_CODE_BUILTINS``: exec, eval, compile, ``__import__`` and ``__builtins__``; ``DYNAMIC_CODE_MODULES``:
importlib, runpy, builtins, ctypes and cffi; ``DYNAMIC_CODE_NAMES``: execute_from_command_line, ManagementUtility, the
lookup functions ``import_string``, ``resolve_name`` and ``locate``, and IPython's ``Macro`` and ``IPyAutocall``; and
``call_command`` of a ``CODE_RUNNING_COMMANDS`` command: shell, shell_plus or runscript). Each R12 name is matched at
the identifiers ``_identifiers`` reads -- a name, an attribute, a keyword argument, a parameter, a function, class or
exception name, a global or nonlocal name, a match capture, a match-mapping rest or a class-pattern attribute, a type
parameter, every part of an imported module's path, and every imported name and alias, with attributes and class-pattern
attributes never matched against the builtins (so ``re.compile`` stays allowed) -- and against every '.'- or
':'-separated part of a str or bytes constant. A string that is itself Python code (it parses and holds a statement or a
call) is read like the cell's own code by R3, R4, R6, R12 and the Django-backed gate, to two levels
(``STRING_CODE_DEPTH``). So importing a listed name from another module (``from webbrowser import subprocess``), binding
it to another name first (``c = connections``, ``run = os.system``), reaching IPython's shell as
``InteractiveShell.instance()`` or looking a name up by a literal string (``import_string('subprocess.run')``) is
reported too. The guard makes no claim of completeness.

What the guard does not see: it reports only the names and patterns its lists contain, in the notebook's own literal
source, so any route to a process, IPython's shell, Django's settings or the database through a name or form the lists
do not contain is out of its reach, deliberate obfuscation included. Such routes include (1) values assembled or looked
up at runtime: a name, module, SQL statement or file path built from pieces (string concatenation, f-strings, joined
path parts, ``str.join``, ``chr``) or fetched with ``getattr``, ``globals()``, ``vars()`` or ``sys.modules`` using such
a string; a module, function or method found at runtime without its name being written (``inspect.getmodule``, a
function's ``__module__``, a search over ``sys.modules`` or a module's ``vars()``); and code decoded at runtime (pickle,
marshal, codecs, base64); (2) code in a string nested more than two strings deep, which R6 and R12 match only by its
'.'- and ':'-separated parts and R3, R4 and the Django-backed gate do not read; (3) rewrites IPython decides at run
time, which R0 does not read: autocall of an autocall object (``exit -1`` runs as ``exit(-1)``) and a one-line cell that
runs an alias or magic the notebook defined while it ran (R12 reports the shell names such a definition writes); (4)
code outside the notebook: a module it imports, IPython's profile and start-up files (with the extensions, aliases and
magics they define), or a library function that starts a process, sets Django up or writes a file under a name these
lists do not contain (for example a path's ``replace`` or ``rename``, ``open`` in a write mode, a path's ``copy`` and
``move`` on Python 3.14 and later, or a ``webbrowser`` controller given a command line); and (5), left out on purpose,
JavaScript or HTML in a cell's output, which a notebook front end may run when the notebook is opened; an nbconvert
regeneration does not run it.
~~~

## Function docstrings, class docstrings and comments the sweep changed

- Module docstring (WHY finding list gains the sixth review; RULE paragraph; the five "What the guard checks" paragraphs; the residual paragraph).
- Function docstrings: `_ipython_line_magic_names`, `_automagic_word`, `_string_code_trees`, `is_django_backed`, `_django_setup_problems`, `_shutil_names`, `_byte_copy_problems`, `_identifiers`, `_database_rerouting_problems`, `isolation_problems`.
- Class docstrings: `TestCapabilityRulesOnSyntheticNotebooks`, `TestR12MatchesEveryIdentifierAndString`, `TestR6CopyAndWriteFunctions`, `TestR0AllowList`.
- Comments: fourteen new or changed comment lines, among them the `STRING_CODE_DEPTH` comment (which rules read string code), the R0 comments (autocall and notebook-defined aliases as run-time rewrites R0 does not read), the R3 and R6 rule comments, the shutil-name comments that said "wherever", and the autocall test's comment that said "the kernel's only autocall objects".
- The full diff of the candidate against 40-17's guard (504 lines) is in `$HOME/tmp/phase40-18-draft.md` (scratch); the committed diff is `git diff ca1cb7d aff2547 -- solsys_code/tests/test_pre_executed_notebooks.py`.

## Recorded README guard paragraph (verbatim)

> `solsys_code/tests/test_pre_executed_notebooks.py` checks every notebook in `pre_executed/`. It reports only the names
> and patterns on its lists, found in a notebook's literal source: its code and imports, the '.'- or ':'-separated parts
> of its str and bytes literals, and code inside its strings (read to two levels). It reports a notebook that does not
> call the helper first and once, with a constant mode; imports Django before that call or calls `django.setup()` itself,
> also in code inside a string; names `FOMO_DATABASE_PATH` or the developer database file in a str or bytes literal, or
> passes a `FOMO_DATABASE_PATH` keyword argument; names `shutil`, tempfile's `_shutil`, shutil's `copy2`, `copyfile`,
> `copytree` or `copyfileobj`, or `write_bytes`, or reads `copy` or `move` from a shutil name; names one of the listed
> ways to choose or reach a database, start a process, reach IPython's shell (`get_ipython`, `InteractiveShell`, the
> IPython, ipykernel and traitlets modules, and the shell's alias, `run_cell` and `autocall` entry points), change the
> environment, or run code built from a string or loaded from another file; has a cell that is not Python IPython runs
> unchanged, apart from `%matplotlib inline` and a first-line `%%time` or `%%capture`, or a one-line cell that starts with
> one of IPython's default magic or alias names, unless that name is a Python builtin or keyword or is followed by `=` or
> `,`; has execution counts other than 1..N or an error output; has no `Resolved database:` line in the setup cell's
> output; has an `assert` in the setup cell or one wrapping a write; or does not end with the teardown call and its
> output. A route to a process, IPython's shell, Django's settings or the database through a name or form the lists do not
> contain is not reported. Its module docstring lists every name, says what a static check of a notebook's source cannot
> see, and makes no claim of completeness. It reads each notebook's JSON and never executes one.

## Recorded override reason (verbatim)

> The guard reports only the names and patterns on its lists, found in a notebook's literal source, and makes no claim of completeness; the guard docstring lists every name. Rule R12, extended in 40-08 [301bd93, f1b4bb6], 40-10 [482606b, 3232341, c6c6322], 40-15 [1c830b1] and 40-17 [c78e517], reports in every notebook the names on its five lists: Django's database settings and connection objects (DATABASES, DATABASE_ROUTERS, settings_dict, override_settings, modify_settings, the helper's internals developer_database_path, prepare_scratch_database and snapshot_database, DatabaseWrapper, load_backend, connections, connection, _connections, ConnectionHandler, create_connection, get_connection_params, get_new_connection, Database and sqlite3, the connections.settings and connections.databases attribute chains and any sqlite3 import); attaching another database through raw SQL (a string holding ATTACH followed by DATABASE, a quote, a placeholder or an opening parenthesis); starting a process or reaching IPython's shell (subprocess, multiprocessing, pty, pexpect, ptyprocess, psutil, jupyter_client, nbclient, nbconvert, IPython, ipykernel, traitlets, system, popen, fork, forkpty, startfile, posix_spawn, posix_spawnp, os's exec* and spawn* functions, Popen, ProcessPoolExecutor, create_subprocess_exec, create_subprocess_shell, getoutput, getstatusoutput, get_ipython, system_raw, run_line_magic, run_cell_magic, getoutputerror, get_output_error_code, process_handler, InteractiveShell, ZMQInteractiveShell, IPKernelApp, alias_manager, define_alias, soft_define_alias, run_cell, run_cell_async, autocall, define_macro, magics_manager and register_magic_function); the environment (environ, environb, putenv, unsetenv); and running code from a string or another file (exec, eval, compile, __import__, __builtins__, importlib, runpy, builtins, ctypes, cffi, execute_from_command_line, ManagementUtility, import_string, resolve_name, locate, Macro, IPyAutocall, and call_command of shell, shell_plus or runscript). Each name is matched at the identifiers the guard docstring lists (attributes never against the builtins exec, eval, compile, __import__ and __builtins__) and against every '.'- or ':'-separated part of a str or bytes constant, and a string that parses and holds a statement or a call is read like the cell's own code, to two levels, so reaching IPython's shell as InteractiveShell.instance() or a process by a literal dotted string (import_string('subprocess.run')) is reported. R3 and R4 (40-17 [fdd8b3e]) report a Django import before the helper call and a call written django.setup(), an import of setup from django or django imported under an alias, in a cell's code and in code inside its strings to two levels, and the Django-backed gate reads string code the same way. R0 reports a cell that does not parse once %matplotlib inline lines and a first-line %%time or %%capture (with only its --no-stderr, --no-stdout and --no-display flags and an output name) are removed, a cell whose meaning IPython's static TransformerManager changes, and, since 40-13 [905408a], a one-line cell whose first word is in IPython's default line-magic and alias set, whatever an earlier cell assigns, imports or defines, unless the word is a builtin or keyword or is followed by = or a comma. R5 and R7 report a str or bytes constant that contains FOMO_DATABASE_PATH or fomo_db.sqlite3 and a keyword argument named FOMO_DATABASE_PATH (40-13 [4f02628]). R6 reports shutil, _shutil, copy2, copyfile, copytree, copyfileobj and write_bytes at the identifiers the docstring lists and as string parts (40-13 [4f02628], 40-15 [bc99b0e]), and copy and move read from an expression that mentions shutil, _shutil or a name the notebook binds to one through an import, an assignment, a loop, a with, a comprehension or a parameter default (40-17 [ca1cb7d]); a cell has at most one write_bytes problem (40-17 [ca1cb7d]). Every literal form in the third review (CR-01, CR-02, WR-01), the fourth review's CR-01 and WR-01 forms, the fifth review's CR-01 and CR-02 forms, the sixth review's CR-01 and CR-02 forms, the re-review's four forms and both round-1 probes is reported on a copy of a real notebook, the fourth review's WR-02 bytes literals on the reference notebook, and every real pre-executed notebook passes the guard and is Django-backed. Any route to a process, IPython's shell, Django's settings or the database through a name or form the lists do not contain is not seen, including values assembled or looked up at runtime (a name, module, SQL statement or file path built from pieces, or fetched with getattr, globals(), vars() or sys.modules using such a string, and a module, function or method found at runtime without its name, through inspect.getmodule, a function's __module__ or a search over sys.modules or a module's vars()), code decoded at runtime (pickle, marshal, codecs, base64), code nested more than two strings deep, rewrites IPython decides at run time (autocall, so exit -1 runs as exit(-1), and a one-line cell that runs an alias or magic the notebook defined while it ran), code outside the notebook (a module it imports, IPython's profile and start-up files with the extensions, aliases and magics they define, or a library function that starts a process, sets Django up or writes a file under a name these lists do not contain, such as a path's copy and move on Python 3.14 and later) and, left out on purpose, JavaScript or HTML in a cell's output, which a notebook front end may run when the notebook is opened.

## Evidence table as presented to the developer

| Claim | Evidence |
|-------|----------|
| The guard reports only the names and patterns on its lists, found in a notebook's literal source | docstring: "reports only the names and patterns listed here, found in a notebook's literal source" |
| The guard makes no claim of completeness | docstring: "The guard makes no claim of completeness." |
| The guard docstring lists every name | docstring: "R12 reports, in every notebook, the names on five lists of capabilities that can route around the scratch database" |
| R12 applies to every notebook, Django-backed or not | test: TestRerouteRuleOnSyntheticNotebooks.test_the_rule_applies_to_a_notebook_that_does_not_import_django |
| Extended in 40-08 [301bd93, f1b4bb6]: the re-review forms and whole capabilities | test: TestReReviewFormsOnARealNotebook.test_a_connection_handler_store_is_reported_on_a_real_notebook, TestCapabilityRulesOnSyntheticNotebooks.test_r12_name_lists_are_disjoint_and_reach_r0 |
| Extended in 40-10 [482606b, 3232341, c6c6322]: identifiers, strings, R0 allow-list | test: TestR12MatchesEveryIdentifierAndString.test_a_listed_name_is_reported_wherever_a_name_appears, TestR0AllowList.test_a_cell_is_a_problem_unless_it_is_python_that_ipython_runs_unchanged |
| Extended in 40-15 [1c830b1]: IPyAutocall is an R12 name, listed with Macro | test: TestR0AllowList.test_autocall_is_not_seen_by_r0_and_the_ipyautocall_name_is_reported |
| Extended in 40-17 [c78e517]: IPython, ipykernel, traitlets and IPython's shell classes and entry points are listed, and the review's four shell forms are reported on a copy of a real notebook | test: TestSixthReviewFormsOnARealNotebook.test_ipython_shell_reached_without_get_ipython_is_reported_on_a_real_notebook |
| Database settings and connection objects, as listed (REROUTE_NAMES) | test: TestRerouteRuleOnSyntheticNotebooks.test_settings_dict_in_any_spelling_is_a_problem, TestRerouteRuleOnSyntheticNotebooks.test_databases_or_routers_in_any_spelling_is_a_problem, TestRerouteRuleOnSyntheticNotebooks.test_connection_handler_settings_are_a_problem, TestRerouteRuleOnSyntheticNotebooks.test_override_or_modify_settings_is_a_problem, TestRerouteRuleOnSyntheticNotebooks.test_helper_internals_are_a_problem, TestRerouteRuleOnSyntheticNotebooks.test_sqlite3_or_a_backend_wrapper_is_a_problem, TestCapabilityRulesOnSyntheticNotebooks.test_replacing_a_database_connection_in_any_spelling_is_a_problem |
| connection, _connections and Database are listed; the connections.settings and connections.databases chains and any sqlite3 import | test: TestR12MatchesEveryIdentifierAndString.test_the_connection_proxy_and_attach_with_a_parenthesis_are_a_problem, TestThirdReviewFormsOnARealNotebook.test_the_connection_proxy_database_connect_and_attach_are_reported_on_a_real_notebook |
| Raw SQL ATTACH followed by DATABASE, a quote, a placeholder or an opening parenthesis | docstring: "a string holding ATTACH followed by DATABASE, a quote or a placeholder, or by an opening parenthesis" |
| Raw SQL ATTACH is reported | test: TestCapabilityRulesOnSyntheticNotebooks.test_attaching_another_database_through_raw_sql_is_a_problem |
| Starting a process or reaching IPython's shell, as listed (PROCESS_MODULES, PROCESS_NAMES) | test: TestCapabilityRulesOnSyntheticNotebooks.test_starting_a_process_in_any_spelling_is_a_problem, TestCapabilityRulesOnSyntheticNotebooks.test_ipython_shell_access_in_a_parseable_cell_is_a_problem |
| IPython.utils helpers getoutputerror, get_output_error_code and process_handler | test: TestThirdReviewFormsOnARealNotebook.test_ipython_process_helpers_are_reported_on_a_real_notebook |
| The environment (environ, environb, putenv, unsetenv) | test: TestCapabilityRulesOnSyntheticNotebooks.test_changing_the_environment_in_any_spelling_is_a_problem |
| Running code from a string or another file, as listed | test: TestCapabilityRulesOnSyntheticNotebooks.test_running_code_from_a_string_or_another_file_is_a_problem |
| import_string, resolve_name, locate, Macro and IPyAutocall are listed | test: TestR12MatchesEveryIdentifierAndString.test_a_listed_name_inside_a_string_or_bytes_constant_is_a_problem |
| Each name is matched at the identifiers the guard docstring lists | docstring: "Each R12 name is matched at the identifiers ``_identifiers`` reads" |
| Every identifier position is matched | test: TestR12MatchesEveryIdentifierAndString.test_a_listed_name_is_reported_wherever_a_name_appears |
| Attributes are never matched against the builtins exec, eval, compile, __import__ and __builtins__ | docstring: "never matched against the builtins (so ``re.compile`` stays allowed)" |
| Every dotted or colon-separated part of a str or bytes constant is matched | docstring: "against every '.'- or ':'-separated part of a str or bytes constant" |
| A string that parses and holds a statement or a call is read like the cell's own code, to two levels | docstring: "A string that is itself Python code (it parses and holds a statement or a call) is read like the cell's own code by R3, R4, R6, R12 and the Django-backed gate, to two levels" |
| String constants and string code are read | test: TestR12MatchesEveryIdentifierAndString.test_a_listed_name_inside_a_string_or_bytes_constant_is_a_problem, TestR12MatchesEveryIdentifierAndString.test_code_inside_a_string_is_checked_like_code |
| Look-alikes stay allowed | test: TestR12MatchesEveryIdentifierAndString.test_names_that_only_look_like_listed_ones_are_not_a_problem, TestCapabilityRulesOnSyntheticNotebooks.test_allowed_look_alikes_are_not_a_problem |
| Reaching IPython's shell as InteractiveShell.instance() is reported | docstring: "reaching IPython's shell as ``InteractiveShell.instance()``" |
| InteractiveShell.instance() with an alias then a one-line cell, run_cell, autocall and ex, on a copy of a real notebook | test: TestSixthReviewFormsOnARealNotebook.test_ipython_shell_reached_without_get_ipython_is_reported_on_a_real_notebook |
| from webbrowser import subprocess and import_string('subprocess.run') are reported on a real notebook | test: TestThirdReviewFormsOnARealNotebook.test_subprocess_imported_from_another_module_is_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_a_process_looked_up_by_a_literal_string_is_reported_on_a_real_notebook |
| R3 reports a Django import before the helper call, in code inside a string | docstring: "R3: no top-level statement before that call imports Django, the project app or a TOM plugin, nested in a block or in code inside its strings" |
| R4 reports django.setup(), an import of setup from django and django imported under an alias, in code and in code inside strings | docstring: "R4: no call written ``django.setup()``, no ``from django import setup`` and no ``import django as`` an alias, in a cell's code or in code inside its strings" |
| The review's timeit probe and the verifier's settings.configure variant are reported on a copy of a real notebook | test: TestSixthReviewFormsOnARealNotebook.test_django_set_up_in_string_code_before_the_helper_is_reported_on_a_real_notebook |
| The Django-backed gate reads string code the same way (docstring) | docstring: "or code inside one of its strings, imports Django" |
| The Django-backed gate reads string code the same way (a bare notebook whose only Django import is in string code) | test: TestSixthReviewFormsOnARealNotebook.test_a_notebook_whose_only_django_import_is_in_string_code_is_django_backed |
| The gate still reads each cell as written | test: TestR0AllowList.test_the_django_backed_gate_reads_each_cell_as_written |
| R0 reports a cell that does not parse once the harmless magic lines are removed, and a cell whose meaning TransformerManager changes | docstring: "IPython's static input transformer (``TransformerManager``) must leave its meaning unchanged" |
| The harmless magic lines and the %%capture flags | docstring: "a first-line ``%%time`` or ``%%capture`` (with only its ``--no-stderr``, ``--no-stdout`` and ``--no-display`` flags and an output name) are removed" |
| R0 allow-list, harmless magics, code under them checked | test: TestR0AllowList.test_a_cell_is_a_problem_unless_it_is_python_that_ipython_runs_unchanged, TestR0AllowList.test_code_under_a_harmless_magic_is_checked_by_every_rule, TestThirdReviewFormsOnARealNotebook.test_harmless_magic_lines_leave_a_real_notebook_clean |
| Since 40-13 [905408a], a one-line cell whose first word is in IPython's default line-magic and alias set is reported | docstring: "is in IPython's default set of line magics and aliases" |
| Whatever an earlier cell assigns, imports or defines | docstring: "is reported whatever an earlier cell assigns, imports or defines" |
| Seven bindings of rm before an alias cell on a copy of a real notebook, each binding alone clean | test: TestR0AllowList.test_a_one_line_magic_or_alias_is_a_problem_even_when_an_earlier_cell_binds_the_name |
| Only a builtin, a keyword or a following = or a comma is exempt | docstring: "only a first word that is a Python builtin or keyword, or one followed by ``=`` or ``,``, is not reported" |
| The automagic set matches a fresh IPython shell | test: TestR0AllowList.test_the_automagic_rule_knows_every_line_magic_a_fresh_ipython_shell_has |
| R5 and R7 report a str or bytes constant that contains FOMO_DATABASE_PATH or fomo_db.sqlite3 (40-13 [4f02628]) | test: TestIsolationRulesOnSyntheticNotebooks.test_a_bytes_literal_naming_the_developer_database_or_its_variable_is_a_problem, TestIsolationRulesOnSyntheticNotebooks.test_developer_database_literal_in_code_is_a_problem |
| A keyword argument named FOMO_DATABASE_PATH is reported | docstring: "no keyword argument is named ``FOMO_DATABASE_PATH``" |
| R6 reports shutil, _shutil, copy2, copyfile, copytree, copyfileobj and write_bytes at the identifiers and as string parts (40-13 [4f02628], 40-15 [bc99b0e]) | test: TestR6CopyAndWriteFunctions.test_every_copy_and_write_function_in_any_spelling_is_a_problem, TestR6CopyAndWriteFunctions.test_shutil_and_tempfile_shutil_are_reported_wherever_they_are_named, TestR6CopyAndWriteFunctions.test_shutil_passed_on_in_any_way_is_reported_on_a_real_notebook, TestR6CopyAndWriteFunctions.test_write_bytes_at_any_identifier_is_reported_on_a_real_notebook |
| R6 reports the module names shutil and _shutil | docstring: "It reports the module names ``shutil`` and ``_shutil``" |
| copy and move read from an expression that mentions a shutil name, through an import, an assignment, a loop, a with, a comprehension or a parameter default | test: TestR6CopyAndWriteFunctions.test_a_rebound_shutil_or_a_distinctive_copy_name_is_a_problem, TestR6CopyAndWriteFunctions.test_a_shutil_name_bound_in_another_cell_or_in_string_code_is_followed |
| An import of tempfile's _shutil is followed (40-17 [ca1cb7d]) | test: TestR6CopyAndWriteFunctions.test_a_name_bound_to_tempfile_shutil_by_an_import_is_followed |
| A cell has at most one write_bytes problem (40-17 [ca1cb7d]) | test: TestR6CopyAndWriteFunctions.test_a_cell_has_at_most_one_write_bytes_problem |
| A cell has at most one write_bytes problem (docstring) | docstring: "a cell has at most one write_bytes problem" |
| Look-alikes (a DataFrame copy, a name not bound to shutil) are not reported | test: TestR6CopyAndWriteFunctions.test_copy_and_move_not_read_from_shutil_are_not_a_problem |
| The third review's CR-01 forms on a real notebook | test: TestThirdReviewFormsOnARealNotebook.test_subprocess_imported_from_another_module_is_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_a_process_looked_up_by_a_literal_string_is_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_ipython_process_helpers_are_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_the_connection_proxy_database_connect_and_attach_are_reported_on_a_real_notebook |
| The third review's CR-02 and WR-01 forms on a real notebook | test: TestThirdReviewFormsOnARealNotebook.test_a_shell_escape_after_a_line_magic_is_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_ipython_aliases_and_shell_magics_are_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_a_one_line_alias_or_magic_that_is_valid_python_is_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_copytree_and_write_bytes_are_reported_on_a_real_notebook |
| The fourth review's CR-01 forms (seven bindings of rm) and WR-01 forms (both rebinding forms) on a copy of a real notebook | test: TestR0AllowList.test_a_one_line_magic_or_alias_is_a_problem_even_when_an_earlier_cell_binds_the_name, TestR6CopyAndWriteFunctions.test_a_rebound_shutil_or_a_distinctive_copy_name_is_a_problem |
| The fifth review's CR-01 and CR-02 forms on a copy of a real notebook | test: TestR6CopyAndWriteFunctions.test_shutil_passed_on_in_any_way_is_reported_on_a_real_notebook, TestR6CopyAndWriteFunctions.test_write_bytes_at_any_identifier_is_reported_on_a_real_notebook |
| The sixth review's CR-01 and CR-02 forms on a copy of a real notebook | test: TestSixthReviewFormsOnARealNotebook.test_ipython_shell_reached_without_get_ipython_is_reported_on_a_real_notebook, TestSixthReviewFormsOnARealNotebook.test_django_set_up_in_string_code_before_the_helper_is_reported_on_a_real_notebook |
| The re-review's four forms on a real notebook | test: TestReReviewFormsOnARealNotebook.test_a_connection_handler_store_is_reported_on_a_real_notebook, TestReReviewFormsOnARealNotebook.test_a_child_manage_py_with_a_bare_environment_is_reported_on_a_real_notebook, TestReReviewFormsOnARealNotebook.test_a_cleared_environment_and_a_child_manage_py_are_reported_on_a_real_notebook, TestReReviewFormsOnARealNotebook.test_a_shell_escape_running_manage_py_is_reported_on_a_real_notebook |
| Both round-1 probes on a real notebook | test: TestRerouteRuleOnARealNotebook.test_a_settings_dict_reroute_after_setup_is_reported_on_a_real_notebook, TestRerouteRuleOnARealNotebook.test_an_override_settings_reroute_after_setup_is_reported_on_a_real_notebook |
| The fourth review's WR-02 bytes literals on the reference notebook | test: TestIsolationRulesOnSyntheticNotebooks.test_a_bytes_literal_naming_the_developer_database_or_its_variable_is_a_problem |
| Every real pre-executed notebook passes the guard and is Django-backed | test: TestPreExecutedNotebooksUseTheHelper.test_every_real_pre_executed_notebook_is_isolated, TestPreExecutedNotebooksUseTheHelper.test_the_real_notebook_check_is_not_vacuous, TestRerouteRuleOnARealNotebook.test_the_unmodified_real_notebook_has_no_problems |
| Any route through a name or form the lists do not contain is not seen | docstring: "any route to a process, IPython's shell, Django's settings or the database through a name or form the lists do not contain is out of its reach" |
| Residual: values assembled or looked up at runtime | docstring: "values assembled or looked up at runtime" |
| Residual: a module, function or method found at runtime without its name (inspect.getmodule, __module__, sys.modules, vars()) | docstring: "(``inspect.getmodule``, a function's ``__module__``, a search over ``sys.modules`` or a module's ``vars()``)" |
| Residual: code decoded at runtime | docstring: "and code decoded at runtime (pickle, marshal, codecs, base64)" |
| Residual: code nested more than two strings deep | docstring: "code in a string nested more than two strings deep" |
| Residual: rewrites IPython decides at run time (autocall) | docstring: "rewrites IPython decides at run time, which R0 does not read" |
| Residual: exit -1 runs as exit(-1); autocall has no R0 problem and the IPyAutocall name is reported | test: TestR0AllowList.test_autocall_is_not_seen_by_r0_and_the_ipyautocall_name_is_reported |
| Residual: a one-line cell that runs an alias or magic the notebook defined while it ran | docstring: "a one-line cell that runs an alias or magic the notebook defined while it ran" |
| Residual: code outside the notebook (a module it imports, IPython's profile and start-up files, a library function) | docstring: "code outside the notebook" |
| Residual: a path's copy and move on Python 3.14 and later | docstring: "path's ``copy`` and ``move`` on Python 3.14 and later" |
| Residual: JavaScript or HTML in a cell's output, left out on purpose | docstring: "JavaScript or HTML in a cell's output" |

## Ledger rows as recorded

- CR-01 (critical, fixed): fixed in 40-17 Task 1 (c78e517) and 40-18 Task 3 (aff2547): R12 lists IPython, ipykernel, traitlets and IPython's shell classes and entry points.
- CR-02 (critical, fixed): fixed in 40-17 Task 2 (fdd8b3e) and 40-18 Task 3 (aff2547): R3, R4 and the Django-backed gate read code inside strings to two levels.
- WR-01 (warning, fixed): fixed in 40-17 Task 3 (ca1cb7d) and 40-18 Task 3 (aff2547): a cell has at most one write_bytes problem.
- IN-01 (info, fixed): fixed in 40-17 Task 3 (ca1cb7d) and 40-18 Task 3 (aff2547): `_shutil_names` follows an import of `_shutil`.
- IN-02 (info, fixed): fixed in 40-18 Task 3 (aff2547): no docstring, README or reason says exit and quit end the kernel.
- IN-03 (info, fixed): fixed in 40-18 Task 3 (aff2547): README and docstring list every rule (D-G18).
- Front matter: `open: 8` became `open: 2` (WR-03 and IN-04 stay open), `total: 12`; exactly 13 lines differ from cd883a6, so every carried row and every "ID reuse across rounds" section is unchanged.

## Verification results

- Guard, helper and projector tests: `Ran 131 tests ... OK` (`test_pre_executed_notebooks`, `test_notebook_helper`, `test_projector_demo_notebook`).
- Check 2 (approved-module equality): `OK: the guard module is the approved candidate byte for byte, its code identical to 40-17's; the README guard paragraph is the approved text and the README is otherwise unchanged`.
- Check 3: `OK: ruff clean; the docs commit holds exactly the guard module and README (aff254725f53d61a7388209ea19527c0ad38f25e) and follows 40-17`.
- Check 4: `OK: CR-01, CR-02, WR-01 and IN-01 fixed citing a 40-17 commit and the 40-18 docs commit, IN-02 and IN-03 fixed citing the docs commit, open 8 -> 2, total 12, exactly 13 lines changed ...`.
- Check 5: `OK: approve -- the approved reason and accepted_at 2026-10-11T13:37:50Z recorded in the one override entry; must_have, accepted_by, every other key and the body unchanged; the override commit is HEAD, after the ledger commit`.
- `git status --short src/` is empty; nothing pushed; `src/fomo_db.sqlite3` untouched; no notebook executed; no IPython shell started.

## Decisions Made

- Recorded the developer-approved wording exactly as drafted; no amendment, no decline.
- Carried out D-G17 and D-G18: each rule is described by what its lists contain, the guard "reports only those names and patterns in a notebook's literal source", and the residual is one broad class with named examples.
- IN-02: dropped the "end the kernel" sentence and described autocall as a run-time rewrite R0 does not read.
- The SUMMARY is committed with the STATE and ROADMAP updates (resume instruction), after the override commit, so Task 3's last check (override commit is HEAD) was run before this commit and cannot be rerun after it.

## Deviations from Plan

None - plan executed exactly as written. Two notes that are not deviations: the plan said to leave the SUMMARY uncommitted for the orchestrator's metadata commit; the resume instructions from the orchestrator say to commit it with the STATE and ROADMAP updates, which this plan's executor did. The README paragraph was inserted from the approved text without re-wrapping (it was already at most 120 columns).

## Issues Encountered

None.

## Known Stubs

None. The files changed are docstrings, comments, README prose and planning records.

## Threat Flags

None. No new network endpoint, auth path, file access pattern or schema change; T-40-65 (wording recorded without approval or claiming more than the code keeps) and T-40-66 (edits outside the one override entry) are mitigated by the blocking-human checkpoint, the byte-for-byte check against the approved module and the front-matter and body comparison with cd883a6.

## Next Phase Readiness

- The verifier applies or rejects the override at re-verification, testing every review route against the reason's exact words. Task 3's last check can pass only while the override (or ledger) commit is HEAD, so later SUMMARY and tracking commits do not invalidate it.
- Verification human items 3 and 4 stay /gsd-verify-work reading items (D-G15); item 2 (reading the README and docstring after the fix) was covered by this plan's checkpoint.
- Still open in the ledger and out of scope (D-G14): WR-03, IN-04, the fifth review's IN-02, the fourth review's IN-01, round-2 IN-03, the IN-07 remainder.

---
*Phase: 40-notebook-isolation-attribution-page*
*Completed: 2026-10-11*

## Self-Check: PASSED

- Commits aff2547, 226acc0, 0f5b540 are ancestors of HEAD.
- 40-18-SUMMARY.md, the guard module, README, ledger and verification report exist; all five task checks printed their OK lines.
