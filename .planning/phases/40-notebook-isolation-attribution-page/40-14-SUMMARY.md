---
phase: 40-notebook-isolation-attribution-page
plan: 14
subsystem: testing
tags: [notebooks, guard, gap-closure, documentation, override, review-ledger]
gap_closure: true
gap_ids: [CR-01, WR-01, WR-02, IN-02]
status: complete

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-13 guard code (R0 automagic rule without the binding exemption, R6 distinctive copy names and shutil-name following, R5/R7 bytes constants, ipykernel in the dev extra)"
provides:
  - "Guard module docstring and docs/notebooks/README.md guard paragraph that state exactly what 40-13's code checks and does not follow (developer-approved wording)"
  - "40-VERIFICATION.md override reason reworded to the guard as 40-13 left it, accepted by Tim Lister at 2026-10-10T23:20:47Z"
  - "40-REVIEW-DISPOSITION.md: CR-01, WR-01, WR-02 and IN-02 fixed, IN-01 open and out of scope (open 8 -> 4)"
affects: [verify-work, 40-re-verification]

actuals:
  tokens: 2626      # chars/4 over the added and removed lines of the three realized diffs; excludes this SUMMARY
  tasks: 3
  commits: 3
plan_head_before: 62053fc
plan_head_after: 8408731

tech-stack:
  added: []
  patterns:
    - "Documentation of a static check is itself checked: every claim in a drafted docstring, README paragraph or override reason is proven against the committed guard by inert calls on placeholder source before a human approves it"

key-files:
  created:
    - .planning/phases/40-notebook-isolation-attribution-page/40-14-SUMMARY.md
  modified:
    - solsys_code/tests/test_pre_executed_notebooks.py
    - docs/notebooks/README.md
    - .planning/phases/40-notebook-isolation-attribution-page/40-REVIEW-DISPOSITION.md
    - .planning/phases/40-notebook-isolation-attribution-page/40-VERIFICATION.md

key-decisions:
  - "D-G7 (3): developer (Tim Lister) approved the drafted docstring, README paragraph and override reason as drafted at the blocking-human checkpoint (decision approve, approved_at 2026-10-10T23:20:47Z)"
  - "IN-01 stays open with an out-of-scope note in its Source cell (developer decision 2026-10-10), recorded for Phase 41"

requirements-completed: [WARN-02, WARN-05]

coverage:
  - id: D1
    description: "The guard docstring and README guard paragraph equal the developer-approved text and the guard code is identical to 40-13's last guard commit"
    requirement: "WARN-02"
    verification:
      - kind: other
        ref: "40-14-PLAN.md Task 3 first <automated> python script (printed 'OK: the module docstring and README guard paragraph are the approved texts ...')"
        status: pass
      - kind: unit
        ref: "python manage.py test solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_notebook_helper solsys_code.tests.test_projector_demo_notebook (Ran 121 tests, OK)"
        status: pass
    human_judgment: false
  - id: D2
    description: "The ledger records CR-01, WR-01, WR-02 and IN-02 fixed with real commits and IN-01 open with its out-of-scope note; exactly 10 lines changed"
    requirement: "WARN-02"
    verification:
      - kind: other
        ref: "40-14-PLAN.md Task 3 third <automated> python script (printed 'OK: CR-01, WR-01, WR-02 and IN-02 fixed ...')"
        status: pass
    human_judgment: false
  - id: D3
    description: "The override entry's reason and accepted_at equal the approved text and the orchestrator-supplied time; every other key and the body are unchanged"
    requirement: "WARN-02"
    verification:
      - kind: other
        ref: "40-14-PLAN.md Task 3 fourth <automated> python script (printed 'OK: approve -- the approved reason and accepted_at 2026-10-10T23:20:47Z recorded ...')"
        status: pass
    human_judgment: false
  - id: D4
    description: "Whether the reworded override applies to ROADMAP SC 2 (first half) is the verifier's judgment at re-verification; verification human items 2-4 are /gsd-verify-work reading items"
    requirement: "WARN-02"
    human_judgment: true
    rationale: "Recording a reason does not apply the override (D-G1, D-G10); the verifier checks it against the guard at re-verification"

duration: 12min (this continuation; Tasks 1 and 2 ran in an earlier executor)
completed: 2026-10-10
---

# Phase 40 Plan 14: Docstring, README and override wording for the round-5 guard Summary

**The guard module docstring, the README guard paragraph and the 40-VERIFICATION.md override reason now say exactly what 40-13's code checks (automagic rule with no binding exemption, R5/R7 over bytes, R6 distinctive copy names everywhere plus shutil-name following) and name only RES and R6's stated boundary as residual, in wording Tim Lister approved.**

## Performance

- **Duration:** this continuation 12 min (2026-10-10T23:21Z to 23:33Z); Task 1 (draft and check) and Task 2 (checkpoint) ran in the previous executor and the orchestrator
- **Tasks:** 3 of 3 (Task 1 tracer, Task 2 blocking-human decision, Task 3 record)
- **Files modified:** 4 (guard module docstring only, README paragraph, ledger, verification front matter) plus this SUMMARY

## Accomplishments

- Task 1 (tracer): the draft at `$HOME/tmp/phase40-14-draft.md` was proven against 40-13's committed guard. The draft check printed: `OK: draft has a splicable docstring (RES kept, code unchanged), a README paragraph and a reason of 8 sentences with 52 evidence rows; every claim checked against the committed guard; no banned phrase; the reason cites 40-13 guard commits; nothing in the repository changed`.
- Task 2 (checkpoint:decision, blocking-human): the developer replied `approve`, approval time `2026-10-10T23:20:47Z` (supplied by the orchestrator, written verbatim to `$HOME/tmp/phase40-14-approved-at.txt`; reply saved to `$HOME/tmp/phase40-14-decision.txt`).
- Task 3: three commits in the required order, the override commit last.

## Decision and approval

- **Decision:** approve (all three texts as drafted), by Tim Lister.
- **Approval time (accepted_at):** `2026-10-10T23:20:47Z`.

## Commits

| Commit | Kind | Files |
| ------ | ---- | ----- |
| 1a315e2 | docs, full pre-commit hook including the Django tests | `solsys_code/tests/test_pre_executed_notebooks.py` (module docstring only), `docs/notebooks/README.md` |
| 45199cc | ledger, `SKIP=django-test` | `.planning/phases/40-notebook-isolation-attribution-page/40-REVIEW-DISPOSITION.md` |
| 8408731 | override, `SKIP=django-test`, last plan commit before this SUMMARY | `.planning/phases/40-notebook-isolation-attribution-page/40-VERIFICATION.md` |

## Recorded texts (verbatim)

### Docstring: finding list in "Why this module exists"

```
Why this module exists (UAT G-37.1-3, then review findings WR-13, WR-14 and WR-17 of the 37.1 review,
WR-01 of the Phase 40 review, WR-01 of the 2026-10-10 re-review, CR-01 and CR-02 of the third review, and CR-01, WR-01
and WR-02 of the fourth review):
```

### Docstring: "What the guard checks", automagic sentence

A one-line cell whose first word, read with IPython's ``split_user_input``, is an IPython line magic or alias (the set
comes from IPython itself, including the kernel's and storemagic's magics) is reported, because IPython's automagic runs
it unless that name is bound when the cell runs, which a static check cannot tell: a bare annotation, an ``if False:``
branch, an empty loop or an ``except ... as`` name binds nothing, and ``del`` unbinds, so the cell is reported whatever
an earlier cell assigns, imports or defines. Only a first word that is a Python builtin or keyword, or one followed by
``=`` or ``,``, is not reported, because IPython always runs such a line as Python.

### Docstring: "What the guard checks", R5, R7 and R6 sentences

R5 and R7: no str or bytes constant in a cell's code (a bytes constant is read as latin-1) names the database variable
``FOMO_DATABASE_PATH`` or the developer database file ``fomo_db.sqlite3``, and no keyword argument is named
``FOMO_DATABASE_PATH``. R6, no byte copy or byte write, reads a cell's code and the code inside its strings. It reports
shutil's ``copy2``, ``copyfile``, ``copytree`` and ``copyfileobj`` at every identifier the parser records, as R12 matches
its names; ``copy`` and ``move``, which are also common method names (a DataFrame's ``copy``), when read as an attribute
of an expression that mentions a shutil name, called or not, or imported with ``from shutil import``; any
``write_bytes``; and any str or bytes constant one of whose '.'- or ':'-separated parts is one of these names, ``copy``
or ``move`` (``getattr(shutil, 'copytree')``). A shutil name is ``shutil`` or a name the notebook binds to it in any
cell: through ``import shutil as X`` or ``from <any module> import shutil [as X]``, or as the target of an assignment, a
loop, a ``with``, a comprehension or a parameter default whose value mentions a shutil name, where a mention is a name,
an attribute or a string part, so ``zipfile.shutil`` and ``sys.modules['shutil']`` count. ``copy`` and ``move`` reached
only through a function's return value, a call's side effect (``setattr``) or a module that holds shutil under another
name (tempfile's ``_shutil``) are not reported.

The "What the guard does not see" paragraph (RES) is unchanged and stays last. The recorded docstring's lines are
wrapped slightly differently from the draft's text above (line breaks only; the checker compares whitespace-normalized
text).

### README guard paragraph

`solsys_code/tests/test_pre_executed_notebooks.py` checks every notebook in `pre_executed/`. It checks that a notebook
calls the helper first; never sets Django up, chooses a database, starts a process, changes the environment or runs code
built from a string or loaded from another file, whether the name is written in its code, in an import or inside a
string; never names `FOMO_DATABASE_PATH` or the developer database file in a str or bytes literal; never byte-copies or
byte-writes a file with shutil's copy and move functions or `write_bytes` (`copy2`, `copyfile`, `copytree` and
`copyfileobj` wherever they are named, `copy` and `move` when they are read from shutil or from a name the notebook
binds to it); contains only Python that IPython runs unchanged, apart from `%matplotlib inline` and a first-line
`%%time` or `%%capture`; has no one-line cell that starts with an IPython magic or alias name, even one an earlier cell
assigns or imports, unless that name is a Python builtin or keyword or is followed by `=` or `,`; and shows output from
one fresh run. Its module docstring says exactly what each rule follows and what a static check of a notebook's source
cannot see. It reads each notebook's JSON and never executes one.

### Override reason (`overrides[0].reason`, approved text; folded and re-wrapped to 104 columns in the front matter)

Rule R12, extended in 40-08 [301bd93, f1b4bb6] and 40-10 [482606b, 3232341, c6c6322], reports in every notebook any name
it lists for five capabilities: Django's database settings and connection objects (DATABASES, DATABASE_ROUTERS,
settings_dict, override_settings, modify_settings, the helper's internals, DatabaseWrapper, load_backend, connections,
connection, _connections, ConnectionHandler, create_connection, get_connection_params, get_new_connection, Database,
sqlite3); attaching another database through raw SQL (a string holding ATTACH followed by DATABASE, a quote, a
placeholder or an opening parenthesis); starting a process (subprocess, multiprocessing, pty, pexpect, ptyprocess,
psutil, jupyter_client, nbclient, nbconvert, system, popen, fork, forkpty, startfile, posix_spawn, posix_spawnp, every
exec* and spawn* function of os, Popen, ProcessPoolExecutor, create_subprocess_exec, create_subprocess_shell, getoutput,
getstatusoutput, IPython's get_ipython, system_raw, run_line_magic and run_cell_magic, and the IPython.utils helpers
getoutputerror, get_output_error_code and process_handler); the environment (environ, environb, putenv, unsetenv); and
running code from a string or another file (exec, eval, compile, __import__, __builtins__, importlib, runpy, builtins,
ctypes, cffi, execute_from_command_line, ManagementUtility, import_string, resolve_name, locate, IPython's Macro, and
call_command of shell, shell_plus or runscript). Each name is matched at every identifier the parser records, as the
guard docstring enumerates them (attributes are never matched against the builtins exec, eval, compile, __import__ and
__builtins__), and against every '.'- or ':'-separated part of a string or bytes constant; a string that parses and
holds a statement or a call is read like the cell's own code, to two levels. So a listed name imported from another
module (from webbrowser import subprocess), bound to another name first or looked up by a literal string
(import_string('subprocess.run')) is reported. R0 reports any code cell that is not Python IPython runs unchanged,
checked with IPython's own TransformerManager, once %matplotlib inline lines and a first-line %%time or %%capture (with
only its --no-stderr, --no-stdout and --no-display flags and an output name) are removed, and, since 40-13 [905408a],
any one-line cell whose first word is an IPython line magic or alias, because IPython's automagic runs it unless the
name is bound when the cell runs, whatever an earlier cell assigns, imports or defines; only a word that is a builtin or
a keyword, or one followed by = or a comma, is exempt. R5 and R7 report a str or bytes constant that names
FOMO_DATABASE_PATH or fomo_db.sqlite3 (40-13 [4f02628]). R6 reports shutil's copy2, copyfile, copytree and copyfileobj
at every identifier the parser records and as a string part, copy and move read from an expression that mentions shutil
or a name the notebook binds to it in any cell through an import, an assignment, a loop, a with, a comprehension or a
parameter default (40-13 [4f02628]), and any write_bytes; copy and move reached only through a function's return value,
a call's side effect or a module that holds shutil under another name are not reported. Every literal form in the third
review (CR-01, CR-02, WR-01), the fourth review's CR-01 and WR-01 forms, the re-review's four forms and both round-1
probes is reported on a copy of a real notebook, the fourth review's WR-02 bytes literals on the reference notebook, and
every real pre-executed notebook passes the guard. The guard docstring makes no claim of completeness and names what
stays out of reach: the guard reads only the notebook's own source, so values assembled or looked up at runtime (a name,
module, SQL statement or file path built from pieces, or fetched with getattr, globals(), vars() or sys.modules using
such a string), code decoded at runtime (pickle, marshal, codecs, base64), code outside the notebook (a module it
imports, IPython's profile and start-up files with the extensions, aliases and magics they define, or a library
function that starts a process or writes a file under a name these lists do not contain) and, left out on purpose,
JavaScript or HTML in a cell's output, which a notebook front end may run when the notebook is opened, are not seen.

The residual the developer accepted: RES (values assembled or looked up at runtime, code decoded at runtime, code
outside the notebook, JavaScript or HTML in outputs) plus R6's stated boundary for copy and move (reached only through a
function's return value, a call's side effect or a module holding shutil under another name).

## Evidence table as presented

| Claim | Evidence |
|-------|----------|
| R12 reports in every notebook any name it lists for five capabilities | docstring: "R12 reports, in every notebook, any use of five capabilities that can route around the scratch database" |
| R12 applies to a notebook that does not import Django | test: TestRerouteRuleOnSyntheticNotebooks.test_the_rule_applies_to_a_notebook_that_does_not_import_django |
| Extended in 40-08 [301bd93, f1b4bb6]: the re-review forms and whole capabilities | test: TestReReviewFormsOnARealNotebook.test_a_connection_handler_store_is_reported_on_a_real_notebook, TestCapabilityRulesOnSyntheticNotebooks.test_r12_name_lists_are_disjoint_and_reach_r0 |
| Extended in 40-10 [482606b, 3232341, c6c6322]: identifiers, strings, R0 allow-list | test: TestR12MatchesEveryIdentifierAndString.test_a_listed_name_is_reported_wherever_a_name_appears, TestR0AllowList.test_a_cell_is_a_problem_unless_it_is_python_that_ipython_runs_unchanged |
| Database settings and connection objects, as listed (REROUTE_NAMES) | test: TestRerouteRuleOnSyntheticNotebooks.test_settings_dict_in_any_spelling_is_a_problem, TestRerouteRuleOnSyntheticNotebooks.test_databases_or_routers_in_any_spelling_is_a_problem, TestRerouteRuleOnSyntheticNotebooks.test_connection_handler_settings_are_a_problem, TestRerouteRuleOnSyntheticNotebooks.test_override_or_modify_settings_is_a_problem, TestRerouteRuleOnSyntheticNotebooks.test_helper_internals_are_a_problem, TestRerouteRuleOnSyntheticNotebooks.test_sqlite3_or_a_backend_wrapper_is_a_problem, TestCapabilityRulesOnSyntheticNotebooks.test_replacing_a_database_connection_in_any_spelling_is_a_problem |
| connection, _connections and Database are listed | test: TestR12MatchesEveryIdentifierAndString.test_the_connection_proxy_and_attach_with_a_parenthesis_are_a_problem, TestThirdReviewFormsOnARealNotebook.test_the_connection_proxy_database_connect_and_attach_are_reported_on_a_real_notebook |
| Raw SQL ATTACH followed by DATABASE, a quote, a placeholder or an opening parenthesis | docstring: "a string holding ATTACH followed by DATABASE, a quote or a placeholder, or by an opening parenthesis" |
| Raw SQL ATTACH is reported | test: TestCapabilityRulesOnSyntheticNotebooks.test_attaching_another_database_through_raw_sql_is_a_problem |
| Starting a process, as listed (PROCESS_MODULES, PROCESS_NAMES) | test: TestCapabilityRulesOnSyntheticNotebooks.test_starting_a_process_in_any_spelling_is_a_problem, TestCapabilityRulesOnSyntheticNotebooks.test_ipython_shell_access_in_a_parseable_cell_is_a_problem |
| IPython.utils helpers getoutputerror, get_output_error_code and process_handler | test: TestThirdReviewFormsOnARealNotebook.test_ipython_process_helpers_are_reported_on_a_real_notebook |
| The environment (environ, environb, putenv, unsetenv) | test: TestCapabilityRulesOnSyntheticNotebooks.test_changing_the_environment_in_any_spelling_is_a_problem |
| Running code from a string or another file, as listed | test: TestCapabilityRulesOnSyntheticNotebooks.test_running_code_from_a_string_or_another_file_is_a_problem |
| import_string, resolve_name, locate and IPython's Macro are listed | test: TestR12MatchesEveryIdentifierAndString.test_a_listed_name_inside_a_string_or_bytes_constant_is_a_problem |
| Each name is matched at every identifier the parser records, as the docstring enumerates them | docstring: "Each R12 name is matched at every identifier the parser records" |
| Every identifier position | test: TestR12MatchesEveryIdentifierAndString.test_a_listed_name_is_reported_wherever_a_name_appears |
| Attributes are never matched against the builtins | docstring: "never matched against the builtins exec, eval, compile, ``__import__`` and ``__builtins__`` (so ``re.compile`` stays allowed)" |
| Every '.'- or ':'-separated part of a string or bytes constant | docstring: "against every '.'- or ':'-separated part of a string or bytes constant" |
| A string that parses and holds a statement or a call is read like code, to two levels | docstring: "(it parses and holds a statement or a call) is read like the cell's own code, to two levels" |
| String constants and string code | test: TestR12MatchesEveryIdentifierAndString.test_a_listed_name_inside_a_string_or_bytes_constant_is_a_problem, TestR12MatchesEveryIdentifierAndString.test_code_inside_a_string_is_checked_like_code |
| Look-alikes stay allowed | test: TestR12MatchesEveryIdentifierAndString.test_names_that_only_look_like_listed_ones_are_not_a_problem, TestCapabilityRulesOnSyntheticNotebooks.test_allowed_look_alikes_are_not_a_problem |
| A name imported from another module, bound to another name or looked up by a literal string is reported | docstring: "So importing a listed name from another module (``from webbrowser import subprocess``), binding it to another name first" |
| from webbrowser import subprocess and import_string('subprocess.run') on a real notebook | test: TestThirdReviewFormsOnARealNotebook.test_subprocess_imported_from_another_module_is_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_a_process_looked_up_by_a_literal_string_is_reported_on_a_real_notebook |
| R0: a code cell must be Python IPython runs unchanged, checked with TransformerManager | docstring: "IPython's own static input transformer (``TransformerManager``) must leave its meaning unchanged" |
| R0's harmless magic lines and %%capture flags | docstring: "``%%time`` or ``%%capture`` (with only its ``--no-stderr``, ``--no-stdout`` and ``--no-display`` flags and an output name) are removed" |
| R0 allow-list, harmless magics, code under them checked | test: TestR0AllowList.test_a_cell_is_a_problem_unless_it_is_python_that_ipython_runs_unchanged, TestR0AllowList.test_code_under_a_harmless_magic_is_checked_by_every_rule, TestThirdReviewFormsOnARealNotebook.test_harmless_magic_lines_leave_a_real_notebook_clean |
| Since 40-13 [905408a], a one-line cell whose first word is an IPython line magic or alias is reported because automagic runs it unless the name is bound when the cell runs | docstring: "IPython's automagic runs it unless that name is bound when the cell runs, which a static check cannot tell" |
| Whatever an earlier cell assigns, imports or defines: seven bindings of rm before an alias cell on a copy of a real notebook, each binding alone clean | test: TestR0AllowList.test_a_one_line_magic_or_alias_is_a_problem_even_when_an_earlier_cell_binds_the_name |
| Only a builtin, a keyword or a following = or a comma is exempt | docstring: "Only a first word that is a Python builtin or keyword, or one followed by ``=`` or ``,``, is not reported" |
| The exempt forms stay clean and the automagic set matches a fresh IPython shell | test: TestR0AllowList.test_a_one_line_magic_or_alias_is_a_problem_even_when_an_earlier_cell_binds_the_name, TestR0AllowList.test_the_automagic_rule_knows_every_line_magic_a_fresh_ipython_shell_has |
| R5 and R7 report a str or bytes constant that names FOMO_DATABASE_PATH or fomo_db.sqlite3 (40-13 [4f02628]) | test: TestIsolationRulesOnSyntheticNotebooks.test_a_bytes_literal_naming_the_developer_database_or_its_variable_is_a_problem, TestIsolationRulesOnSyntheticNotebooks.test_developer_database_literal_in_code_is_a_problem |
| R5 and R7 read bytes as latin-1 and a keyword argument named FOMO_DATABASE_PATH is reported | docstring: "no str or bytes constant in a cell's code (a bytes constant is read as latin-1) names the database variable" |
| R6 reports shutil's copy2, copyfile, copytree and copyfileobj at every identifier the parser records and as a string part (40-13 [4f02628]) | test: TestR6CopyAndWriteFunctions.test_a_rebound_shutil_or_a_distinctive_copy_name_is_a_problem, TestR6CopyAndWriteFunctions.test_every_copy_and_write_function_in_any_spelling_is_a_problem |
| R6 reports the distinctive names at every identifier, as R12 matches its names | docstring: "at every identifier the parser records, as R12 matches its names" |
| copy and move are reported when read from an expression that mentions shutil or a name the notebook binds to it in any cell | test: TestR6CopyAndWriteFunctions.test_a_rebound_shutil_or_a_distinctive_copy_name_is_a_problem, TestR6CopyAndWriteFunctions.test_a_shutil_name_bound_in_another_cell_or_in_string_code_is_followed |
| The binding forms R6 follows: import, assignment, loop, with, comprehension, parameter default; zipfile.shutil and sys.modules count | docstring: "as the target of an assignment, a loop, a ``with``, a comprehension or a parameter default whose value mentions a shutil name" |
| A mention is a name, an attribute or a string part | docstring: "where a mention is a name, an attribute or a string part, so ``zipfile.shutil`` and ``sys.modules['shutil']`` count" |
| Any write_bytes is reported | test: TestR6CopyAndWriteFunctions.test_every_copy_and_write_function_in_any_spelling_is_a_problem, TestThirdReviewFormsOnARealNotebook.test_copytree_and_write_bytes_are_reported_on_a_real_notebook |
| Look-alikes (a DataFrame copy, a name not bound to shutil) are not reported | test: TestR6CopyAndWriteFunctions.test_copy_and_move_not_read_from_shutil_are_not_a_problem |
| copy and move reached only through a function's return value, a call's side effect or a module holding shutil under another name are not reported | docstring: "reached only through a function's return value, a call's side effect (``setattr``) or a module that holds shutil under another name (tempfile's ``_shutil``) are not reported" |
| Third review CR-01 forms on a real notebook | test: TestThirdReviewFormsOnARealNotebook.test_subprocess_imported_from_another_module_is_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_a_process_looked_up_by_a_literal_string_is_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_ipython_process_helpers_are_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_the_connection_proxy_database_connect_and_attach_are_reported_on_a_real_notebook |
| Third review CR-02 and WR-01 forms on a real notebook | test: TestThirdReviewFormsOnARealNotebook.test_a_shell_escape_after_a_line_magic_is_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_ipython_aliases_and_shell_magics_are_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_a_one_line_alias_or_magic_that_is_valid_python_is_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_copytree_and_write_bytes_are_reported_on_a_real_notebook |
| The fourth review's CR-01 forms (seven bindings of rm) and WR-01 forms (both rebinding forms) on a copy of a real notebook | test: TestR0AllowList.test_a_one_line_magic_or_alias_is_a_problem_even_when_an_earlier_cell_binds_the_name, TestR6CopyAndWriteFunctions.test_a_rebound_shutil_or_a_distinctive_copy_name_is_a_problem |
| The re-review's four forms on a real notebook | test: TestReReviewFormsOnARealNotebook.test_a_connection_handler_store_is_reported_on_a_real_notebook, TestReReviewFormsOnARealNotebook.test_a_child_manage_py_with_a_bare_environment_is_reported_on_a_real_notebook, TestReReviewFormsOnARealNotebook.test_a_cleared_environment_and_a_child_manage_py_are_reported_on_a_real_notebook, TestReReviewFormsOnARealNotebook.test_a_shell_escape_running_manage_py_is_reported_on_a_real_notebook |
| Both round-1 probes on a real notebook | test: TestRerouteRuleOnARealNotebook.test_a_settings_dict_reroute_after_setup_is_reported_on_a_real_notebook, TestRerouteRuleOnARealNotebook.test_an_override_settings_reroute_after_setup_is_reported_on_a_real_notebook |
| The fourth review's WR-02 bytes literals on the reference notebook | test: TestIsolationRulesOnSyntheticNotebooks.test_a_bytes_literal_naming_the_developer_database_or_its_variable_is_a_problem |
| Every real pre-executed notebook passes the guard | test: TestPreExecutedNotebooksUseTheHelper.test_every_real_pre_executed_notebook_is_isolated, TestPreExecutedNotebooksUseTheHelper.test_the_real_notebook_check_is_not_vacuous, TestRerouteRuleOnARealNotebook.test_the_unmodified_real_notebook_has_no_problems |
| The guard docstring makes no claim of completeness | docstring: "The guard makes no claim of completeness." |
| The guard reads only the notebook's own source | docstring: "it reads only the notebook's own source, so deliberate obfuscation is out of its reach" |
| Residual (1): values assembled or looked up at runtime | docstring: "(1) Values assembled or looked up at runtime: a name, module, SQL statement or file path built from pieces" |
| Residual (1): getattr, globals(), vars(), sys.modules; code decoded at runtime | docstring: "fetched with ``getattr``, ``globals()``, ``vars()`` or ``sys.modules`` using such a string, and code decoded at runtime (pickle, marshal, codecs, base64)" |
| Residual (2): code outside the notebook | docstring: "(2) Code outside the notebook: a module it imports, IPython's profile and start-up files (with the extensions, aliases and magics they define), or a library function that starts a process or writes a file under a name these lists do not contain" |
| Residual (3): JavaScript or HTML in a cell's output, left out on purpose | docstring: "(3) Left out on purpose: JavaScript or HTML in a cell's output, which a notebook front end may run when the notebook is opened" |

## Ledger rows as recorded (45199cc)

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | fixed | fixed in 40-13 Task 1 (905408a) and 40-14 Task 3 (1a315e2): R0's automagic rule reports a one-line cell that starts with an IPython magic or alias name whatever an earlier cell binds (only a builtin, a keyword or a following = or comma is exempt); the four non-binding forms, a plain assignment, an import and an assignment then del are reported on a copy of import_campaign_csv_demo; the guard docstring, README and override reason say so |
| WR-01 | warning | fixed | fixed in 40-13 Task 2 (4f02628) and 40-14 Task 3 (1a315e2): R6 reports copy2, copyfile, copytree and copyfileobj at every identifier and string part and follows shutil names across the notebook through imports and assignment-shaped bindings for copy and move; both review rebinding forms are reported on a copy of import_campaign_csv_demo; the docstring names what is not followed |
| WR-02 | warning | fixed | fixed in 40-13 Task 2 (4f02628) and 40-14 Task 3 (1a315e2): R5 and R7 read str and bytes constants, so bytes literals naming the developer database file and its variable are reported; the docstring's R5 and R7 sentence says so |
| IN-01 | info | open | out of scope for the gap round (developer decision 2026-10-10); recorded for Phase 41 in 40-11-SUMMARY |
| IN-02 | info | fixed | fixed in 40-13 Task 3 (ccfc578): the dev extra lists ipykernel>=7, which the guard imports; every CI workflow installs .[dev], so none changed |

Front matter: `open: 8` became `open: 4`; `total: 12` unchanged.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] README paragraph re-wrapped so the plan's own check could find its closing phrase**
- **Found during:** Task 3 step 2
- **Issue:** The approved README paragraph's last line break fell inside "and never executes one.", and Task 3's first check locates that closing phrase literally to measure line length.
- **Fix:** moved one line break so the phrase sits on one line (whitespace only; the approved text is unchanged when whitespace-normalized, which is how the check compares it).
- **Files modified:** `docs/notebooks/README.md`
- **Commit:** 1a315e2

### Plan check flaw (no repository change)

The plan's third Task 3 check requires `git diff --name-only e08f8eb -- ... pyproject.toml ...` to be empty, but 40-13 Task 3 (ccfc578) legitimately added `ipykernel>=7` to pyproject.toml after e08f8eb, so the chained command stops before printing its OK line. Every other link in that chain passed (ruff and ruff-format clean, no uncommitted change, the docs commit holds exactly the guard module and README, it follows 40-13's guard commit). The intent was checked directly: `git diff e08f8eb 1a315e2` over `docs/notebooks/pre_executed`, `docs/notebooks/fomo_nb_utils.py`, `docs/runbooks` and `40-REVIEW.md` is empty, and `git log e08f8eb..HEAD -- pyproject.toml` lists only 40-13's ccfc578. 40-14 touched none of them.

**Total deviations:** 1 auto-fixed (whitespace), 1 plan check flaw noted. **Impact:** none on content.

## Paired docs

`docs/notebooks/README.md` is the paired document for the guard and was changed here. No notebook was re-executed, because neither `docs/notebooks/fomo_nb_utils.py` nor any notebook changed this round. No `docs/runbooks/` page documents the guard.

## Verification

- Guard, helper and projector demo tests: `Ran 121 tests ... OK` with the approved docstring and README in place.
- Docs commit full pre-commit hook (ruff, ruff-format, pre-executed notebook check, Django tests): passed.
- Check 1 `OK: the module docstring and README guard paragraph are the approved texts; guard code identical to 40-13; README otherwise unchanged`.
- Check 3 `OK: CR-01, WR-01, WR-02 and IN-02 fixed with real commits touching their files (40-14 docs commit cited for the three guard findings), IN-01 open with its out-of-scope note, open 8 -> 4, exactly 10 lines changed; the ledger commit holds only the ledger and follows the docs commit`.
- Check 4 `OK: approve -- the approved reason and accepted_at 2026-10-10T23:20:47Z recorded in the one override entry; must_have, accepted_by, every other key and the body unchanged; the override commit is HEAD, after the ledger commit`.

## Notes for the verifier

- Recording a reason does not apply the override. The verifier applies or rejects it at re-verification (`overrides_applied` is still 0 as the verifier left it).
- Task 3's last check can pass only while the override commit (8408731) is HEAD, so the SUMMARY and tracking commits that follow do not invalidate it. It was run before them.
- Verification human items 2-4 are /gsd-verify-work reading items (D-G10). The developer read the docstring and README wording at this plan's checkpoint, which covers the wording half of item 2.
- Nothing was pushed.

## Known Stubs

None.

## Threat Flags

None. This plan changed a docstring, a README paragraph, a ledger and one front-matter entry only; T-40-51 and T-40-52 were mitigated as planned (draft check before approval, blocking-human decision, front-matter comparison against e08f8eb).

## Self-Check: PASSED

- `solsys_code/tests/test_pre_executed_notebooks.py`, `docs/notebooks/README.md`, `40-REVIEW-DISPOSITION.md`, `40-VERIFICATION.md`: FOUND, committed.
- Commits 1a315e2, 45199cc, 8408731: FOUND on `issue37-telescope-runs-calendar`, in that order.
