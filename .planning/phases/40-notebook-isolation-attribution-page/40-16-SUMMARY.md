---
phase: 40-notebook-isolation-attribution-page
plan: 16
subsystem: testing
tags: [notebook-isolation-guard, documentation, gap-closure, developer-approval, override]
gap_closure: true
gap_ids: [CR-01, CR-02, WR-01, WR-02]

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-15 guard code (R6 shutil/write_bytes everywhere, IPyAutocall in R12, pin tests) at 1c830b1"
provides:
  - "Guard module docstring and README guard paragraph that say only what 40-15's guard checks, including R6's post-fix boundary and IPython's autocall (developer-approved)"
  - "Review ledger: CR-01, CR-02, WR-01, WR-02 fixed; IN-01, IN-02 out of scope (D-G14)"
  - "Reworded WARN-02 override reason in 40-VERIFICATION.md, accepted_at 2026-10-11T03:35:54Z (applied or rejected by the verifier at re-verification)"
affects: [phase-40-verification, phase-41]

requirements-completed: []

actuals:
  tokens: 21000
  tasks: 3
  commits: 3

plan_head_before: 2d83fe727ecf1cdbb0f6e115a0878e9c69256187
plan_head_after: 88fc09edb0ac6fed59c7cd5da18060598a8e0150

key-files:
  created: []
  modified:
    - solsys_code/tests/test_pre_executed_notebooks.py
    - docs/notebooks/README.md
    - .planning/phases/40-notebook-isolation-attribution-page/40-REVIEW-DISPOSITION.md
    - .planning/phases/40-notebook-isolation-attribution-page/40-VERIFICATION.md

key-decisions:
  - "D-G11/D-G12/D-G13/D-G14: docstring, README and override reason recorded exactly as drafted, approved by Tim Lister (decision approve, 2026-10-11T03:35:54Z)"
  - "WARN-02 and WARN-05 are not marked complete here: recording a reason does not apply the override, and requirements.ready-ids reports nothing ready; the verifier decides at re-verification"

status: complete
completed: 2026-10-11
---

# Phase 40 Plan 16: Guard docs, ledger and WARN-02 override reason Summary

**The guard docstring, README paragraph and WARN-02 override reason now state what 40-15's guard checks (R6 post-fix boundary, IPython autocall, Django-backed gate) in the developer-approved words, with the review ledger updated and no code changed.**

## Performance

- Tasks: 3 (draft in scratch, developer decision, record). Task 1 and Task 2 made no repository change by design.
- Three commits, in the plan's order: docs (full hook), ledger (SKIP=django-test), override (SKIP=django-test, last). Nothing pushed.

## Decision

- Decision: `approve` ("Approve all three as drafted"), Tim Lister, through the orchestrator's question.
- Approval time (supplied by the orchestrator, UTC): `2026-10-11T03:35:54Z`.
- Task 1 draft check printed its OK line before the checkpoint: 8-sentence reason, 77 evidence rows. (It cannot be re-run now: it compares the draft with the pre-approval docstring, which Task 3 replaced.)

## Commits

| Commit | What | Hook |
| ------ | ---- | ---- |
| b04f09f | docs: guard module docstring and README guard paragraph (the only two files; guard code identical to 40-15's) | full hook, passed |
| 930d2df | docs: review ledger, 11 lines | SKIP=django-test |
| 88fc09e | docs: override reason and accepted_at in 40-VERIFICATION.md (the only file; HEAD at the time of the plan's last check) | SKIP=django-test |

No notebook was re-executed because neither the helper nor any notebook changed.

## Ledger rows as recorded

| Finding | Severity | Disposition | Source |
|---|---|---|---|
| CR-01 | critical | fixed | fixed in 40-15 Task 1 (bc99b0e) and 40-16 Task 3 (b04f09f): R6 reports shutil and tempfile's _shutil at every identifier and string part, so all six review forms (a call argument, a lambda, map, a keyword-only parameter, a match capture, a class pattern) are reported on a copy of import_campaign_csv_demo; the guard docstring, README and override reason state R6's post-fix boundary |
| CR-02 | critical | fixed | fixed in 40-15 Task 1 (bc99b0e) and 40-16 Task 3 (b04f09f): R6 matches write_bytes at every identifier and string part; the class-pattern form and a write_bytes parameter and keyword argument are reported on a copy of import_campaign_csv_demo; the docstring, README and reason no longer claim more |
| WR-01 | warning | fixed | fixed in 40-15 Task 2 (1c830b1: isolation_problems docstring and a pin test) and 40-16 Task 3 (b04f09f): the module docstring says R2-R4 and R8-R11 apply only when a cell that parses as written imports Django (D-13 unchanged, developer decision D-G12) |
| WR-02 | warning | fixed | fixed in 40-15 Task 2 (1c830b1) and 40-16 Task 3 (b04f09f): IPyAutocall is an R12 name, and the docstring, README and override reason name IPython's autocall of exit and quit as the rewrite R0 does not see, pinned by a test (D-G13) |
| IN-01 | info | open | out of scope for the gap round (developer decision 2026-10-10, D-G14); kept for Phase 41 triage |
| IN-02 | info | open | out of scope for the gap round (developer decision 2026-10-10, D-G14); kept for Phase 41 triage |

Front matter: `open: 9` became `open: 5`; `total: 12` unchanged; exactly 11 lines changed.

## Evidence table as presented to the developer

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
| Look-alikes (a DataFrame copy, a name not bound to shutil) are not reported | test: TestR6CopyAndWriteFunctions.test_copy_and_move_not_read_from_shutil_are_not_a_problem |
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
| Extended in 40-15 [1c830b1]: IPyAutocall is an R12 name, listed with Macro | test: TestR0AllowList.test_autocall_is_not_seen_by_r0_and_the_ipyautocall_name_is_reported |
| IPython's Macro and IPyAutocall are listed (R12 name list) | docstring: "IPython's ``Macro`` and ``IPyAutocall``, and ``call_command`` of shell, shell_plus or runscript" |
| R0 reports a code cell that is not Python IPython runs unchanged, apart from IPython's autocall | docstring: "every code cell must be Python that IPython runs unchanged, apart from IPython's autocall (below)" |
| R0 does not see IPython's autocall: a one-line cell that starts with an autocall object runs as a call (exit -1 runs as exit(-1)) | docstring: "R0 does not see one rewrite, IPython's autocall: a one-line cell that starts with an autocall object runs as a call (``exit -1`` runs as ``exit(-1)``)" |
| exit -1 and quit -1 give no R0 problem | test: TestR0AllowList.test_autocall_is_not_seen_by_r0_and_the_ipyautocall_name_is_reported |
| The kernel's only autocall objects, exit and quit, end the kernel | docstring: "The kernel's only autocall objects, ``exit`` and ``quit``, end the kernel" |
| R12 reports the name IPyAutocall (40-15 [1c830b1]) | test: TestR0AllowList.test_autocall_is_not_seen_by_r0_and_the_ipyautocall_name_is_reported |
| R12 reports IPyAutocall: the docstring says so | docstring: "R12 reports the name ``IPyAutocall``" |
| An autocall object made without that name (a subclass of type(exit)) runs a call method that is notebook code the rules read | test: TestR0AllowList.test_autocall_is_not_seen_by_r0_and_the_ipyautocall_name_is_reported |
| A subclass of type(exit): its call method is notebook code the rules read (docstring) | docstring: "runs its own call method, which is notebook code the rules read like any other" |
| R6 reports the module names shutil and _shutil and write_bytes at every identifier and as a string part (40-15 [bc99b0e]) | test: TestR6CopyAndWriteFunctions.test_shutil_passed_on_in_any_way_is_reported_on_a_real_notebook, TestR6CopyAndWriteFunctions.test_write_bytes_at_any_identifier_is_reported_on_a_real_notebook, TestR6CopyAndWriteFunctions.test_shutil_and_tempfile_shutil_are_reported_wherever_they_are_named |
| R6 reports the module names shutil and _shutil (docstring) | docstring: "It reports the module names ``shutil`` and ``_shutil`` (tempfile's name for shutil)" |
| write_bytes is reported at every identifier the parser records and as a string part (docstring) | docstring: "``write_bytes`` at every identifier the parser records" |
| Code that names shutil is reported however it then hands the module on (a parameter, a lambda, map over a lambda, a keyword-only parameter, a match capture, a class pattern) | test: TestR6CopyAndWriteFunctions.test_shutil_passed_on_in_any_way_is_reported_on_a_real_notebook |
| Code that names shutil is reported however it hands the module on (docstring) | docstring: "so code that names shutil is reported however it then hands the module on" |
| tempfile._shutil is reported; _shutil is reported wherever it is named | test: TestR6CopyAndWriteFunctions.test_shutil_and_tempfile_shutil_are_reported_wherever_they_are_named |
| write_bytes is reported as a class-pattern attribute, a parameter and a keyword argument, once per cell | test: TestR6CopyAndWriteFunctions.test_write_bytes_at_any_identifier_is_reported_on_a_real_notebook |
| copy and move read from an expression that mentions shutil, _shutil or a name bound to one (docstring) | docstring: "when read as an attribute of an expression that mentions a shutil name, called or not" |
| Not reported: copy or move read from anything that mentions no shutil name (another object's method of that name) | test: TestR6CopyAndWriteFunctions.test_copy_and_move_not_read_from_shutil_are_not_a_problem |
| Not reported: copy or move read from anything that mentions no shutil name (docstring) | docstring: "Not reported: ``copy`` or ``move`` read from anything that mentions no shutil name" |
| Not reported: a path's copy and move on Python 3.14 and later | docstring: "a path's ``copy`` and ``move`` on Python 3.14 and later" |
| Not reported: a shutil module, shutil function or write_bytes method obtained at runtime without writing its name (inspect.getmodule, a function's __module__, a search over sys.modules or a module's vars()) | docstring: "that the notebook obtains at runtime without writing its name (from ``inspect.getmodule``, a function's ``__module__``, or a search over ``sys.modules`` or a module's ``vars()``)" |
| Not reported: code nested more than two strings deep, matched only by its parts | docstring: "code in a string nested more than two strings deep, which R6 and R12 match only by its '.'- and ':'-separated parts" |
| Code in a string is read to two levels (R6 and R12) | test: TestR12MatchesEveryIdentifierAndString.test_code_inside_a_string_is_checked_like_code |
| The fifth review's CR-01 and CR-02 forms are reported on a copy of a real notebook | test: TestR6CopyAndWriteFunctions.test_shutil_passed_on_in_any_way_is_reported_on_a_real_notebook, TestR6CopyAndWriteFunctions.test_write_bytes_at_any_identifier_is_reported_on_a_real_notebook |
| Whether R2-R4 and R8-R11 apply is decided by is_django_backed from the cells that parse as written, before any magic line is removed (D-13); every real notebook is Django-backed | test: TestR0AllowList.test_the_django_backed_gate_reads_each_cell_as_written, TestPreExecutedNotebooksUseTheHelper.test_the_real_notebook_check_is_not_vacuous |
| The gate sentence (docstring) | docstring: "Whether R2-R4 and R8-R11 apply is decided by ``is_django_backed`` from the cells that parse as written, before any magic line is removed (D-13)" |

## Recorded docstring (verbatim)

```text
Guard tests: every Django-backed pre-executed notebook runs on a scratch database through the shared helper.

Why this module exists (UAT G-37.1-3, then review findings WR-13, WR-14 and WR-17 of the 37.1 review,
WR-01 of the Phase 40 review, WR-01 of the 2026-10-10 re-review, CR-01 and CR-02 of the third review, CR-01, WR-01 and
WR-02 of the fourth review, and CR-01, CR-02, WR-01 and WR-02 of the fifth review):
notebooks under ``docs/notebooks/pre_executed/`` once opened the developer database
``src/fomo_db.sqlite3`` and wrote demo rows into it, and a guard that only matched the setup
cell's text was easy to get around (an own ``FOMO_DATABASE_PATH`` assignment with stale good-looking
output, ``from django import setup as _s``, a setup call hidden in a ``def``). Regenerating such a
notebook is routine, so each regeneration would have written to the developer database again.

The rule, enforced by ``isolation_problems()``: every Django-backed notebook calls the shared helper
(``docs/notebooks/fomo_nb_utils.py``) first -- one top-level
``scratch_db_dir = setup_scratch_django(repo_root_path, mode='fresh' | 'snapshot')`` before any Django
import -- never sets Django up or routes the database itself, never byte-copies a database, ends with
``teardown_scratch_django(scratch_db_dir)``, and shows output from one fresh run (execution counts
1..N, the ``Resolved database:`` line inside a ``fomo-notebook-db-*`` directory). The helper's own
behaviour is proven by running it in ``test_notebook_helper.py``.

These tests read notebook JSON only and never execute a notebook. Code is inspected with ``ast`` so a
comment that merely mentions a name cannot satisfy or trip a check.

What the guard checks: R0, every code cell must be Python that IPython runs unchanged, apart from IPython's autocall
(below). Once ``%matplotlib inline`` lines and a first-line ``%%time`` or ``%%capture`` (with only its
``--no-stderr``, ``--no-stdout`` and ``--no-display`` flags and an output name) are removed, the cell must parse, and
IPython's own static input transformer (``TransformerManager``) must leave its meaning unchanged, so an alias, an
escape on any line (also after a line magic, as in ``%time !ls``), any other magic and a prompt prefix (``In [1]:``)
are reported. ``%config`` is not harmless, because ``%config AliasManager.user_aliases = [...]`` defines a shell alias
that a later one-line cell runs; ``%matplotlib`` with any other argument is not harmless either, because a backend
such as ``module://name`` imports a module by name. A one-line cell whose first word, read with IPython's
``split_user_input``, is an IPython line magic or alias (the set comes from IPython itself, including the kernel's and
storemagic's magics) is reported, because IPython's automagic runs it unless that name is bound when the cell runs,
which a static check cannot tell: a bare annotation, an ``if False:`` branch, an empty loop or an ``except ... as``
name binds nothing, and ``del`` unbinds, so the cell is reported whatever an earlier cell assigns, imports or defines.
Only a first word that is a Python builtin or keyword, or one followed by ``=`` or ``,``, is not reported, because
IPython always runs such a line as Python. R0 does not see one rewrite, IPython's autocall: a one-line cell that
starts with an autocall object runs as a call (``exit -1`` runs as ``exit(-1)``). The kernel's only autocall objects,
``exit`` and ``quit``, end the kernel; R12 reports the name ``IPyAutocall``; and an autocall object a notebook makes
without that name (from a subclass of ``type(exit)``) runs its own call method, which is notebook code the rules read
like any other. A cell that does not parse as written and mentions Django, a database, a process, a shell command or
the environment (an IPython shell escape, a process cell magic such as ``%%bash``, a process or environment line magic
such as ``%system`` or ``%env``, ``manage.py``, ``env -i``, ``env -u`` or any name R12 lists) is still reported by its
own message. Every later rule reads each cell's Python view, so the code under a harmless magic is read like any other
code. Whether R2-R4 and R8-R11 apply is decided by ``is_django_backed`` from the cells that parse as written, before
any magic line is removed (D-13): a notebook whose only import of Django, the project app or a TOM plugin sits in a
cell under a harmless magic gets only the rules every notebook gets, and every real notebook makes such an import in a
cell that parses as written, as the not-vacuous test requires. R2-R4 and R8-R11: the helper call comes first and once,
no own Django setup, one fresh run with evidence and teardown output, and no ``assert`` guarding a write. R5 and R7:
no str or bytes constant in a cell's code (a bytes constant is read as latin-1) names the database variable
``FOMO_DATABASE_PATH`` or the developer database file ``fomo_db.sqlite3``, and no keyword argument is named
``FOMO_DATABASE_PATH``. R6, no byte copy or byte write, reads a cell's code and the code inside its strings, to two
levels as R12 does. It reports the module names ``shutil`` and ``_shutil`` (tempfile's name for shutil), shutil's
``copy2``, ``copyfile``, ``copytree`` and ``copyfileobj``, and ``write_bytes`` at every identifier the parser records,
as R12 matches its names, and as a '.'- or ':'-separated part of a str or bytes constant, so code that names shutil is
reported however it then hands the module on (to a parameter, a lambda, a ``match`` capture or a class pattern). It
also reports ``copy`` and ``move``, which are also common method names (a DataFrame's ``copy``), when read as an
attribute of an expression that mentions a shutil name, called or not, when imported with ``from shutil import``, and
as a string part (``getattr(shutil, 'move')``). A shutil name is ``shutil``, ``_shutil`` or a name the notebook binds
to one in any cell through ``import shutil as X`` or ``from <any module> import shutil [as X]``, or as the target of
an assignment, a loop, a ``with``, a comprehension or a parameter default whose value mentions a shutil name, where a
mention is a name, an attribute or a string part, so ``zipfile.shutil`` and ``sys.modules['shutil']`` count. Not
reported: ``copy`` or ``move`` read from anything that mentions no shutil name, such as another object's method of
that name (a DataFrame's ``copy``, or a path's ``copy`` and ``move`` on Python 3.14 and later, which no Python this
project supports has); a shutil module, a shutil function or a ``write_bytes`` method that the notebook obtains at
runtime without writing its name (from ``inspect.getmodule``, a function's ``__module__``, or a search over
``sys.modules`` or a module's ``vars()``); and code in a string nested more than two strings deep, which R6 and R12
match only by its '.'- and ':'-separated parts. R12 reports, in every notebook, any use of five capabilities that can
route around the scratch database (WR-01, D-G1): Django's database settings and connection objects (DATABASES,
DATABASE_ROUTERS, settings_dict, override_settings, modify_settings, the helper's internals, DatabaseWrapper,
load_backend, ``connections``, ``connection`` and ``_connections``, ConnectionHandler, create_connection,
get_connection_params, get_new_connection, ``Database``, ``connections.settings`` and ``connections.databases``, and
any ``sqlite3`` import); attaching another database through raw SQL (a string holding ATTACH followed by DATABASE, a
quote or a placeholder, or by an opening parenthesis, ``ATTACH (``); starting a process (subprocess, multiprocessing,
pty, pexpect, ptyprocess, psutil, Jupyter's kernel clients jupyter_client, nbclient and nbconvert, ``system``,
``popen``, ``fork``, ``forkpty``, ``startfile``, ``posix_spawn``, ``posix_spawnp``, every ``exec*`` and ``spawn*``
function of ``os``, Popen, ProcessPoolExecutor, asyncio's ``create_subprocess_exec`` and ``create_subprocess_shell``,
``getoutput``, ``getstatusoutput``, IPython's ``get_ipython``, ``system_raw``, ``run_line_magic`` and
``run_cell_magic``, and the IPython.utils helpers ``getoutputerror``, ``get_output_error_code`` and
``process_handler``); the environment (``os.environ``, ``os.environb``, ``os.putenv``, ``os.unsetenv``; ``os.getenv``
reads are allowed); and running code from a string or another file (exec, eval, compile, ``__import__``,
``__builtins__``, importlib, runpy, builtins, ctypes, cffi, execute_from_command_line, ManagementUtility, the lookup
functions ``import_string``, ``resolve_name`` and ``locate``, IPython's ``Macro`` and ``IPyAutocall``, and
``call_command`` of shell, shell_plus or runscript). Each R12 name is matched at every identifier the parser records
-- a name, an attribute, a keyword argument, a parameter, a function, class or exception name, a global or nonlocal
name, a match capture or class-pattern attribute, a type parameter, every part of an imported module's path, and every
imported name and alias, with attributes and class-pattern attributes never matched against the builtins exec, eval,
compile, ``__import__`` and ``__builtins__`` (so ``re.compile`` stays allowed) -- and against every '.'- or
':'-separated part of a string or bytes constant; a string that is itself Python code (it parses and holds a statement
or a call) is read like the cell's own code, to two levels. So importing a listed name from another module (``from
webbrowser import subprocess``), binding it to another name first (``c = connections``, ``run = os.system``) or
looking it up by a literal string (``import_string('subprocess.run')``) is reported too. The guard makes no claim of
completeness.

What the guard does not see: it reads only the notebook's own source, so deliberate obfuscation is out of its
reach. (1) Values assembled or looked up at runtime: a name, module, SQL statement or file path built from pieces
(string concatenation, f-strings, joined path parts, ``str.join``, ``chr``) or fetched with ``getattr``,
``globals()``, ``vars()`` or ``sys.modules`` using such a string, and code decoded at runtime (pickle, marshal,
codecs, base64). (2) Code outside the notebook: a module it imports, IPython's profile and start-up files (with the
extensions, aliases and magics they define), or a library function that starts a process or writes a file under a
name these lists do not contain (for example a path's ``replace`` or ``rename``, ``open`` in a write mode, or a
``webbrowser`` controller given a command line). (3) Left out on purpose: JavaScript or HTML in a cell's output,
which a notebook front end may run when the notebook is opened; an nbconvert regeneration does not run it.
```

## Recorded README paragraph (verbatim)

```text
`solsys_code/tests/test_pre_executed_notebooks.py` checks every notebook in `pre_executed/`. It checks that a notebook
calls the helper first; never sets Django up, chooses a database, starts a process, changes the environment or runs
code built from a string or loaded from another file, whether the name is written in its code, in an import, as a part
of a dotted string such as `'subprocess.run'`, or in code inside a string (read to two levels); never names
`FOMO_DATABASE_PATH` or the developer database file in a str or bytes literal; never names `shutil`, tempfile's
`_shutil`, shutil's `copy2`, `copyfile`, `copytree` or `copyfileobj`, or `write_bytes` in any of those places;
contains only Python that IPython runs unchanged, apart from `%matplotlib inline`, a first-line `%%time` or
`%%capture`, and IPython's autocall, which runs `exit -1` as `exit(-1)` (the kernel's own autocall objects, `exit` and
`quit`, only end the kernel); has no one-line cell that starts with an IPython magic or alias name, even one an
earlier cell assigns or imports, unless that name is a Python builtin or keyword or is followed by `=` or `,`; and
shows output from one fresh run. Its module docstring describes each rule and what a static check of a notebook's
source cannot see, and makes no claim of completeness. It reads each notebook's JSON and never executes one.
```

## Recorded override reason (verbatim, rewrapped for the front matter)

```text
Rule R12, extended in 40-08 [301bd93, f1b4bb6], 40-10 [482606b, 3232341, c6c6322] and 40-15 [1c830b1], reports in
every notebook any name it lists for five capabilities: Django's database settings and connection objects (DATABASES,
DATABASE_ROUTERS, settings_dict, override_settings, modify_settings, the helper's internals, DatabaseWrapper,
load_backend, connections, connection, _connections, ConnectionHandler, create_connection, get_connection_params,
get_new_connection, Database, sqlite3); attaching another database through raw SQL (a string holding ATTACH followed
by DATABASE, a quote, a placeholder or an opening parenthesis); starting a process (subprocess, multiprocessing, pty,
pexpect, ptyprocess, psutil, jupyter_client, nbclient, nbconvert, system, popen, fork, forkpty, startfile,
posix_spawn, posix_spawnp, every exec* and spawn* function of os, Popen, ProcessPoolExecutor, create_subprocess_exec,
create_subprocess_shell, getoutput, getstatusoutput, IPython's get_ipython, system_raw, run_line_magic and
run_cell_magic, and the IPython.utils helpers getoutputerror, get_output_error_code and process_handler); the
environment (environ, environb, putenv, unsetenv); and running code from a string or another file (exec, eval,
compile, __import__, __builtins__, importlib, runpy, builtins, ctypes, cffi, execute_from_command_line,
ManagementUtility, import_string, resolve_name, locate, IPython's Macro and IPyAutocall, and call_command of shell,
shell_plus or runscript). Each name is matched at every identifier the parser records, as the guard docstring
enumerates them (attributes are never matched against the builtins exec, eval, compile, __import__ and __builtins__),
and against every '.'- or ':'-separated part of a string or bytes constant; a string that parses and holds a statement
or a call is read like the cell's own code, to two levels. So a listed name imported from another module (from
webbrowser import subprocess), bound to another name first or looked up by a literal string
(import_string('subprocess.run')) is reported. R0 reports any code cell that is not Python IPython runs unchanged,
checked with IPython's own TransformerManager, once %matplotlib inline lines and a first-line %%time or %%capture
(with only its --no-stderr, --no-stdout and --no-display flags and an output name) are removed, apart from IPython's
autocall, which runs a one-line cell that starts with an autocall object as a call (exit -1 runs as exit(-1)): the
kernel's only autocall objects, exit and quit, end the kernel, R12 reports the name IPyAutocall (40-15 [1c830b1]), and
an autocall object made without that name (a subclass of type(exit)) runs a call method that is notebook code the
rules read; and, since 40-13 [905408a], R0 reports any one-line cell whose first word is an IPython line magic or
alias, because IPython's automagic runs it unless the name is bound when the cell runs, whatever an earlier cell
assigns, imports or defines; only a word that is a builtin or a keyword, or one followed by = or a comma, is exempt.
R5 and R7 report a str or bytes constant that names FOMO_DATABASE_PATH or fomo_db.sqlite3 (40-13 [4f02628]). R6
reports the module names shutil and _shutil, shutil's copy2, copyfile, copytree and copyfileobj, and write_bytes at
every identifier the parser records and as a string part (40-13 [4f02628], 40-15 [bc99b0e]), so code that names shutil
is reported however it then hands the module on, and copy and move read from an expression that mentions shutil,
_shutil or a name the notebook binds to one in any cell through an import, an assignment, a loop, a with, a
comprehension or a parameter default; copy and move read from anything that mentions no shutil name (another object's
method of that name, such as a path's copy and move on Python 3.14 and later), a shutil module, shutil function or
write_bytes method obtained at runtime without writing its name (from inspect.getmodule, a function's __module__ or a
search over sys.modules or a module's vars()), and code in a string nested more than two strings deep, which R6 and
R12 match only by its parts, are not reported. Every literal form in the third review (CR-01, CR-02, WR-01), the
fourth review's CR-01 and WR-01 forms, the fifth review's CR-01 and CR-02 forms, the re-review's four forms and both
round-1 probes is reported on a copy of a real notebook, the fourth review's WR-02 bytes literals on the reference
notebook, and every real pre-executed notebook passes the guard. The guard docstring makes no claim of completeness
and names what stays out of reach: the guard reads only the notebook's own source, so values assembled or looked up at
runtime (a name, module, SQL statement or file path built from pieces, or fetched with getattr, globals(), vars() or
sys.modules using such a string), code decoded at runtime (pickle, marshal, codecs, base64), code outside the notebook
(a module it imports, IPython's profile and start-up files with the extensions, aliases and magics they define, or a
library function that starts a process or writes a file under a name these lists do not contain) and, left out on
purpose, JavaScript or HTML in a cell's output, which a notebook front end may run when the notebook is opened, are
not seen.
```

## Verification

- Guard, helper and projector tests: 126 tests OK with the approved docstring and README.
- Task 3 checks printed their OK lines: docstring and README equal the approved texts, guard code identical to 40-15, lines at most 120 columns, README otherwise unchanged; ruff and ruff-format clean; docs commit holds exactly the two files and follows 40-15; ledger commit holds only the ledger and follows the docs commit; override commit holds only 40-VERIFICATION.md, follows the ledger commit and was HEAD.
- Task 3's last check can pass only while the override (or ledger) commit is HEAD, so later SUMMARY and tracking commits do not invalidate it.

## Deviations from Plan

None - plan executed exactly as written. One slip while editing the ledger (the table-row loop was guarded by a wrong line bound, so the four table rows were not rewritten on the first pass) was corrected before the ledger check and commit; the committed ledger changed exactly the expected 11 lines.

## Notes for the verifier

- The verifier applies or rejects the override at re-verification; recording the reason does not apply it.
- WARN-02 and WARN-05 remain `Gaps Found` in REQUIREMENTS.md; `requirements.ready-ids` for phase 40 reported nothing ready.
- Verification human items 2-4 are `/gsd-verify-work` reading items (D-G15).
- IN-01 and IN-02 of the fifth review are out of scope for the gap round (D-G14) and stay open for Phase 41 triage.

## Known Stubs

None.

## Threat Flags

None. No network endpoint, auth path or schema changed; the plan edited a docstring, a README paragraph and planning records only.
