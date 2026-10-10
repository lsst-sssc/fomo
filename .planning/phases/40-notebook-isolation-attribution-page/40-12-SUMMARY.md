---
phase: 40-notebook-isolation-attribution-page
plan: 12
subsystem: docs
tags: [notebooks, guard, override, verification, gap-closure, residual-risk]
gap_closure: true
gap_ids: [CR-01, CR-02]
status: complete

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-10 guard extensions (R12 identifier, string and string-code matching; R0 allow-list and automagic rule; R6 copy and write functions) and 40-11's guard docstring naming exactly the residual (RES)"
provides:
  - "40-VERIFICATION.md's one overrides: entry carries the developer-approved corrected reason (names what the extended guard reports and only the residual the guard docstring names) and accepted_at 2026-10-10T19:14:57Z"
affects: [verify-work, 40-re-verification]

actuals:
  tokens: 1460      # chars/4 over the realized repository diff (40-VERIFICATION.md, 5,849 changed chars); excludes this SUMMARY and the scratch draft (~5,000 more)
  tasks: 3
  commits: 1
plan_head_before: d688504b6f18082d2fde45461201ef4cba49b039
plan_head_after: 44c33c6

tech-stack:
  added: []
  patterns:
    - "Verifier-owned front matter edited by a script that locates the one overrides: entry by its preceding must_have line, re-wraps only the folded reason block and the accepted_at line, then proves every other key and the body unchanged against the base commit"

key-files:
  created:
    - .planning/phases/40-notebook-isolation-attribution-page/40-12-SUMMARY.md
  modified:
    - .planning/phases/40-notebook-isolation-attribution-page/40-VERIFICATION.md

key-decisions:
  - "D-G5 (5): Tim Lister approved the corrected override reason verbatim at the blocking-human checkpoint (2026-10-10T19:14:57Z); it names only what 40-10's tests and 40-11's docstring support, and the residual it accepts is exactly the docstring's 'What the guard does not see' paragraph"

patterns-established:
  - "Override reasons are drafted with a claim-to-evidence table (every claim a real test method or an exact guard-docstring phrase) and recorded only after a blocking-human approval, with the approval time taken when the reply arrives"

requirements-completed: [WARN-02]

coverage:
  - id: D1
    description: "The one override entry in 40-VERIFICATION.md holds exactly the approved reason and the approval time; must_have, accepted_by, every other front-matter key and the body are unchanged from 1930f31; the commit holds only that file and follows the 40-11 ledger commit"
    requirement: "WARN-02"
    verification:
      - kind: other
        ref: "40-12-PLAN.md Task 3 <verify> script (printed 'OK: approve -- ... accepted_at 2026-10-10T19:14:57Z recorded ...')"
        status: pass
    human_judgment: false
  - id: D2
    description: "Every claim in the drafted reason is backed by a real guard test or an exact guard-docstring phrase, with no banned overstatement"
    requirement: "WARN-02"
    verification:
      - kind: other
        ref: "40-12-PLAN.md Task 1 <verify> draft check (printed 'OK: draft has 7 reason sentences and 40 evidence rows ...')"
        status: pass
    human_judgment: false
  - id: D3
    description: "Whether the recorded override applies to ROADMAP SC 2 (first half) / WARN-02"
    verification: []
    human_judgment: true
    rationale: "The verifier applies or rejects the override at re-verification against the guard; recording the reason does not apply it (overrides_applied stays as the verifier left it)"

duration: 2 min (resume segment; Task 1 ran in an earlier executor, draft written 2026-10-10T18:17:34Z)
completed: 2026-10-10
---

# Phase 40 Plan 12: Corrected WARN-02 Override Reason Summary

**The existing override in 40-VERIFICATION.md now carries the developer-approved reason (Tim Lister, 2026-10-10T19:14:57Z). It names what the extended guard reports: R12's five capabilities and every position it matches, R0's allow-list and automagic rule, R6's copy and write functions, and every review form reported on a copy of a real notebook. It names as residual only what the guard docstring names, and it no longer contains the "wherever the name appears" or "IPython's shell helpers" overstatements.**

## Performance

- **Duration:** about 2 min for this resume segment (Task 3 plus close-out). Task 1 (the draft) ran in the earlier executor; its draft file was written at 2026-10-10T18:17:34Z.
- **Started (resume):** 2026-10-10T19:15:47Z
- **Completed:** 2026-10-10T19:17:01Z
- **Tasks:** 3 of 3 (Task 1 draft, Task 2 developer decision, Task 3 record)
- **Files modified:** 1 in the repository (40-VERIFICATION.md), plus this SUMMARY

## Developer Decision

- **Decision:** `approve`. The draft was recorded exactly as written.
- **Approval time (UTC, taken when the reply arrived):** `2026-10-10T19:14:57Z`
- **How it was presented:** the orchestrator showed Tim Lister the full draft reason, the corrections list, the residual paragraph and the evidence summary, with the three options approve, amend and decline. The answer came back through the orchestrator's question prompt.
- **Scratch record (not committed):** `$HOME/tmp/phase40-12-override-decision.txt` (first line `approve`), `$HOME/tmp/phase40-12-override-approved.txt` (byte-identical copy of `$HOME/tmp/phase40-12-reason.txt`, the text that passed the Task 1 draft check), and `$HOME/tmp/phase40-12-override-approved-at.txt` (`2026-10-10T19:14:57Z`, the approval time the orchestrator supplied, not a new `date`).

## Recorded Reason (verbatim, whitespace-normalized)

> Rule R12, extended in 40-08 [301bd93, f1b4bb6] and 40-10 [482606b, 3232341, c6c6322], reports in every notebook any name it lists for five capabilities: Django's database settings and connection objects (DATABASES, DATABASE_ROUTERS, settings_dict, override_settings, modify_settings, the helper's internals, DatabaseWrapper, load_backend, connections, connection, _connections, ConnectionHandler, create_connection, get_connection_params, get_new_connection, Database, sqlite3); attaching another database through raw SQL (a string holding ATTACH followed by DATABASE, a quote, a placeholder or an opening parenthesis); starting a process (subprocess, multiprocessing, pty, pexpect, ptyprocess, psutil, jupyter_client, nbclient, nbconvert, system, popen, fork, forkpty, startfile, posix_spawn, posix_spawnp, every exec* and spawn* function of os, Popen, ProcessPoolExecutor, create_subprocess_exec, create_subprocess_shell, getoutput, getstatusoutput, IPython's get_ipython, system_raw, run_line_magic and run_cell_magic, and the IPython.utils helpers getoutputerror, get_output_error_code and process_handler); the environment (environ, environb, putenv, unsetenv); and running code from a string or another file (exec, eval, compile, __import__, __builtins__, importlib, runpy, builtins, ctypes, cffi, execute_from_command_line, ManagementUtility, import_string, resolve_name, locate, IPython's Macro, and call_command of shell, shell_plus or runscript). Each name is matched at every identifier the parser records, as the guard docstring enumerates them (attributes are never matched against the builtins exec, eval, compile, __import__ and __builtins__), and against every '.'- or ':'-separated part of a string or bytes constant; a string that parses and holds a statement or a call is read like the cell's own code, to two levels. So a listed name imported from another module (from webbrowser import subprocess), bound to another name first or looked up by a literal string (import_string('subprocess.run')) is reported. R0 reports any code cell that is not Python IPython runs unchanged, checked with IPython's own TransformerManager, once %matplotlib inline lines and a first-line %%time or %%capture (with only its --no-stderr, --no-stdout and --no-display flags and an output name) are removed, and any one-line cell whose first word is an IPython line magic or alias, which IPython's automagic would run, unless an earlier cell binds that name, the word is a builtin or a keyword, or it is followed by = or a comma. R6 reports shutil's copy, copy2, copyfile, copytree, copyfileobj and move and any write_bytes, called, referenced, imported or looked up through a string. Every literal form in the third review (CR-01, CR-02, WR-01), the re-review's four forms and both round-1 probes is reported on a copy of a real notebook, and every real pre-executed notebook passes the guard. The guard docstring makes no claim of completeness and names what stays out of reach: the guard reads only the notebook's own source, so values assembled or looked up at runtime (a name, module, SQL statement or file path built from pieces, or fetched with getattr, globals(), vars() or sys.modules using such a string), code decoded at runtime (pickle, marshal, codecs, base64), code outside the notebook (a module it imports, IPython's profile and start-up files with the extensions, aliases and magics they define, or a library function that starts a process or writes a file under a name these lists do not contain) and, left out on purpose, JavaScript or HTML in a cell's output, which a notebook front end may run when the notebook is opened, are not seen.

`accepted_by: "Tim Lister"` (unchanged), `accepted_at: "2026-10-10T19:14:57Z"` (was `"2026-10-10T13:46:17Z"`).

## Draft Check (Task 1)

Re-run on resume, before anything was written, against the same draft:

```
OK: draft has 7 reason sentences and 40 evidence rows, every test and docstring phrase real, no banned phrase, cited hashes touch the guard; verification report untouched
```

## Evidence Table (as presented)

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
| Automagic rule and its three exceptions | docstring: "is reported, because IPython's automagic would run it, unless an earlier cell binds that name, the word is a builtin or a keyword, or it is followed by ``=`` or ``,``" |
| Automagic rule | test: TestR0AllowList.test_a_one_line_magic_or_alias_is_a_problem_unless_the_notebook_defines_the_name, TestR0AllowList.test_the_automagic_rule_knows_every_line_magic_a_fresh_ipython_shell_has |
| R6: shutil's copy functions, move and write_bytes, in any spelling | docstring: "R6 reports shutil's ``copy``, ``copy2``, ``copyfile``, ``copytree``, ``copyfileobj`` and ``move`` (called, referenced or imported, through ``shutil`` or an alias of it), any ``write_bytes``, and any of these names looked up through a string" |
| R6 in any spelling | test: TestR6CopyAndWriteFunctions.test_every_copy_and_write_function_in_any_spelling_is_a_problem, TestIsolationRulesOnSyntheticNotebooks.test_byte_copy_is_a_problem |
| Third review CR-01 forms on a real notebook | test: TestThirdReviewFormsOnARealNotebook.test_subprocess_imported_from_another_module_is_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_a_process_looked_up_by_a_literal_string_is_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_ipython_process_helpers_are_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_the_connection_proxy_database_connect_and_attach_are_reported_on_a_real_notebook |
| Third review CR-02 and WR-01 forms on a real notebook | test: TestThirdReviewFormsOnARealNotebook.test_a_shell_escape_after_a_line_magic_is_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_ipython_aliases_and_shell_magics_are_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_a_one_line_alias_or_magic_that_is_valid_python_is_reported_on_a_real_notebook, TestThirdReviewFormsOnARealNotebook.test_copytree_and_write_bytes_are_reported_on_a_real_notebook |
| The re-review's four forms on a real notebook | test: TestReReviewFormsOnARealNotebook.test_a_connection_handler_store_is_reported_on_a_real_notebook, TestReReviewFormsOnARealNotebook.test_a_child_manage_py_with_a_bare_environment_is_reported_on_a_real_notebook, TestReReviewFormsOnARealNotebook.test_a_cleared_environment_and_a_child_manage_py_are_reported_on_a_real_notebook, TestReReviewFormsOnARealNotebook.test_a_shell_escape_running_manage_py_is_reported_on_a_real_notebook |
| Both round-1 probes on a real notebook | test: TestRerouteRuleOnARealNotebook.test_a_settings_dict_reroute_after_setup_is_reported_on_a_real_notebook, TestRerouteRuleOnARealNotebook.test_an_override_settings_reroute_after_setup_is_reported_on_a_real_notebook |
| Every real pre-executed notebook passes the guard | test: TestPreExecutedNotebooksUseTheHelper.test_every_real_pre_executed_notebook_is_isolated, TestPreExecutedNotebooksUseTheHelper.test_the_real_notebook_check_is_not_vacuous, TestRerouteRuleOnARealNotebook.test_the_unmodified_real_notebook_has_no_problems |
| The guard docstring makes no claim of completeness | docstring: "The guard makes no claim of completeness." |
| The guard reads only the notebook's own source | docstring: "it reads only the notebook's own source, so deliberate obfuscation is out of its reach" |
| Residual (1): values assembled or looked up at runtime | docstring: "(1) Values assembled or looked up at runtime: a name, module, SQL statement or file path built from pieces" |
| Residual (1): getattr, globals(), vars(), sys.modules; code decoded at runtime | docstring: "fetched with ``getattr``, ``globals()``, ``vars()`` or ``sys.modules`` using such a string, and code decoded at runtime (pickle, marshal, codecs, base64)" |
| Residual (2): code outside the notebook | docstring: "(2) Code outside the notebook: a module it imports, IPython's profile and start-up files (with the extensions, aliases and magics they define), or a library function that starts a process or writes a file under a name these lists do not contain" |
| Residual (3): JavaScript or HTML in a cell's output, left out on purpose | docstring: "(3) Left out on purpose: JavaScript or HTML in a cell's output, which a notebook front end may run when the notebook is opened" |

## Record Check (Task 3)

```
OK: approve -- the approved reason and accepted_at 2026-10-10T19:14:57Z recorded in the one override entry; must_have, accepted_by, every other key and the body unchanged; the commit is HEAD, after the 40-11 ledger commit
```

## Task Commits

1. **Task 1: Draft the corrected override reason with a claim-to-evidence table.** No commit. The plan keeps the draft in scratch only (`$HOME/tmp/phase40-12-override-draft.md`).
2. **Task 2: Developer approves, amends or declines (blocking-human).** No commit. The answer was `approve` at 2026-10-10T19:14:57Z.
3. **Task 3: Record the approved reason and the approval time.** Commit `44c33c6`, `docs(40-12): record the developer-approved corrected override reason for the residual WARN-02 risk (Tim Lister, 2026-10-10)`. It holds only 40-VERIFICATION.md and follows the 40-11 ledger commit `7a43251`.

**Commit ordering:** `44c33c6` is the round's last *plan* commit, as D-G5 (5) requires. The SUMMARY commit and the STATE.md/ROADMAP.md tracking commit that follow it are GSD bookkeeping, not plan work. Because of them, Task 3's verify script cannot pass again once they land: it checks that the override commit is HEAD and is the only `(40-12)` commit. It passed at the moment the override commit was HEAD (output above).

## Files Created/Modified

- `.planning/phases/40-notebook-isolation-attribution-page/40-VERIFICATION.md`: only the `overrides[0].reason` folded block (now 37 lines, wrapped to 110 columns, indented six spaces) and `overrides[0].accepted_at` changed. The must_have, `accepted_by: "Tim Lister"`, status, score, gaps, overrides_applied, human_verification, re_verification, covered_files, covered_digest, verified and the whole body are unchanged from `1930f31`.
- `.planning/phases/40-notebook-isolation-attribution-page/40-12-SUMMARY.md`: this file.

## Decisions Made

- D-G5 (5): Tim Lister approved the draft unchanged. The residual WARN-02 risk he accepted is exactly the guard docstring's "What the guard does not see" paragraph:
  1. values assembled or looked up at runtime;
  2. code decoded at runtime;
  3. code outside the notebook;
  4. JavaScript or HTML in a cell's output, left out on purpose.

## Deviations from Plan

**1. [Rule 3 - Blocking] A uniqueness assertion in the edit script was too strict**
- **Found during:** Task 3
- **Issue:** The first run of the edit script asserted that `    reason: >-` occurs only once in the file. It occurs three times (lines 54, 86 and 126; the other two are inside `gaps:` entries). The assertion fired before any write, so the file stayed unchanged (confirmed with `git diff --quiet`).
- **Fix:** Dropped that assertion. The script already finds the right block by its position, the first occurrence, which it checks sits directly under `overrides:` and the WARN-02 must_have line. It also checks that the old `accepted_by`/`accepted_at` lines follow the block.
- **Verification:** The pre-commit YAML check and the plan's Task 3 verify script both print OK.
- **Committed in:** `44c33c6`

**2. [Orchestrator instruction] The SUMMARY is committed**
- Task 3 step 4 of the plan says "Do not commit it". The orchestrator's resume instructions (sequential mode, execute-plan.md close-out) say to commit the SUMMARY with `SKIP=django-test` and then update STATE.md and ROADMAP.md. The orchestrator's instruction was followed. See "Commit ordering" above.

**Total deviations:** 1 auto-fixed (blocking), 1 orchestrator-directed. **Impact:** none on what was recorded.

## Issues Encountered

None.

## User Setup Required

None. No external service configuration is required.

## Next Phase Readiness

- Recording the reason does not apply the override. The verifier applies or rejects it at re-verification, checking it against the guard; `overrides_applied` stays at the verifier's value.
- Verification human items 2-4 (README and docstring wording, snapshot prose, runbook wording) are reading items for `/gsd-verify-work`, not tasks for this plan.
- Gap round 4 is complete (40-10, 40-11, 40-12). The phase is ready for `/gsd-verify-work 40` re-verification.

---
*Phase: 40-notebook-isolation-attribution-page*
*Completed: 2026-10-10*

## Self-Check: PASSED

- FOUND: .planning/phases/40-notebook-isolation-attribution-page/40-VERIFICATION.md (override reason equals the approved text, accepted_at 2026-10-10T19:14:57Z)
- FOUND: commit 44c33c6 (ancestor of HEAD, holds only 40-VERIFICATION.md)
- Draft check and record check both printed OK (above)
