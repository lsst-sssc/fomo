---
phase: 40-notebook-isolation-attribution-page
verified: 2026-10-10T20:24:06Z
status: gaps_found
score: 142/146 must-haves verified (prohibitions 42/42, one superseded by D-G2)
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
covered_digest: "v3:sha256:2b1c16ac43f2d9722914d0702205608153129d587376e64e9610d243284f979c"
behavior_unverified: 0
overrides_applied: 0
overrides:
  - must_have: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale"
    reason: >-
      Rule R12, extended in 40-08 [301bd93, f1b4bb6] and 40-10 [482606b, 3232341, c6c6322], reports in every
      notebook any name it lists for five capabilities: Django's database settings and connection objects
      (DATABASES, DATABASE_ROUTERS, settings_dict, override_settings, modify_settings, the helper's internals,
      DatabaseWrapper, load_backend, connections, connection, _connections, ConnectionHandler,
      create_connection, get_connection_params, get_new_connection, Database, sqlite3); attaching another
      database through raw SQL (a string holding ATTACH followed by DATABASE, a quote, a placeholder or an
      opening parenthesis); starting a process (subprocess, multiprocessing, pty, pexpect, ptyprocess, psutil,
      jupyter_client, nbclient, nbconvert, system, popen, fork, forkpty, startfile, posix_spawn, posix_spawnp,
      every exec* and spawn* function of os, Popen, ProcessPoolExecutor, create_subprocess_exec,
      create_subprocess_shell, getoutput, getstatusoutput, IPython's get_ipython, system_raw, run_line_magic
      and run_cell_magic, and the IPython.utils helpers getoutputerror, get_output_error_code and
      process_handler); the environment (environ, environb, putenv, unsetenv); and running code from a string
      or another file (exec, eval, compile, __import__, __builtins__, importlib, runpy, builtins, ctypes,
      cffi, execute_from_command_line, ManagementUtility, import_string, resolve_name, locate, IPython's
      Macro, and call_command of shell, shell_plus or runscript). Each name is matched at every identifier the
      parser records, as the guard docstring enumerates them (attributes are never matched against the
      builtins exec, eval, compile, __import__ and __builtins__), and against every '.'- or ':'-separated part
      of a string or bytes constant; a string that parses and holds a statement or a call is read like the
      cell's own code, to two levels. So a listed name imported from another module (from webbrowser import
      subprocess), bound to another name first or looked up by a literal string
      (import_string('subprocess.run')) is reported. R0 reports any code cell that is not Python IPython runs
      unchanged, checked with IPython's own TransformerManager, once %matplotlib inline lines and a first-line
      %%time or %%capture (with only its --no-stderr, --no-stdout and --no-display flags and an output name)
      are removed, and any one-line cell whose first word is an IPython line magic or alias, which IPython's
      automagic would run, unless an earlier cell binds that name, the word is a builtin or a keyword, or it
      is followed by = or a comma. R6 reports shutil's copy, copy2, copyfile, copytree, copyfileobj and move
      and any write_bytes, called, referenced, imported or looked up through a string. Every literal form in
      the third review (CR-01, CR-02, WR-01), the re-review's four forms and both round-1 probes is reported
      on a copy of a real notebook, and every real pre-executed notebook passes the guard. The guard docstring
      makes no claim of completeness and names what stays out of reach: the guard reads only the notebook's
      own source, so values assembled or looked up at runtime (a name, module, SQL statement or file path
      built from pieces, or fetched with getattr, globals(), vars() or sys.modules using such a string), code
      decoded at runtime (pickle, marshal, codecs, base64), code outside the notebook (a module it imports,
      IPython's profile and start-up files with the extensions, aliases and magics they define, or a library
      function that starts a process or writes a file under a name these lists do not contain) and, left out
      on purpose, JavaScript or HTML in a cell's output, which a notebook front end may run when the notebook
      is opened, are not seen.
    accepted_by: "Tim Lister"
    accepted_at: "2026-10-10T19:14:57Z"
re_verification:
  previous_status: gaps_found
  previous_score: 112/115 (prohibitions 31/31)
  gaps_closed:
    - "Round-3 CR-01 and CR-02 literal routes: re-exported subprocess, lookups by literal string, a shell escape after a line magic, IPython aliases, %alias, %rehashx and %pip are all reported now (82 guard tests green in the verifier's run)"
    - "CLAUDE.md paired-docs rule / D-G2 (round-3 WR-02): no notebook says 'never opened', 'opened for writing' or 'never ... writes the developer'; T1 is in all eight teardown cells; only six markdown cells changed and every code cell and output is identical to 1930f31"
  gaps_remaining:
    - "ROADMAP SC 2 (first half) / WARN-02: narrowed to two places where the guard's code is narrower than its stated claims (round-4 review CR-01 and WR-01). The override reason makes the CR-01 claim itself, so the override does not apply"
    - "40-08 must_have 8.14 / 40-11 must_have 11.1: the guard docstring, README and override reason still state those two claims"
  regressions: []
gaps:
  - truth: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale (round-4 review CR-01 and WR-01)"
    status: partial
    reason: >-
      Two literal routes in the notebook's own source fall outside the override's residual. The override can't be
      applied because one of its own claims is false. (1) Round-4 CR-01, reproduced by the verifier: the automagic
      rule skips a one-line cell whose first word is in `bound`. `_module_bindings` fills `bound` with every Name
      stored anywhere outside function, class, lambda and comprehension bodies. That includes a bare annotation,
      an `if False:` branch, the target of an empty `for` loop and an `except ... as` name. The verifier's inert
      check found that the guard counts each of these as bound, yet none of them leaves the name in the namespace
      at runtime (control: a plain assignment does). IPython's `is_shadowed` (prefilter.py:62-70) checks
      `ip.user_ns` at runtime. So a later one-line cell that starts with a default alias name is run by automagic
      and the guard reports nothing. Both cells are readable literal source, so this is not in residual 1, 2 or 3.
      The override reason says "unless an earlier cell binds that name", which the code does not check. (2)
      Round-4 WR-01, reproduced: `_byte_copy_problems` adds to `shutil_names` only from `import shutil as X`.
      Placeholder source that rebinds the module with a plain assignment and then calls `copytree` gives R6 0
      problems, R7 0 and R12 0. The reviewer rated this a warning because it said the destination must be built
      from joined path parts. The verifier disagrees. The scratch file has the developer database's base name, as
      R6's own docstring says, so a whole-directory copy can land on the developer database through a literal
      directory path. Such a path is neither assembled nor named. The guard docstring says R6 follows "shutil or
      an alias of it". The override reason says these functions are reported when "called".
    artifacts:
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "_module_bindings (lines 416-441) counts source-level stores that bind nothing at runtime, and ignores a later `del`; the R0 loop (lines 933-964) uses it as the automagic exemption"
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "_byte_copy_problems (lines ~704-760): shutil_names grows only from `import shutil as X`; neither `X = shutil` nor `from <module> import shutil` is followed"
    missing:
      - "Narrow the automagic exemption to bindings that certainly exist: unconditional top-level Assign, AnnAssign with a value, Import/ImportFrom, def and class, minus top-level `del`. Or drop the exemption entirely apart from a following `=` or `,`: no real notebook needs it (the verifier ran _automagic_word(view, set()) on every one-line cell of all eight notebooks and none needed it)"
      - "Add the four non-binding forms (bare annotation, if False, empty for, except-as) and a `del` case to test_a_one_line_magic_or_alias_is_a_problem_unless_the_notebook_defines_the_name, on a copy of a real notebook"
      - "R6: report the distinctive copy names (copy2, copyfile, copytree, copyfileobj) wherever they appear, as R12 does, and grow shutil_names from `X = shutil` and from any `import shutil`; add the two rebinding forms to TestR6CopyAndWriteFunctions"
      - "Keep all eight real notebooks green"
  - truth: "40-08 must_have 8.14 and 40-11 must_have 11.1: the guard docstring says what the guard reports and names exactly what stays out of reach"
    status: failed
    reason: "The root cause is the same as the gap above. Docstring lines 31-32 say 'unless an earlier cell binds that name', and lines 39-40 say R6 follows 'shutil or an alias of it'. README lines 94-96 say the guard checks that a notebook 'never byte-copies ... with shutil's copy functions' and 'has no one-line cell that IPython would run as a magic or an alias'. The developer-approved override reason repeats the first claim. All of these are true only after the fix above."
    artifacts:
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "module docstring lines 29-41"
      - path: "docs/notebooks/README.md"
        issue: "lines 91-97"
    missing:
      - "After the fix, make sure the docstring, the README and the override reason state what the code checks, for example 'unless an earlier cell binds that name with a top-level assignment, import, def or class'. Rewording the override reason needs the developer's approval again"
advisory:
  - finding: "Round-4 WR-02: R5/R7 read only str constants, so a bytes literal naming fomo_db.sqlite3 is not reported by R7"
    category: other
    reason: "Not blocking: R12 still reports it, because a bytes constant is split on '.' and 'sqlite3' is a listed name (verifier inert check: R12 1 problem). The docstring's R7 wording overstates; the _constant_text fix is cheap"
    evidence_status: "inert check run; route is caught by R12"
  - finding: "Round-4 IN-01: the helper's logged warning says 'no database is reachable'; a non-dict DATABASES alias raises TypeError in the switch loop"
    category: other
    reason: "Wording only. The cached wrappers are still dropped after the log line. A non-dict alias is refused by Django at the first query. Already recorded for Phase 41 in 40-11-SUMMARY"
    evidence_status: "code read (fomo_nb_utils.py _disable_django_databases)"
  - finding: "Round-4 IN-02: ipykernel reaches the dev environment only through the unpinned jupyter metapackage"
    category: other
    reason: "If ipykernel is missing, the guard fails loudly with ImportError rather than passing, so it fails closed. Pinning ipykernel in the dev extra is housekeeping"
    evidence_status: "code read (lines 402-406)"
human_verification:
  - test: "Decide on the guard fix and the override wording"
    expected: "Either fix CR-01 and WR-01 as described in gaps[0].missing and approve the override reason again (reworded if needed), or accept these two routes on purpose with a reason that names them. Accepting them would change what SC 2 means, so the wording has to be the developer's"
    why_human: "Accepting residual risk is the developer's decision. The verifier cannot widen an override"
  - test: "Read docs/notebooks/README.md lines 91-99 and the guard module docstring after the fix"
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
**Verified:** 2026-10-10T20:24:06Z (HEAD 4b56819; last code change c6c6322 / 7af64d0 docstrings / 17168ff notebook markdown)
**Status:** gaps_found
**Re-verification:** Yes, after gap-closure plans 40-10, 40-11 and 40-12

## Summary

Gap round 4 closed most of what it set out to close. Every round-3 literal route is now reported: 82 guard
tests, plus the helper and projector modules, ran 117 tests OK in the verifier's run. The six stale notebook
sentences were fixed with markdown-only edits, and the verifier confirmed by a cell diff that no code cell or
output changed. 40-11 changed only docstrings in the guard and helper modules (`ast.dump` is identical with
docstrings removed). The attribution half is unchanged and still passes (111 tests OK).

SC 2's first half is still open, in two narrow places. In both, the code checks less than the guard's own
docstring, the README and the developer-approved override reason say it does:

1. **Automagic exemption (review CR-01, reproduced).** The guard treats a name as defined if any earlier cell
   stores to it anywhere in its source, including statements that bind nothing at runtime. IPython decides from
   the runtime namespace. The override reason's clause "unless an earlier cell binds that name" is therefore
   false of the code. The verifier's rule is that an override applies only when every claim in it is true, so
   it is **not applied**.
2. **R6 shutil rebinding (review WR-01, reproduced; the verifier rates it higher than the review).** A plain
   rebinding of the `shutil` module is not followed. The scratch file has the developer database's base name, so
   copying the whole directory does not need the database to be named or a path to be assembled.

Neither form appears in any real notebook. The fix is small, and the review has already sketched it. No real
notebook relies on the automagic exemption.

To stay safe, the verifier used inert static checks (the guard's own helper functions on placeholder source) and
read IPython's source. It ran no notebook, ran no shell alias, and never opened `src/fomo_db.sqlite3`
(`git status --short src/` is clean).

## Goal Achievement

### Observable Truths: ROADMAP success criteria (the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | No notebook byte-copies or writes the dev DB; snapshot via read-only backup, others fresh; README rule and lists | ✓ VERIFIED | All eight real notebooks: `isolation_problems()` = [] (verifier script); code cells and outputs identical to 1930f31; `test_every_real_pre_executed_notebook_is_isolated` green |
| SC2a | The guard fails a notebook whose current source routes Django to the dev DB, even with stale output | ✗ FAILED (partial). Override checked and **not applicable** | Round-3 routes closed; round-4 CR-01 and WR-01 remain (gap 1) |
| SC2b | DB-path check raises an ordinary exception under `python -O` | ✓ VERIFIED | Helper code unchanged since 1930f31 (ast-equal without docstrings); `TestHelperFailClosedIsExceptionSafe` in the 117 OK |
| SC3 | telescope_runs_demo reports the dip in one unit | ✓ VERIFIED | Dip pin test green; code cells and outputs unchanged |
| SC4 | High-band marker, template test, BS 5.3 only | ✓ VERIFIED | No attribution file changed since 93aedf79; 111 OK |
| SC5 | Undo on a Dismissed row keeps the section open on the same page | ✓ VERIFIED | Unchanged files; the orchestrator's full gate at d688504 ran 40 functional tests OK |

### Override decision (ROADMAP SC 2 first half / WARN-02)

The approved override (Tim Lister, accepted_at 2026-10-10T19:14:57Z) is carried forward verbatim. The verifier
checked each claim in it against the code:

| Claim in the reason | True of the code? | Evidence |
|---------------------|-------------------|----------|
| R12 lists, positions, string parts and string code | Yes | 40-10 tests green; inert checks |
| "a listed name imported from another module, bound to another name first or looked up by a literal string is reported" | Yes | `TestThirdReviewFormsOnARealNotebook` green |
| R0 allow-list over TransformerManager, harmless magics | Yes | `TestR0AllowList` green |
| Automagic "unless an earlier cell binds that name" | **No** | `_module_bindings` counts a bare annotation, an `if False:` store, an empty `for` target and an `except ... as` name as bound. In the inert check, none of the four is bound at runtime, while the control (a plain assignment) is. IPython's `is_shadowed` reads `ip.user_ns` |
| R6 reports the copy functions when "called" | **Not in every literal spelling** | Placeholder source that rebinds `shutil` with an assignment and then calls `copytree`: R6 0, R7 0, R12 0 |
| Every review form and every real notebook pass | Yes | Tests green; 8/8 notebooks clean |
| Residual (runtime values, decoded code, code outside the notebook, output JS/HTML) | Accurate as far as it goes | The two routes above are literal notebook source and fall outside all three residual parts |

One claim is false, and one route falls outside the residual, so the override is not applied (`overrides_applied: 0`).

### Observable Truths: 40-10 must_haves

| # | Truth (abridged) | Status | Evidence |
|---|------------------|--------|----------|
| 10.1 | `from webbrowser/autoreload import subprocess` reported on a real notebook | ✓ VERIFIED | Test green |
| 10.2 | Every R12 table at every identifier position | ✓ VERIFIED | Test green |
| 10.3 | Literal-string lookups, string parts, string code | ✓ VERIFIED | Tests green |
| 10.4 | New names; lists disjoint; IPython helpers on a real notebook | ✓ VERIFIED | Tests green; constants read |
| 10.5 | connection proxy, Database, `ATTACH (` | ✓ VERIFIED | Tests green; ATTACH_SQL line 213 |
| 10.6 | R6 copytree/copyfileobj/move/write_bytes; WR-01 (round 3) forms | ✓ VERIFIED (listed forms) | Tests green. A rebound `shutil` is not followed (gap 1, round-4 WR-01) |
| 10.7 | Look-alikes allowed | ✓ VERIFIED | Test green |
| 10.8 | R0 allow-list via TransformerManager; code under harmless magic checked | ✓ VERIFIED | Tests green; isolation_problems lines 929-967 |
| 10.9 | CR-02 forms reported on a real notebook | ✓ VERIFIED | Tests green |
| 10.10 | Valid-Python automagic forms and prompt prefix reported | ✓ VERIFIED | Test green |
| 10.11 | Automagic rule: exempt only names "the notebook has bound at module level in an earlier cell" | ✗ FAILED (partial) | The exemption covers stores that bind nothing (gap 1, CR-01) |
| 10.12 | Harmless magic lines clean; %config, other %matplotlib reported | ✓ VERIFIED | Test green |
| 10.13 | Empty and ordering edges | ✓ VERIFIED | Code read (one R0 problem per cell, in cell order) |
| 10.14 | No weakening | ✓ VERIFIED | No `def test_` removed from either module since 1930f31; no skip, expectedFailure or tag added |
| 10.15 | Ledger CR-02 and WR-01 fixed | ✓ VERIFIED (historical) | f4f2cd1. The round-4 ledger later reused the ids, and the provenance is kept in fed292c |
| 10.16 | Gate | ✓ VERIFIED | Commit file sets per 40-10-SUMMARY; `git log` order checked |

### Observable Truths: 40-11 must_haves

| # | Truth (abridged) | Status | Evidence |
|---|------------------|--------|----------|
| 11.1 | Docstring "What the guard checks" describes what 40-10 made the guard report | ✗ FAILED (partial) | Accurate except two clauses: "unless an earlier cell binds that name" and "shutil or an alias of it" (gap 2) |
| 11.2 | "What the guard does not see" is exactly RES | ✓ VERIFIED | Lines 69-77 |
| 11.3 | 40-08 phrases kept; guard code identical to 40-10 | ✓ VERIFIED | ast-equal to c6c6322 with docstrings removed |
| 11.4 | README guard paragraph is README_GUARD | ✓ VERIFIED (text) | Lines 91-97. The content overstates the same two points (gap 2) |
| 11.5 | IN-01 helper docstring; helper code unchanged | ✓ VERIFIED | Docstring read; ast-equal to 1930f31 |
| 11.6 | Docstring only, no re-execution | ✓ VERIFIED | Code cells and outputs identical to 1930f31 in all eight notebooks |
| 11.7 | T1 in 46602df0, 234805b6, 31ff3edb; F7/F49 in setup cells | ✓ VERIFIED | Changed cells are exactly those plus 23326d63; T1 present 8/8 |
| 11.8 | Widened style check clean | ✓ VERIFIED | `never opened`, `never .*writes the developer`, `opened for writing`: 0 hits in 8/8 |
| 11.9 | Markdown only | ✓ VERIFIED | Cell diff (verifier script) |
| 11.10 | Ledger rows | ✓ VERIFIED (historical) | 7a43251; later superseded by the round-4 ledger with provenance |
| 11.11 | Gate | ✓ VERIFIED | Commit file sets per summary; no runbook change |

### Observable Truths: 40-12 must_haves

| # | Truth (abridged) | Status | Evidence |
|---|------------------|--------|----------|
| 12.1 | Draft with Reason and Evidence table | ✓ VERIFIED | 40-12-SUMMARY evidence table (40 rows) |
| 12.2 | Every sentence backed; no banned phrase | ✓ VERIFIED (literal) | No "wherever it appears" and no "IPython's shell helpers" in the reason. Being backed by a test does not make the automagic clause true of the code (see Override decision) |
| 12.3 | Blocking-human approval | ✓ VERIFIED | Approved 2026-10-10T19:14:57Z |
| 12.4 | Only reason and accepted_at changed; commit holds only the report | ✓ VERIFIED | `git show --stat 44c33c6`: only 40-VERIFICATION.md; the diff outside the reason block is accepted_at only |

The decline branch (12.5) does not apply.

### Carried truths (40-01 to 40-09)

The previous report's 115 truths were checked for regressions. Nothing they depend on changed in code: the
helper code is ast-identical, notebook code cells and outputs are identical, and the attribution files are
unchanged. PD.1 (paired prose) is now ✓ VERIFIED. SC2a and 8.14 stay failed, now for the round-4 reasons.

**Score:** 142/146 truths verified, 0 present but behavior-unverified. The 146 are 115 carried (113 verified;
SC2a and 8.14 failed), 16 from 40-10 (10.11 failed), 11 from 40-11 (11.1 failed) and 4 from 40-12. All four
failures have one root cause.

### Prohibitions

| Plan | Prohibition | Status | Evidence |
|------|-------------|--------|----------|
| 40-01..40-09 | 31 carried | ✓ VERIFIED (one superseded by D-G2) | No regression |
| 40-10 | Guard not weakened | ✓ VERIFIED | No test removed; no skip or tag added; EXPECTED_NOTEBOOKS unchanged |
| 40-10 | No dev DB, no FOMO_DATABASE_PATH, no IPython shell in the test process | ✓ VERIFIED | `git status --short src/` clean |
| 40-10 | No notebook, README, helper or VERIFICATION edit | ✓ VERIFIED | 40-10 commit file sets |
| 40-10 | Out-of-scope items untouched | ✓ VERIFIED | No attribution, runbook or campaign_views change |
| 40-11 | No code change in guard or helper; no re-execution | ✓ VERIFIED | ast-equal checks; outputs identical |
| 40-11 | No overclaiming wording | ✓ VERIFIED (literal) | Banned phrases absent. The two remaining overstatements are recorded as gap 2 |
| 40-11 | No VERIFICATION, review or runbook edit | ✓ VERIFIED | Commit file sets |
| 40-11 | No dev DB access | ✓ VERIFIED | |
| 40-12 | Nothing written before approval; only approved text | ✓ VERIFIED | Summary timeline; single commit 44c33c6 |
| 40-12 | Other front-matter keys and body unchanged | ✓ VERIFIED | Diff checked |
| 40-12 | No code, test, notebook, README or ledger edit; no push | ✓ VERIFIED | Commit file set |

### Round-4 review findings (40-REVIEW.md): explicit classification

| Finding | Reproduced? | Classification | Reasoning |
|---------|-------------|----------------|-----------|
| CR-01 automagic exemption wider than runtime binding | **Yes** (inert `_module_bindings` check against plain Python namespaces; IPython `is_shadowed` source read) | **Gap** (gap 1 and gap 2) | Literal notebook source outside every residual part, and it falsifies a claim the override itself makes |
| WR-01 R6 follows `shutil` only through `import shutil as X` | **Yes** (inert `_byte_copy_problems` check: 0 problems) | **Gap** (part of gap 1; rated higher than the review) | The scratch file has the dev DB's base name, so a whole-directory copy to a literal directory needs neither a joined path nor the DB name. The review's reason for rating it a warning does not hold |
| WR-02 R5/R7 read only str constants | **Partly.** R7 misses a bytes literal, but R12 reports it through the `sqlite3` part | **Not a gap** (advisory) | The route is still reported; only the R7 docstring wording overstates |
| IN-01 log wording; non-dict alias TypeError | Code read | **Not a gap** (advisory) | Fail-closed behavior holds; wording only. Phase 41 |
| IN-02 ipykernel through the jupyter metapackage | Code read | **Not a gap** (advisory) | A missing ipykernel makes the guard error out, which fails closed |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/tests/test_pre_executed_notebooks.py` | R12, R6, R0 allow-list and automagic rule, real-notebook tests, accurate docstring | ⚠️ VERIFIED with coverage gap | 82 tests green; CR-01 and WR-01 (gap 1); docstring clauses (gap 2) |
| `docs/notebooks/README.md` | Guard paragraph and IN-01 clause | ⚠️ VERIFIED (text) | Overstates the same two points (gap 2) |
| `docs/notebooks/fomo_nb_utils.py` | IN-01 docstring only | ✓ VERIFIED | Code ast-identical to 1930f31 |
| 4 edited notebooks | Six markdown sentences | ✓ VERIFIED | Cell diff; style check clean |
| 40-VERIFICATION.md override | Approved reason recorded | ✓ VERIFIED (recorded, not applicable) | Carried verbatim |
| Attribution artifacts | Unchanged | ✓ VERIFIED | 111 OK |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| R0 loop | `_python_view`, `_ipython_rewrites`, `_automagic_word` | lines 936-962 | ✓ WIRED (exemption too wide: gap 1) |
| R0 loop | later rules | `parsed = views` (line 967) before `problems.extend(...)` | ✓ WIRED |
| `_automagic_word` | IPython magic registry | `_ipython_line_magic_names()` with `default_aliases` | ✓ WIRED |
| `TestThirdReviewFormsOnARealNotebook` | import_campaign_csv_demo.ipynb | deep copies | ✓ WIRED |
| Teardown markdown | teardown output line 2 | T1 | ✓ WIRED (8/8) |
| Override reason | docstring RES | same residual text | ✓ WIRED (one false claim: see Override decision) |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data | Status |
|----------|------|--------|-----------|--------|
| Notebook Resolved line | resolved NAME | lazy settings, checked before setup | Yes | ✓ FLOWING (unchanged) |
| Teardown second line | what setup checked | `check_database_settings` before `django.setup()` | Yes | ✓ FLOWING |
| Attribution page state | return state | unchanged | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Guard, helper, projector | `python manage.py test --noinput --parallel auto ...test_notebook_helper ...test_pre_executed_notebooks ...test_projector_demo_notebook` | Ran 117 tests, OK | ✓ PASS |
| Attribution regression | `... test_attribution_template test_campaign_attribution_views test_bootstrap5_rendering --exclude-tag functional` | Ran 111 tests, OK | ✓ PASS |
| CR-01 binding helper vs runtime | `$HOME/tmp/verify40r4/inert.py` (placeholder name, no alias run) | 4 non-binding forms: guard bound True, runtime False; control True/True | ✗ FAIL (gap 1) |
| WR-01/WR-02 inert | `$HOME/tmp/verify40r4/inert2.py` (placeholder arguments) | rebound shutil: R6 0; bytes DB literal: R12 1 | ✗ FAIL (WR-01) / ✓ (WR-02 caught) |
| Real notebooks | same script | 8/8 `isolation_problems` = []; no one-line cell needs the exemption | ✓ PASS |
| Notebook prose and code | `$HOME/tmp/verify40r4/prose.py` | stale 0 in 8/8; code and outputs same; changed cells as planned; helper and guard code ast-identical | ✓ PASS |
| Dev DB untouched | `git status --short src/` | clean | ✓ PASS |

### Probe Execution

Step 7c: SKIPPED. No `scripts/*/tests/probe-*.sh` exists, and no plan declares one.

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|--------------|-------------|--------|----------|
| WARN-02 | 40-02, 40-06..40-12 | Guard fails a notebook whose current source routes Django to the dev DB | ✗ BLOCKED (partial) | Gap 1; override not applicable |
| WARN-03 | 40-01, 40-02, 40-05, 40-07, 40-08, 40-09, 40-11 | Ordinary exception under `-O` | ✓ SATISFIED | SC2b |
| WARN-04 | 40-03, 40-04 | BS 5.3 only; High-band marker; template test | ✓ SATISFIED | SC4 |
| WARN-05 | 40-01, 40-02, 40-05, 40-07..40-11 | Audit, shared helper, no byte copy or write, README rule and lists | ✓ SATISFIED | SC1; paired prose now fixed |
| WARN-06 | 40-02 | Dip in one unit | ✓ SATISFIED | SC3 |
| WARN-07 | 40-04 | Undo keeps Dismissed open on the same page | ✓ SATISFIED | SC5 |

All six IDs appear in plan frontmatter and in REQUIREMENTS.md (rows 102-107, Phase 40). No requirement is orphaned.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| solsys_code/tests/test_pre_executed_notebooks.py | 416-441 | Exemption built from source-level stores, not runtime bindings | 🛑 Blocker | Gap 1 (carried-forward gap, and the file changed since the prior verification) |
| solsys_code/tests/test_pre_executed_notebooks.py | ~704-760 | `shutil` followed only through `import shutil as X` | 🛑 Blocker | Gap 1 (same evidence basis) |
| solsys_code/tests/test_pre_executed_notebooks.py, docs/notebooks/README.md | 29-41; 91-97 | Wording claims more than the code does | ⚠️ Warning | Gap 2 |
| solsys_code/tests/test_pre_executed_notebooks.py | ~694 | R7 str-only | ℹ️ Info | Advisory (caught by R12) |
| docs/notebooks/fomo_nb_utils.py | 208-210 | Log wording | ℹ️ Info | Advisory (IN-01) |

No TBD, FIXME, XXX, TODO or HACK in lines added since 1930f31.

### Human Verification Required

These items do not change the gaps_found status.

1. **Guard fix and override wording.** Fix CR-01 and WR-01, and approve the override reason again (reworded
   if needed), or accept the two routes on purpose with a reason that names them. Why human: accepting
   residual risk is the developer's decision.
2. **README guard paragraph and guard docstring** after the fix. Why human: wording.
3. **Snapshot notebook prose** (carried). Why human: prose accuracy.
4. **Runbook attribution-pass wording** (round-1 IN-03, kept for Phase 41). Why human: wording.

### Gaps Summary

There is one root cause, which shows up as two gaps: **the guard's code is narrower than what its docstring,
the README and the approved override reason say it checks.**

- The automagic exemption trusts any store of the name in the source. It should trust only statements that
  certainly bind the name: top-level assignment, import, def and class, minus `del`. No real notebook relies on
  the exemption, so dropping it is also safe.
- R6 should follow a rebound `shutil`, or report the distinctive copy names wherever they appear, as R12 does.

Both fixes change only the guard test module, which has no paired notebook. Once they are in, the override
reason's clauses become true of the code. Rewording the reason needs the developer's approval again. Nothing
here is deferred: Phases 41 and 42 do not cover the notebook guard.

---

_Verified: 2026-10-10T20:24:06Z_
_Verifier: Claude (gsd-verifier)_
