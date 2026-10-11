---
phase: 40-notebook-isolation-attribution-page
verified: 2026-10-11T01:51:02Z
status: gaps_found
score: 163/167 must-haves verified (prohibitions 49/49, one superseded by D-G2)
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
covered_digest: "v3:sha256:a46cd003db2a6ed6a21677d9339a0cfe408457ef98518d3ad09ecec17f9d83dd"
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
  previous_score: 142/146 (prohibitions 42/42)
  gaps_closed:
    - "Round-4 CR-01 (automagic exemption): `_module_bindings` is gone and `_automagic_word(view)` exempts only a builtin, a keyword or a word followed by = or ,. The verifier's inert run on a copy of import_campaign_csv_demo: each of the seven rm bindings alone gives [], and with an `rm -r -- '../../src'` cell after the setup cell each gives one R0 automagic problem"
    - "Round-4 WR-01 (rebound shutil): `sh = shutil` then `sh.copytree(...)` and `from zipfile import shutil as s` then `s.copytree(...)` are each reported by R6 (verifier inert run)"
    - "40-10 must_have 10.11: superseded and met, because the exemption it narrowed no longer exists"
    - "Round-4 advisories WR-02 (R5/R7 bytes) and IN-02 (ipykernel in the dev extra): both fixed (4f02628, ccfc578)"
  gaps_remaining:
    - "ROADMAP SC 2 (first half) / WARN-02: now narrowed to the round-5 review's CR-01 and CR-02 routes. They are outside the override's residual, and the override reason itself claims 'any write_bytes', so the override does not apply"
    - "40-08 must_have 8.14 / 40-11 must_have 11.1, joined by 40-14 must_have 14.2: the guard docstring, README and override reason still claim more than the code checks (CR-01, CR-02; WR-01 wording)"
  regressions: []
gaps:
  - truth: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale (round-5 review CR-01 and CR-02)"
    status: partial
    reason: >-
      Two families of literal notebook source escape R6, and neither is named in the override's residual. The
      verifier reproduced both with the guard's own isolation_problems() on a deep copy of
      import_campaign_csv_demo.ipynb with the source appended to its setup cell. (1) CR-01: shutil handed to a
      parameter as a call argument is never added to the shutil names. Six forms give []: `def put(sh): ...
      sh.copy(f, repo_root_path / 'src')` then `put(shutil)`; an immediately called lambda; `map(lambda sh:
      sh.move(...), [shutil])`; a keyword-only parameter given `sh=shutil`; `match shutil: case sh:`; and `match
      shutil: case object(copy=c): c(...)`. The control, a parameter default `def put(sh=shutil)`, is reported. (2)
      CR-02: `match repo_root_path / 'src' / f.name: case object(write_bytes=w): w(f.read_bytes())` gives [],
      because BYTE_WRITE_NAMES is matched only as a Name, an Attribute or a string part, never as a class-pattern
      attribute (`_identifiers` yields kwd_attrs, but R6 uses `_identifiers` only for DISTINCTIVE_COPY_NAMES). The
      control `.write_bytes(...)` is reported. A plain Python check with a stand-in object confirmed that a
      parameter, a lambda argument and `case object(copy=c)` / `case object(write_bytes=w)` all bind the method at
      runtime. The scratch file is named fomo_db.sqlite3, so each form lands on src/fomo_db.sqlite3 through a literal
      directory path. Against the override's exact residual words: the routes are not a function's return value, a
      call's side effect or a module holding shutil under another name (R6's stated boundary), and they are not
      values assembled at runtime, code outside the notebook, or output JavaScript or HTML (RES). They are outside
      the accepted residual. The reason also says R6 reports "any write_bytes", which CR-02 shows is false. So the
      override is carried forward verbatim and not applied. Practical weight, for the developer: residual (2)
      already accepts `Path.replace`, `rename` and `open` in a write mode, which give the same effect, so these
      routes add no new capability. The finding is that the guard's code and its documented boundary disagree, the
      standard this phase has applied in every round.
    artifacts:
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "_binding_pairs / _shutil_names (lines ~722-792) follow assignment-shaped bindings and parameter defaults only; a parameter given shutil as a call argument, a match capture and a class-pattern attribute are not followed"
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "_byte_copy_problems (lines ~858-863) matches BYTE_WRITE_NAMES only as ast.Attribute or ast.Name, not at every identifier"
    missing:
      - "R6: report the module name itself (`shutil`, and tempfile's `_shutil`) at every identifier and string part, like DISTINCTIVE_COPY_NAMES, which also closes the three exceptions R6's boundary now names. No real notebook mentions shutil (grep count 0 in all eight). Or add the parameter-argument, lambda, match-capture and class-pattern routes to the stated boundary in the docstring, README and override reason (that needs the developer's approval)"
      - "R6: match BYTE_WRITE_NAMES through `_identifiers`, as DISTINCTIVE_COPY_NAMES is. No real notebook contains write_bytes (grep count 0 in all eight)"
      - "Add the six CR-01 forms and the CR-02 class-pattern form to TestR6CopyAndWriteFunctions, on a copy of a real notebook"
      - "Keep all eight real notebooks green"
  - truth: "40-08 must_have 8.14, 40-11 must_have 11.1 and 40-14 must_have 14.2: the guard docstring says what the guard reports and names exactly what stays out of reach (and the 40-14 draft names as not followed exactly the forms the code does not follow)"
    status: failed
    reason: >-
      Same root cause as the gap above, plus one wording point. The module docstring (lines 47-54) and README (lines
      95-97) say R6 reports "any write_bytes" and reports copy and move "when they are read from shutil or from a name
      the notebook binds to it". A parameter or a match capture is such a name, and the boundary does not list it.
      README line 100 says the docstring "says exactly what each rule follows", which the CR-01/CR-02 table
      contradicts. Docstring lines 39-40 say "The code under a harmless magic is checked by every later rule", but
      `is_django_backed` (line ~586) parses the raw source, so `%%capture` / `x = 1; import django as dj` /
      `dj.setup()` switches R2-R4 and R8-R11 off in isolation_problems() (verifier inert run: is_django_backed False,
      problems []; the same code without the magic gives two problems). This is the review's WR-01. It is not a
      route in the suite, because test_the_real_notebook_check_is_not_vacuous requires all eight notebooks to be
      is_django_backed on raw source (verifier: the same cell inserted into a copy of a real notebook is reported).
      But the docstring claim is false of the function it describes. The override reason, approved by the
      developer, repeats "any write_bytes" and the R6 boundary.
    artifacts:
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "module docstring lines 39-40 (every later rule), 47 (any write_bytes), 49-54 (R6 boundary); _byte_copy_problems docstring 'any reference to a byte-write method'"
      - path: "docs/notebooks/README.md"
        issue: "lines 95-97 (write_bytes, 'a name the notebook binds to it') and line 100 ('says exactly what each rule follows')"
    missing:
      - "After the R6 fix, make the docstring, README and override reason state what the code checks. Rewording the override reason needs the developer's approval again"
      - "WR-01: either gate R2-R4 and R8-R11 on the Python views (`any(tree is not None and _django_imports(tree) for _n, _c, tree in parsed)`) and add the review's probe to TestR0AllowList, or drop 'every later rule' from the docstring"
advisory:
  - finding: "Round-5 WR-02: R0 does not see IPython's dynamic autocall of IPyAutocall objects; `exit -1` is valid Python, and IPython rewrites it to `exit(-1)` (IPyAutocallChecker, prefilter.py; InteractiveShell.transform_cell runs prefilter on one-line cells)"
    category: other
    reason: "No database route: exit and quit only end the kernel, and the __call__ body of a user-defined autocall object is notebook code that every rule reads. The docstring's 'Python that IPython runs unchanged' overstates. Cheap to fix in the same wording pass ('apart from IPython's autocall of exit and quit', or add IPyAutocall to DYNAMIC_CODE_NAMES)"
    evidence_status: "inert check run (guard returns [] for `exit -1`); IPython 9.10.1 source read"
  - finding: "Round-5 IN-01: README and docstring leave out the R5 keyword check, R10, R8's error-output check and R11's setup-cell assert check"
    category: other
    reason: "Leaving a check out understates coverage and makes no false claim; the README points to the docstring"
    evidence_status: "code read (isolation_problems R8, R10, R11)"
  - finding: "Round-5 IN-02: the magic-name pin test uses a plain InteractiveShell, not ipykernel's ZMQInteractiveShell"
    category: other
    reason: "The two sets match today. Verifier child process (own IPYTHONDIR): ZMQInteractiveShell with storemagic minus the guard set is empty in both directions. A future ipykernel magic would go unnoticed until the pin is switched"
    evidence_status: "child-process check run"
  - finding: "Round-4 IN-01 (carried): the helper's logged warning wording; a non-dict DATABASES alias"
    category: other
    reason: "Out of scope by developer decision 2026-10-10; recorded for Phase 41"
    evidence_status: "code read; unchanged since the previous report"
human_verification:
  - test: "Decide how to close the round-5 R6 routes and approve the override wording again"
    expected: "Either (A) report `shutil` and match write_bytes at every identifier, then reword the docstring, README and override reason to the simpler claim and approve it, or (B) keep the code and approve a reason and docstring whose R6 boundary names a parameter given shutil as a call argument, a match capture and a class-pattern attribute, and that drops 'any write_bytes'"
    why_human: "Accepting residual risk and wording the override are the developer's decisions. The verifier cannot widen an override"
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
**Verified:** 2026-10-11T01:51:02Z (HEAD cb81344; last source change 1a315e2 docstring/README, last code change 4f02628, ccfc578 pyproject)
**Status:** gaps_found
**Re-verification:** Yes, after gap-closure plans 40-13 and 40-14 (gap round 5)

## Summary

Gap round 5 closed both round-4 gaps in code. The automagic exemption is gone. A rebound `shutil` is followed through
every assignment-shaped binding, and the four distinctive copy names are reported wherever they are named. R5 and R7
read bytes, and ipykernel is declared. The developer-approved texts (decision `approve`, 2026-10-10T23:20:47Z) are
exactly what was committed: the docstring, the README paragraph and the override reason each match their saved
approved file. The guard's code is identical to 40-13's (ast check with docstrings removed). Guard, helper and
projector modules: 121 tests OK in the verifier's run. The orchestrator ran the full configured suite at 51da348
(2344 + 40 functional tests OK), and no source has changed since.

SC 2's first half is still open, in one narrow place. R6 only follows shutil through assignment-shaped bindings
and parameter defaults. So shutil handed to a function or lambda as an argument, a `match` capture, or a class-pattern
attribute reaches `copy`/`move` unreported (review CR-01). `write_bytes` read as a class-pattern attribute is not
reported either (CR-02). The verifier reproduced all seven forms on a copy of a real notebook. None is named in R6's
stated boundary or in RES, and the override reason claims "any write_bytes". So the override is **carried forward
verbatim and not applied**.

The practical risk is small, and the developer has already accepted the same effect through other routes:
residual (2) accepts `Path.replace`, `rename` and `open` in a write mode. Both fixes are a few lines: report the
`shutil` module name and `write_bytes` at every identifier. No real notebook contains either name.

To stay safe, the verifier made only inert calls of the guard's own functions on placeholder source or on deep
copies of a real notebook. It used one child process with its own `IPYTHONDIR` to list ipykernel's magics. It ran no
notebook, never called `django.setup()` in-process, never set `FOMO_DATABASE_PATH`, and never opened
`src/fomo_db.sqlite3` (`git status --short src/` is clean).

## Goal Achievement

### Observable Truths: ROADMAP success criteria (the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | No notebook byte-copies or writes the dev DB; snapshot via read-only backup, others fresh; README rule and lists | ✓ VERIFIED | No notebook or helper changed since the previous report (`git diff e08f8eb HEAD` empty for pre_executed/ and fomo_nb_utils.py); all eight real notebooks: `isolation_problems()` = [] (verifier script); `grep -c shutil` and `grep -c write_bytes` are 0 in all eight |
| SC2a | The guard fails a notebook whose current source routes Django to the dev DB, even with stale output | ✗ FAILED (partial). Override checked and **not applicable** | Round-4 routes closed; round-5 CR-01 and CR-02 reproduced (gap 1) |
| SC2b | DB-path check raises an ordinary exception under `python -O` | ✓ VERIFIED | Helper unchanged; `TestHelperFailClosedIsExceptionSafe` among the 121 OK |
| SC3 | telescope_runs_demo reports the dip in one unit | ✓ VERIFIED | Notebook unchanged; dip pin test green |
| SC4 | High-band marker, template test, BS 5.3 only | ✓ VERIFIED | No attribution file changed since the previous report (`git diff 4b56819 HEAD` touches only README, pyproject and the guard module); full suite OK at 51da348 |
| SC5 | Undo on a Dismissed row keeps the section open on the same page | ✓ VERIFIED | Unchanged files; 40 functional tests OK in the orchestrator's full run |

### Override decision (ROADMAP SC 2 first half / WARN-02)

The override (Tim Lister, accepted_at 2026-10-10T23:20:47Z, re-recorded in 8408731) is carried forward verbatim.
The verifier tested each review route against the reason's and the docstring's exact residual words.

| Route | Named by R6's boundary ("a function's return value, a call's side effect, a module that holds shutil under another name")? | Named by RES (runtime-assembled values, decoded code, code outside the notebook, output JS/HTML)? | Verdict |
|-------|------|------|---------|
| CR-01 `def put(sh)` + `put(shutil)` | No: shutil is a call argument, not a return value or a side effect | No: literal source | Outside residual |
| CR-01 immediately called lambda; `map(lambda sh: ..., [shutil])`; `put(sh=shutil)` | No | No | Outside residual |
| CR-01 `match shutil: case sh:` | No: a capture | No | Outside residual |
| CR-01 `case object(copy=c)` | No: a class-pattern attribute | No | Outside residual |
| CR-02 `case object(write_bytes=w)` | No boundary is stated for write_bytes; the reason says "any write_bytes" | No | Outside residual, and it falsifies a claim in the reason |
| WR-01 Django under `%%capture` switching off R2-R4/R8-R11 | n/a | n/a | Not a route in the suite (vacuity test); a docstring claim, which the reason does not make |
| WR-02 `exit -1` autocall | n/a | n/a | No database capability; the reason's R0 clause is qualified by "checked with IPython's own TransformerManager" |
| Accepted, for comparison: `f.replace(...)`, `rename`, `open(..., 'wb')` | n/a | Yes, residual (2) names them | Inside residual |

Five route families are outside the residual, and the reason contains one false claim ("any write_bytes"). So the
override does not apply (`overrides_applied: 0`).

### Observable Truths: 40-13 must_haves

| # | Truth (abridged) | Status | Evidence |
|---|------------------|--------|----------|
| 13.1 | Seven rm bindings alone clean; with an alias cell, only R0 automagic problems | ✓ VERIFIED | Verifier inert run (7/7); test green |
| 13.2 | No `_module_bindings`; `_automagic_word(view)`; `import time` then `time` reported; exemptions only builtin/keyword/=/, | ✓ VERIFIED | `grep -c _module_bindings` = 0; code lines 432-464; tests green |
| 13.3 | Problem text and docstrings say "whatever an earlier cell binds" | ✓ VERIFIED | Problem text at the R0 loop; `_automagic_word` docstring |
| 13.4 | DISTINCTIVE_COPY_NAMES at every identifier; both rebinding forms and copy/move variants reported | ✓ VERIFIED | Line 259; second pass over `_identifiers`; verifier inert run |
| 13.5 | copy/move tied to shutil names through imports and assignment-shaped bindings; listed forms reported | ✓ VERIFIED (listed forms) | Tests green. Call-argument and match-capture routes are not in this truth's list; they are gap 1 |
| 13.6 | Look-alikes allowed | ✓ VERIFIED | Test green |
| 13.7 | R5/R7 read str and bytes | ✓ VERIFIED | `_constant_text(node)` in `_database_routing_problems`; test green |
| 13.8 | `ipykernel>=7` the only pyproject change | ✓ VERIFIED | pyproject line 39; ccfc578 is 1 line |
| 13.9 | Empty edge | ✓ VERIFIED | Code read; tests green |
| 13.10 | Ordering edge: shutil names collected once before any cell | ✓ VERIFIED | `_byte_copy_problems` computes `shutil_names` before its cell loop |
| 13.11 | No weakening | ✓ VERIFIED | No skip, expectedFailure or tag decorator in the module; 86 guard tests (82 before); EXPECTED_NOTEBOOKS unchanged |
| 13.12 | Gate | ✓ VERIFIED | 905408a and 4f02628 hold only the guard module, ccfc578 only pyproject.toml |

### Observable Truths: 40-14 must_haves

| # | Truth (abridged) | Status | Evidence |
|---|------------------|--------|----------|
| 14.1 | Draft with five sections before any change | ✓ VERIFIED | `$HOME/tmp/phase40-14-draft.md` (15:28 local), approved files 16:21, commit 1a315e2 16:23 |
| 14.2 | Drafted "What the guard checks" says exactly what 40-13 checks and names exactly the forms not followed | ✗ FAILED (partial) | R6's boundary omits the CR-01 routes; "any write_bytes" is false (CR-02); "checked by every later rule" is false of `isolation_problems` (WR-01). Gap 2 |
| 14.3 | README paragraph content and phrases | ✓ VERIFIED (text) | Required phrases present; the same overstatements as 14.2 (gap 2) |
| 14.4 | Override reason structure, hashes, evidence rows, no banned phrase | ✓ VERIFIED (literal) | Reason equals the approved text; being approved does not make "any write_bytes" true (see Override decision) |
| 14.5 | Blocking-human checkpoint; reply saved | ✓ VERIFIED | `phase40-14-decision.txt`: `approve`, `approved_at: 2026-10-10T23:20:47Z` |
| 14.6 | Docstring and README equal the approved texts; guard code identical to 40-13 | ✓ VERIFIED | Verifier script: docstring == approved, README contains approved paragraph, ast (docstrings stripped) equal to 4f02628 |
| 14.7 | Ledger | ✓ VERIFIED (historical) | 45199cc changed 10 lines; superseded by the round-5 ledger (83be929, 03d9b0b), with provenance kept |
| 14.8 | Override record: only reason and accepted_at | ✓ VERIFIED | `git show 8408731`: only the reason block and accepted_at change |
| 14.9 | Paired docs | ✓ VERIFIED | README changed; no helper or notebook change, so no re-execution needed |

### Carried truths (40-01 to 40-12)

Nothing they depend on changed: the helper, all eight notebooks and every attribution file are unchanged since the
previous report. 10.11 now passes, because the exemption it narrowed is gone. SC2a, 8.14 and 11.1 stay failed, now
for the round-5 reasons.

**Score:** 163/167 truths verified, 0 present but behavior-unverified. The 167 are 146 carried (143 verified now;
SC2a, 8.14 and 11.1 failed), 12 from 40-13 and 9 from 40-14 (14.2 failed). All four failures have one root cause.

### Prohibitions

| Plan | Prohibition | Status | Evidence |
|------|-------------|--------|----------|
| 40-01..40-12 | 42 carried | ✓ VERIFIED (one superseded by D-G2) | No regression |
| 40-13 | Guard not weakened | ✓ VERIFIED | No test removed; no skip or tag; lists unchanged |
| 40-13 | No dev DB, no FOMO_DATABASE_PATH, no in-process shell, no installer | ✓ VERIFIED | `git status --short src/` clean |
| 40-13 | No notebook, README, helper, docstring, CLAUDE.md, .github or ledger edit | ✓ VERIFIED | Commit file sets |
| 40-13 | Out-of-scope items untouched | ✓ VERIFIED | No attribution, template or runbook change |
| 40-14 | Nothing written before the reply; only approved text | ✓ VERIFIED | File timestamps; approved texts match the committed ones |
| 40-14 | No code change; only reason and accepted_at in VERIFICATION; no push | ✓ VERIFIED | ast check; 8408731 diff |
| 40-14 | No dev DB access | ✓ VERIFIED | Inert checks only |

### Round-5 review findings (40-REVIEW.md): explicit classification

| Finding | Reproduced? | Classification | Reasoning |
|---------|-------------|----------------|-----------|
| CR-01 shutil through a call argument, lambda, map, keyword, match capture, class-pattern attribute | **Yes**, all six forms give [] on a copy of import_campaign_csv_demo; control (parameter default) reported; runtime binding confirmed with a stand-in | **Gap** (gap 1, gap 2) | Literal source outside R6's stated boundary and RES; contradicts the README's "a name the notebook binds to it" |
| CR-02 `case object(write_bytes=w)` | **Yes**, [] (control reported) | **Gap** (gap 1, gap 2) | Outside residual; falsifies "any write_bytes" in the reason, docstring and README |
| WR-01 `is_django_backed` reads raw source | **Yes** in isolation (is_django_backed False, []); **refuted** as a suite route: inserted into a real notebook copy it is reported, and the vacuity test requires raw Django imports in all eight | **Gap in wording only** (part of gap 2); not a route | The docstring sentence "checked by every later rule" is false of the function; the suite holds |
| WR-02 autocall `exit -1` | **Yes** (guard []; IPython source confirms the dynamic rewrite) | **Not a gap** (advisory) | No database capability; wording fix suggested |
| IN-01 README/docstring omit checks | Code read | **Not a gap** (advisory) | Understatement, not a false claim |
| IN-02 pin against a plain shell | Child-process check: sets identical today | **Not a gap** (advisory) | Housekeeping |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/tests/test_pre_executed_notebooks.py` | R0 without exemption, R6 distinctive names and notebook-wide shutil names, R5/R7 bytes, accurate docstring | ⚠️ VERIFIED with coverage gap | 86 tests green; CR-01/CR-02 (gap 1); docstring (gap 2) |
| `pyproject.toml` | `ipykernel>=7` | ✓ VERIFIED | Line 39 |
| `docs/notebooks/README.md` | Approved guard paragraph | ⚠️ VERIFIED (text) | Overstates R6 and "exactly" (gap 2) |
| `40-REVIEW-DISPOSITION.md` | Round-4 rows recorded | ✓ VERIFIED | Round-5 rows open, round-4 decisions kept with provenance |
| `40-VERIFICATION.md` override | Approved reason recorded | ✓ VERIFIED (recorded, not applicable) | Carried verbatim |
| Helper, notebooks, attribution artifacts | Unchanged | ✓ VERIFIED | git diff empty |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| isolation_problems R0 loop | `_automagic_word` | `_automagic_word(view)`, no bound set | ✓ WIRED |
| `_byte_copy_problems` | `_shutil_names`, `_mentions_shutil` | collected once over every cell and string code | ✓ WIRED (does not follow call arguments or match patterns: gap 1) |
| `_byte_copy_problems` | `_identifiers` | DISTINCTIVE_COPY_NAMES only; BYTE_WRITE_NAMES not routed through it | ⚠️ PARTIAL (gap 1, CR-02) |
| `_database_routing_problems` | `_constant_text` | `_constant_text(node)` | ✓ WIRED |
| `_ipython_line_magic_names` | pyproject dev extra | `ipykernel>=7` | ✓ WIRED |
| Gating of R2-R4/R8-R11 | Python views | `is_django_backed(notebook)` reads raw source | ⚠️ PARTIAL (WR-01; the suite is covered by the vacuity test) |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data | Status |
|----------|------|--------|-----------|--------|
| Notebook Resolved line | resolved NAME | lazy settings, checked before setup | Yes | ✓ FLOWING (unchanged) |
| Teardown second line | what setup checked | `check_database_settings` before `django.setup()` | Yes | ✓ FLOWING (unchanged) |
| Attribution page state | return state | unchanged | Yes | ✓ FLOWING (unchanged) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Guard, helper, projector | `python manage.py test --noinput --parallel auto solsys_code.tests.test_pre_executed_notebooks ...test_notebook_helper ...test_projector_demo_notebook` | Ran 121 tests, OK | ✓ PASS |
| Round-4 gap closure | `$HOME/tmp/verify40r5/r4.py` | 7/7 bindings clean alone, alias cell reported; both rebinding forms reported | ✓ PASS |
| CR-01 / CR-02 | `$HOME/tmp/verify40r5/cr.py` | 6 CR-01 forms []; CR-02 class pattern []; controls reported | ✗ FAIL (gap 1) |
| WR-01 / WR-02 | `$HOME/tmp/verify40r5/wr.py` | magic synthetic [] vs 2 problems without the magic; real-copy reported; `exit -1` [] | ✗ (wording) / advisory |
| IN-02 | `$HOME/tmp/verify40r5/zmq_child.py` (child, own IPYTHONDIR) | no difference either way | ✓ PASS |
| Real notebooks and approvals | `real.py`, `appr.py` | 8/8 []; guard code equals 4f02628; texts equal the approved ones | ✓ PASS |
| Dev DB untouched | `git status --short src/` | clean | ✓ PASS |

### Probe Execution

Step 7c: SKIPPED. No `scripts/*/tests/probe-*.sh` exists, and no plan declares one.

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|--------------|-------------|--------|----------|
| WARN-02 | 40-02, 40-06..40-14 | Guard fails a notebook whose current source routes Django to the dev DB | ✗ BLOCKED (partial) | Gap 1; override not applicable |
| WARN-03 | 40-01, 40-02, 40-05, 40-07..40-09, 40-11 | Ordinary exception under `-O` | ✓ SATISFIED | SC2b |
| WARN-04 | 40-03, 40-04 | BS 5.3 only; High-band marker; template test | ✓ SATISFIED | SC4 |
| WARN-05 | 40-01, 40-02, 40-05, 40-07..40-11, 40-13 | Audit, shared helper, no byte copy or write, README rule and lists | ✓ SATISFIED | SC1. The real notebooks are clean; the guard's R6 coverage gap is counted under WARN-02 |
| WARN-06 | 40-02 | Dip in one unit | ✓ SATISFIED | SC3 |
| WARN-07 | 40-04 | Undo keeps Dismissed open on the same page | ✓ SATISFIED | SC5 |

All six IDs appear in plan front matter and in REQUIREMENTS.md (Phase 40). No requirement is orphaned.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| solsys_code/tests/test_pre_executed_notebooks.py | ~722-792 | shutil names grow only from assignment-shaped bindings and parameter defaults | 🛑 Blocker | Gap 1 (carried-forward gap; the file changed since the prior verification) |
| solsys_code/tests/test_pre_executed_notebooks.py | ~858-863 | write_bytes matched only as Name/Attribute | 🛑 Blocker | Gap 1 |
| solsys_code/tests/test_pre_executed_notebooks.py, docs/notebooks/README.md | 39-54; 95-100 | Wording claims more than the code checks | ⚠️ Warning | Gap 2 |
| solsys_code/tests/test_pre_executed_notebooks.py | ~586, ~1093 | Django-backed gate on raw source | ℹ️ Info | WR-01; covered at suite level |

No TBD, FIXME, XXX, TODO or HACK in the lines 40-13 and 40-14 added.

### Human Verification Required

These items do not change the gaps_found status.

1. **Close the R6 routes and approve the override wording again.** Option (A): report `shutil` and match
   `write_bytes` at every identifier, then simplify the wording. Option (B): keep the code and name the
   call-argument, match-capture and class-pattern routes in the boundary, and drop "any write_bytes". Why human:
   accepting residual risk is the developer's decision.
2. **README guard paragraph and guard docstring** after the fix. Why human: wording.
3. **Snapshot notebook prose** (carried). Why human: prose accuracy.
4. **Runbook attribution-pass wording** (round-1 IN-03, kept for Phase 41). Why human: wording.

### Gaps Summary

There is one root cause again: **R6's code is narrower than what the docstring, the README and the approved override
reason say it checks.** R6 follows shutil only through assignment-shaped bindings and parameter defaults. It matches
`write_bytes` only as a name or an attribute, so a call argument, a `match` capture and a class-pattern attribute go
unreported. The cheapest complete fix is the review's: R6 reports the `shutil` module name and `write_bytes` at
every identifier, the way it already reports `copytree`. No real notebook uses either name, so all eight stay
green. The boundary text then shrinks to nothing. While the docstring is open, fix the WR-01 sentence ("checked by
every later rule"), and optionally the WR-02 one. Rewording the override reason needs the developer's approval.

These routes add no capability the developer has not already accepted through `Path.replace` and `open` in a write
mode. If the developer prefers, option (B) closes the gap with wording alone. Nothing here is deferred: Phases 41 and
42 do not cover the notebook guard.

---

_Verified: 2026-10-11T01:51:02Z_
_Verifier: Claude (gsd-verifier)_
