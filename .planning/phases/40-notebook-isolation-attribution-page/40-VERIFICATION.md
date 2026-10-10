---
phase: 40-notebook-isolation-attribution-page
verified: 2026-10-10T16:10:51Z
status: gaps_found
score: 112/115 must-haves verified (prohibitions 31/31, one superseded by D-G2)
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
covered_digest: "v3:sha256:188bdd73cf018663a0f03cb754983521e040b8cdf102a3baf06c585d5050d28c"
behavior_unverified: 0
overrides_applied: 0
overrides:
  - must_have: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale"
    reason: >-
      Rule R12, extended in 40-08 [301bd93, f1b4bb6, c45edbc], reports in every notebook any reference to
      Django's database settings and connection objects (DATABASES, DATABASE_ROUTERS, settings_dict,
      connections, ConnectionHandler, create_connection, get_connection_params, get_new_connection,
      override_settings, modify_settings, DatabaseWrapper, load_backend, sqlite3, raw SQL ATTACH), to
      starting a process (subprocess, multiprocessing, pty, pexpect, ptyprocess, psutil, Jupyter kernel
      clients, os.system, os.popen, os.exec*, os.spawn*, os.posix_spawn*, os.fork, asyncio's
      create_subprocess_*, Popen, ProcessPoolExecutor, get_ipython and IPython's shell helpers), to the
      environment (os.environ, environb, putenv, unsetenv) and to running code from a string or another file
      (exec, eval, compile, __import__, __builtins__, importlib, runpy, builtins, ctypes, cffi, call_command
      of shell, shell_plus or runscript, execute_from_command_line), wherever the name appears, so binding
      one to another name is reported too; R0 reports shell escapes, process magics and manage.py, env -i or
      env -u in an unparseable cell. All four re-review forms and both round-1 probes are reported on a copy
      of a real notebook (40-08 must_haves). What stays out of reach is deliberate obfuscation that no
      static check of a notebook's own source can enumerate, exactly as the guard docstring names it: values
      assembled or looked up at runtime (a name, module, SQL statement or file path built from pieces, or
      fetched with getattr, globals(), vars() or sys.modules), code decoded at runtime, and code moved
      outside the notebook into a module it imports or a library function that starts a process or writes a
      file under a name these lists do not contain.
    accepted_by: "Tim Lister"
    accepted_at: "2026-10-10T13:46:17Z"
re_verification:
  previous_status: gaps_found
  previous_score: 79/80 (prohibitions 21/21)
  gaps_closed:
    - "Re-review WR-01's four forms (a ConnectionHandler stored into connections[...], a child manage.py with a bare environment, a cleared environment plus a child manage.py, a shell escape running manage.py) are now reported on a copy of a real notebook (TestReReviewFormsOnARealNotebook, green in the verifier's run)"
  gaps_remaining:
    - "ROADMAP SC 2 (first half) / WARN-02: narrowed again but still open. Round-3 review CR-01 and CR-02 routes are readable, literal notebook source outside the override's accepted residual, so the override does not apply"
  regressions: []
gaps:
  - truth: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale (round-3 review CR-01, CR-02)"
    status: partial
    reason: >-
      The override does not cover the routes that remain. The verifier added each review form to a copy of the
      real import_campaign_csv_demo.ipynb, with its committed output kept, and isolation_problems() returned []
      for every one. Five of them are literal and run inside the notebook, so they are outside the residual the
      override accepts (values assembled or looked up at runtime, and code outside the notebook):
      (1) `from webbrowser import subprocess` and `from django.utils.autoreload import subprocess as sp`. R12 only
      matches PROCESS_MODULES as an import's first component or as an attribute, never as an imported or bare
      name, so `subprocess` passes here. This contradicts the claim, made in the docstring and repeated in the
      override reason, that each name is matched "wherever it appears".
      (2) `import_string('subprocess.run')` and `pkgutil.resolve_name('subprocess:run')`, looked up by a literal
      string. The residual covers lookups only with a string built from pieces.
      (3) `%time !cd ../.. && env --ignore-environment PATH=/usr/bin python -m django flush ...` in an IPython
      cell. SHELL_ESCAPE only matches `!` at the start of a line, and `--ignore-environment` is not `env -i`.
      (4) the bare IPython alias line `cp /tmp/fomo-notebook-db-*/* ../../src/` (and `%cp`, `rm -r ../../src`,
      `%alias`, `%rehashx`, `%pip`). R0 has no entry for aliases or automagic.
      Runtime check (no write to the developer database): after a real setup_scratch_django(mode='fresh'), routes
      (1), (2) and (3) each started a child Django whose settings.DATABASES['default']['NAME'] was
      /home/tlister/git/fomo_devel/src/fomo_db.sqlite3, while the kernel was on the scratch file. The teardown
      then printed its normal two lines. On dummy files, IPython's run_cell('cp scratch/* src/') replaced
      src/fomo_db.sqlite3 with the scratch copy.
    artifacts:
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "_database_rerouting_problems (lines ~516-549) checks _R12_MODULE_TABLES only against an import's first component and attributes, not against ast.Name or from-import aliases; string constants are matched exactly, so 'subprocess.run' and 'subprocess:run' pass; SHELL_ESCAPE (lines ~214-218) misses a `!` after a line magic; R0 has no IPython alias or automagic entry"
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "module docstring (lines 22-43) claims each R12 name is matched as an imported name and that R0 reports any shell escape or shell command; both claims are false for the forms above"
    missing:
      - "Match the module tables wherever the name tables are matched (ast.Name and every from-import alias); split string constants on '.' and ':' before matching; add import_string, resolve_name and locate to DYNAMIC_CODE_NAMES; add getoutputerror, get_output_error_code and process_handler to PROCESS_NAMES"
      - "Replace R0's deny-list with an allow-list. None of the eight real notebooks has an unparseable code cell (review, checked with ast.parse), so any unparseable cell is a problem unless removing a small set of harmless magic lines (%matplotlib, %config, %%time, %%capture) leaves parseable Python. That covers aliases, automagic, %time !cmd, %alias, %rehashx and %pip without listing them"
      - "Add one real-notebook test per form above (the verifier's probe script is $HOME/tmp/verify40r3/probe_guard.py); keep the eight real notebooks green and keep the %%time look-alike test passing"
      - "Correct the override reason or re-verify after the fix. The reason's 'wherever the name appears' and 'IPython's shell helpers' claims must be true before the override can be applied to the forms left in the residual"
  - truth: "40-08 must_have: the guard module docstring names exactly what stays out of reach (D-G1)"
    status: failed
    reason: "Same root cause as the gap above. The docstring's 'does not see' list names only runtime-assembled values and code outside the notebook, but the literal forms (1)-(4) above are also out of reach. It also says each R12 name is matched as an imported name, which is not true for PROCESS_MODULES and DYNAMIC_CODE_MODULES"
    artifacts:
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "lines 22-49"
    missing:
      - "Fix the guard first (above), then make the docstring and README lines 91-93 describe what the guard actually reports"
  - truth: "CLAUDE.md paired-docs rule / D-G2: every notebook prose sentence that echoes the withdrawn 'developer database never opened' claim moves with the helper's new teardown line (round-3 review WR-02)"
    status: partial
    reason: >-
      Three teardown markdown cells still end with "The developer database was never opened by this notebook run."
      They are backfill_lco_observations_demo 46602df0, campaign_lifecycle_demo 234805b6 and
      load_telescope_runs_demo 31ff3edb. The teardown output printed directly below each one is the new 'Before
      django.setup() ran, the helper checked ...' line. The other five notebooks had the same sentence replaced
      with T1. Two setup cells also still make the stronger claim that the developer database "is never opened at
      all": campaign_lifecycle_demo 7dfef438 and load_telescope_runs_demo 49ff72df. 40-09's interface table missed
      these cells, and its check matched only the literal 'opened for writing'. D-G2 says every prose sentence
      that echoes the old claim moves with it, and CLAUDE.md tells the verifier to treat a stale paired-doc
      update as a must-have gap.
    artifacts:
      - path: "docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb"
        issue: "markdown 46602df0"
      - path: "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb"
        issue: "markdown 234805b6 and 7dfef438"
      - path: "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb"
        issue: "markdown 31ff3edb and 49ff72df"
    missing:
      - "Replace the last sentence of 46602df0, 234805b6 and 31ff3edb with T1. In 7dfef438 and 49ff72df, change 'is never opened at all' to say the developer database is not needed and the helper never opens it in fresh mode. Markdown only, edited in place with nbformat; no re-execution is needed unless the gap plan above changes the helper"
      - "Widen the 40-09 style check to 'never opened' and 'never .* writes the developer'"
human_verification:
  - test: "Decide what to do about the override, given the CR-01/CR-02 reproductions"
    expected: "Either (a) fix the guard (gap 1 'missing') and keep the override for the residual that is genuinely left, with its reason corrected; or (b) widen the override on purpose to cover readable literal routes. Option (b) would change what SC 2 means, so the wording has to be the developer's"
    why_human: "Accepting residual risk is a developer decision. The verifier may not widen an override"
  - test: "Read docs/notebooks/README.md's helper and guard section (lines ~60-100) and the guard module docstring after the gap fix"
    expected: "Clear and honest about what the guard reports and what it cannot see"
    why_human: "Documentation wording; the tests check exact sentences, not their meaning (40-08 D4 human_judgment: true)"
  - test: "Read both snapshot notebooks' prose against this regeneration's output: the reconcile cutover and summary cells 8b703ea8, ce1fd748, e1edbaa7, d2adacf8, 3616c012, the NF-14 labels, and the projector's requirement table in 35debc54"
    expected: "Each claim matches the output beside it (the verifier's script found the labels and citations correct)"
    why_human: "Prose accuracy is a reading judgment (40-09 D2/D3 human_judgment: true)"
  - test: "Read docs/runbooks/telescope_runs_calendar.rst lines ~935-990 (attribution pass)"
    expected: "'each table's page and sort' does not read as covering the worklists' event_page/record_page (round-1 IN-03, kept open for Phase 41 triage by developer decision)"
    why_human: "Documentation wording, carried forward unchanged"
---

# Phase 40: Notebook Isolation & Attribution Page Verification Report

**Phase Goal:** Every pre-executed demo notebook builds and uses its own scratch database, guarded by a check that catches one that does not, and the staff attribution page renders and pages correctly under Bootstrap 5.3, closing the remaining six Phase 37.1 review warnings (WR-13 to WR-18).
**Verified:** 2026-10-10T16:10:51Z (HEAD 9be2c6f; last source change 0020360; no source change since 7e1fdad)
**Status:** gaps_found
**Re-verification:** Yes, after gap-closure plans 40-08 and 40-09

## Summary

Gap plans 40-08 and 40-09 did everything their must_haves list. The four forms from the previous round are now
reported, the helper's fail-closed step is exception-safe, all eight notebooks were re-executed on the new
helper, and the IN-02 prose is fixed. The verifier's own runs confirm each of these.

The phase goal is still not met in one place: the clause "guarded by a check that catches one that does not".
The developer's override accepts only two kinds of route: values assembled or looked up at runtime, and code
outside the notebook. The verifier reproduced five readable, literal routes that are neither, and the guard
reports none of them. Three of them were run after a real fresh setup. Each started a child Django resolved to
`src/fomo_db.sqlite3` while the kernel was on the scratch file, and the teardown then printed its normal lines.
A fourth, the bare IPython `cp` alias, overwrote a dummy `src/fomo_db.sqlite3`. The override's reason also relies
on a claim the code does not keep ("wherever the name appears, so binding one to another name is reported too").
`from webbrowser import subprocess` is a binding to the same name that is not reported. So the override cannot be
applied, and SC 2's first half stays a gap.

A second, smaller gap is a paired-docs miss under D-G2. Three notebooks' teardown prose still says the developer
database "was never opened", directly above output that now says what the helper checked.

The attribution half (WARN-04, WARN-07) is unchanged since 0ad6991 and still passes: 111 tests OK in the
verifier's run.

## Goal Achievement

### Observable Truths: ROADMAP success criteria (the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | No notebook byte-copies or writes the dev DB; snapshot via read-only online backup, others fresh; three byte-copiers on the helper; all re-executed; README rule and mode lists | ✓ VERIFIED | Verifier audit of all eight at HEAD: counts 1..N, 0 error outputs, exact `Django ready:` line, Resolved line in `/tmp/fomo-notebook-db-*` with the right mode text (6 fresh, 2 snapshot), teardown cell last with the new second line, no `/home/` in setup or teardown output. `test_every_real_pre_executed_notebook_is_isolated` green. (Prose miss: gap 3) |
| SC2a | The guard fails a notebook whose current source routes Django to the dev DB, even with stale output | ✗ FAILED (partial). Override checked and **not applicable** | The re-review forms (a)-(d) are now caught. Five literal forms (CR-01 (a)/(e), CR-02) return `[]`, and three of them were shown at runtime to start Django on the dev DB path. See "Override decision" |
| SC2b | Every notebook's DB-path check raises an ordinary exception that fires under `python -O` | ✓ VERIFIED | `check_database_settings` / `check_resolved_database` raise RuntimeError; `TestHelperFailClosedIsExceptionSafe` covers plain and `-O` (green, 98-test run) |
| SC3 | telescope_runs_demo reports the dip in one unit | ✓ VERIFIED | Dip pin test and `test_no_committed_output_reports_arcmin_deg` green |
| SC4 | High-band marker, template test, BS 5.3 names only | ✓ VERIFIED | No attribution file changed since 0ad6991 (`git diff --stat`); 111 non-functional tests OK in the verifier's run |
| SC5 | Undo on a Dismissed row keeps the section open on the same page | ✓ VERIFIED | Unchanged files; the orchestrator's full gate at 7e1fdad ran the browser tests (40 functional OK) |

### Override decision (ROADMAP SC 2 first half / WARN-02)

The developer's override (accepted 2026-10-10T13:46:17Z) is carried forward verbatim in the frontmatter. The
verifier checked each review route against its accepted residual: (1) values assembled or looked up at runtime,
and (2) code outside the notebook.

| Route (review id) | Guard result (verifier probe) | Reaches the developer DB? | Inside the residual? | Classification |
|-------------------|-------------------------------|---------------------------|----------------------|----------------|
| `from webbrowser import subprocess` / `from django.utils.autoreload import subprocess as sp` + `run([... 'manage.py', 'flush'], env={'PATH': ...})` (CR-01 a) | `[]` | **Yes**: the child's Django NAME was `.../src/fomo_db.sqlite3` (runtime probe) | **No.** Literal, in the notebook, and `subprocess` is a listed name; the override reason says binding to another name is reported | **Gap** |
| `import_string('subprocess.run')(...)`, `pkgutil.resolve_name('subprocess:run')` (CR-01 e) | `[]` | **Yes** (runtime probe, import_string) | **No.** Looked up with a literal string; the residual covers lookups only with a string built from pieces | **Gap** |
| `%time !cd ../.. && env --ignore-environment ... python -m django flush ...` (CR-02) | `[]` | **Yes** (runtime probe through IPython `run_cell`) | **No.** Literal shell escape in the notebook; the docstring says R0 reports shell escapes | **Gap** |
| bare `cp /tmp/fomo-notebook-db-*/* ../../src/`, `%cp`, `rm -r ../../src`, `%alias`, `%rehashx`, `%pip` (CR-02) | `[]` | `cp` overwrote a dummy `src/fomo_db.sqlite3` with the scratch copy (IPython `run_cell`) | **No.** Literal and in the notebook | **Gap** (a byte copy over the dev DB, which is exactly what the goal's guard exists to catch) |
| IPython `getoutputerror` / `get_output_error_code` / `process_handler` (CR-01) | `[]` | Could start a child | **Yes, by its words**: a library function that starts a process under a name the lists do not contain. The override reason's "IPython's shell helpers" overstates it | Warning: fix with gap 1 (cheap list entries) |
| `connection._connections.settings[...]`, `connection.connection = connection.Database.connect(...)`, `ATTACH (%s)` (CR-01 b-d) | `[]` for another file | Only with the dev DB's path. A literal `fomo_db.sqlite3` is caught by R7 (verifier control probe); otherwise the path must be built from pieces | **Yes** with respect to the developer database (residual 1) | Warning: they re-route Django to *some* other file, so harden with gap 1 |
| `shutil.copytree(scratch_db_dir, repo_root_path / 'src', ...)`, `write_bytes` loop (WR-01) | `[]` | Overwrites the dev DB (same base name) | **Yes**: the destination is joined path parts (residual 1), and copytree/write_bytes are library functions that write files under unlisted names (residual 2) | Warning, not a gap. The eight real notebooks contain none of these (SC1). Add `copytree`/`copyfileobj`/`move` to R6 in the same fix |

Routes the override does not cover remain, so the override is not applied (`overrides_applied: 0`), and SC2a
counts as failed.

### Observable Truths: 40-08 must_haves (guard capabilities, exception-safe helper)

| # | Truth (abridged) | Status | Evidence |
|---|------------------|--------|----------|
| 8.1-8.4 | Re-review forms (a)-(d) reported on a real-notebook copy | ✓ VERIFIED | `TestReReviewFormsOnARealNotebook` green; verifier control probes (`import subprocess`, `!ls`, `!... manage.py`) caught |
| 8.5 | Process capability: listed modules/names reported as listed | ✓ VERIFIED (literal) | `test_starting_a_process_in_any_spelling_is_a_problem` green. Modules are matched only as an import's first component or an attribute (see gap 1) |
| 8.6 | IPython in parseable and unparseable cells | ✓ VERIFIED (literal) | Tests green. `%time !cmd` and aliases are not covered (gap 1) |
| 8.7 | Stores into / references to `connections`, ConnectionHandler, etc. | ✓ VERIFIED | Test green |
| 8.8 | Environment changes reported | ✓ VERIFIED | Test green |
| 8.9 | R0 text manage.py / env -i / env -u; real setup cells stay clean | ✓ VERIFIED | Test green; real setup cells name `manage.py` in parseable code and pass |
| 8.10 | Code from a string or file | ✓ VERIFIED (literal) | Test green; `import_string` / `resolve_name` are not listed (gap 1) |
| 8.11 | Raw SQL ATTACH | ✓ VERIFIED (literal) | Test green; `ATTACH (` passes (warning) |
| 8.12 | Look-alikes allowed | ✓ VERIFIED | `test_allowed_look_alikes_are_not_a_problem` green |
| 8.13 | Name lists disjoint, R12_ALL_NAMES feeds R0 | ✓ VERIFIED | Test green |
| 8.14 | Docstring names exactly what stays out of reach; matched "wherever it appears" | ✗ FAILED | Gap 2: the docstring claims more than the code does |
| 8.15 | All eight real notebooks pass; no test removed, skipped or tagged; EXPECTED_NOTEBOOKS unchanged | ✓ VERIFIED | `git diff 488a7bb HEAD -- solsys_code/tests/` has no removed `def test_` and no added skip/tag/expectedFailure |
| 8.16 | No `default` alias gives RuntimeError (plain and `-O`); every alias dummy; nothing created | ✓ VERIFIED | `TestHelperFailClosedIsExceptionSafe` green |
| 8.17 | Failing close: RuntimeError kept, all aliases dummy, close failure logged | ✓ VERIFIED | Same class, green |
| 8.18 | Failing disable step never replaces the original error | ✓ VERIFIED | Same class, green |
| 8.19 | Ordering: switch, best-effort close, drop wrappers; handler try/log, finally rmtree and restore, bare raise | ✓ VERIFIED | fomo_nb_utils.py:203-221 and :325-337 read by the verifier |
| 8.20 | New teardown line exactly; old claim gone from the touched files | ✓ VERIFIED | Runtime probe printed the exact line; test green |
| 8.21 | README S1-S3, teardown sentence, guard capabilities; D-04 and lists unchanged | ✓ VERIFIED | Whitespace-normalized match True; `TestReadmeStatesTheIsolationRule` green (wording: human item 2) |
| 8.22 | Ledger WR-01, WR-02 fixed at 40-08 | ✓ VERIFIED (historical) | 7ba970e exists; the round-3 gate later reused the ids and recorded the provenance (9be2c6f) |
| 8.23 | Gate: ruff clean, three full-hook code commits, nothing pushed, no notebook/VERIFICATION change | ✓ VERIFIED | `git show --stat` of 301bd93, f1b4bb6, c45edbc, 7ba970e: only the listed files |

### Observable Truths: 40-09 must_haves (paired notebooks, IN-02, ledger, override)

| # | Truth (abridged) | Status | Evidence |
|---|------------------|--------|----------|
| 9.1 | All eight re-executed on the 40-08 helper with the evidence and teardown lines, no /home/ | ✓ VERIFIED | Verifier audit script (all eight rows true) |
| 9.2 | No old "opened for writing" claim; T1 in the five listed teardown cells, T2 in snapshot, 6a9bd576 clause | ✓ VERIFIED (literal) | T1 found in 9c899021, 317a98bb, 98bba968, c8f6eeb3, 2e2d138e; "opened for writing" in no source or output. The three unlisted teardown cells: gap 3 |
| 9.3 | S1-S3 in every setup markdown | ✓ VERIFIED | 8/8 normalized match |
| 9.4 | reconcile NF-14 labels data-independent; no hard-coded pk | ✓ VERIFIED | Three CommandError lines carry the label beside `duplicate_identity=3`; no `pk=334` |
| 9.5 | Projector table cites real headings only | ✓ VERIFIED | Neither missing heading is cited; "The whole real corpus, not a sample" present |
| 9.6 | Code cells changed since 488a7bb are exactly the three | ✓ VERIFIED | Cell-by-id diff: 35debc54, 2c723427, 48b9f602 only |
| 9.7 | Snapshot notebooks: ids kept, projector test passes, prose recorded | ✓ VERIFIED | `test_projector_demo_notebook` in the 98 OK; prose reading is human item 3 |
| 9.8 | Guard, helper, projector tests pass on the final tree | ✓ VERIFIED | Verifier: `Ran 98 tests ... OK` |
| 9.9 | Home-path inventory | ✓ VERIFIED | No /home/ in setup or teardown output; the spiceypy line is listed in the SUMMARY |
| 9.10 | Ledger IN-01, IN-02 fixed | ✓ VERIFIED (historical) | 93bb514 exists; the later review gate re-recorded the rows (9be2c6f) |
| 9.11 | Override recorded last, nothing else changed | ✓ VERIFIED | `git show b6bfbca --numstat`: 23 added, 0 removed, only 40-VERIFICATION.md. Recording ≠ applicable: see "Override decision" |
| 9.12 | Gate | ✓ VERIFIED | The 40-09 commits touch only the notebooks, the ledger and this report |

### Derived must-have (CLAUDE.md paired-docs rule)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| PD.1 | Notebook prose that echoes the withdrawn claim moves with D-G2 | ✗ FAILED (partial) | Gap 3: 46602df0, 234805b6, 31ff3edb (teardown) and 7dfef438, 49ff72df (setup) |

### Carried truths (40-01 to 40-07, previous report)

The previous report verified 79 truths. Row 6.6, the 40-06 docstring wording, is superseded by 8.14 and not
counted twice. SC2a appears above. The other 78 passed a regression check: the helper and guard rows were re-run
through the 98-test run, the notebook rows through the audit above, and the attribution rows through the 111-test
run. None regressed.

**Score:** 112/115 truths verified, 0 present but behavior-unverified. The 115 are the 79 carried rows (the
previous 80 less 6.6, of which SC2a failed and 78 verified, including SC1, SC2b, SC3, SC4 and SC5), 23 from 40-08
(22 verified, 8.14 failed), 12 from 40-09 (all verified) and PD.1 (failed).
Every behavior-dependent truth (refusal state, close failure, disable failure, environment restore) has a named
child-process test, and those tests ran green in the verifier's run.

### Prohibitions

| Plan | Prohibition | Tier | Status | Evidence |
|------|-------------|------|--------|----------|
| 40-01..40-07 | 21 carried | mixed | ✓ VERIFIED (one superseded) | The 40-05 "teardown lines unchanged" prohibition is superseded by developer decision D-G2; the call itself and the first line are unchanged (fomo_nb_utils.py:258, :345, :387) |
| 40-08 | Guard not weakened | test | ✓ VERIFIED | No removed test, no skip/tag, EXPECTED_NOTEBOOKS unchanged |
| 40-08 | No notebook or VERIFICATION edit | test | ✓ VERIFIED | 40-08 commit file sets |
| 40-08 | Call signature, first teardown line, TEARDOWN_LINE_PREFIX unchanged | test | ✓ VERIFIED | :61, :258, :345, :387 |
| 40-08 | Never open, copy or route to the dev DB in tests | test | ✓ VERIFIED | `git status --short src/` clean after every verifier probe |
| 40-08 | No IN-03 tests; WR-03, IN-04, IN-07, runbook untouched | test | ✓ VERIFIED | No campaign_views or runbook file in the 40-08/40-09 commits |
| 40-09 | No write to the dev DB while re-executing | test | ✓ VERIFIED | Every Resolved line is in `/tmp/fomo-notebook-db-*`; snapshots read with `mode=ro` |
| 40-09 | No `--allow-errors`, counts 1..N | test | ✓ VERIFIED | Audit: 0 error outputs, counts contiguous |
| 40-09 | Only the three code cells changed; helper, guard, README and runbook untouched | test | ✓ VERIFIED | Cell diff; 40-09 commit file sets |
| 40-09 | No credential-shaped text | test | ✓ VERIFIED | Test green |
| 40-09 | Ledger rows IN-03/WR-03/IN-04/IN-07 kept; VERIFICATION only gains overrides | test | ✓ VERIFIED | b6bfbca numstat 23/0 |

### Round-3 review findings (40-REVIEW.md): explicit classification

| Finding | Reproduced by verifier? | Classification | Reasoning |
|---------|-------------------------|----------------|-----------|
| CR-01 (a) re-exported `subprocess`, (e) `import_string` / `resolve_name` by literal string | **Yes.** Guard `[]`; runtime child resolved the dev DB path | **Gap** (gap 1) | Literal and inside the notebook, so outside the override residual; contradicts the "wherever the name appears" claim |
| CR-01 (b)-(d) `connection` proxy, `Database.connect`, `ATTACH (` | **Yes.** Guard `[]` for another file; the dev path as a literal is caught by R7 (control) | **Warning** (fix with gap 1) | Reaching the *developer* DB needs its name, which is caught, or a path built from pieces (residual 1) |
| CR-01 IPython `getoutputerror` etc. | **Yes.** Guard `[]` | **Warning** (fix with gap 1) | Within residual (2) by its words; the override reason overstates "IPython's shell helpers" |
| CR-02 aliases, automagic, `%time !`, `%alias`, `%rehashx`, `%pip` | **Yes.** Guard `[]` x8 forms; `%time !` child resolved the dev DB path; `cp` alias overwrote a dummy dev DB | **Gap** (gap 1) | Literal and inside the notebook; the docstring says R0 reports shell escapes and shell commands |
| WR-01 `copytree` / `write_bytes` | **Yes.** Guard `[]` | **Not a gap** (warning) | Destination built from joined path parts and unlisted writing functions, both within the accepted residual; no real notebook does it. Cheap R6 additions recommended |
| WR-02 "never opened" prose in three notebooks | **Yes** (five cells found) | **Gap** (gap 3, CLAUDE.md paired-docs rule) | D-G2 moved every sentence that echoes the claim; five notebooks were updated and three were missed; the prose now sits directly above output that says something different |
| IN-01 `_disable_django_databases` docstring overstates the switch | By reading (:190-195; cached wrappers are dropped at :217-221) | **Not a gap** (info) | The behavior is fail-closed (the cached wrappers are dropped in every case, and the tests are green); only the wording is wrong. Optional fix with gap 1 |
| IN-03 (round 2), WR-03, IN-04, IN-07 remainder, round-1 IN-03 | Not re-run | **Not counted**: deferred to Phase 41 by the developer | Out of scope by recorded decision |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/tests/test_pre_executed_notebooks.py` | R12 capabilities, R0 escapes, real-notebook replay, docstring | ⚠️ VERIFIED with coverage gap | 63 tests green; module and alias matching gap, R0 deny-list (gap 1) |
| `docs/notebooks/fomo_nb_utils.py` | exception-safe fail-closed step, new teardown line | ✓ VERIFIED | `logger = logging.getLogger(__name__)` :11; ordering read; IN-01 wording only |
| `solsys_code/tests/test_notebook_helper.py` | `TestHelperFailClosedIsExceptionSafe` | ✓ VERIFIED | 30 tests green |
| `docs/notebooks/README.md` | S1-S3, teardown sentence, capabilities | ✓ VERIFIED | Lines 91-93 overstate the guard (gap 2) |
| 8 notebooks | re-executed on the 40-08 helper | ✓ VERIFIED | Audit; prose miss in three (gap 3) |
| 40-VERIFICATION.md overrides | developer override recorded | ✓ VERIFIED (recorded, not applicable) | Carried forward verbatim |
| Attribution artifacts | unchanged | ✓ VERIFIED | 111 OK |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| `isolation_problems` | `_database_rerouting_problems` | `problems.extend(...)` before the Django-backed return | ✓ WIRED |
| R0 loop | SHELL_ESCAPE, UNPARSEABLE_SUSPECT_NAMES, UNPARSEABLE_SUSPECT_TEXT | `SHELL_ESCAPE.search` etc. | ✓ WIRED (coverage gap 1) |
| `TestReReviewFormsOnARealNotebook` | import_campaign_csv_demo.ipynb | deep copy | ✓ WIRED |
| setup failure handler | `_disable_django_databases` | try / `logger.warning` / finally rmtree and restore / bare `raise` (:325-337) | ✓ WIRED |
| notebook teardown output | `teardown_scratch_django` | new second line in 8/8 | ✓ WIRED |
| setup markdown | README | S1-S3 in 8/8 | ✓ WIRED |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data | Status |
|----------|------|--------|-----------|--------|
| Notebook Resolved line | resolved NAME | lazy settings, checked before setup | Yes | ✓ FLOWING |
| Teardown second line | what the setup checked | fixed text, but true of `check_database_settings` before `django.setup()` | Yes (describes a real check) | ✓ FLOWING (unlike the old line, it no longer claims more than was checked) |
| Attribution page state | return state | unchanged | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Helper, guard, projector modules | `python manage.py test --noinput ...test_notebook_helper ...test_pre_executed_notebooks ...test_projector_demo_notebook --parallel auto` | Ran 98 tests, OK | ✓ PASS |
| Attribution regression | `... test_attribution_template test_campaign_attribution_views test_bootstrap5_rendering --exclude-tag functional --parallel auto` | Ran 111 tests, OK | ✓ PASS |
| Guard vs round-3 forms | `$HOME/tmp/verify40r3/probe_guard.py` (append or insert into a copy of import_campaign_csv_demo) | 19 forms `[]`; controls (`import subprocess`, `!ls`, `!... manage.py`, literal `fomo_db.sqlite3`) caught | ✗ FAIL (gap 1) |
| Runtime: do guard-clean routes reach the dev DB? | `$HOME/tmp/verify40r3/runtime_probe.py` (real fresh setup, child prints settings NAME only) | 3/3 children: `/home/tlister/git/fomo_devel/src/fomo_db.sqlite3`; kernel on scratch; teardown printed both normal lines | ✗ FAIL (gap 1 evidence) |
| CR-02 `cp` alias | IPython `run_cell('cp scratch/* src/')` on dummy files | dummy `src/fomo_db.sqlite3` now `SCRATCH` | ✗ FAIL (gap 1 evidence) |
| Notebook audit | `$HOME/tmp/verify40r3/audit.py`, `audit2.py` | all eight rows true; three code cells changed; T1 in five | ✓ PASS |
| Developer DB untouched | `git status --short src/` after every probe | clean | ✓ PASS |

The child processes only read `settings.DATABASES`; none opened a database. All probes ran with
`TMPDIR=$HOME/tmp`, and none set `FOMO_DATABASE_PATH` in the shell.

### Probe Execution

Step 7c: SKIPPED. No `scripts/*/tests/probe-*.sh` exists, and no plan or summary declares one.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| WARN-02 | 40-02, 40-06, 40-07, 40-08, 40-09 | Guard fails a notebook whose current source routes Django to the dev DB | ✗ BLOCKED (partial) | Gap 1; override not applicable |
| WARN-03 | 40-01, 40-02, 40-05, 40-07, 40-08, 40-09 | Ordinary exception under `-O` | ✓ SATISFIED | SC2b; exception-safe fail-closed step |
| WARN-04 | 40-03, 40-04 | BS 5.3 only; High-band marker; template test | ✓ SATISFIED | SC4 |
| WARN-05 | 40-01, 40-02, 40-05, 40-07, 40-08, 40-09 | Audit, shared helper, no byte copy or write, README rule and lists | ✓ SATISFIED (paired-prose miss: gap 3) | SC1 holds for the real notebooks |
| WARN-06 | 40-02 | Dip in one unit | ✓ SATISFIED | SC3 |
| WARN-07 | 40-04 | Undo keeps Dismissed open on the same page | ✓ SATISFIED | SC5 |

All six IDs appear in plan frontmatter and in REQUIREMENTS.md (rows 102-107, all mapped to Phase 40). No
requirement is orphaned.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| solsys_code/tests/test_pre_executed_notebooks.py | ~516-549 | Module tables not matched as names or from-import aliases; exact-match string constants | 🛑 Blocker | Gap 1 (file modified since the previous verification; reproduced) |
| solsys_code/tests/test_pre_executed_notebooks.py | ~214-231 | R0 deny-list misses aliases, automagic, `%time !` | 🛑 Blocker | Gap 1 |
| solsys_code/tests/test_pre_executed_notebooks.py | 22-49 | Docstring claims more than the code does | ⚠️ Warning | Gap 2 |
| solsys_code/tests/test_pre_executed_notebooks.py | ~185 | R6 lists only copy/copy2/copyfile | ⚠️ Warning | WR-01, within the residual |
| docs/notebooks/fomo_nb_utils.py | 190-195 | Docstring overstates the in-place switch | ℹ️ Info | IN-01 |
| 3 notebooks | 46602df0, 234805b6, 31ff3edb, 7dfef438, 49ff72df | Stale "never opened" prose | ⚠️ Warning | Gap 3 |

No TBD, FIXME, XXX, TODO or HACK markers in lines added since 488a7bb.

### Human Verification Required

These do not change the gaps_found status.

1. **Override decision.** Fix the guard and correct the override reason (recommended), or widen the override
   on purpose. Why human: accepting risk is the developer's decision.
2. **README guard section and guard docstring**, after the fix. Why human: wording.
3. **Snapshot notebook prose** against this regeneration's output (reconcile cutover and summary cells, NF-14
   labels, projector requirement table). Why human: prose accuracy.
4. **Runbook attribution-pass wording** (round-1 IN-03, kept for Phase 41). Why human: wording.

### Gaps Summary

There are three gaps, from two root causes.

**Root cause A: the guard still misses readable, literal routes (gaps 1 and 2; BLOCKER for SC 2 / WARN-02).**
This is the third round in which a deny-list guard has let readable source through. This time, though, the
misses are not new categories. `subprocess` is already on the list, but it is not matched when it is imported
through another module. A shell escape is already a stated R0 target, but it is not matched after a line magic.
IPython aliases are shell commands, which R0 says it reports. Two structural fixes, rather than more list entries,
would close these routes and make the override's stated basis true:

1. Match every R12 table wherever a name appears (ast.Name, from-import aliases, and string constants split
   on `.` and `:`).
2. Replace R0's deny-list with an allow-list. No real notebook has an unparseable cell, so any unparseable cell
   is a problem unless removing a few harmless magic lines leaves parseable Python.

After that, the override can be applied with a corrected reason to what genuinely remains: the WR-01 copytree
form, the CR-01 (b)-(d) other-file re-routes, and runtime-assembled values.

**Root cause B: a paired-docs miss under D-G2 (gap 3).** Five markdown cells in three notebooks keep the
withdrawn "never opened" wording. This is a markdown-only edit with no re-execution needed, unless the gap 1
plan changes the helper. Note that a gap 1 plan changes only the guard module (a test file), which has no paired
notebook. Gap 3 can therefore go in the same plan as a small markdown task.

Nothing here is deferred. Phase 41 triages todos and Phase 42 re-verifies v2.4; neither covers the notebook guard.

---

_Verified: 2026-10-10T16:10:51Z_
_Verifier: Claude (gsd-verifier)_
