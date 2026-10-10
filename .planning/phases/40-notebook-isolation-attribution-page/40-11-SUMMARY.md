---
phase: 40-notebook-isolation-attribution-page
plan: 11
subsystem: docs
tags: [notebooks, guard, docstring, readme, gap-closure, scratch-database]
gap_closure: true
gap_ids: [CR-01, WR-02, IN-01]
status: complete

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-10 guard behaviour (R0 allow-list and automagic rule, R6 copy and write functions, R12 identifier, string and string-code matching) and the round-3 review findings CR-01, WR-02 and IN-01"
provides:
  - "Guard module docstring whose 'What the guard checks' paragraph describes the code 40-10 committed and whose 'What the guard does not see' paragraph is exactly the residual RES"
  - "docs/notebooks/README.md guard paragraph (README_GUARD) and refusal sentence (README_IN01)"
  - "_disable_django_databases docstring that says what the in-place switch does on its own"
  - "Six notebook markdown sentences that no longer echo the withdrawn 'never opened' claim"
  - "Review ledger rows for CR-01, WR-02 and IN-01 fixed (open 3 of 12)"
affects: [40-12, verify-work]

tech-stack:
  added: []
  patterns:
    - "Notebook prose edited in place with nbformat by cell id, using a whitespace-tolerant pattern that must match once; no re-execution, outputs byte-identical"

key-files:
  created:
    - .planning/phases/40-notebook-isolation-attribution-page/40-11-SUMMARY.md
  modified:
    - solsys_code/tests/test_pre_executed_notebooks.py
    - docs/notebooks/README.md
    - docs/notebooks/fomo_nb_utils.py
    - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
    - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
    - docs/notebooks/pre_executed/telescope_runs_demo.ipynb
    - .planning/phases/40-notebook-isolation-attribution-page/40-REVIEW-DISPOSITION.md

key-decisions:
  - "The guard docstring describes string code with 40-10's actual test (a string that parses and holds a statement or a call), not the plan's draft 'a string that is itself Python code' alone"
  - "The helper docstring change is a docstring only, so no notebook is re-executed for IN-01 (D-G6 c)"

requirements-completed: [WARN-02, WARN-03, WARN-05]

duration: 9 min
completed: 2026-10-10

actuals:
  tokens: 6300    # chars/4 over the plan's three commits' diff (25235 chars incl. diff headers)
  tasks: 2
  commits: 3
plan_head_before: 506362b6e7d24406ef585394ad0d0791b2831c95
plan_head_after: 7a432513c4a079b357bd2c6b0fdc692751071de3

coverage:
  - id: D1
    description: "Guard module docstring states what the guard reports (R0, R2-R11, R6's copy and write functions, R12's five capabilities and the positions where a name is matched) and ends with RES; guard code identical to 40-10's commit"
    requirement: WARN-02
    verification:
      - kind: command
        ref: "Task 1 second verify command (docstring, README and helper check)"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py (117 tests with helper and projector modules)"
        status: pass
    human_judgment: true
    rationale: "Verification human item 2: the developer reads the docstring and README wording at /gsd-verify-work"
  - id: D2
    description: "README carries README_GUARD and README_IN01; D-04, S1-S3 and both mode lists unchanged"
    requirement: WARN-02
    verification:
      - kind: command
        ref: "Task 1 second verify command"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestReadmeStatesTheIsolationRule"
        status: pass
    human_judgment: true
    rationale: "README wording is a human_judgment item at /gsd-verify-work (D-G5 4)"
  - id: D3
    description: "_disable_django_databases docstring says the switch makes wrappers built afterwards dummy ones, that a cached wrapper with an open connection keeps it, and that the wrappers are dropped whether or not the close worked; helper code identical to 1930f31"
    requirement: WARN-03
    verification:
      - kind: command
        ref: "Task 1 second verify command (ast.dump of the helper with the docstring removed equals 1930f31)"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py"
        status: pass
    human_judgment: false
  - id: D4
    description: "Six notebook sentences replaced; every other cell, code cell and output identical to 1930f31; widened never-opened check clean in all eight notebooks; T1 in all eight teardown cells"
    requirement: WARN-05
    verification:
      - kind: command
        ref: "Task 2 second verify command (notebook check)"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestPreExecutedNotebooksUseTheHelper.test_every_real_pre_executed_notebook_is_isolated"
        status: pass
    human_judgment: false
  - id: D5
    description: "Ledger: CR-01, WR-02 and IN-01 fixed with real commits touching their files, open 3, 7 lines since the 40-10 ledger commit"
    verification:
      - kind: command
        ref: "Task 2 third verify command (with the since-1930f31 count corrected to 11, see Deviations)"
        status: pass
    human_judgment: false
---

# Phase 40 Plan 11: Guard Docstring, README, Helper Docstring and Notebook Prose Match the Guard Summary

**The guard's module docstring and the README now say what 40-10's guard reports, in which positions it matches a name and which magic lines it allows, and they name exactly what a static check cannot see (RES). The helper docstring no longer credits the in-place dummy switch with protection that only dropping the cached wrappers gives. Six notebook sentences no longer say the developer database was "never opened".**

## Performance

- **Duration:** 9 min (2026-10-10T18:00:59Z to 2026-10-10T18:10:23Z)
- **Tasks:** 2 of 2
- **Files modified:** 8 (guard module docstring, README, helper docstring, four notebooks, ledger)
- **Tests:** guard, helper and projector modules `Ran 117 tests` / `OK` after each task
- **Full-hook commits:** 7af64d0 and 17168ff both passed the django-test hook; only the ledger commit used SKIP=django-test

## Commits

| Commit | Task | Closes |
|--------|------|--------|
| 7af64d0 | Task 1: guard module docstring, README guard paragraph and refusal sentence, helper docstring (3 files, full hook) | CR-01 (docstring and README part), IN-01, D-G5 (4), D-G6 (c) |
| 17168ff | Task 2: six markdown sentences in four notebooks (4 files, full hook) | WR-02, verification gap 3, D-G2 |
| 7a43251 | Task 2 ledger (SKIP=django-test, ledger only) | records CR-01, WR-02 and IN-01 fixed, open 6 to 3 |

## "What the guard checks" as committed (40-12 quotes this)

What the guard checks: R0, every code cell must be Python that IPython runs unchanged. Once ``%matplotlib inline``
lines and a first-line ``%%time`` or ``%%capture`` (with only its ``--no-stderr``, ``--no-stdout`` and
``--no-display`` flags and an output name) are removed, the cell must parse, and IPython's own static input
transformer (``TransformerManager``) must leave its meaning unchanged, so an alias, an escape on any line (also
after a line magic, as in ``%time !ls``), any other magic and a prompt prefix (``In [1]:``) are reported.
``%config`` is not harmless, because ``%config AliasManager.user_aliases = [...]`` defines a shell alias that a
later one-line cell runs; ``%matplotlib`` with any other argument is not harmless either, because a backend such as
``module://name`` imports a module by name. A one-line cell whose first word, read with IPython's
``split_user_input``, is an IPython line magic or alias (the set comes from IPython itself, including the kernel's
and storemagic's magics) is reported, because IPython's automagic would run it, unless an earlier cell binds that
name, the word is a builtin or a keyword, or it is followed by ``=`` or ``,``. A cell that does not parse as
written and mentions Django, a database, a process, a shell command or the environment (an IPython shell escape, a
process cell magic such as ``%%bash``, a process or environment line magic such as ``%system`` or ``%env``,
``manage.py``, ``env -i``, ``env -u`` or any name R12 lists) is still reported by its own message. The code under a
harmless magic is checked by every later rule. R2-R4 and R8-R11: the helper call comes first and once, no own
Django setup, one fresh run with evidence and teardown output, and no ``assert`` guarding a write. R5-R7: no code
naming the database variable or the developer database file, and no byte copy or byte write; R6 reports shutil's
``copy``, ``copy2``, ``copyfile``, ``copytree``, ``copyfileobj`` and ``move`` (called, referenced or imported,
through ``shutil`` or an alias of it), any ``write_bytes``, and any of these names looked up through a string
(``getattr(shutil, 'copytree')``), in a cell's code and in code inside its strings. R12 reports, in every notebook,
any use of five capabilities that can route around the scratch database (WR-01, D-G1): Django's database settings
and connection objects (DATABASES, DATABASE_ROUTERS, settings_dict, override_settings, modify_settings, the
helper's internals, DatabaseWrapper, load_backend, ``connections``, ``connection`` and ``_connections``,
ConnectionHandler, create_connection, get_connection_params, get_new_connection, ``Database``,
``connections.settings`` and ``connections.databases``, and any ``sqlite3`` import); attaching another database
through raw SQL (a string holding ATTACH followed by DATABASE, a quote or a placeholder, or by an opening
parenthesis, ``ATTACH (``); starting a process (subprocess, multiprocessing, pty, pexpect, ptyprocess, psutil,
Jupyter's kernel clients jupyter_client, nbclient and nbconvert, ``system``, ``popen``, ``fork``, ``forkpty``,
``startfile``, ``posix_spawn``, ``posix_spawnp``, every ``exec*`` and ``spawn*`` function of ``os``, Popen,
ProcessPoolExecutor, asyncio's ``create_subprocess_exec`` and ``create_subprocess_shell``, ``getoutput``,
``getstatusoutput``, IPython's ``get_ipython``, ``system_raw``, ``run_line_magic`` and ``run_cell_magic``, and
the IPython.utils helpers ``getoutputerror``, ``get_output_error_code`` and ``process_handler``); the environment
(``os.environ``, ``os.environb``, ``os.putenv``, ``os.unsetenv``; ``os.getenv`` reads are allowed); and running
code from a string or another file (exec, eval, compile, ``__import__``, ``__builtins__``, importlib, runpy,
builtins, ctypes, cffi, execute_from_command_line, ManagementUtility, the lookup functions ``import_string``,
``resolve_name`` and ``locate``, IPython's ``Macro``, and ``call_command`` of shell, shell_plus or runscript).
Each R12 name is matched at every identifier the parser records -- a name, an attribute, a keyword argument, a
parameter, a function, class or exception name, a global or nonlocal name, a match capture or class-pattern
attribute, a type parameter, every part of an imported module's path, and every imported name and alias, with
attributes and class-pattern attributes never matched against the builtins exec, eval, compile, ``__import__`` and
``__builtins__`` (so ``re.compile`` stays allowed) -- and against every '.'- or ':'-separated part of a string or
bytes constant; a string that is itself Python code (it parses and holds a statement or a call) is read like the
cell's own code, to two levels. So importing a listed name from another module
(``from webbrowser import subprocess``), binding it to another name first (``c = connections``,
``run = os.system``) or looking it up by a literal string (``import_string('subprocess.run')``) is reported too.
The guard makes no claim of completeness.

## "What the guard does not see" as committed (RES, verbatim from the plan)

What the guard does not see: it reads only the notebook's own source, so deliberate obfuscation is out of its
reach. (1) Values assembled or looked up at runtime: a name, module, SQL statement or file path built from pieces
(string concatenation, f-strings, joined path parts, ``str.join``, ``chr``) or fetched with ``getattr``,
``globals()``, ``vars()`` or ``sys.modules`` using such a string, and code decoded at runtime (pickle, marshal,
codecs, base64). (2) Code outside the notebook: a module it imports, IPython's profile and start-up files (with the
extensions, aliases and magics they define), or a library function that starts a process or writes a file under a
name these lists do not contain (for example a path's ``replace`` or ``rename``, ``open`` in a write mode, or a
``webbrowser`` controller given a command line). (3) Left out on purpose: JavaScript or HTML in a cell's output,
which a notebook front end may run when the notebook is opened; an nbconvert regeneration does not run it.

## Django 5.2.17 source behind each statement in the new helper docstring

Paths are under `site-packages/django/` in `/home/tlister/venv/devel_fomo311_venv`.

- **The handler and its wrappers hold the same dict objects as `settings.DATABASES`:** `utils/connection.py` lines 43-51. `settings` (a cached property) calls `configure_settings(self._settings)`, which with `None` returns `getattr(django_settings, 'DATABASES')`. `db/utils.py` line 147 onward (`ConnectionHandler.configure_settings`) only `setdefault`s keys on those same dicts and returns the same mapping.
- **A wrapper built after the switch is a dummy one, which raises `ImproperlyConfigured` on any query:** `db/utils.py` lines 191-194. `create_connection` does `db = self.settings[alias]`, then `load_backend(db["ENGINE"])`, then `backend.DatabaseWrapper(db, alias)`. `db/backends/base/base.py` lines 53-60 store that same dict as `self.settings_dict`. `db/backends/dummy/base.py` lines 19-24: `complain` raises `ImproperlyConfigured`.
- **A cached wrapper that already holds an open connection keeps using it:** `db/backends/base/base.py` lines 271-279. `ensure_connection` connects only `if self.connection is None`. An open connection is reused whatever `settings_dict` now says.
- **Close is best effort:** `utils/connection.py` lines 83-85. `close_all` closes each initialised wrapper. In the helper, a failure is logged and never raised (code unchanged).
- **Dropping a cached wrapper makes the next access build a new one:** `utils/connection.py` lines 56-64 and 69-70. `__delitem__` does `delattr(self._connections, key)`. `__getitem__` then misses the cache and calls `create_connection(alias)`, which builds a dummy wrapper from the switched dict.

## Notebook cells changed (commit 17168ff)

| Notebook | Cell id | Change |
|----------|---------|--------|
| backfill_lco_observations_demo.ipynb | 46602df0 (teardown markdown) | OLD_T -> T1 |
| campaign_lifecycle_demo.ipynb | 7dfef438 (setup markdown) | OLD7 -> F7 |
| campaign_lifecycle_demo.ipynb | 234805b6 (teardown markdown) | OLD_T -> T1 |
| load_telescope_runs_demo.ipynb | 49ff72df (setup markdown) | OLD49 -> F49 |
| load_telescope_runs_demo.ipynb | 31ff3edb (teardown markdown) | OLD_T -> T1 |
| telescope_runs_demo.ipynb | 23326d63 (setup markdown) | OLD23 -> F23 |

The other four notebooks are byte-identical to 1930f31. No notebook was re-executed, and nbconvert was not run.

## Check OK lines

- `OK: guard, helper and projector tests green after the docstring and README changes` (Ran 117 tests, OK)
- `OK: guard docstring states what the guard reports and ends with RES; guard and helper code unchanged; helper docstring corrected; README carries README_GUARD and README_IN01 with D-04, S1-S3 and both lists unchanged`
- `OK: ruff clean; the docs commit holds exactly the guard module, README and helper (7af64d05aecae28af19b8066c7af6f6b6f6c5d68); no notebook, runbook or verification report changed`
- `OK: guard (every real notebook, the dip pin, credential text), helper and projector tests green with the edited prose` (Ran 117 tests, OK)
- `OK: six sentences replaced exactly, every other cell, code cell and output identical to 1930f31, T1 in all eight teardown cells, the widened never-opened check clean in all eight notebooks`
- `OK: CR-01, WR-02 and IN-01 fixed with real commits touching their files, open 6 -> 3, 7 lines since 40-10 and 11 since 1930f31, other rows and the ID-reuse sections untouched` (the plan's version expects 12, see Deviations)
- `OK: notebook hooks pass; the four notebooks are committed together (17168ff091b323fdba82ac5fb5fcd9af31bab222); the ledger commit follows alone (7a432513c4a079b357bd2c6b0fdc692751071de3); runbooks, the review and the verification report unchanged`

## Accomplishments

- The guard docstring describes the guard as 40-10 left it, rule by rule. It lists every matching position instead of saying "wherever it appears", and its residual is RES and nothing more.
- The README guard paragraph says the guard reports a name whether it is written in code, in an import or inside a string. It names shutil's copy functions, `move` and `write_bytes`, the allowed magic lines and the one-line magic or alias rule.
- The helper docstring and the README refusal sentence now say that dropping the cached wrappers, whether or not the close worked, is what removes a connection that is still open. They no longer give that credit to the dummy switch.
- No notebook prose claims the developer database was never opened or is never written. Each teardown cell says what the helper checked, which matches the output printed below it.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] The ledger check's "12 lines since 1930f31" figure is wrong, and 11 is correct**
- **Found during:** Task 2 (ledger)
- **Issue:** The plan says exactly 7 lines differ from the 40-10 ledger commit and 12 from 1930f31. 40-10 changed 5 lines: two dispositions, two rows and `open: 8 -> 6`. This plan changes 7: three dispositions, three rows and `open: 6 -> 3`. The `open:` line is the same line both times, so 11 distinct lines differ from 1930f31. The plan's third verify command asserts 12, which no correct ledger can satisfy.
- **Fix:** No ledger change. The check was run exactly as written, and it failed only on that count (`found 11`). It was then run again with the count set to 11, and every other assertion passed: 7 lines since 40-10, valid hashes touching each finding's file, open 3, total 12, untouched rows and ID-reuse sections.
- **Commit:** 7a43251

**2. [Rule 1 - Accuracy] The docstring wording differs from the plan's draft where the code differs**
- **Found during:** Task 1
- **Issue and fix (each sentence follows the code in the committed guard):**
  - String code is described as "a string that is itself Python code (it parses and holds a statement or a call)". This matches 40-10's `_parse_string_code` (40-10 deviation 1), not just "a string that is itself Python code".
  - The automagic exceptions also name "a builtin or a keyword", which `_automagic_word` checks. "Not bound earlier" is phrased as "unless an earlier cell binds that name", which is what `bound` holds.
  - "An escape anywhere" became "an escape on any line (also after a line magic, as in ``%time !ls``)". An escape character inside a string is not an escape, so "anywhere" would claim too much.
  - `%%capture`'s allowed flags are listed by name (`--no-stderr`, `--no-stdout`, `--no-display`), as in HARMLESS_CELL_MAGIC.
  - ATTACH is written as its own fifth capability, as `_database_rerouting_problems`'s docstring lists it. The text says what ATTACH_SQL matches: ATTACH followed by DATABASE, a quote or a placeholder, or an opening parenthesis.
  - The first paragraph names "CR-01 and CR-02 of the third review", as the plan asked.
- **Files modified:** solsys_code/tests/test_pre_executed_notebooks.py (docstring only; ast.dump equal to 40-10's commit with the docstring removed)
- **Commit:** 7af64d0

**3. [Rule 3 - Format] README line re-wrapped**
- **Found during:** Task 1
- **Issue:** Putting README_IN01 in place left the following sentence ("When the checks pass, the helper runs ...") on a line longer than 120 columns.
- **Fix:** That line was wrapped at the same point as the surrounding text. Whitespace-normalized, the README is unchanged apart from the two planned replacements.
- **Commit:** 7af64d0

**Total deviations:** 3 auto-fixed (1 plan-count error, 1 wording accuracy, 1 wrap). **Impact:** none of them changes code or behavior. Deviation 1 means 40-12 and /gsd-verify-work should expect 11 changed ledger lines since 1930f31, not 12.

## Out-of-scope observation (not changed)

- `_disable_django_databases`'s warning text in code still says "every alias is already on the dummy backend, so no database is reachable" when the close fails. This sentence overstates in the same way as the old docstring, because a cached wrapper with an open connection is only removed by the drop that follows. It is a string inside code, and the plan forbids any change to the helper's code (ast.dump must equal 1930f31), so it is left for Phase 41 triage alongside IN-01.

## Scope statement

- No code in the guard module or the helper changed. No notebook was re-executed, and no code cell or output changed.
- 40-VERIFICATION.md, 40-REVIEW.md and every `docs/runbooks/` page are unchanged since 1930f31 (checked).
- Round-2 IN-03, WR-03, IN-04, the IN-07 remainder and round-1 IN-03 (runbook text) remain for Phase 41 triage.
- Verification human items 3 and 4 are reading items at /gsd-verify-work. Human item 2 (README and guard docstring wording) is this plan's `<human-check>`.
- Nothing was pushed. The user's own files (`.planning/config.json`, `.planning/state.json`, `.planning/ui-reviews/.gitignore` and the untracked files) were never staged.
- The plan's step 5 says not to commit this SUMMARY. The orchestrator's sequential-mode protocol (Write SUMMARY, then commit, then STATE/ROADMAP) was followed instead, as 40-10 did (3c23dec).

## Issues Encountered

None beyond the ledger count in deviation 1.

## Known Stubs

None. The plan changed only documentation, notebook prose and the ledger.

## Threat Flags

None. No new code, endpoint, file access or schema surface.

## Next Phase Readiness

Ready for 40-12, which puts the corrected override to the developer. It can quote the "What the guard checks" paragraph and RES above, and it cites only 40-10's tests and this docstring.

## Self-Check: PASSED

- FOUND: solsys_code/tests/test_pre_executed_notebooks.py, docs/notebooks/README.md, docs/notebooks/fomo_nb_utils.py, the four notebooks and 40-REVIEW-DISPOSITION.md
- FOUND commits: 7af64d0, 17168ff, 7a43251 (all ancestors of HEAD; `git rev-list --count 506362b..HEAD` = 3 before this SUMMARY)
