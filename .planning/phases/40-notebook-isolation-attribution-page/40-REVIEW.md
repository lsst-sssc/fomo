---
phase: 40-notebook-isolation-attribution-page
reviewed: 2026-10-11T13:54:05Z
depth: deep
files_reviewed: 2
files_reviewed_list:
  - solsys_code/tests/test_pre_executed_notebooks.py
  - docs/notebooks/README.md
findings:
  critical: 1
  warning: 4
  info: 4
  total: 9
status: issues_found
---

# Phase 40: Code Review Report (seventh round: gap plans 40-17 and 40-18)

**Reviewed:** 2026-10-11T13:54:05Z
**Depth:** deep
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Scope: the 2 listed files, compared with `f1369ad` (HEAD `ab72542`). That covers 40-17 (`c78e517`: IPython, ipykernel,
traitlets and IPython's shell classes and entry points added to R12; `fdd8b3e`: R3, R4 and `is_django_backed` read
code inside strings; `ca1cb7d`: one write_bytes problem per cell, and an import of tempfile's `_shutil` is followed) and
40-18 (`aff2547`: module docstrings, comments and the README guard paragraph rewritten). I also checked every claim in
the WARN-02 override reason (`40-VERIFICATION.md`, `overrides[0].reason`, accepted 2026-10-11T13:37:50Z).

What I ran:
- `python manage.py test --noinput solsys_code.tests.test_pre_executed_notebooks`: 96 tests, OK.
- `pre-commit run ruff` and `pre-commit run ruff-format` on the guard module: both pass.
- Static calls of the guard's own helpers (`isolation_problems`, `setup_call`, `notebook_mode`, `is_django_backed`,
  `_automagic_word`, `_insert_before_setup_call`, `_insert_code_cell_before_setup`, `_append_to_setup_cell`). They ran
  on placeholder source or on a deep copy of `import_campaign_csv_demo.ipynb`, with the module loaded by file path
  from a scratchpad copy.
- `inspect.getsource` / `hasattr` on IPython classes. No shell instance was created.
- No notebook was executed, `django.setup()` was not called, `FOMO_DATABASE_PATH` was not set and
  `src/fomo_db.sqlite3` was not opened. No repository file other than this report was written.
- Every commit the override reason cites (`301bd93`, `f1b4bb6`, `482606b`, `3232341`, `c6c6322`, `1c830b1`, `c78e517`,
  `fdd8b3e`, `905408a`, `4f02628`, `bc99b0e`, `ca1cb7d`) exists and matches its description. Every R12, R6 and R11 name
  is in the module docstring; os's `exec*`/`spawn*` functions appear only as that summary, which is accurate.

**Previous round, as of HEAD:**

| ID | Closed? |
|----|---------|
| CR-01 (IPython's shell without `get_ipython`) | **Yes, for the four forms.** `InteractiveShell`, `alias_manager`, `define_alias`, `run_cell`, `autocall` and the `IPython` module are reported on a copy of a real notebook. Other ways to reach the shell remain, for example `exit._ip` (an `IPyAutocall` stores the shell as `_ip`). Those fall inside the stated residual class, but they contradict two docstring sentences (WR-02). |
| CR-02 (R3/R4/gate do not read string code) | **Partly.** String code in statements *before* the helper call is now read. The helper call's own argument is not, and it runs before the helper (CR-01 below). |
| WR-01 (two write_bytes problems per cell) | **Yes.** One `('write', name)` key per cell (`:960-971`, `:989-994`). Tests cover method/string order, string code, parameters and a real notebook. |
| IN-01 (`_shutil` import not followed) | **Yes** (`:874-876`). |
| IN-02, IN-03 (docs) | **Yes.** No "ends the kernel" wording remains. The README lists every rule family. |

The 40-17 code changes have no false positives on the eight real notebooks, and the existing rules did not regress.
The remaining problems:
- One literal route sits in the setup statement itself and contradicts the README and the override reason (CR-01).
- Four documentation sentences or instructions are contradicted by the code or by the module's own tests (WR-01 to
  WR-04).

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: the helper call's own argument runs before the helper and is never read by R3, so Django can be set up on the developer database from string code inside `setup_scratch_django(...)`

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:1266-1279` (R2 argument check, R3 loop); `docs/notebooks/README.md:94-96`; `40-VERIFICATION.md` override reason
**Issue:**
- What R3 reads: only top-level statements strictly before `(setup_number, setup_index)` (`:1275-1277`). The setup
  statement itself is excluded.
- What R2 checks: only that the call has one positional argument and a constant `mode` (`:1268-1272`). It never checks
  what that argument is, even though its own message says the call "must be `scratch_db_dir =
  setup_scratch_django(repo_root_path, mode=...)` with exactly those arguments".
- The gap: Python evaluates the argument before the helper runs. String code in the argument is therefore code that
  runs before the helper call, and nothing reads it for a Django import.

Probe: on a deep copy of `import_campaign_csv_demo.ipynb`, insert `import timeit` before the call, then replace the
call line with

```python
scratch_db_dir = setup_scratch_django(timeit.timeit("from django.conf import settings; import src.fomo.settings as s; settings.configure(s); from django.apps import apps; apps.populate(settings.INSTALLED_APPS); from django.core.management import call_command; call_command('flush', interactive=False)", number=1) and repo_root_path, mode='fresh')
```

Result:
- `setup_call()` still finds the call, and `notebook_mode()` still returns `'fresh'`.
- `isolation_problems()` returns `[]`.
- Control: the same `timeit` string as a statement before the call gets `code cell 1 imports Django before the
  setup_scratch_django call`.

Why R4 does not catch it either: the payload never calls `django.setup()`. It uses `apps.populate(...)`, which is what
`django.setup()` does internally. R4 reports only its three written forms (`:756-763`).

What happens at run time: with `FOMO_DATABASE_PATH` unset, `src.fomo.settings` falls back to `src/fomo_db.sqlite3`.
This is the same mechanism the verifier confirmed for round-6 CR-02, and the flush runs against that database. The
helper's later `RuntimeError` ("already configured") comes after the flush.

Why this is not the accepted residual: it is literal source, one string level deep, in the notebook. It contradicts
three specific sentences:
- README `:95-96`: it reports a notebook that "imports Django before that call ..., also in code inside a string".
- The override reason: "R3 and R4 (40-17 [fdd8b3e]) report a Django import before the helper call ... in a cell's code
  and in code inside its strings to two levels".
- R2's own problem text: "with exactly those arguments".

It is the round-6 CR-02 route moved into the one place 40-17 did not extend R3 to.
**Fix:** Make R2 enforce the shape it already prints: the one positional argument must be a plain name. That removes
any code from the argument. Then have R3 also read the call's arguments, so the rule does not depend on R2:

```python
    if len(call.args) != 1 or keyword is None or keyword.arg != 'mode':
        problems.append(f'code cell {setup_number}: the setup call must be {call_shape} with exactly those arguments')
    elif not isinstance(call.args[0], ast.Name):
        problems.append(f'code cell {setup_number}: the setup call argument must be a plain name ({call_shape})')
    elif not (isinstance(keyword.value, ast.Constant) and keyword.value.value in SCRATCH_MODES):
        ...
    # R3: ... and the setup call's own arguments, which Python evaluates before the helper runs.
    for argument in (*call.args, *(kw.value for kw in call.keywords)):
        if any(_django_imports(scanned) for scanned in _with_string_code(argument)):
            problems.append(f'code cell {setup_number} imports Django in the {SETUP_FUNCTION} call arguments')
```

Add the probe above as a real-notebook test next to
`test_django_set_up_in_string_code_before_the_helper_is_reported_on_a_real_notebook`, with the statement-before-the-call
control.

## Warnings

### WR-01: "no other call of the setup function" (new in 40-18) and the README's "once" are false: an attribute call, a call through another name or a call in string code is not reported

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:53-56, 1247-1265`; `docs/notebooks/README.md:94-95`
**Issue:**
- The claims: the R2 docstring says the guard checks "no other call of the setup function", and the README says it
  reports a notebook that does not call the helper "first and once".
- What the code counts: `total_calls` (`:1247-1253`) counts only calls whose `func` is the bare name
  `setup_scratch_django`, and only in cell trees.
- What I ran: on a copy of the real notebook, each of the following returned `[]`:
  - `import fomo_nb_utils` then `other = fomo_nb_utils.setup_scratch_django(repo_root_path, mode='snapshot')`
  - `s = setup_scratch_django` then `other = s(repo_root_path, mode='snapshot')`
  - `timeit.timeit("setup_scratch_django(repo_root_path, mode='snapshot')", globals=globals())`
  - Control: a second top-level bare call is reported.
- Run-time impact: low. The helper raises on a second call because `apps.ready or settings.configured`
  (`fomo_nb_utils.py:293-299`). The sentences still claim a check the code does not make.

**Fix:** Either scope the sentences ("no other call written `setup_scratch_django(...)` in a cell's code"), or count
every `ast.Call` whose `_call_name` is `SETUP_FUNCTION`, across `_with_string_code(tree)`, plus every reference to
`SETUP_FUNCTION` other than the import and the one call (`ast.Name`/`ast.Attribute` loads).

### WR-02: "R12 reports the IPython shell names a notebook would write to set those up" is contradicted by the module's own test and by `exit._ip`

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:37-40, 125-126, 1174-1177` (contradicting test at `:2723-2725`)
**Issue:** Three sentences say that the run-time rewrites R0 does not read (an autocall object, and an alias or magic
the notebook defines) are set up through IPython shell names that R12 reports:
- `:39-40`: "R12 reports the IPython shell names a notebook would write to set those up".
- `:126`: "R12 reports the shell names such a definition writes".
- `:1176-1177`: the same sentence in `isolation_problems`.

Two counterexamples:
1. **The module's own test.** `test_autocall_is_not_seen_by_r0_and_the_ipyautocall_name_is_reported`, subtest "an
   autocall object made without the name" (`:2723-2725`), sets up an autocall object with
   `class Bye(type(exit)): ...; bye = Bye()` and asserts `[]`.
2. **`exit._ip`.** `IPyAutocall.__init__` stores the shell as `self._ip` (checked with `inspect.getsource`), and the
   kernel's `exit`/`quit` are such objects. On a copy of the real notebook:
   - Appending `exit._ip.find_magic('alias')('wipe cd ../.. && env -i PATH=/usr/bin python manage.py flush --noinput')`
     and adding a later one-line cell `wipe` gives `[]`. `_automagic_word('wipe')` is `None`.
   - `exit._ip.find_line_magic('sx')('<command>')` and `sys.displayhook.shell.find_line_magic('sx')(...)` also give
     `[]`.
   - `find_magic` and `find_line_magic` are `InteractiveShell` attributes.

The routes themselves are inside the accepted residual class ("a route to ... IPython's shell ... through a name or
form the lists do not contain"). The override reason does not repeat these sentences, so the override itself is
unaffected. The three sentences are still false as written.
**Fix:** Reword all three, for example: "R12 reports the listed IPython shell names (`get_ipython`, `InteractiveShell`,
`alias_manager`, `define_alias`, `autocall`, ...) when a notebook writes them; a definition made through an unlisted
name (an autocall object built from `type(exit)`, the shell reached as `exit._ip`) is not seen." If you want these
cheap routes closed rather than documented, add `_ip`, `find_magic`, `find_line_magic` and `find_cell_magic` to
`PROCESS_NAMES`. None of them appears in the eight notebooks. Also add a pinned test that `exit._ip` is reported.

### WR-03: "a comment that merely mentions a name cannot ... trip a check" is false for any cell under a harmless magic, and 40-17's new names widen it

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:21-22, 357, 1192-1197`
**Issue:**
- The cause: a cell with a first-line `%%time`/`%%capture` does not parse as written (`tree is None`). R0's deny-list
  then runs the regexes over the raw source, comments included.
- Probe: `%%time\n# the solar system objects\nx = 1\n` and `%%time\n# works under IPython\nx = 1\n` each give "code cell
  2 is not parseable Python but mentions Django, a database, a process, ...". The same comment in a plain cell gives
  `[]`.
- Why 40-17 makes it worse: R12 now has more words (`IPython`, `traitlets`, `run_cell`, `autocall`), so more ordinary
  comments trip the check. Lower-case "solar system" is a likely comment in a Solar System project.
- The stale comment: line 357 ("An unparseable cell (IPython magics) is tolerated unless it is about Django or a
  database") is also false. Such a cell is reported by the allow-list unless only harmless magic lines make it
  unparseable, and the deny-list also covers processes, shell commands and the environment.

**Fix:** Either strip comments before the deny-list search (tokenize the `_python_view`, or search only the magic lines
plus the parsed view's string constants), or reword `:21-22`: "Code that parses is inspected with `ast` ...; a cell
that does not parse as written, including one under `%%time` or `%%capture`, is searched as text, comments included."
Update the line-357 comment to say what the deny-list covers.

### WR-04: the README sends you to `/tmp` for a leftover snapshot (a full copy of the developer database), but the helper uses `$TMPDIR`

**File:** `docs/notebooks/README.md:59, 89-90` (helper: `docs/notebooks/fomo_nb_utils.py:249`)
**Issue:**
- Where the directory really goes: `prepare_scratch_database` calls `tempfile.mkdtemp(prefix=SCRATCH_DIR_PREFIX)`,
  which honours `TMPDIR`.
- What the README says: "creates a `/tmp/fomo-notebook-db-*` directory", and "If a run fails before the teardown cell,
  delete any leftover `/tmp/fomo-notebook-db-*` directory yourself".
- The README pins `TMPDIR=/tmp` only for the `nbconvert` command. The *Restart and Run All* route it also recommends
  inherits the user's `TMPDIR`; this developer's own setup exports `TMPDIR=$HOME/tmp`.
- The result: after a failed snapshot run, the full copy of the developer database is somewhere the instructions do not
  say to look. This is pre-existing wording, but the file is in scope and the consequence is data left behind.

**Fix:** Write "a `fomo-notebook-db-*` directory in Python's temporary directory (`$TMPDIR`, else `/tmp`)" in both
places. The cleanup line could say `python -c "import tempfile; print(tempfile.gettempdir())"` to find it.

## Info

### IN-01: the README says code inside strings is read, but R5/R7 do not read it, so an escaped `FOMO_DATABASE_PATH` in string code is not reported

**File:** `docs/notebooks/README.md:92-97`; `solsys_code/tests/test_pre_executed_notebooks.py:767-786`
**Issue:** The README says where names are found ("... and code inside its strings (read to two levels)"), then lists
"names `FOMO_DATABASE_PATH` or the developer database file in a str or bytes literal". R5/R7 match only the outer
string's text, as the module docstring correctly says (`:583-584`). So `timeit.timeit("print('\\x46OMO_DATABASE_PATH')")`
gives `[]`: the inner literal's value is `FOMO_DATABASE_PATH`, but the outer text is not. For the file name the gap is
masked, because R12 reports the `sqlite3` part of any string-code literal `fomo_db.sqlite3`. Using the variable name
also needs a listed environment or process name.
**Fix:** In the README, say "code inside its strings (read to two levels) for the Django, shutil and listed-name
checks", or have `_database_routing_problems` walk `_with_string_code(tree)`.

### IN-02: "as the not-vacuous test requires" no longer holds after the gate learned string code

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:50-51, 2985-2994`
**Issue:** The docstring says every real notebook makes a Django import "in a cell that parses as written, as the
not-vacuous test requires". The test asserts `is_django_backed(notebook)`, which since `fdd8b3e` also accepts a notebook
whose only Django import is a string such as `"import django"`.
**Fix:** Either drop "as the not-vacuous test requires", or have the test also assert that some cell tree, not string
code, holds a real import, using `_django_imports(tree)`.

### IN-03 (residual class, accepted): patching `tempfile.mkdtemp` before the helper call makes the developer database the "scratch" file, and the helper's checks pass

**File:** `docs/notebooks/fomo_nb_utils.py:249, 300-312` (guard: no rule covers it)
**Issue:** `import tempfile\ntempfile.mkdtemp = lambda prefix=None: str(repo_root_path / 'src')`, inserted before the
call on a copy of the real notebook, gives `[]`. At run time the helper sets `FOMO_DATABASE_PATH` to
`src/fomo_db.sqlite3`. `check_database_settings` passes because it compares against that same path. `migrate` and every
later demo write then go to the developer database. Only the teardown's prefix check would refuse, after the writes.
`tempfile` and `mkdtemp` are on no list, so this is inside the accepted residual class, not a documentation finding. It
is recorded because it is two literal lines and costs no obfuscation.
**Fix (helper hardening, optional):** in `setup_scratch_django`, before setting the environment, refuse unless
`scratch_db_path.parent.parent.resolve() == Path(tempfile.gettempdir()).resolve()`, the directory name starts with
`SCRATCH_DIR_PREFIX`, and `scratch_db_path.resolve() != developer_database_path(repo_root).resolve()`. These are the
same checks the teardown already makes.

### IN-04: three spellings of "the tree plus its string code"

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:603-605, 916, 1124`
**Issue:** `_with_string_code` was added in 40-17, but `_byte_copy_problems` (`:916`) and
`_database_rerouting_problems` (`:1124`) still build `(tree, *_string_code_trees(tree))` inline. A future change to
what "string code" means (depth, filters) has to be made in three places.
**Fix:** Use `_with_string_code(tree)` in both.

---

_Reviewed: 2026-10-11T13:54:05Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
