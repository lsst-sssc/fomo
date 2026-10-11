---
phase: 40-notebook-isolation-attribution-page
reviewed: 2026-10-11T03:48:28Z
depth: deep
files_reviewed: 2
files_reviewed_list:
  - solsys_code/tests/test_pre_executed_notebooks.py
  - docs/notebooks/README.md
findings:
  critical: 2
  warning: 1
  info: 3
  total: 6
status: issues_found
---

# Phase 40: Code Review Report (sixth round: gap plans 40-15 and 40-16)

**Reviewed:** 2026-10-11T03:48:28Z
**Depth:** deep
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Scope: the 2 listed files, compared with `da88285` (HEAD `ce82278`). That covers 40-15 (`bc99b0e`: R6 reports
`shutil`, `_shutil` and `write_bytes` at every identifier and string part; `1c830b1`: `IPyAutocall` added to R12,
plus tests that pin autocall and the Django-backed gate reading raw cells) and 40-16 (`b04f09f`: the module docstring
and the README guard paragraph rewritten). I also read the override reason in `40-VERIFICATION.md`
(`overrides[0].reason`, `88fc09e`) for the claims it repeats.

What I ran:
- `python manage.py test --noinput solsys_code.tests.test_pre_executed_notebooks`: 91 tests, OK.
- `ruff check --no-fix` and `ruff format --check` on the guard module (venv ruff 0.16.10): both clean.
- Static calls of the guard's own functions (`isolation_problems`, `is_django_backed`, `_string_code_trees`,
  `_django_imports`, `_identifiers`). Each ran on placeholder source or on a deep copy of
  `import_campaign_csv_demo.ipynb`, with source appended to the setup cell (`_append_to_setup_cell`), inserted
  after it (`_insert_code_cell_after_setup`), or inserted before it by a local copy of the same helper. The module
  was loaded by file path from `$HOME/tmp/rv40r6`.
- One child process with its own empty `IPYTHONDIR`. It built `ZMQInteractiveShell.instance()` (no kernel), defined
  an alias and a macro whose only command is `echo placeholder-only`, set `autocall = 2`, and called
  `transform_cell` on one-line sources. `transform_cell` rewrites text and runs none of it.
- `inspect.getsource` of ipykernel's `ZMQExitAutocall`.
- No notebook was executed, `django.setup()` was not called, `src/fomo_db.sqlite3` was not opened and
  `FOMO_DATABASE_PATH` was not set. No repository file other than this report was written.

**Previous round, as of HEAD:**

| ID | Closed? |
|----|---------|
| CR-01 (shutil handed on to a parameter, lambda, match capture or class pattern) | **Yes.** `SHUTIL_MODULE_NAMES` is matched at every identifier `_identifiers` yields and at every string part (`:900-904`, `:916-921`). All six review forms are reported on a copy of a real notebook (`test_shutil_passed_on_in_any_way_is_reported_on_a_real_notebook`). |
| CR-02 (`write_bytes` through a class-pattern attribute) | **Yes.** `BYTE_WRITE_NAMES` goes through `_identifiers` (`:911-915`), so `kwd_attrs`, parameters and keywords are covered. One stated side claim is false (WR-01). |
| WR-01 (Django-backed gate reads raw cells) | **Closed as documentation (D-G12).** The code is unchanged. The docstring (`:44-47`) and the `isolation_problems` docstring now say the gate reads each cell as written, and `test_the_django_backed_gate_reads_each_cell_as_written` pins that. The wording matches the code. |
| WR-02 (R0 does not see autocall) | **Partly.** `IPyAutocall` is on R12's list and the exception is named in the docstring and README. The claim that autocall of `exit`/`quit` objects is the *one* rewrite R0 misses is still too strong, because a notebook can switch IPython's autocall mode on (CR-01). The exit claim itself is inaccurate (IN-02). |

Both new criticals are routes that existed before this round. This round's rewrite made the claims they break
explicit. CR-01 breaks R0's alias claim and the "one rewrite" autocall claim: IPython's shell is reachable without
any R12 name. CR-02 breaks the README's "never sets Django up ... in code inside a string": R3 and R4 never read
string code. Both routes are literal source, and neither falls inside the residuals the docstring and the override
reason list.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: IPython's shell is reachable as `InteractiveShell.instance()`, so a notebook can define a shell alias that a later one-line cell runs, with no name R0 or R12 reports

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:176-203` (`PROCESS_NAMES` lists `get_ipython`, `system_raw`, `run_line_magic`, `run_cell_magic`, but not the shell class or its alias, cell and autocall entry points), `:429-448` (`_ipython_line_magic_names` is IPython's default set only), `:451-483` (`_automagic_word`). Claims: module docstring `:28-35` (alias definition is treated as a threat: "`%config AliasManager.user_aliases = [...]` defines a shell alias that a later one-line cell runs"; "A one-line cell whose first word ... is an IPython line magic or alias ... is reported"), `:36-40` ("R0 does not see one rewrite, IPython's autocall ... The kernel's only autocall objects, `exit` and `quit`"), `:74-78`; `isolation_problems` docstring `:1092-1094`; `docs/notebooks/README.md:98-101`; and the same sentences in the override reason.

**Issue:** `get_ipython` is listed, but `InteractiveShell.instance()` returns the same object. traitlets'
`SingletonConfigurable.instance()` stores the kernel's `ZMQInteractiveShell` on every class in its MRO. The child
check confirmed `InteractiveShell.instance() is ZMQInteractiveShell.instance()` is `True`, and that
`IPython.InteractiveShell` is that class. None of `IPython`, `InteractiveShell`, `ZMQInteractiveShell`, `instance`,
`alias_manager`, `define_alias`, `run_cell` or `autocall` is on any list. Each row below was appended to the setup
cell of a copy of `import_campaign_csv_demo.ipynb`, and `isolation_problems()` returned `[]` for each:

| Form (literal source) | What IPython does | Reported |
|---|---|---|
| `from IPython.core.interactiveshell import InteractiveShell` / `InteractiveShell.instance().alias_manager.define_alias('wipe', 'cd ../.. && env -i PATH=/usr/bin python manage.py flush --noinput')`, then a new one-line cell `wipe` | child shell: `transform_cell('wipe\n')` gives `get_ipython().run_line_magic('wipe', '')`, i.e. the alias runs as a shell command | No |
| same import / `InteractiveShell.instance().run_cell('!cd ../.. && env -i PATH=/usr/bin python manage.py flush --noinput')` | `run_cell` applies IPython's transforms, so `!` starts a shell. The string is not Python, so `_string_code_trees` does not read it, and `manage.py` is checked only in cells that do not parse | No |
| `from IPython import InteractiveShell` / `InteractiveShell.instance().autocall = 2` | child shell: `transform_cell('f [1]\n')` gives `f([1])`, so every later one-line `callable [x]` cell runs as a call, not the subscript the guard reads | No |
| same import / `InteractiveShell.instance().ex('print(1)')` | `ex` is `exec` in the user namespace (a literal Python string is still read by R12) | No |

The `wipe` cell is the alias route the docstring blocks for `%config`, reached without a magic. `_automagic_word`
returns `None` for it because `wipe` is not in IPython's default name set. The command string has no listed part.
`env -i` and `manage.py` are searched for only in cells that do not parse, and the string never names
`FOMO_DATABASE_PATH`. Run from `pre_executed/`, `env -i` clears the helper's `FOMO_DATABASE_PATH`. The child
`manage.py` then falls back to `src/fomo_db.sqlite3` (`settings.py:100`) and flushes the developer database. This is
the re-review's FORM_D (`!cd ../.. && env -i ... flush`), which R0 reports as a bare escape, carried into a parseable
cell. Both rows, with the `env -i` string, were re-probed: `[]`. (The macro
route, `define_macro`, does not give a shell. In the child, the macro text came back as raw `!echo ...`, which then
fails to compile.)

**Inside or outside the stated residual: outside for the alias form.** Residual (2) names "IPython's profile and
start-up files (with the extensions, aliases and magics they define)". This alias is defined by the notebook's own
code, not by a profile. `define_alias` starts no process. The process comes from a one-line notebook cell, and R0
claims to check one-line cells that start with an alias. The `run_cell` form could be argued to fall under residual
(2) ("a library function that starts a process ... under a name these lists do not contain"). But R12 lists
`get_ipython` precisely so the shell object cannot be reached, and this is the same object by another accessor. The
autocall-mode form means the docstring's "R0 does not see one rewrite, IPython's autocall: a one-line cell that
starts with an autocall object" and "the kernel's only autocall objects, `exit` and `quit`" understate what R0
misses. With autocall on, any callable is called.

**Override reason:** the reason says "R0 reports any one-line cell whose first word is an IPython line magic or
alias" and that, apart from autocall of an autocall object, R0 reports any cell that IPython does not run unchanged.
Both are false for the forms above, so the reason does not cover them.

**Practical weight, for the developer's decision:** as in rounds 4-5, residual (2) already accepts process-starting
library calls under unlisted names (for example a `webbrowser` controller given a command line). These forms add a
route the documentation says is closed. They do not add a new capability class.

**Fix:** no real notebook names `IPython`, `ipykernel`, `traitlets`, `InteractiveShell`, `alias_manager`,
`run_cell`, `define_alias` or `autocall` in code (checked over all eight). So report the shell like the process
modules:

```python
# R12: a notebook never needs IPython's shell object; get_ipython() is only one way to reach it.
PROCESS_MODULES = frozenset({..., 'IPython', 'ipykernel', 'traitlets'})
PROCESS_NAMES = frozenset({..., 'InteractiveShell', 'ZMQInteractiveShell', 'IPKernelApp', 'alias_manager',
                           'define_alias', 'soft_define_alias', 'run_cell', 'run_cell_async', 'autocall',
                           'define_macro', 'magics_manager', 'register_magic_function'})
```

Add the four forms above to `TestR0AllowList` or `TestThirdReviewFormsOnARealNotebook`. Then reword the docstring's
autocall sentences to say that R12 reports the shell and its autocall switch, and update the README (`:98-101`) and
the override reason to match.

### CR-02: R3 and R4 do not read code inside strings, so Django set up in a string before the helper call writes the developer database unreported, against the README's explicit claim

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:679-693` (`_django_setup_problems` walks only the cell tree), `:1190-1195` (R3 calls `_django_imports(statement)`, cell tree only), `:594-607` (`_django_imports`, `is_django_backed`). Claims: `docs/notebooks/README.md:92-95` ("It checks that a notebook ... never sets Django up, ... whether the name is written in its code, in an import, as a part of a dotted string ..., or in code inside a string (read to two levels)"); `_string_code_trees` docstring `:527-529` ("Code in a literal string runs as code under `timeit`, ... so the rules read it like the cell's own code"); module docstring `:13-15` ("before any Django import -- never sets Django up").

**Issue:** only R6 and R12 call `_string_code_trees`. R4 (`django.setup`) and R3 (Django import before the helper)
read the cell tree only. A code cell inserted **before** the setup cell of a copy of
`import_campaign_csv_demo.ipynb`:

```python
import timeit
timeit.timeit("import django; django.setup(); from django.core.management import call_command; call_command('flush', interactive=False)", number=1)
```

`isolation_problems()` → `[]`. Control: the same cell with `import django\ndjango.setup()` as plain code gives
`calls django.setup()` and `imports Django before the setup_scratch_django call`. `code.InteractiveInterpreter().
runsource('import django as dj; dj.setup()')` before the helper also gives `[]`. Nothing in the string is on an R12
list: `call_command` is reported only for `shell`, `shell_plus` or `runscript`. Before the helper runs,
`FOMO_DATABASE_PATH` is unset, and `src/fomo/settings.py:100` falls back to `BASE_DIR/fomo_db.sqlite3`. The string
therefore flushes the developer database. The helper's later `RuntimeError` ("Django is already set up or
configured") comes after the write. With fabricated good output, which is the WR-13 threat model this guard exists
for, the committed notebook passes.

`is_django_backed` also ignores string code. A notebook whose only Django use is inside such a string gets none of
R2-R4 or R8-R11 (`is_django_backed` → `False`, `isolation_problems` → `[]` on a one-cell probe). That part is held
in this repo only by `test_the_real_notebook_check_is_not_vacuous`, as in round 5's WR-01.

**Inside or outside the stated residual: outside.** The source is literal, so residual (1) does not apply. The
code is in the notebook, not outside it (residual 2). It is one string level deep, inside STRING_CODE_DEPTH. The
README states the opposite of what the code does.

**Override reason:** the WARN-02 must-have is "fails for a notebook whose current source routes Django to the
developer database". This form does exactly that, and it is not among the residuals the reason lists. So the reason
does not cover it.

**Fix:** have R3, R4 and the gate read string code the way R6 and R12 do. No real notebook has a string with a Django
import or a `django.setup` call (checked over all eight), so nothing real changes:

```python
def _with_string_code(tree):
    return (tree, *_string_code_trees(tree))

# R4: in _django_setup_problems
for scanned in _with_string_code(tree):
    for node in ast.walk(scanned): ...

# R3
if _django_imports(statement) or any(_django_imports(inner) for inner in _string_code_trees(statement)):
    problems.append(...)

# gate
def is_django_backed(notebook):
    return any(tree is not None and any(_django_imports(t) for t in (tree, *_string_code_trees(tree)))
               for _n, _c, tree in _parsed_code_cells(notebook))
```

Add the two before-the-helper probes as cases.

## Warnings

### WR-01: "a cell has at most one write_bytes problem" is false

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:835-836` (`_byte_copy_problems` docstring), `:905-907` (comment: "so a cell never has two write_bytes problems"), `:893-899` (string-part message text).

**Issue:** the identifier pass reuses the reference pass's exact text, so those two collapse. The string-part pass
uses different text ("names write_bytes in a string; ..."). A cell with `p.write_bytes(b'')` and
`getattr(p, 'write_bytes')` gives two write_bytes problems:
`['code cell 1 references write_bytes; ...', 'code cell 1 names write_bytes in a string; ...']`.
`test_write_bytes_at_any_identifier_is_reported_on_a_real_notebook` asserts `== 1` only for forms with no string
part, so the claim is pinned for less than it states. This is not a detection gap. It is a stated invariant that is
false.

**Fix:** either say "at most one write_bytes problem from identifiers, and one more for a string part", or key
`found` by `(kind, name)` and keep the first text. Add a string-plus-method case to the test that asserts whichever
count is documented.

## Info

### IN-01: `from tempfile import _shutil as sh` then `sh.move(a, b)` reports `_shutil` but not the `move`, though the `_byte_copy_problems` docstring says a name bound to `_shutil` by an import counts

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:803-805` (`alias.name == 'shutil'` only), docstring `:829-831` ("``_shutil``, a name bound to either by an import").

**Issue:** `_shutil_names` adds import aliases only when the imported name is `shutil`. The probe returns only
`references _shutil`. Detection holds because `_shutil` is reported at the alias, so this is wording only. The module
docstring (`:58-60`) states the narrower, correct rule.

**Fix:** `if alias.name in SHUTIL_MODULE_NAMES`, or change the function docstring to "a name bound to `shutil` by an
import".

### IN-02: `exit -1` does not end the kernel, though the docstring and README say the kernel's autocall objects "end the kernel" and give `exit -1` as the example

**File:** module docstring `:37-38`; `docs/notebooks/README.md:99-100` ("`exit` and `quit`, only end the kernel"); same words in the override reason.

**Issue:** ipykernel's `ZMQExitAutocall.__call__(self, keep_kernel=False)` sets `keepkernel_on_exit = keep_kernel`,
and `ask_exit` sets `exit_now = not self.keepkernel_on_exit`. So `exit -1` → `exit(-1)` keeps the kernel alive and
only writes an `ask_exit` payload. There is no database route, but the cited example contradicts the sentence it
illustrates.

**Fix:** "end the kernel, or with a true argument only ask the front end to close".

### IN-03: The README and docstring still leave out checks the code makes (round-5 IN-01, unchanged)

**File:** `docs/notebooks/README.md:92-103`; module docstring `:47-48`.

**Issue:** the round-5 IN-01 items are unchanged: the README omits R7's keyword check, R10, R8's error-output check
and R11's setup-cell `assert` rule, and the docstring's R11 summary omits the setup-cell `assert`. The README
paragraph this round rewrote also dropped `copy` and `move` from what it lists. The guard still reports them
whenever they are read from a shutil name, so that is now an omission as well. Detection is unaffected, because any
shutil name is itself reported.

**Fix:** as in round 5, add "ends with the teardown, has no error output, no `assert` in the setup cell" to the
README sentence, and mention `copy`/`move` read from shutil.

---

_Reviewed: 2026-10-11T03:48:28Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
