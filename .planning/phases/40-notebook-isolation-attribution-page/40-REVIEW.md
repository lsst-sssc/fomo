---
phase: 40-notebook-isolation-attribution-page
reviewed: 2026-10-10T19:30:37Z
depth: deep
files_reviewed: 7
files_reviewed_list:
  - docs/notebooks/fomo_nb_utils.py
  - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
  - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
  - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
  - docs/notebooks/pre_executed/telescope_runs_demo.ipynb
  - docs/notebooks/README.md
  - solsys_code/tests/test_pre_executed_notebooks.py
findings:
  critical: 1
  warning: 2
  info: 2
  total: 5
status: issues_found
---

# Phase 40: Code Review Report (fourth round: gap plans 40-10, 40-11 and 40-12)

**Reviewed:** 2026-10-10T19:30:37Z
**Depth:** deep
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Scope: the 7 listed files, compared with `93aedf79` (HEAD `399e426`). That covers the R12 and R6 work in 40-10
(identifier, string and string-code matching, new names, an R0 allow-list over IPython's own input transformer, and
an automagic rule), the docstring, README and notebook prose in 40-11, and the corrected override reason in
40-12 (`40-VERIFICATION.md` front matter, read for the claims it makes).

What I ran:
- `python manage.py test --noinput solsys_code.tests.test_pre_executed_notebooks`: 82 tests, OK.
- A cell-by-cell comparison of the four edited notebooks with `93aedf79`. Code, outputs and execution counts are
  identical; only markdown changed (1, 2, 2 and 1 cells). So no re-execution was needed, and none of the helper's
  code changed (40-11 changed only its docstring).
- Inert unit checks of the guard's own helpers (`_module_bindings`, `_byte_copy_problems`,
  `_database_routing_problems`) on placeholder source, next to plain Python runs that show what those sources bind
  at runtime. All were run under `$HOME/tmp/rv40r4`. No notebook was executed, no repo file was modified, and
  `src/fomo_db.sqlite3` was never opened.
- The installed IPython 9.10.1 source for `is_shadowed`, `AutoMagicChecker.check` and
  `InteractiveShell.transform_cell`.
- `ruff check --no-fix` on the two Python files: clean.

Method limit, stated up front: this round does not run complete bypass notebooks end to end. Each finding rests on
the guard's code, an inert check of the guard function concerned, and the documented Python or IPython behaviour.
Where a route depends on a step I did not run, the finding says so.

**Previous round, as of HEAD:**

| ID | Closed? |
|----|---------|
| CR-01 (R12 literal routes) | **Yes.** Every listed form is reported: a module re-exported from another module, `import_string`/`resolve_name` by literal string, the `connection` proxy with `_connections`/`Database`, `ATTACH (` and the IPython process helpers. `TestThirdReviewFormsOnARealNotebook` and `TestR12MatchesEveryIdentifierAndString` pass. The README guard paragraph (lines 92-99) is now scoped to names. |
| CR-02 (R0 aliases, automagic, escapes after a line magic) | **Yes for the listed forms.** R0 is now an allow-list checked with IPython's own `TransformerManager`, and it has an automagic rule. All seven review cells are reported. The automagic rule's exemption for a name the notebook binds is wider than what Python actually binds; see CR-01 below. |
| WR-01 (R6 `copytree` / `write_bytes`) | **Yes for the listed forms.** `copytree`, `copyfileobj`, `move` and `write_bytes` are reported as calls, references, imports and string lookups. R6 still follows `shutil` only through `import shutil as X`; see WR-01 below. |
| WR-02 (unmeasured "never opened" prose) | **Yes.** Teardown markdown `46602df0`, `234805b6` and `31ff3edb` now give the T1 statement, which matches word for word the second line of the teardown output right below it. Setup markdown `7dfef438` and `49ff72df`, and `telescope_runs_demo`'s `23326d63`, say "is not needed, and the helper never opens it in fresh mode". None of the four notebooks still contains "never opened" or "opened for writing". |
| IN-01 (`_disable_django_databases` docstring) | **Docstring: yes.** It now says the in-place switch only affects wrappers built afterwards, and that dropping the cached wrappers is what removes an open connection. The second half of IN-01 (a non-dict alias) and the matching logged warning text were not addressed; see IN-01 below. |

The override reason recorded by 40-12 now lists names, matching rules and the residual accurately for R12, except
for one claim: "any one-line cell whose first word is an IPython line magic or alias ... unless an earlier cell
binds that name". The code does not check that a name is bound. It checks that a store of the name appears in the
source (CR-01). R6's "an alias of it" and R7's "naming the developer database file" are also narrower in the code
than in the text (WR-01, WR-02).

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: The automagic rule treats any source-level store of a name as a binding, so an alias cell after a store that binds nothing at runtime is not reported

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:416-441` (`_module_bindings`), `:470` (`word in bound`), `:963-964` (R0 loop accumulating `bound`); module docstring `:30-32`; `docs/notebooks/README.md:97`; override reason in `40-VERIFICATION.md` front matter

**Issue:** `_automagic_word` lets a one-line cell pass when its first word is in `bound`. `bound` is filled by
`_module_bindings`, which collects every `Name` in `Store` context, import name, def/class name and except-handler
name anywhere outside function, class, lambda and comprehension bodies. That includes branches that never run. IPython
decides at runtime: `AutoMagicChecker.check` calls `is_shadowed(head, shell)`, which tests
`identifier in ip.user_ns` (IPython 9.10.1 source). Checked in a plain interpreter, each of these makes the guard
count `name` as bound while it is **not** in the namespace afterwards:

| Earlier cell | `_module_bindings` says bound | Bound at runtime |
|---|---|---|
| `name: int` (bare annotation, no value) | True | False |
| `if False:` / `name = 1` | True | False |
| `for name in []: pass` | True | False |
| `except OSError as name: pass` (Python deletes the name after the handler) | True | False |

A later `del name` is not taken into account either. So a notebook can put one of these statements on a default
IPython alias name (`cp`, `mv`, `rm`, `mkdir`, ...) in one cell and an alias command line in a later one-line cell.
IPython's automagic then runs the alias as a shell command, and R0 reports nothing. The third review showed that
the bare `cp` alias can copy the scratch file over `src/fomo_db.sqlite3`. I did not run this combination end to end
(see the method note). Each step was checked on its own, as above.

**Inside or outside the stated residual: outside.** Both cells are readable, literal notebook source. Nothing is
assembled or looked up at runtime (residual 1), the alias is one of IPython's default aliases, not one defined in a
profile or start-up file outside the notebook (residual 2), and no output HTML or JavaScript is involved
(residual 3). The guard docstring (lines 30-32), the README (line 97, "has no one-line cell that IPython would run
as a magic or an alias") and the override reason ("unless an earlier cell binds that name") all claim the
exemption is limited to names a cell binds. The code does not keep that claim. By the standard the previous rounds
used (a readable, literal route that the override says is reported), this keeps SC 2's first half open.

No real notebook needs the exemption: none of the eight has a one-line cell whose first word is an IPython magic or
alias name (checked with `_automagic_word(src, set())` on every one-line cell).

**Fix:** count only bindings that certainly exist when the later cell runs, and let everything else err towards
reporting, as the docstring already intends ("so the check errs towards reporting"):

```python
def _module_bindings(tree: ast.Module) -> set[str]:
    """Names that unconditional top-level statements bind, minus names a top-level ``del`` removes."""
    bound = set()
    for statement in tree.body:  # top level only: no if/for/while/try/with/match bodies
        if isinstance(statement, ast.Assign):
            bound.update(t.id for t in statement.targets if isinstance(t, ast.Name))
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None and isinstance(statement.target, ast.Name):
            bound.add(statement.target.id)
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            bound.update(a.asname or a.name.split('.')[0] for a in statement.names if a.name != '*')
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(statement.name)
        elif isinstance(statement, ast.Delete):
            bound.difference_update(t.id for t in statement.targets if isinstance(t, ast.Name))
    return bound
```

The R0 loop must also apply a later cell's `del` (as above, per cell, in order). Because no real notebook relies on
the exemption, the simpler alternative is to drop it entirely, except for an `=`/`,` right after the word. Add the
four non-binding forms from the table to `test_a_one_line_magic_or_alias_is_a_problem_unless_the_notebook_defines_the_name`.
Then reword the docstring, README line 97 and the override reason to "unless an earlier cell binds that name with a
top-level assignment, import, def or class".

## Warnings

### WR-01: R6 follows `shutil` only through `import shutil as X`, so a plain rebinding of the module hides `copytree` and the other copy functions

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:719-722` (`shutil_names`), `:727` (call check), `:739-743` (reference check); module docstring `:38-41`; `docs/notebooks/README.md:95`

**Issue:** `shutil_names` starts as `{'shutil'}` and only grows from `import shutil as <alias>`. Calls and
references are reported only when the base name is in that set. The inert check
`_byte_copy_problems` on placeholder source gave:

```
'import shutil as sh\nsh.copytree(a, b)\n'           -> ['code cell 1 calls sh.copytree(); ...']
'import shutil\nsh = shutil\nsh.copytree(a, b)\n'     -> []
```

`from <module> import shutil as s` (a module that re-exports shutil) is not followed either. The docstring says R6
reports these functions "called, referenced or imported, through ``shutil`` or an alias of it". The same docstring
says R12 reports a listed name "binding it to another name first", and the override reason sets R6 beside R12, so a
reader will expect the same coverage. The README (line 95) says the guard checks a notebook "never byte-copies or
byte-writes a file with shutil's copy functions, `move` or `write_bytes`".

**Inside or outside the stated residual:** the copy call itself is literal, and `copytree` is on R6's own list, so
residual 2 ("a name these lists do not contain") does not apply. As in the third review's WR-01, landing on the
developer database still needs a destination built from joined path parts (residual 1), which is why this is a
WARNING and not a BLOCKER.

**Fix:** match the distinctive names everywhere, the way R12 does, and keep only the ambiguous ones tied to shutil:

```python
DISTINCTIVE_COPY_NAMES = ('copy2', 'copyfile', 'copytree', 'copyfileobj')  # no real notebook uses any of them
# report any Name/Attribute/alias/keyword equal to one of these (as _identifiers does for R12);
# keep 'copy' and 'move' limited to shutil names, and grow shutil_names from
#   - `X = shutil` / `X = <shutil name>` assignments, and
#   - `from <anything> import shutil [as X]`
```

Add the two rebinding forms as cases in `TestR6CopyAndWriteFunctions`.

### WR-02: R5 and R7 read only `str` constants, so a bytes literal naming `fomo_db.sqlite3` or `FOMO_DATABASE_PATH` is not reported

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:694` (`isinstance(node.value, str)` in `_database_routing_problems`); module docstring `:37-38`

**Issue:** R12 and R6 read str and bytes constants through `_constant_text`, which 40-10 added. R5/R7 still test
`isinstance(node.value, str)`. Inert check:

```
_database_routing_problems on "x = b'fomo_db.sqlite3'"  -> []
_database_routing_problems on "x = 'fomo_db.sqlite3'"   -> ['code cell 1 names fomo_db.sqlite3 in code; only the helper may']
```

Python's file APIs take a bytes path, so the developer database's full path can be written literally without R7
seeing it. The docstring says R5-R7 report "code naming the database variable or the developer database file".

**Inside or outside the stated residual:** the naming claim fails outside the residual, because the path is a
literal, not assembled. Residual 2 already lists `open` in a write mode as a writer the guard does not name, though,
and R7 is what stops such a writer at a literal path. So this is a WARNING: the docstring overstates R7, and the
same `_constant_text` fix as 40-10 closes it.

**Fix:**

```python
for node in ast.walk(tree):
    text = _constant_text(node)          # str, or bytes decoded as latin-1
    if text is not None:
        if DB_ENV_VAR in text: ...
        if DEVELOPER_DB_NAME in text: ...
```

Add a bytes case next to `test_developer_database_literal_in_code_is_a_problem`.

## Info

### IN-01: Carried forward from the third review's IN-01: the logged clean-up warning still overstates, and a non-dict alias stops the switch loop

**File:** `docs/notebooks/fomo_nb_utils.py:208-210`, `:214-217`

**Issue:** (a) When `close_all()` fails, the warning logged says "every alias is already on the dummy backend, so no
database is reachable". As the corrected docstring now says, the switch does not stop a cached wrapper that already
holds an open connection. What removes it is the `del connections[alias]` loop that runs after the log line. 40-11
recorded this under Known Stubs and left it for Phase 41. (b) The second half of the third review's IN-01 was not
addressed. An alias whose value is not a dict raises `TypeError` at line 209. The caller logs and swallows it, so
later aliases are not switched and the close and drop steps never run. Django's own `configure_settings` would
refuse such settings at the first query, so nothing becomes reachable.

**Fix:** (a) change the message to "...; every alias is on the dummy backend and the cached wrappers are dropped
next, so no query made through Django reaches a database". (b) skip a non-dict alias:
`if isinstance(django_settings.DATABASES[alias], dict): ...`, or wrap each alias's switch on its own.

### IN-02: The guard depends on ipykernel and on IPython internals through the `jupyter` metapackage only

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:371-375`, `:402-406`; `pyproject.toml` `[project.optional-dependencies] dev`

**Issue:** R0 imports `ipykernel.zmqshell.KernelMagics`, `IPython.core.inputtransformer2.TransformerManager`,
`IPython.core.splitinput.split_user_input` and `IPython.extensions.storemagic`. `ipython` is a declared dev extra,
but `ipykernel` arrives only through the unpinned `jupyter` metapackage. If an install has no ipykernel, every
`isolation_problems()` call raises `ImportError`, so the whole module fails with errors rather than with a clear
message. The automagic name set also follows whatever IPython version is installed. The child-process test pins it
to a plain `InteractiveShell`, not to the ipykernel shell an nbconvert run uses.

**Fix:** add `ipykernel` to the `dev` extras with a comment naming this guard, and consider a one-line assertion in
`_ipython_line_magic_names()` that the set is non-empty and contains `cp`, `rm` and `mv`, so an IPython refactor
that empties a registry fails loudly.

---

_Reviewed: 2026-10-10T19:30:37Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
