---
phase: 40-notebook-isolation-attribution-page
reviewed: 2026-10-10T23:44:46Z
depth: deep
files_reviewed: 3
files_reviewed_list:
  - docs/notebooks/README.md
  - pyproject.toml
  - solsys_code/tests/test_pre_executed_notebooks.py
findings:
  critical: 2
  warning: 2
  info: 2
  total: 6
status: issues_found
---

# Phase 40: Code Review Report (fifth round: gap plans 40-13 and 40-14)

**Reviewed:** 2026-10-10T23:44:46Z
**Depth:** deep
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Scope: the 3 listed files, compared with `42dc508` (HEAD `51da348`). That covers 40-13 (`905408a`: the automagic
rule no longer exempts names; `4f02628`: R6 distinctive names, notebook-wide shutil names, R5/R7 bytes constants;
`ccfc578`: `ipykernel>=7`) and 40-14 (`1a315e2`: the guard docstring and the README guard paragraph). I also read
the override reason in `40-VERIFICATION.md` (`overrides[0].reason`) for the claims it makes.

What I ran:
- `python manage.py test --noinput solsys_code.tests.test_pre_executed_notebooks`: 86 tests, OK.
- `pre-commit run ruff` and `ruff-format` on the guard module: both pass.
- Static calls of the guard's own functions (`isolation_problems`, `is_django_backed`, `_automagic_word`,
  `_ipython_line_magic_names`). Each ran on placeholder source or on a deep copy of
  `import_campaign_csv_demo.ipynb` with source appended to its setup cell (`_append_to_setup_cell`). The module was
  loaded by file path from `$HOME/tmp/rv40r5`.
- Three child processes, each with its own `IPYTHONDIR`. One listed the line magics of ipykernel's
  `ZMQInteractiveShell`. Two called `InteractiveShell.transform_cell` on one-line sources, which rewrites text and
  runs none of it.
- Plain Python checks with stand-in objects to show what `match` class patterns and call arguments bind at runtime.
  Nothing was written to any file.
- No notebook was executed, `django.setup()` was not called, `src/fomo_db.sqlite3` was not opened and
  `FOMO_DATABASE_PATH` was not set.

**Previous round, as of HEAD:**

| ID | Closed? |
|----|---------|
| CR-01 (automagic exemption trusted source-level bindings) | **Yes.** `_module_bindings` is gone, and `_automagic_word(view)` exempts only a builtin, a keyword, or a word followed by `=` or `,`. A child-shell `transform_cell` check shows IPython keeps `rm = 3`, `rm=3`, `rm, cp = 1, 2` and `rm == 3` as Python, and rewrites `rm += 3`, `cp(a, b)` and `rm -r -- 'x'` to `run_line_magic`. The guard reports exactly the rewritten three. No magic name is also a builtin in the test process. |
| WR-01 (R6 followed shutil only through `import shutil as X`) | **Yes for both review forms**, plus the assignment-shaped bindings 40-13 lists. The stated boundary does not cover every binding the code skips. See CR-01 and CR-02. |
| WR-02 (R5/R7 ignored bytes constants) | **Yes.** `_database_routing_problems` reads constants through `_constant_text`. |
| IN-01 (helper clean-up warning) | Out of this round's file scope (developer decision, Phase 41). |
| IN-02 (ipykernel only through `jupyter`) | **Yes.** `"ipykernel>=7"` is in the dev extra. Installed 7.3.0 declares `Requires-Python >=3.10`, so CI's 3.10 leg can resolve it. The child-process test checks that `cp`, `rm`, `mv` and the rest are present. |

The docstring and README now match the code for R0's automagic rule, for R5/R7 and for the review forms of R6.
They overstate R6 in two places (CR-01, CR-02). The docstring also overstates which rules apply under a harmless
magic (WR-01) and what "IPython runs unchanged" covers (WR-02).

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: R6 does not follow shutil into a function parameter, a lambda, a `match` capture or a class-pattern attribute, and the stated boundary does not name these routes

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:722-743` (`_binding_pairs`), `:760-792` (`_shutil_names`), `:826-857` (the `copy`/`move` checks in `_byte_copy_problems`). Claims: module docstring `:49-54`, `_shutil_names` docstring `:770-771`, `_byte_copy_problems` docstring `:811-813`, `docs/notebooks/README.md:97-98`, and the same boundary in the override reason.

**Issue:** `copy` and `move` are reported only when they appear as an `ast.Attribute` whose value mentions a shutil
name. Shutil names grow only from imports and from `_binding_pairs`: assignments, loop, comprehension and `with`
targets, and parameter **defaults**. A parameter that receives shutil as a call **argument** is never added. Neither
is a `match` capture, and `copy` read through a class-pattern attribute is not an `ast.Attribute` at all. Every row
below was appended to the setup cell of a copy of `import_campaign_csv_demo.ipynb`, and `isolation_problems()`
returned `[]` for each (control: the parameter-default form returns `calls sh.move()`):

| Form (literal source) | Reported |
|---|---|
| `import shutil` / `def put(sh):` / `    for f in scratch_db_dir.iterdir():` / `        sh.copy(f, repo_root_path / 'src')` / `put(shutil)` | No |
| `(lambda sh: [sh.copy(f, repo_root_path / 'src') for f in scratch_db_dir.iterdir()])(shutil)` | No |
| `list(map(lambda sh: sh.move(...), [shutil]))` | No |
| `def put(*, sh): sh.move(...)` / `put(sh=shutil)` | No |
| `match shutil:` / `    case sh:` / `        ... sh.copy(f, repo_root_path / 'src')` | No |
| `match shutil:` / `    case object(copy=c):` / `        ... c(f, repo_root_path / 'src')` | No |

A plain Python check confirms that the parameter and `case object(copy=c)` both bind `shutil.copy`. The scratch
file is named `fomo_db.sqlite3`, so the first form copies it over `src/fomo_db.sqlite3` through a literal directory
path. That is the same landing the verifier used against round 4's WR-01.

**Inside or outside the stated residual: outside.** R6's boundary, in the docstring (`:52-54`) and in the override
reason, lists three exceptions: a function's return value, a call's side effect (`setattr`), and a module that
holds shutil under another name. A call argument bound to a parameter, a `match` capture and a class-pattern
attribute are none of these. Nothing here is assembled at runtime (residual 1). `copy` and `move` are on R6's own
list (residual 2). The README is broader still. It says `copy` and `move` are reported "when they are read from
shutil or from a name the notebook binds to it", and a parameter or capture is such a name. It also says the
docstring "says exactly what each rule follows", which this table contradicts.

**Practical weight, for the developer's decision:** residual (2) already accepts `Path.replace`, `rename` and
`open` in a write mode. So `f.replace(repo_root_path / 'src' / f.name)` gives the same effect today without
naming anything. This finding adds no new capability. It is a false claim in the guard's documentation, which is
the standard rounds 3-5 have applied.

**Fix:** no real notebook mentions `shutil` (`grep -c shutil` gives 0 in all eight). The simplest complete fix is to
report the module itself wherever it is named, as R12 does for `subprocess`. That also closes the three accepted
exceptions:

```python
# R6: a notebook never needs shutil (the helper does every copy), so the module itself is reported
# at every identifier and string part, like the distinctive copy names.
DISTINCTIVE_COPY_NAMES = ('copy2', 'copyfile', 'copytree', 'copyfileobj', 'shutil', '_shutil')
```

After that, `_shutil_names`, `_binding_pairs`, `_target_names` and `_mentions_shutil` are needed only to word the
`copy`/`move` message. Add the six forms above as cases in `TestR6CopyAndWriteFunctions`. Then shorten the
docstring, the README (`:96-98`) and the override reason to "any reference to shutil". The alternative is to keep
the code and add "a parameter given shutil as a call argument, a `match` capture, or a class-pattern attribute"
to the boundary in all three places. Either way, remove "exactly" from README line 101 or make it true.

### CR-02: "any `write_bytes`" is not what R6 checks, and a class-pattern attribute writes the scratch bytes over the developer database unreported

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:858-863` (only `ast.Attribute`/`ast.Name` are matched) and `:864-870` (string parts). Claims: module docstring `:47` ("any ``write_bytes``"), `_byte_copy_problems` docstring `:808` ("any reference to a byte-write method"), `docs/notebooks/README.md:96` ("never byte-copies or byte-writes a file with ... `write_bytes`").

**Issue:** the distinctive copy names are matched at every identifier `_identifiers` yields, and that includes
`ast.MatchClass.kwd_attrs`. `write_bytes` is matched only as a `Name`, an `Attribute` or a string part. A class
pattern looks up an attribute by an identifier that is none of these. Appended to the setup cell of a copy of
`import_campaign_csv_demo.ipynb`, this gives `[]`:

```python
for f in scratch_db_dir.iterdir():
    match repo_root_path / 'src' / f.name:
        case object(write_bytes=w):
            w(f.read_bytes())
```

Control: the same loop with `(repo_root_path / 'src' / f.name).write_bytes(f.read_bytes())` returns `references
write_bytes`. A Python check with a stand-in object confirms that `case object(write_bytes=w)` binds the method.
Since `f.name` is `fomo_db.sqlite3`, the loop overwrites `src/fomo_db.sqlite3` byte for byte. No boundary is stated
for `write_bytes` anywhere: not in the docstring, the README or the override reason.

**Inside or outside the stated residual: outside**, for the same reasons as CR-01: literal source and a name on
R6's own list. The same practical weight applies, because `open` in a write mode is already accepted under residual (2).

**Fix:** match `BYTE_WRITE_NAMES` through `_identifiers`, the same way as `DISTINCTIVE_COPY_NAMES`. No real notebook
uses the name anywhere:

```python
for scanned in scanned_trees:
    for node in ast.walk(scanned):
        for identifier, _groups in _identifiers(node):
            if identifier in (*DISTINCTIVE_COPY_NAMES, *BYTE_WRITE_NAMES) and identifier not in named:
                ...
```

Add the class-pattern form, `def f(write_bytes=...)` and `f(write_bytes=...)` to
`test_every_copy_and_write_function_in_any_spelling_is_a_problem`.

## Warnings

### WR-01: `is_django_backed` reads the raw source, not the Python view, so Django code under a harmless cell magic can switch off R2-R4 and R8-R11

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:586-588` (`is_django_backed` parses `_cell_source`), `:1093` (`if not is_django_backed(notebook): return problems`). Claims: module docstring `:39-40` ("The code under a harmless magic is checked by every later rule"), `isolation_problems` docstring `:1037-1044`.

**Issue:** `isolation_problems` builds the Python views and passes them to every rule. The gate that decides
whether R2-R4 and R8-R11 run, though, calls `is_django_backed(notebook)`, which parses each raw cell. A `%%capture`
or `%%time` cell does not parse raw, so its Django imports are not seen. The unparseable-cell deny-list misses an
import that is not at the start of a line or not the first module named (`UNPARSEABLE_DJANGO_IMPORT` is
`^\s*(from|import)\s+(django|...)`). Probe on placeholder source:

```
nb('%%capture\nx = 1; import django as dj\ndj.setup()\n')
  is_django_backed -> False ; isolation_problems -> []
nb('x = 1; import django as dj\ndj.setup()\n')            # same code, no magic
  -> ['... imports django under an alias ...', 'no top-level `scratch_db_dir = setup_scratch_django(...)` ...']
```

So the function the docstring describes does not check the code under a harmless magic with every later rule.
R4 (own `django.setup`), R2 (helper call) and R3 (Django import before setup) are all skipped.

**Why WARNING:** in this repository, `test_the_real_notebook_check_is_not_vacuous` requires the eight notebooks to
be exactly `EXPECTED_NOTEBOOKS`, and each must be `is_django_backed` and have a raw `setup_call`. A real notebook
therefore always passes the gate, and the views-based rules then catch the code under the magic (R4 reports
`import django as dj`). The suite as a whole holds. The guard function and its docstring do not, and a future
caller of `isolation_problems` (or a change to the vacuity test) would lose these rules without any sign.

**Fix:** decide whether the notebook is Django-backed from the views that every other rule reads:

```python
    parsed = views
    ...
    if not any(tree is not None and _django_imports(tree) for _number, _cell, tree in parsed):
        return problems
```

Add the probe above to `TestR0AllowList.test_code_under_a_harmless_magic_is_checked_by_every_rule`, expecting the
alias and no-helper-call problems.

### WR-02: R0 does not see IPython's autocall of an `IPyAutocall` object, so "Python that IPython runs unchanged" overstates the check

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:432-464` (`_automagic_word` handles only automagic). Claims: module docstring `:23-24`, `isolation_problems` docstring `:1041-1042`, `docs/notebooks/README.md:98`.

**Issue:** IPython's prefilter has more dynamic rewrites than automagic. `IPyAutocallChecker` turns a one-line cell
that starts with an `IPyAutocall` instance into a call. In a child shell, `transform_cell('exit -1\n')` returned
`exit(-1)\n` (the same for `quit`). The guard returns `[]` for `nb('exit -1\n')`, and the same holds for a
user-defined `IPyAutocall` subclass instance. `IPyAutocall` is not on any R12 list.

**Impact:** no database route that I could find. `exit` and `quit` only end the kernel, and a user-defined autocall
object's `__call__` body is ordinary notebook code that every rule reads. The finding is about the claim. The
guard does not establish that every cell is "Python that IPython runs unchanged".

**Fix:** add `'IPyAutocall'` to `DYNAMIC_CODE_NAMES`, so a notebook cannot define new autocall objects. Then either
report a one-line cell whose first word is `exit` or `quit` and that has more after it, or name the exception in the
docstring and README ("apart from IPython's autocall of `exit` and `quit`").

## Info

### IN-01: The README and the docstring leave out checks the code makes

**File:** `docs/notebooks/README.md:92-103`; module docstring `:40-42`

**Issue:** the README guard paragraph does not mention the R5 keyword-argument check (`FOMO_DATABASE_PATH=`), R10
(the last cell must be the teardown of the same directory, with its output), R8's error-output check, or R11 (no
`assert` in the setup cell or wrapping a write). The docstring sums up R11 as "no ``assert`` guarding a write", but
the code (`:1181-1183`) also reports any `assert` in the setup cell. The README points readers to the docstring, so
this is a completeness note rather than a defect.

**Fix:** add "ends with the teardown and has no error output or `assert` in the setup cell" to the README sentence,
and "and no ``assert`` in the setup cell" to the docstring's R11 summary.

### IN-02: The automagic name set is pinned against a plain `InteractiveShell`, not the ipykernel shell that nbconvert runs

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:2410-2437`

**Issue:** the child-process test compares `_ipython_line_magic_names()` with a plain `InteractiveShell` plus
storemagic. Notebooks are regenerated under ipykernel's `ZMQInteractiveShell`. Today the two match: in a child
process, `ZMQInteractiveShell.instance()` with storemagic loaded has exactly the guard's set (no difference in
either direction; IPython 9.10.1, ipykernel 7.3.0), and it builds without a kernel. A future ipykernel that
registers another line magic would go unnoticed.

**Fix:** in the child script, import `ipykernel.zmqshell.ZMQInteractiveShell` and use it instead of
`InteractiveShell`, keeping the subset assertion.

---

_Reviewed: 2026-10-10T23:44:46Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
