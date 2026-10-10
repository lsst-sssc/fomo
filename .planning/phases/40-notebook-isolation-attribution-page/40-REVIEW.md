---
phase: 40-notebook-isolation-attribution-page
reviewed: 2026-10-10T15:36:37Z
depth: deep
files_reviewed: 12
files_reviewed_list:
  - docs/notebooks/README.md
  - docs/notebooks/fomo_nb_utils.py
  - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
  - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
  - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
  - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
  - docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb
  - docs/notebooks/pre_executed/telescope_runs_demo.ipynb
  - solsys_code/tests/test_notebook_helper.py
  - solsys_code/tests/test_pre_executed_notebooks.py
findings:
  critical: 2
  warning: 2
  info: 1
  total: 5
status: issues_found
---

# Phase 40: Code Review Report (third round: gap plans 40-08 and 40-09)

**Reviewed:** 2026-10-10T15:36:37Z
**Depth:** deep
**Files Reviewed:** 12
**Status:** issues_found

## Summary

Scope: the 12 listed files, compared with `269d8c5` (HEAD `7e1fdad`). That covers the extension of guard rules
R12 and R0 from database names to whole capabilities, the exception-safe fail-closed step in
`fomo_nb_utils.py`, the new helper tests, the README, and all eight notebooks re-executed on the new helper.

What I ran:
- `python manage.py test --noinput solsys_code.tests.test_notebook_helper solsys_code.tests.test_pre_executed_notebooks --parallel`:
  93 tests, OK.
- A guard probe (`$HOME/tmp/rv40/probe_guard.py`). It appends candidate source to the setup cell of a copy of
  `import_campaign_csv_demo.ipynb`, keeping the committed output, or inserts a new cell with the counts
  renumbered, and calls `isolation_problems()`. This is the same method as `TestReReviewFormsOnARealNotebook`.
- A runtime probe (`$HOME/tmp/rv40/runtime_probe.py`). It runs a real `setup_scratch_django(mode='fresh')` with
  `TMPDIR` under `$HOME/tmp`, runs each guard-clean form against a throwaway `other.sqlite3`, then runs the
  real teardown. Nothing opened `src/fomo_db.sqlite3`.
- IPython `InteractiveShell.transform_cell` / `run_cell` on dummy files under `$HOME/tmp/rv40/ipy`.

**Previous round, as of HEAD:**

| ID | Verified |
|----|----------|
| WR-01 (four re-review forms) | **Yes.** Forms (a)-(d) are reported, and `TestReReviewFormsOnARealNotebook` passes. The capability extension still leaves plain-source routes open; see CR-01 and CR-02. |
| WR-02 (fail-closed step) | **Yes.** Every alias is switched to the dummy backend before `close_all()`, the close is best-effort, and a clean-up error is logged while the original error is re-raised. All three `TestHelperFailClosedIsExceptionSafe` cases pass, including the case with no `default` alias. |
| IN-01 (refusal wording) | **Yes.** The README and the setup markdown of all eight notebooks now separate failures before the settings are read from failures after. |
| IN-02 (snapshot prose) | **Yes.** The NF-14 labels come from `NF14_EXPECTED_UNEXPLAINED` and agree with the `duplicate_identity=3` output. The PROJ-01 and PROJ-03 rows cite the real heading and the real `Projected events by stage marker:` label. |

The 40-09 override in `40-VERIFICATION.md` accepts only two kinds of gap: values assembled or looked up at
runtime, and code outside the notebook. CR-01 and CR-02 are neither. Each is readable, literal notebook source
that passes the whole extended guard. Five of the forms were run for real and reached a database other than the
scratch one, or overwrote a file, with the helper's teardown still printing its normal lines. Because the
override's stated basis does not hold, SC 2's first half is not met. Both are classified as BLOCKER for that
reason.

## Critical Issues

### CR-01: R12 misses plain-source process, connection and ATTACH routes: module names are not matched as names, the `connection` proxy is not listed, and `ATTACH (` passes

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:520-549` (`_database_rerouting_problems`), `:74-92` (`REROUTE_NAMES`), `:157` (`ATTACH_SQL`), `:28-43` (module docstring); `docs/notebooks/README.md:91-93`

**Issue:** The docstring (lines 39-42) says each R12 name is matched "wherever it appears -- as a name, an
attribute, a keyword, an imported name or an exact string constant". The override repeats that claim. It does
not hold for the module tables (`PROCESS_MODULES`, `DYNAMIC_CODE_MODULES`):

- Line 522 checks a `Name` against `_R12_NAME_TABLES` and `_R12_BUILTIN_TABLES` only.
- Line 532-533 checks a from-import alias against the same two tables only.
- Lines 544 and 548 check the module tables against the first component of the imported module only.

So `subprocess` is reported when it is imported directly or used as an attribute (`webbrowser.subprocess`). It
is not reported when it is imported from any module that re-exports it. Each of the following returned `[]`
from `isolation_problems()` on the real-notebook copy:

```python
# (a) a process, via a module that re-exports subprocess (no runtime assembly; run for real: child started
#     with env={'PATH': '/usr/bin'}, so it has no FOMO_DATABASE_PATH)
from webbrowser import subprocess            # also: from django.utils.autoreload import subprocess as sp
subprocess.run([sys.executable, 'manage.py', 'flush', '--noinput'], cwd=repo_root_path, env={'PATH': '/usr/bin'})

# (b) re-routing the ORM through the singular `connection` proxy. Run for real: Group.objects.create()
#     wrote its row to other.sqlite3
from django.db import connection
connection._connections.settings['default']['NAME'] = '/home/someone/other.sqlite3'
connection.close()

# (c) a new raw connection or a swapped connection without `sqlite3` or a listed name. Both run for real:
#     a row landed in other.sqlite3
connection.connection = connection.Database.connect('/home/someone/other.sqlite3', isolation_level=None)
dev = type(connection.connection)('/home/someone/other.sqlite3')

# (d) ATTACH with a parenthesised expression. Run for real: the INSERT landed in other.sqlite3
cursor.execute('ATTACH (%s) AS dev', [path])          # also "ATTACH('/x.sqlite3') AS dev"

# (e) a dynamic lookup by a literal (not assembled) string
from django.utils.module_loading import import_string
import_string('os.system')('...')                     # also pkgutil.resolve_name('subprocess:run')
```

Why each one passes:
- (b): `connection._connections.settings` has the dotted name `connection._connections.settings`. That does not
  end in `.connections.settings`, so `REROUTE_DOTTED_SUFFIXES` does not match it. `connection`, `_connections`
  and `Database` are on no list.
- (d): `ATTACH_SQL` requires whitespace followed by `database`, a quote or a parameter marker. SQLite takes any
  expression after `ATTACH`.
- (e): the string-constant check is an exact match. `'os.system'` is not `'system'`, and `'subprocess:run'` is
  not `'subprocess'`, although `sys.modules['subprocess']` is caught.
- Related: `IPython.utils.process.getoutputerror`, `get_output_error_code` and `process_handler` also pass. The
  synthetic tests cover the `getoutput` sibling, and the override text says "IPython's shell helpers" are
  reported.

None of these forms assembles a value at runtime or relies on code outside the notebook. README lines 91-93
("never sets Django up, chooses a database, starts a process ...") overstates what the guard checks, for the
same reasons.

**Fix:** match the module tables everywhere the name tables are matched, list the proxy names, widen ATTACH,
and add a real-notebook case for each form:

```python
# _database_rerouting_problems
if isinstance(node, ast.Name):
    note(node.id, _R12_NAME_TABLES, _R12_MODULE_TABLES, _R12_BUILTIN_TABLES)
...
elif isinstance(node, ast.alias):
    for name in (node.name.split('.')[-1], node.asname):
        note(name, _R12_NAME_TABLES, _R12_MODULE_TABLES, _R12_BUILTIN_TABLES)
elif isinstance(node, ast.Constant) and isinstance(node.value, str):
    for piece in re.split(r'[.:]', node.value):          # 'os.system', 'subprocess:run'
        note(piece, _R12_NAME_TABLES, _R12_MODULE_TABLES, _R12_BUILTIN_TABLES)

REROUTE_NAMES |= {'connection', '_connections', 'Database'}  # no real notebook references any of them
PROCESS_NAMES |= {'getoutputerror', 'get_output_error_code', 'process_handler'}
DYNAMIC_CODE_NAMES |= {'import_string', 'resolve_name', 'locate'}
ATTACH_SQL = re.compile(r'\battach\b', re.IGNORECASE)  # any ATTACH; the look-alike prose case still passes
                                                        # if limited to strings passed to execute()/executemany()
```

Then either correct the override reason in `40-VERIFICATION.md` or re-verify after the fix, and narrow README
lines 91-93 to what the guard actually reports.

### CR-02: R0 misses IPython aliases, automagic and shell escapes wrapped in a line magic, so a bare `cp` cell can overwrite the developer database

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:214-218` (`SHELL_ESCAPE`), `:221-231` (`UNPARSEABLE_SUSPECT_TEXT`), `:575-588` (R0 loop)

**Issue:** R0 only knows the escapes listed in `SHELL_ESCAPE`. IPython has more:
- Default aliases on Linux: `mkdir`, `rmdir`, `mv`, `rm`, `cp`, `cat`, `ls`, `ll`, `lf`, `lk`, `ldir` and `lx`
  (checked with `IPython.core.alias.default_aliases()`).
- `automagic` is on by default.
- `%alias` and `%rehashx` define new shell aliases.
- `%time` and `%timeit` transform their argument, so `%time !cmd` runs a shell escape that `^\s*!` does not
  see.

Each of these inserted cells returned `[]` on the real-notebook copy:

```
cp /tmp/fomo-notebook-db-abc123/* ../../src/
%cp /tmp/fomo-notebook-db-abc123/* ../../src/
rm -r ../../src
%time !rm -r ../../src
%alias wipe rm -r ../../src
%rehashx
%pip install -e ../..
```

`InteractiveShell.transform_cell('cp scratch/* src/')` gives `get_ipython().run_line_magic('cp', ...)`. On dummy
files, `run_cell('cp scratch/* src/')` and `run_cell('%time !cp scratch/fomo_db.sqlite3 src/')` each replaced
`src/fomo_db.sqlite3` with the scratch copy. That is a byte copy over the developer database. The file name is
never written, nothing is assembled at runtime, and everything is in the notebook. The docstring (lines 22-26)
says R0 reports any unparseable cell that mentions "a process, a shell command or the environment".

**Fix:** none of the eight real notebooks has an unparseable code cell (checked with `ast.parse` on every code
cell). So replace the R0 deny-list with an allow-list: an unparseable cell is a problem unless removing a small
set of harmless magic lines leaves parseable Python.

```python
HARMLESS_MAGIC_LINE = re.compile(r'^\s*%(?:matplotlib|config)\b.*$|^\s*%%(?:time|capture)\b.*$', re.MULTILINE)
...
if tree is None:
    stripped = HARMLESS_MAGIC_LINE.sub('', source)
    if _parse(stripped) is None or SHELL_ESCAPE.search(source) or ...:
        problems.append(...)
    else:
        parsed[...] = ast.parse(stripped)   # so R5-R12 also check the code under %%time
```

Keep the existing `%%time` look-alike test passing, and add the seven cells above as synthetic cases.

## Warnings

### WR-01: R6 lists only `copy`, `copy2` and `copyfile`, so `shutil.copytree` or `write_bytes` byte-copies over the developer database without naming it

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:185` (`FORBIDDEN_COPY_NAMES`), `:446-464` (`_byte_copy_problems`)

**Issue:** Each of the following passes the guard on the real-notebook copy. The developer database's file name
never appears and nothing is looked up:

```python
import shutil
shutil.copytree(scratch_db_dir, repo_root_path / 'src', dirs_exist_ok=True)
# or
for f in scratch_db_dir.iterdir():
    (repo_root_path / 'src' / f.name).write_bytes(f.read_bytes())
```

The scratch file has the same base name as the developer database (`SCRATCH_DB_NAME`), so both statements
overwrite `src/fomo_db.sqlite3`. I checked the `copytree` overwrite on dummy files. The guard docstring
(line 28) and the README (line 93) say no notebook byte-copies a database.

This is a WARNING, not a BLOCKER, because the destination directory is built by joining path parts. That sits
next to the "joined path parts" wording in the accepted residual. The developer database's own path is never
built, though.

**Fix:** add `copytree`, `copyfileobj` and `move` to `FORBIDDEN_COPY_NAMES`, and report any `scratch_db_dir`
reference outside the setup and teardown statements, because no notebook needs the scratch directory except to
pass it to the teardown. As an alternative, report any path expression that ends in a `'src'` constant joined
onto `repo_root_path`.

### WR-02: three notebooks still publish the unmeasured "developer database was never opened" claim that D-G2 withdrew

**File:** `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` (teardown markdown `46602df0`); `campaign_lifecycle_demo.ipynb` (setup markdown `7dfef438`, teardown markdown `234805b6`); `load_telescope_runs_demo.ipynb` (setup markdown `49ff72df`, teardown markdown `31ff3edb`)

**Issue:** D-G2 replaced the teardown's hard-coded "never opened for writing" line with a statement of what the
helper checked. 40-09 moved five notebooks' teardown markdown to that statement (T1). The 40-09 interface table
listed no teardown markdown cell for these three notebooks, but each has one, unchanged since `269d8c5`:

- `46602df0`, `234805b6` and `31ff3edb` end with "The developer database was never opened by this notebook run."
- The setup cells `7dfef438` ("the developer database is never opened at all (UAT G-33-4)") and `49ff72df`
  ("is never opened at all (T-35-16)") make the same claim.

The teardown output cell right below each of the three teardown markdown cells prints the new "Before
django.setup() ran, the helper checked ..." line. So the prose and the output beside it now make different
claims, which the 40-09 human check (plan line 434) says must not happen. The plan's "no old wording or claim in
any cell" check missed these cells because it matched only the literal "opened for writing". CR-01 and CR-02
show that nothing measures this claim.

**Fix:** in the three teardown markdown cells, replace the last sentence with T1, as was done for the other
five. In `7dfef438` and `49ff72df`, change "is never opened at all" to "is not needed, and the helper never opens
it in fresh mode". Then broaden the 40-09 style check to `never opened` / `never .* writes the developer`.

## Info

### IN-01: `_disable_django_databases` docstring overstates what the in-place switch does on its own

**File:** `docs/notebooks/fomo_nb_utils.py:190-195, 205-208`

**Issue:** The docstring says "the switch alone leaves no real database reachable and a close that fails
afterwards cannot undo it". That is not true for a wrapper that already holds an open connection, for example
after a failure partway through `migrate`. `BaseDatabaseWrapper.ensure_connection()` reconnects only when
`self.connection is None`, so a cached wrapper whose close failed keeps using its open sqlite3 connection
whatever its `settings_dict` now says. What actually removes that route is the `del connections[alias]` loop,
which always runs. `test_a_failing_connection_close_still_disables_every_alias_and_keeps_the_original_error`
does not exercise this case, because its refusal happens before any connection is opened. There is a second,
smaller gap: an alias whose value is not a dict (`None`) raises `TypeError` in the switch loop at line 207.
Aliases later in the loop are then not switched, although Django's own `configure_settings` refuses that
configuration anyway.

**Fix:** reword the docstring to "the switch makes every new wrapper a dummy, and dropping the cached wrappers
removes any connection that is still open". Optionally, add a child case that opens `connections['default']`,
forces `migrate` to fail with `close_all` patched to raise, and asserts that the next query is refused.

---

_Reviewed: 2026-10-10T15:36:37Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
