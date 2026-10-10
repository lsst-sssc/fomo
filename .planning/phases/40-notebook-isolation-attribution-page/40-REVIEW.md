---
phase: 40-notebook-isolation-attribution-page
reviewed: 2026-10-10T05:37:47Z
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
  critical: 0
  warning: 2
  info: 3
  total: 5
status: issues_found
---

# Phase 40: Code Review Report (re-review after gap-closure plans 40-05, 40-06, 40-07)

**Reviewed:** 2026-10-10T05:37:47Z
**Depth:** deep
**Files Reviewed:** 12
**Status:** issues_found

## Summary

Scope: the 12 files listed, compared with `e129b4ff5ff1901696549f22ada2b0bc8c851044`. That covers the
fail-closed rework of `docs/notebooks/fomo_nb_utils.py`, guard rule R12 in
`solsys_code/tests/test_pre_executed_notebooks.py`, the new helper tests, the README, and all eight notebooks
re-executed on the new helper.

What I ran:
- `python manage.py test solsys_code.tests.test_notebook_helper solsys_code.tests.test_pre_executed_notebooks --parallel`:
  77 tests, OK.
- Subprocess probes of the real helper and of `isolation_problems()`, each in a scratch `TMPDIR`. None of them
  opened `src/fomo_db.sqlite3` for writing. The only read was a `mode=ro` check that the developer database's
  journal mode is `delete`, so the `mode=ro` snapshot needs no `-shm` file.
- Notebook source diffs compared with the diff base (prose and code only, outputs left out), and a read of the
  committed output of every snapshot-dependent cell.

**What the previous round's "fixed" items look like now:**

| ID | Ledger | Verified |
|----|--------|----------|
| CR-01 | fixed | **Yes.** `check_database_settings()` runs on the lazy settings before `django.setup()` (`fomo_nb_utils.py:295-300`). After a refusal, `apps.ready` stays `False` and every alias is a dummy. A failure in migrate also leaves only dummy aliases. A second setup is refused. `TestHelperRefusalLeavesNoDatabase` proves all of this in child processes. There is one exception-safety gap in the disable step (WR-02 below). |
| WR-01 | fixed | **Partly.** R12 catches both verifier probes and every spelling it lists. It misses an ORM re-route through a fresh `ConnectionHandler` wrapper, and a child process started with a cleared environment. I reproduced both (WR-01 below). |
| WR-02 | fixed | **Yes** for the configured case: an extra alias or any `DATABASE_ROUTERS` entry is refused before setup. The teardown line is unchanged, by the developer's decision. |
| IN-01 | fixed | **Yes.** `_restore_environ()` runs on the failure path (after the disable step, which needs the async opt-in) and in teardown. `test_teardown_restores_the_routing_environment` covers it. |
| IN-02 | fixed | **Yes** for the helper: the evidence line names only the settings module. As the ledger records, `campaign_lifecycle_demo` still publishes `/home/tlister/venv/...` through spiceypy's stderr line. |
| IN-05 | fixed | **Yes.** All eight `NamedTemporaryFile` calls in `load_telescope_runs_demo` pass `dir=scratch_db_dir`, and the committed output shows the files inside the scratch directory. |
| IN-06 | fixed | **Yes.** Teardown checks the name prefix, that the directory is directly inside the resolved temp directory, and that it matches `_active_setup`. All three checks run before Django is imported. |
| IN-07 | deferred in part | The helper-test part is done: `re.escape(tmp)` in both regexes, a real snapshot subprocess test, and a forced snapshot-failure test. The `campaign_views.py` alias part is outside this round's file scope. |

WR-03, IN-03 and IN-04 from the previous round concern `test_bootstrap5_rendering.py`, `campaign_views.py`,
`campaign_tables.py` and the attribution template. Those files are outside this round's scope, so they stay
`open` in the ledger and are not repeated here.

No new critical issue. The two warnings are a guard blind spot that the module docstring does not admit, and a
disable step that is not exception-safe.

## Warnings

### WR-01: R12 misses a database re-route through a new `ConnectionHandler` wrapper and through a child process with a cleared environment; the teardown still prints "never opened for writing"

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:32-34, 57-74, 116-125, 361-393`; `docs/notebooks/fomo_nb_utils.py:364`
**Issue:** R12 is a deny-list of names. It has no entry for Django's own `ConnectionHandler`, for a store into
`connections[...]`, or for starting a process that runs `manage.py`. I appended each of the following to the
setup cell of a copy of `import_campaign_csv_demo.ipynb`, with its committed output kept, and
`isolation_problems()` returned `[]` every time:

```python
# (a) ORM re-route without any listed name
from django.db import connections
from django.db.utils import ConnectionHandler
dev = str(repo_root_path / 'src' / ('fomo_db' + '.sqlite3'))
connections['default'] = ConnectionHandler({'default': {'ENGINE': 'django.db.backends.' + 'sqlite' + '3', 'NAME': dev}})['default']

# (b) child process that no longer inherits the scratch FOMO_DATABASE_PATH
import subprocess
subprocess.run([sys.executable, 'manage.py', 'migrate'], cwd=repo_root_path, env={'PATH': '/usr/bin'})

# (c) same through os.environ.clear()
import os, subprocess
os.environ.clear()
subprocess.run([sys.executable, str(repo_root_path / 'manage.py'), 'flush', '--noinput'])
```

The IPython shell magic `!cd ../.. && env -i PATH=/usr/bin python manage.py flush --noinput` also gets through
R0, because `manage.py` and `env -i` are not in `UNPARSEABLE_SUSPECT_TEXT`.

I ran (a) for real in a child process after a successful `setup_scratch_django()`, with `NAME` pointed at a
throwaway file. `Group.objects.create(...)` wrote its row to that file, and the teardown then printed
`The developer database (src/fomo_db.sqlite3) was never opened for writing.` R10 requires that line as evidence,
but nothing measures it.

The module docstring's "What the guard does not see" list (lines 32-34) names `exec`/`eval`/`compile`, computed
`getattr`/`importlib`, code inside imported modules, objects bound to another name, and what libraries do. None
of those covers (a), (b) or (c), which are plain, readable Python. The 40-06 summary's claim that R12 "covers
every database re-routing spelling" is therefore too strong. This was the same kind of gap WR-01 was opened for.

**Fix:** extend R12 and R0, add one synthetic case for each, and make the docstring list what is still out of
reach:

```python
REROUTE_NAMES = frozenset({..., 'ConnectionHandler', 'create_connection', 'configure_settings'})
# R12: a store into the connection handler replaces the wrapper the ORM uses.
elif isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
    targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
    for target in targets:
        if isinstance(target, ast.Subscript) and _dotted_name(target.value).split('.')[-1] == 'connections':
            found.add('connections[...] assignment')
# R12: a child process can drop the scratch FOMO_DATABASE_PATH.
SUBPROCESS_NAMES = {'subprocess', 'Popen', 'system', 'spawnv', 'execv'}   # import or call -> problem
UNPARSEABLE_SUSPECT_TEXT = (..., 'manage.py', 'env -i', 'env -u')
```

As an alternative, or in addition, reword the teardown's second line to say what was checked. For example:
"Django's only configured database was the scratch file above." That claim the helper can stand behind.

### WR-02: The fail-closed step is not exception-safe; with no `default` alias it masks the helper's `RuntimeError` and never switches the aliases to the dummy backend

**File:** `docs/notebooks/fomo_nb_utils.py:184-207` (called from `306-314`)
**Issue:** `_disable_django_databases()` calls `connections.close_all()` (line 199) **before** its loop that
switches each alias to `django.db.backends.dummy` (lines 200-207). `close_all()` evaluates
`ConnectionHandler.settings`, which raises `ImproperlyConfigured("You must define a 'default' database.")` when
`DATABASES` has no `default` key. That exception escapes the `try` at line 309. The swap loop never runs, and the
helper's own `RuntimeError` survives only as `__context__`. I reproduced it with an override settings module
whose `DATABASES = {'primary': {...sqlite3...}}`:

```
RAISED ImproperlyConfigured You must define a 'default' database.
CAUSE/CONTEXT RuntimeError A notebook scratch setup supports only the 'default' database, but the settings also define 'primary'; ...
DATABASES after: {'primary': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': '.../primary.sqlite3'}}
```

In this exact case no database is reachable, because Django refuses every connection while `default` is
missing. But that comes from Django's own check, not from the helper. The guarantee the docstring
(lines 263-267) and the README (lines 66-68) state ("switches every database alias to Django's dummy backend",
"raises `RuntimeError`") does not hold. Any other exception from `close_all()` would also skip the swap. For
example, `SynchronousOnlyOperation` from `@async_unsafe close()` when a user has exported
`DJANGO_ALLOW_ASYNC_UNSAFE=''` (which `setdefault` keeps and Django treats as unset) and migrate fails in
Jupyter. The swap is the part that makes the helper fail closed, so it should not depend on the close step
succeeding first.

**Fix:** switch the settings first, close afterwards, and never let the clean-up replace the original error:

```python
def _disable_django_databases():
    from django.conf import settings as django_settings
    from django.db import connections

    if not django_settings.configured:
        django_settings.configure(DATABASES={'default': {'ENGINE': 'django.db.backends.dummy'}})
    # Switch first: cached wrappers hold these same dicts, so this alone makes every one of them unusable.
    for alias_settings in django_settings.DATABASES.values():
        alias_settings['ENGINE'] = 'django.db.backends.dummy'
        alias_settings['NAME'] = ''
    try:
        connections.close_all()  # best effort; every alias is already a dummy
    except Exception:
        pass
    for alias in list(django_settings.DATABASES):
        try:
            del connections[alias]
        except AttributeError:
            pass
```

Wrap the call at line 310 so that a clean-up error is logged and the original exception is re-raised. Add a
`TestHelperRefusalLeavesNoDatabase` case with a settings module that has no `default` alias, and assert that the
child sees `RuntimeError`.

## Info

### IN-01: README and all eight notebooks say a failed setup leaves no reachable database and cannot be re-run, which is not true for failures before the settings are read

**File:** `docs/notebooks/README.md:66-68, 77-78`; the setup markdown cell of every notebook under `docs/notebooks/pre_executed/` (for example `telescope_runs_demo.ipynb` cell 1, `reconcile_campaign_runs_demo.ipynb` cell 1 item 2)
**Issue:** the prose says "after a refusal, or any later failure in the setup, no cell can reach a database until
the kernel is restarted". The README also says that re-running the setup cell "where a setup already failed,
raises `RuntimeError`". Several failures happen in `prepare_scratch_database()` (`fomo_nb_utils.py:281`),
before the `try` and before the settings are read: snapshot mode with no `src/fomo_db.sqlite3`, an unknown
mode, a `mkdtemp` failure, or a backup error. These leave Django unconfigured, do not run the disable step, and
allow a second setup call. That behaviour is fine, since nothing was routed and a retry is safe. Only the prose
overstates it. The function docstring says it precisely ("Once the settings are loaded, any failure ...").
**Fix:** say "after a refusal of the database settings, or any failure once Django's settings are loaded, ..."
and "re-running the setup cell after Django's settings were loaded raises `RuntimeError`".

### IN-02: Snapshot-notebook prose that does not match the committed output

**File:** `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` (code cell 5, the NF-14 cell, lines `print(f'dry run CommandError (expected -- pk=334 plus the duplicate-identity group): ...')` and its two siblings); `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` (code cell 13, the requirement table)
**Issue:** the 40-07 commit `304022a` says the snapshot notebooks' "data-dependent prose match[es] the new
output", but some does not:
- The three NF-14 `print` labels hard-code "expected -- pk=334 plus the duplicate-identity group". The committed
  output on the same lines reads `3 event(s) could not be explained ... (duplicate_identity=3)`, and the BEFORE
  table shows `blank_url: 0`, so the snapshot holds no pk=334. The published output contradicts itself.
- The requirement table cites `cells under "Per-corpus reconciliation"` (PROJ-01) and `"Per-marker tally" cell`
  (PROJ-03). Neither heading exists. The sections are "The whole real corpus, not a sample" and an unheaded tally
  cell.

Both predate this phase, but both notebooks were re-executed and their prose re-checked in this round.
**Fix:** build the NF-14 label from the data, for example `expected -- the duplicate-identity group` plus any
pre-existing unexplained pks found in the BEFORE snapshot, and point the two table rows at the real headings.

### IN-03: Two branches of the fail-closed path have no executed test

**File:** `solsys_code/tests/test_notebook_helper.py` (no case for `fomo_nb_utils.py:197-198` or for the `configured`-only branch of `275`)
**Issue:** `_disable_django_databases()`'s `if not django_settings.configured: configure(dummy)` branch runs when
the settings module itself fails to import, for example a broken `local_settings.py`. No test reaches it. My
first probe hit it by accident (`ModuleNotFoundError: No module named 'src'`) and it behaved correctly, but a
regression there would not be caught. Likewise, the refusal when settings are configured but `apps.ready` is
`False` is tested only indirectly, through the second-setup probe after a refusal.
**Fix:** add a child-process case with `DJANGO_SETTINGS_MODULE` naming a module that raises on import. Assert
that the error propagates, that `connections['default'].cursor()` raises `ImproperlyConfigured`, and that a
second setup is refused.

---

_Reviewed: 2026-10-10T05:37:47Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
