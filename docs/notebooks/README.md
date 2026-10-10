# Jupyter notebooks to run on-demand.

Jupyter notebooks in this directory will be run each time you render your documentation.

This means they should be able to be run with the resources in the repo, and in various environments:

- any other developer's machine
- github CI runners
- ReadTheDocs doc generation

This is great for notebooks that can run in a few minutes, on smaller datasets.

If you would like to include these notebooks in automatically generated documentation
simply add the notebook name to the ``../notebooks.rst`` file, and include a markdown
cell at the beginning of your notebook with ``# Title`` that will be used as the text
in the table of contents in the documentation.

Be aware that you may also need to update the ``../requirements.txt`` file if
your notebooks have dependencies that are not specified in ``../pyproject.toml``.

For notebooks that require large datasets, access to third party APIs, large CPU or GPU requirements, put them in `./pre_executed` instead.

For more information look here: https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks

Or if you still have questions contact us: https://lincc-ppt.readthedocs.io/en/latest/source/contact.html

## Pre-executed notebooks never touch the developer database

No pre-executed notebook byte-copies or writes the developer database (`src/fomo_db.sqlite3`); a notebook that needs real rows reads it only through a read-only online-backup snapshot, and every other notebook starts from a fresh migrated database.

### Notebooks that read a read-only snapshot

- `project_observation_calendar_demo.ipynb` -- reports on the real LCO/SOAR observation corpus and runs the real one-time takeover sweep, so it needs the developer database's rows.
- `reconcile_campaign_runs_demo.ipynb` -- reconciles the real campaign runs and calendar events and runs the classical-allocation cutover against them.

### Notebooks that start from a fresh migrated database

- `backfill_lco_observations_demo.ipynb`
- `campaign_lifecycle_demo.ipynb`
- `import_campaign_csv_demo.ipynb`
- `load_telescope_runs_demo.ipynb`
- `sync_gemini_observation_calendar_demo.ipynb`
- `telescope_runs_demo.ipynb`

Each of these creates every row it shows (or seeds the few `Observatory` rows it needs), so nothing in the
developer database is needed and a copy would only carry unrelated rows into the output.

### How a notebook uses the helper

All eight notebooks use `docs/notebooks/fomo_nb_utils.py`. The setup cell puts the repo root and
`docs/notebooks` on `sys.path`, imports `setup_scratch_django` and `teardown_scratch_django`, and calls

```python
scratch_db_dir = setup_scratch_django(repo_root_path, mode='fresh')  # or mode='snapshot'
```

before anything imports Django. The last cell calls `teardown_scratch_django(scratch_db_dir)`.

The setup function creates a `/tmp/fomo-notebook-db-*` directory, records the current values of
`FOMO_DATABASE_PATH` and `DJANGO_ALLOW_ASYNC_UNSAFE`, and points `FOMO_DATABASE_PATH` at a database in the
directory with a plain assignment (never `setdefault`, so an inherited value cannot win). In snapshot mode it
fills that database with SQLite's online backup of `src/fomo_db.sqlite3`, opened through a read-only (`mode=ro`)
connection so the unattended runner can write the developer database at the same moment. Before `django.setup()`
runs, the helper raises a `RuntimeError` (a real `raise`, so it also fires under `python -O`) unless Django would
use exactly the scratch database file, with no database alias other than `default` and no `DATABASE_ROUTERS`; after
that refusal, or any later failure once the helper has started reading Django's settings, no cell can reach a
database until the kernel is restarted. A failure before the settings are read (for example snapshot mode with no
developer database) routes nothing, and the setup cell can simply be run again. A refusal or failure once the
settings are being read switches every database alias to Django's dummy backend (any query raises
`ImproperlyConfigured`) before it closes the connections, then drops Django's cached connection wrappers whether or
not the close worked, so no query made through Django can reach a database; it then removes the scratch directory,
puts the two variables back and raises the original error. When the checks pass, the helper runs `django.setup()`,
migrates the scratch database to head and prints `Resolved database: ...`.

The teardown refuses any directory other than the one this kernel's setup created directly inside the temporary
directory. For that one it closes Django's connections, removes the directory and restores `FOMO_DATABASE_PATH`
and `DJANGO_ALLOW_ASYNC_UNSAFE` to their values before the setup. Its second printed line says what the setup
checked: before `django.setup()` ran, Django's only configured database was the scratch file.

To regenerate a notebook, run `pre-commit run ruff-format --files <notebook>`, then, from this `pre_executed/`
directory, `env TMPDIR=/tmp jupyter nbconvert --to notebook --execute --inplace <notebook>`, or choose
*Restart and Run All* in Jupyter. Re-running the setup cell once the helper has read Django's settings in this kernel
(after a successful setup, a refusal or a later failure) raises `RuntimeError`; use *Restart and Run All* instead.
Never export `FOMO_DATABASE_PATH` yourself -- the helper sets it -- and commit the notebook with its output.

The two snapshot notebooks need a populated `src/fomo_db.sqlite3` (they raise `RuntimeError` without one), and the
numbers they print change from one regeneration to the next because the developer database changes.

A snapshot holds a full copy of the developer database. If a run fails before the teardown cell, delete any
leftover `/tmp/fomo-notebook-db-*` directory yourself.

`solsys_code/tests/test_pre_executed_notebooks.py` checks every notebook in `pre_executed/`. It checks that a
notebook calls the helper first; never sets Django up, chooses a database, starts a process, changes the environment
or runs code built from a string or loaded from another file, whether the name is written in its code, in an import
or inside a string; never names `FOMO_DATABASE_PATH` or the developer database file in a str or bytes literal;
never byte-copies or byte-writes a file with shutil's copy and move functions or `write_bytes` (`copy2`,
`copyfile`, `copytree` and `copyfileobj` wherever they are named, `copy` and `move` when they are read from shutil
or from a name the notebook binds to it); contains only Python that IPython runs unchanged, apart from
`%matplotlib inline` and a first-line `%%time` or `%%capture`; has no one-line cell that starts with an IPython
magic or alias name, even one an earlier cell assigns or imports, unless that name is a Python builtin or keyword or
is followed by `=` or `,`; and shows output from one fresh run. Its module docstring says exactly what each rule
follows and what a static check of a notebook's source cannot see. It reads each notebook's JSON
and never executes one.
`solsys_code/tests/test_notebook_helper.py` runs the helper itself, and a test in the first module keeps the two
lists above equal to each notebook's `mode`.
