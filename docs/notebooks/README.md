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
use exactly the scratch database file, with no database alias other than `default` and no `DATABASE_ROUTERS`;
after a refusal, or any later failure in the setup, no cell can reach a database until the kernel is restarted.
A refusal or failure switches every database alias to Django's dummy backend (any query raises
`ImproperlyConfigured`), removes the scratch directory and puts the two variables back. When the checks pass, the
helper runs `django.setup()`, migrates the scratch database to head and prints `Resolved database: ...`.

The teardown refuses any directory other than the one this kernel's setup created directly inside the temporary
directory. For that one it closes Django's connections, removes the directory and restores `FOMO_DATABASE_PATH`
and `DJANGO_ALLOW_ASYNC_UNSAFE` to their values before the setup.

To regenerate a notebook, run `pre-commit run ruff-format --files <notebook>`, then, from this `pre_executed/`
directory, `env TMPDIR=/tmp jupyter nbconvert --to notebook --execute --inplace <notebook>`, or choose
*Restart and Run All* in Jupyter. Re-running the setup cell in a kernel that has already set Django up, or where a setup already
failed, raises `RuntimeError`; restart the kernel instead. Never export `FOMO_DATABASE_PATH` yourself -- the helper sets it --
and commit the notebook with its output.

The two snapshot notebooks need a populated `src/fomo_db.sqlite3` (they raise `RuntimeError` without one), and the
numbers they print change from one regeneration to the next because the developer database changes.

A snapshot holds a full copy of the developer database. If a run fails before the teardown cell, delete any
leftover `/tmp/fomo-notebook-db-*` directory yourself.

`solsys_code/tests/test_pre_executed_notebooks.py` checks every notebook in `pre_executed/`: it calls the helper
first, never sets Django up or routes the database itself, never byte-copies, and shows output from one fresh run.
It reads each notebook's JSON and never executes one. `solsys_code/tests/test_notebook_helper.py` runs the helper
itself, and a test in the first module keeps the two lists above equal to each notebook's `mode`.
