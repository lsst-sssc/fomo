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

Every notebook under `pre_executed/` that sets Django up must, before `django.setup()`,
create a `/tmp/fomo-notebook-db-*` directory with `tempfile.mkdtemp`, point
`FOMO_DATABASE_PATH` at a database inside it with a plain assignment (never `setdefault`, so
an inherited value cannot win), check that Django resolved that path, and print
`Resolved database: ...`. It then either migrates a fresh database and creates every row it
needs (most notebooks), or takes a read-only snapshot of `src/fomo_db.sqlite3` with SQLite's
online backup when it needs the real records (`project_observation_calendar_demo`). Its last
code cell removes the directory.

To regenerate a notebook, run `pre-commit run ruff-format --files <notebook>`, then, from this
`pre_executed/` directory, `jupyter nbconvert --to notebook --execute --inplace <notebook>`.
Never export `FOMO_DATABASE_PATH` yourself -- the setup cell sets it -- and commit the notebook
with its output.

`solsys_code/tests/test_pre_executed_notebooks.py` enforces this for every notebook in
`pre_executed/`. It reads each notebook's JSON and never executes one.
