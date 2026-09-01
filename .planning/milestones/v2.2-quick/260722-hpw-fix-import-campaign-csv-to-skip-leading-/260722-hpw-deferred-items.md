# Deferred Items

Out-of-scope discoveries logged during execution of quick task 260722-hpw, per the
executor's scope boundary rule (only auto-fix issues directly caused by the current
task's changes).

## Pre-existing `ruff format --check .` drift (repo-wide, not introduced by this task)

`ruff format --check .` (run at the repo root) reports 7 files needing reformatting:

- `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_nb.py`
- `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_project.py`
- `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`
- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`
- `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb`
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`
- `src/fomo/settings.py`

`ruff check .` (repo root) also reports 5 pre-existing errors (an unsorted-imports block
in `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` and an
`E501` line-too-long in the same notebook, unrelated cells).

Verified pre-existing (not caused by this task): `import_campaign_csv_demo.ipynb`'s
drift is in cell 3 (the Django-setup boilerplate cell, untouched by this task) and was
already present at the pre-dispatch commit (`97b5587`), confirmed via
`git show 97b5587:docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb | ruff
format --check -`. The other 6 files were never touched by this task's tasks.

This task's own quality-gate scope (`./manage.py test
solsys_code.tests.test_import_campaign_csv`, `ruff check .`/`ruff format --check .`
restricted to the 3 files this plan modified) is clean -- see SUMMARY.md.

Not fixed here per the executor's scope boundary rule (pre-existing issues in
unrelated files are out of scope for this quick task).
