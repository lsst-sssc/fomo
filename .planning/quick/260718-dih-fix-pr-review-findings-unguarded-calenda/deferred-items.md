# Deferred Items — Quick Task 260718-dih

Out-of-scope discoveries found while running the repo-wide quality gates
(`ruff check . --fix`, `ruff format .`) during Task 3's verification step.
Not fixed here per the deviation-rules Scope Boundary (only auto-fix issues
directly caused by this task's changes; pre-existing repo-wide drift in
unrelated files is out of scope).

## Pre-existing ruff drift (unrelated to this task)

Running `ruff check . --fix` / `ruff format .` unscoped touches files this
task never modified:

- `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_nb.py`
- `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_project.py`
- `solsys_code/migrations/0001_calendareventtelescopelabel.py`
- `solsys_code/migrations/0005_campaignrun_campaign_run_window_start_end_null_together.py`
- `solsys_code/migrations/0007_campaignrun_contact_public_opt_in.py`
- `solsys_code/solsys_code_observatory/migrations/0001_initial.py`
- `src/fomo/settings.py`
- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` (isort-only import reorder)
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` (isort-only import reorder)

Reverted all of these before committing. Ruff was instead scoped to just the
6 files this task actually touched (`campaign_views.py`, `telescope_runs.py`,
`load_telescope_runs.py`, and the three test modules) via explicit file-list
invocations, plus `--exclude "docs/notebooks/pre_executed/*.ipynb"
--force-exclude` for the whole-repo passes, since CLAUDE.md's paired-notebook
rule and this plan's scope note both prohibit touching the `pre_executed/`
notebooks. The task's own files are ruff-clean; the pre-existing repo-wide
drift above is unresolved and out of scope for this task.
