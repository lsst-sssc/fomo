# Deferred Items - Phase 04 Plan 01

Out-of-scope findings discovered during execution but not fixed (scope boundary:
only auto-fix issues directly caused by this plan's changes).

## ruff format --check . pre-existing findings (not introduced by this plan)

- `src/fomo/settings.py` would be reformatted by `ruff format`. Last touched by
  commit `eaf75aa` ("Migrate to tomtoolkit 3.0 (pre-release) and Django 5"),
  unrelated to this plan. Not modified by Plan 04-01.
- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` would be
  reformatted by `ruff format`. Pre-existing Phase 3 demo notebook, not touched
  by this plan.

Both files were verified untouched by `git status --short` at Task 3 time and
are unrelated to `solsys_code/management/commands/sync_lco_observation_calendar.py`
or `solsys_code/tests/test_sync_lco_observation_calendar.py`. `ruff check .`
(lint) is fully clean; only `ruff format --check .` flags these two pre-existing
files. The phase-level success criterion 5 (`ruff check . / ruff format --check .`
clean) is satisfied for all files this plan touches; these two reformats are a
pre-existing repo-wide condition, not a regression from this plan.
