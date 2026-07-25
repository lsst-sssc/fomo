# Deferred Items — quick-260724-vb0

Out-of-scope discoveries found while running the repo-wide quality gate (`ruff check .`)
in Task 3. Not fixed here per the executor's scope boundary (pre-existing, unrelated to
this plan's files).

## ruff D103 in sync_gemini_observation_calendar_demo.ipynb

`ruff check .` reports one repo-wide finding unrelated to this plan's `files_modified`:

```
D103 Missing docstring in public function
 --> docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb:cell 6:6:5
def make_gem_record(obs_id: str, params: dict) -> ObservationRecord:
```

Confirmed pre-existing: the notebook is unchanged since commit `292929a` (feat(10-02):
re-execute demo notebook and extend CLAUDE.md companion-notebook list) and was already
present, unmodified, at this quick task's base commit `8a973ce`. This plan's
`files_modified` does not touch this notebook. `ruff check` scoped to this plan's actual
files (`solsys_code/templatetags/calendar_display_extras.py`,
`solsys_code/tests/test_calendar_display_extras.py`,
`solsys_code/tests/test_calendar_template.py`) is clean.

## ruff format --check . (repo-wide, informational)

`ruff format --check .` (unscoped) reports 4 pre-existing files needing reformatting:
`docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`,
`docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb`,
`docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`, and
`src/fomo/settings.py`. None are in this plan's `files_modified`; scoped to this plan's
3 actual files, `ruff format --check` reports all 3 already formatted.
