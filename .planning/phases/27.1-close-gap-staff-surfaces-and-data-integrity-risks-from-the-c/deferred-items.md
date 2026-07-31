# Deferred Items -- Phase 27.1

Out-of-scope findings discovered while executing plan 27.1-02, logged per the executor's
SCOPE BOUNDARY rule (not caused by this plan's changes, not fixed here).

## From plan 27.1-02 (Task 3 quality gates)

Global `ruff check .` / `ruff format --check .` (run across the whole repo, not scoped to
this plan's three files) surface pre-existing findings unrelated to this plan's diff:

- `ruff check .`: `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb`
  cell 6 -- `D103 Missing docstring in public function` on `make_gem_record()`. Pre-existing;
  file not touched by this plan.
- `ruff format --check .`: three pre-existing files would be reformatted --
  `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_nb.py`,
  `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_project.py`,
  `src/fomo/settings.py`. None touched by this plan.

This plan's three files (`solsys_code/models.py`, `solsys_code/admin.py`,
`solsys_code/tests/test_admin.py`) are individually clean under both `ruff check` and
`ruff format --check`, matching the plan's action text ("resolving every finding in this
plan's three files").
