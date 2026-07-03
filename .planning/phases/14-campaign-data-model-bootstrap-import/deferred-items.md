# Deferred Items - Phase 14 Plan 03

Out-of-scope findings discovered during execution but not fixed (scope boundary:
only auto-fix issues directly caused by this plan's changes).

## `ruff check .` / `ruff format --check .` pre-existing findings (not introduced by this plan)

- `ruff check .` reports 5 pre-existing errors, all in
  `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` (unused
  `pprint` import, unsorted imports, one `E501` long line). Not touched by this
  plan.
- `ruff format --check .` reports 6 pre-existing files that would be
  reformatted: `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_nb.py`,
  `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_project.py`,
  `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`,
  `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb`,
  `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`,
  `src/fomo/settings.py`. None of these are touched by this plan (which only
  adds `docs/notebooks/pre_executed/fixtures/campaign_sample.csv` and
  `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`).

Both new files this plan adds pass `ruff check .` and `ruff format --check .`
cleanly (verified individually and as part of the full-repo run). This phase's
own `<verification>` block ("ruff gates unaffected (no .py changes in this
plan)") is satisfied; the findings above are a pre-existing repo-wide
condition (some already logged in Phase 04's own `deferred-items.md`), not a
regression introduced by Plan 14-03.
