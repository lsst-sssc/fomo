# Deferred Items

Out-of-scope discoveries found during execution, logged rather than fixed per the executor's
scope-boundary rule (only auto-fix issues directly caused by the current task's changes).

## From 27.1-03 (Task 3 quality gates)

Found while running the repo-wide `ruff check .` / `ruff format --check .` quality gates
required by 27.1-03's Task 3 acceptance criteria. All three files below are untouched by this
plan (confirmed via `git status --short` / `git diff --stat` against the pre-plan HEAD) --
pre-existing repo state, unrelated to campaign_views.py, campaign_list.html,
test_campaign_views.py or the runbook.

- `ruff check .`: `D103 Missing docstring in public function` on
  `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb` (cell 6,
  `make_gem_record`). The notebook's own commit message (`400a176 fix(27): IN-03 make the
  pre-executed notebooks stable under both pinned ruff versions`) suggests this is a
  ruff-version-drift issue (locally installed ruff 0.15.20) rather than a real regression;
  scoped-to-plan-files `ruff check solsys_code/campaign_views.py
  solsys_code/tests/test_campaign_views.py` passes clean.
- `ruff format --check .`: would reformat `src/fomo/settings.py`,
  `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_nb.py`, and
  `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_project.py` --
  none touched by this plan.

Not fixed here (out of scope, per the executor's scope boundary). See 27.1-03-SUMMARY.md for
the full before/after grep output this plan's own acceptance criteria required.
