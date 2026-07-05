# Deferred Items — Phase 15 Plan 02

Out-of-scope discoveries found while executing 15-02-PLAN.md. Not fixed here per the
executor's scope boundary (only auto-fix issues directly caused by this plan's own changes).

## Pre-existing repo-wide `ruff check .` / `ruff format --check .` failures

Found while running the plan's repo-wide quality-gate verification step. All files
below were last modified by commits predating this phase (Phase 5 and an older quick
task), not by any of this plan's 3 task commits (`11f6194`, `7cb6b39`, `b205c09`).
Confirmed via `git log --oneline -1 -- <file>` and `git show --stat` on this plan's
commits (no overlap).

- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` — E401/I001 (unsorted
  multi-import), last touched by `9ca8a29` (partial-night window support)
- `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb` — D103
  (missing docstring), format drift
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` — I001, E501
  (line too long)
- `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` — format drift
- `src/fomo/settings.py` — format drift, last touched by `adc5a61` (Phase 5)
- `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_nb.py` —
  format drift, last touched by `bc5bfdf` (quick task, pre-dispatch scratch script)
- `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_project.py`
  — format drift, same commit

**None of the 5 files this plan actually modified/created are affected** —
`solsys_code/apps.py`, `src/templatetags/solsys_code_extras.py`,
`src/templates/solsys_code/partials/campaign_links.html`,
`src/templates/solsys_code/partials/campaigns_nav_link.html`, and
`solsys_code/tests/test_campaign_views.py` all individually pass
`ruff check` / `ruff format --check`.

**Recommendation:** a future cleanup task (or the next phase touching one of these
notebooks) should run `ruff check --fix` / `ruff format` on this list. Not blocking
for Phase 15 sign-off since VIEW-01..04 have no dependency on these files.
