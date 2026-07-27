# Phase 26 — Deferred Items

Out-of-scope discoveries found during execution that were not fixed, per the executor's
scope-boundary rule (only auto-fix issues directly caused by the current task's changes).

## Pre-existing repo-wide ruff/format drift (found during 26-01 Task 3)

`ruff check .` and `ruff format --check .`, run against the full repository as part of
Task 3's verification, surface failures unrelated to this plan's own changes:

- `ruff check .`: `D103` (missing docstring in public function) in
  `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb` (cell 6).
- `ruff format --check .`: would reformat
  `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_nb.py`,
  `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_project.py`,
  `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`,
  `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`,
  `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb`,
  `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`,
  `src/fomo/settings.py`.

Confirmed pre-existing: `git diff 77e16b5 -- <each path above>` is empty for every one of
these files — none was touched by any 26-01 task. Every file 26-01 actually created or
edited (`solsys_code/models.py`, `solsys_code/migrations/0008_scratch_canonical_record_probe.py`
[scratch branch only], `solsys_code/admin.py`, `solsys_code/management/commands/
sync_lco_observation_calendar.py`, and the four test modules touched by Task 3) is
individually `ruff check`/`ruff format --check` clean.

Not fixed here — out of scope for an investigation-only phase whose only committed
deliverable is `26-DECISION.md`. Whoever next touches these files (or a dedicated cleanup
task) should run `ruff check . --fix && ruff format .` and review the notebook docstring
gap separately, since `docs/notebooks/pre_executed/` content is otherwise intentionally
committed with output per CLAUDE.md's pre-commit convention.
