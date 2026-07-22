# Deferred Items - Phase 14 Plan 03

Out-of-scope findings discovered during execution but not fixed (scope boundary:
only auto-fix issues directly caused by this plan's changes).

## `ruff check .` / `ruff format --check .` pre-existing findings (not introduced by this plan)

- `ruff check .` reports 5 pre-existing errors, all in
  `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` (unused
  `pprint` import, unsorted imports, one `E501` long line). Not touched by this
  plan.
- `ruff format --check .` reports pre-existing files that would be
  reformatted: `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_nb.py`,
  `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_project.py`,
  `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`,
  `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb`,
  `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`,
  `src/fomo/settings.py`, and (added during the 14-REVIEW.md fix pass)
  `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` itself. None of
  the first six are touched by this plan.

**Update (post code-review-fix pass):** `import_campaign_csv_demo.ipynb` was
believed clean when Plan 14-03 completed, but a later `ruff format --check .`
run (during the 14-REVIEW.md fix pass) flagged one cell's `assert (...)`
paren-wrapping. Root cause identified: `.pre-commit-config.yaml` pins
`ruff-pre-commit rev: v0.2.1` (Feb 2024) while the locally installed `ruff`
CLI is `0.12.9` — the two versions disagree on this notebook's Jupyter-cell
formatting, and running `ruff format` under the newer CLI does not converge to
a fixed point the older pre-commit hook also accepts (repeated `git commit`
attempts kept toggling the same 4-line diff back and forth). This is the exact
same version-skew condition already affecting the six files above, just newly
surfaced for this file too. No content/logic change is involved — reformatting
is purely cosmetic, source-only (verified no output/behavior difference).
Not fixed here: doing so would require re-pinning the pre-commit `ruff-format`
hook to a modern version repo-wide, which is out of scope for this phase.

Both new files this plan adds pass `ruff check .` cleanly, and did pass
`ruff format --check .` at the time Plan 14-03's own verification ran. This
phase's own `<verification>` block ("ruff gates unaffected (no .py changes in
this plan)") is satisfied; the findings above are a pre-existing repo-wide
condition (some already logged in Phase 04's own `deferred-items.md`), not a
regression introduced by Plan 14-03 or by the 14-REVIEW.md fix pass.
