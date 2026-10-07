---
phase: 38-sync-with-main
plan: 02
subsystem: infra
tags: [ruff-0.16.9, pre-commit, ci, coverage, ephemeris_segfault, nbsphinx, claude-md]

requires:
  - phase: 38-01
    provides: "merge commit e12158c (origin/main merged), venv on tomtoolkit 3.1.0 / tom_jpl 0.3.0, recorded manage.py check output"
provides:
  - "tree clean under the enforced ruff 0.16.9 (both hooks pass twice, idempotent)"
  - "CI unit-test matrix, daily smoke test and the django-test hook all exclude the ephemeris_segfault class"
  - "pre-executed-nb-never-execute hook repointed at docs/notebooks/pre_executed/; all eight notebooks carry nbsphinx.execute=never"
  - "CLAUDE.md, docs/installation.rst and the GSD codebase maps in step with the merged tooling"
affects: [38-03, 38-04]

actuals:
  tokens: 13200
  tasks: 3
  commits: 5
plan_head_before: 65d6f3229f44341e7e3222ebdafecbe3605d4a97
plan_head_after: 22e1b0aa30b96f88518f615e54fdbef9703d6296

tech-stack:
  added: []
  patterns: ["style commit proven AST-identical file by file before it is trusted", "hooks that rewrite files are run by hand and committed separately from the config change that triggers them"]

key-files:
  created: []
  modified:
    - .pre-commit-config.yaml
    - .github/workflows/testing-and-coverage.yml
    - .github/workflows/smoke-test.yml
    - solsys_code/management/commands/backfill_lco_observations.py
    - docs/conf.py
    - CLAUDE.md
    - docs/installation.rst
    - .planning/codebase/STACK.md
    - .planning/codebase/CONVENTIONS.md

key-decisions:
  - "SIM103 fixed in code, not by widening the ruff ignore list (D-09); pyproject.toml unchanged since the merge"
  - "Smoke test gets the same one-token exclusion as the CI matrix (A8) so it does not hit the ASSIST segfault every morning"
  - "Codebase maps STACK.md and CONVENTIONS.md edited alongside CLAUDE.md because CLAUDE.md's generated blocks come from them (A9)"

requirements-completed: [SYNC-03, SYNC-05, SYNC-06]

duration: 8min
completed: 2026-10-07
status: complete
---

# Phase 38 Plan 02: Adopt main's tooling Summary

**Tree is clean under ruff 0.16.9 (one mechanical reformat commit plus one behavior-preserving SIM103 fix), CI, the smoke test and the django-test hook exclude the one segfaulting test class, the notebook hook points at the real notebook directory, and CLAUDE.md plus the codebase maps describe the merged tooling.**

## Commits (five, each with SKIP=django-test)

| # | Hash | Subject |
|---|------|---------|
| 1 | ff8dd3c | style: reformat with ruff 0.16.9 (ruff-format hook, no other change) |
| 2 | 398fa26 | fix: return the negated condition in backfill_lco_observations (ruff SIM103) |
| 3 | d535ef4 | ci: exclude ephemeris_segfault in CI and the django-test hook; repoint the pre-executed notebook hook |
| 4 | 1ac9a50 | chore: mark pre-executed notebooks nbsphinx.execute=never |
| 5 | 22e1b0a | docs: bring CLAUDE.md, installation and codebase maps in step with the main sync |

`git rev-list --count` from the plan's recorded base to HEAD is 5 (measured, not narrated).

## Task 1: ruff 0.16.9

- Reformat touched 12 files, exactly the set RESEARCH predicted: nine `.py` files (`allocation_projector.py`, `campaign_tables.py`, `campaign_views.py`, `check_unattended.py`, `load_telescope_runs.py`, `reconcile_campaign_runs.py`, `notifications.py`, `templatetags/calendar_display_extras.py`, `tests/test_status_vocabulary.py`) and three pre-executed notebooks (`campaign_lifecycle_demo`, `project_observation_calendar_demo`, `reconcile_campaign_runs_demo`). None of the five files resolved in 38-01 needed reformatting.
- Proof the commit is a pure reformat: every `.py` file parses to the same AST before and after, and every notebook keeps identical outputs and execution counts (script in the plan's verify block, `OK: style commit is a pure reformat of 12 files`).
- Lint found one issue: SIM103 in `backfill_lco_observations.py` (no automatic fix). Rewritten by hand to `return not (created_before is not None and created > created_before)`.
- Both ruff hooks pass twice in a row and change nothing; `git status --porcelain -- . ':(exclude).planning'` shows only the pre-existing untracked `reqgroup_2682493.json`.
- `[tool.ruff.lint]` and the rest of `pyproject.toml` are byte-identical to the merge commit's.
- Targeted tests (backfill, pre-executed notebooks, projector demo notebook, status vocabulary, check_unattended): Ran 239 tests, OK.

### Flag for the developer (D-09): SIM103 rewrite

`solsys_code/management/commands/backfill_lco_observations.py` is a notebook-mapped module (`backfill_lco_observations_demo.ipynb`). The change replaces `if C: return False / return True` with `return not C`, which has the same truth table. It is a pure refactor, so the paired notebook was not re-executed. Please confirm you agree that this is not a behavior change.

## Task 2: CI and hooks

- Verified (not edited): no pytest or Sphinx hook, both ruff-pre-commit entries at `v0.16.9`, `jupyter-nb-clear-output` still excludes `^docs/notebooks/pre_executed`.
- `django-test` hook entry is now `bash -c "coverage run manage.py test --exclude-tag functional --exclude-tag ephemeris_segfault && coverage html"`, with a comment naming `SKIP=django-test` for work-in-progress commits and explaining why TestEphemeris is excluded.
- Notebook hook `files:` and `args:` point at `docs/notebooks/pre_executed/`.
- `testing-and-coverage.yml` unit-test matrix and `smoke-test.yml` each changed by exactly one line (one added `--exclude-tag ephemeris_segfault`); `git diff --numstat origin/main -- .github/workflows` shows `1 1` for each file and nothing else. The functional-tests job (`--tag functional`) is untouched. `@tag('ephemeris_segfault')` appears once under `solsys_code/`.
- A8 smoke-test note: D-05 names only the unit-test matrix; the daily smoke test runs the same suite and would hit the ASSIST segfault every morning without the same exclusion, so it gets the same one-token change.
- `docs/conf.py`: only the comment above `suppress_warnings` was rewritten (it described the removed pre-commit Sphinx hook as current); the setting is unchanged.
- Eight notebooks now carry `metadata.nbsphinx.execute == "never"`, committed alone; the metadata commit changed no cell (source, outputs or execution counts). No notebook was re-executed anywhere in this plan.
- `pre-commit run pre-executed-nb-never-execute --all-files` and `pre-commit run check-github-workflows --all-files` pass.

## Task 3: CLAUDE.md, installation, maps

- CLAUDE.md: note now says `enforces (v0.16.9)`; the test command lines are the five from the plan (including `--exclude-tag ephemeris_segfault`); the Testing section has a third paragraph on TestEphemeris; the pre-commit and copier Conventions bullets describe the django-test hook, `SKIP=django-test` and the local divergences; Key Dependencies names `tomtoolkit>=3.1.0`, `tom_jpl>=0.3.0`, `ruff 0.16.9`, the Django runner and coverage, and no longer lists pytest, pytest-cov or tom_registration. No bare `ruff check` / `ruff format` line exists.
- `docs/installation.rst` lists `* timezonefinder>=6.0`; no registration bullet. `src/fomo/settings.py` has no `tom_registration` entries (verified, not edited).
- `.planning/codebase/STACK.md` and `CONVENTIONS.md` updated for the same stale lines (A9) so the next `generate-claude-md` does not bring them back.
- Runbook check: `python manage.py check` reports only `urls.W005`, which the runbook already describes, so `docs/runbooks/telescope_runs_calendar.rst` is unchanged.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] The notebook hook ran during the ci commit and failed the commit**
- **Found during:** Task 2, step 7
- **Issue:** `pre-executed-nb-never-execute` is an always-run hook. Once `.pre-commit-config.yaml` pointed at `docs/notebooks/pre_executed/`, the hook ran during the ci commit, added the nbsphinx key to the eight notebooks (modifying files outside the commit) and failed it.
- **Fix:** committed step 7 with `SKIP=django-test,pre-executed-nb-never-execute`; then ran the hook by hand (the plan's step 8) and committed the eight notebooks in their own `chore:` commit as planned. Commit order and contents are exactly as the plan specifies.
- **Files modified:** none beyond the plan
- **Commit:** d535ef4 (ci), 1ac9a50 (chore)

**Total deviations:** 1 auto-fixed (Rule 3). **Impact:** none on the delivered tree; the extra SKIP value is the only difference from the plan's commit commands.

## Authentication Gates

None.

## Known Stubs

None.

## Threat Flags

None. Threat mitigations held: T-38-08 (workflow diff is two lines), T-38-09 (reformat AST-proven, SIM103 behavior-preserving), T-38-10 (exclusion covers exactly one tagged class), T-38-11 (notebook outputs and cells verified identical).

## run_unattended cron line: STILL PAUSED

The developer's run_unattended cron line is still paused (`#PHASE38-PAUSED ` prefix). It is restored by 38-03 Task 2 after the database migrate. A stop between this plan and 38-03 is therefore not silent: to restore early, run `crontab "$HOME/tmp/phase38-crontab.bak"`; the normal path is to resume the phase, because restoring before 38-03 migrates the database lets ticks run against an unmigrated database.

## Never staged

`.planning/config.json`, `.planning/state.json`, `.planning/agent-history.json`, `.planning/milestone.lock` and `reqgroup_2682493.json` were never staged into a task commit.

## Self-Check: PASSED

- All five commits are ancestors of HEAD (ff8dd3c, 398fa26, d535ef4, 1ac9a50, 22e1b0a).
- Every plan verify block printed its `OK:` line; both ruff hooks pass and are idempotent after the last commit.
