---
phase: 38-sync-with-main
plan: 01
subsystem: infra
tags: [git-merge, tomtoolkit-3.1.0, tom_jpl-0.3.0, packaging, django-admin, url-routing]

requires:
  - phase: 37
    provides: "branch issue37-telescope-runs-calendar at 088f73b (v2.4 complete, phase 38 planned)"
provides:
  - "one two-parent merge commit e12158c: origin/main (a910c17) merged into issue37-telescope-runs-calendar, unpushed"
  - "nine conflicts resolved keeping both sides (D-03), reviewed and approved by the developer before commit (D-01)"
  - "dev venv on tomtoolkit 3.1.0 and tom_jpl 0.3.0, tom-registration removed, pip check clean"
  - "merged tree boots (only urls.W005), no migration drift, full suite 2178 tests OK"
affects: [38-02, 38-03, 38-04]

actuals:
  tokens: 95000
  tasks: 3
  commits: 63
plan_head_before: 088f73b9fd437ecb3fe9ed16ffc821821cad2ac5
plan_head_after: e12158c396a85814f6d8e91ea6b22a3fa902a380

tech-stack:
  added: [tomtoolkit 3.1.0, tom_jpl 0.3.0, django-allauth 65.19.7, ruff 0.16.10, coverage 7.13.5]
  patterns: ["merge, never rebase: HEAD^1 is the branch line, HEAD^2 is origin/main", "merge commit is a pure sync: outside the nine conflicted paths it equals git's automatic merge"]

key-files:
  created: [.planning/phases/38-sync-with-main/38-MERGE-RESOLUTION.diff]
  modified: [.gitignore, docs/conf.py, docs/design/design.rst, docs/notebooks.rst, pyproject.toml, solsys_code/admin.py, solsys_code/apps.py, solsys_code/tests/test_bootstrap5_rendering.py, src/fomo/urls.py]

key-decisions:
  - "Merge landed with SKIP=django-test,ruff,ruff-format so no reformat or segfaulting TestEphemeris run rides in the merge commit (D-02)"
  - "pytest and pytest-cov left installed in the venv (harmless, no longer declared)"
  - "Cron run_unattended line stays paused until 38-03 restores it"

requirements-completed: [SYNC-01, SYNC-02, SYNC-04]

duration: 35min
completed: 2026-10-07
status: complete
---

# Phase 38 Plan 01: Merge origin/main Summary

**origin/main (62 commits, a910c17) merged into issue37-telescope-runs-calendar in one reviewed two-parent merge commit (e12158c), nine conflicts resolved by keeping both sides, dev venv moved to tomtoolkit 3.1.0 / tom_jpl 0.3.0, and the full 2178-test suite green on the merged tree.**

## Performance

- **Duration:** about 35 min in this continuation (Task 3 and summary); Tasks 1-2 ran in the earlier dispatch
- **Completed:** 2026-10-07T17:28Z
- **Tasks:** 3 of 3
- **Files modified:** 50 files in the merge (+3033 / -268); nine hand-resolved

## Key facts

| Item | Value |
|------|-------|
| Pre-merge head (HEAD^1, ORIG_HEAD) | 088f73b9fd437ecb3fe9ed16ffc821821cad2ac5 |
| origin/main head merged (HEAD^2) | a910c178be2e6e8063f8a262b51934ca05cdbb01 (did not move since planning) |
| Merge commit | e12158c396a85814f6d8e91ea6b22a3fa902a380 |
| Developer's Task 2 answer | approve |
| Pushed | no (Plan 04 pushes) |

Developer's Task 2 answer: approve

## Installed versions (dev venv, /home/tlister/venv/devel_fomo311_venv)

| Package | Version |
|---------|---------|
| tomtoolkit | 3.1.0 (was 3.0.1) |
| tom_jpl | 0.3.0 (replaced the editable ~/git/tom_jpl 0.1.0.post6 install, A6; the clone on disk is untouched) |
| Django | 5.2.17 (unchanged) |
| ruff | 0.16.10 (was 0.2.1) |
| coverage | 7.13.5 |
| django-allauth | 65.19.7 (new, tomtoolkit 3.1.0 requirement) |
| tom-registration | uninstalled (2.0.1) |

`python -m pip check`: "No broken requirements found." pytest and pytest-cov are still installed but no longer declared; left alone (harmless).

## `manage.py check` output (saved at $HOME/tmp/phase38-manage-check.txt; Plan 02 compares it with the runbook paragraph)

```
System check identified some issues:

WARNINGS:
?: (urls.W005) URL namespace 'calendar' isn't unique. You may not be able to reverse all URLs in this namespace

System check identified 1 issue (0 silenced).
```

`makemigrations --check --dry-run`: "No changes detected".
`nav_items()` returns `['solsys_code/partials/campaigns_nav_link.html', 'solsys_code/partials/navbar_list.html']`.
Seven merge-wiring test labels (admin, search, Scout views, Rubin ToO, packaging, TestUserDeleteView, Bootstrap 5 data attributes): Ran 123 tests, OK, run before the commit.

## Full suite

`python manage.py test --noinput solsys_code --exclude-tag=ephemeris_segfault` on the merge commit (log at $HOME/tmp/phase38-01-suite.log):

`Ran 2178 tests in 460.638s` followed by an exact `OK` line; no `^FAILED` line and no `skipped=`. (The log contains "step reconcile: FAILED" lines; those are printed by run_unattended tests exercising the failure path, not test failures.) The known flaky Playwright test did not fail, so no re-run was needed.

## Task Commits

1. **Task 1 (tracer): merge origin/main, resolve nine conflicts, staged** - no commit by design (D-01); previous dispatch.
2. **Task 2: developer approval checkpoint** - answered "approve"; no commit.
3. **Task 3: install, prove the staged tree, commit merge, full suite** - `e12158c` (merge commit).

**Plan metadata:** the docs commit following this file (SUMMARY + 38-MERGE-RESOLUTION.diff), then the tracking-files commit.

`actuals.commits: 63` is the measured `git rev-list --count 088f73b..HEAD`: the merge commit itself plus the 62 origin/main commits that arrived through it. This plan authored one commit.

## Post-commit checks (all passed)

- `git rev-list --parents -n 1 HEAD` prints three hashes; HEAD^2 = origin/main; HEAD^1 = ORIG_HEAD = 088f73b; no HEAD^3.
- origin/main and fb07a66 are ancestors of HEAD.
- Outside the nine conflicted paths HEAD equals `git merge-tree --write-tree HEAD^1 HEAD^2`; no `.planning/` path in the merge.
- Merge deletes exactly tests/fomo/conftest.py and tests/fomo/test_packaging.py; `git ls-files tests` is empty.
- `_commit: v2.2.0` in .copier-answers.yml, .github/pull_request_template.md present, `src/_static/` and `fomo_db_20*.sqlite3` ignored, HEAD not on origin.
- The pre-commit hooks that ran on the merge commit (template version, prevent-main, large files, validate-pyproject, pre-executed notebooks) all passed; ruff, ruff-format and django-test were skipped as planned.
- The leftover untracked `tests/` directory (only `.pyc` caches) was deleted.

## Deviations from Plan

None - plan executed exactly as written.

## Authentication Gates

None.

## Issues Encountered

None. The staged resolution needed no change after the developer's approval.

## Known Stubs

None.

## Threat Flags

None. No new network, auth or file-access surface beyond what the plan's threat model covers (T-38-01 to T-38-07 mitigations held: the `*/local_settings.py` autoapi guard is kept, the user-delete route is kept before tom_common.urls, one Target registration, the index outside the nine paths equals git's merge).

## run_unattended cron line: PAUSED

The developer's 15-minute run_unattended cron line is still paused (`#PHASE38-PAUSED ` prefix). It was not restored in this plan; 38-03 restores it after the database migrate. To restore it early:

```
crontab "$HOME/tmp/phase38-crontab.bak"
```

## Never staged

STATE.md, config.json, state.json, agent-history.json, milestone.lock, `reqgroup_2682493.json` and the `src/fomo_db_20*.sqlite3` backups were never staged into the merge commit.

## Self-Check: PASSED

- e12158c exists and is HEAD at the time of the checks; both parents verified.
- 38-MERGE-RESOLUTION.diff exists (75 KB).
