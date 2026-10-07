---
phase: 38-sync-with-main
plan: 03
subsystem: infra
tags: [tomtoolkit-3.1.0, fresh-install, migrate, full-suite, tom_calendar-overrides]

requires:
  - phase: 38-01
    provides: "merge commit e12158c, dev venv on tomtoolkit 3.1.0 / tom_jpl 0.3.0, cron line paused"
  - phase: 38-02
    provides: "ruff 0.16.9-clean tree, django-test hook excluding ephemeris_segfault"
provides:
  - "fresh-install proof: brand-new venv resolves tomtoolkit 3.1.0 and tom_jpl 0.3.0 from the merged tree's .[dev]"
  - "full suite green on the merged tree in the fresh venv and through the CI/hook command in the dev venv, nothing skipped"
  - "developer database migrated to tomtoolkit 3.1.0's schema behind a verified backup; cron line restored"
  - "38-OVERRIDE-COMPARISON.md for Phase 39"
affects: [38-04, 39]

actuals:
  tokens: 40000
  tasks: 3
  commits: 1
plan_head_before: 29dfb30660ba84e217bfcd168f434e50a0700307
plan_head_after: 4c5e730721129561ebd82411c2659717a8db817a

tech-stack:
  added: []
  patterns: ["fresh-venv install of .[dev] as the SYNC-02 proof, kept at $HOME/venv/fomo_phase38_fresh", "wheel comparison by unzip and diff only, never import"]

key-files:
  created: [.planning/phases/38-sync-with-main/38-OVERRIDE-COMPARISON.md]
  modified: []

key-decisions:
  - "No upstream file FOMO shadows changed between tomtoolkit 3.0.1 and 3.1.0, so no SYNC-07 fix came from the override comparison"
  - "The two observations in the comparison note (leftover Bootstrap 4 class names in calendar.html; upstream's data-bs-url vs dataset.url mismatch) are left for Phase 39 to decide, not fixed here"

requirements-completed: [SYNC-02, SYNC-05, SYNC-07]

duration: 21min
completed: 2026-10-07
status: complete
---

# Phase 38 Plan 03: Fresh-install proof, full suite, database migrate and override comparison Summary

**A brand-new venv installs tomtoolkit 3.1.0 and tom_jpl 0.3.0 from the merged tree, the full suite passes there (2178 tests, OK) and through the CI/hook command in the dev venv (2170 tests, OK), the developer database is migrated behind an integrity-checked backup, the cron line is running again, and no shadowed tomtoolkit file changed between 3.0.1 and 3.1.0.**

## Task 1: fresh install, floors, database

Fresh venv `$HOME/venv/fomo_phase38_fresh` (Python 3.11.13; before the install its `pip list` held only pip and setuptools), `pip install -e '.[dev]'`, log at `$HOME/tmp/phase38-03-fresh-install.log`:

| Package | Version |
|---------|---------|
| tomtoolkit | 3.1.0 |
| tom_jpl | 0.3.0 |
| Django | 5.2.18 |
| django-allauth | 65.19.7 |
| ruff | 0.16.10 |
| coverage | 7.16.2 |

`pip check`: "No broken requirements found." The only versioned line `pyproject.toml` adds relative to origin/main is `"timezonefinder>=6.0"`; none is removed.

Developer database (cron line paused throughout, `FOMO_DATABASE_PATH` unset):

- Backup: `src/fomo_db_20261007_pre_phase38.sqlite3` (made with `sqlite3.Connection.backup`, `PRAGMA integrity_check` returned `ok`, ignored by git).
- `python manage.py migrate` applied (the Django/tomtoolkit lines that matter):

```
  Applying account.0001_initial ... account.0009_emailaddress_unique_primary_email... OK
  Applying mfa.0001_initial ... mfa.0003_authenticator_type_uniq... OK
  Applying tom_common.0005_profile_password_changed_at_profile_phone_number... OK
  Applying tom_common.0006_termsofserviceacceptance... OK
  Applying tom_dataproducts.0019_alter_astrometryreduceddatum_options_and_more... OK
  Applying tom_dataproducts.0020_remove_astrometryreduceddatum_unique_astrometry_and_more... OK
  Applying tom_jpl.0002_scoutdetail_dec_scoutdetail_ra_scoutdetail_rate_and_more... OK
  Applying tom_jpl.0003_scoutdetail_active... OK
  Applying tom_jpl.0004_alter_scoutdetailhistory_options... OK
  Applying tom_jpl.0005_scoutdetail_merged_into_scoutdetail_mpc_reference_and_more... OK
```

- `migrate --check` exited 0; `showmigrations tom_common tom_dataproducts` shows 0005, 0006, 0019 and 0020 applied and no `[ ]` row. No portal-calling command was run.

## Task 2: full suite

| Run | Where | Result |
|-----|-------|--------|
| `manage.py test --noinput solsys_code --exclude-tag=ephemeris_segfault` | fresh venv, log `$HOME/tmp/phase38-03-fresh-suite.log` | `Ran 2178 tests in 456.988s`, `OK` |
| `pre-commit run django-test --all-files --verbose` | dev venv, log `$HOME/tmp/phase38-03-hook.log` | hook Passed, `Ran 2170 tests in 549.950s`, `OK` |

The 8-test difference is the `functional` Playwright tests the hook excludes. Neither log has a `^FAILED` line or `skipped=`. (Both logs contain "step reconcile: FAILED" text printed by run_unattended tests exercising the failure path; those are not test failures.) The known flaky Playwright test did not fail, so no re-run was needed and nothing was tagged or skipped. The diff from the pre-merge head (e12158c^1) to HEAD under `solsys_code/` adds no skip decorator, `skipTest`, `expectedFailure` or tag other than main's `@tag('functional')`.

**Crontab restored at 2026-10-07T18:29:10Z (11:29:10 PDT)** with `crontab "$HOME/tmp/phase38-crontab.bak"`; `crontab -l` is byte-identical to the backup and the run_unattended line is active (the next tick runs the merged code against the migrated database).

## Task 3: override comparison

`38-OVERRIDE-COMPARISON.md` (committed as 4c5e730). The installed `tom_calendar` is identical to the 3.1.0 wheel's, and the 3.0.1 and 3.1.0 `tom_calendar` packages are byte-identical to each other. Conclusion: no upstream file FOMO shadows (`calendar_urls.py` vs `tom_calendar/urls.py`, `partials/calendar.html`, `partials/event_form.html`, `tom_common/index.html`, `tom_targets/partials/module_buttons.html`) changed between 3.0.1 and 3.1.0, so no upstream change is newly hidden and there is no SYNC-07 fix from this comparison. Two pre-existing observations are recorded for Phase 39 (leftover Bootstrap 4 utility class names in FOMO's `calendar.html`; upstream's own `data-bs-url` / `dataset.url` mismatch, which FOMO's `data-url` avoids).

## Task Commits

1. **Task 1** - no commit by design (new venv outside the repo, migrated ignored database).
2. **Task 2** - no commit (no SYNC-07 fix needed).
3. **Task 3** - `4c5e730` (docs(38): record the tom_calendar override comparison for Phase 39).

`actuals.commits: 1` is the measured `git rev-list --count 29dfb30..HEAD` taken before the metadata commit that follows this file.

## Deviations from Plan

None - plan executed exactly as written. RESEARCH's forecast held: zero code changes were needed on tomtoolkit 3.1.0, so no paired notebook or runbook changed.

## Authentication Gates

None.

## Known Stubs

None.

## Threat Flags

None.

## Self-Check: PASSED

- 38-OVERRIDE-COMPARISON.md exists; commit 4c5e730 is an ancestor of HEAD.
- Fresh-venv log: exact `OK`, no `FAILED`/`skipped=`; hook log: Passed, exact `OK`.
- `crontab -l` matches `$HOME/tmp/phase38-crontab.bak` byte for byte; backup file present and ignored by git.
