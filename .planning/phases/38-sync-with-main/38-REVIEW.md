---
phase: 38-sync-with-main
reviewed: 2026-10-07T00:00:00Z
depth: deep
files_reviewed: 27
files_reviewed_list:
  - CLAUDE.md
  - docs/conf.py
  - docs/design/design.rst
  - docs/index.rst
  - docs/installation.rst
  - docs/notebooks.rst
  - .github/workflows/smoke-test.yml
  - .github/workflows/testing-and-coverage.yml
  - .gitignore
  - .pre-commit-config.yaml
  - pyproject.toml
  - solsys_code/admin.py
  - solsys_code/allocation_projector.py
  - solsys_code/apps.py
  - solsys_code/campaign_tables.py
  - solsys_code/campaign_views.py
  - solsys_code/management/commands/backfill_lco_observations.py
  - solsys_code/management/commands/check_unattended.py
  - solsys_code/management/commands/load_telescope_runs.py
  - solsys_code/management/commands/reconcile_campaign_runs.py
  - solsys_code/notifications.py
  - solsys_code/solsys_code_observatory/views.py
  - solsys_code/templatetags/calendar_display_extras.py
  - solsys_code/tests/test_bootstrap5_rendering.py
  - solsys_code/tests/test_status_vocabulary.py
  - src/fomo/settings.py
  - src/fomo/urls.py
findings:
  critical: 1
  warning: 2
  info: 4
  total: 7
status: issues_found
---

# Phase 38: Code Review Report

**Reviewed:** 2026-10-07
**Depth:** deep
**Files Reviewed:** 27
**Status:** issues_found

## Summary

This review covers the merge of `origin/main` (a910c17) into `issue37-telescope-runs-calendar`
(merge `e12158c`), the ruff 0.16.9 reformat (`ff8dd3c`), the SIM103 fix (`398fa26`), the CI and
pre-commit edits (`d535ef4`) and the docs commit (`22e1b0a`). I compared each file against both
`origin/main` and the pre-merge head `088f73b`. I also checked runtime behaviour read-only against
the installed tomtoolkit 3.1.0 / Django 5.2.17 environment: URL resolution, template lookup, and
`makemigrations --check --dry-run`, which reported no changes.

Checked and found correct:

- **`apps.py`:** there is one `nav_items` and it returns both entries. The campaigns link sets
  `position: 'left'`, and main's Rubin ToO partial uses the default, which is also `left`.
  `data_services` includes `ScoutDataService`.
- **`admin.py`:** `Target` is registered once, through `SolsysTargetAdmin`.
  `TargetAdmin.inlines` (`[TargetExtraInline]`) plus `TargetNameInline` gives no duplicate inline.
  The branch's old local `TargetAdmin` has been removed, so nothing shadows the imported name.
- **`urls.py`:** each shadow route comes before `tom_common.urls`. This covers `targets/`,
  `targets/export/`, `calendar/`, `campaigns/`, `users/<pk>/delete/` and `scout/rubin-too*`.
- **`docs/conf.py`:** it keeps the `*/local_settings.py` exclusion from `autoapi_ignore` and adds
  main's `_skip_version_module`/`setup`. It correctly drops `*/_version.py` from the ignore list so
  the `fomo/__init__.py` import still resolves.
- **The SIM103 rewrite** in `_within_created_window` gives the same result as before.
  `not (A and B)` returns False exactly when the old `if A and B: return False` did, and True
  otherwise.
- **The CI/hook exclusion tag:** `ephemeris_segfault` is on exactly one class, `TestEphemeris` at
  `solsys_code/tests/test_views.py:98`.
- **The reformatted modules:** the changes are only quote-style and line-wrap changes, with no
  change in behaviour.
- **Settings the branch depends on:** every `settings.X` the branch reads still exists after the
  auto-merge. That covers `FOMO_*`, `FACILITIES`, `HOOKS`, `TOM_FACILITY_CLASSES`,
  `GENERAL_SEARCH_FUNCTIONS` and `EMAIL_BACKEND`.

Problems found:

- **Blocker (CR-01):** the `urls.py` conflict resolution restores the `alerts/` include. Main
  deliberately deleted it because tomtoolkit 3.1 no longer installs `tom_alerts`.
- **Deployment risk (WR-01):** the merged `local_settings` import path does not match main's.
- **CI coverage gap (WR-02):** no CI job now runs `TestEphemeris`.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: The merge restores main's deleted `alerts/` route for an app that is no longer installed. Every page view 500s.

**File:** `src/fomo/urls.py:32-35`
**Issue:** Main's commit `ada2000` ("Updates for tom_toolkit 3.1") removed
`path('alerts/', include('tom_alerts.urls', namespace='alerts'))`, with the commit note
"tom_alerts is gone/going". The same commit switched `INSTALLED_APPS` to
`TOMTOOLKIT_INSTALLED_APPS + [...]`, and tomtoolkit 3.1.0's `TOMTOOLKIT_INSTALLED_APPS` does not
include `tom_alerts`. Plan 38-01 resolved the conflict by keeping the branch's `alerts/` entry and
its comment. The comment now says something false: "tom_alerts is still an installed app".

I verified the result against the merged settings:

- `apps.is_installed('tom_alerts')` is `False`, but `/alerts/query/list/` still resolves to
  `tom_alerts.views.BrokerQueryListView`.
- Calling that view as a logged-in user raises
  `TemplateDoesNotExist: tom_alerts/brokerquery_list.html`. Because the app is not installed, the
  `APP_DIRS` template loader never searches its templates. The query create, update and delete
  views would fail the same way.
- On a fresh install, `migrate` never creates `tom_alerts_brokerquery`, because the app is not
  installed. The `RunQueryView` and list querysets would also fail with "no such table".
  Developer databases only have the table because it predates the switch.
- `CreateTargetFromAlertView` and `SubmitAlertUpstreamView` redirect instead of rendering, so they
  still run. `get_service_classes()` falls back to tom_alerts' built-in default brokers (Lasair,
  ALeRCE, Gaia, Fink, Scout). This leaves a target-creating POST endpoint live, run from an app the
  project has dropped.

The include was first added in quick task `260719-rmx` only so the `alerts` namespace would
resolve for `tom_alerts`' own navbar partial. That partial is no longer rendered, and nothing in
tomtoolkit 3.1.0's templates or this repo reverses `alerts:*`. No test does either.
`manage.py check` stays quiet, so the full test suite cannot catch this.

**Fix:** Take main's side. Delete the entry and its comment:

```python
    path('alerts/', include('tom_alerts.urls', namespace='alerts')),
```

Delete that line along with the three comment lines above it (lines 32-34). Also add a test that
`/alerts/query/list/` returns 404, so the route cannot quietly come back in a later merge.

## Warnings

### WR-01: The `local_settings.py` import path differs from main's. A host set up for main would silently fall back to dev settings.

**File:** `src/fomo/settings.py:414-423` (with `docs/installation.rst:112-119` and
`.planning/phases/38-sync-with-main/38-PR43-BODY.md:45`)
**Issue:** `origin/main` loads overrides with `from local_settings import *` and a bare
`except ImportError: pass`. That is a top-level module found on `sys.path`, i.e. the repo root or
`src/`. The merged settings use the branch's `from fomo.local_settings import *` (branch-only
commit `c0f883d`) and swallow the `ImportError` when `exc.name == 'fomo.local_settings'`.

After PR #43 lands, any host that followed main's layout (`src/local_settings.py` or
`<repo>/local_settings.py`) no longer loads its overrides, and nothing reports it. On that host:

- The web process runs with `DEBUG=True`, the committed `SECRET_KEY` and the dev `ALLOWED_HOSTS`.
- Cron-run management commands such as `run_unattended` run with empty LCO/SOAR API keys and the
  console email backend. Their failure emails are printed to stdout instead of being sent.

Phase 38's job includes checking this deployment contract (emphasis 2). Neither the PR body's
Settings checklist line nor `installation.rst` says the file must now be at
`src/fomo/local_settings.py`. The CLAUDE.md line "fallback: no error on missing" describes exactly
this silent fallback.

**Fix:** Accept both locations during the transition, and say so in the PR body and
installation docs:

```python
try:
    from fomo.local_settings import *  # noqa
except ImportError as exc:
    if exc.name != 'fomo.local_settings':
        raise
    try:
        from local_settings import *  # noqa  -- main's pre-merge location
    except ImportError as exc2:
        if exc2.name != 'local_settings':
            raise
```

As an alternative, keep the single location but add one line to `38-PR43-BODY.md` and
`docs/installation.rst`: "local_settings.py moves to src/fomo/local_settings.py". Also consider
having `check_unattended` fail hard when `SECRET_KEY` still equals the committed dev key.

### WR-02: No CI job runs `TestEphemeris` any more. The core ephemeris view loses the CI coverage it had on main.

**File:** `.github/workflows/testing-and-coverage.yml:41`, `.github/workflows/smoke-test.yml:43`,
`.pre-commit-config.yaml:86`
**Issue:** On `origin/main`, the unit-test matrix ran `manage.py test --exclude-tag functional`,
which includes `TestEphemeris`. That is the only test of `Ephemeris`/`MakeEphemerisView`, the
first of FOMO's two core features. The merge adds `--exclude-tag ephemeris_segfault` to all three
places that run the suite, and no job runs the tagged class on its own. CLAUDE.md:97-101 says the
class "still runs when you name it directly", but nothing in CI does that.

The crash is in native ASSIST when the class shares a process with the rest of the suite (commit
`1850322`). The exclusion avoids that crash, but it also means a real ephemeris regression in a
future PR would reach `main` untested.

**Fix:** In `testing-and-coverage.yml`, add a separate step (or job) that runs the class in its
own process. If the crash does reproduce in CI, mark the step non-blocking so the run still shows
whether it passed:

```yaml
    - name: Run ephemeris tests in their own process
      run: |
        python manage.py test --tag ephemeris_segfault
```

## Info

### IN-01: `suppress_warnings = ['toc.excluded']` is kept even though the reason for it is gone

**File:** `docs/conf.py:45-51`
**Issue:** The new comment says the `sphinx-build` pre-commit hook that caused this warning was
removed, and that the setting "is kept because it is harmless". It is not harmless. With no
exclusion override left, the setting can only hide real problems: in the CI/ReadTheDocs builds, a
toctree entry that points at a page excluded by `exclude_patterns` would produce no warning.
**Fix:** Delete the setting and its comment. If you want to keep it, have the comment describe it
as dead configuration rather than harmless.

### IN-02: Stale stack lines remain in CLAUDE.md beside lines this phase updated

**File:** `CLAUDE.md:264, 267-268, 283, 310, 354, 369`
**Issue:** Plan 38-04 updated the stack section (lines 273-277, 282, 287) but left lines beside
them that are now wrong:

- "Django 2.1+": Django 5.2 is installed.
- "`crispy_bootstrap4` / Bootstrap 4": the site uses BS5.
- "tom_fink>=1.0.0": `pyproject.toml` pins `>=2.0.1`.
- "enforced by `ruff` and `black`" and "Profile: `black`": this phase deleted `[tool.black]` and
  `[tool.isort]` from `pyproject.toml`.
- "local_settings.py import (fallback: no error on missing)": gives no path; see WR-01.

Subagents read this file as instructions, so a wrong entry here can lead them to wrong actions.
**Fix:** Correct these lines in the same edit that fixes WR-01.

### IN-03: The ruff `exclude` list does not apply under the enforced pre-commit gate

**File:** `pyproject.toml:60-65`
**Issue:** The merged `exclude` adds `.planning`, `.planning/**` and
`docs/notebooks/ESO_How_to_download_data.ipynb`. However, ruff-pre-commit passes file paths
explicitly, and without `force-exclude = true` ruff ignores `exclude` for files passed that way.
So the 13 tracked `.py` files under `.planning/` and the ESO notebook are still linted and
formatted by `pre-commit run --all-files`. Also, setting `exclude` (rather than `extend-exclude`)
replaces ruff's default excludes (`.venv`, `build`, `dist`, ...) for a bare `ruff check .`.
**Fix:** Rename the key to `extend-exclude`, and add `force-exclude = true` under `[tool.ruff]`.

### IN-04: The `setuptools>=62` build floor is too low for a PEP 639 `license` string

**File:** `pyproject.toml:3-4, 49`
**Issue:** `license = "MIT"` (an SPDX string) and top-level `license-files` are supported only from
setuptools 77. Builds with build isolation fetch the latest setuptools and work. A
`--no-build-isolation` install against setuptools 62-76 fails with a validation error. This was
already the case on both sides, but the file was hand-resolved in this phase.
**Fix:** Change the floor to `"setuptools>=77"`.

---

_Reviewed: 2026-10-07_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
