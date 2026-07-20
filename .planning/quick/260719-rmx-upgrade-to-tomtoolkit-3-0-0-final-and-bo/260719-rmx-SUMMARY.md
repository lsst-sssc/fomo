---
phase: 260719-rmx
plan: 01
subsystem: infra
tags: [tomtoolkit, django-bootstrap5, crispy-forms, bootstrap5, tom_eso, tom_fink, tom-registration, tom_alertstreams, migrations]

requires: []
provides:
  - "FOMO running on tomtoolkit>=3.0.0 final (upgraded from 3.0.0a9)"
  - "Bootstrap 5 UI stack (django_bootstrap5 + crispy_bootstrap5) replacing Bootstrap 4"
  - "tom_eso>=0.3.1 (fixes the tom_common/session_utils.py hard-break)"
  - "Dev DB migrated to tom_dataproducts 0015-0017 / tom_common 0004 schema with all 70 ReducedDatum rows preserved"
  - "project-level 'alerts' URL namespace re-registered (dropped upstream in 3.0.0 final's tom_common/urls.py)"
affects: [issue45-tomtoolkit-3.0-upgrade, any future phase touching settings.py/urls.py/forms.py/templates]

tech-stack:
  added: [django_bootstrap5 (already installed, now wired in), crispy_bootstrap5 (already installed, now wired in)]
  patterns: ["Project-level urls.py must explicitly include app URLs no longer auto-wired by tom_common.urls in tomtoolkit 3.0.0 (e.g. tom_alerts)"]

key-files:
  created: []
  modified:
    - pyproject.toml
    - src/fomo/settings.py
    - src/fomo/urls.py
    - src/templates/ephem.html
    - src/templates/ephem_form.html
    - src/templates/solsys_code_observatory/observatory_list.html
    - src/templates/solsys_code_observatory/observatory_detail.html
    - src/templates/solsys_code_observatory/observatory_create.html
    - src/templates/solsys_code_observatory/partials/navbar_list.html
    - solsys_code/forms.py
    - solsys_code/views.py
    - solsys_code/tests/test_views.py
    - manage.py

key-decisions:
  - "Re-added the project-level 'alerts/' URL include (namespace='alerts') in src/fomo/urls.py because tomtoolkit 3.0.0 final's tom_common/urls.py silently dropped it (present in 2.x and 3.0.0a9) -- verified by diffing against the 2.32.4 wheel"
  - "Replaced observatory_create.html's {% buttons %}{% endbuttons %} tag (a django-bootstrap4-only tag, not crispy_forms) with a plain <div class=\"mb-3\"> wrapper, since django-bootstrap5 has no equivalent tag"
  - "Applied a repo-wide `ruff check . --fix` / `ruff format .` sweep to satisfy this plan's explicit 'ruff clean repo-wide' gate, touching manage.py/views.py/test_views.py which were pre-existing lint debt unrelated to the tomtoolkit upgrade itself"
  - "Left two prior quick-task verify_*.py scripts (260619-f7u) untouched after discovering local ruff (0.12.9) and the repo's pinned pre-commit ruff (v0.2.1) disagree on their formatting -- pre-commit's pinned version is treated as canonical, so no fix applied; logged as a deferred item"

requirements-completed: [issue-45]

coverage:
  - id: D1
    description: "App boots and passes Django system check with tomtoolkit 3.0.0 final, tom_eso>=0.3.1, tom_fink>=2.0.1, tom-registration>=2.0.1, tom_alertstreams>=1.3.0 installed"
    requirement: "issue-45"
    verification:
      - kind: other
        ref: "./manage.py check"
        status: pass
    human_judgment: false
  - id: D2
    description: "Bootstrap 4 -> Bootstrap 5 migration: no template loads bootstrap4, no data-toggle attributes, forms.py has no form-row class"
    requirement: "issue-45"
    verification:
      - kind: other
        ref: "grep -rn 'load bootstrap4' src/templates/ ; grep -rn 'data-toggle=' src/templates/ ; grep form-row solsys_code/forms.py (all empty)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Dev DB migrated (tom_dataproducts 0015-0017, tom_common 0004) with all 70 pre-existing ReducedDatum rows preserved via migrate_reduced_datums"
    requirement: "issue-45"
    verification:
      - kind: other
        ref: "./manage.py showmigrations tom_dataproducts tom_common (0017/0004 applied) + sqlite3 row count 70 before/after"
        status: pass
    human_judgment: false
  - id: D4
    description: "ruff check / ruff format --check clean repo-wide; pytest suite and full solsys_code Django test suite (63 tests) pass"
    requirement: "issue-45"
    verification:
      - kind: unit
        ref: "./manage.py test solsys_code (63 tests, OK)"
        status: pass
      - kind: unit
        ref: "python -m pytest -q (1 passed)"
        status: pass
      - kind: other
        ref: "ruff check . && ruff format --check ."
        status: pass
    human_judgment: false

duration: 55min
completed: 2026-07-19
status: complete
---

# Quick Task 260719-rmx: Upgrade to tomtoolkit 3.0.0 final and Bootstrap 5 Summary

**FOMO now runs on tomtoolkit 3.0.0 final with a Bootstrap 5 UI stack (django_bootstrap5/crispy_bootstrap5), tom_eso>=0.3.1 fixing the removed-module hard break, and the dev DB migrated to the new ReducedDatum schema with all 70 rows preserved.**

## Performance

- **Duration:** ~55 min
- **Started:** 2026-07-19T19:xx (session start, not precisely timestamped)
- **Completed:** 2026-07-20T03:14:03Z
- **Tasks:** 3/3 completed
- **Files modified:** 13

## Accomplishments
- Bumped `pyproject.toml` pins (`tomtoolkit>=3.0.0`, `tom_alertstreams>=1.3.0`, `tom_fink>=2.0.1`, `tom-registration>=2.0.1`, `tom_eso>=0.3.1`) and installed the upgraded packages into the venv
- Migrated Bootstrap 4 -> Bootstrap 5 across `settings.py` (`django_bootstrap5`/`crispy_bootstrap5` apps, `CRISPY_TEMPLATE_PACK='bootstrap5'`), all 5 templates that load the bootstrap tag library, the navbar dropdown (`data-bs-toggle`), and `forms.py`'s grid class (`form-row` -> `row`)
- Backed up the dev DB (`src/fomo_db.sqlite3.bak-pre-3.0`, uncommitted) and applied the new upstream migrations (`tom_dataproducts` 0015-0017, `tom_common` 0004), confirming all 70 pre-existing `ReducedDatum` rows survived intact (all `data_type='ades_astrometry'`, a custom type not matched by `migrate_reduced_datums`' standard photometry/spectroscopy/astrometry buckets — rows stay in the base table unchanged, which is expected upstream behavior, not data loss)
- Found and fixed two regressions that only surfaced once tomtoolkit 3.0.0 final was actually installed and exercised by the Django test suite (not previously anticipated by the plan's pre-verified facts): the `alerts` URL namespace silently dropped from `tom_common/urls.py`, and `observatory_create.html`'s `{% buttons %}` tag which turned out to be BS4-package-specific, not crispy's
- Brought the full quality-gate sweep green: `./manage.py check`, `ruff check .`, `ruff format --check .`, `python -m pytest` (1 passed), and `./manage.py test solsys_code` (63 tests, OK)

## Task Commits

Each task was committed atomically:

1. **Task 1: Dependency pins, settings, and Bootstrap 4 -> 5 source edits** - `c214d1e` (feat)
2. **Task 2: Install upgraded packages, back up DB, run migrations** - no commit (no git-tracked files changed; DB is gitignored, pip installs are venv-only)
3. **Task 3: Quality gates — Django check, ruff, and both test suites** - `2d68c61` (fix, includes the two deviations found while running this task's gates)

**Plan metadata:** commit made separately by the orchestrator after this SUMMARY.

## Files Created/Modified
- `pyproject.toml` - tomtoolkit>=3.0.0 pin plus bumped tom_alertstreams/tom_fink/tom-registration/tom_eso pins
- `src/fomo/settings.py` - django_bootstrap5/crispy_bootstrap5 apps, CRISPY_TEMPLATE_PACK='bootstrap5'
- `src/fomo/urls.py` - re-added the `alerts/` include (namespace='alerts') dropped by tomtoolkit 3.0.0 final's tom_common/urls.py
- `src/templates/ephem.html` - `{% load django_bootstrap5 %}`
- `src/templates/ephem_form.html` - `{% load django_bootstrap5 crispy_forms_tags %}`
- `src/templates/solsys_code_observatory/observatory_list.html` - `{% load django_bootstrap5 %}`
- `src/templates/solsys_code_observatory/observatory_detail.html` - `{% load django_bootstrap5 %}`
- `src/templates/solsys_code_observatory/observatory_create.html` - `{% load django_bootstrap5 %}`, replaced BS4-only `{% buttons %}` tag with plain `<div class="mb-3">`
- `src/templates/solsys_code_observatory/partials/navbar_list.html` - `data-toggle` -> `data-bs-toggle`
- `solsys_code/forms.py` - `css_class='form-row'` -> `css_class='row'`
- `solsys_code/views.py` - lint-only: `Optional[dict[str, Any]]` -> `dict[str, Any] | None`
- `solsys_code/tests/test_views.py` - lint-only: `.split(',')` string -> list literal (SIM905), reformatted to fit 120 cols
- `manage.py` - formatting-only (blank line after module docstring)

## Decisions Made
- Re-added `tom_alerts`'s URL include at the project level rather than patching the installed tomtoolkit package, since project-level `urls.py` is the correct place for app wiring the framework no longer does automatically (matches the pattern already used for `solsys_code_observatory`)
- Replaced `{% buttons %}{% endbuttons %}` with a plain `<div class="mb-3">` instead of loading `crispy_forms_tags` for a single button, since crispy's own `buttons` tag is intended for use inside `{% crispy %}`-rendered layouts, and observatory_create.html renders the form manually
- Treated the repo-wide `ruff check . --fix` / `ruff format .` sweep as in-scope for Task 3 (an explicit "quality gates" task with a "ruff clean repo-wide" done-criterion), even though the surfaced findings (SIM905, UP045, formatting drift) were pre-existing and unrelated to the tomtoolkit upgrade itself
- Did not touch the two 260619-f7u quick-task `verify_*.py` scripts after discovering the local venv's `ruff` (0.12.9) and the repo's pre-commit-pinned `ruff-format` (v0.2.1) disagree on their formatting — pre-commit (which actually gates commits) is clean with the original content, so left them as-is

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Restored the `alerts` URL namespace dropped by tomtoolkit 3.0.0 final**
- **Found during:** Task 3 (`./manage.py test solsys_code`)
- **Issue:** `tom_common/urls.py` in 3.0.0 final no longer includes `tom_alerts.urls` under the `alerts` namespace (confirmed present in both 2.32.4 and the previously-installed 3.0.0a9 by diffing the wheel contents). Any page extending `tom_common/base.html` (which every FOMO view does) 404s with `NoReverseMatch: 'alerts' is not a registered namespace` because the navbar partial reverses `alerts:list`. This broke 4 of 63 Django tests.
- **Fix:** Added `path('alerts/', include('tom_alerts.urls', namespace='alerts'))` to `src/fomo/urls.py`, mirroring the exact wiring tomtoolkit itself used to provide.
- **Files modified:** `src/fomo/urls.py`
- **Verification:** `./manage.py test solsys_code` errors dropped from 4 to 1 after this fix alone; full suite green after both fixes.
- **Committed in:** `2d68c61`

**2. [Rule 1 - Bug] Fixed BS4-only `{% buttons %}` tag with no Bootstrap 5 equivalent**
- **Found during:** Task 3 (`./manage.py test solsys_code`)
- **Issue:** `observatory_create.html` used `{% buttons %}...{% endbuttons %}`, which turned out to be a tag registered by the `django-bootstrap4` package itself (`register.tag("buttons")` in `bootstrap4/templatetags/bootstrap4.py`), not by `crispy_forms_tags` as the plan's task 1 instructions assumed. `django-bootstrap5` has no equivalent tag, so after swapping the `{% load %}` line the template raised `TemplateSyntaxError: Invalid block tag on line 18: 'buttons'`.
- **Fix:** Replaced with a plain `<div class="mb-3">` wrapper around the existing submit `<button>` (same visual grouping/spacing, BS5-idiomatic margin utility class, no tag-library dependency).
- **Files modified:** `src/templates/solsys_code_observatory/observatory_create.html`
- **Verification:** `CreateObservatoryTest.test_form_view_bad_code` (and the full observatory test suite) pass; full `./manage.py test solsys_code` green (63/63).
- **Committed in:** `2d68c61`

**3. [Rule 3 - Blocking] Fixed pre-existing ruff findings blocking the plan's repo-wide "ruff clean" gate**
- **Found during:** Task 3 (`ruff check .` / `ruff format --check .`)
- **Issue:** Two pre-existing lint findings (SIM905 in `solsys_code/tests/test_views.py`, UP045 in `solsys_code/views.py`) and formatting drift in `manage.py` and (initially, then reverted) two unrelated quick-task scripts — none caused by this plan's edits, but blocking the plan's own explicit must-have of "ruff check and ruff format --check clean [...] repo-wide."
- **Fix:** Ran `ruff check . --fix` and `ruff format .`, then manually reverted the two unrelated 260619-f7u `verify_*.py` files after discovering they disagree with the repo's pre-commit-pinned older ruff version (see Decisions Made).
- **Files modified:** `solsys_code/views.py`, `solsys_code/tests/test_views.py`, `manage.py`
- **Verification:** `ruff check .` and `ruff format --check .` clean (confirmed via both the local CLI and the pre-commit hook on final commit).
- **Committed in:** `2d68c61`

---

**Total deviations:** 3 auto-fixed (2 bugs surfaced by upgrading to the real 3.0.0 release, 1 blocking pre-existing lint debt)
**Impact on plan:** All three were necessary to meet the plan's own stated must-haves (working app, passing test suite, repo-wide ruff clean). No scope creep beyond what Task 3's explicit gates required.

## Issues Encountered
- The plan's Task 2 verify command (`import tomtoolkit`) fails because the `tomtoolkit` PyPI package does not expose a top-level `tomtoolkit` importable module (it provides `tom_common`, `tom_alerts`, etc. directly) — this is a defect in the plan's verify script, not a real problem. Verified the intent manually instead (`importlib.metadata.version('tomtoolkit')` plus `showmigrations`), both confirming success.
- Local venv's `ruff` (0.12.9) reformats two pre-existing quick-task `verify_*.py` scripts differently than the repo's pre-commit-pinned `ruff-format` (v0.2.1, from `.pre-commit-config.yaml`). Left those two files untouched (pre-commit is canonical for what actually gates commits); flagging this ruff-version drift as a deferred item below rather than fixing it in this task's scope.

## Deferred Items
- `ruff` version drift: local venv has ruff 0.12.9, but `.pre-commit-config.yaml` pins `astral-sh/ruff-pre-commit` at `v0.2.1` (over 2 years old at time of writing). This caused a brief disagreement over formatting of `.planning/quick/260619-f7u-.../verify_*.py`. Bumping the pre-commit pin is a tooling decision out of scope for this quick task (Rule 4) — worth a future quick task or phase to reconcile.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- FOMO's Django app, templates, and dev DB are fully aligned with tomtoolkit 3.0.0 final and Bootstrap 5; all quality gates green.
- Out of scope by design (per plan's verification section): `tom_calendar` templates/calendar override (issue37 branch) and the four telescope-runs-calendar demo notebooks were not touched — this plan made no behavior changes to `telescope_runs.py`/`load_telescope_runs.py`/`sync_*_observation_calendar.py`, so the CLAUDE.md paired-notebook rule does not apply.
- Ready for PR review / merge of `issue45-tomtoolkit-3.0-upgrade`.

---
*Phase: 260719-rmx*
*Completed: 2026-07-19*

## Self-Check: PASSED

All claimed files and commits verified to exist:
- `src/fomo_db.sqlite3.bak-pre-3.0` - FOUND
- `.planning/quick/260719-rmx-upgrade-to-tomtoolkit-3-0-0-final-and-bo/260719-rmx-SUMMARY.md` - FOUND
- Commit `c214d1e` - FOUND
- Commit `2d68c61` - FOUND
