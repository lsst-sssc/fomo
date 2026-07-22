---
phase: 260719-rmx
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - pyproject.toml
  - src/fomo/settings.py
  - src/templates/ephem.html
  - src/templates/ephem_form.html
  - src/templates/solsys_code_observatory/observatory_list.html
  - src/templates/solsys_code_observatory/observatory_detail.html
  - src/templates/solsys_code_observatory/observatory_create.html
  - src/templates/solsys_code_observatory/partials/navbar_list.html
  - solsys_code/forms.py
autonomous: true
requirements:
  - issue-45  # GitHub issue #45 — upgrade to tomtoolkit 3.0.0 final (non-calendar scope)

must_haves:
  truths:
    - "`./manage.py check` passes with tomtoolkit 3.0.0 final installed (app boots, no removed-module imports)"
    - "No template loads the Bootstrap 4 tag library or uses the BS4 data-toggle attribute"
    - "Dev DB migrated (tom_dataproducts 0015-0017, tom_common 0004) with existing ReducedDatum rows preserved via migrate_reduced_datums"
    - "ruff check and ruff format --check clean; pytest suite and solsys_code Django test suite pass"
  artifacts:
    - "pyproject.toml pinning tomtoolkit>=3.0.0 plus BS5/3.0-compatible tom_eso, tom_fink, tom-registration, tom_alertstreams"
    - "src/fomo/settings.py with django_bootstrap5 / crispy_bootstrap5 apps and CRISPY_TEMPLATE_PACK='bootstrap5'"
  key_links:
    - "INSTALLED_APPS bootstrap app names must match installed packages (django_bootstrap5, crispy_bootstrap5) or Django fails at startup"
    - "Each template `{% load %}` tag-library name must match the installed package or the template fails to render"
    - "crispy_bootstrap5 registers the 'bootstrap5' pack; CRISPY_TEMPLATE_PACK must equal it or crispy forms error"
---

<objective>
Upgrade FOMO from tomtoolkit 3.0.0a9 to 3.0.0 final (GitHub issue #45), covering the
non-calendar scope: dependency pins, Bootstrap 4 -> Bootstrap 5 migration in settings and
templates, the tom_eso hard-break fix, environment reinstall, and the new upstream DB
migrations. Work happens on the already-checked-out branch `issue45-tomtoolkit-3.0-upgrade`.

Purpose: 3.0.0 final removed `tom_common/session_utils.py` (breaking the installed
tom_eso 0.2.4) and dropped Bootstrap 4 for Bootstrap 5. The app will not boot until deps,
settings, and templates are aligned with the released packages.

Output: pyproject.toml + settings.py + templates + forms.py updated, upgraded packages
installed into the venv, dev DB migrated with ReducedDatum rows preserved, all quality
gates green.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md

# Pre-verified facts (trust these; no re-research needed)
# - django-bootstrap5 26.2 and crispy-bootstrap5 2026.3 are ALREADY installed in the venv.
# - tom_common/session_utils.py was REMOVED in 3.0.0 final; tom_eso 0.2.4 imports it (hard break) -> tom_eso>=0.3.1.
# - tom_alerts and tom_catalogs SURVIVE in 3.0.0 (only Hermes broker removed); existing TOM_ALERT_CLASSES/BROKERS stay valid.
# - New upstream migrations: tom_dataproducts 0015-0017 (ReducedDatum refactor) + tom_common 0004 (delete UserSession).
#   New command `migrate_reduced_datums` moves existing rows. Dev DB has 70 ReducedDatum rows.
# - venv: /home/tlister/venv/fomo_venv. tom-jpl is a LOCAL editable install at ~/git/tom_jpl -- do not touch it.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Dependency pins, settings, and Bootstrap 4 -> 5 source edits</name>
  <files>pyproject.toml, src/fomo/settings.py, src/templates/ephem.html, src/templates/ephem_form.html, src/templates/solsys_code_observatory/observatory_list.html, src/templates/solsys_code_observatory/observatory_detail.html, src/templates/solsys_code_observatory/observatory_create.html, src/templates/solsys_code_observatory/partials/navbar_list.html, solsys_code/forms.py</files>
  <action>
Edit dependency pins in pyproject.toml (the `dependencies` list, lines ~20-24):
- Replace the `tomtoolkit>=3.0.0a9` line (including its trailing inline comment) with `"tomtoolkit>=3.0.0",` — drop the now-stale pre-release comment entirely.
- Bump `tom_alertstreams` to `>=1.3.0`, `tom_fink` to `>=2.0.1`, `tom-registration` to `>=2.0.1`, `tom_eso` to `>=0.3.1`. Leave numpy, sbpy, sorcha untouched.

Edit src/fomo/settings.py:
- In INSTALLED_APPS (line ~50-51) rename the `bootstrap4` app entry to `django_bootstrap5` and the `crispy_bootstrap4` app entry to `crispy_bootstrap5`. Leave `crispy_forms` as-is.
- Set `CRISPY_TEMPLATE_PACK` (line ~109) from the BS4 value to `'bootstrap5'`.

Templates — swap the tag library the `{% load %}` line pulls in, from the BS4 library name to `django_bootstrap5`, in each of: ephem.html (line 2), observatory_list.html (line 2), observatory_detail.html (line 2), observatory_create.html (line 2). In ephem_form.html (line 2) the load line also pulls in `crispy_forms_tags` alongside it — keep `crispy_forms_tags`, only swap the bootstrap library name. The `{% bootstrap_form form %}` tag in observatory_create.html (line 16) keeps the same tag name under django_bootstrap5, so leave the tag call itself unchanged.

In src/templates/solsys_code_observatory/partials/navbar_list.html (line 2), change the dropdown attribute from the BS4 `data-toggle` form to the BS5 `data-bs-toggle` form (value stays `"dropdown"`).

In solsys_code/forms.py (line ~70), change the outer Div `css_class` from the removed BS4 `'form-row'` grid class to `'row'` (BS5 dropped form-row; the two inner `css_class='col'` Divs stay unchanged).

Then run `ruff format .` and `ruff check . --fix` to keep the two edited Python files (settings.py, forms.py) clean.
  </action>
  <verify>
    <automated>! grep -rn "load bootstrap4" src/templates/ && ! grep -rn 'data-toggle=' src/templates/ && grep -rl "load django_bootstrap5" src/templates/ | wc -l | grep -qx 5 && grep -q "tomtoolkit>=3.0.0\"" pyproject.toml && grep -q "django_bootstrap5" src/fomo/settings.py && grep -q "crispy_bootstrap5" src/fomo/settings.py && grep -q "CRISPY_TEMPLATE_PACK = 'bootstrap5'" src/fomo/settings.py && ! grep -q "form-row" solsys_code/forms.py && ruff check solsys_code/forms.py src/fomo/settings.py && ruff format --check solsys_code/forms.py src/fomo/settings.py</automated>
  </verify>
  <done>pyproject.toml pins tomtoolkit>=3.0.0 and the four bumped TOM packages; settings.py uses django_bootstrap5/crispy_bootstrap5 with CRISPY_TEMPLATE_PACK='bootstrap5'; all 5 templates load django_bootstrap5, navbar uses data-bs-toggle, forms.py uses css_class='row'; ruff clean on edited Python files.</done>
</task>

<task type="auto">
  <name>Task 2: Install upgraded packages, back up DB, run migrations</name>
  <files>src/fomo_db.sqlite3 (untracked dev DB — not committed)</files>
  <action>
Operate inside the active venv (/home/tlister/venv/fomo_venv). Do NOT reinstall or modify tom-jpl (local editable install at ~/git/tom_jpl).

1. Install the upgraded upstream packages by name (this only touches the named packages, leaving the tom-jpl editable install intact):
   `pip install -U 'tomtoolkit>=3.0.0' 'tom_eso>=0.3.1' 'tom_fink>=2.0.1' 'tom-registration>=2.0.1' 'tom_alertstreams>=1.3.0'`

2. Back up the dev DB BEFORE migrating (the .bak file is untracked and must NOT be committed — DB files are gitignored):
   `cp src/fomo_db.sqlite3 src/fomo_db.sqlite3.bak-pre-3.0`

3. Apply the new upstream schema migrations (tom_dataproducts 0015-0017 refactor ReducedDatum into typed sub-models; tom_common 0004 deletes UserSession):
   `./manage.py migrate`

4. Move the 70 existing ReducedDatum rows into the new typed sub-models with the new upstream command:
   `./manage.py migrate_reduced_datums`

If `./manage.py migrate` errors on a removed-module import (e.g. tom_eso still importing session_utils), that means the pip upgrade did not take — re-run step 1 and confirm `pip show tom_eso` reports >=0.3.1 before retrying.
  </action>
  <verify>
    <automated>python -c "import tom_eso, tomtoolkit; from importlib.metadata import version; assert tuple(int(x) for x in version('tomtoolkit').split('.')[:2]) >= (3,0), version('tomtoolkit')" && ./manage.py showmigrations tom_dataproducts tom_common | grep -E '0017|0004' | grep -vq '\[ \]'</automated>
  </verify>
  <done>tomtoolkit>=3.0.0, tom_eso>=0.3.1, tom_fink>=2.0.1, tom-registration>=2.0.1, tom_alertstreams>=1.3.0 installed in the venv; DB backed up to src/fomo_db.sqlite3.bak-pre-3.0 (uncommitted); `./manage.py migrate` and `./manage.py migrate_reduced_datums` completed with tom_dataproducts 0017 and tom_common 0004 applied.</done>
</task>

<task type="auto">
  <name>Task 3: Quality gates — Django check, ruff, and both test suites</name>
  <files>(verification only — no file changes)</files>
  <action>
Run the full quality-gate sweep and confirm each is green. Fix any failure surfaced by these gates before declaring the plan done (a leftover BS4 load or a missed dep bump will surface here).

1. `./manage.py check` — Django system check; proves the app boots with the 3.0.0 apps and no removed-module imports.
2. `ruff check .` and `ruff format --check .` — must stay clean repo-wide.
3. `python -m pytest` — the packaging/pure-Python suite (collects tests/, src/, docs/ only per pyproject testpaths).
4. `./manage.py test solsys_code` — the DB-dependent Django app suite. CAUTION: importing solsys_code.ephem_utils triggers a ~1.6 GB SPICE-kernel download on first use; ~/.cache/sorcha should already be populated on this machine, so this should be fast. This run also builds a fresh test DB, which re-applies all upstream migrations and independently confirms they are clean.
  </action>
  <verify>
    <automated>./manage.py check && ruff check . && ruff format --check . && python -m pytest -q && ./manage.py test solsys_code</automated>
  </verify>
  <done>`./manage.py check` reports no issues; ruff check and ruff format --check clean repo-wide; pytest suite passes; `./manage.py test solsys_code` passes (test DB builds cleanly with the new migrations).</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| PyPI -> local venv | Upgraded packages pulled over the network during `pip install` |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-260719-SC | Tampering | pip install of tomtoolkit/tom_eso/tom_fink/tom-registration/tom_alertstreams | low | accept | All five are pre-existing FOMO dependencies already pinned in pyproject.toml — this is a version bump of trusted upstream TOM Toolkit packages, not introduction of new/unknown packages. No new-package legitimacy checkpoint required. |
| T-260719-01 | Denial of Service | `./manage.py migrate` on dev DB | low | mitigate | DB backed up to src/fomo_db.sqlite3.bak-pre-3.0 before migrating (Task 2) so a failed migration is recoverable. |
</threat_model>

<verification>
- Branch is `issue45-tomtoolkit-3.0-upgrade` (already checked out from main).
- No template references the Bootstrap 4 tag library or the BS4 data-toggle attribute; forms.py has no form-row class.
- pyproject.toml, settings.py, and templates align with the released django_bootstrap5 / crispy_bootstrap5 / tomtoolkit 3.0.0 packages.
- Migrations applied and existing ReducedDatum rows preserved; DB backup present and uncommitted.
- `./manage.py check`, ruff check, ruff format --check, pytest, and `./manage.py test solsys_code` all pass.
- Out of scope (do NOT touch): tom_calendar templates / calendar override (issue37 branch), demo notebooks (no behavior change to telescope_runs.py / load_telescope_runs.py / sync_*_observation_calendar.py — this is a dependency + UI-class migration, so the CLAUDE.md paired-notebook rule does not apply).
</verification>

<success_criteria>
All three tasks complete, every quality gate in Task 3 green, the DB migrated with ReducedDatum rows preserved, and no Bootstrap 4 / removed-module references remaining in the non-calendar scope.
</success_criteria>

<output>
Create `.planning/quick/260719-rmx-upgrade-to-tomtoolkit-3-0-0-final-and-bo/260719-rmx-SUMMARY.md` when done.
</output>
