# Phase 38: Sync with main - Pattern Map

**Mapped:** 2026-10-07
**Files analyzed:** 18 (9 conflict files, 5 config/CI/doc follow-ups, 4 PR/branch artifacts)
**Analogs found:** 18 / 18 (the analogs are `origin/main`'s own versions; this phase has no new application modules)

All line numbers for the branch refer to HEAD of `issue37-telescope-runs-calendar`; "main" means `git show origin/main:<path>`.
All paths are git-tracked source.

## File Classification

| File | Role | Data Flow | Analog (source of the pattern) | Match |
|------|------|-----------|--------------------------------|-------|
| `solsys_code/apps.py` | config (AppConfig) | request-response | branch `nav_items` (lines 92-102) + main `nav_items` (lines 8-15) | merge: one list |
| `solsys_code/admin.py` | config (admin) | CRUD | main `SolsysTargetAdmin` (whole file); branch local `TargetAdmin` (583-587) is dropped | main wins |
| `docs/conf.py` | config | batch | main `_skip_version_module` + `setup` (61-70); branch `autoapi_ignore` (line 70) | merge, edit |
| `solsys_code/tests/test_bootstrap5_rendering.py` | test | request-response | both imports | union import |
| `src/fomo/urls.py` | route | request-response | main urls (scout paths) + branch urls (19-39) | merge: union |
| `.gitignore`, `docs/design/design.rst`, `docs/notebooks.rst` | config/docs | - | keep both sides | union |
| `pyproject.toml` | config | - | main `[dev]` extras; one hunk conflicts | main + `graphifyy` |
| `.pre-commit-config.yaml` | config | - | main file (auto-merges) | edit 2 spots |
| `.github/workflows/testing-and-coverage.yml` | config (CI) | batch | main file | one-token edit |
| `.github/workflows/smoke-test.yml` | config (CI) | batch | main file line 43 | one-token edit |
| `CLAUDE.md`, `docs/installation.rst` | docs | - | main's text auto-merged | edit stale lines |
| `issue37-code-only` snapshot + PR #43 body | git/PR | - | 2026-09-01 "Sync code-only branch ..." commit | recipe in RESEARCH |

## Pattern Assignments

### `solsys_code/apps.py` (nav_items) - ONE method, two entries

Branch (lines 92-102) already has `nav_items` returning a context-driven dict. Main (lines 8-15) adds a second `def nav_items`. Do NOT keep both methods (the later silently wins). Target:

```python
    def nav_items(self):
        """
        Integration point for adding entries to the navbar (VIEW-02/D-03).
        """
        return [
            {
                'partial': f'{self.name}/partials/campaigns_nav_link.html',
                'context': 'src.templatetags.solsys_code_extras.campaigns_nav_link',
                'position': 'left',
            },
            {'partial': f'{self.name}/partials/navbar_list.html'},  # Rubin ToO menu, from main
        ]
```
Check: `git grep -c "def nav_items" solsys_code/apps.py` prints 1. Main's `target_detail_buttons`/`ready` hunk stays as the branch has it.
`navbar_list.html` arrives from main (`src/templates/solsys_code/partials/navbar_list.html`, an `<li class="nav-item dropdown">` with `scout_rubin_too` / `scout_rubin_too_stats` links).

### `solsys_code/admin.py` - main's TargetAdmin subclass replaces the local class

Branch imports (lines 1-5): `from tom_targets.models import Target`. Branch local class (583-587) and registration (593-594):
```python
class TargetAdmin(admin.ModelAdmin):  # noqa: D101
    list_display = ['name', 'type', 'ra', 'dec']
    ...
admin.site.unregister(Target)
admin.site.register(Target, TargetAdmin)
```
Resolution: delete that local class; change import to
```python
from tom_targets.admin import TargetAdmin
from tom_targets.models import Target, TargetName
```
(keeping the branch's `from solsys_code.models import (...)` block); add main's `TargetNameInline` and `SolsysTargetAdmin(TargetAdmin)` (search_fields `('name', 'aliases__name')`, `get_list_display` swapping in ra/dec for sidereal filter, `inlines = TargetAdmin.inlines + [TargetNameInline]`) verbatim from `origin/main:solsys_code/admin.py`; register once at the file end:
```python
admin.site.unregister(Target)
admin.site.register(Target, SolsysTargetAdmin)
```
Keep the branch's four `admin.site.register(CampaignRun, ...)` etc. lines (589-592) before it. Tests: `test_admin.TargetAdminChangelistAndTypeFilterTests`, `test_search.TestTargetAdminOverride`.

### `docs/conf.py`

Branch line 70: `autoapi_ignore = ['*/__main__.py', '*/_version.py', '*/local_settings.py']`. Result: `autoapi_ignore = ['*/__main__.py', '*/local_settings.py']` (drop `_version.py`, keep CR-03 secret guard). Add main's lines 61-70 verbatim:
```python
def _skip_version_module(app, what, name, obj, skip, options):
    # fomo._version must stay parsed (not in autoapi_ignore) so that ...
    return True if name == 'fomo._version' else None


def setup(app):
    """Register the Sphinx event handlers."""
    app.connect('autoapi-skip-member', _skip_version_module)
```
Keep branch `nbsphinx_allow_errors = True` (line 76). Optional: comment at ~line 51 mentions the removed sphinx-build hook.

### `solsys_code/tests/test_bootstrap5_rendering.py`

Branch line 26: `from django.test import SimpleTestCase` ; main line 14: `from django.test import tag`. Result: `from django.test import SimpleTestCase, tag` (isort order). `@tag('functional')` goes on `TestBootstrap5Rendering` (main line 22); `TestTemplatesUseBootstrap5DataAttributes(SimpleTestCase)` (branch line 312) stays untagged.

### `src/fomo/urls.py`

Imports (union):
```python
from solsys_code.scout_views import (
    RubinTooScoutListView,
    RubinTooScoutStatsView,
    ScoutTargetExportView,
    ScoutTargetListView,
)
from solsys_code.views import Ephemeris, MakeEphemerisView, ProtectedUserDeleteView
```
URL list: main's two shadow routes `targets/` (`scout_target_list`) and `targets/export/` (`scout_target_export`) go first (with main's comment; earlier pattern wins), then branch's `observatory/`, `ephem/`, `makeephem/`, main's `scout/rubin-too/` and `scout/rubin-too/stats/`, branch's `calendar/` (with DISPLAY-09 comment), `campaigns/`, `alerts/`, `users/<int:pk>/delete/`, and last `path('', include('tom_common.urls'))`. Keep the branch's explanatory comments (urls.py lines 25-37). Expected warning: `urls.W005` for namespace `calendar`.

### `.pre-commit-config.yaml` (auto-merges; two edits afterwards)

Main's file as arrived (hooks `pre-executed-nb-never-execute` at `lincc-frameworks/pre-commit-hooks` v0.2.2; `django-test` local hook). Edits per D-06/D-07 (exact text in 38-RESEARCH.md "Hook entries after D-06 / D-07"):
- `files: ^docs/pre_executed/.*\.ipynb$` -> `^docs/notebooks/pre_executed/.*\.ipynb$`; `args: ["docs/pre_executed/"]` -> `["docs/notebooks/pre_executed/"]`
- `entry: bash -c "coverage run manage.py test --exclude-tag functional && coverage html"` -> add `--exclude-tag ephemeris_segfault`; add `# Takes ~10 minutes. For work-in-progress commits: SKIP=django-test git commit ...` comment.
Existing convention: comments sit above `- repo:` / hook blocks; ruff hooks both `rev: v0.16.9`.
Executors must use `SKIP=django-test` on task commits (suite is ~613 s).

### `.github/workflows/testing-and-coverage.yml` and `smoke-test.yml`

Main, unit-test step:
```yaml
    - name: Run Django unit tests with coverage
      run: |
        coverage run manage.py test --exclude-tag functional
        coverage xml
```
-> `coverage run manage.py test --exclude-tag functional --exclude-tag ephemeris_segfault`. `smoke-test.yml` line 43 `python manage.py test --exclude-tag functional` gets the same token. Leave `functional-tests` job (`python manage.py test --tag functional`) untouched. Workflow style: `actions/checkout@v7`, 2-space YAML, matrix 3.10-3.12.
Note: `testing-and-coverage.yml` triggers on `push: branches: [ main ]` and `pull_request` to main only, so a push of the v2.5 branch does not run CI (RESEARCH Open Question 1).

### `pyproject.toml`

One conflict in `[project.optional-dependencies] dev` (main lines): take
```toml
    "factory_boy >3.2.1,<3.4", # ...
    "ipython",
    "jupyter",
    "playwright", # ...
    "coverage", # Used to report total code coverage of the Django test suite
    "pre-commit",
    "ruff>=0.16", # ...
```
plus the branch's `"graphifyy"` line; drop `pytest`, `pytest-cov`, `ruff==0.2.1`. Everything else (floors `tomtoolkit>=3.1.0`, `tom_jpl>=0.3.0`, `timezonefinder>=6.0`, ruff exclude union, main's coverage block) auto-merges.

### `.gitignore`, `docs/design/design.rst`, `docs/notebooks.rst`

Union of both sides in order: .gitignore branch blocks then main's `src/_static/`; design.rst toctree = branch seven entries + `target_origin_tracking`, `scout_element_history` + shared `fink_sso_support`, `target_search_limitations`; notebooks.rst branch "Demonstration Notebooks" toctree then `Scout candidate lifecycle <notebooks/scout_lifecycle_exploration>`.

### `CLAUDE.md` / `docs/installation.rst` (follow-up commits, not in merge)

Edit stale lines only: `(v0.2.1)` -> `(v0.16.9)` (Commands comment, D-07 note); Key Dependencies drop `pytest`, `pytest-cov`, `tom_registration`, `ruff 0.2.1+` -> `0.16.9`; Configuration line "(ruff, pytest, Sphinx, validation)"; add `--exclude-tag ephemeris_segfault` to test commands. Add `timezonefinder>=6.0` bullet to `docs/installation.rst` mirroring its existing `tomtoolkit>=3.1.0` bullet. Keep `pre-commit run ruff --all-files` / `ruff-format` lines; no bare `ruff check` lines.

### Lint fix: `solsys_code/management/commands/backfill_lco_observations.py:176` (SIM103)

```python
    if created_before is not None and created > created_before:
        return False
    return True
```
-> `return not (created_before is not None and created > created_before)`; flag to developer (D-09), no notebook update needed.

### PR #43 refresh (git recipe, not code)

Analog: the 2026-09-01 "Sync code-only branch ... through v2.2" commit on `issue37-code-only` (`git log --oneline issue37-code-only` to find the style). Use a separate worktree (`git worktree add ../fomo_code_only issue37-code-only`), `merge -s ours origin/main`, `git read-tree -u --reset issue37-telescope-runs-calendar`, `git rm -r -q --cached .planning`, one snapshot commit, plain push. PR body via `gh pr edit 43 --body-file`; stays a draft. Run `git branch --show-current` before each branch-implicit command (CLAUDE.md).

## Shared Patterns

### Merge commit discipline (D-02)
Merge commit holds origin/main plus conflict-resolution edits only; reformat, lint, CI, CLAUDE.md edits are separate later commits. Never run `ruff --fix` before the merge commit.

### Verification commands
`git grep -c "def nav_items" solsys_code/apps.py` (1); `git diff --name-only --diff-filter=U` (exactly the 9 files); `python manage.py check` (only urls.W005); `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` (not bare ruff); `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` (expect ~2178 OK).

### Code style for any edited Python
Single quotes, 120 col, Google docstrings, `# noqa: D101` on undocumented admin classes as the branch does; tests use `NonSiderealTargetFactory` for Targets.

## No Analog Found

| File | Reason |
|------|--------|
| `.planning/phases/38-sync-with-main/38-OVERRIDE-COMPARISON.md` | New note; no precedent, content defined in RESEARCH ("tom_calendar overrides": upstream `tom_calendar` byte-identical 3.0.1 vs 3.1.0, so no SYNC-07 fix) |
| PR body text | No prior template; follow D-12 sections |

## Metadata

**Analog search scope:** `origin/main` versions of apps.py, admin.py, urls.py, docs/conf.py, test_bootstrap5_rendering.py, .pre-commit-config.yaml, workflows; branch HEAD versions; `git diff 756680f origin/main`
**Pattern extraction date:** 2026-10-07
