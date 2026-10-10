---
phase: 40-notebook-isolation-attribution-page
reviewed: 2026-10-10T00:35:32Z
depth: deep
files_reviewed: 28
files_reviewed_list:
  - CLAUDE.md
  - docs/notebooks/README.md
  - docs/notebooks/fomo_nb_utils.py
  - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
  - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
  - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
  - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
  - docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb
  - docs/notebooks/pre_executed/telescope_runs_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/campaign_tables.py
  - solsys_code/campaign_views.py
  - solsys_code/tests/test_attribution_template.py
  - solsys_code/tests/test_bootstrap5_rendering.py
  - solsys_code/tests/test_campaign_attribution_views.py
  - solsys_code/tests/test_notebook_helper.py
  - solsys_code/tests/test_pre_executed_notebooks.py
  - src/templates/campaigns/approval_queue.html
  - src/templates/campaigns/attribution_queue.html
  - src/templates/campaigns/campaign_list.html
  - src/templates/campaigns/campaignrun_gap_analysis.html
  - src/templates/campaigns/campaignrun_submit_form.html
  - src/templates/campaigns/campaignrun_table.html
  - src/templates/campaigns/partials/attribution_return_state.html
  - src/templates/campaigns/submission_thanks.html
  - src/templates/solsys_code/partials/campaign_links.html
findings:
  critical: 1
  warning: 3
  info: 7
  total: 11
status: issues_found
---

# Phase 40: Code Review Report

**Reviewed:** 2026-10-10T00:35:32Z
**Depth:** deep
**Files Reviewed:** 28
**Status:** issues_found

## Summary

Scope: `git diff 04796447af044d5c1d6c81bc71b8f9efd5c65137 HEAD` for the 28 listed files. That covers the new
shared notebook helper (`docs/notebooks/fomo_nb_utils.py`), the eight pre-executed notebooks moved onto it,
the AST guard that replaces the old source checks, the Bootstrap 5.3 class sweep, and the attribution page's
return state (hidden inputs, `attribution_return_state()` whitelist, fragment redirect).

What I checked by running things:
- `python manage.py test` for test_notebook_helper, test_pre_executed_notebooks, test_attribution_template,
  test_campaign_attribution_views and test_bootstrap5_rendering (`--parallel --exclude-tag functional`):
  163 tests, OK. The two new functional Playwright tests run on their own: OK.
- `ruff check --no-fix` and `ruff format --check` on the changed Python files: clean.
- No Bootstrap 4 class names or data attributes remain in `src/templates/campaigns`,
  `src/templates/solsys_code`, `campaign_tables.py` or `campaign_views.py`.
- The attribution redirect is safe: the path is always `reverse('campaigns:attribution')`, the query holds
  only values that passed the regexes, the fragment is one of two constants, and `_return_state_inputs()`
  escapes its values through `format_html_join`. No open redirect, header injection or XSS path found.
  django-tables2 3.0.0's `RequestConfig` is silent on `EmptyPage`, so the D-27 "past the end" claim holds.

The main problem is in the helper's fail-closed logic. When the helper refuses a non-scratch database, the
kernel is still left with Django set up against that database. I reproduced this: after the `RuntimeError`,
the next statement wrote a table into the refused database. The static guard also misses a notebook that
re-routes the connection after setup. The rest are robustness, test-reliability and documentation items.

## Critical Issues

### CR-01: The helper refuses a non-scratch database but leaves Django set up against it, so the next cell writes there

**File:** `docs/notebooks/fomo_nb_utils.py:199-212`
**Issue:** `setup_scratch_django()` calls `django.setup()` (line 201) **before** it checks the resolved
database (lines 206-207). When the check fails (the case it exists for: a `local_settings.py` or
`DJANGO_SETTINGS_MODULE` whose `DATABASES` beats `FOMO_DATABASE_PATH`), the `except BaseException`
block only removes the scratch directory and re-raises. `apps.ready` stays `True`, and
`connections['default']` still points at the refused database. In an interactive Jupyter session the user
can run the next cell, or "Run All Below", and the ORM then writes straight into the developer database (or
whatever `local_settings` names). This is the outcome WARN-05/WARN-03 exist to prevent. The setup cell
cannot be re-run either, because `apps.ready` now makes it refuse with "restart the kernel", so the poisoned
state lasts until the kernel restarts.

Reproduced in a subprocess with an override settings module whose `DATABASES` pointed at a throwaway file:

```
raised: A notebook must never write to the developer database -- resolved DB '...other.sqlite3' ...
apps.ready after refusal: True
next cell would use: .../other.sqlite3
write succeeded            # CREATE TABLE ran against the refused database
```

Running the check after `django.setup()` also lets every installed app's `ready()` run against a database
nobody has verified yet. `test_a_databases_override_is_refused_before_migrate` only proves that no current
`ready()` hook opens the database. A future hook that does would write before the guard fires.
`test_notebook_helper.py` never checks the state left behind after a refusal.

**Fix:** settings can be read without populating the app registry, so resolve and check them **before**
`django.setup()`. Also make any failure after setup leave the ORM unusable:

```python
        os.environ['FOMO_DATABASE_PATH'] = str(scratch_db_path)
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'src.fomo.settings')
        os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'true')

        from django.conf import settings as django_settings

        # Loads the settings module (and local_settings) without populating apps or opening a connection.
        resolved_name = django_settings.DATABASES['default']['NAME']
        check_resolved_database(resolved_name, scratch_db_path)   # refuses while apps.ready is still False

        import django

        django.setup()
        from django.core.management import call_command

        call_command('migrate', verbosity=0)
    except BaseException:
        from django.apps import apps

        if apps.ready:  # failure after setup (e.g. in migrate): make sure no later cell can reach a database
            from django.db import connections

            connections.close_all()
            for alias in connections:
                connections[alias].settings_dict['NAME'] = str(scratch_db_path)  # deleted below -> open fails
        shutil.rmtree(scratch_db_path.parent, ignore_errors=True)
        raise
```

Then add a `TestHelperRunsForReal` case that catches the refusal in the child process and asserts that
`django.apps.apps.ready` is `False` and that a raw `connection.cursor()` raises.

## Warnings

### WR-01: The static guard does not catch a notebook that re-routes the database after setup

**File:** `solsys_code/tests/test_pre_executed_notebooks.py:281-295` (R5/R7), and `isolation_problems()` generally
**Issue:** R5/R7 only look for the string constants `FOMO_DATABASE_PATH` and `fomo_db.sqlite3`. A notebook
that has called the helper properly can still point Django at the developer database afterwards, and the
guard returns no problems. I confirmed that this synthetic cell yields `isolation_problems(...) == []`:

```python
from django.db import connections
connections['default'].close()
connections['default'].settings_dict['NAME'] = str(repo_root_path / 'src' / ('fomo_db' + '.sqlite3'))
from django.conf import settings
settings.DATABASES['default']['NAME'] = connections['default'].settings_dict['NAME']
```

`override_settings(DATABASES=...)` and `sqlite3.connect(...)` aimed at a computed path get through the same
way. The module docstring says the only blind spot is code the AST cannot read (`exec()`). This gap is
plain, readable Python, so that statement is too strong.
**Fix:** add an R-rule over every parsed cell that reports any attribute or subscript reference to
`DATABASES` or `settings_dict`, any `override_settings`/`modify_settings` name, and any `sqlite3` import or
`sqlite3.connect` call. None of the eight notebooks uses any of these today. Alternatively, narrow the
docstring's claim to the specific bypasses that are checked.

### WR-02: The resolved-path check covers only the `default` alias, but the teardown says the developer database was never written

**File:** `docs/notebooks/fomo_nb_utils.py:206-207, 238-240`
**Issue:** `check_resolved_database()` checks only `DATABASES['default']['NAME']`. A `local_settings.py`
that adds a second alias plus a `DATABASE_ROUTERS` entry, which is a normal production pattern, can still
send reads and writes to a real database, and the check passes. The teardown still prints "The developer
database (src/fomo_db.sqlite3) was never opened for writing." That line is printed unconditionally and is
not measured. R10 in the guard treats it as required evidence, so the published notebooks carry a promise
the helper never verifies.
**Fix:** check every alias, and refuse routers:

```python
for alias, cfg in django_settings.DATABASES.items():
    if alias != 'default':
        raise RuntimeError(f'Notebook scratch setup supports only the default database; found alias {alias!r}.')
check_resolved_database(django_settings.DATABASES['default']['NAME'], scratch_db_path)
if django_settings.DATABASE_ROUTERS:
    raise RuntimeError('DATABASE_ROUTERS is set; a notebook cannot prove all writes reach the scratch database.')
```

Alternatively, reword the teardown line so it states what was checked ("Django's only database was the
scratch file above").

### WR-03: The functional marker test's check that the unmarked row has no bar can pass without testing anything

**File:** `solsys_code/tests/test_bootstrap5_rendering.py:545, 564-566`
**Issue:** the fixture assertion `bands.get(other_run.pk, 'none') != BAND_HIGH` also passes when `other_run`
is **not a candidate at all**. In that case no non-High candidate row is rendered, and the locator
`table.table tbody tr:not(.bg-light):not(.border-start)` with `.first` falls through to the first row of
the Dismissed table (django-tables2 also gives it `class="table"`), i.e. its "No dismissals recorded yet."
row. The "no 4px left bar" assertion then passes about a row that is not a candidate row. The unit-level
`AttributionHighBandMarkerRenderTests` covers the server side, but this browser half of D-21 can pass
without testing anything.
**Fix:** require the second candidate (`self.assertIn(other_run.pk, bands)`). Then scope the locator to the
worklist, for example locate the `<tbody>` containing the event's group header and pick its
`tr:not(.bg-light):not(.border-start)`, or add a `data-candidate-run="{{ candidate.run.pk }}"` attribute
and select `tr[data-candidate-run="<other_run.pk>"]`.

## Info

### IN-01: `FOMO_DATABASE_PATH` (and `DJANGO_ALLOW_ASYNC_UNSAFE`) stay set after teardown or a failed setup

**File:** `docs/notebooks/fomo_nb_utils.py:193-197, 210-212, 238`
**Issue:** neither the failure path nor `teardown_scratch_django()` restores the environment. Afterwards
the kernel's `os.environ['FOMO_DATABASE_PATH']` names a deleted directory, and any later `!python manage.py ...`
or `subprocess` call from that kernel inherits it. Today that fails safe ("unable to open database file"),
but it is surprising state.
**Fix:** record the previous values in `setup_scratch_django()` (return them or keep them in a module-level
dict) and restore or pop them in the `except` path and in teardown.

### IN-02: Committed notebook output publishes the developer's absolute home path

**File:** `docs/notebooks/fomo_nb_utils.py:213`; for example `docs/notebooks/pre_executed/*.ipynb` setup-cell output
**Issue:** the new evidence line prints `repo_root='/home/tlister/git/fomo_devel'`, and every notebook,
including the docs-published ones, now carries the developer's username and directory layout. The path adds
no evidence (the guard checks only the `Resolved database:` line).
**Fix:** print `repo_root` relative to the notebook (or only its name), or drop it from the line.

### IN-03: Attribution worklists have no pager, and their page keys are outside the return state

**File:** `src/templates/campaigns/attribution_queue.html:48-123, 153-212`; `solsys_code/campaign_views.py:1209-1236`
**Issue:** this predates the phase but bears on WARN-07. `event_groups`/`record_groups` are paginated on
`event_page`/`record_page` (25 groups each), but the template renders no pager, so orphans beyond the first
25 can be reached only by editing the URL. Neither key is in `attribution_return_state()`, so Confirm or
Dismiss on such a page lands back on page 1. The runbook's new "The page keeps your place ... each table's
page and sort" paragraph (`docs/runbooks/telescope_runs_calendar.rst:967-978`) reads as if it covered the
worklists.
**Fix:** add a pager for the two worklists and whitelist `event_page`/`record_page` with the same
`_RETURN_PAGE_RE`. Alternatively, say in the runbook that only the Dismissed and Confirmed tables' place is
kept.

### IN-04: Undo posts a page number computed under `per_page`, but the redirect drops `per_page`

**File:** `solsys_code/campaign_tables.py:582-596`; `solsys_code/campaign_views.py:1209-1236`
**Issue:** `RequestConfig` honours `dismissed-per_page`/`confirmed-per_page` from the URL, and
`_return_state_inputs()` posts `table.page.number` under that page size. The whitelist then drops
`per_page`, so the redirect reopens page N at the default 25 rows, which can be a different page. The page
offers no per-page control today, so this happens only with a hand-edited URL.
**Fix:** either whitelist `*-per_page` with a bounded regex, or omit the own-page override when a non-default
`per_page` is active.

### IN-05: `load_telescope_runs_demo` still leaves eight temp files in `/tmp` per regeneration

**File:** `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` (eight `tempfile.NamedTemporaryFile(..., delete=False)` cells)
**Issue:** this predates the phase, but the notebook was re-executed here, and its sibling
`import_campaign_csv_demo` was changed to write its CSV inside `scratch_db_dir` so that teardown removes it.
`load_telescope_runs_demo` never unlinks its schedule files, so each regeneration leaves eight
`/tmp/tmp*.txt` files behind.
**Fix:** create them with `dir=scratch_db_dir` (as `import_campaign_csv_demo` now does) so the existing
teardown removes them.

### IN-06: `teardown_scratch_django()` will delete any directory named `fomo-notebook-db-*`, wherever it is

**File:** `docs/notebooks/fomo_nb_utils.py:229-238`
**Issue:** the only safety check is on the basename. A wrong argument such as
`~/projects/fomo-notebook-db-archive` would be `rmtree`d.
**Fix:** also require `path.resolve().parent == Path(tempfile.gettempdir()).resolve()` (or check against the
directory `setup_scratch_django()` actually created, recorded at module level).

### IN-07: Small test and code tidy-ups

**File:** `solsys_code/tests/test_notebook_helper.py:241`; `solsys_code/campaign_views.py:1188-1189`
**Issue:**
- `test_fresh_setup_routes_migrates_and_tears_down` puts the temp directory into a regex without escaping
  it (`rf"Resolved database: '{tmp}/..."`). A `TMPDIR` containing `+`, `(` or `.` would make the test
  flaky.
- `_DISMISSED_TABLE_PREFIX = DISMISSED_TABLE_PREFIX` and `_CONFIRMED_TABLE_PREFIX = CONFIRMED_TABLE_PREFIX`
  are pure aliases of the imported constants, and only add a name to keep in sync.
- `prepare_scratch_database()`'s snapshot-failure cleanup branch (`fomo_nb_utils.py:153-157`) and snapshot
  mode through `setup_scratch_django()` have no executed test. D-11 lists the snapshot as a "run the real
  function" case, and only `snapshot_database()` is exercised.
**Fix:** use `re.escape(tmp)`; use the imported constants directly; add a subprocess test that runs
`setup_scratch_django(mode='snapshot')` against a temp repo root containing a small `src/fomo_db.sqlite3`,
and a unit test that forces `snapshot_database` to raise and asserts the scratch directory is gone.

---

_Reviewed: 2026-10-10T00:35:32Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
