# Phase 40: Notebook Isolation & Attribution Page - Pattern Map

**Mapped:** 2026-10-09
**Files analyzed:** 22 new/modified (excluding the eight notebooks, grouped as one row)
**Analogs found:** 21 / 22 (one partial: subprocess-based test)

All analog paths below are git-tracked source (checked: `docs/notebooks/fomo_nb_utils.py` is NOT tracked on this branch, so it is a new file seeded from `origin/feature/skymap`). Notebooks were read as JSON, never executed.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `docs/notebooks/fomo_nb_utils.py` (new) | utility (dev tooling helper) | file-I/O + batch (scratch DB setup/teardown) | setup cell `7022f987` of `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` + `git show origin/feature/skymap:docs/notebooks/fomo_nb_utils.py` (seed) | exact (logic) / seed file |
| `docs/notebooks/pre_executed/*.ipynb` x8 (setup cell + last cell) | notebook | batch | `project_observation_calendar_demo.ipynb` setup cell | exact |
| `solsys_code/tests/test_pre_executed_notebooks.py` (rewrite) | test (guard) | transform (AST over JSON) | itself (current `isolation_problems()`, `_notebook()` builder) | exact |
| `solsys_code/tests/test_notebook_helper.py` (new) | test | file-I/O + subprocess | `test_projector_demo_notebook.py` (SimpleTestCase, file-reading) + `test_check_unattended.py` (`subprocess` usage, mocked only) | partial (no real subprocess test exists) |
| `solsys_code/tests/test_projector_demo_notebook.py` (keep green) | test | file reads | itself | exact |
| `solsys_code/tests/test_bootstrap5_rendering.py` (extend scans + add functional marker test) | test | transform + browser | `TestTemplatesUseBootstrap5DataAttributes` (line 459) and `test_attribution_page_sections_open_and_close_with_bootstrap5` (line 385) | exact |
| `solsys_code/tests/test_attribution_template.py` (add GET-only marker test) | test | request-response (GET) | same module (`AttributionViewTestBase` subclass, line 800-830 area) | exact |
| `solsys_code/tests/test_campaign_attribution_views.py` (add return-state tests) | test | request-response (POST+follow) | `TestDismissAndUndoDismissal` (line 421) | exact |
| `src/templates/campaigns/attribution_queue.html` | template | request-response | itself | exact |
| `src/templates/campaigns/{campaign_list,campaignrun_table,approval_queue,campaignrun_gap_analysis,campaignrun_submit_form,submission_thanks}.html`, `src/templates/solsys_code/partials/campaign_links.html` | template | request-response | mechanical rename; Phase 39 `calendar.html` precedent | exact |
| `solsys_code/campaign_tables.py` (`ml-1` at 433; `render_actions` at 629, 731 get hidden state) | component (django-tables2) | request-response | `AttributionDismissedTable.render_actions` (629) / `AttributionConfirmedTable.render_actions` (731) | exact |
| `solsys_code/campaign_views.py` (`_redirect` ~1386, new `attribution_return_state`, context boolean) | controller | request-response | `AttributionDecisionView._redirect` (1386) | exact |
| `docs/notebooks/README.md`, `docs/runbooks/telescope_runs_calendar.rst` (~956-990), `CLAUDE.md` paired-docs map | docs | n/a | existing sections | exact |
| `.planning/ROADMAP.md`, `REQUIREMENTS.md`, 37.1 REVIEW-DISPOSITION.md | ledger | n/a | existing rows | exact |

## Pattern Assignments

### `docs/notebooks/fomo_nb_utils.py` (utility, file-I/O)

**Seed:** `git show origin/feature/skymap:docs/notebooks/fomo_nb_utils.py` (not tracked here; write it byte-identical for `special_locations()`, including U+2212 minus signs, then append new functions BELOW it; ruff v0.16.9 passes it, no RUF family selected).

Seed header (imports at top are astropy; D-07 forbids Django at module load, so new functions import Django lazily):
```python
import astropy.units as u
from astropy.coordinates import SkyCoord


def special_locations(skycoords=False):
    """Special locations for Rubin. ...
```
Add stdlib imports (`os`, `sqlite3`, `sys`/`tempfile`, `shutil`, `pathlib.Path`) alongside; keep the seed function untouched.

**Snapshot pattern to lift** (`project_observation_calendar_demo.ipynb` cell `7022f987`):
```python
dev_db_path = repo_root_path / 'src' / 'fomo_db.sqlite3'
if not dev_db_path.exists():
    raise RuntimeError(f'No developer database at {dev_db_path}; run `python manage.py migrate` first.')

scratch_db_dir = Path(tempfile.mkdtemp(prefix='fomo-notebook-db-'))
scratch_db_path = scratch_db_dir / 'fomo_db.sqlite3'
source_connection = sqlite3.connect(f'{dev_db_path.as_uri()}?mode=ro', uri=True)
scratch_connection = sqlite3.connect(scratch_db_path)
try:
    source_connection.backup(scratch_connection)
finally:
    scratch_connection.close()
    source_connection.close()
os.environ['FOMO_DATABASE_PATH'] = str(scratch_db_path)   # plain assignment, never setdefault
```
Order fix required by RESEARCH: the missing-developer-database `RuntimeError` must come BEFORE `mkdtemp`/env assignment.

**Env + setup + check + migrate** (same cell):
```python
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'src.fomo.settings')
os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'true')
django.setup()                                   # lazy `import django` inside the function
from django.conf import settings as django_settings
from django.core.management import call_command as _call_command

resolved_db_name = django_settings.DATABASES['default']['NAME']
if Path(resolved_db_name).resolve() != scratch_db_path.resolve():
    raise RuntimeError(
        'This notebook must never write to the developer database -- resolved DB '
        f'{resolved_db_name!r} is not the scratch copy {str(scratch_db_path)!r}.'
    )
_call_command('migrate', verbosity=0)
print(f'Django ready: settings module={os.environ["DJANGO_SETTINGS_MODULE"]!r}, repo_root={repo_root!r}')
print(f'Resolved database: {resolved_db_name!r} (scratch copy of the developer database, '
      "taken read-only with SQLite's online backup and migrated to head)")
```
Output contract other tests depend on: line `Resolved database: '/tmp/fomo-notebook-db-XXXX/fomo_db.sqlite3' (...)`; snapshot mode text must contain `scratch copy` (`test_projector_demo_notebook.py:42` regex `Resolved database: '/tmp/fomo-notebook-db-[^/']+/fomo_db\.sqlite3'`); teardown prints `Removed scratch database directory: <dir>`. Teardown: lazy `connections.close_all()` tolerating not-set-up Django, then `shutil.rmtree(dir, ignore_errors=True)`.

Style: Google-style docstrings with `Args/Returns/Raises` (D103 selected), no `assert` anywhere in the module, single quotes, 120 cols.

---

### Pre-executed notebooks x8 (setup cell, last cell)

**Analog:** `project_observation_calendar_demo.ipynb` setup cell bootstrap (keep in each notebook; extend with `docs/notebooks` on `sys.path`):
```python
repo_root_path = Path.cwd().resolve().parents[2]
if not (repo_root_path / 'manage.py').exists():
    raise RuntimeError(f'No manage.py at {repo_root_path}; run Jupyter from docs/notebooks/pre_executed/')
repo_root = str(repo_root_path)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
```
Then (RESEARCH Pattern 2): add `repo_root_path / 'docs' / 'notebooks'` to `sys.path`, `from fomo_nb_utils import setup_scratch_django  # noqa: E402`, `scratch_db_dir = setup_scratch_django(repo_root_path, mode='fresh'|'snapshot')`. No `import django` before it.

Verdicts: fresh = campaign_lifecycle, load_telescope_runs, backfill_lco, import_campaign_csv, sync_gemini, telescope_runs; snapshot = reconcile_campaign_runs, project_observation_calendar. Keep project notebook cell id `7022f987`; keep `sync_gemini` extra `src` sys.path entry and `campaign_lifecycle`'s `setup_test_environment()` (`except RuntimeError`) in a later cell. Edit via `nbformat` preserving `id`/metadata (`nbsphinx.execute = never`); regenerate with `jupyter nbconvert --to notebook --execute --inplace` from `docs/notebooks/pre_executed/`. Last code cell calls the helper teardown. Markdown prose cell ids to rewrite are listed in RESEARCH Pitfall 10.

---

### `solsys_code/tests/test_pre_executed_notebooks.py` (test, AST transform)

**Analog:** itself. Keep these helpers verbatim (lines ~34-100): `_cell_source`, `_cell_output_text`, `_parse`, `_dotted_name`, `_is_constant`, `_calls`, `_code_cells`; keep `_notebook()` synthetic builder (lines ~198-216) but change `REFERENCE_SETUP` / `REFERENCE_TEARDOWN` to the helper form.

Header/imports to keep:
```python
import ast
import json
import re
from pathlib import Path

from django.test import SimpleTestCase

NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / 'docs' / 'notebooks' / 'pre_executed'
SCRATCH_PREFIX = 'fomo-notebook-db-'
RESOLVED_LINE = re.compile(r"Resolved database: '([^']+)'")
TEARDOWN_OUTPUT = 'Removed scratch database directory:'
```
Replace `isolation_problems()` body with the D-11..D-14 rules: iterate `tree.body` (not `ast.walk`) for ordering; Django-backed by `ast.Import/ImportFrom` top package in `django`/`solsys_code`/`tom_*`; execution counts `== list(range(1, N+1))`; read JSON with `encoding='utf-8'`. Keep the SubTest loop style:
```python
for path in notebooks:
    with self.subTest(notebook=path.name):
        problems = isolation_problems(json.loads(path.read_text(encoding='utf-8')))
        self.assertEqual(problems, [], f'{path.name} is not isolated:\n  ' + '\n  '.join(problems))
```
Not-vacuous test: assert the set of names equals the eight expected files and each is Django-backed. Add negative tests per WR-13 bypass (own `FOMO_DATABASE_PATH` assignment with stale output, aliased `from django import setup as _s`, call inside `def`/`if False:`, missing teardown, non-contiguous counts, `shutil.copy2`). Optionally add: no committed output contains `arcmin deg` (WARN-06).

---

### `solsys_code/tests/test_notebook_helper.py` (test, file-I/O + subprocess)

**Analog:** `test_projector_demo_notebook.py` for module docstring/`SimpleTestCase`/`Path(__file__).resolve().parents[2]` style:
```python
_REPO_ROOT = Path(__file__).resolve().parents[2]
...
from django.test import SimpleTestCase
```
No existing test runs a real subprocess (`test_check_unattended.py` only patches `subprocess.run`, with `subprocess.CompletedProcess(...)` fakes at lines 196-247), so use RESEARCH Pattern 4 directly: load helper via `importlib.util.spec_from_file_location('fomo_nb_utils', HELPER_PATH)` (safe because Django is lazy), in-process tests of snapshot against a tiny SQLite file and the missing-source `RuntimeError`; `subprocess.run([sys.executable, '-c', ...], env=..., cwd=repo_root, timeout=...)` for fresh setup (about 4 s), and the `local_settings`-style `DATABASES` override expecting `RuntimeError` in stderr and the other DB never created, run with `sys.executable -O` and `PYTHONOPTIMIZE=1`. Add a static test: helper module contains no `ast.Assert`. Do not call `django.setup()` in the test process; do not write `src/fomo/local_settings.py`.

---

### `solsys_code/tests/test_bootstrap5_rendering.py` (test)

**Analog 1:** `TestTemplatesUseBootstrap5DataAttributes` (lines 459-505). Widen `_template_files()` (lines 470-475) to every `TEMPLATES[0]['DIRS']` entry plus app template dirs inside the repo; fix the `relative_to(DIRS[0])` at line ~482 (IN-12) by computing relative against whichever collected directory contains the file:
```python
@staticmethod
def _template_files():
    files = []
    for template_dir in settings.TEMPLATES[0]['DIRS']:
        files.extend(sorted(Path(template_dir).rglob('*.html')))
    return files
```
Hit-reporting loop to copy (lines 478-486):
```python
for path in files:
    for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if self.BOOTSTRAP4_ATTRIBUTE.search(line):
            relative = path.relative_to(Path(settings.TEMPLATES[0]['DIRS'][0]))
            hits.append(f'{relative}:{lineno}: {line.strip()}')
self.assertEqual(hits, [], 'Bootstrap 4 plugin data attributes found:\n' + '\n'.join(hits))
```
New class guard sits beside it as a sibling `SimpleTestCase` with RESEARCH's `BOOTSTRAP4_CLASS` regex (lookarounds, not `\b`), plus the same three-test shape: scan, `test_the_scan_is_not_vacuous` (`>= 20` templates, names `campaigns/attribution_queue.html`, `campaigns/campaign_list.html`), and a pattern-matches test with good/bad strings (lines 496-505 style). Third-party app dirs are out of scope (D-20 amended); filter with `Path(d).resolve().is_relative_to(REPO_ROOT)`. Write forbidden names split (e.g. `'border-' + 'left'`) so the test file does not itself contain them.

**Analog 2 (functional test):** `test_attribution_page_sections_open_and_close_with_bootstrap5` (line 385, inside `@tag('functional') class TestBootstrap5Rendering`). Fixture, login and navigation to copy:
```python
staff = get_user_model().objects.create_user(username='bs5-attribution-staff', password='pw', is_staff=True)
campaign = TargetList.objects.create(name='BS5 Attribution Campaign')
target = NonSiderealTargetFactory.create()
campaign.targets.add(target)
observatory = Observatory.objects.create(obscode='E10', name='Siding Spring', short_name='SSO')
run = CampaignRun.objects.create(campaign=campaign, telescope_instrument='FTS/MuSCAT4',
    window_start=date(2026, 7, 7), window_end=date(2026, 7, 21), site=observatory, telescope_class='')
...
self._log_in_browser(staff)          # line 130: force_login + add_cookies(session cookie)
page_errors = []
self.page.on('pageerror', lambda exc: page_errors.append(str(exc)))
self.page.goto(f'{self.live_server_url}{reverse("campaigns:attribution")}')
...
assert page_errors == []
```
The new test needs an orphan event/record that yields a High-band candidate (see `_make_event`/`_make_record` in `AttributionViewTestBase`), then `page.evaluate` on `getComputedStyle(row)`: left `4px`/`rgb(25, 135, 84)`, top/right/bottom `0px`. Optional Undo test: `self.page.on('dialog', lambda d: d.accept())`.

---

### `solsys_code/tests/test_attribution_template.py` (test, GET render)

**Analog:** same module; it subclasses `AttributionViewTestBase` and imports:
```python
from solsys_code.tests.test_campaign_attribution_views import AttributionViewTestBase
TEMPLATE_NAME = 'campaigns/attribution_queue.html'
```
Module rule (docstring lines 1-10): GET/structure only, never a client POST. Add the D-21 `TestCase` half here: render the page with a High-band and a non-High candidate and assert `border-start border-4 border-success` appears for High rows only. Strengthen `test_evidence_columns_score_and_band_badge_all_present` (~line 815), which currently asserts only `'border-success'`.

Template change (`attribution_queue.html:62-64` and ~169), today:
```html
<tr class="{% if candidate.band == 'high' %}border-left border-success{% endif %}{% if not forloop.first %} border-top{% endif %}"
    {% if candidate.band == 'high' %}style="border-left-width: 4px;"{% endif %}>
```
Target (RESEARCH, verified in Chromium):
```html
<tr class="{% if candidate.band == 'high' %}border-0 border-start border-4 border-success{% elif not forloop.first %}border-top{% endif %}">
```
Dismissed section markup to dedupe into one context boolean (lines 244-262; condition repeated 3x, IN-15):
```html
<button class="btn btn-link p-0 font-weight-bold{% if dismissed_table.prefixed_page_field in request.GET or dismissed_table.prefixed_order_by_field in request.GET %}{% else %} collapsed{% endif %}" ...
<div id="attribution-dismissed-section" class="collapse{% if ... %} show{% endif %}">
```

---

### `solsys_code/tests/test_campaign_attribution_views.py` (test, POST + follow)

**Analog:** `TestDismissAndUndoDismissal` (line 421). Pattern to copy:
```python
class TestDismissAndUndoDismissal(AttributionViewTestBase):
    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def test_dismiss_with_reason_creates_row_with_who_when_reason(self):
        event = self._make_event()
        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'dismiss', 'kind': 'event', 'orphan_pk': event.pk,
             'run_pk': self.campaign_run.pk, 'reason': 'Different campaign entirely.'},
        )
        self.assertEqual(response.status_code, 302)
```
New tests (D-23, D-26, D-27, D-28): same POST dict plus `band`, `dismissed-page`, `confirmed-page`, sort keys; use `follow=True` and assert on rendered `#attribution-dismissed-section` having `show`; `response.redirect_chain[-1][0]` for the fragment (client drops fragments). Create 26 dismissals to reach page 2 for the clamp test. Tampering cases (`band=evil`, `dismissed-page=0`/`abc`/`-1`, sort with spaces) are dropped. Keep `TestUndoDismissalPrompt` (line 1377) green: it calls `AttributionDismissedTable([], request=None).render_actions(dismissal)`, so `render_actions` must tolerate `request=None` and an unpaginated table. Fixtures: `NonSiderealTargetFactory` only (already in base), per CLAUDE.md.

---

### `solsys_code/campaign_views.py` (controller, request-response)

**Analog:** `AttributionDecisionView._redirect` (lines 1386-1393), the single sink for all five actions:
```python
def _redirect(self, request):
    band = request.POST.get('band')
    if band in _ATTRIBUTION_BANDS:
        return redirect(f'{reverse("campaigns:attribution")}?band={band}')
    return redirect('campaigns:attribution')
```
Replace with a call to a new module-level `attribution_return_state(querydict) -> dict[str, str]` (RESEARCH sketch: `band` in `_ATTRIBUTION_BANDS`; `*-page` positive decimal; `*-sort` `re.fullmatch(r'-?[a-z_]+', ...)`), `urllib.parse.urlencode`, and a constant fragment (`#attribution-dismissed-section` / `#attribution-confirmed-section`). Action allow-list and dispatch to keep: `_ATTRIBUTION_ACTIONS` tuple at line 1301, `post()` at ~1357. Table construction with `prefix='dismissed-'`/`'confirmed-'` and `RequestConfig` is at ~1277-1284 (use `table.prefixed_page_field` / `prefixed_order_by_field`, not literals). django-tables2 already clamps out-of-range pages; only validate the number.

---

### `solsys_code/campaign_tables.py` (component)

**Analog:** `AttributionDismissedTable.render_actions` (line 629) and `AttributionConfirmedTable.render_actions` (line 731). Existing form builder:
```python
decide_url = reverse('campaigns:attribution_decide')
csrf_token = get_token(self.request) if self.request is not None else ''
return format_html(
    '<form method="post" action="{0}">'
    '<input type="hidden" name="csrfmiddlewaretoken" value="{1}">'
    '<input type="hidden" name="action" value="undo_confirmation">'
    '<input type="hidden" name="kind" value="{2}">'
    '<input type="hidden" name="orphan_pk" value="{3}">'
    '<input type="hidden" name="run_pk" value="{4}">'
    '<button type="submit" class="btn btn-sm btn-outline-secondary" onclick="return confirm(...)">Undo</button>'
    '</form>',
    decide_url, csrf_token, kind, orphan.pk, record.run.pk,
)
```
Add hidden inputs for the validated return state via `format_html`/`format_html_join` (autoescaped); Dismissed table always sends `dismissed-page` = current page number (default 1) per D-24. Fallback to page 1 when `self.request is None` or no `page` attribute. Also change `class="small ml-1"` at line 433 to `ms-1`. Leave `ApprovalQueueTable.render_actions` (line 492) alone.

## Shared Patterns

### Raise, never assert, for the DB guard
**Source:** `project_observation_calendar_demo.ipynb` cell `7022f987` (resolved-path check, quoted above)
**Apply to:** `fomo_nb_utils.py` only (one copy for all eight notebooks); remaining ~236 demonstration asserts untouched (D-15).

### Test conventions
**Source:** `solsys_code/tests/test_pre_executed_notebooks.py`, `test_attribution_template.py`
**Apply to:** all new tests. Django test runner only; `SimpleTestCase` for file-reading guards, `TestCase` for views, `@tag('functional')` + `StaticLiveServerTestCase` for Playwright; `NonSiderealTargetFactory` for Target fixtures; module docstring stating why the module exists; tag any migration test `@tag('migration')` (none expected).

### Form hidden-input and CSRF discipline
**Source:** `campaign_tables.py` 629/731 (`csrfmiddlewaretoken` hidden input, `format_html` escaping)
**Apply to:** every form carrying return state, including the template's confirm/dismiss forms (events ~107, records ~197) and bulk forms (`bulk-confirm-events` ~134, `bulk-confirm-records` ~220), which already carry one `band` hidden field.

### Bootstrap 5.3 renames
**Source:** RESEARCH "Bootstrap 5.3 sweep" inventory table (file:line to replacement)
**Apply to:** the eight templates + `campaign_tables.py:433`. `font-weight-bold`->`fw-bold`, `mr-N`->`me-N`, `form-inline`->`d-flex align-items-center gap-2`, `form-control`(select)->`form-select`, `form-group`->`mb-3`, `badge-info/light`->`text-bg-info/light`. `solsys_code/filters.py:63` (`form-group col-md-3`, Python-built) is seen and deliberately left.

### Paired docs / ledgers
**Apply to:** `docs/notebooks/README.md` (D-04 rule in the same words as ROADMAP SC 1 / REQUIREMENTS WARN-05; list snapshot vs fresh notebooks; helper usage; regeneration with "Restart & Run All"; delete leftover `fomo-notebook-db-*` dirs; snapshot notebooks need a populated `src/fomo_db.sqlite3`), `docs/runbooks/telescope_runs_calendar.rst` ~956-990, `CLAUDE.md` paired-docs map line for `docs/notebooks/fomo_nb_utils.py`, and the 37.1 disposition ledger rows WR-13..WR-18 and IN-12. `campaign_lifecycle_demo.ipynb` is re-executed last, after the attribution changes land.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| real-subprocess portion of `test_notebook_helper.py` | test | subprocess | Only mocked `subprocess.run` exists (`test_check_unattended.py`); use RESEARCH Pattern 4 |

## Metadata

**Analog search scope:** `docs/notebooks/`, `solsys_code/tests/`, `solsys_code/campaign_views.py`, `solsys_code/campaign_tables.py`, `src/templates/campaigns/`, `origin/feature/skymap` helper
**Files scanned:** about 12 read directly (plus RESEARCH/CONTEXT)
**Pattern extraction date:** 2026-10-09
