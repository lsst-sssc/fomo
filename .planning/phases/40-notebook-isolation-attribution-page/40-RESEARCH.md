# Phase 40: Notebook Isolation & Attribution Page - Research

**Researched:** 2026-10-09
**Domain:** Jupyter pre-executed demo notebooks (Django + SQLite scratch databases), an AST-based notebook guard, Bootstrap 5.3 class sweep, a Django staff worklist's post-then-redirect state
**Confidence:** HIGH (almost every claim was checked by reading the file, running the code, or rendering in Chromium this session; the few `[ASSUMED]` items are listed in the Assumptions Log)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Mixed per notebook. A notebook migrates a fresh, empty scratch database and creates every row it shows, unless its committed output depends on rows it did not create itself, then it takes a read-only snapshot. The researcher applies that test to each of the three byte-copy notebooks (`campaign_lifecycle_demo`, `load_telescope_runs_demo`, `reconcile_campaign_runs_demo`) and records the verdict per notebook. Expected (not locked): `reconcile_campaign_runs_demo` snapshots (its sweep shows 45 real runs / 264 real events); the other two likely go fresh because they `update_or_create` their own Observatory/campaign rows. Verify, don't assume.
- **D-02:** Snapshots use SQLite's online backup only: a `mode=ro` URI connection plus `Connection.backup()`, the pattern `project_observation_calendar_demo` already uses. No `shutil.copy*` of any database anywhere.
- **D-03:** `project_observation_calendar_demo` keeps its read-only snapshot (the reference pattern). It moves onto the shared helper like every other notebook.
- **D-04:** Success criterion 1 is reworded, in `ROADMAP.md` (Phase 40 SC 1) and `REQUIREMENTS.md` (WARN-05), from "no pre-executed notebook reads or copies `src/fomo_db.sqlite3`" to the rule actually adopted: no notebook byte-copies or writes the developer database; a notebook that needs real rows reads it only through a read-only online-backup snapshot; every other notebook starts from a fresh migrated database. `docs/notebooks/README.md` states the same rule in the same words, and lists which notebooks snapshot and which start fresh.
- **D-05:** A snapshot notebook on a machine with no `src/fomo_db.sqlite3` raises a clear `RuntimeError` (today's behavior). No silent fallback to a fresh database.
- **D-06:** One shared helper module, not a per-notebook copy of the setup code. It lives at `docs/notebooks/fomo_nb_utils.py` (the name and location the developer already used on the unmerged `feature/skymap` branch). Start from `feature/skymap`'s version of the file (`git show origin/feature/skymap:docs/notebooks/fomo_nb_utils.py`, which carries `special_locations()`), and add the scratch-database functions to it, so the later skymap merge has nothing to reconcile in this file. Reversibility: costly (every pre-executed notebook and the guard depend on the helper's name, location and call signature).
- **D-07:** Django imports inside the helper are lazy (inside the function bodies), so a docs-build notebook importing only `special_locations()` never pulls Django in. The helper must not import FOMO models at module load; it runs before `django.setup()`.
- **D-08:** Import path. Pre-executed notebooks start their kernel in `docs/notebooks/pre_executed/`, so their setup cell adds `docs/notebooks` to `sys.path` alongside the repo root it already adds, then imports the helper by bare name (`from fomo_nb_utils import ...`). The repo-root/`sys.path` bootstrap stays in each notebook; everything after it (scratch dir via `tempfile.mkdtemp(prefix='fomo-notebook-db-')`, plain assignment of `FOMO_DATABASE_PATH` (never `setdefault`), fresh `migrate` or `mode=ro` backup snapshot, `django.setup()`, the resolved-path check, printing `Resolved database: ...`) moves into the helper.
- **D-09:** The resolved-path check is a `raise RuntimeError`, never an `assert` (WARN-03), using `Path(...).resolve()` comparison as `project_observation_calendar_demo` does today. Living in the helper, it is written once for all eight notebooks.
- **D-10:** The helper provides the teardown too (removing the scratch directory); every Django-backed notebook's last code cell calls it.
- **D-11:** The guard checks notebooks' use of the helper; tests execute the helper for real. Guard rules, per notebook (top-level statements only, iterate `tree.body`, not `ast.walk`, so a call inside an unexecuted `def`/`if False:` does not count): the helper's setup call appears before any `django` / `solsys_code` / `tom_*` import or use; the notebook itself never calls `django.setup()` and never assigns `FOMO_DATABASE_PATH`; no `shutil.copy`, `shutil.copy2` or `shutil.copyfile` anywhere in the notebook. Helper tests (Django test runner, in `solsys_code/tests/`) run the real functions: fresh-database routing, the `mode=ro` backup snapshot, the missing-developer-database `RuntimeError`, and a `RuntimeError` when a `local_settings`-style `DATABASES` override wins over the scratch path. These executed tests replace today's source-pattern checks of the setup cell (mkdtemp prefix, assignment form, resolved-name read) rather than duplicating them.
- **D-12:** Require one fresh top-to-bottom run: code-cell `execution_count`s must be exactly `1..N`. This closes WR-13's stale-output hole; `nbconvert --execute --inplace` always produces it.
- **D-13:** Django-backed detection is by import: any notebook that imports `django`, `solsys_code` or any `tom_*` package is Django-backed and must satisfy the guard. Renaming or aliasing `django.setup` can no longer hide a notebook. The not-vacuous test covers all eight notebooks, not just four.
- **D-14:** Keep the committed-output checks, via the helper: the setup cell's committed output must contain the helper's `Resolved database: ...fomo-notebook-db-...` line, and the last code cell must call the helper's teardown with its committed output present. With D-12 that output is known to belong to the current source.
- **D-15:** The ~250 demonstration `assert`s in the eight notebooks stay as they are. WARN-03 is about the database-path guard, which D-09 moves into the helper. The audit (all eight notebooks, per the ROADMAP scope note) records that each remaining `assert` was reviewed and protects no write.
- **D-16:** WARN-06: the horizon dip prints in degrees (`dip.to(u.deg)`), matching the Stage-1 constraint (1.44 deg +/- 0.02 at 2402 m), the cell comment and the notebook intro.
- **D-17:** Tidy every FOMO template, not only the attribution page: all eight templates the scan found (`src/templates/campaigns/attribution_queue.html`, `campaign_list.html`, `campaignrun_table.html`, `approval_queue.html`, `campaignrun_gap_analysis.html`, `campaignrun_submit_form.html`, `submission_thanks.html`, and `src/templates/solsys_code/partials/campaign_links.html`), plus the `ml-1` in the HTML built in `solsys_code/campaign_tables.py:433`. Mechanical renames (`font-weight-*` -> `fw-*`, `mr-/ml-` -> `me-/ms-`, `border-left` -> `border-start`, ...); `form-inline` -> flex utilities (`d-flex align-items-center gap-2`), `<select>` -> `form-select form-select-sm`. Same spirit as Phase 39 D-11 for `calendar.html`.
- **D-18:** High-band marker = `border-start border-4 border-success` (drop the inline `border-left-width` style), the green left bar UI-SPEC gave confirmable rows.
- **D-19:** New class guard next to `TestTemplatesUseBootstrap5DataAttributes`: a regex for removed Bootstrap 4 class names matched inside `class="..."` values (start from WR-15's pattern; the planner extends it as needed, e.g. `badge-*`, `custom-select`, `btn-block`). It scans every template, so D-17 must leave it green.
- **D-20:** Close IN-12 as well: both the data-attribute scan and the new class scan cover every `TEMPLATES[0]['DIRS']` entry and the app template directories (`APP_DIRS` is True), not just `DIRS[0]`. Mark IN-12 fixed in the disposition ledger. Templates only; Python-built HTML is not scanned (D-17 fixes its one hit by hand).
- **D-21:** The marker is proven twice: a `TestCase` asserts High-band rows carry `border-start border-4 border-success` and other rows don't (success criterion 4), plus a `@tag('functional')` Playwright test asserting the computed left border of a High-band row is 4px and green.
- **D-22:** Hidden inputs + a whitelisted redirect (the reviewer's fix), no HTMX. Forms carry the worklist state as hidden fields; `_redirect()` (`solsys_code/campaign_views.py:1386`) re-applies only whitelisted, validated keys: `band` (must be in `_ATTRIBUTION_BANDS`), `dismissed-page`, `dismissed-sort`, `confirmed-page`, `confirmed-sort` (the actual names come from the tables' `prefixed_page_field` / `prefixed_order_by_field`).
- **D-23:** All actions keep the state: confirm, dismiss, bulk confirm and both Undos all go through the one redirect. One code path, no special case for Undo.
- **D-24:** Forms inside the Dismissed table always send `dismissed-page` (the current page, defaulting to 1), because opening the section by click on page 1 leaves nothing in the URL. Posting from inside Dismissed therefore always reopens it.
- **D-25:** The redirect adds a section fragment (`#attribution-dismissed-section` or `#attribution-confirmed-section`, whichever the action came from) so the browser lands on the section being worked.
- **D-26:** Undo on a Confirmed row lands on the Confirmed section, same page; Dismissed keeps whatever state it had (it is not force-opened to show where the pair went).
- **D-27:** If the acted-on row was the only one on its page, land on the last valid page of that section (clamp): no error, no jump to page 1.
- **D-28:** A render test posts an Undo carrying `dismissed-page`, follows the redirect, and asserts `#attribution-dismissed-section` has `show` (WR-18's fix), plus tests for D-23, D-26 and D-27.
- **D-29:** Mark WR-13 to WR-18 and IN-12 fixed in `.planning/milestones/v2.4-phases/37.1-close-gap-alloc-06-exact-identity-system-links-on-ingest-int/37.1-REVIEW-DISPOSITION.md` as each lands.
- **D-30:** Paired docs in scope: every notebook moved onto the helper (all eight) re-executed with `jupyter nbconvert --to notebook --execute --inplace` from `docs/notebooks/pre_executed/` and committed with output; `docs/notebooks/README.md` (D-04 rule, helper usage, regeneration steps); `docs/runbooks/telescope_runs_calendar.rst` attribution-pass section (~lines 956-990: Dismissed "folded away until you click" / "stays open while you page", now also stays open across Undo and every other action); CLAUDE.md's paired-docs map gains a line for `docs/notebooks/fomo_nb_utils.py` (helper -> every pre-executed notebook; a behavior change re-executes them).

### Claude's Discretion

- Helper function names and signature (e.g. one `setup_scratch_django(mode='fresh'|'snapshot')` vs two functions), so long as D-07 to D-10 hold.
- Exact Bootstrap 4 class regex, as long as it covers every class D-17 removes.
- Whether the WARN-06 cell also adds a value check on the dip (the reviewer suggested one).
- Plan split and wave layout, within the ROADMAP's fixed order.

### Deferred Ideas (OUT OF SCOPE)

- Merging `feature/skymap` itself (its notebooks and `special_locations()` callers). Not this phase; D-06 only seeds the helper file so that merge is clean later.
- The other 19 pending todos belong to Phase 41's triage.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| WARN-02 | Guard fails a notebook whose current source routes Django to the developer database, regardless of stale output | "Guard design" below: top-level-statement rules, import-based Django detection, execution counts 1..N, executed helper tests |
| WARN-03 | No notebook relies on `assert` for the database-path check; raises an ordinary exception that survives `python -O` | Helper owns the check as `raise RuntimeError`; subprocess test under `python -O` and `PYTHONOPTIMIZE`; 7 setup-cell asserts found (all but `project_observation_calendar_demo`); remaining 236 reviewed |
| WARN-04 | Attribution page uses Bootstrap 5.3 class names only; High-band marker renders; template test asserts it | Complete class inventory and replacements; **D-18's class string alone draws a full box on a `<tr>` (verified in Chromium), needs `border-0`** |
| WARN-05 | Every notebook audited for live developer-database access; byte-copiers rebuilt; README states the rule | Per-notebook verdicts below (executed evidence); helper design; README rewrite list |
| WARN-06 | `telescope_runs_demo` reports the horizon dip in a single unit | **Already fixed by Phase 39.1-01 (commit 7953b1f)**; remaining work is keeping it true and optionally pinning it |
| WARN-07 | Undo on a Dismissed row keeps the section open on the same page | `AttributionDecisionView._redirect` map; django-tables2 field names; existing clamping; hidden-input sites |
</phase_requirements>

## Summary

Phase 40 has two independent halves that meet only in `campaign_lifecycle_demo.ipynb`. The notebook half is a refactor: eight pre-executed notebooks each carry a hand-copied setup cell (create a temp directory, point `FOMO_DATABASE_PATH` at it, set Django up, check the resolved path, migrate), and the guard test checks that cell's *text*. Moving that cell into one importable helper (`docs/notebooks/fomo_nb_utils.py`) lets the helper be tested by really running it, and lets the guard shrink to "does each notebook call the helper first, never set Django up itself, and show fresh output". I executed patched copies of the three byte-copy notebooks against a fresh migrated scratch database (never touching `src/fomo_db.sqlite3`): **`campaign_lifecycle_demo` and `load_telescope_runs_demo` run cleanly from empty with only incidental differences (primary-key numbers and row counts), so they go fresh; `reconcile_campaign_runs_demo` fails on an empty database (a `DoesNotExist` in cell 5 and an `AssertionError` in cell 8) because its whole point is the sweep over the real runs and events, so it snapshots.**

The attribution half is mostly mechanical, with four findings that change what the plan should say. (1) WARN-06 is already done: Phase 39.1-01 re-executed `telescope_runs_demo` and the committed output now reads `1.4211 deg`. (2) D-18's marker (`border-start border-4 border-success`) on a `<tr>` renders a 4px green box on all four sides, not a left bar, because Bootstrap's reboot gives table rows `border-style: solid` and `.border-4` sets the width of every side; adding `border-0` first fixes it (verified in Chromium; screenshots in the scratchpad). (3) D-20's "app template directories" would pull in third-party templates (Django REST framework alone has 63 matching lines) that FOMO does not own, so the scan must be limited to directories inside the repo (FOMO has none today; all its templates are under `src/templates`). (4) django-tables2 3.0.0 already clamps an out-of-range page to the last page, so D-27 needs validation of the posted page number, not a clamping calculation.

**Primary recommendation:** Land in this order inside two or three plans: (A) helper + helper tests (green on their own); (B) guard rewrite + all eight notebooks converted and re-executed in one commit stack that ends green (the old and new guards cannot both pass a half-converted tree); (C) Bootstrap sweep + marker (using `border-0 border-start border-4 border-success`) + class guard; (D) return-state redirect + tests + runbook; then re-execute `campaign_lifecycle_demo.ipynb` last so its committed output matches the final templates and view.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Scratch database setup, resolved-path check, teardown | Dev tooling helper (`docs/notebooks/fomo_nb_utils.py`) | Notebook setup cell (calls it) | Runs before `django.setup()`; must not import FOMO models at load (D-07) |
| Notebook isolation enforcement | Test suite (`solsys_code/tests`) | pre-commit `django-test` hook | The guard reads notebook JSON only; helper tests run the helper in a subprocess |
| High-band row marker | Template (`attribution_queue.html`) | Browser CSS (Bootstrap 5.3.3 from CDN) | Class names are server-rendered; the visual result depends on Bootstrap's table reboot |
| Worklist return state (band, page, sort) | API / backend (`AttributionDecisionView._redirect`) | Table classes + template (emit hidden inputs) | The redirect is the only sink for all six actions; the forms only carry state |
| Section open/closed state | Template conditional on `request.GET` | django-tables2 `prefixed_page_field` | Server renders `show`; Bootstrap collapse JS only handles later clicks |
| Page clamping | django-tables2 `RequestConfig` (already silent) | View validation of the posted number | Library already maps `EmptyPage` to the last page |
| Bootstrap 4 class guard | Test suite (`test_bootstrap5_rendering.py`) | — | Static scan of template files, no browser |

## Standard Stack

No new packages. Everything below is already installed and in use.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Django test runner (`python manage.py test`) | Django 5.2.17 installed | All new tests | The only runner (CLAUDE.md); `SimpleTestCase` for file-reading guards, `TestCase` for view tests, `StaticLiveServerTestCase` for Playwright |
| `sqlite3` stdlib (`Connection.backup`, `mode=ro` URI) | Python 3.11.13 | Snapshot of the developer database | The pattern `project_observation_calendar_demo` already uses; copes with a concurrent writer (D-02) |
| `ast` stdlib | Python 3.11 | Guard rules over notebook cells | Already used by `test_pre_executed_notebooks.py` |
| nbconvert | 7.17.1 [VERIFIED: `python -c "import nbconvert"`] | Re-execute notebooks (`--execute --inplace`) | The documented regeneration path (README) |
| django-tables2 | 3.0.0 [VERIFIED: `pip show django-tables2`] | Dismissed/Confirmed tables, page and sort field names | Already the table library |
| Playwright (sync API) + Chromium | chromium-1243 present in `~/.cache/ms-playwright` [VERIFIED: `ls`] | D-21 browser test | Existing `TestBootstrap5Rendering` class |
| Bootstrap | 5.3.3 via jsDelivr CDN [VERIFIED: rendered `{% bootstrap_css %}` output] | The CSS the pages really load | `tom_common/base.html` loads it through `django_bootstrap5` |
| ruff | pinned v0.16.9 (pre-commit); 0.16.10 on PATH | Lint/format | Use `pre-commit run ruff --files ...`; unpinned `ruff` can differ (CLAUDE.md) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Subprocess tests of the helper | In-process `django.setup()` | Not possible: the test process already has settings configured and a test database, so the resolved-path check would (correctly) refuse. A subprocess takes about 4 s for a full fresh migrate [VERIFIED: timed 3.8 s] |
| `importlib.util.spec_from_file_location` to load the helper in tests | `sys.path.insert` | spec loading does not mutate the test process's `sys.path`; use it for the in-process unit tests |

**Installation:** none.

**Version verification:** versions above were read from the installed environment on 2026-10-09, not from a registry, because nothing new is installed.

## Package Legitimacy Audit

No external packages are installed or recommended by this phase. Audit not applicable.

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Per-Notebook Isolation Verdicts (D-01, WARN-05)

**Method.** For the three byte-copy notebooks I made patched copies in the scratchpad (setup cell edited to build a fresh migrated scratch database instead of copying `src/fomo_db.sqlite3`; repo-root line hard-coded), executed them with `jupyter nbconvert --execute --allow-errors`, and diffed every code cell's output against the committed output. The real notebooks and the developer database were never opened by these runs (the patched setup cell has no reference to `dev_db_path`). `[VERIFIED: executed this session]`

| Notebook | Today | Verdict | Evidence |
|----------|-------|---------|----------|
| `campaign_lifecycle_demo` | `shutil.copy2` of dev DB | **Fresh** | 25 of 25 cells ran, 0 errors. 19 cells differ only in primary-key numbers and counts that were dev-database residue: `attribution_count=8` became `1`, `attribution_count after confirming: 7` became `0`, `orphans_needing_attribution_count(): 7` became `0`, `CalendarEvent.objects.count()` `346` became `7`. None of these numbers appear in markdown prose. It creates its own Observatory rows (`update_or_create` Y21/Y22/Y23), campaign, runs, staff user and orphan event. |
| `load_telescope_runs_demo` | `shutil.copy2` of dev DB | **Fresh** | 16 of 16 cells ran, 0 errors. Differences: the four Observatory rows print `created` instead of `updated/unchanged`; `CampaignRun.objects.count() before: 45` becomes `0`, `CalendarEvent` `233` becomes `0`; run pks. Its prose says the copy "already carries real production `Observatory`" rows, which must be rewritten. |
| `reconcile_campaign_runs_demo` | `shutil.copy2` of dev DB | **Snapshot** | Fresh run **fails**: cell 5 `DoesNotExist: CampaignRun matching query does not exist`, cell 8 `AssertionError: expected every remaining blank-url event to be one the cutover reported as unexplained, got remaining=[1, 2, 3, 4, 5, 6] vs. reported=[]`. Cell 4's cutover reports `unresolvable_site=6` (`resolved site has no Observatory record or no timezone set`) instead of the committed `duplicate_identity=3`, because the real `NTT` Observatory row is absent. The committed cells 1-3 and 6-7 print real-data counts (`total: 316`, `facility_url: 264`, `CampaignRun rows: 41`, `Done. runs: 41 ... unchanged: 52 ... retired: 5`) that become `0` on an empty database. |
| `project_observation_calendar_demo` | `mode=ro` + `Connection.backup()` | **Snapshot** (keep, D-03) | Source read through `sqlite3.connect(f'{dev_db_path.as_uri()}?mode=ro', uri=True)`; migrate after the snapshot. |
| `backfill_lco_observations_demo` | fresh scratch, migrate from empty | **Fresh** (already) | Setup comment says it is deliberately not a copy (a copy would bring the host's watched proposals). |
| `import_campaign_csv_demo` | fresh scratch | **Fresh** (already) | Setup comment. |
| `sync_gemini_observation_calendar_demo` | fresh scratch | **Fresh** (already) | Setup comment; also inserts `src/` on `sys.path` before the setup (keep in that notebook's bootstrap). |
| `telescope_runs_demo` | fresh scratch | **Fresh** (already) | Setup cell read directly. |

**Why snapshot output is not reproducible, and why byte-copy is unsafe (observed).** While I ran the experiment above, `src/fomo_db.sqlite3` changed on disk (mtime 13:32:25, my before/after md5 differed) because the developer's cron job `manage.py run_unattended` was running (PID 1892414, started 13:30) `[VERIFIED: ps, ls --time-style]`. That is the exact hazard WR-16 describes for `shutil.copy2`, and it also means a snapshot notebook's committed output reflects whatever the developer database held at regeneration time. That is acceptable for `reconcile_campaign_runs_demo` and `project_observation_calendar_demo` (their purpose), but the README should say these two cannot be regenerated on a machine without a populated developer database (D-05), and that their numbers will differ run to run.

**Source-pattern audit result (WARN-03, D-15).** An AST count finds 243 `assert` statements across the eight notebooks: backfill 1, lifecycle 94, import 7, load 28, project 8, reconcile 87, sync 17, telescope 1. Seven are the setup-cell database-path check (every notebook except `project_observation_calendar_demo`); these disappear when the helper owns the check. The remaining 236 are demonstration assertions; none has a write or management-command call in its test expression `[VERIFIED: AST scan this session]`. Record that sentence in the plan's audit note.

## Architecture Patterns

### System Architecture Diagram

```
Developer / CI                                   pre-commit django-test hook
    |                                                      |
    | jupyter nbconvert --execute --inplace                | python manage.py test (parallel)
    v                                                      v
pre_executed/<nb>.ipynb  (kernel cwd = docs/notebooks/pre_executed)      solsys_code/tests/
  cell 0: sys.path bootstrap (repo root + docs/notebooks)                  test_pre_executed_notebooks.py
          from fomo_nb_utils import setup_..., teardown_...                  reads notebook JSON only (ast, top-level statements)
          scratch = setup_...(repo_root, mode)  ----+                       rules: helper first, no own django.setup,
  cells 1..N-1: Django imports + demo                |                              no FOMO_DATABASE_PATH, no shutil.copy*,
  last cell: teardown_...(scratch)  ---------+       |                              exec counts 1..N, output lines present
                                             |       v                    test_notebook_helper.py (new)
                                             |   docs/notebooks/fomo_nb_utils.py                   loads helper by file path;
                                             |     mkdtemp(prefix='fomo-notebook-db-')             pure parts in-process,
                                             |     mode='snapshot': sqlite3 mode=ro + backup()     full setup in a subprocess
                                             |     os.environ['FOMO_DATABASE_PATH'] = scratch      (also under -O / PYTHONOPTIMIZE)
                                             |     django.setup()   (lazy import)
                                             |     if Path(resolved).resolve() != scratch.resolve(): raise RuntimeError
                                             |     migrate; print('Resolved database: ...')
                                             +--> connections.close_all(); shutil.rmtree(scratch dir)
                                                          |
                                                          v
                                                src/fomo/settings.py  DATABASES NAME = os.getenv('FOMO_DATABASE_PATH') or <dev db>
                                                (local_settings may override DATABASES -> the check must refuse)

Attribution page (WARN-04 / WARN-07)
Browser GET /campaigns/attribution/?band=&dismissed-page=&dismissed-sort=&confirmed-page=
   -> AttributionQueueView builds two django-tables2 tables (prefix 'dismissed-' / 'confirmed-'), RequestConfig clamps pages
   -> template: forms emit hidden state; Dismissed gets `show` when its page/sort field is in request.GET
Browser POST attribution_decide (confirm | confirm_selected | dismiss | undo_confirmation | undo_dismissal) + hidden state
   -> AttributionDecisionView -> _redirect(request): validate whitelist -> urlencode -> redirect(url + '#section')
   -> same GET as above
```

### Recommended Project Structure
```
docs/notebooks/
├── fomo_nb_utils.py                 # NEW (seeded from origin/feature/skymap, then extended)
├── README.md                        # rewrite isolation section (D-04)
└── pre_executed/*.ipynb             # all eight converted + re-executed
solsys_code/tests/
├── test_pre_executed_notebooks.py   # rewrite to helper-based rules (D-11..D-14)
├── test_notebook_helper.py          # NEW: runs the real helper
├── test_bootstrap5_rendering.py     # widen scans (D-19/D-20), add class guard; add functional marker test (D-21)
├── test_attribution_template.py     # GET-only marker render test (module rule: no client POST here)
└── test_campaign_attribution_views.py  # POST-then-follow return-state tests (D-23, D-26, D-27, D-28)
```

### Pattern 1: The helper (names are discretion; this is a recommended API)
**What:** Two entry points plus pure parts the tests can call directly.
```python
# docs/notebooks/fomo_nb_utils.py  (new functions go AFTER special_locations(), whose bytes stay identical to skymap's)
def setup_scratch_django(repo_root, mode='fresh'):
    """Route Django to a scratch database, set it up, check the route, migrate, print the evidence line.

    Args:
        repo_root: path of the repository (the directory holding manage.py).
        mode: 'fresh' migrates an empty database; 'snapshot' takes a read-only online-backup copy of
            src/fomo_db.sqlite3 and migrates it to head.

    Returns:
        pathlib.Path: the scratch directory, to hand to teardown_scratch_django().

    Raises:
        RuntimeError: snapshot requested but src/fomo_db.sqlite3 is missing (raised before anything is
            created or exported), or Django resolved a database other than the scratch one.
    """
```
Order inside (each point is a pitfall below): (1) snapshot only: raise if the developer database is missing, **before** `mkdtemp` or touching the environment; (2) `mkdtemp(prefix='fomo-notebook-db-')`; (3) snapshot only: `mode=ro` backup; (4) plain assignment `os.environ['FOMO_DATABASE_PATH'] = ...`; (5) `os.environ.setdefault` for `DJANGO_SETTINGS_MODULE` (`src.fomo.settings`) and `DJANGO_ALLOW_ASYNC_UNSAFE` (so a test can substitute settings); (6) lazy `import django; django.setup()`; (7) **the `raise RuntimeError` path check before migrate**; (8) `call_command('migrate', verbosity=0)`; (9) print `Resolved database: '<path>' (...)`.

Keep `scratch_db_dir` the scratch directory and the database file name `fomo_db.sqlite3`: `test_projector_demo_notebook.py` matches `Resolved database: '/tmp/fomo-notebook-db-[^/']+/fomo_db\.sqlite3'` (line 42) `[VERIFIED: read]`.

### Pattern 2: Notebook setup cell after conversion
```python
import sys
from pathlib import Path

# NOTE: parents[2] is correct only when the kernel CWD is docs/notebooks/pre_executed/.
repo_root_path = Path.cwd().resolve().parents[2]
if not (repo_root_path / 'manage.py').exists():
    raise RuntimeError(f'No manage.py at {repo_root_path}; run Jupyter from docs/notebooks/pre_executed/')
for entry in (repo_root_path, repo_root_path / 'docs' / 'notebooks'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fomo_nb_utils import setup_scratch_django  # noqa: E402

scratch_db_dir = setup_scratch_django(repo_root_path, mode='fresh')  # or mode='snapshot'
```
Nothing Django-related (not even `import django`) may precede the helper call (D-11). `campaign_lifecycle_demo`'s `django.test.utils.setup_test_environment()` (guarded by `except RuntimeError`) stays in that notebook, in a cell after the helper call; `sync_gemini_observation_calendar_demo` keeps its extra `src` `sys.path` entry in its bootstrap.

### Pattern 3: Guard rules (AST over notebook JSON; replaces `isolation_problems()`)
Statement order is `(code-cell index, top-level statement index)`.
- **Django-backed** = any `ast.Import` / `ast.ImportFrom` anywhere in any code cell (use `ast.walk`) whose top package is `django`, `solsys_code` or starts with `tom_` (D-13). For an unparseable cell fall back to a regex on `^\s*(from|import)\s+(django|solsys_code|tom_)` and report the cell (today's code reports unparseable cells that mention `django.setup`; every code cell in all eight notebooks currently parses `[VERIFIED: AST scan]`).
- **Setup call** = a top-level `Expr`/`Assign` whose value is a `Call` to the helper's setup function, resolved through `from fomo_nb_utils import <name>` (reject `as` aliases and `import *`) or `fomo_nb_utils.<name>(...)`. Exactly one. It must precede every Django-ish import in statement order. Use `tree.body` (not `ast.walk`) so a call inside an unexecuted `def`/`if False:` does not count.
- **Never in a notebook:** a call to `django.setup` (dotted name ending `django.setup`), `from django import setup`, any assignment to or call mutating `os.environ['FOMO_DATABASE_PATH']` (`Assign` subscript, `os.environ.setdefault/update`, `os.putenv`), any `shutil.copy`, `shutil.copy2`, `shutil.copyfile` (suggest also `copytree`/`copyfileobj`/`move` on a database; optional).
- **Teardown:** the last code cell's last top-level statement calls the helper's teardown with the scratch directory, and its committed output contains `Removed scratch database directory:` (keep that exact text; today's constant is `TEARDOWN_OUTPUT`).
- **Fresh run (D-12):** `[c.get('execution_count') for c in code_cells] == list(range(1, N+1))`. All eight notebooks already satisfy this `[VERIFIED: scan this session]`. Suggested extra (discretion): no code cell has an `output_type == 'error'` output.
- **Committed output (D-14):** setup cell output matches `Resolved database: '([^']+)'` and the path's parent directory name starts with `fomo-notebook-db-`.
- **Not vacuous (D-13):** every `*.ipynb` under `pre_executed/` is Django-backed (all eight) and passes; also assert the set of names equals the eight expected files so an unexpected ninth notebook is noticed.
- Read notebook JSON with `encoding='utf-8'` (closes IN-14 for free).
- Keep synthetic-notebook negative tests (the existing `_notebook()` builder) but update the reference cells to the helper form, and add one test per bypass WR-13 named (own `FOMO_DATABASE_PATH` assignment with a stale output; aliased `from django import setup as _s`; routing inside `def`/`if False:`; missing teardown; non-contiguous execution counts).

### Pattern 4: Executing the real helper from the Django test runner
Load the file without touching `sys.path`:
```python
spec = importlib.util.spec_from_file_location('fomo_nb_utils', HELPER_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)   # safe in-process only because the helper imports Django lazily (D-07)
```
- In-process (no Django needed): the snapshot function against a tiny SQLite file standing in for the developer database (assert contents copied, source bytes unchanged, URI contains `mode=ro` by wrapping `sqlite3.connect`), the missing-source `RuntimeError` (assert no `FOMO_DATABASE_PATH` change and no new `fomo-notebook-db-*` directory), and the pure path-check function with equal and unequal paths.
- Subprocess (`subprocess.run([sys.executable, '-c', ...], env=..., cwd=repo_root, timeout=...)`): (a) full fresh setup, assert exit 0, the printed `Resolved database:` line, a migrated file at the scratch path (about 4 s); (b) **a `local_settings`-style override**: put a one-file settings module on `PYTHONPATH` doing `from src.fomo.settings import *` then `DATABASES = {...other path...}`, set `DJANGO_SETTINGS_MODULE` to it, expect non-zero exit with `RuntimeError` in stderr **and the other database file never created** (proves the check runs before migrate); run (b) with `sys.executable -O` and with `PYTHONOPTIMIZE=1` (success criterion 2: still fires under `python -O`). Do not write `src/fomo/local_settings.py` (it is gitignored and host-specific; this machine's copy does not override `DATABASES`).
- Also add a cheap static test: the helper module contains no `ast.Assert`.

### Anti-Patterns to Avoid
- **Calling `django.setup()` in the test process to test the helper.** Settings are already configured to the in-memory test database; the (correct) path check will refuse, or worse, you test nothing.
- **Putting the missing-developer-database check after `mkdtemp`/`os.environ` assignment.** It leaks a directory and leaves the variable set for later tests in the same process.
- **Hard-coding names in the guard that the helper's final names later change.** Keep the helper's function names as module-level constants imported by the test (`from ... import` is not possible without executing the module, so read them from the spec-loaded module).
- **`ast.walk` for ordering rules.** It descends into `def`/`if False:` bodies (WR-13's last bullet).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Consistent copy of a database another process writes | `shutil.copy*` | `sqlite3` `mode=ro` source + `Connection.backup()` | Byte copies can catch a half-written file; verified live: the cron runner changed the file during this research |
| Out-of-range page handling | A custom clamp in `_redirect` | `django_tables2.RequestConfig.configure` (silent by default) | Source: `EmptyPage` -> `table.paginator.page(table.paginator.num_pages)`, `PageNotAnInteger` -> page 1 (`django_tables2/config.py`) `[VERIFIED: read]` |
| Query-string building in the redirect | f-string concatenation | `urllib.parse.urlencode` | Escapes values; path stays `reverse('campaigns:attribution')` so there is no open redirect |
| Hidden-input escaping | Manual string joining | `format_html` / template autoescape | Already how `render_actions` builds forms |
| Left bar on a table row | Inline `style` | Bootstrap utilities `border-0 border-start border-4 border-success` | Inline style loses to the `!important` utilities (verified); the utility combination is verified |
| Bootstrap 4 -> 5 name mapping | Guessing | The official migration guide table below | `[CITED: getbootstrap.com/docs/5.3/migration/]` |

**Key insight:** every hazard here already has a standard-library or already-installed answer; the work is wiring and tests, not new mechanisms.

## Runtime State Inventory

This is a refactor of notebook plumbing, not a rename, but one runtime-state question matters.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | `src/fomo_db.sqlite3` is read (snapshot notebooks only) and is mutated by the live cron `run_unattended` while this research ran. Five dated backup copies sit beside it in `src/` (untracked, irrelevant). | Code edit only. Never write the developer database; document that snapshot output varies. |
| Live service config | The cron entry runs `manage.py run_unattended` against the developer database every few minutes | None (reason to prefer the backup API over byte copy) |
| OS-registered state | None for this phase | None |
| Secrets/env vars | `FOMO_DATABASE_PATH` is read by `src/fomo/settings.py:100` as `os.getenv('FOMO_DATABASE_PATH') or os.path.join(BASE_DIR, 'fomo_db.sqlite3')`, so an **empty-string** value silently falls back to the developer database; the helper's plain non-empty assignment plus the resolved-path check covers it | Code edit (helper) |
| Build artifacts | `docs/notebooks/__pycache__/fomo_nb_utils.cpython-311.pyc` exists (stale, gitignored; Python ignores a `__pycache__` file whose source is absent). `docs/_build/` holds an old HTML build with the "arcmin deg" text (stale artifact, not tracked) | None; do not treat `docs/_build` as evidence of the current notebook |

## Bootstrap 5.3 sweep (D-17 to D-21)

**Complete inventory** (regex over every `*.html` under `src/templates` and every non-test `*.py` under `solsys_code`, `src/templatetags`, `src/fomo`; this session):

| File:line | Today | Replacement |
|-----------|-------|-------------|
| `attribution_queue.html` 6, 27, 42, 53, 56, 85-88, 143, 148, 159, 162, 187-190, 229, 249-250, 261-262 | `font-weight-bold` / `font-weight-normal` | `fw-bold` / `fw-normal` |
| `attribution_queue.html:8` | `<form class="form-inline mb-4">` | `class="d-flex align-items-center gap-2 mb-4"` |
| `attribution_queue.html:9` | `mr-2` on the label | `me-2` (or drop it, since `gap-2` now spaces) |
| `attribution_queue.html:10` | `<select class="form-control form-control-sm">` | `form-select form-select-sm` |
| `attribution_queue.html:63, 169` | `border-left border-success` + `style="border-left-width: 4px;"` | **`border-0 border-start border-4 border-success`**, no inline style (see below) |
| `campaign_list.html:6, 29, 40` | `font-weight-bold`, `mr-2` x2 | `fw-bold`, `me-2` |
| `campaignrun_table.html:24, 41, 52` | `font-weight-bold` | `fw-bold` |
| `campaignrun_table.html:27, 29, 30` | `mr-2` | `me-2` |
| `campaignrun_table.html:90` | `mr-3` | `me-3` |
| `campaignrun_table.html:40, 51` | `form-group` | not in D-17; removed in 5. Use `mb-3` |
| `campaignrun_table.html:58` | `<select class="form-control w-auto">` | `form-select w-auto` |
| `campaignrun_gap_analysis.html:6, 39, 50, 68, 85, 102` | `font-weight-bold` | `fw-bold` |
| `campaignrun_gap_analysis.html:56, 58` | `badge badge-info`, `badge badge-light` | not in D-17. `badge text-bg-info`, `badge text-bg-light` (this template family already uses `badge text-bg-success`) |
| `campaignrun_gap_analysis.html:68, 85, 102` | `<i class="fa ... mr-2">` | `me-2` |
| `approval_queue.html:6, 9, 19, 22` | `font-weight-bold` | `fw-bold` |
| `campaignrun_submit_form.html:6`, `submission_thanks.html:6` | `font-weight-bold` | `fw-bold` |
| `solsys_code/partials/campaign_links.html:1` | `btn btn-info mr-2 mb-2` | `me-2` |
| `solsys_code/campaign_tables.py:433` | `class="small ml-1"` (inside `ApprovalQueueTable`, the site cell, not an attribution table) | `ms-1` |
| `solsys_code/filters.py:63` (not scanned, not in D-17) | `css_class='form-group col-md-3'` (crispy layout) | Out of scope per D-20; mention in the plan as a known leftover |

Mapping sources: `.font-weight-*` -> `.fw-*`, `.ml-*`/`.mr-*` -> `.ms-*`/`.me-*`, `.border-left`/`.border-right` -> `.border-start`/`.border-end`, "Use our grid and utilities instead of `.form-group`, `.form-row`, or `.form-inline`", `.custom-select` -> `.form-select`, `.badge-*` colors dropped for background utilities, `.btn-block` dropped for `.d-grid` + `.gap-*` `[CITED: getbootstrap.com/docs/5.3/migration/]`. `text-muted` still works in 5.3 and is not in scope.

**D-18 correction (verified in Chromium against the exact CSS the pages load).** `border-start border-4 border-success` on a `<tr>` computes to `4px solid` on **all four sides** (`['4px','4px','4px','4px']`, green): Bootstrap's reboot gives `tr`/`td` `border-style: solid; border-width: 0`, and `.border-4` sets the width of every side, so the whole row gets a green box. Adding `border-0` first fixes it: `border-0 border-start border-4 border-success` computes to top/right/bottom `0px none` and left `4px solid rgb(25, 135, 84)`. Two consequences: the High-band row can no longer also carry `border-top` (a 4px top border appears when combined), so a non-first High row loses its thin separator line; and the old inline `style="border-left-width: 4px"` cannot work (the `!important` utility wins, it computed `1px`). `[VERIFIED: Playwright/Chromium run on Bootstrap 5.3.3 CSS; screenshots in the scratchpad]`. The D-21 `TestCase` substring check `border-start border-4 border-success` still passes with `border-0` in front; the Playwright test should assert all four computed border widths (`4px`, `0px`, `0px`, `0px`) and the left color `rgb(25, 135, 84)`.

**D-19 regex.** Start from WR-15's pattern and add the names this inventory found. Keep it matching inside `class="..."` only. Caveat: `class="[^"]*` stops at the first double quote, so a template tag containing `"` inside a class value ends the match early (none do today: the templates use single quotes inside tags).
```python
BOOTSTRAP4_CLASS = re.compile(
    r'class="[^"]*(?<![\w-])(?:font-weight-(?:bold|normal|light|bolder|lighter)|border-(?:left|right)|'
    r'[mp][lr]-(?:\d|auto)|form-inline|form-group|form-row|text-(?:left|right)|float-(?:left|right)|'
    r'custom-select|btn-block|badge-[a-z]+|sr-only|no-gutters|font-italic)(?![\w-])'
)
```
Use the lookarounds, not `\b`: `\b` treats `-` as a boundary, so `\bmr-2` would also match inside `me-mr-2`-like tokens and `border-left` would match `border-left-width`. Follow Phase 39's precedent of writing the forbidden names so the test file itself does not contain them (for example `'border-' + 'left'`, `test_calendar_template.py:2228`) when a test also asserts absence by `assertNotIn`.

**D-20 trap.** The widened scan must not include third-party app template directories. FOMO's own `solsys_code` has no `templates/` directory; `get_app_template_dirs('templates')` returns only installed third-party apps, and running the regex over them finds matches that FOMO does not own: `rest_framework` 63 lines, `tom_targets` 5, `tom_calendar` 4, `tom_jpl` 3, `tom_common` 2, `crispy_bootstrap5` 1, `tom_dataproducts` 1 `[VERIFIED: ran regex over every app template dir]`. A guard over all of them would fail forever. Recommended implementation: collect `TEMPLATES[0]['DIRS']` **plus** the entries of `django.template.utils.get_app_template_dirs('templates')` filtered to paths inside the repository (`Path(d).resolve().is_relative_to(REPO_ROOT)`), build the `relative` display path against whichever collected directory contains the file (this is what removes IN-12's `ValueError` on a second `DIRS` entry), and keep `test_the_scan_is_not_vacuous`'s `>= 20` templates check (26 templates found today under `src/templates`). Today the filtered app-directory list is empty; the code is future-proofing, which should be said in the test's docstring. This is a clarification of D-20, flagged for the user in the Assumptions Log (A1).

## Attribution return state (D-22 to D-28)

**Facts, with source** `[VERIFIED: read this session]`:
- The view is `AttributionDecisionView` (`solsys_code/campaign_views.py:1343`), not "AttributionActionView". All five actions (`confirm`, `confirm_selected`, `dismiss`, `undo_confirmation`, `undo_dismissal`, tuple at line 1301) finish with `return self._redirect(request)`, so one change covers D-23. Current body, lines 1386-1393: `band = request.POST.get('band')`; `if band in _ATTRIBUTION_BANDS: return redirect(f'{reverse("campaigns:attribution")}?band={band}')`; `return redirect('campaigns:attribution')`.
- The tables are built in `AttributionQueueView.get_context_data` with `prefix='dismissed-'` / `prefix='confirmed-'` and `RequestConfig(self.request).configure(...)` (lines 1277-1284). django-tables2 3.0.0 defaults are `order_by_field='sort'`, `page_field='page'`, `per_page=25`, `per_page_field='per_page'` (`django_tables2/tables.py:142-145`), so the field names are **`dismissed-page`, `dismissed-sort`, `confirmed-page`, `confirmed-sort`** and the existing tests use `confirmed-per_page` / `dismissed-per_page` (`test_attribution_template.py`). Use `table.prefixed_page_field` / `prefixed_order_by_field` rather than literals.
- Dismissed opens only when `dismissed_table.prefixed_page_field` or `prefixed_order_by_field` is a key of `request.GET` (condition repeated three times, template lines 250-257, IN-15). Confirmed is always `show`.
- Per-row forms are emitted in three places: `AttributionDismissedTable.render_actions` (`campaign_tables.py:629`, `undo_dismissal`), `AttributionConfirmedTable.render_actions` (line 731, `undo_confirmation`), and the template's three `<form>`s: the per-candidate confirm/dismiss forms for events (template line 107) and records (line 197), and the two bulk forms (`bulk-confirm-events` line 134, `bulk-confirm-records` line 220). All five of those already carry one `band` hidden input; they must carry the same state as the table forms.
- The two worklists (`event_page`, `record_page`) have no paging links in the template, so they need no preserved state.
- The rows lists are capped at 50 (`limit=50`) and the table default is 25 per page, so each table has at most two pages.
- **`ApprovalQueueTable.render_actions` is at `campaign_tables.py:492`** (the CONTEXT's "lines 492, 629, 731" includes it, but it is the approval queue, not the attribution page; leave it alone).

**Clamping (D-27) is already done by the library, with one quirk.** `RequestConfig.configure` is silent by default: `PageNotAnInteger` -> page 1, `EmptyPage` -> last page, and a non-integer `?dismissed-page=abc` is dropped before paginating (`int(...)` in a `try`). Django's paginator treats page `0` or a negative number as `EmptyPage`, so `dismissed-page=0` also lands on the **last** page. So `_redirect` should (a) pass the posted page through unchanged when it is a positive integer, and (b) drop it otherwise; no clamping computation is needed, and a deleted-only-row-on-page-2 redirect `?dismissed-page=2` renders page 1 of 1 with no error. A test needs more than 25 dismissals to reach page 2 (create 26, undo the 26th), or construct the expectation against `table.paginator.num_pages`.

**Recommended design (all of D-22 to D-26):**
1. One function, for example `attribution_return_state(querydict) -> dict[str, str]`, returning only whitelisted, validated pairs: `band` in `_ATTRIBUTION_BANDS`; each `*-page` a positive decimal integer; each `*-sort` matching `^-?[a-z_]+$` (django-tables2 drops invalid or unorderable names itself, `Table.order_by` setter). The view's `_redirect` calls it on `request.POST`; the page's own GET context calls it on `request.GET` to feed the template forms; the table classes call it on `self.request.GET`.
2. `_redirect`: `params = attribution_return_state(request.POST)`, `url = reverse('campaigns:attribution')`, then `f'{url}?{urlencode(params)}'` when `params` is non-empty, plus a fragment: `#attribution-dismissed-section` for `undo_dismissal` (and any action posted from the Dismissed table), `#attribution-confirmed-section` for `undo_confirmation`, none for the worklist actions (D-25 only names the two sections; pick and document the worklist fragment, or omit it). The fragment is a fixed constant, never taken from input. Django's test client drops fragments when following redirects, so the D-28 test asserts on the rendered page, not on the URL fragment (assert `response.redirect_chain[-1][0]` ends with the fragment if the fragment itself is to be pinned).
3. Dismissed-table forms always include `dismissed-page` = `self.page.number` (D-24). Confirmed-table forms include `confirmed-page` and every other whitelisted key present in the request, so Dismissed keeps whatever state it had (D-26).
4. **Pitfall:** existing tests construct the tables as `AttributionDismissedTable([], request=None).render_actions(dismissal)` (`test_campaign_attribution_views.py`, `TestUndoDismissalPrompt`). `render_actions` must therefore tolerate `request is None` and a table that was never paginated (no `page` attribute); fall back to page 1 and no extra state.
5. Template: replace the repeated `{% if dismissed_table.prefixed_page_field in request.GET or ... %}` with one context boolean computed in `get_context_data` (closes IN-15), and render the hidden inputs from a context list through one small include or a loop, not five hand-copied blocks.

**Where each test goes.** `test_attribution_template.py`'s docstring forbids client POSTs in that module ("do not add one"); put the POST-then-follow tests (D-23, D-26, D-27, D-28) in `test_campaign_attribution_views.py` (`AttributionViewTestBase` provides `staff_user`, `campaign_run`, `_make_event`, `_make_record`), and keep the GET-only marker test (D-21 `TestCase` half) in `test_attribution_template.py`. The existing `test_evidence_columns_score_and_band_badge_all_present` (line 815) asserts only `'border-success'`, which passed while the marker was broken (WR-15); strengthen it or add the exact-class assertion beside it.

**Browser test (D-21 second half).** Add to `TestBootstrap5Rendering` (the `@tag('functional')` class at the top of `test_bootstrap5_rendering.py`); reuse the fixture style of `test_attribution_page_sections_open_and_close_with_bootstrap5` and `self._log_in_browser(staff)`. It needs the page's CSS from the jsDelivr CDN (network access; the existing functional tests already need it for Bootstrap JS). Assert via `page.evaluate` on `getComputedStyle(row)`. A second browser test for Undo is optional but cheap: register `page.on('dialog', lambda d: d.accept())` because the Undo button's `onclick="return confirm(...)"` would otherwise block.

## Common Pitfalls

### Pitfall 1: Old and new guard cannot both pass a half-converted tree
**What goes wrong:** The commit hook runs the whole suite (about 80 s). If notebooks are converted before the guard is rewritten (or vice versa) the suite is red at that commit.
**How to avoid:** Land helper + helper tests first (green, additive). Then guard rewrite and all eight notebook conversions plus their re-executions as one stack that ends green (or use `SKIP=django-test` only for a deliberately red intermediate commit, per CLAUDE.md).

### Pitfall 2: The second notebook guard
**What goes wrong:** `solsys_code/tests/test_projector_demo_notebook.py` also reads `project_observation_calendar_demo.ipynb`: it addresses the setup cell by its nbformat id `7022f987` (line 36) and requires that cell's output to contain `scratch copy`, match `Resolved database: '/tmp/fomo-notebook-db-[^/']+/fomo_db\.sqlite3'` (line 42) and not contain `the developer database itself`. It also reads cell ids `05528b38`, `556d2a9f`, `250b5d0b`.
**How to avoid:** When rewriting that notebook's setup cell keep its `id`; make the helper's snapshot-mode evidence line contain the words `scratch copy`. Fresh-mode wording is free. Edit notebooks with `nbformat`, preserving `id`, `metadata` and `nbsphinx.execute = never`.

### Pitfall 3: The path check must come before `migrate`, and must not be an `assert`
Seven notebooks check with `assert resolved_db_name == str(scratch_db_path)` immediately before `migrate`. Under `python -O` it vanishes and `migrate` would run against whatever `local_settings.py` selected. The helper uses `Path(resolved).resolve() != scratch.resolve()` and `raise RuntimeError`.

### Pitfall 4: Re-running the setup cell in one kernel
`django.setup()` is a no-op the second time, so the freshly made scratch path no longer matches the resolved one and the helper (correctly) raises. Today's `setup_test_environment()` `except RuntimeError` in `campaign_lifecycle_demo` hints that people re-run cells. README should say "Restart & Run All" (and `nbconvert` always does).

### Pitfall 5: Teardown under an open connection (IN-17) and leftovers (IN-13)
`shutil.rmtree` while Django holds the file open works on Linux but hides failures elsewhere. In the helper's teardown call `connections.close_all()` (lazy import, tolerate Django not set up) before removing. A failed run never reaches teardown, leaving a full developer-database snapshot in `/tmp` (mode 0700 from `mkdtemp`); README should tell operators to delete leftover `fomo-notebook-db-*` directories. `import_campaign_csv_demo` also leaves a second unprefixed temp directory (IN-13); using `scratch_db_dir` for it is a one-line fix if folded in.

### Pitfall 6: Add/add conflict with `feature/skymap` is not zero
D-06 says seeding from the skymap file leaves "nothing to reconcile". I reproduced the later merge in a scratch repository: the two branches both adding `docs/notebooks/fomo_nb_utils.py` with different content gives `CONFLICT (add/add)` with **one** conflict hunk, even when this branch's file is the skymap file plus appended functions `[VERIFIED: git merge in a temp repo]`. Keep `special_locations()` byte-identical (including the U+2212 minus signs) and append the new functions below it so the resolution is "keep both"; do not reformat or reorder the seeded part.

### Pitfall 7: The skymap helper passes the pinned ruff
Concern from CONTEXT: do the U+2212 minus signs trip RUF001? `ruff` v0.16.9 with the repo config: `check` reports `All checks passed!` and `format --diff` is empty `[VERIFIED: pinned binary from the pre-commit cache, stdin-filename docs/notebooks/fomo_nb_utils.py]`. The repo's `[tool.ruff.lint] select` has no `RUF` family, so RUF001 is never enabled (it does fire if `RUF001` is selected explicitly, which is why I looked). The `ruff` hook's `types_or: [python, pyi]` covers any `.py` under `docs/notebooks/`; `ruff-format` also covers `jupyter` (notebooks are formatted, not linted). New helper functions need docstrings (`D103` is selected) with Google-style sections.

### Pitfall 8: Snapshot against a developer database behind head
The snapshot copy can be behind the current migrations; migrate after the backup (as `project_observation_calendar_demo` does).

### Pitfall 9: Regenerating `campaign_lifecycle_demo` twice
It must be executed when converted (the new guard needs fresh helper output) and again after the attribution changes (new cells showing the marker and the Undo return state, if added). It runs on an empty database in roughly a minute, so two executions are cheap; plan the second as the last task of the phase.

### Pitfall 10: Prose in every notebook's setup markdown
Markdown cells that describe the old setup must be rewritten, by cell id: `campaign_lifecycle_demo` 5c49e87b, 7dfef438, 061b7205, 32a7b0c9, 6a3dae2b, bbf027b0, 234805b6; `load_telescope_runs_demo` 5febc666, 49ff72df, 041476f5, 78e0bb2f, 31ff3edb; `reconcile_campaign_runs_demo` 8b703ea8, dfcc0343, e1edbaa7, c4cd78c1, 9a9bfc44, e5a1c3d7, 680797dd, d2adacf8, 98bba968 (these mention "scratch copy of the developer database"; reconcile stays a snapshot but should say "online-backup snapshot", not "copies"); the other five notebooks' cell 1 setup description also changes to "calls the helper". `[VERIFIED: regex over markdown cells]`

## Code Examples

### Read-only online backup (pattern to lift into the helper)
```python
# Source: docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb (setup cell, read this session)
source_connection = sqlite3.connect(f'{dev_db_path.as_uri()}?mode=ro', uri=True)
scratch_connection = sqlite3.connect(scratch_db_path)
try:
    source_connection.backup(scratch_connection)
finally:
    scratch_connection.close()
    source_connection.close()
```

### Resolved-path check (D-09)
```python
# Source: same notebook (WR-08's fix)
if Path(resolved_db_name).resolve() != scratch_db_path.resolve():
    raise RuntimeError(
        'This notebook must never write to the developer database -- resolved DB '
        f'{resolved_db_name!r} is not the scratch copy {str(scratch_db_path)!r}.'
    )
```

### Verified High-band marker
```html
<!-- verified computed style: left 4px solid rgb(25,135,84); top/right/bottom 0px none -->
<tr class="{% if candidate.band == 'high' %}border-0 border-start border-4 border-success{% elif not forloop.first %}border-top{% endif %}">
```

### Whitelisted redirect (sketch)
```python
_RETURN_STATE_PAGE_KEYS = ('dismissed-page', 'confirmed-page')
_RETURN_STATE_SORT_KEYS = ('dismissed-sort', 'confirmed-sort')

def attribution_return_state(data) -> dict[str, str]:
    """Return the validated worklist state (band, page and sort per table) found in a QueryDict."""
    state = {}
    if data.get('band') in _ATTRIBUTION_BANDS:
        state['band'] = data['band']
    for key in _RETURN_STATE_PAGE_KEYS:
        if data.get(key, '').isdecimal() and int(data[key]) >= 1:
            state[key] = str(int(data[key]))
    for key in _RETURN_STATE_SORT_KEYS:
        if re.fullmatch(r'-?[a-z_]+', data.get(key, '')):
            state[key] = data[key]
    return state
```
(The key strings should come from the tables' `prefixed_*` properties at the call sites; the literals here are for the sketch.)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `font-weight-*`, `ml-*`/`mr-*`, `border-left`, `form-inline`, `form-group`, `badge-*` | `fw-*`, `ms-*`/`me-*`, `border-start`, flex utilities / grid, `text-bg-*` | Bootstrap 5.0 | These names are inert under the 5.3.3 that `tom_common/base.html` loads |
| Dismissal state in URL only on page/sort links | State in hidden inputs, replayed by `_redirect` | this phase | Undo no longer folds the section |

**Deprecated/outdated:** `.text-muted` is deprecated but still works in 5.3 `[ASSUMED]` (not in scope); `docs/_build/` HTML is a stale artifact.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | D-20's "app template directories" should mean only directories inside the repository (none exist today), not third-party apps' templates | Bootstrap 5.3 sweep | If the user wants third-party templates scanned, the guard fails on code FOMO cannot edit. Needs user confirmation. |
| A2 | D-18 should be amended to `border-0 border-start border-4 border-success`, and High-band rows drop `border-top` | Bootstrap 5.3 sweep | If left as written the row shows a full green box; the amended class string still contains the D-18 substring so D-21's test text is unaffected. Needs user confirmation (small visual change). |
| A3 | WARN-06 needs no code change because Phase 39.1-01 already fixed it; only a re-execution check and an optional pin remain | Summary | If the developer wants a value assertion added (the reviewer suggested one), that is an extra `assert abs(horizon_dip(2402).to_value(u.deg) - 1.44) <= 0.02` in the notebook and a re-execution |
| A4 | Worklist actions (confirm, dismiss, bulk) need no section fragment; only the two Undo/section cases do | Attribution return state | D-25 only names the two section fragments; if the user wants the worklist to scroll somewhere, a fragment must be chosen |
| A5 | `.text-muted` still renders in 5.3 so it is left alone | State of the Art | Cosmetic only |
| A6 | Snapshot notebooks' output varying between regenerations is acceptable | Per-notebook verdicts | If reviewers expect reproducible outputs, those two notebooks need a different approach (a fixture database) |

## Open Questions (RESOLVED)

All three were decided at planning (2026-10-09); each resolution names the plan and task that implements it.

1. **Should the guard also reject database-path string literals (`'fomo_db.sqlite3'`, a bare `sqlite3` import) in notebook code?** D-11 lists three rules only. Today the literal appears only as the scratch file name inside the setup cell and in teardown prints, both of which move into the helper. A literal ban would close a route D-11 does not (`open('src/fomo_db.sqlite3', 'rb')`). Recommendation: add it as a fourth rule; it costs one AST pass and no notebook needs the literal after conversion. Planner's call.
   **RESOLVED -- adopted.** The guard gains a fourth rule: no notebook code string may name `fomo_db.sqlite3` (a comment may; the check is AST-based, so only `ast.Constant` strings count). Implemented by **40-02 Task 1** as rule R7 (of R0-R11, defined in its interfaces block) in the rewritten `solsys_code/tests/test_pre_executed_notebooks.py`, with the synthetic-notebook tests `test_developer_database_literal_in_code_is_a_problem` and `test_developer_database_name_in_a_comment_is_not_a_problem`; recorded in 40-02's objective ("Research Open Question 1 -- adopted"). After conversion the literal lives only in the helper `docs/notebooks/fomo_nb_utils.py` (40-01 Task 1).
2. **Does the helper print the closing "The developer database ... was never opened for writing" line in teardown?** The notebooks print it today (last cell of every notebook). Moving it into the helper keeps the committed output the same text; the guard only requires `Removed scratch database directory:`.
   **RESOLVED -- yes.** `teardown_scratch_django()` prints both closing lines the notebooks print today: `Removed scratch database directory: <dir>` and then `The developer database (src/fomo_db.sqlite3) was never opened for writing.`, so the committed output keeps the same text. Implemented by **40-01 Task 1** (the helper in `docs/notebooks/fomo_nb_utils.py`; recorded in 40-01's objective under "Teardown (research Open Question 2, IN-17)" and in its interfaces block); every notebook's last cell becomes exactly `teardown_scratch_django(scratch_db_dir)` (40-01 Tasks 1 and 2 for four notebooks, 40-02 Task 1 for the other four), and the guard still requires only the `Removed scratch database directory:` line (40-02 rule R10).
3. **`solsys_code/filters.py:63` still has `form-group col-md-3`.** Out of scope under D-20 (Python-built HTML not scanned); recommend one line in the plan noting it was seen and left.
   **RESOLVED -- left alone and recorded.** `solsys_code/filters.py:63` (`css_class='form-group col-md-3'`, a Python-built crispy layout) is not changed in this phase: D-20 does not scan Python-built HTML and D-17 names only `campaign_tables.py:433`. Recorded in **40-03**'s objective ("Left alone"), and **40-03 Task 3** leaves the file unchanged -- its `<verify>` command fails if that `css_class` line is gone or the file has uncommitted changes.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | everything | yes | 3.11.13 | — |
| jupyter / nbconvert | re-executing notebooks | yes | nbconvert 7.17.1 | — |
| Playwright + Chromium | D-21 browser test, WARN-04 verification | yes | chromium-1243 | — |
| Network to `cdn.jsdelivr.net` | functional tests need Bootstrap CSS/JS | yes (HTTP 200, 232803 bytes) | Bootstrap 5.3.3 | none; the functional tests already depend on it |
| pinned ruff 0.16.9 | lint/format | yes, in `~/.cache/pre-commit/repof44_zo6b/py_env-python3.11/bin/ruff` | 0.16.9 | `pre-commit run ruff ...` |
| `src/fomo_db.sqlite3` | snapshot notebooks (reconcile, project_observation) | yes, written live by cron `run_unattended` | — | none by design (D-05 RuntimeError) |
| SPICE kernels (`~/.cache/sorcha`) | importing `solsys_code.views` in test runs | yes (existing suite runs) | — | — |
| Node 22 | `gsd-tools.cjs` only (`nvm use 22`), not this phase's code | n/a | — | — |

**Missing dependencies with no fallback:** none.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django test runner (`python manage.py test`), the project's only runner |
| Config file | none (settings in `src/fomo/settings.py`; test-only overrides under `sys.argv[1] == 'test'`) |
| Quick run command | `python manage.py test solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_notebook_helper solsys_code.tests.test_projector_demo_notebook solsys_code.tests.test_attribution_template solsys_code.tests.test_bootstrap5_rendering.TestTemplatesUseBootstrap5DataAttributes` (baseline of the existing four modules: 42 tests in 0.4 s, 4.7 s wall) |
| Full suite command | `python manage.py test --exclude-tag functional --exclude-tag ephemeris_segfault --exclude-tag migration --parallel` (the hook's set; about 80 s) |
| Browser tests | `python manage.py test solsys_code.tests.test_bootstrap5_rendering --tag functional` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| WARN-02 | Guard flags: own `FOMO_DATABASE_PATH` assignment with stale output, aliased `django.setup`, setup inside `def`/`if False:`, missing helper call, helper call after a Django import, `shutil.copy*`, non-contiguous execution counts | unit (SimpleTestCase, synthetic notebooks) | `python manage.py test solsys_code.tests.test_pre_executed_notebooks` | rewrite existing |
| WARN-02 | All eight real notebooks pass; not vacuous (all eight Django-backed) | unit | same | rewrite existing |
| WARN-03 | Helper raises `RuntimeError` (not `AssertionError`) when a `DATABASES` override wins, under `python -O` and `PYTHONOPTIMIZE=1`, before any migrate; helper has no `ast.Assert` | subprocess + static | `python manage.py test solsys_code.tests.test_notebook_helper` | Wave 0 |
| WARN-05 | Fresh setup routes, migrates, prints evidence; snapshot uses `mode=ro` backup and leaves the source unchanged; missing source raises `RuntimeError` first | in-process + subprocess | same | Wave 0 |
| WARN-05 | README names every snapshot/fresh notebook | manual review (or a small test comparing README text to the guard's lists) | — | n/a |
| WARN-06 | Committed `telescope_runs_demo` output has the dip line in degrees and no "arcmin deg" | unit (add to guard: no committed output anywhere contains `arcmin deg`) | `python manage.py test solsys_code.tests.test_pre_executed_notebooks` | Wave 0 add |
| WARN-04 | No removed Bootstrap 4 class in any repo-owned template; data-attribute scan covers every `DIRS` entry | unit | `python manage.py test solsys_code.tests.test_bootstrap5_rendering.TestTemplatesUseBootstrap5DataAttributes` (+ new class) | extend |
| WARN-04 | High-band rows render `border-start border-4 border-success` (and `border-0`), other rows do not | GET render test | `python manage.py test solsys_code.tests.test_attribution_template` | Wave 0 add |
| WARN-04 | High-band row computed left border 4px green, other sides 0 | functional (Playwright) | `python manage.py test solsys_code.tests.test_bootstrap5_rendering --tag functional` | Wave 0 add |
| WARN-07 | Undo (dismissal) carrying `dismissed-page`: redirect keeps it, Dismissed has `show`; confirmed Undo lands on Confirmed same page with Dismissed state unchanged; clamp when the acted-on row was alone on page 2; all five actions keep band/page/sort; tampered values (`band=evil`, `dismissed-page=0`, `abc`, `-1`, sort with spaces) are dropped | `TestCase` (POST + follow) | `python manage.py test solsys_code.tests.test_campaign_attribution_views` | Wave 0 add |
| WARN-07 | `render_actions` still works with `request=None` | unit | same | exists, keep green |

### Sampling Rate
- **Per task commit:** the quick run command above (about 5 s plus new subprocess tests, roughly 10 s more).
- **Per wave merge:** the full suite command.
- **Phase gate:** full suite green, plus `--tag functional` for the Bootstrap file, plus a fresh `nbconvert --execute` of every converted notebook (an executed notebook with `execution_count` 1..N and no error output is itself the end-to-end check of the helper).

### Wave 0 Gaps
- [ ] `solsys_code/tests/test_notebook_helper.py` covers WARN-03, WARN-05 (new file)
- [ ] `docs/notebooks/fomo_nb_utils.py` must exist before the new tests can load it
- [ ] Guard rewrite in `solsys_code/tests/test_pre_executed_notebooks.py` covers WARN-02, WARN-06
- [ ] Marker, class-guard and return-state tests (the three files named in the map) cover WARN-04, WARN-07
- No framework install needed.

## Security Domain

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no (unchanged) | existing `StaffRequiredMixin` |
| V3 Session Management | no | — |
| V4 Access Control | yes (unchanged, must stay) | `AttributionQueueView` and `AttributionDecisionView` keep `StaffRequiredMixin`; a test that an anonymous or non-staff POST still writes nothing already exists (`TestAttributionStaffGating`) and must keep passing |
| V5 Input Validation | **yes** | `_redirect` accepts only whitelisted keys; `band` against `_ATTRIBUTION_BANDS`; page fields positive decimal integers; sort fields `^-?[a-z_]+$`; build the query with `urlencode`; path from `reverse()`; fragment a constant |
| V6 Cryptography | no | — |

### Known Threat Patterns
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Open redirect or header injection through a reflected return parameter | Tampering | Redirect target is `reverse('campaigns:attribution')` plus an `urlencode`d whitelist; never use a posted URL or fragment |
| Reflected markup in hidden inputs | Tampering / XSS | Emit values through `format_html` or template autoescape; the values are already validated to digits, letters and underscores |
| Developer-database corruption or leak into committed notebook output | Tampering / Information disclosure | `mode=ro` backup, resolved-path `RuntimeError` before migrate, guard bans `shutil.copy*`; snapshot output (user names, runs) lands in committed notebooks, so keep the existing "no credential-shaped text" test (`test_no_credential_shaped_text_in_the_notebook_file`) and consider extending it to every pre-executed notebook |
| Leftover full snapshot of the developer database in `/tmp` after a failed run (password hashes, tokens, sessions) | Information disclosure | `mkdtemp` is mode 0700; README line telling operators to remove leftover `fomo-notebook-db-*` directories (IN-17) |
| CSRF on the decision endpoint | Tampering | Unchanged: every form carries `csrfmiddlewaretoken`; new hidden inputs must not remove it |

## Sources

### Primary (HIGH confidence)
- Repository files read this session: `40-CONTEXT.md`, `solsys_code/tests/test_pre_executed_notebooks.py`, `test_projector_demo_notebook.py`, `test_bootstrap5_rendering.py`, `test_attribution_template.py`, `test_campaign_attribution_views.py`, `solsys_code/campaign_views.py` (1175-1700), `solsys_code/campaign_tables.py` (420-770), `src/templates/campaigns/*.html`, `src/fomo/settings.py`, `pyproject.toml`, `.pre-commit-config.yaml`, `docs/notebooks/README.md`, `docs/runbooks/telescope_runs_calendar.rst` (940-1000), all eight pre-executed notebooks (as JSON)
- Executed this session: patched-copy notebook runs on a fresh database; Chromium/Playwright computed-style experiments against the real Bootstrap 5.3.3 CSS; timing of a fresh `migrate`; pinned-ruff checks of the skymap helper; scratch-repository merge experiment; AST scans of assert counts and execution counts; regex scan of every app template directory
- `git show origin/feature/skymap:docs/notebooks/fomo_nb_utils.py` (fetched read-only); `git show fe15b0c:.planning/phases/37.1-.../37.1-REVIEW.md` (original WR-13..WR-18, IN-12..IN-17); commit `7953b1f` (WARN-06 already fixed)
- django-tables2 3.0.0 source: `config.py`, `tables.py:142-145, 552-602`

### Secondary (MEDIUM confidence)
- `[CITED: getbootstrap.com/docs/5.3/migration/]` (fetched): the Bootstrap 4 to 5 rename table used in the sweep

### Tertiary (LOW confidence)
- none

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH, nothing new and all versions read from the environment.
- Architecture: HIGH, helper/guard/redirect designs follow code I read and ran; the helper API names are discretion.
- Pitfalls: HIGH, each was reproduced or read in source this session.

**Research date:** 2026-10-09
**Valid until:** 30 days (stable code; the one fast-moving part is the Bootstrap CDN version pinned by `django_bootstrap5`)
