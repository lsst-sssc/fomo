# Phase 40: Notebook Isolation & Attribution Page - Context

**Gathered:** 2026-10-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Close the six remaining Phase 37.1 review warnings (WR-13 to WR-18 → WARN-02 to WARN-07):

- **Notebooks (first):** every pre-executed demo notebook builds and uses its own scratch database
  through one shared helper; nothing byte-copies or writes `src/fomo_db.sqlite3`; the notebook guard
  test checks each notebook's *current source* (not stale output); the database-path check is an
  ordinary exception that survives `python -O`; `telescope_runs_demo`'s horizon dip prints in one unit.
- **Attribution page (second):** FOMO's templates use Bootstrap 5.3 class names only, the High-band
  row marker renders again (proven by a template test and a browser test), and every action on the
  attribution page — Undo in particular — returns the staff member to the same section, page, sort
  and band filter.

Order inside the phase is fixed by ROADMAP: WARN-05 → WARN-02/WARN-03 (→ WARN-06) → WARN-04 → WARN-07.
`campaign_lifecycle_demo.ipynb` is rebuilt on the new helper and re-executed **once**, after the
attribution-page changes it documents have landed (or re-executed again if the attribution work
changes its output — the planner sequences this so the committed output matches the final code).

</domain>

<decisions>
## Implementation Decisions

### Scratch database source (WARN-05)
- **D-01:** **Mixed per notebook.** A notebook migrates a fresh, empty scratch database and creates
  every row it shows, **unless its committed output depends on rows it did not create itself** — then
  it takes a read-only snapshot. The researcher applies that test to each of the three byte-copy
  notebooks (`campaign_lifecycle_demo`, `load_telescope_runs_demo`, `reconcile_campaign_runs_demo`)
  and records the verdict per notebook. Expected (not locked): `reconcile_campaign_runs_demo` snapshots
  (its sweep shows 45 real runs / 264 real events); the other two likely go fresh because they
  `update_or_create` their own Observatory/campaign rows — verify, don't assume.
- **D-02:** **Snapshots use SQLite's online backup only** — a `mode=ro` URI connection plus
  `Connection.backup()`, the pattern `project_observation_calendar_demo` already uses. No
  `shutil.copy*` of any database anywhere.
- **D-03:** **`project_observation_calendar_demo` keeps its read-only snapshot** (it is the reference
  pattern). It moves onto the shared helper like every other notebook.
- **D-04:** **Success criterion 1 is reworded**, in `ROADMAP.md` (Phase 40 SC 1) and
  `REQUIREMENTS.md` (WARN-05), from "no pre-executed notebook reads or copies `src/fomo_db.sqlite3`" to
  the rule actually adopted: no notebook byte-copies or writes the developer database; a notebook
  that needs real rows reads it only through a read-only online-backup snapshot; every other notebook
  starts from a fresh migrated database. `docs/notebooks/README.md` states the same rule in the same
  words, and lists which notebooks snapshot and which start fresh.
- **D-05:** **A snapshot notebook on a machine with no `src/fomo_db.sqlite3` raises a clear
  `RuntimeError`** (today's behavior) — no silent fallback to a fresh database.

### Shared helper module
- **D-06:** **One shared helper module, not a per-notebook copy of the setup code.** It lives at
  **`docs/notebooks/fomo_nb_utils.py`** — the name and location the developer already used on the
  unmerged `feature/skymap` branch. **Start from `feature/skymap`'s version** of the file
  (`git show origin/feature/skymap:docs/notebooks/fomo_nb_utils.py`, which carries
  `special_locations()`), and add the scratch-database functions to it, so the later skymap merge has
  nothing to reconcile in this file. — **Reversibility:** costly — every pre-executed notebook and
  the guard depend on the helper's name, location and call signature.
- **D-07:** **Django imports inside the helper are lazy** (inside the function bodies), so a
  docs-build notebook importing only `special_locations()` never pulls Django in. The helper must not
  import FOMO models at module load — it runs before `django.setup()`.
- **D-08:** **Import path.** Pre-executed notebooks start their kernel in
  `docs/notebooks/pre_executed/`, so their setup cell adds `docs/notebooks` to `sys.path` alongside the
  repo root it already adds, then imports the helper by bare name (`from fomo_nb_utils import ...`).
  The repo-root/`sys.path` bootstrap stays in each notebook (it has to run before the helper can be
  imported); everything after it — scratch dir via `tempfile.mkdtemp(prefix='fomo-notebook-db-')`,
  plain assignment of `FOMO_DATABASE_PATH` (never `setdefault`), fresh `migrate` or `mode=ro` backup
  snapshot, `django.setup()`, the resolved-path check, printing `Resolved database: ...` — moves into
  the helper.
- **D-09:** **The resolved-path check is a `raise RuntimeError`, never an `assert`** (WARN-03), using
  `Path(...).resolve()` comparison as `project_observation_calendar_demo` does today. Living in the
  helper, it is written once for all eight notebooks.
- **D-10:** **The helper provides the teardown too** (removing the scratch directory); every
  Django-backed notebook's last code cell calls it.

### Guard strictness (WARN-02, WARN-03)
- **D-11:** **The guard checks notebooks' use of the helper; tests execute the helper for real.**
  Guard rules, per notebook (top-level statements only — iterate `tree.body`, not `ast.walk`, so a
  call inside an unexecuted `def`/`if False:` does not count):
  - the helper's setup call appears before any `django` / `solsys_code` / `tom_*` import or use;
  - the notebook itself never calls `django.setup()` and never assigns `FOMO_DATABASE_PATH`;
  - no `shutil.copy`, `shutil.copy2` or `shutil.copyfile` anywhere in the notebook.
  Helper tests (Django test runner, in `solsys_code/tests/`) run the real functions: fresh-database
  routing, the `mode=ro` backup snapshot, the missing-developer-database `RuntimeError`, and a
  `RuntimeError` when a `local_settings`-style `DATABASES` override wins over the scratch path. These
  executed tests replace today's source-pattern checks of the setup cell (mkdtemp prefix, assignment
  form, resolved-name read) rather than duplicating them.
- **D-12:** **Require one fresh top-to-bottom run:** code-cell `execution_count`s must be exactly
  `1..N`. This closes WR-13's stale-output hole; `nbconvert --execute --inplace` always produces it.
- **D-13:** **Django-backed detection is by import:** any notebook that imports `django`,
  `solsys_code` or any `tom_*` package is Django-backed and must satisfy the guard. Renaming or
  aliasing `django.setup` can no longer hide a notebook. The not-vacuous test covers all eight
  notebooks, not just four.
- **D-14:** **Keep the committed-output checks, via the helper:** the setup cell's committed output
  must contain the helper's `Resolved database: ...fomo-notebook-db-...` line, and the last code cell
  must call the helper's teardown with its committed output present. With D-12 that output is known
  to belong to the current source.
- **D-15:** **The ~250 demonstration `assert`s in the eight notebooks stay as they are.** WARN-03 is
  about the database-path guard, which D-09 moves into the helper. The audit (all eight notebooks,
  per the ROADMAP scope note) records that each remaining `assert` was reviewed and protects no write.
- **D-16:** **WARN-06: the horizon dip prints in degrees** (`dip.to(u.deg)`), matching the Stage-1
  constraint (1.44° ± 0.02° at 2402 m), the cell comment and the notebook intro.

### Bootstrap 5.3 sweep (WARN-04)
- **D-17:** **Tidy every FOMO template, not only the attribution page:** all eight templates the scan
  found (`src/templates/campaigns/attribution_queue.html`, `campaign_list.html`,
  `campaignrun_table.html`, `approval_queue.html`, `campaignrun_gap_analysis.html`,
  `campaignrun_submit_form.html`, `submission_thanks.html`, and
  `src/templates/solsys_code/partials/campaign_links.html`), plus the `ml-1` in the HTML built in
  `solsys_code/campaign_tables.py:433`. Mechanical renames (`font-weight-*` → `fw-*`, `mr-/ml-` →
  `me-/ms-`, `border-left` → `border-start`, …); `form-inline` → flex utilities
  (`d-flex align-items-center gap-2`), `<select>` → `form-select form-select-sm`. Same spirit as Phase
  39 D-11 for `calendar.html`.
- **D-18:** **High-band marker** = `border-0 border-start border-4 border-success` (drop the inline
  `border-left-width` style, and High-band rows drop `border-top`) — the green left bar UI-SPEC gave
  confirmable rows. *Amended 2026-10-09 after research (RESEARCH A2, developer-confirmed): on a `<tr>`
  the original `border-start border-4 border-success` draws a 4px green box on all four sides under
  Bootstrap 5.3.3, because reboot gives rows `border-style: solid` and `.border-4` sets every side;
  `border-0` first leaves a left bar only.*
- **D-19:** **New class guard** next to `TestTemplatesUseBootstrap5DataAttributes`: a regex for removed
  Bootstrap 4 class names matched inside `class="..."` values (start from WR-15's pattern; the
  planner extends it as needed, e.g. `badge-*`, `custom-select`, `btn-block`). It scans **every**
  template, so D-17 must leave it green.
- **D-20:** **Close IN-12 as well:** both the data-attribute scan and the new class scan cover every
  `TEMPLATES[0]['DIRS']` entry **and** any app template directory that lives inside this repository
  (none exist today), not just `DIRS[0]`. *Amended 2026-10-09 after research (RESEARCH A1,
  developer-confirmed): third-party apps' template directories (rest_framework, tom_targets,
  tom_calendar, tom_jpl, ...) are out of scope; they carry Bootstrap 4 names FOMO cannot edit.* Mark IN-12 fixed in the disposition ledger. Templates only — Python-built HTML is not
  scanned (D-17 fixes its one hit by hand).
- **D-21:** **The marker is proven twice:** a `TestCase` asserts High-band rows carry
  `border-start border-4 border-success` and other rows don't (success criterion 4), **plus** a
  `@tag('functional')` Playwright test asserting the computed left border of a High-band row is 4px
  and green.

### Attribution page return state (WARN-07)
- **D-22:** **Hidden inputs + a whitelisted redirect** (the reviewer's fix), no HTMX. Forms carry the
  worklist state as hidden fields; `_redirect()` (`solsys_code/campaign_views.py:1386`) re-applies only
  whitelisted, validated keys: `band` (must be in `_ATTRIBUTION_BANDS`), `dismissed-page`,
  `dismissed-sort`, `confirmed-page`, `confirmed-sort` (the actual names come from the tables'
  `prefixed_page_field` / `prefixed_order_by_field`).
- **D-23:** **All actions keep the state** — confirm, dismiss, bulk confirm and both Undos all go
  through the one redirect. One code path, no special case for Undo.
- **D-24:** **Forms inside the Dismissed table always send `dismissed-page`** (the current page,
  defaulting to 1), because opening the section by click on page 1 leaves nothing in the URL. Posting
  from inside Dismissed therefore always reopens it.
- **D-25:** **The redirect adds a section fragment** (`#attribution-dismissed-section` or
  `#attribution-confirmed-section`, whichever the action came from) so the browser lands on the
  section being worked.
- **D-26:** **Undo on a Confirmed row lands on the Confirmed section, same page**; Dismissed keeps
  whatever state it had (it is not force-opened to show where the pair went).
- **D-27:** **If the acted-on row was the only one on its page, land on the last valid page** of that
  section (clamp) — no error, no jump to page 1.
- **D-28:** A render test posts an Undo carrying `dismissed-page`, follows the redirect, and asserts
  `#attribution-dismissed-section` has `show` (WR-18's fix), plus tests for D-23, D-26 and D-27.

### Ledgers and paired docs
- **D-29:** Mark WR-13 to WR-18 and IN-12 fixed in
  `.planning/milestones/v2.4-phases/37.1-close-gap-alloc-06-exact-identity-system-links-on-ingest-int/37.1-REVIEW-DISPOSITION.md`
  as each lands.
- **D-30:** Paired docs in scope: every notebook moved onto the helper (all eight) re-executed with
  `jupyter nbconvert --to notebook --execute --inplace` from `docs/notebooks/pre_executed/` and
  committed with output; `docs/notebooks/README.md` (D-04 rule, helper usage, regeneration steps);
  `docs/runbooks/telescope_runs_calendar.rst` attribution-pass section (~lines 956-990: Dismissed
  "folded away until you click" / "stays open while you page" — now also stays open across Undo and
  every other action); CLAUDE.md's paired-docs map gains a line for `docs/notebooks/fomo_nb_utils.py`
  (helper → every pre-executed notebook; a behavior change re-executes them).

### Claude's Discretion
- Helper function names and signature (e.g. one `setup_scratch_django(mode='fresh'|'snapshot')` vs two
  functions), so long as D-07 to D-10 hold.
- Exact Bootstrap 4 class regex, as long as it covers every class D-17 removes.
- Whether the WARN-06 cell also adds a value check on the dip (the reviewer suggested one).
- Plan split and wave layout, within the ROADMAP's fixed order.

### Folded Todos
- **2026-10-02 "Run pre-executed demo notebooks against a scratch DB copy, never the live dev DB"**
  (`.planning/todos/pending/2026-10-02-run-pre-executed-demo-notebooks-against-a-scratch-db-copy-ne.md`,
  `resolves_phase: 40`). Its problem (regenerations leaking demo rows into the real DB) is WARN-05's;
  its suggested "shared setup cell / one helper" is D-06. Phase 41 records it closed.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope and requirements
- `.planning/ROADMAP.md` §"Phase 40: Notebook Isolation & Attribution Page" — goal, order, scope notes, success criteria (SC 1 to be reworded per D-04)
- `.planning/REQUIREMENTS.md` — WARN-02 to WARN-07 (WARN-05 wording to be updated per D-04)

### The review findings being closed
- `git show fe15b0c:.planning/phases/37.1-close-gap-alloc-06-exact-identity-system-links-on-ingest-int/37.1-REVIEW.md` — **original full text of WR-13 to WR-18** (bypasses found, reviewer's suggested fixes and code). The archived `37.1-REVIEW.md` was overwritten by later review rounds and no longer contains them.
- `.planning/milestones/v2.4-phases/37.1-close-gap-alloc-06-exact-identity-system-links-on-ingest-int/37.1-REVIEW-DISPOSITION.md` — ledger rows WR-13..WR-18 and IN-12 to mark fixed (D-29)

### Shared helper precedent
- `git show origin/feature/skymap:docs/notebooks/fomo_nb_utils.py` — the developer's existing notebook helper to start from (D-06); used by `rubin_obsloctap.ipynb` / `ztf_obsloctap.ipynb` on that branch via bare `from fomo_nb_utils import special_locations`. Note: its SkyCoord strings use U+2212 minus signs (astropy parses them correctly, verified); check ruff doesn't flag them (RUF001) before landing.

### Notebooks and guard
- `docs/notebooks/pre_executed/*.ipynb` — the eight pre-executed notebooks
- `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` — reference `mode=ro` + `Connection.backup()` snapshot and `raise`-not-`assert` path check
- `solsys_code/tests/test_pre_executed_notebooks.py` — current guard (`isolation_problems()`, `django_setup_cells()`), to be rewritten per D-11 to D-14
- `docs/notebooks/README.md` — isolation-rule section to rewrite (D-04)
- `docs/notebooks.rst` — toctree listing the pre-executed notebooks
- `.pre-commit-config.yaml` — pre-executed notebooks keep output and must be marked never-execute
- `solsys_code/telescope_runs.py` `horizon_dip()` — returns arcminutes (WARN-06)

### Attribution page
- `src/templates/campaigns/attribution_queue.html` — High-band rows (lines ~63, ~169), Dismissed/Confirmed collapse sections (~235-270)
- `solsys_code/campaign_views.py` — `AttributionActionView.post()` / `_redirect()` (~1360-1393)
- `solsys_code/campaign_tables.py` — `render_actions()` (lines 492, 629, 731) where hidden inputs go; `ml-1` at 433
- `solsys_code/tests/test_bootstrap5_rendering.py` — `TestTemplatesUseBootstrap5DataAttributes` (line 459), to extend (D-19, D-20)
- `docs/runbooks/telescope_runs_calendar.rst` ~lines 956-990 — attribution-pass Dismissed/Undo text (D-30)

### Project rules
- `CLAUDE.md` — paired-docs rule and notebook map (D-30), `NonSiderealTargetFactory` for Target fixtures, `@tag('functional')` for Playwright tests, Django runner only

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `project_observation_calendar_demo`'s setup cell: working `mode=ro` + `backup()` snapshot and `Path.resolve()` raise — lift into the helper.
- `feature/skymap`'s `fomo_nb_utils.py`: file to seed the helper (D-06).
- `TestTemplatesUseBootstrap5DataAttributes._template_files()` and its not-vacuous test: the scan scaffold to widen (D-19, D-20).
- `_ATTRIBUTION_BANDS` and the existing band hidden field (Phase 28 D-10): the start of the D-22 whitelist.

### Established Patterns
- Setup cells today: repo root found via `Path.cwd().resolve().parents[2]` with a `manage.py` check; `DJANGO_SETTINGS_MODULE` / `DJANGO_ALLOW_ASYNC_UNSAFE` via `setdefault`; `FOMO_DATABASE_PATH` via plain assignment before `django.setup()`; `src/fomo/settings.py` reads `FOMO_DATABASE_PATH`.
- `campaign_lifecycle_demo` also calls `django.test.utils.setup_test_environment()` — the helper (or that notebook) keeps this.
- Dismissed section opens only when `dismissed_table.prefixed_page_field` or `prefixed_order_by_field` is in `request.GET`; Confirmed is always `show`.
- `_undo_confirmation()` writes a dismissal row, so an undone confirmation lands in Dismissed.

### Integration Points
- Every pre-executed notebook's setup cell and last cell (all eight move to the helper).
- `src/fomo/settings.py` `TEMPLATES` (`DIRS` + `APP_DIRS=True`) for the widened scans.
- The `functional` Playwright suite (`StaticLiveServerTestCase`) for D-21.

</code_context>

<specifics>
## Specific Ideas

- The developer wants one merged `fomo_nb_utils.py` in `docs/notebooks/` serving both docs-build and pre-executed notebooks, matching the helper they wrote on `feature/skymap`, rather than a new FOMO-app module or a second helper file.
- `pre_executed/` already sits inside `docs/notebooks/`; the split is by execution policy (docs-build-executed vs never-execute with committed output), not by location.

</specifics>

<deferred>
## Deferred Ideas

- Merging `feature/skymap` itself (its notebooks and `special_locations()` callers) — not this phase; D-06 only seeds the helper file so that merge is clean later.

### Reviewed Todos (not folded)
- The other 19 pending todos matched only on generic keywords (score 0.6 or below). They belong to Phase 41's triage, per ROADMAP.

</deferred>

---

*Phase: 40-notebook-isolation-attribution-page*
*Context gathered: 2026-10-09*
