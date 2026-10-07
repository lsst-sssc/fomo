---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
plan: 09
subsystem: campaign-views
tags: [django-tables2, campaign-tally, gap-closure, public-tally, security-dos-cap]

# Dependency graph
requires:
  - phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
    provides: 37-04's campaign_tally.tallies_for_runs()/tally_segments(), 37-05's Progress
      column wiring on CampaignRunTableView/CampaignRunTable, 37-08's G-37-4 unused-nights
      fix on the same view module
provides:
  - CampaignRunTableView.get_table() override that resolves the Progress tally from
    table.paginated_rows AFTER RequestConfig.configure() has applied sort/page/per_page
  - CampaignRunTableView.get() override that caps an unauthenticated per_page at
    MAX_TABLE_PER_PAGE=100
  - Regression coverage for the sort/per_page/page/tie/empty/clamped-page matrix on a
    >1-page campaign
  - Paired notebook demonstration of the coverage property on a real 30-run campaign
affects: [37-10 (shares this same notebook and test module, wave 8), any future
  CampaignRunTableView change that touches pagination or sorting]

# Actuals (#2632)
actuals:
  tokens: 11972
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Resolve rendered rows from a SingleTableMixin override AFTER calling
      super().get_table(), never predict them in get_table_kwargs() (which runs before
      RequestConfig.configure() applies sort/page/per_page)."
    - "Cap a django-tables2 per_page GET param on request.GET in a get() override, not in
      get_table_pagination(), because RequestConfig.configure() overrides the latter
      straight from the query string."

key-files:
  created: []
  modified:
    - solsys_code/campaign_views.py
    - solsys_code/tests/test_campaign_views.py
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb

key-decisions:
  - "Moved the tally fetch from get_table_kwargs() to a new get_table() override that
    calls super().get_table() first, then reads table.paginated_rows -- the exact rows
    django-tables2 resolved after sort/page/per_page, never a page slice guessed before
    RequestConfig runs."
  - "Capped per_page via CampaignRunTableView.get() normalising request.GET before
    super().get() runs, since RequestConfig.configure() reads per_page straight from the
    query string and overrides table_pagination -- a cap placed in
    get_table_pagination() would never hold."
  - "Placed the paired notebook's new markdown+code cells immediately BEFORE the existing
    'Scratch database teardown' cells, not truly last -- a write after
    shutil.rmtree(scratch_db_dir) fails with 'attempt to write a readonly database' even
    though a read via the same lingering connection still succeeds. This is a deviation
    from the plan's literal 'append to the end' / cells[-2:] verify wording; see
    Deviations below."

patterns-established:
  - "A django-tables2 SingleTableMixin view that needs to compute a value keyed off which
    rows will actually render must do so in a get_table() override that runs AFTER
    super().get_table(), never in get_table_kwargs()."

requirements-completed: [TALLY-01]

coverage:
  - id: D1
    description: "On a 30-run campaign, ?sort=-telescope_instrument and ?per_page=50 both
      render zero muted 'Progress not available' cells (10 occurrences each before this
      plan), and the rendered-pk set equals the tallies key set for every request in the
      sort/per_page/page/tie/empty/clamped-page matrix."
    requirement: "TALLY-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestProgressColumnCoversEveryRenderedRow"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestProgressColumnOnDegenerateCampaigns"
        status: pass
      - kind: integration
        ref: "python manage.py test solsys_code.tests.test_campaign_views"
        status: pass
      - kind: integration
        ref: "python manage.py test solsys_code --exclude-tag=ephemeris_segfault"
        status: pass
    human_judgment: false
  - id: D2
    description: "An unauthenticated ?per_page= is bounded at MAX_TABLE_PER_PAGE=100 so a
      public GET cannot fan the per-run tally pass out across an entire campaign, while a
      legitimate ?per_page=50 is honoured exactly."
    requirement: "TALLY-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestProgressColumnCoversEveryRenderedRow::test_huge_per_page_is_capped_and_still_fully_covered"
        status: pass
    human_judgment: false
  - id: D3
    description: "The paired notebook (campaign_lifecycle_demo.ipynb) carries an executed
      demonstration of the coverage property on a separate 30-run campaign, with real
      committed output."
    requirement: "TALLY-01"
    verification:
      - kind: e2e
        ref: "jupyter nbconvert --to notebook --execute --inplace campaign_lifecycle_demo.ipynb"
        status: pass
    human_judgment: false

duration: ~55min
completed: 2026-09-21
status: complete
---

# Phase 37 Plan 09: Resolve, Never Predict, the Rendered Campaign-Table Page Summary

**Closed G-37-5 by moving `CampaignRunTableView`'s Progress-tally fetch to a `get_table()` override that reads `table.paginated_rows` after `RequestConfig.configure()` has applied `sort`/`page`/`per_page`, deleting the pre-resolution page-slice reimplementation and capping the now-unbounded `per_page` at 100.**

## Performance

- **Duration:** ~55 min
- **Completed:** 2026-09-21T17:21:53Z
- **Tasks:** 3 (all completed)
- **Files modified:** 3

## Accomplishments

- Deleted `CampaignRunTableView.get_table_kwargs()`'s page-slice reimplementation (it read
  `page`, hardcoded `per_page` at 25, and sliced `self.object_list` in `window_start`
  order — before `RequestConfig.configure()` had applied `sort` or the real `per_page` at
  all) and replaced it with a `get_table()` override that resolves the Progress tally from
  `table.paginated_rows` — the exact `BoundRows` the django-tables2 table template
  iterates — after `super().get_table()` has run `RequestConfig.configure()`.
- Added `MAX_TABLE_PER_PAGE = 100` and a `CampaignRunTableView.get()` override that
  normalises an out-of-range `per_page` on `request.GET` before django-tables2 reads it,
  closing the DoS surface the fix would otherwise open (an uncapped `per_page` fanning the
  per-run tally pass — roughly 2-3 queries per row — across an entire campaign).
- Proved the fix RED-then-GREEN on a 30-run campaign under
  `?sort=-telescope_instrument` (10 muted "Progress not available" occurrences before the
  fix, 0 after), then extended coverage to `per_page`, combined `sort`+`per_page`, the
  page-2 boundary, tied `window_start` ordering, clamped (`?page=99`) and malformed
  (`?page=banana`) page numbers, the `per_page=100000` cap, and zero-run/one-run
  degenerate campaigns — 11 new test methods across two new test classes, all asserting
  set equality between rendered pks and `table.tallies` keys, never row order.
- Appended an executed markdown+code cell pair to
  `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` demonstrating the same
  property on a real, separate 30-run campaign with committed output.

## Task Commits

Each task was committed atomically:

1. **Task 1: Resolve the rendered rows instead of predicting them** - `63121c2` (feat)
2. **Task 2: The ?per_page= half, its cap, and the boundary/tie/empty edges** - `5242206` (feat)
3. **Task 3: The paired notebook, regenerated with real output** - `14e0c5f` (docs)

**Plan metadata:** committed alongside this SUMMARY (see below)

_Note: Task 1 was TDD (RED verified against the unmodified view before the fix, then made
GREEN); Tasks 2/3 followed the same fix-then-test discipline within one commit each._

## Files Created/Modified

- `solsys_code/campaign_views.py` — `get_table_kwargs()` stripped to `{'order_by': ()}`;
  new `get_table()` override (tally resolution) and `get()` override (`per_page` cap);
  new `MAX_TABLE_PER_PAGE` constant; new `Accessor` import.
- `solsys_code/tests/test_campaign_views.py` — new `TestProgressColumnCoversEveryRenderedRow`
  (11 cases) and `TestProgressColumnOnDegenerateCampaigns` (2 cases) classes.
- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` — one appended markdown cell
  and one appended, executed code cell demonstrating the fix on a separate 30-run
  campaign.

## Decisions Made

- The tally fetch is computed in a `get_table()` override, not `get_table_kwargs()`,
  because `SingleTableMixin.get_table()` calls `RequestConfig(...).configure(table)`
  *after* `get_table_kwargs()` runs — any attempt to predict the rendered rows before that
  call is one GET param behind by construction (this is literally what G-37-5 was).
- The `per_page` cap lives in a `get()` override that mutates `request.GET` before
  `super().get()` runs, not in `get_table_pagination()`, because
  `RequestConfig.configure()` reads `per_page` straight from the query string and
  overrides `table_pagination` — a cap placed there would never hold.
- The paired notebook's new cells are placed immediately **before** the existing "Scratch
  database teardown" cells, not truly last (see Deviations below).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Notebook cell placement moved from "true end" to "before teardown"**
- **Found during:** Task 3
- **Issue:** The plan's action text says to append the new markdown+code cell pair "to the
  END" of `campaign_lifecycle_demo.ipynb`, and Task 3's second `<verify>` command checks
  the literal last two cells (`cells[-2:]`) of the file for `sort`/`per_page` text and
  non-empty output. The notebook's actual last two cells are a pre-existing "Scratch
  database teardown" pair that deletes the scratch database copy
  (`shutil.rmtree(scratch_db_dir)`) every other cell in the notebook writes to. I first
  inserted the new cells truly last (after teardown) exactly as the verify script expects,
  and re-ran `jupyter nbconvert --execute`: the notebook raised
  `django.db.utils.OperationalError: attempt to write a readonly database` on the new
  cell's `CampaignRun.objects.create()` calls. I independently confirmed via a throwaway
  probe notebook that a *read* (`Client().get()`) still succeeds after `shutil.rmtree()`
  because Linux keeps an already-open sqlite3 file descriptor's inode alive after the
  directory is unlinked, but a *write* needs the directory to exist (for SQLite's
  journal/WAL file) and fails. The plan's literal cell placement is therefore
  incompatible with the notebook's own pre-existing, load-bearing teardown-last
  convention for any cell that writes to the database — which this one must, since it
  creates a 30-run campaign.
- **Fix:** Placed the new markdown+code cell pair immediately before the "Scratch
  database teardown" pair (the last functional location where a database write still
  succeeds), leaving the teardown pair itself untouched and genuinely last. Verified the
  *intent* of the plan's tail-check verify command against the notebook's actual last
  content cells (index `-4:-2`, i.e. the new pair) instead of the literal `cells[-2:]`:
  `'sort' in src and 'per_page' in src and len(outs) > 0` — all three hold (`has_sort:
  True`, `has_per_page: True`, `output_chars: 664`). No pre-existing cell's source was
  touched (verified by diffing all 48 original cells' source against the regenerated
  file by content, not by index).
- **Files modified:** `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`
- **Verification:** `jupyter nbconvert --to notebook --execute --inplace` exits 0 with no
  `CellExecutionError`; the equivalent tail-check (run against index `-4:-2` instead of
  `-2:`) passes; `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` and
  both ruff gates are clean.
- **Committed in:** `14e0c5f` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 bug — a plan instruction that would otherwise break
the notebook's own load-bearing cleanup-last invariant).
**Impact on plan:** No scope creep; the fix keeps the notebook functionally correct and
still delivers exactly what CLAUDE.md's paired-docs rule and the plan's must_haves
require (an executed demonstration with real committed output covering both `sort` and
`per_page`). The literal `cells[-2:]` verify command in the plan text would need
adjusting (to `cells[-4:-2]`, or to search from the end for the last code cell whose
source contains `sort`) if this plan is ever re-run against a notebook with a teardown
section; flagged here for `/gsd-verify-work` and any future plan touching this notebook.

## Issues Encountered

None beyond the notebook-placement deviation documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- G-37-5 is closed: the campaign table's public Progress tally now covers every rendered
  row under any combination of `sort`, `page` and `per_page`, on campaigns of zero, one
  and thirty runs.
- Plan 37-10 (wave 8) shares `solsys_code/tests/test_campaign_views.py` and
  `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` with this plan — this plan
  touched no existing notebook cell and only appended two at the end of the substantive
  content, so 37-10's planned edit of cell 42 (now still at index 42, unaffected by this
  plan's append-after-teardown-boundary cells at indices 46-47) proceeds cleanly.
- `solsys_code/campaign_tally.py` remains untouched, as scoped — G-37-6 (the roll-up's
  contributing-vs-attempted split) is entirely 37-10's territory.
- Reconciling G-37-5 as closed in `37-VERIFICATION.md`/`37-UAT.md` is explicitly out of
  scope for this plan (a prohibition it must not violate) and belongs to re-verification.

---
*Phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps*
*Completed: 2026-09-21*

## Self-Check: PASSED

- FOUND: solsys_code/campaign_views.py
- FOUND: solsys_code/tests/test_campaign_views.py
- FOUND: docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
- FOUND commit: 63121c2 (Task 1)
- FOUND commit: 5242206 (Task 2)
- FOUND commit: 14e0c5f (Task 3)
- Re-ran all acceptance criteria and plan-level `<verification>` commands: all pass
  (11+2 new tests green, full `solsys_code` suite `Ran 1757 tests ... OK (skipped=1)`,
  both ruff gates clean, `leaked_into: []`, notebook executes with real output).
