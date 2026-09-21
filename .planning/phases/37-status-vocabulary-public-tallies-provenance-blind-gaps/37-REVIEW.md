---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
reviewed: 2026-09-21T18:06:59Z
depth: deep
scope: incremental
diff_base: 2c5906c4a61f1fd6745d6f4c1aa60ede4f3e15d0
files_reviewed: 7
files_reviewed_list:
  - solsys_code/campaign_views.py
  - solsys_code/campaign_tally.py
  - solsys_code/tests/test_campaign_views.py
  - solsys_code/tests/test_campaign_tally.py
  - src/templates/campaigns/campaignrun_table.html
  - docs/runbooks/telescope_runs_calendar.rst
  - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
findings:
  critical: 1
  warning: 7
  info: 6
  total: 14
status: issues_found
---

# Phase 37: Code Review Report (incremental — gap-closure plans 37-09 and 37-10)

**Reviewed:** 2026-09-21T18:06:59Z
**Depth:** deep
**Diff base:** `2c5906c` (commit of the previous 37-REVIEW.md)
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Only the work landed since `2c5906c` was reviewed: plan 37-09 (`CampaignRunTableView.get_table()`
rewrite + `MAX_TABLE_PER_PAGE`) and plan 37-10 (contributing-vs-attempted unused split,
`unused_unknown_runs`, fourth strip branch), plus their tests and paired docs.

**What holds up.** The core of 37-09 is correct and was verified against the installed
django-tables2 3.0.0 source, not assumed:

- `SingleTableMixin.get_table()` (views.py:111-120) runs `RequestConfig.configure()` before
  returning, and `configure()` is the only place `sort`/`page`/`per_page` are read, so reading
  `table.paginated_rows` afterwards is genuinely definitional.
- `paginated_rows` returns `self.page.object_list`, a `BoundRows` wrapping ONE sliced
  `QuerySet` object; `{% render_table %}` (`RenderTableNode.render`) does not re-configure or
  re-paginate a `Table` instance, and all five stock body templates iterate
  `table.paginated_rows` — so the view's iteration and the template's iteration share one
  `_result_cache`. The "costs no extra query" claim is true.
- `Accessor.resolve` short-circuits on `isinstance(context, dict) and self in context`, and
  `ALLOWED_FIELDS_FOR_NON_STAFF[0] == 'pk'`, so the anonymous `.values()` dict-row path
  resolves; the staff model-instance path falls through `TypeError` to `getattr`. Both branches
  are covered by the new page tests, which I ran green (23 tests, `manage.py test`).
- `RequestConfig`'s own `silent` handling (`PageNotAnInteger` → page 1, `EmptyPage` → last page)
  reassigns `table.page` *before* `get_table()` reads it, so the out-of-range/non-integer `page`
  tests are testing a real property, not a coincidence.
- The clamp itself resists the bypasses I tried: repeated `per_page` params (`QueryDict.get`
  and `RequestConfig`'s `GET[name]` both take the last value; `QueryDict.__setitem__` replaces
  the whole value list), `+1000`, `1_000`, whitespace-padded, `100.9`, `1e9`, `0x64`. There is
  no table `prefix`, so the hardcoded `'per_page'` matches `prefixed_per_page_field`.
- 37-10's accounting is sound. `unused_is_estimate = bool(contributing_codes)` is a strict
  improvement over `bool(estimate_codes)`; the per-RUN unknown count matches what the rows
  beneath actually read; `_apply_rollup_unused_fields()` is called on both the cache-hit and
  cache-miss paths; `_without_unused_fields()` resets the new key before `cache.set()`; and
  `tally_segments()`'s `.get(..., 0)` keeps per-run tallies inert. `campaign_list.html` renders
  only `rollup.nights_observed`, so there is no second roll-up surface that silently absorbs
  unknown runs as zero. `pre-commit run ruff` and `ruff-format` are clean on all changed files.

**Key concerns.** The security control 37-09 introduced is backwards on one branch and is
effectively untested; a committed notebook carries a silently-broken assertion that no
project gate can catch; and the roll-up's "not yet known" state now has two inconsistent
shapes.

---

## Critical Issues

### CR-01: the `per_page` cap clamps *below-range* values UP to the maximum, quadrupling the anonymous page cost it exists to bound

**File:** `solsys_code/campaign_views.py:160-170`
**Issue:**

```python
if per_page is not None and (per_page > MAX_TABLE_PER_PAGE or per_page < 1):
    mutable_get = request.GET.copy()
    mutable_get['per_page'] = str(MAX_TABLE_PER_PAGE)
```

A `per_page` that is *too small* (`0`, `-1`, `-9999`) is rewritten to `100` — the **largest**
page the view will serve. That is the opposite of what a cost cap should do, and it is a real
regression against pre-change behaviour, not a theoretical one:

- Before this change, `?per_page=0` reached `Table.paginate()`, whose first line is
  `per_page = per_page or self._meta.per_page` (django_tables2/tables.py:568, with
  `TableOptions.per_page` defaulting to 25 at tables.py:144). `0` is falsy, so the request
  rendered **25 rows**.
- After this change, `?per_page=0` renders **100 rows**.

The per-rendered-row tally cost is pinned by this codebase's own test at exactly 3 queries
(`test_page_query_count_grows_by_a_bounded_per_row_amount_not_unboundedly`,
`test_campaign_views.py:1044-1082`). So the single-character query string `?per_page=0` on a
public, unauthenticated, unthrottled page went from roughly 75 tally queries to roughly 300 —
a 4× amplification handed to anonymous clients by the very code whose module-level comment
(campaign_views.py:116-123) says it exists to stop "one GET [fanning] that cost out".

The `get()` docstring states the intent correctly — *"this is a ceiling, not a re-hardcoding of
the old 25-row default"* — and then the low branch does exactly the thing the sentence
disclaims. (The negative-value case is a genuine improvement: `per_page=-1` is truthy, so it
previously reached `Paginator`, whose `page(1)` computes `object_list[0:-1]` and hits Django's
"Negative indexing is not supported" — an unauthenticated 500. The fix for that is to clamp to
a *valid* value, which the fix below still does.)

**Fix:** clamp low values to the view's own default page size (the same constant the
`table_pagination` dict already carries), not to the maximum:

```python
MAX_TABLE_PER_PAGE = 100
DEFAULT_TABLE_PER_PAGE = 25  # D-11; table_pagination below reads this too

class CampaignRunTableView(SingleTableMixin, FilterView):
    table_pagination = {'per_page': DEFAULT_TABLE_PER_PAGE}  # D-11

    def get(self, request, *args, **kwargs):
        raw_per_page = request.GET.get('per_page')
        if raw_per_page is not None:
            try:
                per_page = int(raw_per_page)
            except (TypeError, ValueError):
                per_page = None
            if per_page is not None and not (1 <= per_page <= MAX_TABLE_PER_PAGE):
                clamped = MAX_TABLE_PER_PAGE if per_page > MAX_TABLE_PER_PAGE else DEFAULT_TABLE_PER_PAGE
                mutable_get = request.GET.copy()
                mutable_get['per_page'] = str(clamped)
                request.GET = mutable_get
        return super().get(request, *args, **kwargs)
```

and update the `get()` docstring, which currently documents the wrong behaviour for the low
branch. Add the `?per_page=0` / `?per_page=-1` cases to the tests (see WR-01).

---

## Warnings

### WR-01: `test_huge_per_page_is_capped_and_still_fully_covered` is vacuous — the cap has zero real test coverage

**File:** `solsys_code/tests/test_campaign_views.py:1211-1226`
**Issue:** The test fixtures **30** runs and then asserts `rendered_count <= MAX_TABLE_PER_PAGE`
(100). With `?per_page=100000` and only 30 rows in the campaign, all 30 render whether or not
the cap exists — `assertLessEqual(30, 100)` is true either way, and
`len(table.tallies) == rendered_count` and `_assert_full_coverage()` hold too. **Delete the
entire `get()` override and this test still passes.** The notebook's new G-37-5 cell has the
same hole: it prints `MAX_TABLE_PER_PAGE=100` in its narration but its largest request is
`?per_page=50` against 30 runs.

This is not an abstract complaint — it is why CR-01 shipped. No test exercises `per_page=0`,
`per_page=-1`, `per_page=101` (the first value actually above the cap), or `per_page=100`
(the boundary that must be honoured).

**Fix:** fixture more runs than the cap and assert the *exact* clamped page size, plus the
degenerate values:

```python
def test_huge_per_page_is_capped_at_the_maximum(self):
    for i in range(120):                       # > MAX_TABLE_PER_PAGE
        d = _BASE_DATE + timedelta(days=i)
        self._make_run(window_start=d, window_end=d)
    url = reverse('campaigns:table', kwargs={'pk': self.campaign.pk})
    response = self.client.get(url, {'per_page': '100000'})
    table = response.context['table']
    self.assertEqual(len(list(table.paginated_rows)), campaign_views.MAX_TABLE_PER_PAGE)
    self._assert_full_coverage(response)

def test_degenerate_per_page_falls_back_to_the_default_not_the_maximum(self):
    self._make_thirty_runs()
    url = reverse('campaigns:table', kwargs={'pk': self.campaign.pk})
    for value in ('0', '-1', '-9999'):
        with self.subTest(per_page=value):
            response = self.client.get(url, {'per_page': value})
            self.assertEqual(len(list(response.context['table'].paginated_rows)), 25)

def test_per_page_exactly_at_the_cap_is_honoured(self):
    ...  # 120 runs, ?per_page=100 -> exactly 100 rendered rows
```

### WR-02: broken multi-line `assert` in the committed notebook — two dead string literals, silently truncated message

**File:** `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` (code cell 42, the roll-up
cell)
**Issue:**

```python
assert (
    rollup['unused_is_estimate'] is False
), 'no run here carries a fetched proposal estimate yet -- correctly absent under D-20 '
'because no code contributed, not merely because none was collected (pre-G-37-6, it was '
'absent only for that weaker reason)'
```

The `assert` statement ends at the first string. The following two lines are a separate,
implicitly-concatenated **expression statement with no effect**. I confirmed the resulting
message by executing the exact fragment:

```
MESSAGE: 'no run here carries a fetched proposal estimate yet -- correctly absent under D-20 '
```

The half of the message that carries the actual G-37-6 reasoning never reaches anyone. Worse,
nothing in this repo can catch it: the ruff **lint** hook is declared `types_or: [ python, pyi ]`
(`.pre-commit-config.yaml`) so notebooks are never linted, only `ruff-format`ed — and I verified
that even `ruff check --select B,F,W,E` does not flag a bare string expression statement (no
`B018` hit). The notebook is a committed deliverable under the CLAUDE.md paired-docs rule, so
this is production doc content, not scratch.

**Fix:** make it one string, and re-execute the notebook:

```python
assert rollup['unused_is_estimate'] is False, (
    'no run here carries a fetched proposal estimate yet -- correctly absent under D-20 '
    'because no code contributed, not merely because none was collected (pre-G-37-6, it was '
    'absent only for that weaker reason)'
)
```

Consider also widening the lint hook to `types_or: [ python, pyi, jupyter ]` so the next one of
these is caught by the gate rather than by review.

### WR-03: the roll-up now has two different "nothing is known" shapes — `unused_is_estimate` disagrees with itself

**File:** `solsys_code/campaign_tally.py:586-591`, `:637-640`, `:645-655`, `:682-691`
**Issue:** 37-10 redefined `unused_is_estimate` as "an estimate contributed a number", but only
applied that redefinition to one of the two paths that produce a not-yet-known roll-up:

| state | `nights_unused` | `unused_known` | `unused_is_estimate` | `unused_unknown_runs` |
|---|---|---|---|---|
| empty campaign (`:586-591`) | `None` | `False` | **`True`** | `0` |
| runs present, nothing contributed (`:637-640`) | `None` | `False` | **`False`** | `N` |
| `campaign_rollup()` initial dict (`:682-691`) | `None` | `False` | **`True`** | `0` |
| `_without_unused_fields()` cached reset (`:645-655`) | `None` | `False` | **`True`** | `0` |

Three of the four write `True`; the one this plan touched writes `False`. The docstring at
`:606-608` still asserts the old uniformity — *"An empty `runs` list writes the same
not-yet-known defaults (`None`/`True`/`False`/`0`)"* — which is now false for any non-empty
campaign in the same logical state.

Not currently visible, because every renderer (`campaignrun_table.html:92`,
`render_progress()`, `_segment_summary_words()`) checks `known` first and short-circuits. But it
is exactly the kind of divergence this phase has repeatedly had to close: the moment any
consumer reads `is_estimate` without gating on `known`, two identical states render
differently, and `tally_segments()` hands out both keys side by side with no ordering contract
between them.

**Fix:** pick one and make all four agree. Given the new definition, the honest value when
nothing contributed is `False`; if the cached-reset convention (`True`, pinned by
`test_cached_value_never_carries_a_computed_unused_figure`) is the one to keep, then the
`:637-640` branch should write `True` when `unused_known` is `False`:

```python
if exact_known or estimate_known:
    rollup['nights_unused'] = exact_total + estimate_total
    rollup['unused_known'] = True
    rollup['unused_is_estimate'] = bool(contributing_codes)
else:
    rollup['nights_unused'] = None
    rollup['unused_known'] = False
    rollup['unused_is_estimate'] = True   # matches every other not-yet-known writer
rollup['unused_unknown_runs'] = unknown_runs
```

Either way, add a test asserting the two unknown states are key-for-key identical apart from
`unused_unknown_runs`, and fix the `:606-608` docstring.

### WR-04: stale key counts left in `campaign_tally.py` docstrings after the fourth `unused_*` key was added

**File:** `solsys_code/campaign_tally.py:663`, `:673`, `:746`
**Issue:** 37-10 diligently updated `_apply_rollup_unused_fields()` ("three" → "four") and
`_without_unused_fields()` ("three" → "four") but missed the two public entry points that
describe the same set:

- `:663` — `campaign_rollup()`: *"The three ``unused_*`` keys come from
  `_apply_rollup_unused_fields()`"* → now four.
- `:673` — `campaign_rollup()` Returns: *"the same eight tally keys plus ``runs``"* → the
  roll-up dict now has nine keys plus `runs`.
- `:746` — `get_or_compute_rollup()`: *"The three ``unused_*`` keys are never served from the
  cache"* → now four.

In a module where the docstrings are the stated contract (and where `tally_segments()`'s
Returns block *was* correctly updated), a wrong count is a live mis-statement, not cosmetics —
a reader counting keys off `:673` will not know `unused_unknown_runs` exists.

**Fix:** change "three" → "four" at `:663` and `:746`, and "eight tally keys" → "nine tally
keys" at `:673`. Note `:277`, `:317`, `:407` legitimately still say "three" (the per-run tally
never gained the key) — leave those.

### WR-05: `CampaignRunTable.__init__`'s `tallies` keyword argument is now dead, and two comments still describe it as the live mechanism

**File:** `solsys_code/campaign_tables.py:156-165` and `:346-349`; caused by
`solsys_code/campaign_views.py:275-280`
**Issue:** 37-09 replaced the constructor kwarg with post-construction attribute assignment
(`table.tallies = campaign_tally.tallies_for_runs(runs)`). A repo-wide grep for `tallies=`
returns exactly one hit — the parameter's own definition. No production code, no test, and no
template passes it any more.

Two consequences:

1. `__init__`'s docstring (`:157-164`) says `tallies` is *"pre-computed for the WHOLE table in
   one pass by the view"* — no view does that any more. `ApprovalQueueTable.Meta`'s comment
   (`:346-349`) says *"the three approval-queue construction sites in ApprovalQueueView never
   pass a `tallies` kwarg"*, which reads as if some other site does. None does.
2. It is a live trap: a future caller who passes `tallies=` to `CampaignRunTable` through
   `CampaignRunTableView` will have it silently overwritten by `get_table()` two lines later,
   with no error. Two mechanisms now exist for one piece of state, and only one of them works.

**Fix:** either drop the kwarg and set the default in `__init__` (`self.tallies = {}`), updating
both comments to say the view attaches it after `RequestConfig.configure()`; or keep the kwarg
and have `get_table()` pass it through `kwargs` instead of assigning the attribute — but
`get_table()` must build the table first to know the rendered rows, so dropping the kwarg is
the coherent option.

### WR-06: the cap hardcodes the literal `'per_page'` instead of django-tables2's `prefixed_per_page_field`

**File:** `solsys_code/campaign_views.py:160`
**Issue:** `RequestConfig.configure()` reads `getattr(table, 'prefixed_per_page_field')`
(config.py:51-55), which is `f"{table.prefix}{table.per_page_field}"`. The cap reads the bare
literal `'per_page'`. They agree today only because `CampaignRunTable.Meta` sets neither
`prefix` nor `per_page_field`. Add either (a `prefix` is the normal way to put a second table on
a page — `ApprovalQueueView` already renders three tables at once) and the cap silently stops
matching the parameter django-tables2 actually reads. Nothing fails; the ceiling just
disappears, and WR-01 means no test would notice.

**Fix:** resolve the name from the table class rather than restating it, e.g.

```python
per_page_field = self.table_class._meta.prefix + self.table_class._meta.per_page_field
raw_per_page = request.GET.get(per_page_field)
```

or, at minimum, add an assertion-style test pinning
`CampaignRunTable(data=[]).prefixed_per_page_field == 'per_page'` so a future `prefix` breaks
loudly.

### WR-07: `MAX_TABLE_PER_PAGE = 100` allows ~300 SQL queries per anonymous GET; the stated precedent does not carry the cost it is being used to justify

**File:** `solsys_code/campaign_views.py:116-123`
**Issue:** The comment justifies 100 as *"match[ing] the existing CampaignListView.paginate_by
bound below -- the precedent for what one anonymous page load may cost"*. But this codebase
already pins the per-row cost of *this* page at exactly 3 queries per rendered row
(`test_page_query_count_grows_by_a_bounded_per_row_amount_not_unboundedly`). 100 rows therefore
authorises roughly **300 queries per unauthenticated request**, against the ~75 that the
previous (accidental) 25-row tally bound allowed. The page has no throttle —
`_check_and_increment_throttle` is wired to the submission form only.

Borrowing a number from `CampaignListView` is a category error: that page's 100 items are
campaigns whose five record-derived keys are *fully cached*, whereas here every one of the 100
rows pays two uncacheable live unused-rule queries by D-15 design. "Same number" is not "same
cost".

This is a deliberate-tradeoff finding rather than a defect — but the tradeoff was made by
analogy rather than by arithmetic, and the arithmetic is already written down in the test
suite next door.

**Fix:** either re-derive the constant from the measured per-row cost and state the resulting
query budget in the comment, or lower it (50 keeps the documented `?per_page=50` use case and
halves the ceiling). If 100 is kept, say so explicitly: *"100 rows x 3 queries/row ~= 300
queries is the accepted anonymous ceiling."*

---

## Info

### IN-01: `get_table()`'s docstring calls the page queryset "already-evaluated" when the method's own iteration is what evaluates it

**File:** `solsys_code/campaign_views.py:260-262`
**Issue:** *"Iterating it costs no extra query: it wraps the ONE already-evaluated,
already-sliced page queryset, whose result cache is shared…"*. At the moment `get_table()` runs,
`page.object_list`'s underlying `QuerySet` is sliced but **not** evaluated — the set
comprehension on the next line is the first consumer and is what populates `_result_cache`; the
template's later iteration is the one that reuses it. The conclusion is right, the ordering in
the explanation is backwards, and this docstring is the only place the no-extra-query property
is recorded.
**Fix:** *"…it wraps the ONE sliced page queryset; this iteration evaluates it once and the
template's own `{% for row in table.paginated_rows %}` reuses that result cache."*

### IN-02: `request.GET = mutable_get` leaves `request.GET` mutable, and only on the clamped branch

**File:** `solsys_code/campaign_views.py:167-169`
**Issue:** `QueryDict.copy()` returns a *mutable* QueryDict. After the clamp fires,
`request.GET` is mutable for the remainder of the request (filterset, forms, `{% querystring %}`
tag, any middleware) — while an unclamped request keeps the normal immutable one. Nothing in
the current stack mutates it, but two different immutability contracts for the same attribute
depending on the query string is a surprise waiting to be found.
**Fix:** `mutable_get._mutable = False` before assigning, and say why in the docstring.

### IN-03: the notebook hand-duplicates the template's four-branch conditional with nothing tying them together

**File:** `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` (cell 42, `_segment_text`);
mirrors `src/templates/campaigns/campaignrun_table.html:92`
**Issue:** The helper's own comment says *"this helper must move when that template's
conditional moves"* — an acknowledged manual-sync obligation with no mechanism behind it. It
also emits the literal `≈` where the template emits `&approx;`, so its `assert _segment_text(...)
in collapsed_content` only works because this demo campaign happens to have no contributing
estimate (the comment says so). The one branch combining "at least" with the estimate sign —
the case WR-03 and the strip template are most fragile around — is exercised nowhere in the
notebook or the runbook.
**Fix:** add a small demo campaign to the notebook that *does* carry a fetched
`ProposalTimeAllocation` plus an unfetched code, and assert the entity spelling
(`at least &approx;N (M runs not yet known)`) against the rendered body directly, dropping the
`_segment_text` round-trip for that case.

### IN-04: the runbook does not describe the roll-up's "nothing known at all" state

**File:** `docs/runbooks/telescope_runs_calendar.rst:1992-2016`
**Issue:** The new prose covers the partially-known strip (`at least 2 (1 run not yet known)`)
and the estimate qualifier, but not the first template branch: when **no** run in the campaign
has a known figure, the strip reads a bare `not yet known` with *no* run count at all, even
though `unused_unknown_runs` is non-zero. An operator reading this section would reasonably
expect `at least 0 (3 runs not yet known)` and be confused by the plain form.
**Fix:** one sentence: *"When no run in the campaign has a figure yet, the strip reads simply
`not yet known` — there is no partial total to qualify."*

### IN-05: `_segment_summary_words()` and `render_progress()` accept any tally dict but ignore `unknown_runs`

**File:** `solsys_code/templatetags/calendar_display_extras.py:552-563`;
`solsys_code/campaign_tables.py:168-200`
**Issue:** Both take "a segment from `tally_segments()`" with no run/roll-up distinction, and
both fall straight from `known` to `is_estimate`, dropping `unknown_runs` on the floor. Today
neither is ever handed a roll-up, and `campaign_tally.py:440-444` documents the restriction —
but the restriction lives only in prose. Passing a roll-up to either would reproduce the exact
G-37-6 defect (a partial total rendered as if complete) one function over from where it was
just fixed.
**Fix:** cheapest guard is an explicit branch in both that raises or renders the partial form,
rather than silently ignoring a key that is always present in the dict they receive.

### IN-06: `campaign_rollup()`'s `unused_unknown_runs: 0` initialiser is unreachable

**File:** `solsys_code/campaign_tally.py:689`
**Issue:** Every exit from `campaign_rollup()` calls `_apply_rollup_unused_fields()`, which
unconditionally writes all four `unused_*` keys on both of its branches. The four initialisers
in the literal dict (`:686-689`) can never be observed. Harmless and arguably documentary, but
it is a fourth place a reader must check when asking "what is the default", and it is one of
the four rows in WR-03's disagreement table.
**Fix:** leave as-is if kept deliberately, but add a one-line comment saying the applier always
overwrites these — or drop them and let the applier own the keys outright.

---

_Reviewed: 2026-09-21T18:06:59Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep (incremental, `git diff 2c5906c..HEAD`)_
