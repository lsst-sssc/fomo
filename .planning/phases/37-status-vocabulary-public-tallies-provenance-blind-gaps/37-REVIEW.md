---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
reviewed: 2026-09-21T05:20:27Z
depth: deep
files_reviewed: 21
files_reviewed_list:
  - solsys_code/campaign_tally.py
  - solsys_code/campaign_views.py
  - solsys_code/campaign_tables.py
  - solsys_code/campaign_gap.py
  - solsys_code/status_vocabulary.py
  - solsys_code/proposal_allocation.py
  - solsys_code/observation_projector.py
  - solsys_code/admin.py
  - solsys_code/templatetags/calendar_display_extras.py
  - solsys_code/tests/test_campaign_tally.py
  - solsys_code/tests/test_campaign_views.py
  - solsys_code/tests/test_campaign_gap.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/tests/test_calendar_display_extras.py
  - solsys_code/tests/test_observation_projector.py
  - solsys_code/tests/test_proposal_allocation.py
  - solsys_code/tests/test_admin.py
  - src/templates/campaigns/campaign_list.html
  - src/templates/tom_calendar/partials/calendar.html
  - docs/runbooks/telescope_runs_calendar.rst
findings:
  critical: 2
  warning: 11
  info: 6
  total: 19
status: issues_found
---

# Phase 37: Code Review Report

**Reviewed:** 2026-09-21T05:20:27Z
**Depth:** deep
**Files Reviewed:** 21
**Status:** issues_found

## Summary

Scope: everything changed since `957417f` — the phase's review-fix round (CR-01/CR-02/CR-03,
WR-01..WR-11) plus plan 37-08's gap closure (G-37-4), which made the campaign roll-up's unused
figure compute live on every call.

Both quality gates are clean (`pre-commit run ruff` and `ruff-format` pass on every changed
Python file), and the mechanical half of G-37-4 holds up: `_rollup_runs()` genuinely is the one
run-fetch both roll-up paths use, `_without_unused_fields()` genuinely keeps a computed unused
figure out of the roll-up cache, and the new tests for the two time-driven drivers are real RED
→ GREEN tests, not tautologies.

Two defects nevertheless survive, both verified by executing the code rather than by reading it:

1. The public Progress column silently blanks out the moment a reader clicks a column header or
   passes `?per_page=`, because the page slice added for WR-04 mirrors django-tables2's page
   *number* but not its *ordering* or *page size*. Measured: `?sort=-telescope_instrument&page=2`
   renders 5 rows, **all 5** showing "Progress not available".
2. The campaign roll-up strip publishes a definite-looking `[U] ≈2` for a campaign in which one
   run's unused figure is genuinely unknown — silently counting "unknown" as zero, which is the
   one thing every docstring in `campaign_tally.py`/`proposal_allocation.py` says must never
   happen, and a roll-up-vs-cells disagreement on a single page response, which is exactly the
   invariant G-37-4 exists to guarantee.

Beyond those, the WR-07 path-traversal mitigation does not reject the value its own comment
cites (`..` passes the charset and `urljoin` still normalises it away), the WR-08 prune is
destructive in a way that is now unrecoverable through the admin, and the CR-01 cache-key
"link_version" watermark does not close the delete-then-create hole its docstring claims it
closes. Several of the new tests assert weaker properties than their names and docstrings
promise.

No structural-findings block was supplied with this review.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Progress cells blank out whenever the table is sorted or `?per_page=` is passed

**File:** `solsys_code/campaign_views.py:214-226` (`CampaignRunTableView.get_table_kwargs`)

**Issue:**
The WR-04 fix slices the tally pk set with

```python
per_page = self.table_pagination['per_page']
page = max(int(self.request.GET.get('page', 1)), 1)
page_pks = self.object_list.values_list('pk', flat=True)[(page - 1) * per_page : page * per_page]
```

It mirrors django-tables2's page *number* only. `RequestConfig.configure()` (verified against the
installed `django_tables2/config.py`) does **three** things after `get_table_kwargs()` has already
run:

- `order_by = request.GET.getlist(table.prefixed_order_by_field)` → `?sort=` re-orders the
  queryset via `TableQuerysetData.order_by()` (`django_tables2/data.py:179-204`), which calls
  `queryset.order_by(...)` and *replaces* `get_queryset()`'s `window_start desc nulls_last`
  ordering;
- `kwargs['per_page'] = int(request.GET['per_page'])` → `?per_page=` overrides the 25 this method
  hard-reads from `self.table_pagination`;
- `EmptyPage` → the rendered page is clamped to the last page, which the slice does not do.

So the 25 pks handed to `tallies_for_runs()` are computed from a *different ordering and a
different page size* than the rows django-tables2 actually renders. `render_progress()` then falls
into its `tally is None` branch (`campaign_tables.py:181-186`) and emits the muted "Progress not
available" token. The `get_table_kwargs` docstring's own claim — "Interactive column-header sorting
(RequestConfig) still works normally on top of this" — is false.

Measured against this working tree (30 approved runs, one campaign, anonymous client; the token is
emitted twice per cell, once in `title=` and once as text):

| request | rows rendered | rows showing "Progress not available" |
|---|---|---|
| (no params) | 25 | 0 |
| `?sort=-telescope_instrument` | 25 | **5** |
| `?sort=-telescope_instrument&page=2` | 5 | **5 (all of them)** |
| `?per_page=30` | 30 | **5** |

This silently destroys the phase's headline public deliverable (TALLY-01) for any reader who
clicks a sortable column header, and no existing test covers a sorted or `per_page`-overridden
request.

**Fix:** stop predicting the page and read it. Build the tallies from the rows django-tables2
actually paginated, after `RequestConfig` has run:

```python
def get_table_kwargs(self):
    # 'order_by': () still suppresses django-tables2's own default sort (D-04).
    return {'order_by': ()}

def get_table(self, **kwargs):
    table = super().get_table(**kwargs)  # RequestConfig has now applied sort/page/per_page
    page_pks = [
        pk
        for pk in (Accessor('pk').resolve(row, quiet=True) for row in table.page.object_list)
        if pk is not None
    ]
    runs = CampaignRun.objects.filter(pk__in=page_pks).select_related('site')
    table.tallies = campaign_tally.tallies_for_runs(runs)
    return table
```

(`Accessor` is already imported in `campaign_tables.py`; import it here too, or reuse the same
dict-vs-model resolution.) Add regression tests for `?sort=<col>`, `?sort=-<col>&page=2` and
`?per_page=<n>` asserting that **no** rendered row carries "Progress not available".

---

### CR-02: Campaign roll-up counts an unknown run's unused nights as zero and labels an exact total as an estimate

**File:** `solsys_code/campaign_tally.py:576-598` (`_apply_rollup_unused_fields`)

**Issue:**
For each run the applier either adds an exact count or records the run's `proposal_code` in
`estimate_codes`. A code whose `estimated_unused_nights()` returns `None` (never fetched, or a
semester with no stored rows) contributes **nothing** to `estimate_total`, but the final branch is

```python
if exact_known or estimate_known:
    rollup['nights_unused'] = exact_total + estimate_total
    rollup['unused_known'] = True
...
rollup['unused_is_estimate'] = bool(estimate_codes)
```

So a campaign mixing one run with real allocation events and one run whose figure is genuinely
unknown reports a definite number, with the unknown run silently treated as 0. Verified by
execution (campaign with run A = 2 elapsed `ALLOC:` nights, run B = `proposal_code`
`'NEVER-FETCHED-001'` with no `ProposalTimeAllocation` rows):

```
run A cell  : {'nights_unused': 2,    'unused_known': True,  'unused_is_estimate': False}
run B cell  : {'nights_unused': None, 'unused_known': False, 'unused_is_estimate': True}
rollup strip: {'nights_unused': 2,    'unused_known': True,  'unused_is_estimate': True}
```

Two separate faults in one page response:

1. **Unknown rendered as a number.** The strip renders `[U] ≈2` while the row directly beneath it
   renders `[U] not yet known`. That contradicts `_apply_unused_fields()`'s own docstring
   ("callers must render this as not-yet-known, never as zero"), `unused_hours_for()`'s identical
   contract, and D-15's "the strip and the rows agree by construction" — the very invariant G-37-4
   was written to establish. It also under-reports wasted telescope time, the number the whole
   feature exists to surface.
2. **Exact total mislabelled.** `unused_is_estimate` is set from the codes that were *attempted*,
   not the codes that *contributed*. Here every contributing figure (2) is exact, yet the strip
   prints the `≈` estimate marker.

The existing tests miss this: `test_d06_unknown_contract_survives_the_warm_path` uses a campaign
whose *only* run is unknown, so `exact_known` is False and the `else` branch rescues it.

**Fix:** distinguish "no figure attempted" from "a figure was attempted and came back unknown",
and derive the estimate flag from the contributing codes:

```python
    estimate_total = 0
    known_codes: set[str] = set()
    unknown_codes: set[str] = set()
    for code in estimate_codes:
        estimate = proposal_allocation.estimated_unused_nights(code)
        if estimate is None:
            unknown_codes.add(code)
        else:
            estimate_total += estimate
            known_codes.add(code)

    # A run with neither an exact count nor a resolvable estimate makes the WHOLE campaign
    # total unknown -- never silently zero (the same contract _apply_unused_fields() keeps
    # per run).
    if unknown_codes or blank_code_runs:
        rollup['nights_unused'] = None
        rollup['unused_known'] = False
        rollup['unused_is_estimate'] = True
        return

    rollup['nights_unused'] = exact_total + estimate_total
    rollup['unused_known'] = bool(exact_known or known_codes)
    rollup['unused_is_estimate'] = bool(known_codes)
```

where `blank_code_runs` is set in the first loop for a run with no allocation events and no
`proposal_code`. If a fully-unknown strip is judged worse UX than a partial one, add an explicit
`unused_partial` key and render `[U] ≥2` — but do not keep publishing `≈2` for a campaign whose
real total is unknown. Add a test for the mixed exact + unknown-code campaign above.

---

## Warnings

### WR-01: `_PROPOSAL_CODE_RE` admits the exact `..` value its own comment says it rejects

**File:** `solsys_code/proposal_allocation.py:80`, `:121-135`

**Issue:** The comment states the charset closes "a value containing `..`, `?` or `#` [which]
would redirect the authenticated portal request to a different endpoint when `urljoin()`
normalises it". The charset is `^[A-Za-z0-9._-]{1,100}$` — `.` is in it, so `..` and `.` both
match. `quote(code, safe='')` does not escape `.` (unreserved), and `urljoin` removes dot
segments per RFC 3986. Verified:

```python
>>> urljoin('https://observe.lco.global/', '/api/proposals/../')
'https://observe.lco.global/api/'
>>> urljoin('https://observe.lco.global/', '/api/proposals/./')
'https://observe.lco.global/api/proposals/'
```

So a `WatchedProposal.proposal_code` (admin-editable free text) or a classical schedule file's
`[..]` token still sends the credentialed, API-key-bearing GET to a different portal endpoint.
Practical blast radius is limited (one level up, same host, GET only, and the response fails the
`timeallocation_set` check), but the mitigation does not do what it claims and the test named for
it (WR-11 below) never exercises the admitted value.

**Fix:** reject dot-only segments explicitly, e.g.

```python
_PROPOSAL_CODE_RE = re.compile(r'^(?!\.+$)[A-Za-z0-9._-]{1,100}$')
```

and add `self.assertRaises(pa.PortalUnavailable)` cases for `'..'` and `'.'` alongside the
existing `'../requestgroups'` case.

---

### WR-02: `link_version` does not catch the delete-then-create pair its docstring says it catches

**File:** `solsys_code/campaign_tally.py:99-110`, `:145`

**Issue:** `build_tally_cache_key()`'s docstring states `link_version` "additionally catches a
delete-then-create pair that leaves the count unchanged (e.g. re-attributing one record to a
different, same-count link set)". `link_version` is `Max('observation_record_id')`, so it only
moves when the *maximum* linked record id changes. Counter-example: a run linked to records
`{5, 9}`; staff re-attribute record 5 away and attach record 7 instead. Afterwards:
`records = 2` (unchanged), `link_version = 9` (unchanged), and `records_version =
Max(modified)` is still record 9's stamp because neither `ObservationRecord` row was saved.
Identical cache key → the stale five-key tally is served for up to `TALLY_CACHE_TTL_SECONDS`
(one hour), which is precisely the hole CR-01 set out to close.

The two new tests only cover the cases where the max *does* move
(`test_link_version_is_the_newest_linked_record_id`,
`test_removing_a_non_newest_link_is_reflected_with_no_clock_advance` — the latter changes the
count).

**Fix:** use an order-insensitive watermark that changes for any membership change, e.g.
`link_version=Sum('observation_record_id')` (cheap, and combined with `records` it distinguishes
essentially every real swap), or `link_version=Max('observation_record_id')` plus
`link_checksum=Sum('observation_record_id')` folded into the key. Either way, correct the
docstring so it does not promise coverage the key does not have, and add the
`{5, 9} → {7, 9}` swap test.

---

### WR-03: `CampaignListView` pays an unbounded, per-run live unused recomputation for a value the template never renders

**File:** `solsys_code/campaign_views.py:279-292`, `:317-319`; `src/templates/campaigns/campaign_list.html:48-50`

**Issue:** `get_context_data()` calls `campaign_tally.get_or_compute_rollup(campaign)` for every
listed campaign. Since G-37-4, that call is never fully cached: `_apply_rollup_unused_fields()`
re-runs live on every hit, issuing one `allocation_events()` query per run plus one
`estimated_unused_nights()` query per distinct proposal code, plus a `_rollup_runs()` fetch and a
`campaign_records_version()` probe per campaign.

The template renders only `campaign.run_count` (a queryset annotation) and
`campaign.rollup.nights_observed` (a fully cached key). **The three `unused_*` keys — the only
part that is recomputed live — are never displayed on this page at all.** The entire marginal
cost is dead work.

The code comment's justification is also factually wrong: "Bounding the page to 100 campaigns
bounds both costs per request" bounds the *campaign* count, not the *run* count. Measured on a
fully warm cache, anonymous client:

```
WARM campaign-list queries for 3 campaigns (1 + 5 + 20 = 26 runs): 38
WARM campaign-list queries after adding a 50-run campaign:         90
```

i.e. +52 queries for one added campaign — one SQL query per run, on every anonymous GET, forever.
At `paginate_by = 100` with campaigns averaging 20 runs that is ~2 000 queries per unauthenticated
page load, which is a request-amplification/availability risk on a public endpoint.

**Fix:** do not compute the live unused split on a page that does not show it. Either give
`CampaignListView` a cheap cached-only entry point:

```python
# campaign_tally.py
def get_or_compute_rollup(campaign, records_version=None, *, live_unused: bool = True):
    ...
    if cached is not None:
        rollup = dict(cached)
        if live_unused:
            _apply_rollup_unused_fields(rollup, _rollup_runs(campaign))
        return rollup
```

and call it with `live_unused=False` from the list view, or (cleaner) replace the list view's call
with a dedicated `campaign_observed_nights(campaign)` helper that reads only the cached
record-derived half. Keep the live path for `CampaignRunTableView`, which does render the figure.

---

### WR-04: The test that claims to bound the campaign-list query cost cannot detect growth with run count

**File:** `solsys_code/tests/test_campaign_views.py` — `TestCampaignRollup.test_campaign_list_query_count_bound_with_three_campaigns`

**Issue:** The test pins `MARGINAL_QUERIES_PER_CAMPAIGN = 3` and its docstring claims a second
added campaign "proves the cost stays CONSTANT per campaign rather than growing". Every fixture
campaign has exactly **one** run (`self._make_run(campaign=c, ...)` once per campaign), so the
constant it measures is `2 + n_runs` evaluated at `n_runs == 1`. The regression WR-03 describes —
per-run query fan-out — is invisible to it, and the `assertEqual` gives false confidence that it
is pinned.

**Fix:** parameterise the fixture over run count and assert the shape, not a single point:

```python
for n_runs in (1, 4):
    extra = TargetList.objects.create(name=f'Bound Extra {n_runs}')
    for i in range(n_runs):
        self._make_run(campaign=extra, telescope_instrument=f'FTN/x{n_runs}-{i}')
    campaign_tally.get_or_compute_rollup(extra)
    with CaptureQueriesContext(connection) as ctx:
        self.client.get(reverse('campaigns:list'))
    marginal[n_runs] = len(ctx.captured_queries) - previous
self.assertEqual(marginal[4], marginal[1])  # constant per campaign, independent of run count
```

---

### WR-05: `store_proposal_allocations()`'s prune can wipe a proposal's stored allocations, with no admin recovery path

**File:** `solsys_code/proposal_allocation.py:201-204`; `solsys_code/admin.py:519-523`

**Issue:** The new prune deletes every stored row for `proposal_code` whose
`(semester, instrument_type, allocation_type)` key was not in this response. When `rows == []`
(portal returns a syntactically valid body with an empty `timeallocation_set` — an outage state
that `fetch_proposal_allocations()` accepts, since only a non-list fails the check), `seen_keys`
is empty and **all** rows for that proposal are deleted. The test
`test_prune_never_touches_a_different_proposal_codes_rows` encodes this behaviour deliberately.

Consequences: the public unused estimate flips from a number to "not yet known" for every run
carrying that code, and — because WR-10 in the same round added
`has_add_permission() == False` and `has_delete_permission() == False` to
`ProposalTimeAllocationAdmin` — there is no longer any manual way for staff to restore or even
inspect-and-correct the table. Recovery depends entirely on the next successful unattended fetch.

**Fix:** make the prune conditional on having written something, and guard the destructive case:

```python
    if not seen_keys:
        # An empty timeallocation_set is indistinguishable from a partial/degraded portal
        # response -- never let it silently destroy a proposal's stored allocations.
        logger.warning('store_proposal_allocations: empty response for %r; keeping stored rows.', proposal_code)
        return written
    stale = ProposalTimeAllocation.objects.filter(proposal_code=proposal_code)
    ...
```

If a genuinely-empty allocation set must be representable, add an explicit
`--prune-empty` path on the management command rather than making it the default of the
unattended runner.

---

### WR-06: `unused_hours_for()`'s new `semester` parameter has no production caller, and the estimate is still not scoped to the run

**File:** `solsys_code/proposal_allocation.py:208-240`, `:257`

**Issue:** WR-08 widened `unused_hours_for()` with a `semester` argument, but the only production
caller, `estimated_unused_nights()`, never passes it (`unused_hours_for(proposal_code)`), and
`campaign_tally` calls `estimated_unused_nights(code)` only. Grep confirms the parameter is
exercised exclusively by `test_explicit_semester_overrides_the_most_recent_default` and
`test_no_rows_in_an_explicitly_requested_semester_is_none` — it is test-only API surface.

The substantive gap is that the default (`max(semester)`) is not the semester the *run* belongs
to. A 2026A run whose proposal also carries 2026B allocations now reports 2026B's unused hours as
"this run's wasted time". The docstring's "currently-relevant wasted time" claim only holds for
runs in the newest stored semester.

**Fix:** either plumb the run's own semester through —

```python
def estimated_unused_nights(proposal_code: str, semester: str | None = None) -> int | None:
    unused_hours = unused_hours_for(proposal_code, semester=semester)
```

with `campaign_tally._apply_unused_fields()` deriving the semester from `run.window_start` — or
drop the unused parameter and state plainly in the docstring that the estimate is
proposal-latest-semester-wide, not run-scoped, so no reader mistakes it for the latter.

---

### WR-07: Cache-write asymmetry — the per-run paths rely on backend serialisation to keep `unused_*` out of the cache

**File:** `solsys_code/campaign_tally.py:305-309`, `:346-350` vs `:601-611`, `:743`

**Issue:** The roll-up path defensively caches an explicit copy
(`cache.set(key, _without_unused_fields(rollup), ...)`). The two per-run paths instead do

```python
cache.set(key, tally, timeout=TALLY_CACHE_TTL_SECONDS)  # cached WITHOUT unused_* fields
_apply_unused_fields(tally, run)                        # mutates the same dict afterwards
```

and depend on the backend copying/pickling at `set()` time. That holds for `LocMemCache`,
`DatabaseCache`, memcached and redis, so it is not a live bug today — but the invariant "no
computed unused figure is ever written to the cache" is enforced by the backend, not by this
module, and a future in-process by-reference cache would silently break the D-15 guarantee with
no test failing (the existing
`test_cached_value_never_carries_a_computed_unused_figure` exists only for the roll-up).

**Fix:** use the same explicit copy on both paths, and reuse it for the per-run shape:

```python
cache.set(key, _without_unused_fields(tally), timeout=TALLY_CACHE_TTL_SECONDS)
_apply_unused_fields(tally, run)
```

and add the per-run twin of `test_cached_value_never_carries_a_computed_unused_figure`.

---

### WR-08: CLAUDE.md paired-docs rule — `campaign_views.py` and `observation_projector.py` changed behaviour with no notebook regeneration

**File:** `solsys_code/campaign_views.py`, `solsys_code/observation_projector.py` (no
`docs/notebooks/pre_executed/**` change in `957417f..HEAD`)

**Issue:** CLAUDE.md maps `campaign_views.py` → `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`
(the v2.2 campaign-surface collective mapping) and `observation_projector.py` →
`docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`, and states that a change to
those modules' *behaviour* must include the paired notebook in scope "up front, not as a
follow-up". This round changed `CampaignListView` behaviour (added `order_by('name')` and
`paginate_by = 100`), changed `CampaignRunTableView.get_table_kwargs()` behaviour (page slicing),
and added a new public `observation_projector.facility_for_or_none()`. The last notebook
regeneration is `df40929` ("docs(37-07): regenerate the four pre-executed notebooks"), which is an
ancestor of the review base — so the notebooks predate all of it.

`campaign_lifecycle_demo.ipynb` cells 41-44 demonstrate the roll-up and per-run tally; nothing in
them exercises the live-on-cache-hit property that is this round's entire point, nor the new
campaign-list pagination. CLAUDE.md's breach history (Phase 5, Phase 6, quick task `260726-kdp`,
Phase 35 NF-24) is explicitly about exactly this omission.

**Fix:** add a cell to `campaign_lifecycle_demo.ipynb` that warms the roll-up cache, makes a staff
`run_status` edit that touches no `ObservationRecord`, and shows the strip and the row cells
agreeing on the second call; add a cell to `project_observation_calendar_demo.ipynb` showing
`facility_for_or_none()` returning `None` for a stale facility name. Regenerate both with
`jupyter nbconvert --to notebook --execute --inplace` and commit with output.

---

### WR-09: Runbook lists the campaign-list badge as a surface that shows the unused figure; it does not

**File:** `docs/runbooks/telescope_runs_calendar.rst:2001-2022`; `src/templates/campaigns/campaign_list.html:48-50`

**Issue:** The updated runbook text reads: "The unused figure is recomputed **on every page
load**, on every surface that shows it -- the run row's Progress cell, the campaign roll-up strip
above the runs table, **the campaign-list badge**, and the calendar pop-up's attributed-run
block." The campaign-list badge renders only
`{{ campaign.run_count }} run(s)` and, conditionally, `{{ campaign.rollup.nights_observed }}
night(s) observed`. It never renders `nights_unused`. An operator reading this will look for a
number that is not there — and the sentence is also the stated justification for the wasted cost
in WR-03.

**Fix:** drop "the campaign-list badge" from that list, or add the unused segment to the badge if
it was meant to be there. Whichever is chosen, keep the runbook and the template in agreement.

---

### WR-10: The anonymous campaign list still leaks pending-review run counts, the same leak WR-06 hardened the calendar against

**File:** `solsys_code/campaign_views.py:269-273` (`run_count = Count('campaign_runs')`); `src/templates/campaigns/campaign_list.html:49`

**Issue:** WR-06 in this same round added
`if run is None or not run.is_publicly_visible: return None` to `unused_night_decoration()`, with
the stated rationale "leaking the existence of an unreviewed run". Two lines of the same page's
own view still expose exactly that: `run_count` is an unfiltered `Count('campaign_runs')` and is
rendered in the badge to anonymous visitors, while `campaign.rollup['runs']` (computed on the same
request) excludes `PENDING_REVIEW`. A visitor who submits a run can watch the badge increment to
confirm it landed, and staff-side pending volume is publicly countable.

The annotation predates this phase, but the inconsistency is now internal to a single request and
directly contradicts the rationale written into this round's own fix.

**Fix:** filter the annotation to publicly visible runs, matching `_rollup_runs()`'s discipline:

```python
.annotate(
    run_count=Count(
        'campaign_runs',
        filter=~Q(campaign_runs__approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW),
    )
)
```

(or simply render `campaign.rollup.runs`, which is already correct, and drop the annotation).

---

### WR-11: `test_path_traversal_like_code_is_rejected_before_any_request` does not test a value the regex admits

**File:** `solsys_code/tests/test_proposal_allocation.py` — `FetchProposalAllocationsTests.test_path_traversal_like_code_is_rejected_before_any_request`

**Issue:** The test asserts `'../requestgroups'` is rejected. That value is rejected by the `/`
character, not by any handling of `..`. The values that actually traverse — `'..'` and `'.'` —
both pass `_PROPOSAL_CODE_RE` (see WR-01) and are not tested. The test therefore proves nothing
about the property its name and docstring claim.

**Fix:** add the two admitted values to the test, which will fail until WR-01 is fixed:

```python
    def test_bare_dot_segments_are_rejected(self):
        for code in ('..', '.'):
            with self.subTest(code=code), patch('solsys_code.proposal_allocation.make_request') as mock_request:
                with self.assertRaises(pa.PortalUnavailable):
                    pa.fetch_proposal_allocations(code, _mock_facility())
                mock_request.assert_not_called()
```

---

## Info

### IN-01: `_rollup_runs()` requests a column nothing reads

**File:** `solsys_code/campaign_tally.py:535`

`.only('pk', 'proposal_code', 'run_status', 'site_id', 'site__timezone', 'site__obscode')` — the
docstring justifies `run_status` and `site__timezone` by naming the exact readers, but nothing on
the roll-up path reads `site.obscode` (`night_counts_for_run()` reads only `site.timezone`;
`unused_nights_for_run()` reads only `pk` and `run_status`). Drop `site__obscode` or name its
reader, so the `.only()` list stays the audit trail its docstring claims it is.

### IN-02: `claimed_site_unknown_count` now includes non-claiming records

**File:** `solsys_code/campaign_gap.py:220-226`

The `facility is None → site_unknown_count += 1; continue` guard runs *before* the
`classify_record(...) not in _CLAIMING_DISPLAY_STATES` filter, so a QUEUED or otherwise
non-claiming record with a stale facility name now inflates the "N observation(s) on this
campaign's calendar could not be assigned to a site" banner
(`campaignrun_gap_analysis.html:66-72`), which previously counted only records that would
otherwise have claimed a night. Defensible (an unclassifiable record genuinely is unknown), but
the banner copy no longer quite matches what is counted.

### IN-03: `unused_night_decoration()` pays a query before the cheap namespace check

**File:** `solsys_code/templatetags/calendar_display_extras.py:674-679`

The tag fetches `event.telescope_label_meta` (one query per event) before testing
`(event.url or '').startswith(ALLOC_URL_NAMESPACE)`, so every non-allocation event on every
calendar cell pays a companion-row lookup for nothing. Reordering the two guards is free.
Pre-existing (Plan 06), not introduced here.

### IN-04: `link_counts_for_runs()` indexes the result dict without a guard

**File:** `solsys_code/campaign_tally.py:148-160`

`result[row['run_id']][...] = ...` assumes the `values(run_id=F('observation_records__campaign_run_links__run_id'))`
re-traversal reuses the join the `.filter()` set up. It does in the installed Django, but a
join-reuse surprise here is a `KeyError`/500 on the anonymous campaign table rather than a
degraded count. `result.setdefault(row['run_id'], {...})` or a `if row['run_id'] in result` guard
costs nothing.

### IN-05: `facility_for_or_none()` catches only `ImportError`

**File:** `solsys_code/observation_projector.py:104-107`

`facility_for()` also *instantiates* the service class (`get_service_class(name)()`,
`observation_projector.py:82`). A facility whose `__init__` raises (bad settings, missing API key
object) still escapes the "never raises" promise into the anonymous campaign table. Widen to
`except (ImportError, Exception)`-with-logging, or at minimum document that instantiation errors
are out of scope.

### IN-06: Campaign-list pagination links drop other query parameters

**File:** `src/templates/campaigns/campaign_list.html:58`, `:66`

`href="?page={{ ... }}"` discards any other querystring. Harmless today (the list view takes no
other params), but it will silently break the first time a filter or search is added. Prefer a
`querystring` template tag.

---

_Reviewed: 2026-09-21T05:20:27Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
