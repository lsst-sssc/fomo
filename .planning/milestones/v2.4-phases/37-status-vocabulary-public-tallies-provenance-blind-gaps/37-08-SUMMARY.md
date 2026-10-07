---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
plan: 08
subsystem: tally
tags: [campaign-tally, caching, django, gap-closure]

# Dependency graph
requires:
  - phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
    provides: "37-04's per-run live-unused-split contract (CR-02/get_or_compute_tally), 37-05/37-07's campaign roll-up (campaign_rollup/get_or_compute_rollup) that this plan restructures"
provides:
  - "campaign_tally.py: _rollup_runs()/_apply_rollup_unused_fields()/_without_unused_fields() -- one shared route to the campaign roll-up's unused figure, live on both the cache-hit and cache-miss paths"
  - "get_or_compute_rollup() never serves a computed unused figure from the cache -- closes G-37-4"
  - "Re-pinned, relaxed-but-real query-cost bounds for both the runs-page per-row cost and the anonymous campaign-list per-campaign cost, the latter now also proving the cost is CONSTANT per campaign"
  - "Corrected freshness documentation: two campaign_tally.py docstrings, the WR-05 comment in campaign_views.py, and the runbook's How fresh is the tally? paragraph"
affects: [gap-closure re-verification (reads gap_ids: [G-37-4]), any future phase touching campaign_tally.py's roll-up helpers]

# Actuals (#2632)
actuals:
  tokens: 10500    # chars/4 over the 5 changed files (42002 chars diff 3c34e2e..HEAD)
  tasks: 3
  commits: 4

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Campaign-level cache-or-compute split: cache only the record-derived half of a dict, recompute a live-only half on every call (cache hit included), mirroring the per-run get_or_compute_tally()/tallies_for_runs() contract at the campaign level"

key-files:
  created: []
  modified:
    - solsys_code/campaign_tally.py
    - solsys_code/campaign_views.py
    - solsys_code/tests/test_campaign_tally.py
    - solsys_code/tests/test_campaign_views.py
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "Took the developer's chosen option (b) from 37-UAT.md: extend the per-run live-unused-split contract (CR-02) to the campaign roll-up, rather than shortening the cache TTL or removing caching from the roll-up entirely."
  - "Accepted a deliberate, documented double allocation-event lookup on the cache-miss path (once inside tallies_for_runs()'s own per-run split, once inside the new campaign-level applier) rather than passing per-run tallies into the applier as a shortcut -- the shortcut would reintroduce the two-sources-for-one-figure structure the gap is made of."
  - "Query-count regressions that are direct, provable consequences of the fix (test_only_does_not_trigger_deferred_field_queries 5->6, the runs-page per-row bound 2->3, and the campaign-list per-campaign bound) were re-measured and re-pinned as exact constants rather than loosened to inequalities, per the plan's own must-haves."

patterns-established:
  - "A campaign-level '_apply_*_unused_fields(dict, runs)' applier as the single route to a derived figure, called identically from both the cache-hit and cache-miss branches of its cache wrapper -- extends the shape already established for the per-run tally to the campaign roll-up."

requirements-completed: [TALLY-01, TALLY-02, UNUSED-01]

coverage:
  - id: D1
    description: "The roll-up's unused figure (nights_unused/unused_known/unused_is_estimate) is computed live on every call to get_or_compute_rollup(), cache hit included, through one shared campaign-level applier -- closing G-37-4."
    requirement: "TALLY-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRollup.test_rollup_strip_agrees_with_progress_cells_after_a_staff_status_edit_not_a_records_change"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestGetOrComputeRollupFreshness (9 cases)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Nothing computed is ever cached: cache.set() for the roll-up always carries the not-yet-known defaults for the three unused_* keys."
    requirement: "TALLY-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestGetOrComputeRollupFreshness.test_cached_value_never_carries_a_computed_unused_figure"
        status: pass
    human_judgment: false
  - id: D3
    description: "D-10's counting rule (each run's exact still-standing nights added directly; the D-06 proposal estimate added once per distinct proposal code) and D-06's not-yet-known-never-zero contract both survive the refactor unchanged, including on the warm path."
    requirement: "TALLY-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestCampaignRollup (existing, values untouched) + TestGetOrComputeRollupFreshness (warm-path re-assertion)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Both moved query-count bounds (the runs-page per-row cost, the anonymous campaign-list per-campaign cost) are re-measured and re-pinned as exact, enumerated constants; the campaign-list bound additionally proves the marginal cost is constant rather than growing."
    requirement: "TALLY-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunTableProgressColumn.test_page_query_count_grows_by_a_bounded_per_row_amount_not_unboundedly"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRollup.test_campaign_list_query_count_bound_with_three_campaigns"
        status: pass
    human_judgment: false
  - id: D5
    description: "Freshness documentation (two campaign_tally.py docstrings, the WR-05 comment above CampaignListView.paginate_by, and the runbook's How fresh is the tally? paragraph) accurately describes what is now live versus what the cache still bounds."
    verification:
      - kind: other
        ref: "grep-based paragraph-scoped verify gates in 37-08-PLAN.md Task 3 + pre-commit sphinx-build"
        status: pass
    human_judgment: false

duration: 75min
completed: 2026-09-21
status: complete
---

# Phase 37 Plan 08: Live-Compute the Campaign Roll-up's Unused Figure Summary

**Closed G-37-4 by giving `campaign_rollup()`/`get_or_compute_rollup()` the same live-unused split the per-run tally (`get_or_compute_tally()`) already had, through one new shared campaign-level applier -- no more up-to-an-hour disagreement between the roll-up strip and the Progress cells beneath it.**

## Performance

- **Duration:** 75 min
- **Started:** 2026-09-21T03:31:00Z (approx.)
- **Completed:** 2026-09-21T04:46:32Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- `_rollup_runs()`, `_apply_rollup_unused_fields()` and `_without_unused_fields()` give the campaign roll-up exactly one route to its unused figure, shared by both `campaign_rollup()` (cache-miss) and `get_or_compute_rollup()` (cache-hit) -- proven end-to-end on a real rendered `campaigns:table` response across a staff `run_status` edit that moves neither the clock nor `campaign_records_version()`.
- Nine new unit tests (`TestGetOrComputeRollupFreshness`) pin both blind-spot drivers (the clock, a staff status edit) on a cache hit, the "nothing computed is ever cached" contract read directly off `cache.get()`, cold/warm dict equality, D-10's once-per-distinct-proposal-code rule, D-06's not-yet-known-never-zero contract, the pending-review exclusion, the empty-campaign shape, and an enumerated 3-query warm-path cost.
- Both query-count regressions the fix directly causes were re-measured and re-pinned as exact constants rather than loosened: the roll-up's own `assertNumQueries` (5 -> 6), the runs-page per-added-row bound (2 -> 3), and the anonymous campaign-list per-campaign bound (relaxed to a named `MARGINAL_QUERIES_PER_CAMPAIGN = 3` constant, now additionally proving the cost is *constant* per campaign via a second added campaign).
- Two `campaign_tally.py` docstrings, the WR-05 comment above `CampaignListView.paginate_by`, and the runbook's `How fresh is the tally?` paragraph were all corrected to state what is now live (the unused split, everywhere) versus what the cache still bounds (the five record-derived keys plus run count, keyed by `campaign_records_version()`).

## Task Commits

Each task was committed atomically (Task 1 followed the RED -> GREEN TDD cycle as a `type="tracer"` task):

1. **Task 1 RED: failing end-to-end test for the roll-up cache-hit agreement** - `e8b7219` (test)
2. **Task 1 GREEN: one live route to the roll-up's unused figure** - `55349cf` (feat)
3. **Task 2: unit coverage for both blind-spot drivers, the cache contract, and the moved query cost** - `2271865` (test)
4. **Task 3: re-pin both moved query bounds and correct the freshness docs** - `9886c0c` (fix)

_Note: Task 1 is `type="tracer" tdd="true"` — RED and GREEN landed as separate commits per the TDD commit-scope contract; no REFACTOR commit was needed (the GREEN implementation needed no follow-up cleanup)._

## Files Created/Modified

- `solsys_code/campaign_tally.py` - New `_rollup_runs()`/`_apply_rollup_unused_fields()`/`_without_unused_fields()` helpers; `campaign_rollup()` and `get_or_compute_rollup()` rewritten to route through them; both functions' docstrings corrected
- `solsys_code/campaign_views.py` - WR-05 comment above `CampaignListView.paginate_by` corrected to describe the live per-load cost (comment-only change)
- `solsys_code/tests/test_campaign_tally.py` - New `TestGetOrComputeRollupFreshness` (9 cases); retargeted `inspect.getsource(...)` subject and the one moved `assertNumQueries` constant in two pre-existing tests
- `solsys_code/tests/test_campaign_views.py` - New end-to-end `TestCampaignRollup` case; two moved query-count constants re-pinned with the campaign-list one gaining a constant-marginal-cost assertion
- `docs/runbooks/telescope_runs_calendar.rst` - `How fresh is the tally?` paragraph rewritten to describe the corrected caching contract

## Decisions Made

- Took the developer's chosen option (b) from `37-UAT.md`: extend the per-run live-unused-split contract to the campaign roll-up, rather than shortening the cache TTL or dropping caching from the roll-up entirely.
- Accepted a deliberate, documented double allocation-event lookup on the cache-miss path (once inside `tallies_for_runs()`'s own per-run split, once inside the new campaign-level applier) rather than reusing the per-run tallies as a shortcut -- the shortcut would reintroduce the two-sources-for-one-figure structure the gap is made of. Recorded in an in-code comment at the call site.
- Query-count regressions that are direct, provable consequences of the fix (`test_only_does_not_trigger_deferred_field_queries` 5->6, the runs-page per-row bound 2->3, and the campaign-list per-campaign bound) were re-measured and re-pinned as exact constants, per the plan's own must-haves -- never loosened to inequalities.

## Deviations from Plan

### Auto-fixed Issues

None in the Rule 1-3 sense -- no bugs, missing critical functionality, or blocking issues were discovered outside what the plan itself specified.

### Noted, Not Auto-fixed (documented, out of scope)

**1. Pre-existing flaky Playwright test observed during the full-suite verify gate**
- **Found during:** Task 3's `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` verification
- **Issue:** `solsys_code.tests.test_bootstrap5_rendering.TestBootstrap5Rendering.test_observatory_create_form_submits_to_observatory_url` failed once (`AssertionError` on a Playwright `self.page.url` check) in a 1746-test run
- **Why not fixed:** entirely outside this plan's `files_modified` and imports none of the changed modules -- out of scope per the executor's scope-boundary rule. Confirmed transient: passes in isolation, passes when run together with both files this plan changed (154 tests, OK), and an immediate identical full-suite retry completed `OK (skipped=1)` with no failure
- **Action taken:** logged in `.planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/deferred-items.md` and in the cross-phase `WINDOWS.md` ledger (kind: deviation) rather than fixed
- **Files modified:** none (documentation only)

---

**Total deviations:** 0 auto-fixed; 1 pre-existing issue documented and deferred (out of scope).
**Impact on plan:** None. The plan's own diff-scope gate (`git diff --name-only 1b7e68f -- solsys_code/ docs/ src/`) confirms exactly the five `files_modified` files changed.

## Issues Encountered

None beyond the flaky-test observation documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

G-37-4 is closed: the campaign roll-up's unused figure now agrees with its rows by construction, on every surface, at every instant. `.planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-UAT.md` and `37-VERIFICATION.md` were intentionally NOT edited by this plan (per its own prohibition) -- re-verification reads this plan's `gap_ids: [G-37-4]` to reconcile the gap. No blockers for Phase 37 re-verification or milestone completion.

---
*Phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps*
*Completed: 2026-09-21*

## Self-Check: PASSED
