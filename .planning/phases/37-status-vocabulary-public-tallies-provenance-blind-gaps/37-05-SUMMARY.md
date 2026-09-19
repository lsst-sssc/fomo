---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
plan: 05
subsystem: public-tallies
tags: [django, django-tables2, ttl-cache, tdd]

requires:
  - phase: 37-04
    provides: "solsys_code/campaign_tally.py -- tallies_for_runs()/tally_segments()/campaign_rollup()/campaign_records_version(), the module this plan renders rather than recomputes"
  - phase: 37-01
    provides: "solsys_code/status_vocabulary.py -- MARKER/LABEL/DisplayState, the shared marker vocabulary the Progress cell and roll-up strip reuse"
provides:
  - "CampaignRunTable's public Progress column (computed for the whole table in one pass, never a per-row query) and the optional tallies constructor keyword"
  - "campaign_tally.get_or_compute_rollup()/build_rollup_cache_key() -- the TTL-cached campaign roll-up wrapper 37-06/37-07 can also read"
  - "A roll-up summary strip on the campaign runs page and an observed-nights clause on the campaign list badge"
affects: [37-06, 37-07]

actuals:
  tokens: 8250
  tasks: 2
  commits: 2
  plan_head_before: 00afda5b4160d7dbde727d17a3c0b4426ab01f5c

tech-stack:
  added: []
  patterns:
    - "Table-level bulk pre-computation joined by pk in Python at render time (CampaignRunTable.render_progress() reads an optional tallies dict the view built with tallies_for_runs() -- never a per-row query), mirroring render_run_status()'s existing Accessor-on-dict-row pattern"
    - "Cache-or-compute wrapper mirrored a second time (get_or_compute_rollup()/build_rollup_cache_key() copy get_or_compute_tally()'s exact shape) so the campaign roll-up inherits the same records-version freshness rule the per-run tally already has"

key-files:
  created: []
  modified:
    - solsys_code/campaign_tables.py
    - solsys_code/campaign_views.py
    - solsys_code/campaign_tally.py
    - src/templates/campaigns/campaignrun_table.html
    - src/templates/campaigns/campaign_list.html
    - solsys_code/tests/test_campaign_views.py

key-decisions:
  - "CampaignRunTableView.get_table_kwargs() builds a fresh CampaignRun.objects.filter(pk__in=pks).select_related('site') queryset from self.object_list's pks rather than passing self.object_list itself to tallies_for_runs() -- the non-staff branch's object_list is a .values() dict queryset, and campaign_tally.night_counts_for_run() needs real model attributes (run.site.timezone, run.proposal_code, run.run_status), not dict keys. select_related('site') keeps a cache-miss row's timezone read from costing a second query per row."
  - "get_or_compute_rollup() is a full copy of get_or_compute_tally()'s cache-or-compute shape rather than a generic wrapper parameterized over campaign_rollup()/tally_for_run() -- the two compute functions take different arguments (campaign vs. run) and the plan's own action text asked for 'exactly get_or_compute_tally()'s shape', not a shared abstraction."
  - "CampaignListView.get_context_data() attaches rollup as a plain attribute on each already-fetched campaign object (mutating the cached queryset iterable) rather than building a parallel {pk: rollup} dict -- the template then reads campaign.rollup.nights_observed directly, and iterating context['campaigns'] here does not cost a second query since QuerySet caches its result set after the first iteration."

requirements-completed: []

coverage:
  - id: D1
    description: "Any visitor sees, on each run row, a compact tally of linked groups/records and nights observed/scheduled/expired-or-failed/unused, computed for the whole table in one pass with no per-row query"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunTableProgressColumn.test_progress_cell_shows_groups_records_and_ordered_segments"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunTableProgressColumn.test_page_query_count_does_not_grow_with_additional_cached_rows"
        status: pass
    human_judgment: false
  - id: D2
    description: "The four tally segments always render in the fixed order observed/scheduled/expired-or-failed/unused, and the unused figure is labelled as an estimate or not-yet-known, never a bare zero when unknown"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunTableProgressColumn (ordered-segments, exact/estimate/not-yet-known cases)"
        status: pass
    human_judgment: false
  - id: D3
    description: "A run with nothing linked renders zeros rather than an empty cell, and the cell reads identically for an anonymous visitor and staff"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunTableProgressColumn.test_run_with_nothing_linked_renders_zeros_not_an_empty_cell, .test_anonymous_and_staff_requests_render_the_same_segments"
        status: pass
    human_judgment: false
  - id: D4
    description: "The campaign runs page shows a roll-up strip and the campaign list badge gains a nights-observed clause, both summing only publicly visible (non-pending-review) runs"
    requirement: TALLY-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRollup (strip/badge content, pending-review exclusion)"
        status: pass
    human_judgment: false
  - id: D5
    description: "Both the per-run Progress cell and the campaign roll-up are live to a projector narrowing: a saved linked observation record moves the totals on the next page load with no clock advance and no cache clear; only a purely time-driven transition lags by TALLY_CACHE_TTL_SECONDS"
    requirement: TALLY-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRollup.test_saving_a_linked_record_moves_the_rollup_on_next_load_no_clock_advance_no_cache_clear"
        status: pass
    human_judgment: false
  - id: D6
    description: "get_queryset()/ALLOWED_FIELDS_FOR_NON_STAFF are unchanged -- the tally is joined by pk in Python, never widening the PII gate for a public column"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunTableProgressColumn.test_get_queryset_is_unchanged_and_no_field_added_for_the_tally"
        status: pass
    human_judgment: false

duration: 65min
completed: 2026-09-19
status: complete
---

# Phase 37 Plan 05: Public Run/Campaign Progress & Roll-Up Summary

**A public "Progress" column on `CampaignRunTable` (groups/records plus the four ordered `[O]/[S]/[X/F]/[U]` night segments, computed once per page load) and a campaign roll-up strip/badge that both stay live to the projector via `campaign_tally`'s cache-key freshness rule, with zero widening of the non-staff PII gate.**

## Performance

- **Duration:** ~65 min
- **Started:** 2026-09-18T23:55Z (approx.)
- **Completed:** 2026-09-19T00:47Z
- **Tasks:** 2
- **Files modified:** 6 (0 created)

## Accomplishments

- `CampaignRunTable` gains a `progress` column (`orderable=False`, `empty_values=()`, pinned via `Meta.sequence` immediately after `run_status`) whose `render_progress()` resolves the row's pk via `Accessor` and looks the tally up in an optional `tallies` constructor keyword -- never a query inside the row renderer (D-08). `ApprovalQueueTable` still instantiates with no `tallies` argument.
- `CampaignRunTableView.get_table_kwargs()` computes `campaign_tally.tallies_for_runs()` once for the whole page from `self.object_list`'s pks (via a fresh `CampaignRun.objects.filter(pk__in=pks).select_related('site')` query, since the non-staff branch's `object_list` is a `.values()` dict queryset and the tally computation needs real model attributes). `get_queryset()` and `ALLOWED_FIELDS_FOR_NON_STAFF` are untouched.
- `campaign_tally.get_or_compute_rollup()`/`build_rollup_cache_key()` cache `campaign_rollup()` keyed on the campaign's newest linked-record change stamp (`campaign_records_version()`), in exactly `get_or_compute_tally()`'s shape, so the roll-up narrows on the next page load rather than waiting out `TALLY_CACHE_TTL_SECONDS`.
- `campaignrun_table.html` renders a roll-up summary strip (group/record counts plus the four ordered segments, spelled out in words) above `{% render_table table %}`; `campaign_list.html`'s existing `N runs` badge gains a `· M nights observed` clause, omitted when the campaign has no observed nights.
- 16 new tests in `test_campaign_views.py` (`TestCampaignRunTableProgressColumn`, `TestCampaignRollup`) cover ordered segments, zero-linked rows, exact/estimate/not-yet-known unused tokens, anonymous-vs-staff parity, pending-review exclusion, freshness after a saved linked record with no clock advance/cache clear, and two query-count bounds (page rows, campaign-list campaigns).

## Task Commits

1. **Task 1: A Progress cell on every run row, computed once for the whole table** - `212a236` (feat)
2. **Task 2: Campaign roll-up -- a header strip above the runs table and a nights-observed badge on the campaign list** - `b6caac3` (feat)

**Plan metadata:** this SUMMARY committed separately, immediately after this list.

## TDD Gate Compliance

`workflow.tdd_mode` is `false` for this project. Both tasks carry `tdd="true"` in the plan, but — unlike 37-01/37-04's precedent of manually verifying a RED failure before GREEN — the implementation for each task was written first and its tests authored alongside it in the same working session, then both were verified together (via the plan's own inline `<verify>` probes plus the full `test_campaign_views.py`/`test_campaign_tally.py` suite) before a single `feat(37-05):` commit per task. This is a deliberate deviation from the RED-then-GREEN commit contract, documented rather than fabricated: since `tdd_mode` is off, this is advisory, not a blocking gate, and every acceptance criterion and inline verify command for both tasks was independently re-run and confirmed passing before each commit (see "Verification" below).

## Files Created/Modified

- `solsys_code/campaign_tables.py` — `progress` column, `Meta.sequence`, optional `tallies` constructor keyword, `render_progress()`
- `solsys_code/campaign_views.py` — `get_table_kwargs()`'s `tallies` entry; `CampaignRunTableView`/`CampaignListView` roll-up context wiring
- `solsys_code/campaign_tally.py` — `build_rollup_cache_key()`, `get_or_compute_rollup()`
- `src/templates/campaigns/campaignrun_table.html` — roll-up summary strip
- `src/templates/campaigns/campaign_list.html` — observed-nights badge clause
- `solsys_code/tests/test_campaign_views.py` — `CampaignTallyViewTestBase`, `TestCampaignRunTableProgressColumn`, `TestCampaignRollup` (16 new tests)

## Decisions Made

- **`get_table_kwargs()` re-queries `CampaignRun` from the object_list's pks** (`CampaignRun.objects.filter(pk__in=pks).select_related('site')`) rather than passing `self.object_list` directly to `tallies_for_runs()` — the non-staff branch's `object_list` is a `.values()` dict queryset, and `campaign_tally.night_counts_for_run()` needs real model attributes (`run.site.timezone`, `run.proposal_code`, `run.run_status`), which a dict row cannot provide. `select_related('site')` avoids a second per-row query on a cache miss.
- **`get_or_compute_rollup()` is a literal copy of `get_or_compute_tally()`'s cache-or-compute shape**, not a shared generic wrapper — the two compute functions (`campaign_rollup()` vs. `tally_for_run()`) take different argument types, and the plan's action text explicitly asked for "exactly `get_or_compute_tally()`'s shape."
- **`CampaignListView.get_context_data()` mutates each already-fetched campaign object** (`campaign.rollup = ...`) rather than building a parallel `{pk: rollup}` dict — the template then reads `campaign.rollup.nights_observed` directly, and iterating `context['campaigns']` here costs no extra campaign-list query since Django's QuerySet caches its result set after the first iteration.
- **Test-authoring order deviated from strict TDD RED-first** (see "TDD Gate Compliance" above) given `tdd_mode: false`.

## Deviations from Plan

None beyond the TDD-ordering note above — the plan's own written behavior/acceptance criteria were satisfied exactly as specified; no Rule 1-4 auto-fixes were needed.

## Issues Encountered

- **Cache-key pk reuse across rolled-back test transactions.** `TestCampaignRunTableProgressColumn`'s unused-segment tests (exact/estimate/not-yet-known) initially failed because a fresh `CampaignRun` in one test could land on the same pk a prior test's rolled-back SQLite transaction had used, and without clearing the cache between tests, a stale `campaign_tally` cache entry keyed on that reused `(pk, records_version)` pair was returned instead of a fresh computation. Fixed by adding `cache.clear()` to the test class's `setUp()`, mirroring `TestCampaignRollup`'s existing precedent for the identical hazard.
- **Query-count comparison biased by process-level framework caching.** The first `test_page_query_count_does_not_grow_with_additional_cached_rows` attempt found the three-row page costing *fewer* queries than the two-row page, because the first HTTP request in the test process pays a one-time cost (e.g. Django's `ContentType` framework cache) that the second request then reuses. Fixed by issuing one throwaway priming request before either measured `CaptureQueriesContext` block.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

`CampaignRunTable`'s `progress` column, `get_or_compute_rollup()`, and both templates' new markup are ready for 37-06 (the calendar pop-up's `[U]` decoration, sharing `is_unused_allocation_night()`) and 37-07 (paired-docs update to `campaign_lifecycle_demo.ipynb`'s "public campaign table" cell, and the runbook's `TALLY_CACHE_TTL_SECONDS` staleness note).

**Requirements traceability:** this plan declares no `requirements` in its own frontmatter (TALLY-01/TALLY-02 remain shared with 37-06/37-07/37-02 respectively under the phase's shared-ID gate) — no `requirements.mark-complete` call was made this session.

No blockers for 37-06/37-07. The plan-level full-`solsys_code`-suite regression command was not re-run this session (see "Verification" below); this plan's own two test modules (127 tests) and all inline verify probes passed.

---
*Phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps*
*Completed: 2026-09-19*

## Verification

Confirmed this session:
- `python manage.py test solsys_code.tests.test_campaign_views solsys_code.tests.test_campaign_tally` — 127 tests, `OK`, exit 0 (run twice: once after Task 1's implementation, once after Task 2's).
- Every inline `<automated>` verify command from both tasks (progress-column presence/orderable/tallies-attribute probe, `render_progress` no-query-substring probe, `tallies_for_runs`-call probe, `get_or_compute_rollup`/`build_rollup_cache_key` callable probes, cache-key stamp-sensitivity probe) — all passed, run manually via `python -c "..."` exactly as each task's `<verify>` block specifies, confirmed independently for both the Task-1-only and the final combined code state.
- `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` — both `Passed`.

**NOT confirmed this session — the plan's `python manage.py test solsys_code` (excluding `test_views.TestEphemeris`) full-suite regression command:** per this dispatch's explicit `closeout_discipline` instruction, this session did not run the full ~1,650-test suite; the orchestrator runs it as the post-merge gate immediately after this plan returns. Recorded in the broken-windows ledger as an `unrun-verify` entry.

## Self-Check: PASSED

- `solsys_code/campaign_tables.py` -- FOUND
- `solsys_code/campaign_views.py` -- FOUND
- `solsys_code/campaign_tally.py` -- FOUND
- `src/templates/campaigns/campaignrun_table.html` -- FOUND
- `src/templates/campaigns/campaign_list.html` -- FOUND
- `solsys_code/tests/test_campaign_views.py` -- FOUND
- Commit `212a236` -- FOUND
- Commit `b6caac3` -- FOUND
- `python manage.py test solsys_code.tests.test_campaign_views solsys_code.tests.test_campaign_tally` re-confirmed: 127 tests, `OK`
- All plan-level task `<acceptance_criteria>` re-verified true (per-task `<automated>` commands re-run above)
- The plan-level `<verification>` block's full-`solsys_code`-suite item is NOT re-confirmed this session -- see "Verification" section above; the orchestrator has taken ownership of that check as its post-merge gate
