---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
verified: 2026-09-20T23:10:00Z
status: gaps_found
score: 4/5 must-haves verified
covered_files:
  - ".planning/REQUIREMENTS.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-01-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-01-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-02-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-02-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-03-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-03-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-04-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-04-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-05-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-05-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-06-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-06-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-07-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-07-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-08-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-08-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-REVIEW-FIX.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-REVIEW.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-UAT.md"
  - "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb"
  - "docs/runbooks/telescope_runs_calendar.rst"
  - "solsys_code/allocation_projector.py"
  - "solsys_code/calendar_utils.py"
  - "solsys_code/campaign_gap.py"
  - "solsys_code/campaign_reconciler.py"
  - "solsys_code/campaign_tables.py"
  - "solsys_code/campaign_tally.py"
  - "solsys_code/campaign_views.py"
  - "solsys_code/observation_projector.py"
  - "solsys_code/proposal_allocation.py"
  - "solsys_code/status_vocabulary.py"
  - "solsys_code/templatetags/calendar_display_extras.py"
  - "solsys_code/tests/test_campaign_tally.py"
  - "solsys_code/tests/test_campaign_views.py"
  - "solsys_code/unattended.py"
  - "src/templates/campaigns/campaign_list.html"
  - "src/templates/campaigns/campaignrun_table.html"
  - "src/templates/tom_calendar/partials/calendar.html"
  - "src/templates/tom_calendar/partials/event_form.html"
covered_digest: "v1:sha256:cf79cd5415355ecaad18b85558395fd0f62b73ee296453ba84d85be360aec5b3"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 5/5
  gaps_closed:
    - "G-37-4: the campaign roll-up strip's [U] total could disagree with the [U] values in the Progress cells beneath it for up to TALLY_CACHE_TTL_SECONDS after a purely time-driven or staff-run_status-driven change"
  gaps_remaining: []
  regressions: []
  human_items_resolved:
    - "Prior item 1 (calendar visual/interaction check) — resolved by 37-UAT.md tests 1, 2 and 3, all `result: pass`"
    - "Prior item 2 (product decision on roll-up unused staleness) — decided by the developer as Option (b) in 37-UAT.md test 4, implemented by plan 37-08 and verified closed below"
gaps:
  - truth: "Any visitor — not only staff — sees on each run a live tally of linked observation groups and records and of nights observed / scheduled / expired-or-failed / unused so far, updating as the projector narrows, and the campaign page rolls the same tally up across its runs"
    status: partial
    reason: "CR-01, independently reproduced. `CampaignRunTableView.get_table_kwargs()` mirrors only django-tables2's page NUMBER when slicing the pks it hands to `tallies_for_runs()`, but `RequestConfig.configure()` afterwards also applies `?sort=` (via `table.prefixed_order_by_field`) and `?per_page=` (via `table.prefixed_per_page_field`). Both change which rows the table actually renders, so rows outside the mirrored slice have no entry in `self.tallies` and `render_progress()` falls into its `Progress not available` branch. Measured on a 30-run campaign: the plain page renders 0 `Progress not available` cells, `?sort=-telescope_instrument` renders 5 of 25 rows without a tally, and `?per_page=50` renders 5 of 30 rows without a tally. REQUIREMENTS.md TALLY-01 names the campaign table row as one of the two surfaces the tally must appear on, and the view's own docstring advertises that interactive column-header sorting still works — so this is a supported interaction that silently removes the tally."
    artifacts:
      - path: "solsys_code/campaign_views.py"
        issue: "get_table_kwargs() (lines ~216-226) resolves only the `page` GET param and hardcodes `per_page = self.table_pagination['per_page']`; it ignores the `sort` param entirely, so the slice is taken in get_queryset()'s order rather than the order the table will render. The docstring's stated worst case ('an out-of-range/non-integer page number degrades to page 1's pks') does not cover sorting or a per_page override."
      - path: "solsys_code/tests/test_campaign_views.py"
        issue: "No test exercises the Progress column under `?sort=` or `?per_page=`, and no test asserts that a rendered runs page contains zero `Progress not available` cells. `TestCampaignRunTableProgressColumn` only covers the default, unsorted, default-per_page view."
    missing:
      - "Resolve the rendered row set the same way django-tables2 will: apply the `?sort=` ordering (and the `?per_page=` override) before slicing, or move the tally fetch to after `RequestConfig.configure()` and read the pks off `table.page.object_list` — so the tallies dict always covers exactly the rows rendered"
      - "A regression test on a campaign with more than one page of runs asserting that `?sort=<column>` and `?per_page=<n>` both render every row with a real tally and zero `Progress not available` cells"
      - "If a page-1 degradation is genuinely acceptable for some param combination, say so in the docstring explicitly rather than leaving sorting undescribed"
deferred: []
human_verification:
  - test: "Decide how the campaign roll-up strip should report a partially-unknown unused figure (CR-02). Reproduce with a campaign holding one allocation run with elapsed still-standing ALLOC: nights and one container run with a non-blank `proposal_code` whose `ProposalTimeAllocation` has never been fetched."
    expected: "A decision, not a code state. Today the strip renders `[U] ≈2` directly above rows reading `[U] 2` and `[U] not yet known`: `_apply_rollup_unused_fields()` adds nothing for the unknown code (so the unknown is absorbed as zero into a displayed total, against D-06's never-zero contract) while `unused_known` stays True from the other run's exact count, and `unused_is_estimate` is set from `bool(estimate_codes)` — the codes ATTEMPTED — so the `≈` appears even when no estimate contributed a single night. Either accept the `≈` hedge as the intended signal for 'incomplete total', or track contributing-vs-attempted separately and give the strip its own not-fully-known rendering."
    why_human: "This is pre-existing counting-rule semantics from plan 37-04, not something plan 37-08 introduced or was allowed to change (its 'No tally value changes' prohibition forbids it explicitly), and the right display for a partially-unknown roll-up total is a product decision that nothing in the codebase determines. D-10 fixes the counting rule but says nothing about how a partially-unknown sum should read."
---

# Phase 37: Status Vocabulary, Public Tallies & Provenance-Blind Gaps — Verification Report

**Phase Goal:** The layered calendar reads correctly to everyone — one status vocabulary instead of three that agree by convention, an ongoing public tally of what each run and campaign actually got, unused awarded nights that look unused, and coverage gaps that count every observation.
**Verified:** 2026-09-20T23:10:00Z
**Status:** gaps_found
**Re-verification:** Yes — after the `--gaps-only` re-execution of plan 37-08 (G-37-4). Supersedes the 2026-09-19 pass.

## Goal Achievement

### Observable Truths

| # | Truth (ROADMAP Success Criterion) | Status | Evidence |
|---|---|---|---|
| 1 | One status vocabulary drives every calendar title prefix and status ring — the three parallel prefix maps are gone, a placed-but-unobserved night has its own named state, and terminal-state detection goes through one facility-aware classifier instead of a hardcoded `status == 'COMPLETED'` | ✓ VERIFIED | Regression check only — `solsys_code/status_vocabulary.py` (12,319 bytes, mtime 2026-09-19 07:46) was not touched by this round's commits (`git diff --stat e8b7219~1 HEAD` lists only `campaign_tally.py`, `campaign_views.py`, two test modules and the runbook). Behavioral re-run this pass: `test_status_vocabulary` green inside a 67-test run. |
| 2 | Any visitor — not only staff — sees on each run a live tally of linked observation groups and records and of nights observed / scheduled / expired-or-failed / unused so far, updating as the projector narrows, and the campaign page rolls the same tally up across its runs | ✗ FAILED | **The liveness half is now fully delivered** (G-37-4 closed — see below), but the "any visitor sees on each run a tally" half breaks under a supported interaction. Measured on a 30-run campaign through the Django test client: plain page → 0 `Progress not available` cells; `?sort=-telescope_instrument` → 5 of 25 rendered rows have no tally; `?per_page=50` → 5 of 30 rendered rows have no tally. Root cause is structural and reproducible: `get_table_kwargs()` slices by page number only while `RequestConfig.configure()` applies `sort` and `per_page` afterwards (`django_tables2/config.py`, `for arg in ("page", "per_page")` plus `table.order_by = request.GET.getlist(table.prefixed_order_by_field)`). See Gaps. |
| 3 | A run's `run_status` never changes by itself: whatever its linked records did, it stays what a staff member set | ✓ VERIFIED | The one module changed this round that participates in the computation path, `campaign_tally.py`, gained three helpers (`_rollup_runs()`, `_apply_rollup_unused_fields()`, `_without_unused_fields()`) — all read-only: the applier reads `run.run_status` and `run.proposal_code` and writes neither. Behavioral: `TestTallyNeverWritesRunStatus` (including its AST check that no computation-path module assigns to the attribute) re-run green this pass. |
| 4 | An awarded night that came and went with nothing scheduled or observed is visibly different on the calendar from a night that was actually observed | ✓ VERIFIED | Regression check plus the human half now closed. `calendar_display_extras.py` and `src/templates/tom_calendar/partials/calendar.html` untouched this round; `.cal-event-unused` class and `[U]` token still wired at `calendar.html:185/278/281/298`. The prior pass left this present-but-visually-unconfirmed; **37-UAT.md tests 1 and 2 both `result: pass`** — a human confirmed the muted/dashed unused chip is distinguishable from a `[C]` cancelled night and that the `[U]` legend swatch toggles the filter. |
| 5 | Coverage-gap analysis counts every observation on the campaign calendar, so classical and queue time is no longer reported as unclaimed | ✓ VERIFIED | Regression check only — `campaign_gap.py` (23,032 bytes, mtime 2026-09-19 07:53) untouched this round; `observation_claimed_dates()` still present at line 172 and unioned into `claimed_dates()`. Behavioral: `test_campaign_gap` green inside this pass's 67-test run. |

**Score:** 4/5 truths verified (0 present, behavior-unverified)

### G-37-4 Closure Assessment (the carried-forward gap)

**G-37-4 is CLOSED.** Verified against the code and behaviorally, not from SUMMARY claims:

- `get_or_compute_rollup()` (`campaign_tally.py:696`) no longer returns the cached dict on a hit. Its hit branch is `rollup = dict(cached); _apply_rollup_unused_fields(rollup, _rollup_runs(campaign)); return rollup`, and its miss branch caches `_without_unused_fields(rollup)` — so no computed unused figure is ever written to the cache at all.
- There is exactly one route to the figure: `_apply_rollup_unused_fields()` is called from both `campaign_rollup()` (including the no-runs early return) and `get_or_compute_rollup()`'s hit branch. `git show 55349cf` confirms D-10's counting rule was **moved**, not rewritten — the estimate-per-distinct-code loop is byte-identical, and the exact-count derivation changed from `tally['unused_known'] and not tally['unused_is_estimate']` to `unused_nights_for_run(run) is not None`, which is the same predicate by construction (`_apply_unused_fields()` sets those two keys exactly when `unused_nights_for_run()` returns non-None).
- The end-to-end test is real, not vacuous: `test_rollup_strip_agrees_with_progress_cells_after_a_staff_status_edit_not_a_records_change` (`test_campaign_views.py:1173`) renders `campaigns:table` twice around a `run_b.save(update_fields=['run_status'])` that touches no `ObservationRecord`, asserts `[U] 3` / `[U] 2` / `[U] 1` in a whitespace-collapsed body before and `[U] 2` / `[U] 0` with `assertNotIn('[U] 3', ...)` after, then re-asserts the strip total equals the computed per-run sum.
- Independently re-run this pass: `TestGetOrComputeRollupFreshness` + `TestCampaignRollup` (both modules) — 25 tests, OK in 15.3 s.

### CR-01 (code review BLOCKER) — CONFIRMED, and it is the phase's gap

Verified from source **and** reproduced. `django_tables2/config.py`'s `RequestConfig.configure()` runs after `get_table_kwargs()`:

```python
order_by = self.request.GET.getlist(table.prefixed_order_by_field)   # ?sort=
if order_by:
    table.order_by = order_by
...
for arg in ("page", "per_page"):                                      # ?per_page=
    kwargs[arg] = int(self.request.GET[name])
```

`get_table_kwargs()` mirrors only `page` and hardcodes `per_page = self.table_pagination['per_page']` (25), and slices `self.object_list` in `get_queryset()`'s order. Reproduction (throwaway test, 30 runs in one campaign, removed after the run — the working tree is clean):

| Request | `Progress not available` occurrences in body (2 per cell) | Rows without a tally |
|---|---|---|
| plain | 0 | 0 |
| `?sort=-telescope_instrument` | 10 | 5 of 25 rendered |
| `?per_page=50` | 10 | 5 of 30 rendered |

Manifests only on campaigns with more than one page of runs (>25), and degrades gracefully (a muted token, never a wrong number). It is nonetheless an observable failure of Success Criterion 2 / TALLY-01 under an interaction the page explicitly offers.

Evidence gate (#3304): `campaign_views.py` was git-modified since the prior `verified:` timestamp (2026-09-19T17:05:00Z) by commit `9886c0c`, so this finding is in-contract and blocks regardless; it is additionally backed by the deterministic reproduction above, so the gate is satisfied on both routes.

### CR-02 (code review BLOCKER) — CONFIRMED as behavior, DOWNGRADED to a decision item

The reviewer's description of `_apply_rollup_unused_fields()` is accurate. Reproduced with stubbed dependencies (no DB):

```
ROLLUP unused fields: {'nights_unused': 2, 'unused_known': True, 'unused_is_estimate': True}
STRIP renders: [U] ≈2
  ROW run1 renders: [U] 2
  ROW run2 renders: [U] not yet known
```

A `proposal_code` whose `estimated_unused_nights()` returns `None` contributes nothing to the total but is still counted in `estimate_codes`, so `unused_is_estimate = bool(estimate_codes)` sets the `≈` with no estimate behind it, and `unused_known` stays True from the other run's exact count.

**It is not G-37-4 reached by another route, and it is not a regression.** `git show 55349cf` shows the estimate loop and the `unused_is_estimate = bool(estimate_codes)` line are unchanged from plan 37-04's original inline implementation — this semantics predates the gap-closure round and was passed by the prior verification. Plan 37-08's prohibitions forbid it explicitly: *"This plan changes WHEN the roll-up's unused figure is computed, never WHAT it counts."* The gap it was dispatched to close was the caching staleness, and it closed that.

**It is not silently accepted either.** The `≈` is a hedge rather than a flat contradiction (a reader sees "roughly 2" over "2" and "not yet known"), but the roll-up does absorb an unknown as zero into a displayed number, against D-06's "never zero" contract, and the `≈` fires on attempted rather than contributing codes. Routed to the human decision item above rather than raised as a blocker.

### CLAUDE.md Paired-Docs Rule (WR-08) — NOT a gap this round

The flag's premise does not hold for this round's commits:

- `solsys_code/observation_projector.py` was **not modified at all** by 37-08 — `git diff --stat e8b7219~1 HEAD` lists only `campaign_tally.py`, `campaign_views.py`, `tests/test_campaign_tally.py`, `tests/test_campaign_views.py`, the runbook and planning docs. Its paired `project_observation_calendar_demo.ipynb` has nothing to be stale against. (The prior pass separately established that Phase 37's earlier change to that module was a pure refactor for `PROJECTED_FACILITIES = ('LCO', 'SOAR')`.)
- `solsys_code/campaign_views.py`'s only change this round is the 19-line comment block above `CampaignListView.paginate_by` (`git show 9886c0c -- solsys_code/campaign_views.py` — comment lines only, zero code lines). CLAUDE.md's rule triggers on a module's *behavior* changing and carves out pure refactors and typo fixes, so `campaign_lifecycle_demo.ipynb` is not stale.
- `campaign_tally.py` — the module whose behavior actually changed — has no paired notebook in CLAUDE.md's map, and its values are unchanged (the notebook's cell-42 roll-up assertions still hold; `TestCampaignRollup`'s expected numbers were untouched and re-run green).
- The directory-scoped half of the rule **was** honoured: `docs/runbooks/telescope_runs_calendar.rst`'s `How fresh is the tally?` paragraph was updated in the same round (22 lines, commit `9886c0c`) and no longer ties an elapsed awarded night to the cache lifetime.

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `solsys_code/campaign_tally.py` | One shared campaign-level unused applier; cache never carries a computed unused figure | ✓ VERIFIED | 744 lines; `_rollup_runs()` (504), `_apply_rollup_unused_fields()` (539), `_without_unused_fields()` (601); both cache branches call the applier |
| `solsys_code/campaign_views.py` | Roll-up + per-row tallies wired into the runs page | ⚠️ PARTIAL | `get_context_data()` → `get_or_compute_rollup()` ✓; `get_table_kwargs()` → `tallies_for_runs()` ✓ but the page slice covers the wrong rows under `?sort=`/`?per_page=` (CR-01) |
| `solsys_code/tests/test_campaign_tally.py` | Cache-hit freshness coverage for both blind-spot drivers | ✓ VERIFIED | `TestGetOrComputeRollupFreshness` added (+166 lines, 9 cases); green |
| `solsys_code/tests/test_campaign_views.py` | End-to-end strip-vs-rows agreement + re-pinned query bounds | ✓ VERIFIED | `test_rollup_strip_agrees_with_progress_cells_...` substantive (not a vacuous substring check — collapses whitespace first and uses both `assertIn` and `assertNotIn`); `MARGINAL_QUERIES_PER_CAMPAIGN = 3` with a second added-campaign constancy assertion |
| `docs/runbooks/telescope_runs_calendar.rst` | `How fresh is the tally?` corrected | ✓ VERIFIED | Updated in `9886c0c`; no longer ties an elapsed awarded night or a refreshed allocation to `TALLY_CACHE_TTL_SECONDS` |
| `solsys_code/status_vocabulary.py` | Single marker/label/legend/classifier home | ✓ VERIFIED (regression) | Unchanged this round; 67-test regression run green |
| `solsys_code/campaign_gap.py` | Provenance-blind claim source | ✓ VERIFIED (regression) | Unchanged this round; `observation_claimed_dates()` at line 172 |
| `solsys_code/templatetags/calendar_display_extras.py` + `calendar.html` | `[U]` decoration, two channels | ✓ VERIFIED (regression) | Unchanged this round; `cal-event-unused` at `calendar.html:185/278/281/298`; UAT tests 1-2 pass |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `get_or_compute_rollup()` cache-HIT branch | `_apply_rollup_unused_fields()` | direct call on a `dict(cached)` copy before return | ✓ WIRED | The single line whose absence *was* G-37-4 |
| `get_or_compute_rollup()` cache-MISS branch | `cache.set(..., _without_unused_fields(rollup))` | explicit stripping helper | ✓ WIRED | Nothing computed is ever stored |
| `campaign_rollup()` | `_apply_rollup_unused_fields()` | called on both exits, no-runs return included | ✓ WIRED | One counting rule, no second copy |
| `_apply_rollup_unused_fields()` | `unused_nights_for_run()` → `is_unused_allocation_night()` | D-15's shared rule, the same one `unused_night_decoration()` reads | ✓ WIRED | No re-derivation |
| both roll-up paths | `_rollup_runs()`'s `.exclude(approval_status=PENDING_REVIEW)` | one shared queryset | ✓ WIRED | `test_pending_review_exclusion_is_applied_at_the_queryset_level` retargeted to `_rollup_runs`, assertion lines unchanged |
| `CampaignRunTableView.get_table_kwargs()` page slice | the rows django-tables2 actually renders | mirrors `?page` only | ✗ NOT_WIRED | `?sort=` and `?per_page=` are applied later by `RequestConfig.configure()`; the slice misses those rows (CR-01) |
| `CampaignListView` | `get_or_compute_rollup()` per listed campaign | `get_context_data()` | ✓ WIRED | WR-05 comment corrected to describe the live per-load cost |

### Data-Flow Trace (Level 4)

| Rendered value | Source | Produces real data | Status |
|---|---|---|---|
| Roll-up strip `[U]` total | `get_or_compute_rollup()` → `_apply_rollup_unused_fields()` → live `unused_nights_for_run()` per run | Yes, on every call including cache hits | ✓ FLOWING |
| Roll-up strip's five record-derived keys | cached `campaign_rollup()` half, keyed by `campaign_records_version()` | Yes | ✓ FLOWING |
| Progress cell `[U]` figure | `tallies_for_runs()` → `_apply_unused_fields()` live per call | Yes — for rows inside the page slice | ⚠️ PARTIAL — rows outside the slice render `Progress not available` under `?sort=`/`?per_page=` |
| Campaign-list nights badge | `campaign.rollup.nights_observed` | Yes | ✓ FLOWING |
| Calendar `[U]` chip | `unused_night_decoration()` → `is_unused_allocation_night()` | Yes | ✓ FLOWING |
| Gap page claimed/site-unknown | `_compute_gap()` | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Roll-up unused figure live on cache hits; strip agrees with rows end-to-end | `python manage.py test solsys_code.tests.test_campaign_tally.TestGetOrComputeRollupFreshness solsys_code.tests.test_campaign_tally.TestCampaignRollup solsys_code.tests.test_campaign_views.TestCampaignRollup --exclude-tag=ephemeris_segfault` | Ran 25 tests in 15.3 s — OK | ✓ PASS |
| TALLY-03 guard, status vocabulary, provenance-blind gap claims | `python manage.py test solsys_code.tests.test_campaign_tally.TestTallyNeverWritesRunStatus solsys_code.tests.test_status_vocabulary solsys_code.tests.test_campaign_gap --exclude-tag=ephemeris_segfault` | Ran 67 tests in 10.1 s — OK | ✓ PASS |
| Progress cells survive `?sort=` / `?per_page=` on a >1-page campaign | throwaway test, 30 runs, counting `Progress not available` in the rendered body (probe file removed; `git status --porcelain solsys_code/` clean) | plain 0, sorted 10, per_page=50 10 | ✗ FAIL |
| CR-02 roll-up aggregation semantics | standalone script stubbing `unused_nights_for_run` / `estimated_unused_nights`, no DB | strip `[U] ≈2` over rows `[U] 2` and `[U] not yet known` | ✗ FAIL (downgraded to decision item — see above) |

Full-suite and lint gates were run by the orchestrator (post-merge test gate exit 0; both ruff gates clean) and not repeated here.

### Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| — | — | No `scripts/*/tests/probe-*.sh` exist and no PLAN/SUMMARY declares a probe | ? SKIP |

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|---|---|---|---|---|
| STATUS-01 | 37-01, 37-07 | One status vocabulary replaces the three parallel prefix maps | ✓ SATISFIED | Truth 1 (regression-verified) |
| STATUS-02 | 37-01 | General terminal-state classifier replaces `status == 'COMPLETED'` | ✓ SATISFIED | Truth 1 |
| TALLY-01 | 37-02, 37-04, 37-05, 37-06, 37-07, 37-08 | Public per-run tally on the campaign table row and run detail, updating as the projector narrows | ✗ BLOCKED | Truth 2 — the tally disappears from the campaign table row for rows outside the mirrored page slice under `?sort=`/`?per_page=` (CR-01). The liveness clause itself is satisfied. |
| TALLY-02 | 37-04, 37-05, 37-07, 37-08 | Campaign page rolls the same tally up | ✓ SATISFIED | G-37-4 closed; strip is live on every call, proven end-to-end. CR-02's partially-unknown-total rendering routed to a decision item, not a blocker. |
| TALLY-03 | 37-04 | `run_status` never set automatically from linked records | ✓ SATISFIED | Truth 3 — new applier is read-only; AST guard green |
| UNUSED-01 | 37-04, 37-06, 37-07, 37-08 | Unused awarded night visually distinct | ✓ SATISFIED | Truth 4 — plus 37-UAT.md tests 1 and 2 human-passed |
| GAPB-01 | 37-03, 37-07 | `claimed_dates()` counts every observation | ✓ SATISFIED | Truth 5 |

No orphaned requirements: REQUIREMENTS.md maps exactly these seven IDs to Phase 37 and each appears in at least one plan's `requirements` field. **Note:** REQUIREMENTS.md line 52 already marks TALLY-01 `[x]` and line 125 `Complete`; that should not be taken as settled while the gap above is open.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `solsys_code/campaign_views.py` | ~216-226 | Page slice mirrors only one of the three GET params django-tables2 consumes | 🛑 Blocker | Rows rendered under `?sort=`/`?per_page=` have no tally; see Gaps |
| `solsys_code/campaign_tally.py` | `_apply_rollup_unused_fields()` 539-599 | Unknown contributor absorbed as zero into a displayed total; `unused_is_estimate` set from attempted rather than contributing codes | ⚠️ Warning | Strip can read `[U] ≈2` above a row reading `[U] not yet known`. Pre-existing (plan 37-04), explicitly out of 37-08's mandate. Routed to a human decision. |
| `solsys_code/campaign_tally.py` | `campaign_rollup()` 655-663 | Deliberate double allocation-event lookup per run on the cache-miss path | ℹ️ Info | Documented in an inline comment as the price of one route to the figure; the alternative reintroduces the two-sources structure G-37-4 is made of. Bounded and pinned by `assertNumQueries` and the re-pinned marginal-cost tests. |
| `solsys_code/management/commands/backfill_lco_observations.py` | 303 | Bare `block.get('state') == 'COMPLETED'` outside `status_vocabulary` | ℹ️ Info | Carried forward unchanged from the prior pass and from 37-UAT.md's notes; outside every Phase 37 plan's `files_modified`. Follow-up quick task. |
| `src/templates/tom_calendar/partials/event_form.html` | 211 | Segment labels not pluralised (`15 Unused awarded night`) | ℹ️ Info | Observed and accepted as non-blocking by the developer during UAT test 3; 37-08 was explicitly prohibited from touching it. Follow-up quick task. |
| `solsys_code/campaign_views.py`, `tests/*`, runbook | various | `TBD` string occurrences | ℹ️ Info (not a debt marker) | Every hit is the domain term "TBD window"/"TBD Telescope"/"TBD Coordinator" on `CampaignRun`, present since Phase 15. No `FIXME`/`XXX` in any file changed this round. The debt-marker gate does not fire. |

### Deferred Items

None actionable. Phase 37 is the final phase of the v2.4 milestone (ROADMAP.md holds phases 33-37 only), so no later phase exists to defer a gap to. `deferred-items.md` records one order-dependent Playwright flake in `test_bootstrap5_rendering.py`, confirmed non-reproducible on an immediate identical re-run and outside every Phase 37 plan's `files_modified` — a follow-up quick task, not a Phase 37 gap.

### Human Verification Required

#### 1. Product decision: how a partially-unknown roll-up unused total should read (CR-02)

**Test:** Build a campaign with one allocation run holding elapsed still-standing `ALLOC:` nights and one container run whose `proposal_code` is set but whose `ProposalTimeAllocation` has never been fetched. Load `campaigns:table` and read the strip against the rows.
**Expected:** A decision. Today: strip `[U] ≈2`, rows `[U] 2` and `[U] not yet known`. Either accept `≈` as the agreed signal for "total incomplete", or separate contributing from attempted estimate codes and give the strip its own not-fully-known rendering.
**Why human:** Pre-existing semantics from plan 37-04, explicitly outside 37-08's mandate, and D-10 fixes the counting rule without saying how a partially-unknown sum should read. Nothing in the codebase determines the answer.

### Gaps Summary

**G-37-4 is genuinely closed.** The caching defect UAT raised is gone by construction, not by TTL: `get_or_compute_rollup()` strips the three `unused_*` keys before caching and re-applies them live on every call including hits, through a single campaign-level applier both paths share. The counting rule was moved, not rewritten, and every pre-existing expected value in `TestCampaignRollup` still passes. The end-to-end page test is substantive and I re-ran it green.

**One new gap blocks the phase goal.** Success Criterion 2's "any visitor sees on each run a live tally" fails for a supported interaction: the tally fetch mirrors only django-tables2's page number, so sorting the runs table — or overriding `per_page` — on a campaign with more than 25 runs renders rows whose tallies were never fetched, and those Progress cells read "Progress not available". Reproduced deterministically (5 of 25 rows under `?sort=`, 5 of 30 under `?per_page=50`). The fix is confined to `CampaignRunTableView.get_table_kwargs()` plus one regression test; it is independent of everything 37-08 built, and closing it does not disturb the roll-up work.

**Two escalated findings resolved differently from how they were framed.** CR-02 is real behavior but is pre-existing, hedged by `≈`, and explicitly outside the gap-closure plan's mandate — a decision item, not a blocker, and not G-37-4 by another route. WR-08's paired-docs concern does not hold for this round at all: `observation_projector.py` was not touched, `campaign_views.py`'s only change is a comment block, and the one directory-scoped paired doc that *was* in scope — the runbook's freshness paragraph — was updated in the same commit as the code.

---

_Verified: 2026-09-20T23:10:00Z_
_Verifier: Claude (gsd-verifier)_
