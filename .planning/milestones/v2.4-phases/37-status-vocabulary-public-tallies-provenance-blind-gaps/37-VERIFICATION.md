---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
verified: 2026-09-21T18:41:52Z
status: passed
score: 5/5 must-haves verified
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
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-09-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-09-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-10-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-10-SUMMARY.md"
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
covered_digest: "v1:sha256:8b81d2f652422084716be42e359b833727784ecdc4d387d1b5fc8a542126bf72"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "G-37-5 (prior CR-01): the public Progress cell disappeared from rows rendered under ?sort= or ?per_page= on a campaign with more than one page of runs, because get_table_kwargs() reimplemented django-tables2's page resolution before RequestConfig.configure() applied those GET params"
    - "G-37-6 (prior CR-02, carried as a human product decision, since decided as D-20): the campaign roll-up strip absorbed an unknown contributor into its total as zero and rendered an approximation qualifier derived from proposal codes ATTEMPTED rather than codes that actually contributed"
  gaps_remaining: []
  regressions: []
  human_items_resolved:
    - "Prior item 1 (product decision on how a partially-unknown roll-up unused total should read) — decided by the developer as D-20, implemented by plan 37-10, verified closed below"
  review_fixes_verified:
    - "37-REVIEW CR-01 (the per_page cap clamped below-range values UP to the maximum) — fixed in a1d4328 and behaviorally re-verified: ?per_page=0/-1/-9999 render DEFAULT_TABLE_PER_PAGE (25) rows, ?per_page=100000/101 clamp DOWN to MAX_TABLE_PER_PAGE (100)"
    - "37-REVIEW WR-01..WR-07 — all present in the source and green; none regressed either gap closure (39 gap-closure tests + 101 tally/calendar tests + 67 vocabulary/gap tests re-run in this pass)"
gaps: []
deferred: []
advisory:
  - finding: "An unauthenticated GET to CampaignRunTableView may now cost ~300 queries at the accepted MAX_TABLE_PER_PAGE=100 ceiling (100 rows x the 3 queries/row this view's own test pins), on an endpoint with no throttle"
    category: security
    reason: "Deliberate, documented tradeoff recorded in campaign_views.py's MAX_TABLE_PER_PAGE comment during the WR-07 fix (the review offered lowering the constant and the fixer chose the lower-risk option of keeping 100). Bounded and pinned by tests, and strictly better than the pre-cap unbounded state. Lowering the constant to 50 would halve the ceiling if the developer wants it lower; no code change is required for this phase's goal."
    evidence_status: "none provided — the cost is bounded and test-pinned; no exploit or degradation was observed"
  - finding: ".planning/REQUIREMENTS.md's traceability table still marks TALLY-03, STATUS-01, STATUS-02 and GAPB-01 as `Gaps Found` (and leaves their checkboxes unticked), which the codebase now contradicts"
    category: other
    reason: "Bookkeeping churn, not a codebase gap: commit 283f6e8 ('revert premature Complete requirements after gaps found') blanket-reverted four IDs after the 2026-09-20 gaps_found verdict, even though the only open gap then was on TALLY-01. All four are implemented and behaviorally green in this pass. The orchestrator should reconcile the table to `Complete` for all seven Phase 37 IDs on phase completion."
    evidence_status: "none needed — planning-artifact state, verified against the code it describes"
---

# Phase 37: Status Vocabulary, Public Tallies & Provenance-Blind Gaps — Verification Report

**Phase Goal:** The layered calendar reads correctly to everyone — one status vocabulary instead of three that agree by convention, an ongoing public tally of what each run and campaign actually got, unused awarded nights that look unused, and coverage gaps that count every observation.
**Verified:** 2026-09-21T18:41:52Z
**Status:** passed
**Re-verification:** Yes — after the 37-09/37-10 gap-closure wave and the 37-REVIEW/37-REVIEW-FIX pass. Supersedes the 2026-09-20 `gaps_found` report.

## Goal Achievement

### Observable Truths

| # | Truth (ROADMAP Success Criterion) | Status | Evidence |
|---|---|---|---|
| 1 | One status vocabulary drives every calendar title prefix and status ring — the three parallel prefix maps are gone, a placed-but-unobserved night has its own named state, and terminal-state detection goes through one facility-aware classifier instead of a hardcoded `status == 'COMPLETED'` | ✓ VERIFIED | Structural: `grep -rn '_CLASSICAL_STATUS_PREFIX\|_FAILURE_PREFIX_BY_STATUS\|_RUN_STATUS_CALENDAR_PREFIX\|_TERMINAL_PREFIXES'` over `solsys_code/` + `src/` returns only two *docstring/comment* references naming the retired maps — zero live definitions or uses. `status_vocabulary.py` holds the single `DisplayState`/`MARKER`/`LABEL`/legend home including `DisplayState.SCHEDULED` (`[S]`, the placed-but-unobserved state) and `terminal_observing_states_for(facility)` built from `facility.get_terminal_observing_states() - failed_states_for(facility)` (line 235). The only surviving `== 'COMPLETED'` in non-test code is `backfill_lco_observations.py:303`, and it tests an LCO API *block payload* key (`block.get('state')`), not an `ObservationRecord.status` — outside every Phase 37 plan's `files_modified` (Info below). Behavioral: `test_status_vocabulary` re-run green in this pass. Regression-safe: the module is byte-unchanged since the prior report (`git diff --stat f29210c HEAD` lists eight files, none of them `status_vocabulary.py`). |
| 2 | Any visitor — not only staff — sees on each run a live tally of linked observation groups and records and of nights observed / scheduled / expired-or-failed / unused so far, updating as the projector narrows, and the campaign page rolls the same tally up across its runs | ✓ VERIFIED | **The prior report's one gap is closed, re-measured, not taken on trust.** `get_table_kwargs()` no longer predicts the page (it returns `{'order_by': ()}`); a new `CampaignRunTableView.get_table()` calls `super().get_table()` first — which is where `RequestConfig(...).configure(table)` applies `sort`/`page`/`per_page` — and then resolves pks from `table.paginated_rows`, the exact `BoundRows` `{% render_table table %}` iterates (`campaignrun_table.html:98`). Re-measured this pass on the same 30-run fixture: **zero** `Progress not available` occurrences under `?sort=-telescope_instrument` (25 rows), under `?per_page=50` (30 rows), under both combined, on the page-2 tail, on tied `window_start`, and under `?page=99`/`?page=banana` — against the prior report's 10 occurrences each for the first two. The assertion is not vacuous: `_assert_full_coverage()` also asserts *set equality* between the rendered-pk set and `table.tallies`' key set, and the `'Progress not available'` literal it counts is the exact string `render_progress()` emits (`campaign_tables.py:192`). Liveness half: `test_saving_a_linked_record_moves_the_rollup_on_next_load_no_clock_advance_no_cache_clear` and `test_rollup_strip_agrees_with_progress_cells_after_a_staff_status_edit_not_a_records_change` both green. Public half: `test_anonymous_and_staff_requests_render_the_same_segments` green. 39 + 101 tests re-run by me, OK. |
| 3 | A run's `run_status` never changes by itself: whatever its linked records did, it stays what a staff member set | ✓ VERIFIED | The one computation-path module changed this wave, `campaign_tally.py`, gained only read-only logic: `_apply_rollup_unused_fields()` reads `run.proposal_code` and `unused_nights_for_run(run)` and writes nothing back. Behavioral: `TestTallyNeverWritesRunStatus` — including its AST check that no computation-path module contains an assignment to the attribute — re-run green in this pass (inside a 67-test run). |
| 4 | An awarded night that came and went with nothing scheduled or observed is visibly different on the calendar from a night that was actually observed | ✓ VERIFIED | Regression + a standing human pass. `solsys_code/templatetags/calendar_display_extras.py` and `src/templates/tom_calendar/partials/calendar.html` are byte-unchanged since the prior report (not in `git diff --stat f29210c HEAD`); `cal-event-unused` still appears 5× in `calendar.html`, and `[U]`/`Unused awarded night` are still sourced from `MARKER[DisplayState.UNUSED]`/`LABEL[DisplayState.UNUSED]`. The visual half was human-confirmed in `37-UAT.md` tests 1 and 2, both `result: pass`, against code identical to today's. `test_calendar_display_extras` (101-test run with the Progress-column class) green — confirming 37-10's new `unknown_runs` segment key stays inert on the per-run/pop-up path (`tally.get('unused_unknown_runs', 0)` → 0). |
| 5 | Coverage-gap analysis counts every observation on the campaign calendar, so classical and queue time is no longer reported as unclaimed | ✓ VERIFIED | Regression. `campaign_gap.py` unchanged since the prior report; `observation_claimed_dates()` (line 172) is still **unioned**, never substituted, into `claimed_dates()` (`claimed \|= observation_claimed`), with the site-unknown count returned alongside. Behavioral: `test_campaign_gap` green in this pass's 67-test run. |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### G-37-5 Closure Assessment (prior gap)

**CLOSED.** Verified from the source and re-measured behaviorally, not from SUMMARY claims:

- `get_table_kwargs()` (`campaign_views.py:256`) is now three lines returning `{'order_by': ()}`; the page-slice reimplementation the prior report faulted is *gone*, not merely patched.
- `get_table()` (`campaign_views.py:275`) is ordered correctly: `table = super().get_table(**kwargs)` runs `RequestConfig.configure()` first, then `pks = {Accessor('pk').resolve(row.record, quiet=True) for row in table.paginated_rows}`, then one `tallies_for_runs()` pass over `CampaignRun.objects.filter(pk__in=pks).select_related('site')`. `Accessor` (not `record.pk` or a dict subscript) is required because a staff row is a model instance and an anonymous row is a `.values()` dict — both branches are exercised by the tests.
- The coverage tests assert the *property* (every rendered row has a tally) as set equality, never row order — so they cannot pass by accidentally re-encoding the old ordering.
- Re-measurement (my run, 39 tests, OK in 4.5 s): `plain 0 / sorted 0 / per_page=50 0` `Progress not available` occurrences, against the prior report's `plain 0 / sorted 10 / per_page=50 10`.

### G-37-6 Closure Assessment (prior human decision item, decided as D-20)

**CLOSED.** The prior report escalated this as a product decision rather than a blocker; the developer decided it (D-20) and plan 37-10 implemented it:

- `_apply_rollup_unused_fields()` (`campaign_tally.py:549`) now collects `estimate_codes` (attempted) and `contributing_codes` (those whose `estimated_unused_nights()` returned non-`None`) separately, derives `rollup['unused_is_estimate'] = bool(contributing_codes)` — the exact substitution D-20 required — and counts `unused_unknown_runs` **per run**, adding one unit for every run with no allocation events and either a blank `proposal_code` or a code that did not contribute.
- `tally_segments()` carries `unknown_runs` on all four segment dicts (literal `0` on three, `tally.get('unused_unknown_runs', 0)` on the unused one), so a per-run tally is inert by construction and only the strip branches on it.
- `campaignrun_table.html:92` has the fourth branch: `{% if not segment.known %}not yet known{% elif segment.unknown_runs %}at least {% if segment.is_estimate %}&approx;{% endif %}{{ segment.count }} ({{ segment.unknown_runs }} run{{ ...|pluralize }} not yet known){% elif segment.is_estimate %}&approx;{{ segment.count }}{% else %}{{ segment.count }}{% endif %}`.
- End-to-end on the developer's own reproduction, re-run by me and green: the strip renders `[U] at least 2 (1 run not yet known)` over rows reading `[U] 2` and `[U] not yet known`, with `assertNotIn('[U] ≈2', body)` — the exact contradiction the prior report reproduced is now asserted absent. A second test proves the two signals co-occur rather than cancel: `[U] at least &approx;3 (1 run not yet known)`.
- `_without_unused_fields()` resets the new key to `0` before `cache.set()`, so G-37-4's "nothing computed is ever cached" contract still holds with the fourth key.

### 37-REVIEW CR-01 (the review's critical) — FIX HOLDS

The review's critical was the *inverse* of a cap: `?per_page=0` was rewritten to `100`, taking a public page from 25 rendered rows to 100 — a 4× query amplification introduced by the control meant to prevent it.

Source today (`campaign_views.py:201-205`):

```python
if per_page is not None and not (1 <= per_page <= MAX_TABLE_PER_PAGE):
    clamped = MAX_TABLE_PER_PAGE if per_page > MAX_TABLE_PER_PAGE else DEFAULT_TABLE_PER_PAGE
```

Two directions, two targets. Behaviorally re-verified by me, not read: `test_degenerate_per_page_falls_back_to_the_default_not_the_maximum` sub-tests `0`, `-1`, `-9999` all render exactly `DEFAULT_TABLE_PER_PAGE` (25) rows on a 30-run campaign; `test_huge_per_page_is_capped_at_the_maximum` and `test_first_value_above_the_cap_is_clamped_down` render exactly `MAX_TABLE_PER_PAGE` (100) on a **120**-run fixture (WR-01's point: the old 30-run fixture made the cap test vacuous); `test_per_page_exactly_at_the_cap_is_honoured` pins the boundary. All green.

The other seven fixes are present and did not regress either gap closure: WR-01 (real boundary fixture), WR-02 (the notebook's split `assert (...), '...' '...'` is now one parenthesized message and the notebook re-executes clean), WR-03 (`unused_is_estimate` writes `True` in the not-yet-known branch, agreeing with all four not-known writers), WR-04 (docstring counts now "four"/"nine"), WR-05 (`tallies=` constructor kwarg dropped — `grep -rn 'tallies=' solsys_code/ src/` returns only docstring prose; `ApprovalQueueTable.Meta` excludes `progress` so no staff queue page renders a permanently-dead column), WR-06 (the capped field name resolves as `_meta.prefix + _meta.per_page_field`, pinned by `test_per_page_field_name_used_by_the_cap_matches_the_table`), WR-07 (comment-only).

### CLAUDE.md Paired-Docs Rule — SATISFIED for this wave

Both wave-8 plans declared their paired artifacts up front and both actually carry real, executed content:

- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` — 50 cells, output committed. Cell 47 (37-09) is executed with 664 chars of real output reading `plain rendered_rows=25 not_available_count=0 / sorted (?sort=-telescope_instrument) rendered_rows=25 not_available_count=0 / widened (?per_page=50) rendered_rows=30 not_available_count=0` on its own separate 30-run campaign, and names the `MAX_TABLE_PER_PAGE=100` cap. Cell 42 (37-10) is executed with 734 chars of real output reading `Unused awarded night: [U] at least 3 (4 runs not yet known)` above per-run rows of `[U] 3` and `[U] not yet known` — the strip's new fourth branch demonstrated on real data, with its own `unused_unknown_runs >= 3` / `< rollup['runs']` assertions.
- `docs/runbooks/telescope_runs_calendar.rst` — the "What does a run's or a campaign's public tally show?" section gained the roll-up's `at least N (M runs not yet known)` explanation and a new paragraph stating that the `≈`/`&approx;` qualifier now means "an estimate contributed", explicitly noting an operator who remembers the old behavior "was seeing the defect, not the design".
- 37-09's SUMMARY documents one deviation (new cells placed before the notebook's load-bearing teardown pair rather than literally last, because a write after `shutil.rmtree(scratch_db_dir)` fails). I confirmed the placement independently: the new pairs sit at indices 41/42 and 46/47, the teardown pair is still last, and both new code cells carry committed output — the rule's intent (executed demonstration of the new behavior) is met.

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `solsys_code/campaign_views.py` | Tally resolved from the rows actually rendered; a two-direction `per_page` clamp | ✓ VERIFIED | `get_table()` reads `table.paginated_rows` after `super().get_table()`; `get()` clamps high→100, low→25 on a field name resolved from the table's own Meta; `get_table_kwargs()` reduced to `{'order_by': ()}` |
| `solsys_code/campaign_tally.py` | Contributing-vs-attempted split; per-run unknown count; cache never carries a computed unused figure | ✓ VERIFIED | `_apply_rollup_unused_fields()` 549-646 (`contributing_codes`, `attempted_not_contributing`, `unused_unknown_runs`); `_without_unused_fields()` resets all four keys; `tally_segments()` defaults `unknown_runs` inertly |
| `src/templates/campaigns/campaignrun_table.html` | Roll-up strip's fourth ("at least … not yet known") branch | ✓ VERIFIED | Line 92, four branches, `&approx;` only inside the estimate sub-branch; pluralised run count |
| `solsys_code/campaign_tables.py` | Progress cell reads an attribute-attached tallies dict; no dead kwarg; approval queue excludes the column | ✓ VERIFIED | `self.tallies = {}` in `__init__`, populated only by the view; `ApprovalQueueTable.Meta.exclude` includes `'progress'` |
| `solsys_code/tests/test_campaign_views.py` | Coverage matrix over sort/per_page/page/tie/empty/clamped + the cap boundaries | ✓ VERIFIED | `TestProgressColumnCoversEveryRenderedRow` (12 cases, set-equality based), `TestProgressColumnOnDegenerateCampaigns` (2), two new `TestCampaignRollup` cases for D-20 |
| `solsys_code/tests/test_campaign_tally.py` | Contributing/attempted/blank-code matrix + cache reset + accounting invariant | ✓ VERIFIED | `TestRollupPartiallyKnownUnusedTotal` green; the invariant (known contributors + `unused_unknown_runs` == `rollup['runs']`) would break loudly on a new unaccounted contributor kind |
| `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` | Executed demonstration of both closures | ✓ VERIFIED | Cells 42 and 47, real committed output (see paired-docs section) |
| `docs/runbooks/telescope_runs_calendar.rst` | Public-tally section describes the partially-known roll-up and the contribution-based qualifier | ✓ VERIFIED | Lines ~1992-2013 |
| `solsys_code/status_vocabulary.py` | Single marker/label/legend/classifier home | ✓ VERIFIED (regression) | Unchanged since the prior report; 67-test run green |
| `solsys_code/campaign_gap.py` | Provenance-blind claim source | ✓ VERIFIED (regression) | Unchanged; `observation_claimed_dates()` unioned at line 353/`claimed \|= observation_claimed` |
| `solsys_code/templatetags/calendar_display_extras.py` + `calendar.html` | `[U]` decoration, two channels | ✓ VERIFIED (regression) | Unchanged; `cal-event-unused` ×5; UAT tests 1-2 pass against identical code |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `CampaignRunTableView.get_table()` | the rows django-tables2 actually renders | `table.paginated_rows` read after `super().get_table()` (which runs `RequestConfig.configure()`) | ✓ WIRED | The link whose absence *was* G-37-5. Same `BoundRows` `{% render_table table %}` iterates (`campaignrun_table.html:98`); no per-row query added |
| `CampaignRunTableView.get()` | django-tables2's own `per_page` read | `self.table_class._meta.prefix + _meta.per_page_field` on a mutable `request.GET` copy | ✓ WIRED | Matches `table.prefixed_per_page_field`, pinned by a test; both clamp directions exercised |
| `CampaignRunTable.render_progress()` | `table.tallies` | attribute assignment from the view only | ✓ WIRED | No `tallies=` kwarg anywhere; unresolvable pk degrades to the muted token, never a query |
| `_apply_rollup_unused_fields()` | `unused_nights_for_run()` → `is_unused_allocation_night()` | D-15's shared rule, the same one `unused_night_decoration()` reads | ✓ WIRED | No re-derivation; table and calendar still agree by construction |
| `_apply_rollup_unused_fields()` | `proposal_allocation.estimated_unused_nights()` | second pass over the attempted code set | ✓ WIRED | Contributing subset drives both the total and the `≈` qualifier |
| `tally_segments()` unused segment | `campaignrun_table.html` fourth branch | `unknown_runs` | ✓ WIRED | Non-zero only for a roll-up; `0` on the per-run/pop-up path |
| `get_or_compute_rollup()` (both branches) | `_apply_rollup_unused_fields()` | one campaign-level applier | ✓ WIRED | G-37-4's fix intact after 37-10's widening; `_without_unused_fields()` resets the new key too |
| `campaign_gap.claimed_dates()` | `observation_claimed_dates()` | set union | ✓ WIRED | Never substitution |

### Data-Flow Trace (Level 4)

| Rendered value | Source | Produces real data | Status |
|---|---|---|---|
| Progress cell groups/records/segments | `get_table()` → `tallies_for_runs()` over the pks of the rows actually rendered | Yes — for **every** rendered row now, under any sort/page/per_page combination | ✓ FLOWING |
| Roll-up strip `[U]` total + unknown-run count | `get_or_compute_rollup()` → `_apply_rollup_unused_fields()` → live `unused_nights_for_run()` / `estimated_unused_nights()` per call | Yes, on cache hits too; unknown contributors counted, never zeroed | ✓ FLOWING |
| Roll-up strip's record-derived keys | cached `campaign_rollup()` half, keyed by `campaign_records_version()` | Yes | ✓ FLOWING |
| Campaign-list nights badge | `campaign.rollup.nights_observed` (the only roll-up field that template reads) | Yes | ✓ FLOWING |
| Calendar `[U]` chip | `unused_night_decoration()` → `is_unused_allocation_night()` | Yes | ✓ FLOWING |
| Gap page claimed/site-unknown | `_compute_gap()` → `claimed_dates()` ∪ `observation_claimed_dates()` | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Every rendered row keeps its tally under `?sort=`/`?per_page=`/`?page=`/ties/empty/clamped; both `per_page` clamp directions; D-20 strip rendering end-to-end; roll-up liveness | `python manage.py test solsys_code.tests.test_campaign_views.TestProgressColumnCoversEveryRenderedRow solsys_code.tests.test_campaign_views.TestProgressColumnOnDegenerateCampaigns solsys_code.tests.test_campaign_views.TestCampaignRollup solsys_code.tests.test_campaign_tally.TestRollupPartiallyKnownUnusedTotal --exclude-tag=ephemeris_segfault` | Ran 39 tests in 4.5 s — OK | ✓ PASS |
| Status vocabulary single source; provenance-blind gap claims; TALLY-03 no-self-write AST guard | `python manage.py test solsys_code.tests.test_status_vocabulary solsys_code.tests.test_campaign_gap solsys_code.tests.test_campaign_tally.TestTallyNeverWritesRunStatus --exclude-tag=ephemeris_segfault` | Ran 67 tests in 2.9 s — OK | ✓ PASS |
| Calendar/pop-up tally unaffected by the new `unknown_runs` key; per-row query budget still bounded at 3/row | `python manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_campaign_views.TestCampaignRunTableProgressColumn --exclude-tag=ephemeris_segfault` | Ran 101 tests in 2.2 s — OK | ✓ PASS |
| Paired notebook carries real executed output for both closures | `json.load()` on `campaign_lifecycle_demo.ipynb`, reading cells 42 and 47 outputs | `[U] at least 3 (4 runs not yet known)`; `not_available_count=0` for plain/sorted/widened | ✓ PASS |
| Debt-marker gate on every file changed this wave | `grep -n -E "FIXME\|XXX\|HACK\|TODO\|PLACEHOLDER"` over the 8 changed files | Only two hits, both the domain term "tier-3 PLACEHOLDER Observatory" (Phase 22); zero `TODO`/`FIXME`/`XXX` | ✓ PASS |

Full-suite and lint gates were run by the orchestrator before this verification (1774 tests exit 0; both ruff gates Passed; post-merge build gate exit 0) and were not repeated — per the one-full-run-per-verification constraint, this pass ran only named classes.

### Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| — | — | No `scripts/*/tests/probe-*.sh` exist and no PLAN/SUMMARY declares a probe | ? SKIP |

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|---|---|---|---|---|
| STATUS-01 | 37-01, 37-07 | One status vocabulary replaces the three parallel prefix maps | ✓ SATISFIED | Truth 1 — zero live references to the three retired maps; `DisplayState.SCHEDULED` present |
| STATUS-02 | 37-01 | General terminal-state classifier replaces `status == 'COMPLETED'` | ✓ SATISFIED | Truth 1 — `terminal_observing_states_for()` / `failed_states_for()`; only remaining literal is an LCO API block-payload check outside phase scope |
| TALLY-01 | 37-02, 37-04, 37-05, 37-06, 37-07, 37-08, 37-09 | Public per-run tally on the campaign table row and run detail, updating as the projector narrows | ✓ SATISFIED | Truth 2 — the prior report's blocker is closed and re-measured at zero missing cells; anonymous and staff render identically |
| TALLY-02 | 37-04, 37-05, 37-07, 37-08, 37-10 | Campaign page rolls the same tally up | ✓ SATISFIED | Truth 2 + G-37-6 closure — strip live on every call and now honest about what it could not account for |
| TALLY-03 | 37-04 | `run_status` never set automatically from linked records | ✓ SATISFIED | Truth 3 — new applier read-only; AST guard green |
| UNUSED-01 | 37-04, 37-06, 37-07, 37-08, 37-10 | Unused awarded night visually distinct | ✓ SATISFIED | Truth 4 — unchanged code plus `37-UAT.md` tests 1-2 human-passed |
| GAPB-01 | 37-03, 37-07 | `claimed_dates()` counts every observation | ✓ SATISFIED | Truth 5 |

No orphaned requirements: REQUIREMENTS.md maps exactly these seven IDs to Phase 37 and each appears in at least one plan's `requirements` field. **Bookkeeping note (Advisory 2):** REQUIREMENTS.md still shows four of the seven as `Gaps Found` with unticked checkboxes — a blanket revert (commit `283f6e8`) after the 2026-09-20 verdict that the codebase now contradicts. It needs reconciling to `Complete` on phase completion.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `solsys_code/campaign_views.py` | 132 | `MAX_TABLE_PER_PAGE = 100` authorises ~300 queries for one anonymous GET on an unthrottled endpoint | ℹ️ Info (Advisory) | Deliberate, arithmetic-justified tradeoff (WR-07); bounded and test-pinned; lowering to 50 halves it. Not a phase-goal gap |
| `solsys_code/management/commands/backfill_lco_observations.py` | 303 | Bare `block.get('state') == 'COMPLETED'` outside `status_vocabulary` | ℹ️ Info | Carried forward unchanged for the third pass; it reads an LCO API block payload, not an `ObservationRecord.status`, and the file is outside every Phase 37 plan's `files_modified`. Follow-up quick task |
| `src/templates/tom_calendar/partials/event_form.html` | 211 | Segment labels not pluralised (`15 Unused awarded night`) | ℹ️ Info | Observed and accepted as non-blocking by the developer during UAT test 3. Follow-up quick task |
| `solsys_code/campaign_tally.py` | `campaign_rollup()` | Deliberate double allocation-event lookup per run on the cache-miss path | ℹ️ Info | Documented inline as the price of a single route to the figure; bounded by `assertNumQueries` |
| `campaign_views.py`, `campaign_tables.py` | 785, 442 | `PLACEHOLDER` string | ℹ️ Info (not a debt marker) | The Phase 22 domain term "tier-3 PLACEHOLDER Observatory". No `TODO`/`FIXME`/`XXX` in any file changed this wave — the debt-marker gate does not fire |

### Deferred Items

None actionable. Phase 37 is the final phase of the v2.4 milestone (ROADMAP.md holds phases 33-37), so no later phase exists to defer to. `deferred-items.md`'s one entry — an order-dependent Playwright flake in `test_bootstrap5_rendering.py`, confirmed non-reproducible on an immediate identical re-run and outside every Phase 37 plan's `files_modified` — remains a follow-up quick task, not a Phase 37 gap.

### Human Verification Required

None. The prior report's single human item (the product decision on a partially-unknown roll-up unused total) was decided by the developer as D-20, implemented by plan 37-10, and is verified closed above. The visual must-have (Truth 4) was human-passed in `37-UAT.md` tests 1 and 2 against code that is byte-unchanged since, so it needs no re-confirmation.

### Gaps Summary

No gaps. Both items the prior report left open are closed against the codebase, not against their SUMMARYs:

**G-37-5** was a structural ordering defect — the tally was computed before django-tables2 had decided which rows to render. The fix removes the prediction entirely rather than extending it to a third GET param: `get_table()` now reads the rows after `RequestConfig.configure()`. I re-ran the prior report's own measurement and the 10-occurrence counts under `?sort=` and `?per_page=50` are now 0, with set equality between rendered pks and tally keys asserted across a 12-case matrix that includes the page-2 tail, tied orderings, clamped and malformed page numbers, and zero/one-run campaigns.

**G-37-6** was a counting-semantics defect the prior report escalated rather than blocked. D-20's decision is implemented exactly as stated: the `≈` qualifier now fires on codes that *contributed*, and an unaccounted run is counted rather than absorbed as zero, so the strip reads `at least 2 (1 run not yet known)` over the rows it sits above instead of `≈2` over a row reading `not yet known`.

The code review's own critical — a cost cap that clamped the cheapest query string to the most expensive page — is fixed in the right direction and, unlike its predecessor, is now tested against a fixture large enough for the cap to be observable at all. The remaining seven review fixes are present in the source and regressed neither closure: 207 tests across the affected modules were re-run in this pass, all green, on top of the orchestrator's clean full suite and lint gates.

---

_Verified: 2026-09-21T18:41:52Z_
_Verifier: Claude (gsd-verifier)_
