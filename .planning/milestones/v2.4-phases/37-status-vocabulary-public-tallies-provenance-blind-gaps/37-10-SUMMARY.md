---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
plan: 10
subsystem: campaign-tally
tags: [campaign-tally, gap-closure, public-tally, unused-nights, tdd]

# Dependency graph
requires:
  - phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
    provides: 37-04's campaign_tally.py module and tally_segments() render contract, 37-05's
      Progress column wiring, 37-08's G-37-4 live-unused-figure fix on the same applier,
      37-09's get_table()/per_page cap on campaign_views.py and the notebook cells it
      appended
provides:
  - _apply_rollup_unused_fields() widened to track contributing-vs-attempted proposal
    codes separately, writing a new unused_unknown_runs roll-up key
  - tally_segments()'s unknown_runs field (0 on three segments, tally.get(...) on the
    unused segment) -- strip-only, inert for every per-run tally
  - campaignrun_table.html's roll-up strip fourth branch ("at least N (M runs not yet
    known)")
  - Unit coverage of the full contributing/attempted/blank-code matrix and the
    known-contributors-plus-unknown-runs-equals-total-runs invariant
  - Both paired docs (runbook + campaign_lifecycle_demo.ipynb) brought up to date with
    real committed output showing the partially-known roll-up line
affects: [any future contributor kind added to the unused-nights roll-up (the
  assumption-delta's accepted invariant test will catch an unaccounted-for one),
  37-VERIFICATION.md/37-UAT.md reconciliation of G-37-6 (owned by re-verification, not
  this plan)]

# Actuals (#2632)
actuals:
  tokens: 13260
  tasks: 3
  commits: 4

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A campaign-level applier that sums a per-contributor figure must also count what it
      could NOT include, one unit per contributor that failed to resolve -- never silently
      treat an unresolved contributor as a zero addend to the total."
    - "A render-contract field meant for exactly one consumer (the roll-up strip) is added
      to every dict the shared render function returns, defaulted inertly (0/false) on
      every other consumer, rather than branching the render function itself by caller."

key-files:
  created: []
  modified:
    - solsys_code/campaign_tally.py
    - src/templates/campaigns/campaignrun_table.html
    - solsys_code/tests/test_campaign_tally.py
    - solsys_code/tests/test_campaign_views.py
    - docs/runbooks/telescope_runs_calendar.rst
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb

key-decisions:
  - "unused_is_estimate now derives from whether a proposal-derived code actually
    CONTRIBUTED a number to the total (estimated_unused_nights() returned non-None), never
    from whether a code was merely attempted (bool(estimate_codes)) -- the single
    substitution D-20 required."
  - "unused_unknown_runs is counted PER RUN, not per distinct proposal code -- two runs
    sharing one unfetched code count as two unknown runs, matching the two Progress rows
    that each read 'not yet known' one-for-one. D-10's once-per-distinct-code rule
    continues to govern only the contributing estimate."
  - "A run with a BLANK proposal_code and no allocation events counts as unknown
    immediately (D-20's 'never absorbed as zero' sentence forces the wider reading, not
    just non-blank unfetched codes)."
  - "The new unknown_runs field lives on all four tally_segments() dicts (0 on
    observed/scheduled/expired-or-failed, tally.get('unused_unknown_runs', 0) on unused)
    rather than being a per-caller branch, so a per-run tally is inert by construction and
    only the roll-up strip template reads it."

patterns-established:
  - "Contributing-vs-attempted split for any future 'estimate from an external source'
    figure: collect the attempted set during the first pass, resolve which entries
    actually contributed during a second pass over that set only, and derive both the
    total and the estimate qualifier from the CONTRIBUTING subset, never the attempted
    one."

requirements-completed: [TALLY-01, TALLY-02, UNUSED-01]

coverage:
  - id: D1
    description: "On the developer's own two-run reproduction (one allocation run with two
      elapsed still-standing ALLOC: nights, one container run with a non-blank
      never-fetched proposal_code), the campaign roll-up strip reads '[U] at least 2 (1 run
      not yet known)' -- never '[U] ≈2' -- and a third run with a fetched proposal code
      added makes the two signals (the approximation qualifier and the unknown-run count)
      co-occur without cancelling."
    requirement: "TALLY-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRollup::test_an_unknown_contributor_is_never_absorbed_as_zero_into_the_strip_total"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRollup::test_the_approximation_qualifier_and_the_unknown_run_count_can_co_occur"
        status: pass
      - kind: integration
        ref: "python manage.py test solsys_code.tests.test_campaign_tally solsys_code.tests.test_campaign_views"
        status: pass
      - kind: integration
        ref: "python manage.py test solsys_code --exclude-tag=ephemeris_segfault"
        status: pass
    human_judgment: false
  - id: D2
    description: "Every combination of exact/contributing-estimate/attempted-but-unfetched/
      blank-code runs has a pinned expectation for both the total and the unknown-run
      count; the cache still resets the new key to 0; and an accepted invariant test
      (known contributors + unused_unknown_runs == rollup['runs']) would go red if a future
      contributor kind were added without being accounted for on either side."
    requirement: "TALLY-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestRollupPartiallyKnownUnusedTotal"
        status: pass
      - kind: integration
        ref: "python manage.py test solsys_code.tests.test_campaign_tally"
        status: pass
    human_judgment: false
  - id: D3
    description: "Both paired docs tell the truth about the new strip: the runbook's
      public-tally section explains the roll-up's partially-known total and the
      contribution-based estimate qualifier, and campaign_lifecycle_demo.ipynb's own
      _segment_text() reimplementation of the strip has the same four branches, with
      committed output showing the campaign's real partially-known roll-up line."
    requirement: "UNUSED-01"
    verification:
      - kind: e2e
        ref: "jupyter nbconvert --to notebook --execute --inplace campaign_lifecycle_demo.ipynb"
        status: pass
      - kind: other
        ref: "sed -n '/What does a run.s or a campaign.s public tally show?/,/How do I find nights/p' docs/runbooks/telescope_runs_calendar.rst | grep -c 'at least'/'not yet known'"
        status: pass
    human_judgment: false

# Metrics
duration: ~50min
completed: 2026-09-21
status: complete
---

# Phase 37 Plan 10: Contributing vs Attempted, and the Strip's Own Partially-Known Total Summary

**Closed G-37-6 by splitting the campaign roll-up's estimate qualifier from "a proposal code was attempted" to "a proposal code actually contributed a number," and adding a per-run `unused_unknown_runs` count so an incomplete total reads as `at least N (M runs not yet known)` instead of silently absorbing an unknown contributor as zero.**

## Performance

- **Duration:** ~50 min
- **Started:** 2026-09-21T~16:35Z
- **Completed:** 2026-09-21T~17:25Z
- **Tasks:** 3 (all completed)
- **Files modified:** 6

## Accomplishments

- Widened `_apply_rollup_unused_fields()` in `solsys_code/campaign_tally.py` so it tracks
  which proposal codes were merely attempted versus which ones actually contributed a
  number to the total, and derives `unused_is_estimate` from the CONTRIBUTING subset
  instead of `bool(estimate_codes)` (the exact mis-wiring G-37-6 was).
- Added a new `unused_unknown_runs` roll-up key: the count of runs whose own unused
  figure is not yet known (no allocation events, and either a blank `proposal_code` or a
  code with no stored `ProposalTimeAllocation`), counted per RUN rather than per distinct
  code so it matches the Progress rows one-for-one.
- Gave `campaignrun_table.html`'s roll-up strip a fourth branch, inserted second, that
  renders `at least {count} ({n} run(s) not yet known)` -- with the `&approx;` entity only
  when the total's estimate half genuinely contributed -- so an incomplete total is never
  shown as if it were complete.
- Proved the fix RED-then-GREEN end-to-end on the developer's own two-run reproduction
  (`TestCampaignRollup` in `test_campaign_views.py`), then pinned the full
  contributing/attempted/blank-code matrix, the cache-reset contract, the pending-review
  exclusion, and the assumption-delta's accepted accounting invariant with 12 new unit
  cases in `TestRollupPartiallyKnownUnusedTotal` (`test_campaign_tally.py`) -- 14 new test
  methods total across the two modules, with no pre-existing expected value edited.
- Updated both paired docs: the runbook's "What does a run's or a campaign's public tally
  show?" section now explains the strip's own partially-known rendering and the
  contribution-based estimate qualifier, and `campaign_lifecycle_demo.ipynb` cell 42's
  `_segment_text()` reimplementation was brought to the same four branches and regenerated
  with real committed output showing `[U] at least 3 (4 runs not yet known)` on the
  notebook's own five-run campaign.

## Task Commits

Each task was committed atomically:

1. **Task 1: Contributing vs attempted, and the strip's own not-fully-known rendering** -
   `93daff0` (test, RED) / `5a7a64f` (feat, GREEN)
2. **Task 2: Unit coverage for the contributing/attempted matrix, the blank-code case, the
   cache contract and the accounting invariant** - `553213d` (test)
3. **Task 3: Both paired docs — the runbook's public-tally section and the notebook's
   strip reimplementation — and the phase gates** - `c84af6d` (docs)

**Plan metadata:** committed alongside this SUMMARY (see below)

_Note: Task 1 was TDD (RED verified against the unmodified module -- `unused_is_estimate`
read True with nothing behind it, `unused_unknown_runs` did not exist -- then made GREEN
by the applier/template change); Task 2 followed the same test-only discipline within one
commit; `workflow.tdd_mode` is `false` for this project, so `gsd_run check
tdd-red-evidence` was not invoked, but the RED failure was verified manually against the
stated reason before any implementation code was touched._

## Files Created/Modified

- `solsys_code/campaign_tally.py` — `_apply_rollup_unused_fields()` widened with the
  contributing/attempted split and the new `unused_unknown_runs` count;
  `campaign_rollup()`'s defaults dict and `_without_unused_fields()` both carry the new
  key; `tally_segments()` reads it on the unused segment and sets it to `0` on the other
  three.
- `src/templates/campaigns/campaignrun_table.html` — roll-up strip's segment conditional
  widened from three branches to four.
- `solsys_code/tests/test_campaign_tally.py` — new `TestRollupPartiallyKnownUnusedTotal`
  (12 cases); one added assertion in `test_cached_value_never_carries_a_computed_unused_figure`.
- `solsys_code/tests/test_campaign_views.py` — two new cases added to the existing
  `TestCampaignRollup`.
- `docs/runbooks/telescope_runs_calendar.rst` — the public-tally section's third bullet
  amended and one new paragraph added.
- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` — cell 42's `_segment_text()`
  helper and roll-up block extended; cell 41's markdown extended; regenerated with output.

## Decisions Made

See `key-decisions` in the frontmatter above. The most consequential: `unused_is_estimate`
now means "an estimate contributed," never "an estimate was looked for," and the unknown
count is per run (not per distinct proposal code) so it corresponds one-for-one with the
Progress rows a reader actually sees.

## Deviations from Plan

None - plan executed exactly as written. One in-flight self-correction during Task 3 is
worth recording as process, not a deviation: the class-level and one test-level docstring
in `TestRollupPartiallyKnownUnusedTotal` originally named the private helpers
(`_apply_rollup_unused_fields()`, `_rollup_runs()`, `_without_unused_fields()`) in prose to
explain what the class deliberately does NOT call; the plan's own verify script strips only
`#`-comment lines, not docstring text, so those literal names tripped the
"no private helper reaches" gate. Reworded both docstrings to describe the discipline
without naming the exact private symbols, re-ran the gate to confirm `private_helper_reaches=0`,
and re-ran the full module to confirm no test behavior changed — caught and fixed before
the Task 2 commit, not after.

## Issues Encountered

None. The notebook regeneration needed one follow-up edit: the first render of a new
code comment estimated "three of five roll-up runs" render not-yet-known, but the real
fixture (a fifth, unnamed run beyond the four the cell's per-run loop prints by name) put
the true count at four; corrected the comment to state a floor ("at least three") rather
than an exact count, re-ran `jupyter nbconvert --execute --inplace`, and confirmed the
regenerated output (`[U] at least 3 (4 runs not yet known)`) still satisfies both
`unused_unknown_runs >= 3` and `< rollup['runs']`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- G-37-6 is closed: the campaign roll-up strip now reports what it actually knows, on both
  the developer's own reproduction and a real five-run campaign in the paired notebook.
- Reconciling G-37-6 as closed in `37-VERIFICATION.md`/`37-UAT.md` is explicitly out of
  scope for this plan (a prohibition it must not violate) and belongs to re-verification.
- `solsys_code/campaign_views.py` was untouched, as scoped — it remains plan 37-09's this
  wave.
- This was the last plan in phase 37's wave structure per the plan's own frontmatter
  (`depends_on: ["37-04", "37-05", "37-08", "37-09"]`, wave 8); phase-level verification is
  the next step.

---
*Phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps*
*Completed: 2026-09-21*

## Self-Check: PASSED

- FOUND: solsys_code/campaign_tally.py
- FOUND: src/templates/campaigns/campaignrun_table.html
- FOUND: solsys_code/tests/test_campaign_tally.py
- FOUND: solsys_code/tests/test_campaign_views.py
- FOUND: docs/runbooks/telescope_runs_calendar.rst
- FOUND: docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
- FOUND commit: 93daff0 (Task 1 RED)
- FOUND commit: 5a7a64f (Task 1 GREEN)
- FOUND commit: 553213d (Task 2)
- FOUND commit: c84af6d (Task 3)
- Re-ran all acceptance criteria and plan-level `<verification>` commands: all pass
  (172 tests across `test_campaign_tally`/`test_campaign_views`, full `solsys_code` suite
  green with `--exclude-tag=ephemeris_segfault`, both ruff gates clean, `leaked_into: []`,
  notebook executes with real committed output showing the partially-known roll-up line).
