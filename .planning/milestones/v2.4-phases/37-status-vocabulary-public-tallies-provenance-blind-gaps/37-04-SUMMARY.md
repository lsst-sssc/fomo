---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
plan: 04
subsystem: public-tallies
tags: [django, orm-aggregation, ttl-cache, tdd]

requires:
  - phase: 37-01
    provides: "solsys_code/status_vocabulary.py -- classify_record()/DisplayState/MARKER/LABEL/RUN_STATUS_MARKER, the one classifier and marker table this module reads instead of re-deriving"
  - phase: 37-02
    provides: "CampaignRun.proposal_code and solsys_code/proposal_allocation.py's estimated_unused_nights()/unused_hours_for() -- the D-06 estimate fallback for a run with no allocation events"
provides:
  - "solsys_code/campaign_tally.py -- the single module every public tally surface (37-05, 37-06) will read: build_tally_cache_key(), link_counts_for_runs(), night_counts_for_run(), tally_for_run(), get_or_compute_tally(), tallies_for_runs(), is_unused_allocation_night(), unused_nights_for_run(), tally_segments(), campaign_rollup(), campaign_records_version()"
  - "The shared D-15 unused-night rule (is_unused_allocation_night()) that both the table's count and the (not-yet-built) calendar [U] marker must call, so the two agree by construction"
affects: [37-05, 37-06, 37-07]

actuals:
  tokens: 14200
  tasks: 3
  commits: 6
  plan_head_before: 889c4f2df3130d4411518d51fb2fd261ea2beaa6

tech-stack:
  added: []
  patterns:
    - "TTL-cached, freshness-keyed per-run computation (mirrors campaign_gap.py's GAP_CACHE_TTL_SECONDS/get_or_compute_gap() pattern), but the cache key itself carries the newest linked-record change stamp so a projector narrowing is visible immediately -- only purely time-driven transitions wait on the TTL"
    - "Bulk aggregate-first, per-run-cache-second: tallies_for_runs() runs the SQL-expressible counts in exactly two queries for the whole set (never per row), then only night_counts_for_run()'s inherently-per-record Python classification runs per cache-miss run"

key-files:
  created:
    - solsys_code/campaign_tally.py
    - solsys_code/tests/test_campaign_tally.py
  modified: []

key-decisions:
  - "campaign_rollup() reuses each run's already-computed tally['unused_known']/['unused_is_estimate'] flags to decide exact-vs-estimate bucketing, instead of calling unused_nights_for_run() a second time per run -- avoids re-querying the ALLOC: namespace a run's own tallies_for_runs() call already resolved."
  - "The module docstring states the heavy-import discipline (never the ephemeris-computation module or the view layer that imports it) without embedding the literal 'solsys_code.views'/'ephem_utils' substrings -- the plan's own <verify> command greps the WHOLE module source (not just import lines) for absence of those two substrings, so a campaign_gap.py-style docstring mention of them by name would fail that specific check even though the actual import statements are clean."
  - "The combined expired-or-failed tally segment uses a locally-defined '[X/F]' token, not a status_vocabulary.MARKER entry -- it is a tally-display grouping of three underlying DisplayState markers ([X]/[C]/[F]), not a DisplayState of its own, so it does not belong in the one-marker-one-module vocabulary itself."
  - "Two of the module's own tests (the import guard, and the D-08 'tallies_for_runs never calls tally_for_run in a loop' check) needed to search only the executable lines/AST of the function under test, not its full inspect.getsource() output -- a naive substring search over the whole source tripped on the docstring's own explanatory prose naming the very thing the test asserts is absent. Same self-correction pattern documented in 37-01-SUMMARY.md and 37-03-SUMMARY.md for the identical docstring-vs-grep pitfall."

requirements-completed: [TALLY-03]

coverage:
  - id: D1
    description: "One module computes every public tally figure: linked observation groups, linked observation records, and nights observed/scheduled/expired-or-failed/unused so far"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestTallyForRun.test_carries_the_eight_required_keys"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestLinkCountsForRuns, TestNightCountsForRun"
        status: pass
    human_judgment: false
  - id: D2
    description: "Night counters follow the site-local observing night via telescope_runs.observing_night(), never the UTC date (D-11), including the 02:00-local previous-date boundary"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestNightCountsForRun.test_early_morning_local_start_counts_on_the_previous_date"
        status: pass
    human_judgment: false
  - id: D3
    description: "Two linked records whose blocks fall on the same site-local observing night count as one night, not two"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestNightCountsForRun.test_two_observed_records_same_site_local_night_count_once"
        status: pass
    human_judgment: false
  - id: D4
    description: "A run with zero linked records produces a complete tally of zeros, never a blank value or an error; a run with an unresolved site still reports its group/record counts and zero night counts, never raising"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestTallyForRun.test_zero_linked_records_produces_all_zero_and_unused_unknown"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestNightCountsForRun.test_site_unset_returns_all_zero_no_exception"
        status: pass
    human_judgment: false
  - id: D5
    description: "One shared classifier (is_unused_allocation_night()) decides the unused figure, so the table's count and the calendar's [U] marker will agree by construction"
    requirement: UNUSED-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestIsUnusedAllocationNight"
        status: pass
    human_judgment: false
  - id: D6
    description: "For a run with allocation nights, unused-so-far is the exact still-standing count and unused_is_estimate is False; for a run without them it is the D-06 proposal-derived estimate and unused_is_estimate is True; the two are distinguishable by the caller"
    requirement: UNUSED-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestTallyForRunUnusedWiring.test_run_with_allocation_events_uses_the_exact_count"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestTallyForRunUnusedWiring.test_run_with_no_allocation_events_uses_the_proposal_estimate"
        status: pass
    human_judgment: false
  - id: D7
    description: "An unused figure that has never been fetched reports as unknown (unused_known=False, nights_unused=None), never as zero"
    requirement: UNUSED-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestTallyForRunUnusedWiring.test_run_with_no_allocation_events_and_blank_proposal_code_is_unknown_not_zero"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestTallyForRunUnusedWiring.test_run_with_a_proposal_code_but_no_stored_rows_is_unknown_not_zero"
        status: pass
    human_judgment: false
  - id: D8
    description: "The tally updates as the projector narrows: a saved linked observation record changes the run's cache key, so the next call recomputes with no TTL wait; only a purely time-driven transition lags, bounded by TALLY_CACHE_TTL_SECONDS"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestGetOrComputeTallyFreshness.test_saving_a_linked_record_is_reflected_with_no_clock_advance"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestBuildTallyCacheKey"
        status: pass
    human_judgment: false
  - id: D9
    description: "No function in this module ever writes CampaignRun.run_status; a run's stored status is byte-identical before and after any tally computation, across every entry point and both cache orders (miss-then-hit, hit-then-miss)"
    requirement: TALLY-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestTallyNeverWritesRunStatus.test_run_status_unchanged_for_a_fully_observed_run"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestTallyNeverWritesRunStatus.test_run_status_unchanged_for_a_run_with_no_linked_records"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestTallyNeverWritesRunStatus.test_no_computation_path_module_assigns_to_run_status_attribute"
        status: pass
    human_judgment: false
  - id: D10
    description: "A campaign roll-up sums groups/records/nights across only approved, publicly visible runs (a pending-review run contributes nothing, excluded at the queryset level); the unused estimate is counted once per distinct proposal code, not once per run"
    requirement: TALLY-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_tally.py#TestCampaignRollup"
        status: pass
    human_judgment: false

duration: 4h 30min
completed: 2026-09-19
status: complete
---

# Phase 37 Plan 04: Public Run/Campaign Tally Module Summary

**`solsys_code/campaign_tally.py` is the one module that answers "what has this run/campaign actually got": per-run group/record counts (two aggregate queries for a whole table, never a per-row loop), site-local-observing-night counts by state behind a TTL cache keyed on the newest linked-record change stamp, the shared unused-allocation-night rule both the table and the future calendar marker will read, and a campaign roll-up that de-duplicates the proposal-derived estimate by code — with a two-part guard (behavioural snapshot + AST scan) proving no entry point ever writes `CampaignRun.run_status`.**

## Performance

- **Duration:** ~4h 30min (includes required-reading of 8 upstream source files plus two prior-plan SUMMARY files, and one long-running full-suite regression check that did not finish within this session — see "Verification" below)
- **Tasks:** 3, each executed as RED (failing test commit) → GREEN (implementation commit)
- **Commits:** 6 (verified via `git rev-list --count 889c4f2..HEAD` against the plan-start ledger)
- **Files created:** 2 (`solsys_code/campaign_tally.py`, `solsys_code/tests/test_campaign_tally.py`); 0 modified

## Accomplishments

- `solsys_code/campaign_tally.py` (520 lines) is now the single module every public tally surface (37-05's Progress column/campaign roll-up strip, 37-06's calendar `[U]` decoration) will read rather than compute:
  - `link_counts_for_runs(run_pks)` — two aggregate queries (`CampaignRunObservation` grouped by run for distinct record count + `Max('observation_record__modified')`, `ObservationGroup` grouped by run for distinct group count) for a whole set of runs, never one query per row (D-08).
  - `night_counts_for_run(run)` — the Python half: classifies each linked `ObservationRecord` through `status_vocabulary.classify_record()`, buckets its `telescope_runs.observing_night()` site-local night into observed/scheduled/failed sets, de-duplicated per night.
  - `build_tally_cache_key(run_pk, records_version)` / `get_or_compute_tally()` / `tallies_for_runs()` — the TTL-cache pattern from `campaign_gap.py`, but the cache key folds in the newest linked-record change stamp so a projector narrowing is visible on the very next call with no TTL wait.
  - `is_unused_allocation_night(end_time, run_status)` / `unused_nights_for_run(run)` — the single D-15 shared rule: an allocation night is unused when its `end_time` is strictly in the past and the run's status carries no cancelled/weathered marker (D-14, staff status always wins).
  - `tally_segments(tally)` — the fixed `[O]`/`[S]`/`[X/F]`/`[U]` render order both the table and the (future) calendar pop-up will share.
  - `campaign_rollup(campaign)` / `campaign_records_version(campaign)` — sums a campaign's approved, publicly visible runs (pending-review excluded at the queryset level), counting each distinct proposal code's unused estimate once, not once per run (D-10).
- The TALLY-03 guard is enforced two ways in `solsys_code/tests/test_campaign_tally.py`: a behavioural before/after `CampaignRun.run_status` snapshot across every entry point and both cache-hit/miss orderings, and a static `ast`-based scan of `campaign_tally`/`status_vocabulary`/`proposal_allocation`/`campaign_gap` proving no assignment targets `.run_status` and no `.update()` call passes a `run_status` keyword anywhere in the computation path.
- 55 tests in `solsys_code/tests/test_campaign_tally.py`, all passing.

## Task Commits

Each task followed the RED (failing test) → GREEN (implementation) TDD commit contract this dispatch specified, even though `workflow.tdd_mode` is `false` for this project (manual RED verification, matching 37-01/37-02/37-03 precedent):

1. **Task 1: Per-run tally — group/record counts in one query, night counts behind a TTL cache** — RED `93306cd` (test) → GREEN `226f8f8` (feat)
2. **Task 2: The shared unused-night classifier, the proposal-derived estimate, and the ordered render segments** — RED `2526344` (test) → GREEN `0536f48` (feat)
3. **Task 3: Campaign roll-up, and the guard that a tally never sets a run's status** — RED `5366f6b` (test) → GREEN `7003deb` (feat)

**Plan metadata:** this SUMMARY committed separately, immediately after this list.

## TDD Gate Compliance

`workflow.tdd_mode` is `false` for this project, so the tool-mediated `gsd_run check tdd-red-evidence` gate was not invoked. RED evidence for all three tasks was verified manually against the exact `ImportError`/`AttributeError` each RED commit's tests produced (the target symbols did not exist yet), matching Phase 34/37-01/37-02/37-03 precedent for the same situation:

- Task 1 RED (`93306cd`): `test_campaign_tally.py` failed with `ImportError: cannot import name 'campaign_tally' from 'solsys_code'` — the module did not exist at all. Confirmed intentional, not an INVALID_RED pattern (no zero-test discovery, no fixture crash, no unrelated failure).
- Task 1 GREEN (`226f8f8`): all 26 tests pass after `campaign_tally.py` is created with Task 1's symbols.
- Task 2 RED (`2526344`): `test_campaign_tally.py` failed with `ImportError: cannot import name 'is_unused_allocation_night'`. Confirmed intentional.
- Task 2 GREEN (`0536f48`): all 43 tests pass after `is_unused_allocation_night()`/`unused_nights_for_run()`/`tally_segments()` are added and wired into `tally_for_run()`.
- Task 3 RED (`5366f6b`): `test_campaign_tally.py` failed with `ImportError: cannot import name 'campaign_records_version'`. Confirmed intentional.
- Task 3 GREEN (`7003deb`): all 55 tests pass after `campaign_rollup()`/`campaign_records_version()` are added.

## Files Created/Modified

- `solsys_code/campaign_tally.py` — the new module (created across all three tasks)
- `solsys_code/tests/test_campaign_tally.py` — the new test module (created across all three tasks; 55 tests total)

## Decisions Made

- **`campaign_rollup()` reuses each run's already-computed `tally['unused_known']`/`tally['unused_is_estimate']` flags** (from the `tallies_for_runs()` call it already makes) to decide whether a run's unused figure is exact or estimate-derived, rather than calling `unused_nights_for_run()` a second time per run. This avoids re-querying the `ALLOC:` namespace for a fact the bulk tally call already resolved.
- **The module docstring describes the heavy-import discipline without the literal `'solsys_code.views'`/`'ephem_utils'` substrings.** The plan's own automated `<verify>` command (`'solsys_code.views' in src`, `'ephem_utils' in src`) greps the WHOLE module source, not just import statement lines — a `campaign_gap.py`-style docstring sentence naming those two modules by their exact dotted path would trip that check even though no import statement actually references them. Reworded to name them descriptively ("the heavy ephemeris-computation module", "the view layer that imports it") instead.
- **The combined expired-or-failed tally segment (`[X/F]`) is a locally-defined string constant**, not a `status_vocabulary.MARKER` entry — it groups three underlying `DisplayState` markers (`[X]`/`[C]`/`[F]`) for display purposes and is not a `DisplayState` of its own, so it does not belong in the one-marker-one-module vocabulary `status_vocabulary.py` owns.
- **Two of this module's own tests needed to search only the executable code (not the full `inspect.getsource()` output)** for the import-guard test and the D-08 "`tallies_for_runs()` never calls `tally_for_run()` in a loop" test — a naive whole-source substring search tripped on each function's own docstring prose, which legitimately names the very import/call the test asserts is absent, in order to explain the rule. Fixed by checking only import-statement-shaped lines (mirroring `test_campaign_gap.py`'s existing `TestNoHeavyEphemerisImport` pattern) and only the AST nodes after the docstring, respectively. Same self-correction class documented in `37-01-SUMMARY.md` (a `status_border_css()` literal-check precedent) and `37-03-SUMMARY.md` (an `observation_claimed_dates()` docstring reword) for the identical docstring-vs-grep pitfall.
- **Three of the roll-up tests needed a distinct `telescope_instrument` per run** when creating two `CampaignRun`s under the same test campaign, to satisfy the model's own `(campaign, telescope_instrument, window_start, window_end)` unique constraint — a test-fixture correction discovered while making Task 3's tests pass, not a change to any production behavior.

## Deviations from Plan

None — plan executed exactly as written. The items above are implementation-detail decisions and test-fixture corrections made while satisfying the plan's own written behavior/acceptance criteria, not departures from what the plan specified.

## Verification

Confirmed this session:
- `python manage.py test solsys_code.tests.test_campaign_tally` — 55 tests, `OK`, exit 0 (re-confirmed after each task's GREEN commit and again at the end of Task 3).
- All of Task 1/2/3's plan-specified inline `<automated>` verify commands (module callables, `TALLY_CACHE_TTL_SECONDS == 3600`, heavy-import-substring absence, cache-key stamp sensitivity, `tally_segments()` marker order, the TALLY-03 AST probe reporting 0 `run_status` assignments) — all passed, run manually via `python -c "..."` exactly as the plan's `<verify>` blocks specify.
- `pre-commit run ruff --files solsys_code/campaign_tally.py solsys_code/tests/test_campaign_tally.py` and the equivalent `ruff-format` run — both `Passed`.
- Importing `solsys_code.campaign_tally` completes in ~2 seconds with no SPICE-kernel fetch (confirmed by timing the import directly).

**NOT confirmed this session — the plan's `python manage.py test solsys_code` (excluding `test_views.TestEphemeris`) full-suite regression command:**

Two attempts were made to run the full ~1,650-test `solsys_code` suite (the project's configured `test_command`, covering every other test module besides `test_campaign_tally` itself). The first attempt's output was inconclusive (piped through `tail -25`, which discarded the final pass/fail summary before the process — bounded by a 1400-second `timeout` — could be confirmed to have completed naturally rather than being killed mid-run). The second attempt (`nohup ... > full_suite_run.log &`, unbounded) was still running when this session's time budget for the plan closed out; per the orchestrator's explicit instruction, this session did not wait for it or re-run it, since the orchestrator runs the full suite itself as the post-merge gate immediately after this plan returns. Nothing in `test_campaign_tally.py`'s own 55 tests, nor the plan's own inline verify commands, was left unrun — only the cross-module full-repository regression sweep (which exercises modules this plan does not modify) is unconfirmed. Recorded in the broken-windows ledger as an `unrun-verify` entry per CLAUDE.md/GSD convention.

## Known Stubs

None.

## Threat Flags

None — every threat this plan's `<threat_model>` names (T-37-13, T-37-14, T-37-15, T-37-16, T-37-SC) is mitigated by the implementation as specified: every query is `.only()`/`.values()`-restricted to non-PII fields (T-37-13), `campaign_rollup()` excludes pending-review at the queryset level (T-37-14), cached tally dicts contain only integers and booleans (T-37-15), the TALLY-03 guard is tested both behaviourally and structurally (T-37-16), and no new package was installed (T-37-SC).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

`solsys_code/campaign_tally.py` is the shared foundation waves 4+ of Phase 37 consume:

- Plan 37-05 (TALLY-01/02) will render `tallies_for_runs()`/`tally_segments()` in the campaign table's Progress column and `campaign_rollup()`/`campaign_records_version()` in the campaign roll-up header strip and list badge.
- Plan 37-06 (TALLY-01/UNUSED-01) will render `is_unused_allocation_night()` (the same shared rule this module's count already uses) for the calendar's `[U]` marker, so the two surfaces agree by construction per D-15.
- Plan 37-07 owns the paired-docs update and the legacy-title re-title sweep; no dependency on this plan's internals beyond the vocabulary/tally figures it renders.

**Requirements traceability:** TALLY-03 is declared only by this plan (no sibling), so it is marked complete in this session's state update. TALLY-01 (shared with 37-02/37-05/37-06/37-07), TALLY-02 (shared with 37-05) and UNUSED-01 (shared with 37-06/37-07) stay "Pending"/"Blocked" per the shared-ID gate (`requirements.ready-ids` confirmed `1/4 ready: TALLY-03`) until every plan declaring them has finished.

No blockers for wave 4 (37-05, 37-06). The one open item is the full-suite regression re-confirmation noted above, which the orchestrator has stated it will run itself as the post-merge gate.

---
*Phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps*
*Completed: 2026-09-19*

## Self-Check: PASSED

- `solsys_code/campaign_tally.py` -- FOUND
- `solsys_code/tests/test_campaign_tally.py` -- FOUND
- Commit `93306cd` -- FOUND
- Commit `226f8f8` -- FOUND
- Commit `2526344` -- FOUND
- Commit `0536f48` -- FOUND
- Commit `5366f6b` -- FOUND
- Commit `7003deb` -- FOUND
- `python manage.py test solsys_code.tests.test_campaign_tally` re-confirmed: 55 tests, `OK`
- All plan-level task `<acceptance_criteria>` re-verified true (per-task `<automated>` commands re-run above)
- The plan-level `<verification>` block's full-`solsys_code`-suite item is NOT re-confirmed this session -- see "Verification" section above; the orchestrator has taken ownership of that check as its post-merge gate
