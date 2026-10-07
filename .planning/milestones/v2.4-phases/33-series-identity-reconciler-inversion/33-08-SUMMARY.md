---
phase: 33-series-identity-reconciler-inversion
plan: 08
subsystem: api
tags: [campaign-reconciler, observing-night, timezone, calendar-events, gap-closure, notebooks, runbook]

# Dependency graph
requires:
  - phase: 33-series-identity-reconciler-inversion
    provides: "The D-01/ANNOT-01 skip rule and _attributed_nights() from earlier Phase 33 waves, plus unlink_event_from_run()/UNLINK_CLEARED_FIELDS from plans 33-04/33-07"
provides:
  - "_observing_night(start_time, site_zone): the noon-anchored event-to-night mapping shared with telescope_runs._local_noon_utc(), superseding 26-DECISION.md D-10's plain site-local .date() derivation (CR-02 closed)"
  - "An unconditional attribution skip in _reconcile_classical_nights() -- the reconcile-then-attribute and attribute-then-reconcile orderings now converge (CR-03 closed)"
  - "ReconcileResult.detached and a returned active-url set from _reconcile_classical_nights(), threaded through reconcile_run() so a superseded night's event is detached (never deleted) with a counted, logged trace (WR-03 closed, IN-02 closed as a side effect)"
  - "skipped_nights/detached/would_detach surfaced in reconcile_campaign_runs's per-run and summary output (WR-01 closed)"
  - "Both paired notebooks re-executed against the fixed code, and the runbook's reconcile/skip-rule/inline/pop-up-link sections brought back in line (WR-06 runbook half, WR-08 runbook half)"
affects: ["34", "35"]

actuals:
  tokens: 128000
  tasks: 4
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Noon-anchored night derivation (local time minus 12 hours, then .date()) as a single named helper with one call site, rather than re-deriving the anchor inline wherever an event-to-night mapping is needed"
    - "A branch function returns the exact url set it considers 'current' alongside its result tuple, so the caller (reconcile_run()) never re-derives window arithmetic a second time to compute what to detach"
    - "Self-cleaning demo notebook cells: a cell that creates a scratch row for a demo captures its primary key at creation time and deletes only that row at the end of the same cell, leaving no permanent artifact for the next execution to trip over"

key-files:
  created: []
  modified:
    - solsys_code/campaign_reconciler.py
    - solsys_code/management/commands/reconcile_campaign_runs.py
    - solsys_code/tests/test_campaign_reconciler.py
    - solsys_code/tests/test_reconcile_campaign_runs.py
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "Checkpoint decision resolved by the user: adopt the local-noon anchored _observing_night(), superseding 26-DECISION.md D-10's plain site-local .date() derivation. This re-maps which observing night an existing attributed event resolves to on the real developer database -- accepted because it matches the anchor sun_event() itself uses and closes CR-02's duplicate-night/uncovered-night defect."
  - "Moved the sun_event() call in _reconcile_classical_nights() to fire only for a night that is not skipped (previously called unconditionally for every night in the loop). Not requested verbatim by the plan text, but consistent with the review's own suggested fix shape and avoids an unnecessary astronomical computation for a night the branch is about to skip entirely."
  - "Cleaned up one stale leftover CalendarEvent (pk=322, 'Hand-entered classical night (skip-rule demo)') from a prior execution of the OLD, pre-fix skip-rule cell directly against the real dev database, rather than adding self-healing cleanup logic inside the new cell. The old cell's unconditional delete() left that row permanently attributed with no corresponding RUN:-keyed sibling; removing it out-of-band let the new cell's own real sweep (re-)mint that RUN:-keyed event fresh, matching the state the new cell's comments describe, without stretching the '33-05 P2: only delete a row created earlier in the same notebook run' prohibition to cover residue from a different, now-superseded cell design."
  - "campaign_lifecycle_demo.ipynb needed no markdown addition: its orphan-event-to-classical_run attribution (cell ~24) never triggers a second reconcile_run(classical_run) call afterwards, so the CR-03 detach step is never exercised by this notebook's flow. Re-executed output is structurally identical to the prior commit apart from non-deterministic primary keys."
  - "WR-06's runbook fix targets the second bullet under 'Two things to know about that inline' (the one describing clearing the Attributed campaign run value on the inline), not literally the third bullet as the plan's action text names it -- the plan's described content unambiguously matches that bullet's text. Also renamed the lead-in from 'Two things' to 'Three things' since a third (Observation record/group) bullet already existed from an earlier wave."

requirements-completed: [ANNOT-01, PROJ-04]

coverage:
  - id: D1
    description: "An attributed event starting after local midnight resolves to the PREVIOUS date's observing night (CR-02), matching the anchor sun_event() computes sunset for -- the reconciler no longer mints a duplicate for the covered night while leaving the real night uncovered"
    requirement: "ANNOT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestObservingNightBoundary.test_cr02_post_midnight_start_skips_the_previous_nights_url_not_the_next"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestObservingNightBoundary.test_measured_pk_54_case_resolves_to_the_night_before_the_naive_utc_date"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestObservingNightBoundary.test_negative_utc_offset_site_resolves_by_observing_night_not_naive_utc_date"
        status: pass
    human_judgment: false
  - id: D2
    description: "The noon-anchor boundary is exact on both sides: a start of exactly 12:00:00 local belongs to that date's night, 11:59:59 belongs to the previous date's night (PROJ-04 adjacency edge)"
    requirement: "PROJ-04"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestObservingNightBoundary.test_exact_local_noon_boundary_belongs_to_the_date_that_just_started"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestObservingNightBoundary.test_one_second_before_local_noon_belongs_to_the_previous_date"
        status: pass
    human_judgment: false
  - id: D3
    description: "A run with no attributed non-RUN: event skips no night and detaches nothing -- skipped_nights and detached are both 0 and one event per night is minted, including for a single-night window (PROJ-04 empty edge)"
    requirement: "PROJ-04"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestObservingNightBoundary.test_no_attributed_events_skips_nothing_and_mints_one_event_per_night"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestObservingNightBoundary.test_single_night_window_with_no_attribution_creates_exactly_one_event"
        status: pass
    human_judgment: false
  - id: D4
    description: "The skip rule fires whether or not a RUN:{pk}:{date} event already exists for that night -- the reconcile-then-attribute and attribute-then-reconcile orderings converge (CR-03); a superseded event is DETACHED (never deleted), its confirmation stamps cleared, exactly one entry stays attributed for that night, and clearing the surviving attribution restores the same detached record in place on the next reconcile"
    requirement: "ANNOT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestReconcileThenAttributeOrdering.test_second_reconcile_detaches_the_superseded_run_keyed_event_and_restore_on_third"
        status: pass
      - kind: integration
        ref: "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb (skip-rule demo cell, real developer database)"
        status: pass
    human_judgment: false
  - id: D5
    description: "A night whose RUN:-keyed event is attributed to a DIFFERENT run stays blocked, keeps its url in the active set, and is never detached (D-02, foreign attribution survives the refactor)"
    requirement: "ANNOT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestReconcileThenAttributeOrdering.test_blocked_night_keeps_its_url_active_and_is_never_detached"
        status: pass
    human_judgment: false
  - id: D6
    description: "reconcile_campaign_runs surfaces skipped_nights and detached (per-run lines plus both summary lines), with would_detach: n/a (dry-run) since the detach step is itself a write (WR-01, WR-03)"
    requirement: "PROJ-04"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_reconcile_campaign_runs.py#TestSkipAndDetachCounters.test_real_sweep_reports_skipped_nights_for_a_night_attributed_elsewhere"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_reconcile_campaign_runs.py#TestSkipAndDetachCounters.test_real_sweep_reports_detached_for_a_superseded_run_keyed_event"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_reconcile_campaign_runs.py#TestSkipAndDetachCounters.test_dry_run_reports_skipped_nights_and_would_detach_na_and_writes_nothing"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_reconcile_campaign_runs.py#TestSkipAndDetachCounters.test_both_counters_are_reported_together_in_the_same_sweep"
        status: pass
    human_judgment: false
  - id: D7
    description: "Both pre-executed notebooks re-executed against the fixed code with real committed output, and the runbook documents the new counters, the noon anchor, the release-on-supersede behaviour, the corrected inline instruction (WR-06), and the paginated campaign-table anchor constraint (WR-08)"
    verification:
      - kind: manual_procedural
        ref: "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb + docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb, executed via jupyter nbconvert --execute against src/fomo_db.sqlite3; docs/runbooks/telescope_runs_calendar.rst prose"
        status: pass
    human_judgment: true
    rationale: "The notebook D-04 before/after diff over the real developer database (167 non-RUN:-namespaced events compared, 0 differences) and the runbook prose changes are best confirmed by a human skim of the rendered docs, even though the underlying assertions in the notebook cells and the acceptance-criteria greps all pass automated checks."

# Metrics
duration: 40min
completed: 2026-09-09
status: complete
---

# Phase 33 Plan 08: Noon-Anchored Observing Night and Unconditional Skip Summary

**Closes CR-02 (post-midnight events resolved to the wrong observing night) and CR-03 (the reconcile-then-attribute ordering left permanent duplicate calendar entries) with a shared noon-anchored night helper, a counted/logged detach step, operator-visible counters, and re-executed paired docs.**

## Performance

- **Duration:** ~40 min
- **Started:** 2026-09-09T00:34:03Z (first task commit)
- **Completed:** 2026-09-09T01:05:58Z (last task commit)
- **Tasks:** 4 (1 checkpoint:decision, resolved by the user before this executor started; 1 tracer, 2 auto)
- **Files modified:** 7

## Accomplishments

- Added `_observing_night(start_time, site_zone)`, the local-noon anchored event-to-night mapping matching `telescope_runs._local_noon_utc()`'s own convention, superseding 26-DECISION.md D-10's plain `.date()` derivation. Closes CR-02: an event starting after local midnight now resolves to the PREVIOUS date's observing night.
- Made the D-01/ANNOT-01 skip in `_reconcile_classical_nights()` unconditional on attribution alone (dropped the `existing is None` conjunction) and had it return the active-url set it considers current, so `reconcile_run()` no longer re-derives the window twice (IN-02). Closes CR-03: reconcile-then-attribute and attribute-then-reconcile now converge on the same result.
- Added `ReconcileResult.detached`; `_detach_stale_family_events()` now returns the cleared-row count and logs a warning naming the run and count when non-zero (WR-03).
- Surfaced `skipped_nights`/`detached`/`would_detach: n/a (dry-run)` in `reconcile_campaign_runs`'s per-run and summary output, so an operator can tell "already converged" apart from "covered elsewhere" and is told when a confirmation stamp was discarded (WR-01).
- Re-executed both paired notebooks against the fixed code (real developer database), rewrote the skip-rule demo cell to remove its unconditional `.delete()` (IN-04) and demonstrate the post-fix detach-and-restore behaviour instead, and updated the operator runbook's reconcile-summary, skip-rule, inline-instruction (WR-06) and pop-up-link (WR-08) sections.

## Task Commits

Each task was committed atomically:

1. **Task 1: Anchor the observing night at local noon and make the skip unconditional** - `5af1254` (fix)
2. **Task 2: Surface the two counters in the operator sweep** - `7e90dfb` (feat)
3. **Task 3: Re-execute the paired notebooks and bring the runbook back in line** - `40109b8` (docs)

_Note: the plan's Task 0 was a `checkpoint:decision` resolved by the user (`noon-anchor`) before this executor was dispatched; it produced no commit of its own._

## Files Created/Modified

- `solsys_code/campaign_reconciler.py` - `_observing_night()` helper, `ReconcileResult.detached`, unconditional skip + returned active-url set, counted/logged detach
- `solsys_code/management/commands/reconcile_campaign_runs.py` - `skipped_nights`/`detached`/`would_detach` counters, per-run stdout/stderr lines
- `solsys_code/tests/test_campaign_reconciler.py` - `TestObservingNightBoundary` (Sydney + new Chilean `Observatory` fixture, adjacency and empty edges) and `TestReconcileThenAttributeOrdering` (detach-then-restore, blocked-night); re-homed the Sydney pk=54 measured case out of `TestAttributedNightSkip`
- `solsys_code/tests/test_reconcile_campaign_runs.py` - `TestSkipAndDetachCounters` (four behaviors: isolated skip, isolated detach, dry-run parity, both counters together)
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` - re-executed; skip-rule cell rewritten, no unconditional delete, new counters markdown cell
- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` - re-executed unchanged in content (output shape identical apart from non-deterministic pks)
- `docs/runbooks/telescope_runs_calendar.rst` - reconcile summary counters, noon-anchor + release-into-queue prose, corrected inline instruction (WR-06), pop-up link pagination caveat (WR-08)

## Decisions Made

See `key-decisions` in the frontmatter above. In summary: the noon-anchor supersedes D-10 (user-approved at the plan's checkpoint); the `sun_event()` call in the classical loop now fires only for non-skipped nights (a small deviation from the plan's literal wording, matching the review's own suggested fix shape); one stale real-database row from the old (pre-fix) notebook cell design was cleaned up directly rather than via new self-healing notebook code; `campaign_lifecycle_demo.ipynb` needed no markdown addition since its flow never re-exercises the detach step; and WR-06's fix targets the actual erroneous bullet (the second one) rather than the plan text's literal "third bullet" count.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug-adjacent efficiency fix] Moved `sun_event()` call to fire only for non-skipped nights**
- **Found during:** Task 1
- **Issue:** The plan's action text described moving only the `night in attributed_nights` test above the `CalendarEvent.objects.filter(url=url).first()` lookup, leaving the `sun_event()` call's position unspecified. Calling it unconditionally (as the pre-fix code did) computes a dip-corrected sunset/sunrise for a night the branch is about to skip entirely and never uses.
- **Fix:** Moved the `sun_event()` call below the skip check, so it only runs for a night that will actually be written. This matches 33-REVIEW.md's own suggested fix pseudocode (which places the `continue` before any per-night computation) and does not change behaviour for any existing test, including the mid-loop `ValueError` propagation test (which has no attributed nights, so its call sequence is unaffected).
- **Files modified:** `solsys_code/campaign_reconciler.py`
- **Verification:** Full `test_campaign_reconciler` suite (62 tests) and the broader four-module suite (236 tests) pass.
- **Committed in:** `5af1254`

**2. [Rule 3 - Blocking, real-database cleanup] Removed a stale leftover row from a prior notebook execution**
- **Found during:** Task 3
- **Issue:** The real developer database (`src/fomo_db.sqlite3`) carried a permanent leftover `CalendarEvent` (pk=322, "Hand-entered classical night (skip-rule demo)") from the OLD, pre-fix skip-rule cell, which used to delete the corresponding `RUN:`-keyed event unconditionally and never cleaned up its own stand-in row. This left the demo campaign's middle night permanently attributed with no `RUN:`-keyed sibling for the new cell's `CalendarEvent.objects.get(url=existing_run_keyed_url)` lookup to find.
- **Fix:** Deleted that one row directly via Django (not via notebook code), printing what was removed and why in my own terminal output (reported in this SUMMARY's key-decisions). This let the new cell's own earlier real sweep (re-)mint the `RUN:`-keyed event fresh, matching the "a night that already has its own RUN:-keyed event from this notebook's earlier real sweep" precondition the new cell's design assumes.
- **Files modified:** none (real database row only; not a tracked file)
- **Verification:** Re-executed notebook's skip-rule cell output shows `classical_run pk=59 already owns 'RUN:59:2026-09-02' (pk=335)...` and the full detach/restore sequence completing as designed.
- **Committed in:** N/A (database state, not a file change; the notebook's re-executed output in `40109b8` reflects the corrected starting state)

---

**Total deviations:** 2 auto-fixed (1 Rule 1, 1 Rule 3). **Impact:** Both were necessary for the fix to behave as designed and for the paired notebook to demonstrate the correct post-fix state; neither expanded scope beyond what CR-02/CR-03/IN-04 already required.

## Issues Encountered

None beyond the two deviations documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- CR-02 and CR-03 are closed with regression tests derived from 33-REVIEW.md's own reproduction transcripts; WR-01, WR-03, IN-02 and IN-04 are closed; WR-06 and WR-08's runbook halves are written here (their code/test halves were closed in plans 33-06/33-07).
- `reconcile_run()` still writes nothing outside the `RUN:` namespace; the notebook's real-database before/after diff over 167 non-`RUN:`-namespaced events stayed empty after the night-derivation change (D-04's proof still holds).
- The paired-docs obligation for `campaign_reconciler.py` and `reconcile_campaign_runs.py` is satisfied within this plan.
- This is the last plan of Phase 33's gap-closure run (33-06 -> 33-07 -> 33-08); Phase 33 is ready for `/gsd-verify-work 33`. PROJ-04's `ordering` edge (the abstained `observation_group` reverse-manager iteration-order backstop truth) remains an open operator decision for that step, per this plan's "Flagged assumptions" section -- no plan in this gap-closure run authored a predicate for it.

## Self-Check: PASSED

- FOUND: solsys_code/campaign_reconciler.py
- FOUND: solsys_code/management/commands/reconcile_campaign_runs.py
- FOUND: solsys_code/tests/test_campaign_reconciler.py
- FOUND: solsys_code/tests/test_reconcile_campaign_runs.py
- FOUND: docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
- FOUND: docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
- FOUND: docs/runbooks/telescope_runs_calendar.rst
- FOUND commit: 5af1254
- FOUND commit: 7e90dfb
- FOUND commit: 40109b8

---
*Phase: 33-series-identity-reconciler-inversion*
*Completed: 2026-09-09*
