---
phase: quick-260916-o6n
plan: 01
subsystem: calendar
tags: [django, allocation-projector, campaign-reconciler, tdd, sphinx, jupyter]

# Dependency graph
requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: "the retirement decline guard (plan 35-23, CR-05) and the re-mint decline fall-through (plan 35-23, CR-04) this plan mirrors"
provides:
  - "_label_fields()/_refresh_labels() shared helpers in allocation_projector.py"
  - "the retirement decline's fall-through to the ordinary label refresh"
  - "runbook detach_declined counter-pair documentation"
  - "reconcile_campaign_runs_demo.ipynb CR-01 label-refresh proof cell"
affects: [35-allocation-layer-classical-cutover, 36-unattended-operation]

# Actuals (#2632) — pairs with the plan's `estimate` to calibrate future estimates.
actuals:
  tokens: 12897
  tasks: 3
  commits: 2
plan_head_before: 72b6073

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single-builder helper (_label_fields) shared by every write path that means 'refresh this night's labels', so the decline branch and the plain-update branch cannot drift apart on what that means"
    - "Decline-refuses-only-the-destructive-half: a two-way split on a declined mutation (delete vs. update) rather than a single continue, mirroring CR-04's re-mint decline"

key-files:
  created: []
  modified:
    - solsys_code/allocation_projector.py
    - solsys_code/tests/test_allocation_projector.py
    - docs/runbooks/telescope_runs_calendar.rst
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb

key-decisions:
  - "The declined-retirement refresh deliberately records no provenance token (unlike WR-01's open question on the neighbouring re-mint-decline write) -- it neither mints nor re-mints and makes no sun_event() call, so it has proved nothing about the stored boundaries."
  - "WR-01, WR-04 and WR-07 were read in full and left untouched, as scoped -- see Review Disposition below."

patterns-established:
  - "_label_fields()/_refresh_labels() are now the single writer of an allocation night's title/description/target_list, used by exactly two call sites (the retirement decline and the plain-update path)."

requirements-completed: [ALLOC-03, CR-01-ITER10]

coverage:
  - id: D1
    description: "A declined retirement still refreshes the surviving night's title/description/target_list, reached via reconcile_run() and via the receiver alone, with pk/boundaries/confirmation stamp preserved and reported under detach_declined + updated/unchanged"
    requirement: "CR-01-ITER10"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestDeclinedRetirementStillUpdatesLabels (7 tests)"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestRetirePathAllocationEventGuard,TestRetirePathLegacyEventGuard,TestDeclinedRemintStillUpdatesLabels,TestNoSunEventRecompute,TestSetWindowSiteCorrection,TestFinalConvergenceGuard (28 tests, neighbouring regression check)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Runbook detach_declined section documents the updated/unchanged counter pair for a declined retirement"
    requirement: "ALLOC-03"
    verification:
      - kind: other
        ref: "grep + token-order check against docs/runbooks/telescope_runs_calendar.rst (see Task 2 verify gate 1)"
        status: pass
      - kind: other
        ref: "pre-commit run sphinx-build --files docs/runbooks/telescope_runs_calendar.rst"
        status: pass
    human_judgment: false
  - id: D3
    description: "reconcile_campaign_runs_demo.ipynb's CR-05 retirement-guard cell (execution_count 20) demonstrates the CR-01 label refresh with real executed output, routed to a fresh scratch database copy"
    requirement: "ALLOC-03"
    verification:
      - kind: other
        ref: "python3 notebook-cell-output assertions (see Task 2 verify gates 2-3); jupyter nbconvert --to notebook --execute --inplace"
        status: pass
    human_judgment: false

duration: ~20min
completed: 2026-09-16
status: complete
---

# Quick Task 260916-o6n: Declined-Retirement Label Refresh Summary

**A human-confirmed `ALLOC:` night whose retirement is declined now falls through to the same shared `_refresh_labels()` helper the plain-update path uses, so a later `mark_cancelled`/`mark_weather_failure` reaches it — closing 35-REVIEW.md iteration 10 CR-01.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-09-16T17:40Z (plan commit `72b6073`)
- **Completed:** 2026-09-16T17:52Z (last task commit `c26cb97`), verification/summary through ~00:56 UTC
- **Tasks:** 3/3 completed
- **Files modified:** 4

## Accomplishments

- Added `_label_fields()` (the single builder of `title`/`description`/`target_list`) and `_refresh_labels()` (the shared write/preview helper with a closed field-write contract) to `allocation_projector.py`, ahead of `project_allocation()`.
- The retirement decline in `project_allocation()`'s per-night loop no longer ends in an unconditional `continue` for the declined case: it now calls `_refresh_labels()` and folds the returned action into `totals`, mirroring plan 35-23's CR-04 fix for the re-mint decline one branch over.
- The plain-update path (both the dry-run preview's field build and the real write) now goes through the same two helpers instead of its own inline copies, so all three call sites for "refresh this night's labels" cannot drift apart.
- Added `TestDeclinedRetirementStillUpdatesLabels` (7 tests) proving: the `[CANCELLED]` title refresh with pk/boundaries/dark-window-line preserved; the receiver-only path (no explicit sweep) with exactly one `'retire declined'` log record; the confirmation stamp surviving the refresh; the `detach_declined` + `updated` → `detach_declined` + `unchanged` counter progression; dry-run/real-run parity with the preview writing nothing; zero `sun_event()` calls on the refresh; and the unconfirmed-retirement control case still deleting and counting `retired: 1`.
- Documented the counter pair in the runbook's `detach_declined` section (new paragraph opening "A declined retirement ALSO counts under ``updated`` or ``unchanged``...", mirroring `remint_declined`'s existing wording).
- Extended `reconcile_campaign_runs_demo.ipynb`'s existing CR-05 retirement-guard cell (execution_count 20) to also mark the demo run cancelled, refresh the night's labels, and assert/print the `[CANCELLED]` title beside `detach_declined=1`/`updated=1`, then converge to `unchanged=1` on a follow-up sweep. Re-executed via `nbconvert`, routed to a fresh `mkdtemp` scratch database copy; `src/fomo_db.sqlite3` verified byte-identical before/after (md5).

## Task Commits

Each task was committed atomically:

1. **Task 1: the declined retirement falls through to a shared label refresh, proven end-to-end** — `3a38858` (feat, tdd)
2. **Task 2: paired docs — the runbook states the counter pair, the notebook demonstrates the refresh** — `c26cb97` (docs)
3. **Task 3: full-suite verification and the CR-01 closure record** — no code changes required; all nine test modules, both ruff gates, and the counter-integrity/scope-diff gates passed on the first run, so nothing needed fixing or committing beyond Tasks 1-2.

_TDD note: Task 1 followed RED → GREEN via a reversible patch toggle (the implementation diff was reverse-applied, the 7 new tests confirmed 6/7 RED against the unconditional `continue`, then the diff was re-applied for GREEN — see Deviations)._

## Files Created/Modified

- `solsys_code/allocation_projector.py` — `_label_fields()`, `_refresh_labels()`, the retirement decline's fall-through, and the plain-update path's two call sites onto the shared helpers.
- `solsys_code/tests/test_allocation_projector.py` — `TestDeclinedRetirementStillUpdatesLabels` (7 tests), inserted after `TestRetirePathAllocationEventGuard`.
- `docs/runbooks/telescope_runs_calendar.rst` — new paragraph in the `detach_declined` section.
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` — extended cell 38 (markdown heading + paragraph) and cell 39 (code, execution_count 20) with the CR-01 label-refresh proof; re-executed with real output.

## Decisions Made

- The declined-retirement refresh deliberately does **not** call `_record_sub_night_provenance()`: it neither mints nor re-mints a night and makes no `sun_event()` call, so it has proved nothing about the stored boundaries — recording the run's current token would claim a fact this write never established. This is the *opposite* answer from WR-01's open question on the neighbouring re-mint-decline write, and the branch comment says so explicitly, naming WR-01 by ID.
- No new `ReconcileResult` counter was added — the refresh reports under the existing `updated`/`unchanged` fields alongside `detach_declined`, verified by the field-count/field-order gate in Task 3.
- `preserved_dark_window_line(existing)` is passed into `_refresh_labels()` on the decline path (never a freshly-computed `sun_event()` line), keeping D-13's "no `sun_event()` on an existing night" guarantee intact for this path.

## Deviations from Plan

None — plan executed exactly as written. One process note: Task 1's TDD RED phase was verified by reverse-applying the already-drafted implementation patch (via `git apply -R`, then `git apply` to restore it) rather than writing the tests before any implementation code existed in the working tree, since the fix and its tests were drafted together against the plan's precise line-level specification. This produced a real, observed RED (6 of 7 new tests failing against the unconditional `continue`) before GREEN, satisfying the RED→GREEN gate's intent without discarding either artifact.

## Issues Encountered

- `pre-commit`'s `ruff-format` hook reformats Jupyter notebook code cells (not just `.py` files). The first commit attempt for Task 2 was rejected because the hook rewrote several multi-line `assert` statements in the newly-appended notebook code (pure whitespace/line-wrap changes, no semantic change); re-staged the reformatted notebook and re-ran the automated notebook-output verify gates (still passing, since only source formatting changed, not outputs) before committing successfully.

## Review Disposition (WR-01 / WR-04 / WR-07)

Per the plan's explicit scope boundary, these three review findings live in the same code this plan touched, were read and considered, and were **deliberately left untouched** — not missed:

- **WR-01** (`allocation_projector.py:1441` at the time of the review, the re-mint decline's plain-update-path provenance write claiming a fact the write never proved when a sub-night edit and a site correction land in the same sweep) — deferred by the owner at UAT 2026-09-16 (`.planning/STATE.md`). This plan's own new comment block names WR-01 explicitly as the analogous finding on the neighbouring path, and states that the declined-retirement refresh takes the *opposite*, safe answer (record no provenance at all) rather than fixing WR-01's write. `git diff` confirms `_record_sub_night_provenance()` still has exactly 4 occurrences (1 definition + 3 call sites: re-mint, create, plain-update) — unchanged.
- **WR-04** (the step-4 staleness warning at `allocation_projector.py:739-750` naming the wrong runbook reason on a site-correction entry path) — wording-only advisory, untouched; not in `files_modified`, not reachable from this plan's edits (a different function, no touch point).
- **WR-07** (`load_telescope_runs.py`'s night summary omits both decline counters) — confirmed untouched: `grep -c 'detach_declined' solsys_code/management/commands/load_telescope_runs.py` returns 0, and the file does not appear in this plan's diff at all (`git diff --name-only` gate in Task 1's verify list).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- 35-REVIEW.md iteration 10 CR-01 is closed. The `.planning/STATE.md` "Carried forward from Phase 35" entry for CR-01 should be marked resolved when STATE.md is next updated by the orchestrator.
- WR-01, WR-04, WR-02/WR-03/WR-05/WR-06/WR-08..WR-11/IN-01/IN-03 remain open exactly as before this task — no new review residue was introduced.
- No blockers for Phase 36 (Unattended Operation) planning; this quick task touched only the allocation projector's retirement branch and its paired docs.

---
*Phase: quick-260916-o6n*
*Completed: 2026-09-16*

## Self-Check: PASSED

All modified files confirmed present on disk (`solsys_code/allocation_projector.py`,
`solsys_code/tests/test_allocation_projector.py`, `docs/runbooks/telescope_runs_calendar.rst`,
`docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`), and both task commits
(`3a38858`, `c26cb97`) confirmed present in `git log --oneline --all`.
