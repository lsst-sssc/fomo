---
phase: 35-allocation-layer-classical-cutover
plan: 07
subsystem: calendar-sync
tags: [django, docs, jupyter, campaign-reconciler, allocation-projector, classical-cutover, sphinx]

requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: "plan 35-05's rewritten load_telescope_runs (one CampaignRun per line, --dry-run, [proposal] token) and plan 35-06's cutover_classical_allocations command plus the real scratch-copy before/after numbers this plan's paired docs reproduce and quote"
provides:
  - "docs/runbooks/telescope_runs_calendar.rst -- rewritten classical-ingest section, D-10 queue-dispatch consequence in 'Can I correct a run's source?', a new 'How do I run the one-time classical cutover?' section with the four-step sequence, a cutover cheat-sheet row, and two new troubleshooting entries"
  - "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb -- re-executed against a throwaway scratch copy of the developer database, demonstrating the allocation path, --dry-run, run+night idempotency, sun_event()-drift immunity, the [proposal] token and its collision report, sub-night fields, and --campaign"
  - "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb -- the executed cutover before/after diff (241/56/16/10/0/45 -> 233/0/16/1/57/48) with four end-state assertions, all four current dispatch branches, and the D-05/D-07 observation-handoff demo"
  - "CLAUDE.md notebook pairing map extended with cutover_classical_allocations.py -> reconcile_campaign_runs_demo.ipynb"
affects: []

actuals:
  tokens: 52111
  tasks: 3
  commits: 3
  plan_head_before: 3f90cc8a6b27f37e9754c1cae3bd6d261d757c49

tech-stack:
  added: []
  patterns:
    - "Pristine-baseline notebook ordering: the reconciler demo's cutover section runs immediately after Django setup, before any of the notebook's own synthetic CampaignRun fixtures are seeded -- a deliberate deviation from the plan's literal action-item ordering (which lists the cutover section fourth, after the fixture/dispatch demo). Seeding fixtures first and then calling reconcile_campaign_runs as part of that demo would have already rekeyed/deleted the real database's own RUN:{pk}:{date} family before the cutover section's own before-state capture ran, contaminating the before/after diff this plan exists to prove. Running the cutover first preserves the exact worked-example numbers the runbook quotes."
    - "Both cutover cells (--dry-run and real) drive cutover_classical_allocations through call_command() inside an identical try/except CommandError pattern, printing the D-18 report as real cell output instead of letting the exception end the notebook run -- the same pattern for both invocations, per the plan's own instruction, so a reader sees one idiom, not two."
    - "Pure-Python key-family classification (no ORM __regex lookup) for the before/after CalendarEvent counts: comparing the RUN:/ALLOC: prefix and colon-presence of a small, already-materialized list of urls in Python avoids any sqlite REGEXP-function backend dependency."

key-files:
  created: []
  modified:
    - docs/runbooks/telescope_runs_calendar.rst
    - CLAUDE.md
    - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb

key-decisions:
  - "The reconciler demo's cutover section runs BEFORE the fixture/dispatch-branch demo (opposite of the plan's action-item listing order), so its before-state capture reflects the pristine real developer database rather than a database already partially converged by the fixture demo's own reconcile_campaign_runs calls. This reproduces plan 35-06's exact real-database numbers (241/56/16/10/0/45 before, 233/0/16/1/57/48 after, 48 rekeyed + 8 legacy_deleted = 56) as the executed proof, rather than a diluted or trivially-zero diff."
  - "ReconcileResult.retired reports 1 for every reconcile call while a linked record's block still occupies a night, not just on the call that performed the delete -- the observation-handoff demo's assertions were corrected mid-execution to match this (retired means 'this night IS retired', per plan 35-01's own Task 2 Test 1 contract), rather than expecting a one-time transition to 0."
  - "Deleting a CampaignRunObservation link restores its allocation night automatically, with no explicit reconcile_run() call needed at all: D-11's post_delete receiver re-projects the linked run as part of the delete itself. The demo's own final reconcile_run() call was corrected to assert a no-op (already converged), not a fresh creation -- the first version of this cell asserted `created == 1` and failed because the night was already restored by the time `.delete()` returned."
  - "The classical-ingest notebook's fixture Observatory rows and the reconciler notebook's cutover section both run against a scratch copy of the developer database that is migrated to head (migration 0018) inside the setup cell, because the real src/fomo_db.sqlite3 has not had that migration applied and CampaignRun.night_start_utc/night_end_utc are read unconditionally by both load_telescope_runs and the allocation projector."

requirements-completed: [ALLOC-04, ALLOC-05]

coverage:
  - id: D1
    description: "The operator runbook documents the load_telescope_runs allocation contract (one CampaignRun per line, [proposal] token, --dry-run, two summary lines), the D-10 queue-dispatch consequence, and a new four-step classical cutover section with the real 35-06 worked-example numbers, a cheat-sheet row and two troubleshooting entries"
    requirement: ALLOC-04
    verification:
      - kind: other
        ref: "python -c token-count check over telescope_runs_calendar.rst -> 5/5; CLAUDE.md cutover_classical_allocations.py count -> 1; CLAUDE.md diff --stat -> 2 lines"
        status: pass
      - kind: other
        ref: "pre-commit run sphinx-build --all-files -> Passed"
        status: pass
    human_judgment: false
  - id: D2
    description: "load_telescope_runs_demo.ipynb re-executed against a throwaway scratch copy of the developer database, demonstrating the allocation path (CampaignRun + ALLOC: nights), --dry-run, run+night-level idempotency, sun_event()-drift immunity (0 calls on an unchanged re-import), the [proposal] token and its collision report, sub-night fields, and --campaign on both the run and its nights"
    requirement: ALLOC-04
    verification:
      - kind: other
        ref: "jupyter nbconvert --to notebook --execute --inplace -> exit 0, no CellExecutionError/Traceback"
        status: pass
      - kind: other
        ref: "python -c cell-output/execution_count check -> 14/14; token check (FOMO_DATABASE_PATH, ALLOC:, source_identifier, --dry-run) -> 4/4; retired blank-url lookup absent -> False"
        status: pass
      - kind: other
        ref: "git status --porcelain -- src/fomo_db.sqlite3 -> empty, both before and after execution"
        status: pass
    human_judgment: false
  - id: D3
    description: "reconcile_campaign_runs_demo.ipynb carries the executed cutover before-and-after diff (D-15) with four end-state assertions in code: zero date-bearing RUN: nights, only the reported unexplained blank-url row remains, containers unchanged in count, and every facility-url observation event byte-identical -- reproducing plan 35-06's real-database numbers exactly"
    requirement: ALLOC-05
    verification:
      - kind: other
        ref: "jupyter nbconvert --to notebook --execute --inplace -> exit 0, no CellExecutionError/Traceback"
        status: pass
      - kind: other
        ref: "python -c cell-output/execution_count check -> 16/16; token check (cutover_classical_allocations, ALLOC:, legacy_deleted) -> 3/3; assert count -> 18 (>=4); except CommandError count -> 3 (>=2)"
        status: pass
      - kind: integration
        ref: "Real scratch-copy execution: before 241/56/16/10/0/45, after 233/0/16/1/57/48; sweep line 48 rekeyed + 8 legacy_deleted = 56, matching plan 35-06's proof exactly"
        status: pass
      - kind: other
        ref: "pre-commit run sphinx-build --all-files -> Passed; git status --porcelain -- src/fomo_db.sqlite3 -> empty throughout"
        status: pass
    human_judgment: true
    rationale: "The plan's own <human-check> asks a human to open the rendered notebook and confirm the before/after table shows real numbers, the four end-state assertions executed and passed, and the unexplained list contains only recognisable events -- a judgement about what the calendar means, not a property a test alone can assert. I inspected the actual executed output myself and confirmed it reproduces plan 35-06's numbers exactly and the sole unexplained row is the documented pre-existing junk `tmp` event (pk=334), but the plan explicitly routes this confirmation to a human reviewer."
  - id: D4
    description: "The D-05/D-07 observation handoff is demonstrated end to end in the reconciler notebook: linking a real ObservationRecord (NonSiderealTargetFactory target) whose placed block occupies an allocation night retires that night (ReconcileResult.retired == 1, no ALLOC: event); deleting the link restores it automatically via the post_delete receiver (D-11), with no explicit reconcile call needed"
    verification:
      - kind: other
        ref: "Executed notebook cell: retire_result.retired == 1, event absent after linking; event present (same title) immediately after link.delete(), before any further reconcile_run() call"
        status: pass
    human_judgment: false

duration: 195min
completed: 2026-09-13
status: complete
---

# Phase 35 Plan 07: Classical Cutover Paired Docs Summary

**The operator runbook, the classical-ingest demo notebook, and the reconciler demo notebook are all rewritten for the Phase 35 allocation path, with the reconciler notebook's new cutover section reproducing plan 35-06's real-database before/after numbers (241/56/16/10/0/45 -> 233/0/16/1/57/48) as executed proof, four end-state properties asserted in code.**

## Performance

- **Duration:** ~195 min
- **Started:** 2026-09-13T08:27:00Z (approx, from the prior plan's completion commit)
- **Completed:** 2026-09-13T11:42:00Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- `docs/runbooks/telescope_runs_calendar.rst`: rewrote "How do I load a classical telescope schedule?" for the allocation contract (one `CampaignRun` per line, the `[proposal]` token and its collision message/fix, `--dry-run` and its two summary lines, byte-identical calendar output); updated the queue-vs-per-night dispatch description and "Can I correct a run's source?" for the D-10 queue-container consequence and the `LEGACY`-stays-per-night rule; added a new "How do I run the one-time classical cutover?" section with the four-step sequence and the real 35-06 worked-example numbers; added a cutover cheat-sheet row and two troubleshooting entries (a reported `source_identifier` collision, a reported unexplainable cutover event).
- `CLAUDE.md`: extended the notebook pairing map with `cutover_classical_allocations.py` -> `reconcile_campaign_runs_demo.ipynb` (2-line diff, per the plan's own bound).
- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`: fully rewritten and re-executed against a throwaway scratch copy of the developer database (migrated to head for migration 0018). Demonstrates: the run+night two-line summary; `--dry-run` writing nothing; the created `CampaignRun`'s fields alongside its `ALLOC:`-keyed nights; a cancelled run's `run_status`/`[CANCELLED]` title prefix; run-level and night-level idempotency; a drift-tolerance cell proving `allocation_projector.sun_event()` is called zero times on an unchanged re-import (D-13); the two sub-night fields on a partial-night run; the `[proposal]` token producing two distinct runs versus the same lines without a token colliding onto one; and `--campaign` setting the campaign on both the run and every night it produces.
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`: fully rewritten and re-executed. Its new cutover section (placed immediately after Django setup, before any of the notebook's own fixtures, to preserve the pristine baseline) captures the before-state, drives `cutover_classical_allocations --dry-run` and for real through `call_command()` inside `try`/`except CommandError` (rendering the expected D-18 report as real output both times), runs one `reconcile_campaign_runs` sweep, captures the after-state, and asserts all four end-state properties in code. The real numbers exactly reproduce plan 35-06's own proof: 241 events (56 date-bearing `RUN:` nights, 16 containers, 10 blank-url, 0 `ALLOC:`, 45 runs) before, 233 events (0 date-bearing, 16 containers, 1 blank-url -- the same junk `tmp` row, 57 `ALLOC:`, 48 runs) after, with the sweep's own line confirming `48 rekeyed + 8 legacy_deleted = 56`. The fixture/dispatch-branch demo below it was rewritten for D-09/D-10 (a queue-sourced run now gets a container regardless of site; the classical fixture is campaign-less and gets `ALLOC:` nights instead of `RUN:` nights), the namespace-isolation proof was extended to cover `ALLOC:` alongside `RUN:` (the retired `run_night_url()` import was removed), and the stale skip-rule/confirm-erase cells were replaced with a new D-05/D-07 observation-handoff demo: linking a real `ObservationRecord` retires an allocation night, and deleting the link restores it automatically via the `post_delete` receiver, with no explicit reconcile call needed.
- Both notebooks re-executed via `jupyter nbconvert --to notebook --execute --inplace`, committed with real output; `src/fomo_db.sqlite3` confirmed unmodified (`git status --porcelain` empty) before, during, and after every execution across both notebooks and every commit.

## Task Commits

Each task was committed atomically:

1. **Task 1: Runbook sections and the CLAUDE.md notebook map** - `08249c2` (docs)
2. **Task 2: Re-execute the classical ingest demo notebook against a scratch database** - `7d7b9d4` (feat)
3. **Task 3: The cutover before-and-after diff in the reconciler demo notebook** - `72edd1b` (feat)

**Plan metadata:** committed alongside this SUMMARY.

## Files Created/Modified

- `docs/runbooks/telescope_runs_calendar.rst` - rewritten classical-ingest section, D-10 consequence, new cutover section, cheat-sheet row, two troubleshooting entries
- `CLAUDE.md` - notebook pairing map extended with the cutover command
- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` - full rewrite for the allocation path, scratch-database routing
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` - full rewrite: new cutover section, D-09/D-10 dispatch demo, D-05/D-07 observation-handoff demo

## Decisions Made

See `key-decisions` in the frontmatter above.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug, found during Task 3 execution] Observation-handoff demo asserted the wrong steady-state value for `ReconcileResult.retired`**
- **Found during:** Task 3, first execution of the observation-handoff demo cell
- **Issue:** The cell asserted `unchanged_result.retired == 0` on a second `reconcile_run()` call made while the observation link was still in place. `retired_nights()`/`project_allocation()` report `retired` for every night currently covered by a linked, placed record on every call (per plan 35-01's own Task 2 Test 1 contract: "retired" means "this night IS retired," not "a delete just happened"), so the correct expectation is `retired == 1` again, not `0`.
- **Fix:** Corrected the assertion to `retired == 1` and updated the adjacent comment/markdown to state the idempotent-convergence property accurately.
- **Files modified:** `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`
- **Verification:** Re-executed via `jupyter nbconvert`; the cell now passes and the printed `ReconcileResult` matches the corrected expectation.
- **Committed in:** `72edd1b` (Task 3 commit)

**2. [Rule 1 - Bug, found during Task 3 execution] Observation-handoff demo asserted the wrong outcome for the unlink step, and read `link.pk` after it was already cleared**
- **Found during:** Task 3, first execution of the observation-handoff demo cell
- **Issue:** The cell called `link.delete()` then printed `link.pk` (already `None` post-delete -- Django clears an instance's pk after `.delete()`) and asserted `reconcile_run(classical_run).created == 1`, expecting the explicit reconcile call to restore the night. In fact `receiver_on_run_observation_delete()` (D-11) re-projects the linked run as part of the `post_delete` signal itself, so the `ALLOC:` event was already restored by the time `.delete()` returned -- the subsequent explicit `reconcile_run()` call correctly reported `created: 0` (already converged), and the assertion failed.
- **Fix:** Captured `link_pk = link.pk` before calling `.delete()`; replaced the post-delete assertion with a direct check that the `ALLOC:` event already exists (same title) immediately after `.delete()` returns, then asserted a further explicit `reconcile_run()` call is a no-op. Updated the adjacent markdown to describe the automatic-restore behavior.
- **Files modified:** `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`
- **Verification:** Re-executed via `jupyter nbconvert`; the cell now passes, and the demonstrated behavior (automatic restore on unlink, no operator command) is a stronger, more accurate illustration of D-11 than the original plan text anticipated.
- **Committed in:** `72edd1b` (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 bugs in the observation-handoff demo cell, found by this task's own execution -- neither is a defect in production code; both are corrections to the demo's own expectations about already-correct `ReconcileResult`/receiver behavior). **Impact:** Both fixes make the notebook's assertions accurate and the demonstrated behavior a strictly better illustration of D-05/D-07/D-11 than the plan's literal wording anticipated (the automatic-restore-on-unlink property is more compelling than an explicit-reconcile-required one). No scope creep.

### Interpretive Decision (not a defect)

**Cutover section placement reordered relative to the plan's action-item listing.** The plan's Task 3 action items list the cutover section fourth (after seeding/sweep/attributed-night), but placing it there in the notebook would mean the fixture-seeding section's own `reconcile_campaign_runs` calls (items 1-2) had already run at least once against the whole database before the cutover section's own "before-state" capture -- since `reconcile_campaign_runs` alone (independent of `cutover_classical_allocations`) already rekeys/deletes an EXISTING `CampaignRun`'s own leftover `RUN:{pk}:{date}` family the moment it dispatches to a different branch. Placing the cutover section immediately after Django setup instead (before any fixture is seeded) preserves the pristine pre-cutover baseline, so the before/after diff genuinely reproduces plan 35-06's real numbers rather than showing an already-diluted delta. This is a notebook-cell-ordering judgment call within the scope the plan left to discretion ("Test file layout... and how the notebooks express the before/after diffs" is explicitly listed under 35-CONTEXT.md's "Claude's Discretion"), not a deviation from any stated must-have.

## Issues Encountered

None beyond the two auto-fixed deviations documented above, both caught and corrected by this plan's own re-execution-and-verify loop before this SUMMARY was written.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- ALLOC-04 and ALLOC-05 are now fully discharged across their three shared plans (35-05, 35-06, this one): the classical ingest rewrite, the cutover command, and the paired documentation proving the four-step sequence against real data are all shipped.
- The paired-docs obligation in CLAUDE.md is discharged for this phase: both changed modules (`load_telescope_runs.py`, `cutover_classical_allocations.py`) have re-executed demo notebooks with real output, the affected runbook page is updated, and the notebook pairing map covers the new command for next time.
- ROADMAP Success Criterion 5 is demonstrated, not merely asserted: `reconcile_campaign_runs_demo.ipynb` carries a real before/after diff with four executed end-state assertions.
- **Carried forward, not resolved here (per the plan's own explicit scope note):** `project_observation_calendar_demo.ipynb`'s owed un-routed re-execution (the Phase 34 paired-docs debt, which also repairs 14 stale `KEY2026B-004` LCO events) remains open, recorded in STATE.md as "Phase 35 or a quick task." This plan's scratch-copy notebooks never touch that facility-url-keyed candidate set.
- No blockers. Phase 35 "Allocation Layer & Classical Cutover" is now complete (7/7 plans).

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-13*

## Self-Check: PASSED

- FOUND: docs/runbooks/telescope_runs_calendar.rst
- FOUND: CLAUDE.md
- FOUND: docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
- FOUND: docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
- FOUND commit: 08249c2
- FOUND commit: 7d7b9d4
- FOUND commit: 72edd1b
- Task 1 `<verify>`: runbook token check -> 5/5; CLAUDE.md occurrence count -> 1; `pre-commit run sphinx-build --all-files` -> Passed
- Task 2 `<verify>`: nbconvert exit 0; code-cell/output check -> 14/14; token check -> 4/4; retired blank-url lookup absent; `git status --porcelain -- src/fomo_db.sqlite3` -> empty
- Task 3 `<verify>`: nbconvert exit 0; code-cell/output check -> 16/16; token check -> 3/3; assert count -> 18 (>=4); `except CommandError` count -> 3 (>=2); `git status --porcelain -- src/fomo_db.sqlite3` -> empty; `pre-commit run sphinx-build --all-files` -> Passed
- Plan-level `<verification>`: all four items re-run and passing (sphinx-build green, both notebooks re-executed with non-null execution counts, runbook cutover section states the four steps in order, `src/fomo_db.sqlite3` unmodified)
