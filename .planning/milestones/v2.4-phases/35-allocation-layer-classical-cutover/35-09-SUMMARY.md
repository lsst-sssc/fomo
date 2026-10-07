---
phase: 35-allocation-layer-classical-cutover
plan: 09
subsystem: allocation-layer
tags: [django-management-command, calendar-projection, zoneinfo, dry-run, tdd]

# Dependency graph
requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: load_telescope_runs allocation cutover (plans 35-01..35-07), campaign_reconciler/allocation_projector dispatch split, cutover_classical_allocations database-scoped identity guard (35-08)
provides:
  - "load_telescope_runs.handle() has a dedicated except ZoneInfoNotFoundError clause ahead of (ValueError, Observatory.DoesNotExist), so a mistyped Observatory.timezone skips one line and the import reaches its summary (NF-21)"
  - "_raise_if_set_window_inverted(run, night) -- a shared, astropy-free dry-run inversion guard called from both project_allocation() caller branches of _mint_fields() (the re-mint branch and the create branch), so a dry-run preview of an operator-inverted sub-night window fails the same way the real run does on either branch (NF-20)"
  - "project_allocation()'s takeover branch claims legacy_urls_claimed on every decision including a block, matching the retired branch's NF-09 fix, so campaign_reconciler's downstream foreign fold never double-counts the same legacy event (NF-22)"
  - "_detach_stale_family_events()'s return annotation now states tuple[int, int, int, int], matching its own Returns: docstring and its actual four-value return statement (NF-23)"
affects: [35-10, 35-11]

# Actuals (#2632)
actuals:
  tokens: 4514
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Clause-order-sensitive exception handling: ZoneInfoNotFoundError subclasses KeyError, not ValueError, so a dedicated except clause must be placed AHEAD of an existing (ValueError, X) clause on the same try -- order, not breadth, is what decides which handler sees a raised exception."
    - "Shared dry-run guard extraction: when the same real-mode check (an inversion guard resolved with zoneinfo alone) is duplicated inline across two dry-run short-circuits, hoist it into one named helper called from both -- a third caller later has one guard to reuse instead of a third inline copy, closing the class of defect (NF-10 -> NF-20) that recurrence produces."
    - "Claim-on-every-decision symmetry: a per-decision 'this url has been handled' set (legacy_urls_claimed) must be populated the moment ANY outcome is decided (create, block, or decline) -- claiming only on the success path leaves the blocked/declined outcome visible to a downstream convergence step that double-counts the same single decision."

key-files:
  created: []
  modified:
    - solsys_code/management/commands/load_telescope_runs.py
    - solsys_code/tests/test_load_telescope_runs.py
    - solsys_code/allocation_projector.py
    - solsys_code/tests/test_allocation_projector.py
    - solsys_code/campaign_reconciler.py

key-decisions:
  - "Task 2 and Task 3 both touch solsys_code/allocation_projector.py and solsys_code/tests/test_allocation_projector.py with adjacent, non-overlapping edits; committed as two fully separate diffs (temporarily reverting Task 3's lines, committing Task 2, then re-applying Task 3's lines and committing separately) rather than one combined commit, to honour the plan's per-task atomic-commit requirement."
  - "TestMalformedTimezoneSkipsOneLine implemented as its own TestCase class (the plan's primary-preference option) with a minimal two-Observatory fixture, rather than a method added to TestLoadTelescopeRuns -- avoids running that class's full existing fixture/test set a second time under a new class name while still matching the plan's literal verify-command label."

requirements-completed: [ALLOC-01, ALLOC-02, ALLOC-04]

coverage:
  - id: D1
    description: "A schedule file line resolving to an Observatory with a mistyped IANA timezone is skipped and logged per-line -- the command reaches its summary reporting skipped: 1, and the following line is still processed; the bad line's CampaignRun does not exist (transaction.atomic() rollback)."
    requirement: ALLOC-04
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_load_telescope_runs.TestMalformedTimezoneSkipsOneLine#test_malformed_timezone_skips_only_its_own_line"
        status: pass
    human_judgment: false
  - id: D2
    description: "reconcile_run(run, dry_run=True) over an existing night whose operator-edited sub-night pair is inverted raises the identical ValueError the immediately following real run raises -- the re-mint branch, not only the create branch, is guarded."
    requirement: ALLOC-02
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_allocation_projector.TestSubNightWindowSiteDirection#test_dry_run_of_a_remint_inverted_window_also_raises"
        status: pass
    human_judgment: false
  - id: D3
    description: "A single legacy RUN:{pk}:{date} event attributed to a different run, reached on the takeover path, is reported as blocked == 1 with exactly one log line, not blocked == 2 with two."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code.tests.test_allocation_projector.TestTakeoverBlockedCountedOnce#test_foreign_attributed_legacy_event_on_takeover_path_counted_once"
        status: pass
    human_judgment: false
  - id: D4
    description: "_detach_stale_family_events()'s return annotation, docstring, and return statement all state a four-element tuple."
    requirement: ALLOC-01
    verification:
      - kind: other
        ref: "python -c NF-23 signature-annotation check (tuple[int, int, int, int] present in the def line)"
        status: pass
    human_judgment: false
  - id: D5
    description: "Full solsys_code test suite (test_views excluded) and both pinned ruff/ruff-format gates stay green after all three tasks."
    requirement: ALLOC-01
    verification:
      - kind: integration
        ref: "python manage.py test (43 module labels excluding test_views: 1273 tests, OK skipped=1); pre-commit run ruff --all-files; pre-commit run ruff-format --all-files"
        status: pass
    human_judgment: false

duration: ~75min
completed: 2026-09-15
status: complete
---

# Phase 35 Plan 09: Loader Exception Routing, Dry-Run Inversion Parity & Takeover Double-Count Summary

**Closed the four remaining 35-REVIEW.md write-path warnings: a mistyped `Observatory.timezone` no longer aborts the whole `load_telescope_runs` import (NF-21); a dry-run preview of an operator-inverted sub-night window now fails identically to the real run on both `_mint_fields()` caller branches (NF-20); a blocked legacy takeover event is counted once, not twice (NF-22); and `_detach_stale_family_events()`'s return annotation matches its actual four-value return (NF-23).**

## Performance

- **Duration:** ~75 min
- **Completed:** 2026-09-15T14:42:28Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- **NF-21:** `load_telescope_runs.handle()` gained a dedicated `except ZoneInfoNotFoundError` clause, placed ahead of the existing `(ValueError, Observatory.DoesNotExist)` clause on the same per-line `try` (clause order matters -- `ZoneInfoNotFoundError` subclasses `KeyError`, not `ValueError`). A mistyped site timezone now writes one stderr line naming the `Observatory`, its obscode and the offending timezone string, increments `run_skipped`, and lets the loop continue -- the runbook's "one bad row never aborts the whole run" invariant holds again for this command.
- **NF-20:** extracted `_raise_if_set_window_inverted(run, night)` in `allocation_projector.py` and call it from both `if dry_run:` short-circuits inside `project_allocation()`'s per-night loop -- the `_span_needs_remint()` re-mint branch (previously unguarded) and the `existing is None` create branch (whose inline duplicate the helper now replaces). A null sub-night field is never checked (matching `_span_needs_remint()`'s own convention), so the guard adds zero `sun_event()` calls on any path.
- **NF-22:** moved `legacy_urls_claimed.add(legacy_url)` to the first statement inside the takeover branch's `if existing is None and legacy_event is not None:` block, ahead of the `_may_write()` ownership check -- exactly mirroring the retired branch's existing NF-09 fix. A legacy event blocked on the takeover path is now excluded from `campaign_reconciler._stale_dated_events()`'s downstream `foreign` fold, so `ReconcileResult.blocked` reports the single decision once.
- **NF-23:** corrected `_detach_stale_family_events()`'s return annotation from `tuple[int, int, int]` to `tuple[int, int, int, int]`, matching its own `Returns:` docstring (already correct) and its actual four-value return statement.
- Each fix carries a regression test that failed against the pre-fix tree with the exact failure mode 35-REVIEW.md's probes reproduced (an uncaught `ZoneInfoNotFoundError`; a dry run reporting `created=1, retired=1` and raising nothing where the real run raised; `blocked == 2` with two log records where one decision occurred).
- Full `solsys_code` suite (1273 tests, `test_views` excluded per project convention) and both pinned `ruff`/`ruff-format` gates pass clean after all three fixes.

## Task Commits

1. **Task 1: Give the loader's write/reconcile call its own ZoneInfoNotFoundError handler** - `0eceeb5` (fix)
2. **Task 2: Hoist the dry-run inversion guard so both _mint_fields() caller branches share it** - `eb79263` (fix)
3. **Task 3: Count a blocked legacy takeover event once, correct the four-tuple annotation, re-green the suite** - `0b7599f` (fix)

**Plan metadata:** pending (this commit)

## Files Created/Modified
- `solsys_code/management/commands/load_telescope_runs.py` - `from zoneinfo import ZoneInfoNotFoundError` import; new `except ZoneInfoNotFoundError` clause on the per-line `try`, ahead of `(ValueError, Observatory.DoesNotExist)`.
- `solsys_code/tests/test_load_telescope_runs.py` - new `TestMalformedTimezoneSkipsOneLine` class pinning the per-line skip-and-log invariant.
- `solsys_code/allocation_projector.py` - new `_raise_if_set_window_inverted(run, night)` helper; called from the re-mint branch's `if dry_run:` short-circuit and the create branch's (replacing its inline duplicate); `legacy_urls_claimed.add(legacy_url)` moved ahead of the takeover branch's `_may_write()` check.
- `solsys_code/tests/test_allocation_projector.py` - re-mint twin of `test_dry_run_of_a_brand_new_inverted_window_also_raises`; new `TestTakeoverBlockedCountedOnce` class.
- `solsys_code/campaign_reconciler.py` - `_detach_stale_family_events()`'s return annotation corrected to `tuple[int, int, int, int]`.

## Decisions Made
- Split Task 2 and Task 3's overlapping edits to `allocation_projector.py`/`test_allocation_projector.py` into two fully independent commits (temporarily reverting Task 3's lines to isolate and verify Task 2's diff alone, then re-applying and committing Task 3 separately) rather than landing them together, to honour per-task atomic commits even though both tasks touch the same two files.
- Implemented `TestMalformedTimezoneSkipsOneLine` as a standalone `TestCase` class with its own minimal two-`Observatory` fixture (the plan's primary-preference option), rather than as a method on `TestLoadTelescopeRuns` -- avoids re-running that class's full existing fixture/test set under a new class name while still matching the plan's literal verify-command label.

## Deviations from Plan

None - plan executed exactly as written. All three tasks' regression tests were confirmed to fail against the pre-fix tree (RED) before implementing the fix (GREEN), per the plan's `tdd="true"` requirement.

## Issues Encountered
None. The full regression suite (1273 tests) and both pinned lint gates ran clean on the first pass after each task's fix.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- NF-21, NF-20, NF-22 and NF-23 -- the four remaining 35-REVIEW.md write-path warnings -- are closed in code, each with a regression test that failed against the pre-fix tree.
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` and `docs/runbooks/telescope_runs_calendar.rst` still owe the regeneration/correction this plan's objective explicitly deferred to **35-10-PLAN.md Tasks 1 and 2** and **35-11-PLAN.md Task 1** (both `depends_on` this plan) -- not part of this plan's `files_modified`.
- Plans 35-10 and 35-11 are untouched by this dispatch, as instructed.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-15*

## Self-Check: PASSED
- FOUND: solsys_code/management/commands/load_telescope_runs.py
- FOUND: solsys_code/tests/test_load_telescope_runs.py
- FOUND: solsys_code/allocation_projector.py
- FOUND: solsys_code/tests/test_allocation_projector.py
- FOUND: solsys_code/campaign_reconciler.py
- FOUND: commit 0eceeb5
- FOUND: commit eb79263
- FOUND: commit 0b7599f
