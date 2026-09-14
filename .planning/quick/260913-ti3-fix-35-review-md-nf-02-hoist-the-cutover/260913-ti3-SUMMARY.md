---
phase: quick-260913-ti3
plan: 01
subsystem: management-command
tags: [django, cutover, management-command, tdd, cutover_classical_allocations]

# Dependency graph
requires:
  - phase: quick-260913-npq
    provides: "the two key_collision preconditions (in-run + existing-url) on both dry-run and real cutover branches, and the WR-08 window-containment check on the real branch alone (the gap this plan closes)"
provides:
  - "A single _check_event_night() helper applying all three per-event preconditions (window containment, in-run collision, existing-url collision) on both the --dry-run and real cutover branches"
  - "A dedicated window_mismatch reason category, distinct from the generic 'other' bucket, on both paths"
  - "A dry-run branch that reads its window bounds from the previewed fields dict rather than a possibly-stale pre-existing CampaignRun row"
  - "A regression-guard test (TestDryRunAndRealRunAgree) proving dry-run/real parity across all three preconditions in one fixture"
  - "Operator runbook coverage of the window_mismatch reason and the dry-run/real parity contract"
affects: [cutover_classical_allocations, campaign_reconciler, allocation_projector]

# Actuals (#2632)
actuals:
  tokens: 8001
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Hoisted per-event precondition helper (_check_event_night()) shared by two structurally-different call sites (dry-run preview loop vs. real per-event savepoint) to make future drift between the two impossible by construction rather than by discipline"
    - "Window bounds passed as explicit parameters rather than read off a possibly-stale ORM row, so both branches provably evaluate the same window"

key-files:
  created: []
  modified:
    - solsys_code/management/commands/cutover_classical_allocations.py
    - solsys_code/tests/test_cutover_classical_allocations.py
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "_check_event_night() takes window_start/window_end as explicit parameters (not read off `run`) so the dry-run branch can pass fields['window_start']/fields['window_end'] -- the freshly re-derived window -- even when run is a stale pre-existing row or None"
  - "The existing-ALLOC-url probe inside the helper is skipped only when run is None (dry-run path, no CampaignRun yet); window containment and in-run collision always apply on both paths"
  - "window_mismatch reported via a dedicated exception (_WindowMismatchError) with its own except clause, mirroring the existing _KeyCollisionError pattern, so both remain distinguishable from the generic catch-all"

requirements-completed: [NF-02]

coverage:
  - id: D1
    description: "A legacy blank-url event whose derived night falls outside the run's window is reported as window_mismatch (not the generic 'other' bucket) on both --dry-run and the real path"
    requirement: "NF-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_cutover_classical_allocations.py#TestDryRunAppliesTheWindowCheck.test_dry_run_reports_window_mismatch_and_does_not_count_it_as_rekeyed"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_cutover_classical_allocations.py#TestDryRunAppliesTheWindowCheck.test_real_run_also_reports_window_mismatch_not_the_generic_bucket"
        status: pass
    human_judgment: false
  - id: D2
    description: "--dry-run applies all three per-event preconditions the real path applies, so a dry run can no longer exit 0 over a fixture the immediately following real run rejects"
    requirement: "NF-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_cutover_classical_allocations.py#TestDryRunAndRealRunAgree.test_dry_run_and_real_run_report_identical_counts_and_reasons"
        status: pass
    human_judgment: false
  - id: D3
    description: "The dry run and the immediately following real run agree on the re-key count, unexplained count, every per-reason breakdown line, and exit status, over a fixture tripping all three preconditions at once (window containment, in-run collision, existing-url collision)"
    requirement: "NF-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_cutover_classical_allocations.py#TestDryRunAndRealRunAgree.test_dry_run_and_real_run_report_identical_counts_and_reasons"
        status: pass
    human_judgment: false
  - id: D4
    description: "An out-of-window event is left byte-identical on both paths (url stays blank, no CalendarEventMeta row created), including under a deliberately wider pre-existing run window"
    requirement: "NF-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_cutover_classical_allocations.py#TestDryRunAndRealRunAgree.test_out_of_window_event_is_byte_identical_after_both_passes"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_cutover_classical_allocations.py#TestDryRunAppliesTheWindowCheck.test_dry_run_reports_window_mismatch_and_does_not_count_it_as_rekeyed"
        status: pass
    human_judgment: false
  - id: D5
    description: "The three preconditions live in exactly one helper (_check_event_night()) called by both branches; the real path's conversion behavior is otherwise unchanged; every pre-existing test class stays green"
    requirement: "NF-02"
    verification:
      - kind: unit
        ref: "python manage.py test solsys_code.tests.test_cutover_classical_allocations (26 tests, OK)"
        status: pass
      - kind: other
        ref: "grep -c 'window_start <= night' over non-comment code == 1; grep -c '_check_event_night(' over non-comment code >= 3"
        status: pass
    human_judgment: false
  - id: D6
    description: "Module docstring's reason vocabulary and the operator runbook's cutover reason list both name window_mismatch with its operator action, in the style of the existing key_collision/foreign_attribution entries"
    requirement: "NF-02"
    verification:
      - kind: other
        ref: "grep -c 'window_mismatch' docs/runbooks/telescope_runs_calendar.rst == 4 (>= 2 required)"
        status: pass
    human_judgment: false
  - id: D7
    description: "Paired demo notebook audit: cutover cells in reconcile_campaign_runs_demo.ipynb only exercise no_source_line output, unaffected by this change; no re-execution needed"
    verification:
      - kind: other
        ref: "manual notebook cell inspection (see Deviations/Paired Notebook Audit section below)"
        status: pass
    human_judgment: false

duration: 8min
completed: 2026-09-14
status: complete
---

# Phase quick-260913-ti3 Plan 01: Hoist the Cutover Window Check onto Both Paths Summary

**Hoisted `cutover_classical_allocations`'s three per-event preconditions (window containment, in-run collision, existing-`ALLOC:`-url collision) into a single `_check_event_night()` helper called by both `--dry-run` and the real path, giving the out-of-window case its own `window_mismatch` reason instead of the generic `other` bucket — closing 35-REVIEW.md NF-02.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-09-14T04:17:40Z
- **Completed:** 2026-09-14T04:25:45Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- `_check_event_night()` module-level helper now performs, in order, window containment, in-run collision, and existing-ALLOC-url collision, taking `window_start`/`window_end` as explicit parameters rather than reading them off `run` — this is what lets the dry-run branch evaluate the same window the real pass would write (`fields['window_start']`/`fields['window_end']`), even before any `CampaignRun` row exists or when a stale pre-existing row's window differs.
- New `_WINDOW_MISMATCH = 'window_mismatch'` reason constant, `_REASON_LABELS` entry, and `_WindowMismatchError` exception, routed through a dedicated `except` clause on both branches ahead of the generic catch-all.
- Both per-event loops (dry-run preview and real per-event savepoint) now call the same helper and nothing else for their checks — the window comparison and both collision probes exist in exactly one place in the module.
- `TestDryRunAppliesTheWindowCheck` (RED verified before implementation: dry run exited 0, real run reported the generic bucket) pins the fix on the WR-08 fixture directly.
- `TestDryRunAndRealRunAgree` is a structural regression guard: one fixture trips all three preconditions at once (a pre-existing run with a *deliberately wider* window than the schedule line implies, an existing-url collision, an in-run collision, and a window-mismatch outlier), and a single `assertEqual` compares a parsed dry-run summary structure against a parsed real-run summary structure — any future check added to only one branch breaks this test immediately.
- Runbook (`docs/runbooks/telescope_runs_calendar.rst`) updated in both operator-facing reason lists, naming `window_mismatch` with its operator action, plus a new sentence recording the dry-run/real parity contract.

## Task Commits

Each task was committed atomically:

1. **Task 1: Hoist the three per-event preconditions into one helper and give the window mismatch its own reason** - `94f6c85` (fix)
2. **Task 2: Pin dry-run/real parity as a regression guard across all three preconditions** - `1cb39a4` (test)
3. **Task 3: Add the new reason to the operator runbook, audit the paired notebook, and run the quality gates** - `bc15c4d` (docs)

**Plan metadata:** SUMMARY.md/STATE.md commit handled by the orchestrator (not this executor, per Quick task convention).

_Note: Task 1 followed RED→GREEN — `TestDryRunAppliesTheWindowCheck` was written and confirmed failing (both assertions, for the stated reasons) before any implementation change, then implementation made it green alongside all 22 pre-existing tests._

## Files Created/Modified

- `solsys_code/management/commands/cutover_classical_allocations.py` — added `_WINDOW_MISMATCH`/`_REASON_LABELS` entry, `_WindowMismatchError`, `_check_event_night()` helper; rewrote both per-event loops to call it; updated module docstring.
- `solsys_code/tests/test_cutover_classical_allocations.py` — added `TestDryRunAppliesTheWindowCheck` (Task 1) and `TestDryRunAndRealRunAgree` plus `_parse_cutover_summary()` (Task 2); added `import re`.
- `docs/runbooks/telescope_runs_calendar.rst` — named `window_mismatch` in the "Always run `--dry-run` first" enumeration (with its Cause/Fix framing and a new dry-run parity sentence) and the "A reported unexplainable event during the classical cutover" troubleshooting section's Cause/Fix.

## Decisions Made

- `_check_event_night()`'s window bounds are parameters, never read off `run`, so both branches provably evaluate the same window — this is the structural fix, not a copy of a third check onto the dry-run branch (matches the plan's explicit intent).
- The existing-ALLOC-url probe (the third precondition) is skipped inside the helper only when `run is None` — legitimate, since a run with no primary key can hold no `ALLOC:` url; this asymmetry preserves parity rather than breaking it.
- `window_mismatch` gets its own exception class (`_WindowMismatchError`) rather than reusing `_KeyCollisionError` with a flag, mirroring the existing pattern for maintainability and matching the plan's `except`-ladder ordering requirement.

## Deviations from Plan

None - plan executed exactly as written. The RED-phase test failures matched exactly what the plan predicted (dry run exits 0; real run names the generic `other` bucket), and the `TestDryRunAndRealRunAgree` fixture produced the exact expected counts (`events re-keyed: 2`, `unexplained: 3` with `key_collision: 2`/`window_mismatch: 1`) on the first run, with no debugging needed.

### Paired Notebook Audit (required by CLAUDE.md paired-docs rule)

`docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` is the paired notebook for `cutover_classical_allocations.py`. Audited both cells that invoke the command:

- Cell 6 (`--dry-run`): recorded stdout is `Done (dry run). candidates: 10, groups: 3, runs created: 3, updated: 0, unchanged: 0, events re-keyed: 9, unexplained: 1` with `unexplained (no_source_line): 1 -- no parseable Source line: marker`. No `key_collision` or `window_mismatch` output present.
- Cell 7 (real run): recorded stdout is `Done. candidates: 10, groups: 3, runs created: 3, updated: 0, unchanged: 0, events re-keyed: 9, unexplained: 1` with the same `no_source_line` breakdown line, no other reason categories.

Both cells' committed output exercises only the `no_source_line` reason category on the notebook's fixture data. This change alters neither the code path that produces `no_source_line` output nor any other output these two cells exhibit. **Outcome: confirmed unaffected — no re-execution needed.** The notebook was not touched, added to the commit, or included in the diff, consistent with the plan's conditional `files_modified` entry.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 35-REVIEW.md NF-02 (BLOCKER) is closed: `--dry-run` can no longer exit 0 over a fixture the immediately following real run rejects.
- The command module's per-event precondition surface is now single-sourced (`_check_event_night()`), so any future fourth precondition (should one arise) cannot be added to only one branch without the existing `TestDryRunAndRealRunAgree` regression guard catching the drift.
- No blockers for Phase 35 verification/UAT; this quick task closes a review finding from `35-REVIEW.md` ahead of that.

---
*Phase: quick-260913-ti3*
*Completed: 2026-09-14*

## Self-Check: PASSED

All 3 changed files found on disk; all 3 task commits (`94f6c85`, `1cb39a4`, `bc15c4d`) found in git history. `commits: 3` matches `git rev-list --count 634fa96..HEAD` measured from the plan-head ledger; `plan_head_before: 634fa96b185a9e9b4d2af45ae76bf2c6cc9fffc2`.
