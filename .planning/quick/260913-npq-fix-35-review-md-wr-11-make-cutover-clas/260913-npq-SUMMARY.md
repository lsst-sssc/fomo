---
phase: quick-260913-npq
plan: 01
subsystem: campaign-management
tags: [django, management-command, calendar, campaign-run, cutover, wr-11]

# Dependency graph
requires:
  - phase: 35
    provides: "cutover_classical_allocations command (D-17/D-18), ALLOC:-keyed allocation nights, source_identifier"
provides:
  - "Dedicated key_collision reason category in cutover_classical_allocations, refusing to write a duplicate ALLOC: url"
  - "In-run night-claim check (per group, never shared across groups) and existing-url probe on both the real and --dry-run paths"
affects: [35-allocation-layer-classical-cutover, load_telescope_runs, campaign_reconciler]

actuals:
  tokens: 5506
  tasks: 3
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Raise-inside-savepoint for a new unexplainable reason (_KeyCollisionError), caught by a dedicated except clause placed BEFORE the existing broad except Exception, so the failure routes to its own named reason category instead of falling into the generic 'other' bucket."
    - "Per-group claimed-nights set populated only after a savepoint commits successfully, so a later event can still claim a night an earlier event failed to convert for an unrelated reason."

key-files:
  created: []
  modified:
    - solsys_code/management/commands/cutover_classical_allocations.py
    - solsys_code/tests/test_cutover_classical_allocations.py
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "key_collision is a dedicated reason category, not folded into 'other' -- it calls for a distinct operator action (delete or re-attribute the duplicate row) that a generic unexpected-error reason cannot communicate."
  - "The exception class is named _KeyCollisionError (not _KeyCollision as originally planned) to satisfy ruff's N818 naming rule (Exception subclasses require an Error suffix) -- caught during Task 3's quality-gate run, fixed before commit."
  - "The existing-url probe is skipped on the dry-run path when the group's CampaignRun does not exist yet (run is None), since a run with no primary key can hold no ALLOC: url in the table -- matches the plan's own key_links guidance."

patterns-established:
  - "A new D-18 unexplainable reason is added by: (1) a module-level string constant placed in report order, (2) a _REASON_LABELS entry, (3) a private exception subclass with an Error-suffixed name, (4) a dedicated except clause ahead of the broad except Exception, and (5) a docstring sentence naming the case and the operator action."

requirements-completed: [WR-11]

coverage:
  - id: D1
    description: "Two blank-url legacy events sharing a group and the same observing night: the first claimant is re-keyed, the second is reported under key_collision and left byte-identical (url='', no CalendarEventMeta row), non-zero exit"
    requirement: WR-11
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_cutover_classical_allocations.py::TestKeyCollisionDetection::test_in_run_collision_rekeys_the_first_claimant_reports_the_second"
        status: pass
    human_judgment: false
  - id: D2
    description: "A legacy event whose derived ALLOC:{run_pk}:{night} url is already held by a different CalendarEvent row (import-ran-first case) is reported under key_collision and left byte-identical; the pre-existing row is untouched"
    requirement: WR-11
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_cutover_classical_allocations.py::TestKeyCollisionDetection::test_existing_url_collision_reports_legacy_event_leaves_pre_existing_row_untouched"
        status: pass
    human_judgment: false
  - id: D3
    description: "--dry-run previews the same collision outcome with no writes: detects the in-run collision, counts it as unexplained not a would-be re-key, and its events-re-keyed count equals what the real run then performs on the same fixture"
    requirement: WR-11
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_cutover_classical_allocations.py::TestKeyCollisionDetection::test_dry_run_predicts_in_run_collision_without_writing"
        status: pass
    human_judgment: false
  - id: D4
    description: "Every pre-existing test class in test_cutover_classical_allocations.py stays green, unmodified -- a group with no colliding nights converts exactly as it did before"
    verification:
      - kind: unit
        ref: "python manage.py test solsys_code.tests.test_cutover_classical_allocations (22/22 tests pass)"
        status: pass
    human_judgment: false
  - id: D5
    description: "The runbook's cutover section names the collision reason with its operator action and states the cutover-before-import ordering; the troubleshooting entry covers it too"
    requirement: WR-11
    verification:
      - kind: manual_procedural
        ref: "docs/runbooks/telescope_runs_calendar.rst, 'How do I run the one-time classical cutover?' and 'A reported unexplainable event during the classical cutover'"
        status: pass
    human_judgment: true
    rationale: "Prose-quality/clarity judgment (does the operator language read clearly and match the printed reason token) is a human call; the plan's own <verify> block treats this section as a human-check item."

duration: ~35min
completed: 2026-09-13
status: complete
---

# Quick Task 260913-npq: Detect ALLOC: Key Collisions in the Classical Cutover Summary

**Closed 35-REVIEW.md WR-11: `cutover_classical_allocations` now refuses to silently write
a second event onto an already-claimed `ALLOC:{run_pk}:{night}` url, reporting the
collision under its own named reason instead.**

## Performance

- **Duration:** ~35 min
- **Tasks:** 3 (all completed)
- **Files modified:** 3 (`cutover_classical_allocations.py`, `test_cutover_classical_allocations.py`, `telescope_runs_calendar.rst`)

## Accomplishments

- Added a dedicated `key_collision` reason category to `cutover_classical_allocations`,
  distinct from the generic `other` bucket, with its own `_REASON_LABELS` entry and a
  code comment justifying the split (an operator needs a different action -- delete or
  re-attribute the duplicate row -- than for a generic unexpected error).
- Added a private `_KeyCollisionError` exception (renamed from the plan's suggested
  `_KeyCollision` to satisfy ruff's N818 naming rule) raised inside the existing per-event
  savepoint (real path) or a new read-only preview loop (dry-run path), always checked
  BEFORE any write so the losing event stays byte-identical.
- Added a per-group `claimed_nights: set[date]`, created fresh for every group (never
  shared across groups, since the url is keyed by run primary key too) and populated only
  after a night's write/preview succeeds -- so an event that failed to re-key for some
  other reason never blocks a later event from claiming the same night.
- Added an existing-url probe
  (`CalendarEvent.objects.filter(url=url).exclude(pk=event.pk).first()`) on both paths,
  skipped on the dry-run path when the group's `CampaignRun` does not exist yet (`run is
  None`), since a run with no primary key can hold no `ALLOC:` url in the table.
- Replaced the dry-run path's unconditional `events_rekeyed += len(writable_events)` with
  a read-only loop that applies the identical two collision checks and only counts an
  event that claims its night cleanly.
- Extended the module docstring's reason list to name the collision case and its
  rationale (`CalendarEvent.url` carries no unique constraint).
- Wrote 3 new tests FIRST in `TestKeyCollisionDetection` (TDD RED confirmed against the
  unmodified command before implementing): in-run collision, existing-url collision
  (import-ran-first), and `--dry-run` prediction. All pass GREEN after implementation;
  every pre-existing test class (19 tests across 11 other classes) stays green unmodified
  -- 22/22 total.
- Updated `docs/runbooks/telescope_runs_calendar.rst`: named `key_collision` in the
  cutover section's operator-facing reason list with its operator action, pinned the
  cutover-before-import ordering as an explicit sentence with the numbered step list, and
  extended the troubleshooting entry's **Cause**/**Fix** to cover it.
- Completed the mandatory paired-docs audit (CLAUDE.md) of
  `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`'s committed cutover
  cells (see "Paired-Notebook Audit" below) -- confirmed no re-execution was needed.

## Task Commits

Each task was committed atomically:

1. **Task 1: Detect ALLOC: key collisions in the cutover and report them instead of writing
   a duplicate** - `556c6c0` (fix) -- includes both the RED-then-GREEN test class and the
   command implementation in one commit, since the plan scoped both files to the same
   task's `<files>` list and both were needed together for the task's own `<verify>` step
   to pass.
2. **Task 2: Document the new reason and pin the cutover-before-import ordering in the
   runbook, and audit the paired notebook** - `338625d` (docs)
3. **Task 3: Run the project's quality gates on the changed files** - no separate commit;
   `pre-commit run ruff` and `ruff-format` were already clean against the tree left by
   Task 1's second (corrected) commit, and `python manage.py test
   solsys_code.tests.test_cutover_classical_allocations` passed 22/22 on the final tree.
   Verified, nothing further to stage.

**Plan metadata:** SUMMARY.md and STATE.md updates handled by the orchestrator after this
report (per this executor's constraints -- docs artifacts are not committed here).

_Note: TDD task (Task 1) had two attempts at the commit step -- the first `git commit`
invocation was rejected by pre-commit's pinned ruff (N818: an `Exception` subclass name
must end in `Error`) and reformatted one line via `ruff-format`; both were fixed
(`_KeyCollision` renamed to `_KeyCollisionError` throughout the command module) and a
fresh commit created, per the deviation rules' "fix inline, re-verify, continue" protocol
for Rule 3 (blocking issue)._

## Files Created/Modified

- `solsys_code/management/commands/cutover_classical_allocations.py` - `_KEY_COLLISION`
  reason constant, `_REASON_LABELS` entry, `_KeyCollisionError` exception, per-group
  `claimed_nights` set, in-run and existing-url collision checks on both the real and
  `--dry-run` paths, extended module docstring.
- `solsys_code/tests/test_cutover_classical_allocations.py` - new
  `TestKeyCollisionDetection` class (3 tests: in-run collision, existing-url collision,
  dry-run prediction).
- `docs/runbooks/telescope_runs_calendar.rst` - `key_collision` named in the cutover
  section's reason list with its operator action, cutover-before-import ordering sentence,
  troubleshooting entry extended.

## Decisions Made

- **`_KeyCollisionError`, not `_KeyCollision`.** The plan's own `<action>` text suggested
  `_KeyCollision`; ruff's N818 (`Exception name should be named with an Error suffix`)
  rejected that at the pre-commit gate on the first commit attempt. Renamed throughout
  the command module (the exception is private to this module and never referenced by the
  test file, so the rename was a single-file `sed`). This is the deviation history the
  plan's own naming suggestion didn't anticipate -- documented here rather than silently
  diverging from the plan's stated name.
- **The night-derivation exception guard in the dry-run loop deliberately mirrors the real
  path's generic `except Exception` treatment** (an `observing_night()` failure -- e.g. an
  ambiguous local time -- is reported under `_OTHER`, not `_KEY_COLLISION`), per the plan's
  "Guard the night derivation with the same report-not-crash treatment the real path gives
  it" instruction.
- **No `UniqueConstraint` was added to `tom_calendar.CalendarEvent`.** Per the plan's own
  scope lock, this remains explicitly out of scope; the fix lives entirely in this
  command's own per-event checks.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Renamed `_KeyCollision` to `_KeyCollisionError` for ruff N818**
- **Found during:** Task 1's first commit attempt (pre-commit hook)
- **Issue:** ruff's N818 naming rule rejected `class _KeyCollision(Exception)` --
  exception subclasses must carry an `Error` suffix. This is a pre-commit-pinned ruff
  rule, not a project convention the plan's own suggested name anticipated.
- **Fix:** Renamed the class and every reference (`raise`, `except`, docstring mention)
  to `_KeyCollisionError` via a targeted `sed` across the one file that defines/uses it.
- **Files modified:** `solsys_code/management/commands/cutover_classical_allocations.py`
- **Verification:** `pre-commit run ruff` and `ruff-format` both pass; full test module
  re-run, 22/22 green.
- **Committed in:** `556c6c0` (the corrected, successful Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking, Rule 3)
**Impact on plan:** Cosmetic rename only, forced by a pre-commit lint rule; no behavior,
test, or scope change. No scope creep.

## Issues Encountered

None beyond the ruff N818 naming fix documented above.

## Paired-Notebook Audit (CLAUDE.md, plan Task 2 step 4)

Inspected `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`'s two committed
cutover cells and their recorded output directly from the notebook JSON:

- **Cell 6** (`--dry-run` `call_command('cutover_classical_allocations', ...)`): recorded
  stdout reads `candidates: 10, groups: 3, runs created: 3, updated: 0, unchanged: 0,
  events re-keyed: 9, unexplained: 1` with `unexplained (no_source_line): 1` as the only
  reason line; stderr names `pk=334 ('tmp')`.
- **Cell 7** (real `call_command('cutover_classical_allocations', ...)`): recorded stdout
  reads the same `events re-keyed: 9, unexplained: 1`, same single `no_source_line` reason
  line, same `pk=334` on stderr.
- **Cell 12**'s downstream assertions independently confirm the end-state: zero
  date-bearing `RUN:` nights, the only remaining blank-url row is the reported
  unexplained one (pk 334), bare containers unchanged in count, every facility-url event
  byte-identical.

The only reason category present in either cell's recorded output is `no_source_line`
(the known `tmp` junk row); `key_collision` does not appear anywhere. The dry-run and real
run's `events re-keyed` counts are equal (9 == 9), which is exactly the invariant that
would break if any night on this database had been claimed twice -- confirming
independently, not just citing quick task 260913-ng8's prior evidence, that this database's
cutover run contains zero collisions of either kind this change detects.

**Outcome: no re-execution needed.** The notebook file is unchanged by this quick task.

## WR-08 Dry-Run Non-Goal (plan Task 1 step 5, recorded as required)

Per the plan's explicit instruction, this task did NOT extend the dry-run path with the
WR-08 window-containment check. The dry-run path still does not verify that an event's
derived night falls inside the run's own window before counting it as a would-be re-key,
so a dry run can over-count a would-be re-key for an out-of-window event (the real run
would then report that same event as `_OTHER` via the window-containment `ValueError`).
This gap predates WR-11, is out of this task's scope, and was deliberately left as-is --
not fixed here.

## Next Phase Readiness

- WR-11 is closed; the cutover command's collision surface is now fully covered by named,
  operator-actionable reasons.
- No blockers for Phase 35 verification or the next milestone step.
- The pre-existing WR-08 dry-run non-goal above remains a known, documented gap for a
  future quick task if ever prioritized -- not introduced or worsened by this change.

---
*Phase: quick-260913-npq*
*Completed: 2026-09-13*

## Self-Check: PASSED

All created/modified files found on disk; both task commits (`556c6c0`, `338625d`) found
in git history.
