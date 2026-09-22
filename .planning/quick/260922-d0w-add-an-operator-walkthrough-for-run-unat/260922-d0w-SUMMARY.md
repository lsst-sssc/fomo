---
phase: 260922-d0w
plan: 01
subsystem: docs
tags: [runbook, sphinx, run_unattended, unattended-operation, docutils]

# Dependency graph
requires:
  - phase: 36-unattended-operation
    provides: solsys_code/unattended.py (STEPS registry, run_tick(), per-step StepResult summaries)
provides:
  - "Walking through a first tick" operator walkthrough subsection in docs/runbooks/telescope_runs_calendar.rst
  - Corrected --step reference in "Running it by hand" (now names all five steps)
  - Corrected WatchedProposal bookkeeping labels (Last swept at / Last sweep summary) in three places
  - Fixed pre-existing docutils inline-literal warning at the .gitignore'd construct
affects: [docs/runbooks/telescope_runs_calendar.rst, future unattended-operation runbook edits]

# Actuals (#2632)
actuals:
  tokens: 2507
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Runbook walkthrough sections pair a numbered operator step with an indented .. code-block:: console using the >> prompt, followed by a literal healthy-result block (::) and a prose does-not-do note -- matching the file's existing 'Setting it up on a fresh host' pattern"
    - "Cross-references to sibling subsections within this file use plain quoted-title text (see \"X\" above), never :ref:, matching the file's existing convention"

key-files:
  created: []
  modified:
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "New walkthrough content uses python3 manage.py throughout, matching the file's existing 27 occurrences of that form, rather than the task brief's literal python manage.py wording (within-document consistency)"
  - "Any example failure output shows only an exception's class name (e.g. 'raised ConnectionError'), matching run_tick()'s own catch-all behavior -- never a message, per T-d0w-03"
  - "Neither discovery nor proposal_allocation was run locally to harvest example output (both make credentialed portal calls); only the argparse-rejection check (--step bogus) was run to verify prose claims, per T-d0w-04"

requirements-completed: [SCHED-08, SCHED-09, SCHED-10, DISCOVER-01]

coverage:
  - id: D1
    description: "Operator walkthrough subsection lets a reader drive all five run_unattended steps individually under --dry-run and read each one's real result line and does-not-do note"
    requirement: "SCHED-08"
    verification:
      - kind: other
        ref: "sphinx-build zero-docutils-warning gate + grep-based --step-name coverage check (all 5 present in the new subsection)"
        status: pass
    human_judgment: true
    rationale: "Prose accuracy and operator readability (does the walkthrough actually make sense to someone unfamiliar with the runner) require human review; automated checks only confirm structure and literal-text correctness, not comprehensibility."
  - id: D2
    description: "Full dry-run tick documented with start/end banners, per-step lines, and exit-code reading, including that a dry run can still exit non-zero"
    requirement: "SCHED-09"
    verification:
      - kind: other
        ref: "sphinx-build zero-docutils-warning gate"
        status: pass
    human_judgment: true
    rationale: "Whether the exit-code explanation is clear to an operator following along is a judgment call, not something a grep/build check can confirm."
  - id: D3
    description: "--step flag semantics documented (argparse validation, --step alone does not suppress work) and stale --step reference in 'Running it by hand' corrected to include proposal_allocation"
    requirement: "SCHED-10"
    verification:
      - kind: other
        ref: "awk-scoped grep confirming 'proposal_allocation' appears in the 'Running it by hand' subsection"
        status: pass
    human_judgment: false
  - id: D4
    description: "Discovery step's dry-run behavior documented (no bookkeeping write, empty-list healthy case) and WatchedProposal admin labels corrected to Last swept at / Last sweep summary in three places"
    requirement: "DISCOVER-01"
    verification:
      - kind: other
        ref: "grep -c 'Last sweep summary' >= 2 and grep -c 'Last run summary|Last run at' == 0"
        status: pass
    human_judgment: false

duration: ~25min
completed: 2026-09-22
status: complete
---

# Quick Task 260922-d0w: Operator Walkthrough for run_unattended Summary

**Added a "Walking through a first tick" runbook subsection covering all five `run_unattended --dry-run --step <name>` invocations plus the full tick, and corrected three stale facts (a missing `--step` name and three wrong `WatchedProposal` admin labels) the new content exposed.**

## Performance

- **Duration:** ~25 min
- **Tasks:** 3
- **Files modified:** 1 (`docs/runbooks/telescope_runs_calendar.rst`)

## Accomplishments
- New "Walking through a first tick" subsection between "Adding a proposal to watch" and "The two failure signals", covering: adding a watched proposal, each of the five `run_unattended` steps individually with its real dry-run result line and a does-not-do note, a stderr-redirect note, a `check_unattended` pointer, the full dry-run tick (start/end banners, per-step lines, exit code, non-zero-despite-dry-run caveat), an example class-name-only failure line, `--step` flag semantics, and a "what to check afterwards" step.
- Fixed the pre-existing docutils "Inline literal start-string without end-string" warning at the ``.gitignore``d construct using RST's escaped-space form (``\ d``), clearing the zero-docutils-warning gate for the whole file.
- Corrected `--step` reference in "Running it by hand" to name all five steps (was missing `proposal_allocation`).
- Corrected three occurrences of stale `WatchedProposal` field-name labels (`Last run at`/`Last run summary`) to the admin's actual verbose names (`Last swept at`/`Last sweep summary`): the backfill section's per-row failure sentence, "Adding a proposal to watch"'s history sentence, and item 1 of "When nothing has appeared".

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end — new subsection heading, intro, and the first step, warning-free** - `314fd2e` (docs)
2. **Task 2: Expand — the remaining four steps, the full dry-run tick, and what to check afterwards** - `9c89cb4` (docs)
3. **Task 3: Correct the three stale facts in the neighbouring subsections and run the full gate** - `b5b5073` (docs)

_Note: this was a docs-only plan; all three commits use the `docs` type._

## Files Created/Modified
- `docs/runbooks/telescope_runs_calendar.rst` - New "Walking through a first tick" operator walkthrough subsection; fixed pre-existing docutils warning; corrected `--step` reference and three `WatchedProposal` bookkeeping labels

## Decisions Made
- Used `python3 manage.py` throughout the new content (matching the file's existing 27 occurrences), a deliberate deviation from the task brief's literal `python manage.py` wording, for within-document consistency (recorded in the plan's own `<planning_observations>`).
- The example failure line (`step project_sweep: FAILED | raised ConnectionError`) is grounded in `run_tick()`'s own outer `except Exception` handler (`solsys_code/unattended.py`), which builds `summary=f'raised {type(exc).__name__}'` -- the only failure-line shape the plan sanctioned for direct quotation, since it is the sole path guaranteed to never carry an exception message.
- Did not run `discovery` or `proposal_allocation` locally (both make credentialed LCO portal calls, per T-d0w-04); only ran the `--step bogus` argparse-rejection check, confirmed it fails before any step/lock/log line appears, and described that behavior in prose without quoting the version-specific error text.

## Deviations from Plan

None - plan executed exactly as written. The `python3 manage.py` invocation form and the `raised <ClassName>` failure-line grounding above are both explicitly pre-authorized by the plan's own `<planning_observations>` and Task 2 `<action>` text, not deviations from it.

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required. This is a docs-only change.

## Next Phase Readiness
- The unattended-operation section of the runbook now fully satisfies CLAUDE.md's paired-docs exemption for `unattended.py`, `notifications.py`, `run_unattended.py`, and `check_unattended.py` -- no notebook is added or regenerated.
- `pre-commit run --all-files` (including the Sphinx build hook and the Django test suite) passes clean; the runbook page builds with zero docutils-category warnings of its own (the pre-existing `[ref.doc]` `campaign_lifecycle_demo` warning remains, out of scope per the plan).
- No blockers. This quick task closes the milestone v2.4 documentation gap identified for the unattended path.

---
*Phase: 260922-d0w*
*Completed: 2026-09-22*

## Self-Check: PASSED

- FOUND: `docs/runbooks/telescope_runs_calendar.rst`
- FOUND: `.planning/quick/260922-d0w-add-an-operator-walkthrough-for-run-unat/260922-d0w-SUMMARY.md`
- FOUND commit `314fd2e` (Task 1)
- FOUND commit `9c89cb4` (Task 2)
- FOUND commit `b5b5073` (Task 3)
