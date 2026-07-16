---
phase: 23-weather-storm-cancellation-handling-give-staff-a-way-to-mark
plan: 01
subsystem: calendar
tags: [django, tom-calendar, management-command, jupyter]

# Dependency graph
requires:
  - phase: 02-03 (classical run ingest, v1.1)
    provides: load_telescope_runs command, insert_or_create_calendar_event() no-churn update path
  - phase: 08-09 (calendar visual clarity, v1.4)
    provides: calendar_display_extras._TERMINAL_PREFIXES (already includes '[CANCELLED]')
provides:
  - "[CANCELLED] title prefix on CalendarEvents for classical runs whose schedule line carries the cancelled status word"
  - Fresh-every-ingest title computation routed through the existing no-churn update path (revert-on-re-ingest works for free)
  - Paired demo notebook cell demonstrating the behavior with executed output
affects: [23-02 (LCO/queue cancellation handling, if it follows the same title-prefix idiom)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Status->title-prefix lookup via a fixed dict keyed on a validated enum value (parsed.status), never interpolated from raw line text — mirrors sync_lco_observation_calendar's _FAILURE_PREFIX_BY_STATUS/_title_for idiom"

key-files:
  created: []
  modified:
    - solsys_code/management/commands/load_telescope_runs.py
    - solsys_code/tests/test_load_telescope_runs.py
    - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb

key-decisions:
  - "Title is recomputed fresh from parsed.status/telescope/instrument on every handle() invocation (never appended to the stored event.title), so insert_or_create_calendar_event()'s existing field-diff update naturally reverts a stale [CANCELLED] prefix when the status word is later removed from the source line (RESEARCH Pitfall 4)."
  - "No templatetag change: '[CANCELLED]' was already a member of calendar_display_extras._TERMINAL_PREFIXES, so the terminal box-shadow ring is inherited for free."

requirements-completed: [D-01, D-02]

coverage:
  - id: D1
    description: "A cancelled classical-schedule line produces a CalendarEvent whose title begins with '[CANCELLED] '"
    requirement: D-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_load_telescope_runs.py#test_cancelled_line_gets_bracket_cancelled_title_prefix"
        status: pass
    human_judgment: false
  - id: D2
    description: "The other four KNOWN_STATUSES words leave the title unprefixed, unchanged from today's behavior"
    requirement: D-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_load_telescope_runs.py#test_non_cancelled_statuses_keep_unprefixed_title"
        status: pass
    human_judgment: false
  - id: D3
    description: "Re-ingesting the same line after the cancelled word is removed reverts the title to unprefixed in place, no duplicate row"
    requirement: D-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_load_telescope_runs.py#test_reingest_without_cancelled_reverts_title_prefix"
        status: pass
    human_judgment: false
  - id: D4
    description: "Paired demo notebook demonstrates the [CANCELLED] title prefix with real executed output"
    verification:
      - kind: other
        ref: "jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb (cell cc02ce02 output: 'Title      : [CANCELLED] NTT EFOSC2')"
        status: pass
    human_judgment: false

# Metrics
duration: 15min
completed: 2026-07-16
status: complete
---

# Phase 23 Plan 01: Classical-Run [CANCELLED] Title Prefix Summary

**Cancelled classical-schedule runs now render a `[CANCELLED] {telescope} {instrument}` title on the calendar, computed fresh every ingest and reverting cleanly when the status word is removed.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-07-16T20:56:29Z
- **Completed:** 2026-07-16T21:10:30Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- `load_telescope_runs.py` gains `_CLASSICAL_STATUS_PREFIX = {'cancelled': '[CANCELLED]'}` and a fresh-every-invocation title computation, wired into the existing `insert_or_create_calendar_event()` no-churn update path
- Three new Django tests cover the cancelled-prefix case, the four unprefixed statuses (each on its own night to avoid natural-key collision, per REVIEW finding #4), and the revert-on-re-ingest case (asserting both unchanged event count and surviving `pk`)
- Paired demo notebook (`load_telescope_runs_demo.ipynb`) gained a new markdown + code cell after the "Inspect the created CalendarEvent rows" section, executed end-to-end against the dev DB and committed with output showing `Title : [CANCELLED] NTT EFOSC2`

## Task Commits

Each task was committed atomically:

1. **Task 1: Add [CANCELLED] title prefix for cancelled classical runs, fresh every ingest** - `5254c3c` (feat)
2. **Task 2: Demonstrate the [CANCELLED] prefix in the paired demo notebook (executed output committed)** - `49873fa` (docs)

## Files Created/Modified
- `solsys_code/management/commands/load_telescope_runs.py` - `_CLASSICAL_STATUS_PREFIX` dict + fresh title computation in `Command.handle()`
- `solsys_code/tests/test_load_telescope_runs.py` - 3 new tests (`test_cancelled_line_gets_bracket_cancelled_title_prefix`, `test_non_cancelled_statuses_keep_unprefixed_title`, `test_reingest_without_cancelled_reverts_title_prefix`)
- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` - new markdown+code cell pair demonstrating the `[CANCELLED]` prefix, regenerated and committed with executed output

## Decisions Made
- Title recomputed fresh from `parsed.*` on every `handle()` call (never `event.title + ' [CANCELLED]'`) so the revert-on-re-ingest case works via the existing field-diff update path, with no special-casing needed.
- No change to `calendar_display_extras.py` — `'[CANCELLED]'` was already in `_TERMINAL_PREFIXES`.

## Deviations from Plan

None - plan executed exactly as written. `ruff-format` (via pre-commit) reformatted one line of the new notebook code cell for line length on first commit attempt; the reformatted content was re-executed via `jupyter nbconvert` and committed successfully on retry — this is standard pre-commit tooling behavior, not a deviation from the plan's intent.

## Issues Encountered
- Running `ruff format` directly (outside pre-commit) on the notebook surfaced a pre-existing, out-of-scope formatting drift in an unrelated cell (the Django-setup `assert` statement, cell `c3d4e5f6`), caused by a ruff-version skew between the CLI and the pre-commit-pinned version. Confirmed via `git show` that this drift predates this plan (present in the notebook before Task 2's edits). Left untouched per the deviation-rules scope boundary (pre-existing issue in unrelated code, not introduced by this plan).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `./manage.py test solsys_code` — all 523 tests pass (14 pre-existing + 3 new in `test_load_telescope_runs.py`), no regressions.
- `ruff check .` / `ruff format --check .` clean on both files this plan modified (`load_telescope_runs.py`, `test_load_telescope_runs.py`).
- Ready for plan 23-02 (parallel wave-1 sibling) or the next phase; no blockers.

---
*Phase: 23-weather-storm-cancellation-handling-give-staff-a-way-to-mark*
*Completed: 2026-07-16*

## Self-Check: PASSED

All created/modified files found on disk; all task commit hashes (5254c3c, 49873fa, 1f58c51) found in git log.
