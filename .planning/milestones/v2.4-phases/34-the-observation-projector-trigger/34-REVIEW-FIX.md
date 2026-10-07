---
phase: 34-the-observation-projector-trigger
fixed_at: 2026-09-11T22:20:47Z
review_path: .planning/phases/34-the-observation-projector-trigger/34-REVIEW.md
iteration: 1
findings_in_scope: 14
fixed: 12
skipped: 2
status: partial
---

# Phase 34: Code Review Fix Report

**Fixed at:** 2026-09-11T22:20:47Z
**Source review:** .planning/phases/34-the-observation-projector-trigger/34-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope (Critical + Warning, per `fix_scope`): 14 (3 critical, 11 warning; REVIEW.md
  reports 0 info findings)
- Fixed: 12
- Skipped: 2 (WR-02, WR-11 -- both explicitly marked out of scope for this run by the
  orchestrator's project constraints, not fix failures)

All fixes were applied inside an isolated git worktree
(`.claude/worktrees/rf-34-1710532-1789163316`, branch `gsd-reviewfix/34-1710532`) and fast-forward
merged onto `issue37-telescope-runs-calendar` on cleanup, per `workflow.use_worktrees=true`.

## Fixed Issues

### CR-01: `coerce_schedule_datetime()` never converts an aware non-UTC value to UTC

**Files modified:** `solsys_code/calendar_utils.py`, `solsys_code/tests/test_calendar_utils.py`,
`solsys_code/tests/test_observation_projector.py`
**Commit:** `1d6ef6c`
**Applied fix:** Changed the aware-datetime branch to `return value.astimezone(dt_timezone.utc)`
instead of returning the value unconverted. Updated the `Returns:` docstring to drop the
"already-aware datetime is returned as-is" claim. Renamed
`test_aware_datetime_is_returned_with_value_and_tzinfo_unchanged` to
`test_aware_non_utc_datetime_is_converted_to_utc_with_the_same_instant` and updated its
assertions to check the converted UTC tzinfo. Added a regression test in
`test_observation_projector.py` (`test_non_utc_offset_schedule_renders_utc_wall_clock_in_description`)
asserting `event_fields_for()`'s description renders the UTC wall clock for a `-04:00` input.

### CR-02: `record_time_window()`'s parameters branch rejects `Z`-suffixed values on Python 3.10

**Files modified:** `solsys_code/calendar_utils.py`, `solsys_code/observation_projector.py`,
`solsys_code/tests/test_calendar_utils.py`
**Commit:** `d996b21`
**Applied fix:** Replaced `datetime.fromisoformat(...).replace(tzinfo=utc)` with
`coerce_schedule_datetime(...)` in both `record_time_window()`'s parameters-fallback branch and
`observation_projector.py`'s duplicate `'inconsistent'`-stage branch (removing the now-unused
`datetime`/`dt_timezone` imports there and adding `coerce_schedule_datetime` to the import list).
Added `test_both_scheduled_none_falls_back_to_z_suffixed_parameters_start_end` with a
`'2026-07-20T00:00:00Z'`-shaped `parameters['start']`/`['end']`.

### CR-03: notebook SCHED-06 cell published scratch-copy state under the baseline heading

**Files modified:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`
**Commit:** `46d8390` (combined with WR-06/WR-07/WR-08/WR-09 -- see note below)
**Applied fix:** On a scratch-routed run, the SCHED-06 cell now substitutes the committed
baseline file's own `records` for `baseline_records` before the following per-record table
renders, instead of showing the scratch copy's already-swept rows under the "SCHED-06 baseline"
heading. Re-executed the notebook against the scratch DB copy from plan 34-06
(`tmp/fomo_g34_2_copy.sqlite3`, `FOMO_DATABASE_PATH`-routed, cwd
`docs/notebooks/pre_executed/`, run in the isolated worktree -- see Verification below). The
cell's output now reports `record_count=74`, stage tally `{'placed': 18, 'queued': 56}`, matching
`project_observation_calendar_demo.sched06-baseline.json` exactly (same `captured_at`). The
committed baseline JSON file's checksum is unchanged before and after
(`c85e176eb048178bfdabcd159f03c80f`); the developer database (`src/fomo_db.sqlite3`) was never
opened or written by this fix (mtime/size verified unchanged:
`1789157248 1232896`).

### WR-01: the diagnostic `ValueError` message is never logged, only the exception class name

**Files modified:** `solsys_code/observation_projector.py`
**Commit:** `1f6786b`
**Applied fix:** Both `project_record()`'s and `project_queryset()`'s `except`-and-log sites now
log `'%s: %s', type(exc).__name__, exc` instead of just the class name, so
`coerce_schedule_datetime()`'s diagnostic message (naming the rejected value) reaches the log.

### WR-03: a bare ISO date string is silently accepted as midnight

**Files modified:** `solsys_code/calendar_utils.py`, `solsys_code/tests/test_calendar_utils.py`
**Commit:** `38aeb0d`
**Applied fix:** `coerce_schedule_datetime()` now rejects any string of 10 characters or fewer
(the bare `YYYY-MM-DD` form) with `ValueError: Schedule value is a date, not a datetime: ...`,
matching the function's existing "never silently degrade" contract. Documented in the `Raises:`
block; added `test_bare_iso_date_string_raises_value_error`.

### WR-04: the non-`str`/non-`datetime` raise branch and end-to-end `unprojectable` contract were untested

**Files modified:** `solsys_code/tests/test_calendar_utils.py`,
`solsys_code/tests/test_observation_projector_signals.py`
**Commit:** `bac6161`
**Applied fix:** Added `test_non_string_non_datetime_value_raises_value_error` (covers the
`elif not isinstance(value, datetime)` branch with an int and a `date`). Added
`test_updatestatus_with_unparseable_schedule_value_leaves_the_event_untouched` to
`TestUpdateObservationStatusPath`, using a bare ISO date (`'2026-09-16'`) as the fake facility's
`scheduled_start` -- chosen deliberately because Django's own `DateTimeField.to_python()` accepts
a bare date (as midnight) and so the value reaches the `post_save` receiver as a genuine
in-memory string, unlike a truly garbage string which Django's own field validation rejects
before the receiver ever runs. Asserts the save does not raise, the pre-existing `[Q]` event span
is untouched, and `'unprojectable'` is logged.

### WR-05: `record_time_window()`'s return annotation disagrees with the helper it calls

**Files modified:** `solsys_code/calendar_utils.py`
**Commit:** `e5b6a13`
**Applied fix:** `cast(datetime, coerce_schedule_datetime(...))` on both calls in the
both-populated (`scheduled_start`/`scheduled_end`) branch, narrowing the type now that the
guarding `elif` has already proven neither value is `None`.

### WR-06: notebook's dead `existing_baseline` read and two now-false comments

**Files modified:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`
**Commit:** `46d8390` (combined with CR-03/WR-07/WR-08/WR-09)
**Applied fix:** `existing_baseline['records']` is now consumed (via CR-03's fix) instead of
discarded. Removed the false "both branches build ... identically" comment and replaced it with
an accurate one. Corrected the "What happens next" markdown cell's parenthetical from "(which
this run has just overwritten)" to "(which an un-routed run overwrites; a scratch-routed run
leaves it alone -- see below)".

### WR-07: the scratch branch opens the baseline JSON with no existence guard

**Files modified:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`
**Commit:** `46d8390` (combined with CR-03/WR-06/WR-08/WR-09)
**Applied fix:** Added an explicit `RuntimeError` naming the missing committed artifact before
opening `SCHED06_BASELINE_PATH` in the scratch branch, matching every other precondition in this
notebook.

### WR-08: the scratch-copy guard can be satisfied by a relative path that still opens the dev DB

**Files modified:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`
**Commit:** `46d8390` (combined with CR-03/WR-06/WR-07/WR-09)
**Applied fix:** Replaced the tautological `assert` (`resolved_db_name == SCRATCH_DB_OVERRIDE`,
always true when the env var is set) with a resolved-path comparison
(`Path(resolved_db_name).resolve() == dev_db_path.resolve()`) and a `raise RuntimeError` (not
`assert`, which `python -O` strips) if they match.

### WR-09: `SCRATCH_DB_OVERRIDE` is a cross-cell global consumed eleven cells later

**Files modified:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`
**Commit:** `46d8390` (combined with CR-03/WR-06/WR-07/WR-08)
**Applied fix:** The SCHED-06 cell now re-reads `os.environ.get('FOMO_DATABASE_PATH') or None`
itself rather than relying on the Django-setup cell's earlier binding, so the guard is
self-contained against a fresh-kernel re-run of just that cell.

### WR-10: the "in-memory instance" test used a real database INSERT

**Files modified:** `solsys_code/tests/test_calendar_utils.py`
**Commit:** `d21b293`
**Applied fix:** `test_in_memory_instance_with_portal_iso_strings_returns_aware_utc_pair` now
constructs the `ObservationRecord` directly (`ObservationRecord(...)`, no `.objects.create()`),
so no INSERT and no `post_save` signal fire -- pinning the actual in-memory contract the test's
name and docstring claim, rather than relying on Django happening not to refresh field values
after `save()`.

## Skipped Issues

### WR-02: `coerce_schedule_datetime()` duplicates `_parse_datetime_value()`

**File:** `solsys_code/calendar_utils.py:460-506` vs
`solsys_code/management/commands/backfill_lco_observations.py:81-107`
**Reason:** Explicitly out of scope per the orchestrator's project constraints for this run:
consolidating with `_parse_datetime_value()` would pull `backfill_lco_observations.py`'s paired
notebook (`backfill_lco_observations_demo.ipynb`) into scope under CLAUDE.md's paired-docs rule,
and plan 34-05 already explicitly decided against that consolidation. Not attempted.

### WR-11: signal tests hand-roll monkeypatching instead of `patch.object`, and triplicate a fixture

**File:** `solsys_code/tests/test_observation_projector_signals.py:94-108, 124-134, 149-162`
**Reason:** Explicitly marked optional per the orchestrator's project constraints for this run.
Not attempted; the new WR-04 test added to this same class intentionally followed the existing
hand-rolled pattern for local consistency rather than introducing a third pattern into a file
this finding already flags for consolidation.

## Verification

- **Where the gates ran:** entirely inside the isolated worktree
  (`/home/tlister/git/fomo_devel/.claude/worktrees/rf-34-1710532-1789163316`, branch
  `gsd-reviewfix/34-1710532`), not the main checkout. The worktree carries no `node_modules`
  concern (pure-Python project); Django/pytest dependencies came from the shared venv
  (`/home/tlister/venv/devel_fomo311_venv`), which is valid for both trees since it is installed
  editable against the *package name*, not a specific working-tree path, and every test/lint
  invocation below `cd`'d into the worktree first. Numbers below are reproducible by checking out
  `gsd-reviewfix/34-1710532` (or, after cleanup, the fast-forwarded tip of
  `issue37-telescope-runs-calendar`) and re-running the same commands from the repo root -- they
  are not an artifact of worktree-only state.
- Targeted tests (`solsys_code.tests.test_calendar_utils
  solsys_code.tests.test_observation_projector_signals solsys_code.tests.test_observation_projector
  solsys_code.tests.test_project_observation_calendar`): **151 tests, OK** (run after every
  individual fix's own targeted subset also passed before each commit).
- `pre-commit run ruff --all-files`: **Passed**.
- `pre-commit run ruff-format --all-files`: **Passed** (auto-reformatted the notebook's cell
  source once during the CR-03/WR-06/WR-07/WR-08/WR-09 commit; re-staged and re-committed with
  the reformatted content).
- Full-suite gate (`test_command` from `.planning/config.json`, run verbatim): first invocation
  (all `solsys_code` test modules except `test_views.py`) ran **1127 tests in 135.5s, OK**, exit
  code 0; second invocation
  (`solsys_code.tests.test_views.TestSplitNumberUnitRegex solsys_code.tests.test_views.TestJPLSBDBQuery`)
  ran **40 tests, OK**, exit code 0.
- Notebook re-execution: `jupyter nbconvert --to notebook --execute --inplace` run from
  `docs/notebooks/pre_executed/` inside the worktree with
  `FOMO_DATABASE_PATH=<worktree>/tmp/fomo_g34_2_copy.sqlite3` (the scratch copy from plan 34-06,
  copied into the worktree since untracked files do not follow `git worktree add`). Exit code 0.
  The committed baseline JSON (`project_observation_calendar_demo.sched06-baseline.json`) and the
  developer database (`src/fomo_db.sqlite3`) were both verified byte-for-byte/mtime-unchanged in
  the main checkout before and after (see CR-03 above) -- neither was ever touched by this fix or
  its verification.
- `LCO_API_KEY` / token-bearing URLs: none printed or pasted at any point.

---

_Fixed: 2026-09-11T22:20:47Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
