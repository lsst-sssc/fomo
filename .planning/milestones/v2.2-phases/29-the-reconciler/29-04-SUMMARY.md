---
phase: 29-the-reconciler
plan: 04
subsystem: calendar-projection
tags: [django, calendar-events, campaign-run, reconciler, staff-actions, testing]

# Dependency graph
requires:
  - phase: 29-the-reconciler
    plan: 02
    provides: campaign_reconciler.py's complete reconcile_run() (container branch, D-02
      adopt-and-rekey classical branch, RECON-05 ownership guard, RECON-06 dry-run support)
provides:
  - "campaign_views.py's four staff actions (approve, resolve_site, mark_cancelled,
    mark_weather_failure) call campaign_reconciler.reconcile_run() as their sole
    calendar-projection mechanism -- no projection logic of their own"
  - "_project_calendar_event()/_calendar_event_title()/_RUN_STATUS_CALENDAR_PREFIX deleted
    outright from campaign_views.py"
  - "backfill_range_calendar_events management command and its test module deleted outright"
  - "test_campaign_approval.py rewritten onto RUN: keys, campaign_reconciler patch targets,
    and the reconciler's event_title()/event_description()/RUN_STATUS_CALENDAR_PREFIX as
    the single source of truth for title/description assertions"
affects: [29-05-runbook-and-demo-notebook]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Staff actions are thin wrappers over reconcile_run(): each of the four call sites
       does its own business-logic guard/staleness check, then delegates all calendar state
       to the shared reconciler, never touching CalendarEvent directly"
    - "Test assertions against derived reconciler helpers (event_title()/event_description()/
       run_night_url()/run_container_url()/owned_events()) instead of re-derived literal
       key/title strings, so the tests cannot silently drift from the single source of truth"

key-files:
  created: []
  modified:
    - solsys_code/campaign_views.py
    - solsys_code/tests/test_admin.py
    - solsys_code/tests/test_campaign_approval.py
  deleted:
    - solsys_code/management/commands/backfill_range_calendar_events.py
    - solsys_code/tests/test_backfill_range_calendar_events.py

key-decisions:
  - "event_title(run) already applies RUN_STATUS_CALENDAR_PREFIX internally (looked up from
     run.run_status) -- test assertions call it directly rather than re-concatenating the
     prefix a second time, which the first draft did and which double-prefixed the title"
  - "The calendar-sync-failure test in TestRunStatusChange now patches
     campaign_reconciler.update_calendar_event_key_and_fields, not insert_or_create_calendar_event
     -- the target CalendarEvent already exists (created by the run's earlier approve), so
     reconcile_run()'s update path is what actually executes for that scenario"
  - "docs/runbooks/telescope_runs_calendar.rst and calendar_utils.py's docstring mention of
     the retired backfill command are left untouched -- both are explicitly out of this
     plan's files_modified scope and are plan 29-05's stated responsibility (confirmed via
     29-05-PLAN.md's must_haves and Task 1 scope)"

patterns-established:
  - "Task-boundary test-file commits: when a single test file's rewrite spans two plan
     tasks, the executor can still produce two atomic per-task commits by reconstructing an
     intermediate file state (task N's target classes converted, task N+1's target classes
     temporarily reverted to their pre-plan form) rather than bundling both tasks into one
     commit -- used here to keep Task 2 and Task 3 each independently verifiable against the
     commit history."

requirements-completed: [RECON-08, RECON-09]

# Metrics
duration: ~50min
completed: 2026-08-05
---

# Phase 29 Plan 04: The Staff Action Rewire Summary

**`campaign_views.py`'s four staff actions (approve/resolve_site/mark_cancelled/mark_weather_failure) now call `campaign_reconciler.reconcile_run()` exclusively; the retired `_project_calendar_event()`/`_calendar_event_title()` projection code and the `backfill_range_calendar_events` command are deleted, and the approval-queue test suite (124 tests) is rewritten onto `RUN:` keys and the reconciler's own title/prefix builders.**

## Performance

- **Duration:** ~50 min
- **Tasks:** 3 completed
- **Files modified:** 3 modified, 2 deleted

## Accomplishments

- Rewired `approve()`, `_resolve_site()`, and `_set_run_status()` (serving `mark_cancelled`/
  `mark_weather_failure`) in `campaign_views.py` to call `campaign_reconciler.reconcile_run()`
  instead of the retired `_project_calendar_event()` helper -- `_resolve_site()` now derives
  its two success messages from `ReconcileResult.skipped_reason is None`, and
  `_set_run_status()` no longer queries `CalendarEvent` or builds titles/descriptions itself;
  the reconciler's stage-0 guard is the sole authority on whether an event exists.
- Deleted `_calendar_event_title()`, `_project_calendar_event()`, and
  `_RUN_STATUS_CALENDAR_PREFIX` from `campaign_views.py` (110 lines of projection code), and
  pruned the imports ruff flagged as unused (`sun_event`, `insert_or_create_calendar_event`,
  `time as dt_time`, `timedelta`, `django.db.models.Q`).
- Deleted `solsys_code/management/commands/backfill_range_calendar_events.py` and
  `solsys_code/tests/test_backfill_range_calendar_events.py` outright via `git rm` (RECON-09)
  -- the reconciler now covers every gap the backfill command patched.
- Rewrote all seven affected test classes in `test_campaign_approval.py`
  (`TestApproval`, `TestCalendarProjection`, `TestRunStatusChange`, `TestSitesNeedingReview`,
  `TestPlaceholderSiteReplacement`, `TestCalendarNoChurn`, `TestGeminiFtScenario`) onto
  `run_container_url()`/`run_night_url()`/`owned_events()` for key assertions and
  `event_title()`/`event_description()`/`RUN_STATUS_CALENDAR_PREFIX` for title/description
  assertions, moved every `campaign_views.sun_event`/`campaign_views.
  insert_or_create_calendar_event` patch target to `campaign_reconciler`, and encoded the two
  new D-01 behavior changes: a single-night classical run's key is date-bearing with no bare
  sibling, and an APPROVED site-resolved run with no prior event now gets one created (not
  just updated) when marked cancelled, while a TBD-window run marked cancelled still ends
  with zero events.
- Full `solsys_code` suite (808 tests, excluding `test_views`/`test_ephem_utils` per project
  memory) passes green; the 124-test `test_campaign_approval` module and the 73-test
  `test_admin` module were also independently verified.

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewire the four staff actions onto reconcile_run() and delete the retired code paths** - `6e9da58` (feat)
2. **Task 2: Rewrite the approval-queue projection tests -- TestApproval, TestCalendarProjection, TestCalendarNoChurn, TestGeminiFtScenario** - `61acb32` (test)
3. **Task 3: Rewrite the remaining approval-queue classes and restore a fully green suite** - `cd84437` (test)

## Files Created/Modified

- `solsys_code/campaign_views.py` - three staff-action call sites now call `reconcile_run()`; `_calendar_event_title()`/`_project_calendar_event()`/`_RUN_STATUS_CALENDAR_PREFIX` deleted; unused imports pruned
- `solsys_code/tests/test_admin.py` - one comment-only reference to the retired projection helper reworded to name `campaign_reconciler.reconcile_run()`'s `'TBD window'` skip reason
- `solsys_code/tests/test_campaign_approval.py` - all `CAMPAIGN:` key literals and `campaign_views`-targeted patches replaced with `RUN:`-scheme reconciler helpers and `campaign_reconciler` patch targets; two new tests added for the D-01 create-on-cancel behavior change
- `solsys_code/management/commands/backfill_range_calendar_events.py` - deleted
- `solsys_code/tests/test_backfill_range_calendar_events.py` - deleted

## Decisions Made

- `event_title(run)` already folds in `RUN_STATUS_CALENDAR_PREFIX` for the run's current
  `run_status`, so test assertions call it directly (`self.assertEqual(event.title,
  event_title(run))`) rather than re-concatenating the prefix, which double-prefixed the
  title in an initial draft (`[WEATHERED] [WEATHERED] ...`) and was caught by the first test
  run against the rewritten classes.
- The `test_mark_cancelled_survives_calendar_sync_failure` patch target moved to
  `campaign_reconciler.update_calendar_event_key_and_fields` (not
  `insert_or_create_calendar_event`), since the target event already exists from the run's
  earlier approve -- `reconcile_run()`'s **update** path, not its create path, is what
  actually executes when `mark_cancelled` fires against an already-projected run.
- Left `docs/runbooks/telescope_runs_calendar.rst` and a docstring mention in
  `calendar_utils.py` untouched: both still reference the retired backfill command, but
  neither is in this plan's `files_modified` list, and `29-05-PLAN.md`'s own must-haves and
  Task 1 scope explicitly own "no runbook prose left behind assumes
  backfill_range_calendar_events still exists" -- verified by reading `29-05-PLAN.md` before
  deciding not to touch these files.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Missing `src/fomo/_version.py` build artifact blocked all Django test runs in this worktree**
- **Found during:** First `python manage.py test` invocation in this worktree
- **Issue:** Same environment-only gap plans 29-01/29-02 hit: `src/fomo/__init__.py` imports the gitignored, `setuptools_scm`-generated `._version` module, absent from this fresh worktree checkout.
- **Fix:** Copied the file from the main repo's working tree into this worktree -- a harmless, gitignored, environment-only file, never staged or committed.
- **Files modified:** none tracked by git (gitignored build artifact)
- **Verification:** `python manage.py test` now runs in this worktree.
- **Committed in:** n/a (not a git-tracked file)

**2. [Rule 1 - Bug] Fixed a double-prefixed calendar-event title assertion introduced while encoding Task 3's "assert against event_title()" instruction**
- **Found during:** Task 3 verification (first run of `TestRunStatusChange`/`TestGeminiFtScenario` against the rewritten assertions)
- **Issue:** An initial draft asserted `event.title == f'{RUN_STATUS_CALENDAR_PREFIX[...]} {event_title(run)}'`, but `event_title(run)` already includes the prefix internally (it looks up `RUN_STATUS_CALENDAR_PREFIX.get(run.run_status)`), producing a spurious `'[WEATHERED] [WEATHERED] ...'` expected value that failed against the real single-prefixed title.
- **Fix:** Asserted `event.title.startswith(RUN_STATUS_CALENDAR_PREFIX[...])` (proves the prefix literal) alongside `event.title == event_title(run)` (proves the full derived string), without re-concatenating the prefix.
- **Files modified:** `solsys_code/tests/test_campaign_approval.py`
- **Verification:** `python manage.py test solsys_code.tests.test_campaign_approval` passes (124/124).
- **Committed in:** `cd84437` (Task 3 commit)

**3. [Rule 3 - Blocking] `python manage.py test solsys_code` (unfiltered) segfaults on `test_views.TestEphemeris`**
- **Found during:** Verifying the plan's own `<verification>` command (`python manage.py test solsys_code` green)
- **Issue:** The native ASSIST integrator segfaults inside `test_views.TestEphemeris` (a pre-existing environment issue unrelated to any recent phase, documented in project memory and CLAUDE.md).
- **Fix:** Ran the full suite as the explicit list of all `solsys_code/tests/*.py` modules except `test_views`/`test_ephem_utils`/`helpers`, per the project's own established exclusion convention (also used in plans 29-01/29-02's regression runs).
- **Files modified:** none (verification-command substitution only)
- **Verification:** 808 tests, OK.
- **Committed in:** n/a (verification-only, no source change)

---

**Total deviations:** 3 auto-fixed (1 Rule 3 environment-only fix, 1 Rule 1 bug fix to a test assertion I introduced, 1 Rule 3 verification-command substitution for a pre-existing unrelated segfault)
**Impact on plan:** All three were necessary to complete verification as specified; none changed scope or added functionality beyond what the plan already required.

## Known Residual (not fixed, out of scope)

The plan's acceptance criteria include `grep -rn '_project_calendar_event\|_calendar_event_title' --include=*.py solsys_code/` returning nothing repo-wide. After this plan, that grep still matches five docstring/comment mentions inside `solsys_code/campaign_reconciler.py` (a file plan 29-01 created and this plan's `files_modified` list does not include) -- they explain the reconciler's behavior relative to the now-deleted helpers (e.g. "byte-identical to `campaign_views._calendar_event_title()`'s cancelled/weathered output"). Editing `campaign_reconciler.py` was out of scope for this plan (per the parallel-execution file-ownership boundary with the sibling 29-03 worktree, and because it is not in `29-04-PLAN.md`'s `files_modified`), so these are left as historical documentation rather than fixed. They are prose-only; no code imports, calls, or patches the deleted symbols anywhere. A future plan touching `campaign_reconciler.py` should reword these five mentions.

## Issues Encountered

Mid-execution, the agent paused waiting on a backgrounded full-suite test run; the background process was orphaned when the turn ended before a notification arrived. Resumed per the coordinator's instruction by re-running the same verification synchronously (foreground, bounded timeout) rather than backgrounding it again. No work was lost -- Task 1 was already committed, and Tasks 2/3's test-file edits were still present on disk uncommitted; both were re-verified and committed cleanly afterward.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All four staff actions reconcile through the shared function; the retired projection path and the backfill command are gone from the codebase (RECON-08/09 code-and-tests complete).
- Plan 29-05 (runbook + demo notebook) can now write accurate operator documentation: `docs/runbooks/telescope_runs_calendar.rst` still describes the retired `backfill_range_calendar_events` command and must be rewritten to document `reconcile_campaign_runs` instead -- this was explicitly scoped to 29-05, not this plan, and confirmed by reading `29-05-PLAN.md` before deciding not to touch it here.
- `solsys_code/calendar_utils.py` line 553 still contains one docstring mention of the retired `backfill_range_calendar_events` command name; harmless (prose only) but a natural cleanup opportunity for whichever future plan next touches that file.
- No blockers.

## Self-Check: PASSED

- Files verified present: `solsys_code/campaign_views.py`, `solsys_code/tests/test_admin.py`, `solsys_code/tests/test_campaign_approval.py`, `.planning/phases/29-the-reconciler/29-04-SUMMARY.md`.
- Files verified absent (deleted): `solsys_code/management/commands/backfill_range_calendar_events.py`, `solsys_code/tests/test_backfill_range_calendar_events.py` (`git ls-files` on both returns nothing).
- Commits verified present in `git log --oneline --all`: `6e9da58`, `61acb32`, `cd84437`.
- Full regression run: `python manage.py test` across all `solsys_code` test modules except `test_views`/`test_ephem_utils`/`helpers` (excluded per project memory -- native ASSIST segfault, unrelated to this plan) -- **808 tests, OK** (~325s).
- `ruff check .` and `ruff format --check .`: clean except three pre-existing, untouched-by-this-plan issues (`docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb` D103, and `.planning/quick/260619-f7u-.../verify_nb.py`/`verify_project.py`/`src/fomo/settings.py` formatting) -- confirmed via `git status --short` that none of these files were touched by this plan.
- `grep -c 'CAMPAIGN:' solsys_code/campaign_views.py solsys_code/tests/test_campaign_approval.py`: both 0.
- `grep -rn "patch('solsys_code.campaign_views._project_calendar_event" solsys_code/`: no matches.
- `grep -n 'from .campaign_reconciler import reconcile_run' solsys_code/campaign_views.py` matches; `grep -c 'reconcile_run(run)' solsys_code/campaign_views.py` is 3.

---
*Phase: 29-the-reconciler*
*Completed: 2026-08-05*
