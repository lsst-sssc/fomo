---
phase: 29-the-reconciler
plan: 05
subsystem: docs
tags: [sphinx, runbook, jupyter, reconciler, campaign-run, calendar-events]

# Dependency graph
requires:
  - phase: 29-the-reconciler
    plan: 03
    provides: solsys_code/management/commands/reconcile_campaign_runs.py -- the command
      this plan documents
  - phase: 29-the-reconciler
    plan: 04
    provides: the four staff actions (approve/resolve_site/mark_cancelled/mark_weather_failure)
      calling campaign_reconciler.reconcile_run() directly, and the deletion of
      backfill_range_calendar_events -- both of which this plan's runbook rewrite documents
provides:
  - "docs/runbooks/telescope_runs_calendar.rst rewritten: backfill_range_calendar_events
    fully retired from the page (section, cheat-sheet row, skip-and-log bullet,
    troubleshooting mention, cross-reference), replaced with a new 'How do I get every
    campaign run onto the calendar?' section documenting reconcile_campaign_runs"
  - "The 'Campaign run block' section rewritten: reconciler-owned events get the ownership
    link automatically; the manual admin path is scoped to events the reconciler never
    touches"
  - "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb -- new pre-executed
    demo notebook, wired into docs/notebooks.rst's toctree and CLAUDE.md's notebook
    pairing map"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Runbook prose describing a retired command name must paraphrase it (e.g. 'the
       retired per-gap range-window backfill command') rather than spell the literal
       identifier, to avoid self-tripping the phase's own zero-occurrence grep check --
       the same self-tripping pattern plans 29-01/29-02 hit for different literal strings"

key-files:
  created:
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  modified:
    - docs/runbooks/telescope_runs_calendar.rst
    - docs/notebooks.rst
    - CLAUDE.md

key-decisions:
  - "The demo notebook seeds its own dedicated Observatory pair (obscodes X29/X30, not
     reusing any real MPC code) and its own campaign TargetList, so it never collides
     with fixture data any sibling notebook might have seeded in the same dev DB"
  - "Queue run explicitly sets source=CampaignRun.Source.LCO_QUEUE and is given a resolved
     site (not left site-less) -- the reconciler's dispatch order checks telescope_class,
     then satellite site, then QUEUE_SOURCES, so a resolved-site queue run still correctly
     exercises the queue/container branch, matching what a real LCO queue CampaignRun looks
     like once source is corrected"

patterns-established: []

requirements-completed: [RECON-01, RECON-06, RECON-08, RECON-09]

# Metrics
duration: ~55min
completed: 2026-08-05
---

# Phase 29 Plan 05: The Runbook and Demo Notebook Summary

**Rewrote `docs/runbooks/telescope_runs_calendar.rst` to retire every trace of `backfill_range_calendar_events` and document `reconcile_campaign_runs` in its place, corrected the "Campaign run block" section's now-superseded manual-only claim, and shipped a new pre-executed `reconcile_campaign_runs_demo.ipynb` proving the dry-run/real-sweep/idempotency contract end-to-end.**

## Performance

- **Duration:** ~55 min
- **Tasks:** 2 completed
- **Files modified:** 3 (2 modified, 1 new; 1 of the modified files, CLAUDE.md, gets a one-line map addition)

## Accomplishments

- Deleted the "How do I backfill calendar events for older approved range-window runs?"
  section, its `backfill_range_calendar_events` cheat-sheet row, its per-line/per-record
  skip-and-log bullet, its Observatory-missing-timezone troubleshooting mention, and the
  stale `backfill_lco_observation_records` cross-reference -- all four locations now name
  `reconcile_campaign_runs` instead.
- Added a new "How do I get every campaign run onto the calendar?" section covering what
  the command does, the dry-run-first workflow with a matching console example pair, the
  real summary-line counter names for both modes, all four `_skip_reason()` strings quoted
  verbatim with operator guidance for each, the separate `reconcile failed (...)` timezone
  case, what an operator sees on the calendar afterwards, and that the four staff actions
  now reconcile automatically without a command run.
- Rewrote the "Why doesn't the calendar pop-up show a 'Campaign run' block?" section: the
  old claim that "nothing in FOMO fills that link in automatically" is corrected to state
  that reconciler-owned events get the link set automatically, with the manual admin path
  now scoped explicitly to events the reconciler never touches (un-attributed sync-command
  events, hand-created entries). Both existing notes (frozen calendar-event field, clearing
  the owning-run value) are preserved verbatim.
- Created and executed `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`:
  seeds a ground + satellite `Observatory` pair and a campaign `TargetList`, seeds three
  `CampaignRun` rows exercising all three reconciler dispatch branches (classical,
  queue, class-wide) with a markdown cell explaining why `source` must be set explicitly
  for real (pre-v2.2 `legacy`-carrying) data, demonstrates `--dry-run` leaving
  `CalendarEvent.objects.count()` unchanged, a real sweep with a printed per-run loop over
  `owned_events()` showing the date-bearing `RUN:{pk}:{date}` family for the classical run
  versus the bare `RUN:{pk}` family for the queue and class-wide runs, and a second real
  sweep reporting `created: 0, updated: 0`.
- Wired the new notebook into `docs/notebooks.rst`'s Sphinx toctree and added the
  `campaign_reconciler.py`/`reconcile_campaign_runs.py` pair to CLAUDE.md's notebook
  pairing map.

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite the operator runbook for the reconciler and finish retiring the backfill command** - `4bafffa` (docs)
2. **Task 2: Create and execute the paired reconcile_campaign_runs demo notebook** - `f9dcbbd` (docs)

## Files Created/Modified

- `docs/runbooks/telescope_runs_calendar.rst` - backfill command fully retired from the
  page; new reconciler section; corrected "Campaign run block" section; new cheat-sheet row
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` - new pre-executed
  demo notebook (executed, committed with output)
- `docs/notebooks.rst` - new toctree entry for the reconciler demo notebook
- `CLAUDE.md` - notebook pairing map gains the `campaign_reconciler.py`/
  `reconcile_campaign_runs.py` -> `reconcile_campaign_runs_demo.ipynb` entry

## Decisions Made

- The demo notebook uses its own dedicated `Observatory` obscodes (`X29` ground, `X30`
  satellite) rather than reusing any real MPC code from sibling notebooks, so it stays
  collision-free and independently re-runnable against any dev DB.
- The queue-run fixture is given a resolved `site` (not left site-less) alongside its
  explicit `source=CampaignRun.Source.LCO_QUEUE` -- the reconciler's dispatch order checks
  `telescope_class`, then a satellite `site`, then `source in QUEUE_SOURCES` before falling
  through to the classical per-night branch, so a resolved-site queue run still correctly
  exercises the queue/container branch, matching a real corrected LCO queue `CampaignRun`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Missing `src/fomo/_version.py` build artifact blocked all Django
management-command/test invocations in this worktree**
- **Found during:** Environment setup before Task 2 (first `python manage.py migrate --check`)
- **Issue:** Same environment-only gap plans 29-01 through 29-04 hit: `src/fomo/__init__.py`
  imports the gitignored, `setuptools_scm`-generated `._version` module, absent from this
  fresh worktree checkout.
- **Fix:** Copied the file from the main repo's working tree
  (`/home/tlister/git/fomo_devel/src/fomo/_version.py`) into this worktree -- a harmless,
  gitignored, environment-only file, never staged or committed.
- **Files modified:** none tracked by git (gitignored build artifact)
- **Verification:** `python manage.py migrate` and the notebook's Django-setup cell now run
  in this worktree.
- **Committed in:** n/a (not a git-tracked file)

**2. [Rule 3 - Blocking] Dev-DB schema not migrated in this fresh worktree**
- **Found during:** Same environment-setup step, immediately after the `_version.py` fix
- **Issue:** `src/fomo_db.sqlite3` was a fresh, unmigrated gitignored dev-DB file (0
  `CampaignRun` rows, no schema) -- confirmed via `showmigrations`, every app's migrations
  were unapplied.
- **Fix:** Ran `python manage.py migrate` once in this worktree -- an environment-only
  action against a gitignored local file, not a code or migration change.
- **Files modified:** none tracked by git (`src/fomo_db.sqlite3` is gitignored)
- **Verification:** the demo notebook's seed/reconcile cells now execute against a real
  migrated (initially empty) schema.
- **Committed in:** n/a (not a git-tracked file)

**3. [Rule 3 - Blocking] The demo notebook's own opening markdown cell self-tripped the
plan's zero-occurrence `backfill_range_calendar_events` grep**
- **Found during:** Task 2 verification (running `grep -rn 'backfill_range_calendar_events'
  docs/` per the plan's phase-wide `<verification>` block, after the notebook was already
  executed)
- **Issue:** The notebook's title cell explained what it replaces by naming the retired
  command literally ("replacing the retired per-gap `backfill_range_calendar_events`
  command"), which is accurate but tripped the same-named grep check the runbook edits were
  designed to satisfy -- the identical self-tripping pattern plans 29-01/29-02 documented
  for different literal strings in `campaign_reconciler.py`'s own docstrings.
- **Fix:** Reworded the sentence to "replacing the retired per-gap range-window backfill
  command", preserving the same meaning without the literal identifier. Edited the
  already-executed notebook's markdown-cell JSON directly (no code cells touched, so
  re-execution was not required) and re-validated with `nbformat.validate()`.
- **Files modified:** `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`
- **Verification:** `grep -rn 'backfill_range_calendar_events' docs/` returns nothing;
  `nbformat.validate()` passes; all 6 code cells still carry their original committed
  output (verified by count before and after the edit).
- **Committed in:** `f9dcbbd` (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (2 Rule 3 environment-only fixes, 1 Rule 3 fix to a
docstring/prose literal that self-tripped the plan's own grep check)
**Impact on plan:** All three were necessary to complete verification as specified; none
changed scope or added functionality beyond what the plan already required.

## Issues Encountered

None beyond the auto-fixed items above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- RECON-09 is now complete in both code (plan 29-04) and documentation (this plan) -- no
  runbook prose anywhere references `backfill_range_calendar_events` as a still-available
  command.
- The paired demo notebook and its cross-references (toctree, CLAUDE.md pairing map) are
  in place; a full end-to-end human read-through of the runbook is 29-06's stated
  responsibility per the threat register (T-29-16 disposition), not this plan's.
- No blockers.

## Self-Check: PASSED

- Files verified present: `docs/runbooks/telescope_runs_calendar.rst`,
  `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`, `docs/notebooks.rst`,
  `CLAUDE.md`, `.planning/phases/29-the-reconciler/29-05-SUMMARY.md`.
- Commits verified present in `git log --oneline --all`: `4bafffa`, `f9dcbbd`.
- `grep -rn 'backfill_range_calendar_events' docs/` returns nothing.
- `grep -c 'reconcile_campaign_runs' docs/runbooks/telescope_runs_calendar.rst` is 9 (>= 4
  required).
- `grep -c 'reconcile_campaign_runs_demo' docs/notebooks.rst` is 1;
  `grep -c 'reconcile_campaign_runs_demo.ipynb' CLAUDE.md` is 1.
- `python -m sphinx -b html docs docs/_build/html -q` completes with no errors or warnings
  naming `telescope_runs_calendar.rst`, the new notebook, or `notebooks.rst` (pre-existing,
  unrelated warnings/errors in `docs/autoapi/`, `sync_lco_observation_calendar_demo.ipynb`,
  and `ESO_How_to_download_data.ipynb` are untouched by this plan).
- `ruff check .` and `ruff format --check .`: clean except the same 3 pre-existing,
  untouched-by-this-plan issues 29-04-SUMMARY.md already documented
  (`sync_gemini_observation_calendar_demo.ipynb` D103,
  `.planning/quick/260619-f7u-.../verify_nb.py`/`verify_project.py`, `src/fomo/settings.py`
  formatting) -- confirmed via `git status --short` that none of these files were touched
  by this plan.
- Notebook code-cell output count verified: 6/6 code cells carry committed output both
  immediately after execution and again after the post-execution markdown wording fix.

---
*Phase: 29-the-reconciler*
*Completed: 2026-08-05*
