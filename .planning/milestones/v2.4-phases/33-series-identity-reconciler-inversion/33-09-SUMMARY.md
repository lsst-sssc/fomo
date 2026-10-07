---
phase: 33-series-identity-reconciler-inversion
plan: 09
subsystem: campaign-coordination
tags: [django, sqlite, jupyter, notebooks, data-cleanup, environment-variable]

# Dependency graph
requires:
  - phase: 33 (plans 01-08)
    provides: the reconciler inversion, CalendarEventMeta link fields, and the two
      pre-executed demo notebooks whose write-through-to-dev-db defect this plan fixes
provides:
  - FOMO_DATABASE_PATH environment-variable branch in src/fomo/settings.py
  - a scratch-copy setup/teardown pattern in both pre-executed demo notebooks, reusable by
    any future notebook regeneration
  - a developer database with the two demo campaigns' residue removed, including detached
    event pk 335
affects: [33-10, 33-11, future notebook regenerations under docs/notebooks/pre_executed/]

# Actuals (#2632)
actuals:
  tokens: 92000
  tasks: 3
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Notebook scratch-database pattern: copy src/fomo_db.sqlite3 to a tempfile.mkdtemp()
      directory, set FOMO_DATABASE_PATH to the copy BEFORE django.setup(), assert the
      resolved DATABASES['default']['NAME'] equals the copy path, and rmtree the scratch
      directory in a final teardown cell -- makes any docs/notebooks/pre_executed/ notebook
      re-runnable without corrupting the developer database."

key-files:
  created: []
  modified:
    - src/fomo/settings.py
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
    - src/fomo_db.sqlite3 (untracked/gitignored -- local-only cleanup, no committed diff)

key-decisions:
  - "FOMO_DATABASE_PATH resolved via `os.getenv('FOMO_DATABASE_PATH') or os.path.join(BASE_DIR, ...)` (using `or`, not a two-arg os.getenv) so an empty-string value also falls back to the default path."
  - "Event pk 335 (RUN:59:2026-09-02, detached from run 59) was DELETED, not re-attached, per the plan's explicit instruction -- it is a reconciler-minted entry for a demo campaign that no longer exists."
  - "Both notebooks were edited and re-executed twice each in this plan's execution to preserve one commit per task: once with only Task 1's scratch-copy mechanism (committed), then again with Task 2's contact-field removal layered on top (committed separately) -- avoiding a single commit that silently bundled both tasks' changes to campaign_lifecycle_demo.ipynb."

requirements-completed: []  # ANNOT-01/ANNOT-02 not yet markable complete -- see Requirements Status below

coverage:
  - id: D1
    description: "Both demo notebooks copy the developer database to a scratch file, point FOMO_DATABASE_PATH at the copy before django.setup(), print+assert the resolved path, and tear the copy down at the end"
    requirement: "ANNOT-01"
    verification:
      - kind: other
        ref: "manual re-execution: jupyter nbconvert --to notebook --execute --inplace on both notebooks, md5sum src/fomo_db.sqlite3 before/after identical (1dc2914614c036f061259dcbb4137c82)"
        status: pass
      - kind: other
        ref: "FOMO_DATABASE_PATH=/tmp/fomo-settings-probe.sqlite3 python manage.py shell -c \"...\" -- override path printed; unset -- default src/fomo_db.sqlite3 path printed"
        status: pass
    human_judgment: false
  - id: D2
    description: "campaign_lifecycle_demo.ipynb's public-campaign-table cell prints only pk, telescope_instrument and approval_status -- contact_person/contact_email removed from that cell's code and committed output"
    requirement: "ANNOT-02"
    verification:
      - kind: other
        ref: "grep -c 'contact_person\\|contact_email' on cell 36's own source/output: 0; whole-file grep: 11 (all in unrelated submission-form cells 9/10/22 -- see Deviations)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Developer database holds no row from either demo campaign, including event pk 335; a fixture receipt (surviving campaign-attributed row count and date range) is recorded for plan 33-11"
    verification:
      - kind: other
        ref: "python manage.py shell -c residue probe -- DEMO_RESIDUE=0 TOTAL_EVENTS=238 ATTRIBUTED_SURVIVING=65"
        status: pass
    human_judgment: false

# Metrics
duration: 38min
completed: 2026-09-10
status: complete
---

# Phase 33 Plan 09: Notebook Scratch-Database Isolation & Demo Residue Cleanup Summary

**Both demo notebooks now execute against a `tempfile.mkdtemp()` scratch copy of the developer database instead of writing through to it, the lifecycle notebook's public table no longer prints contact fields, and the dev database's demo-campaign residue (including detached event pk 335) is gone.**

## Performance

- **Duration:** ~38 min
- **Started:** 2026-09-10T03:40:00Z (approx.)
- **Completed:** 2026-09-10T04:18:00Z
- **Tasks:** 3
- **Files modified:** 3 tracked (`src/fomo/settings.py`, 2 notebooks) + 1 untracked/gitignored (`src/fomo_db.sqlite3`)

## Accomplishments

- `src/fomo/settings.py` resolves `DATABASES['default']['NAME']` from `FOMO_DATABASE_PATH` when set and non-empty, falling back to today's `src/fomo_db.sqlite3` path otherwise -- verified both branches directly via `manage.py shell`.
- Both `docs/notebooks/pre_executed/` demo notebooks copy the developer database to a scratch file, point `FOMO_DATABASE_PATH` at the copy before `django.setup()`, assert the resolved database path equals the scratch copy, and tear the scratch directory down in a final cell -- proven by two full end-to-end executions of each notebook with `src/fomo_db.sqlite3`'s md5 unchanged across both.
- `campaign_lifecycle_demo.ipynb`'s public-campaign-table cell prints only `pk`, `telescope_instrument` and `approval_status` -- `contact_person`/`contact_email` are gone from that cell's code and its committed output.
- The developer database's demo-campaign residue (9 `CampaignRun` rows, 19 `CalendarEvent` rows including detached pk 335, 2 `TargetList` rows, 1 `ObservationRecord`, 1 `ObservationGroup`, 1 `Target`, 5 `Observatory` rows, 1 `User`) is removed, backed up first outside the repo. The developer database's remaining campaign-attributed row count (65, down from 82 -- a drop of exactly 17, matching 33-UAT.md's count of demo-campaign September attributions) is recorded below as the fixture receipt for plan 33-11.

## Task Commits

1. **Task 1: Notebooks execute against a scratch copy of the developer database** - `0219ccd` (fix)
2. **Task 2: Drop the two contact fields from the lifecycle notebook's public-table output** - `1f1b89d` (fix)
3. **Task 3: Clean the demo residue out of the developer database** - no commit (target file `src/fomo_db.sqlite3` is gitignored/untracked; effect recorded here instead)

_Note: to keep each task's commit strictly scoped to that task's own diff, `campaign_lifecycle_demo.ipynb` was edited and re-executed twice: once with only Task 1's scratch-copy mechanism (committed in `0219ccd`), then again with Task 2's contact-field removal layered on top (committed separately in `1f1b89d`)._

## Files Created/Modified

- `src/fomo/settings.py` - added the `FOMO_DATABASE_PATH` environment-variable branch to `DATABASES['default']['NAME']`
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` - scratch-copy setup/teardown cells, updated markdown prose, re-executed
- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` - scratch-copy setup/teardown cells, updated markdown prose, contact-field removal in the public-table cell, re-executed twice (once per task)
- `src/fomo_db.sqlite3` (untracked, local-only) - demo-campaign residue removed; see "Task 3 Cleanup Record" below

## Task 3 Cleanup Record

**Backup path (outside the repo, undo path):** `/tmp/fomo-db-backups/33-09-backup-oWQ6P9/fomo_db.sqlite3.bak`

**Before/after counts:**

| Category | Before | After | Notes |
|---|---|---|---|
| `CampaignRun` (either demo campaign) | 9 | 0 | Deleted in Step 1; cascade removed their own `RUN:`-namespaced events |
| `CalendarEvent` linked via `target_list` to either demo campaign (incl. pk 335) | 19 | 0 | Most already removed by Step 1's per-run cascade; the residual (including pk 335) removed explicitly in Step 2 |
| `CalendarEvent` with the lifecycle legacy url prefix | 0 | 0 | None existed at cleanup time (already resolved/consumed by prior notebook runs) |
| `TargetList` (either demo campaign) | 2 | 0 | Deleted in Step 4 |
| `ObservationRecord` (`campaign-lifecycle-demo-obs-1`) | 1 | 0 | Deleted in Step 5 |
| `ObservationGroup` (`Campaign Lifecycle Demo Group`) | 1 | 0 | Deleted in Step 5 |
| `Target` (`Campaign Lifecycle Demo Series-Identity Target`) | 1 | 0 | Deleted in Step 5 |
| Total `CalendarEvent` | 257 | 238 | Non-zero after cleanup -- proves the sweep was scoped, not a blanket empty |
| `CalendarEventMeta.objects.filter(run__isnull=False)` | 82 | 65 | Drop of 17, matching 33-UAT.md's count of demo-campaign September attributions |

**Observatory rows (Step 6):** all five (X29, X30, Y21, Y22, Y23) were deleted -- every `CampaignRun` that had referenced any of them belonged to one of the two demo campaigns (confirmed by query before deletion), so none was left referenced. No obscode was skipped.

**Staff user (Step 7):** `campaign_lifecycle_demo_staff` was deleted -- no surviving `CalendarEventMeta` row named it as `confirmed_by` (confirmed 0 both before and after the cleanup).

**Fixture receipt for plan 33-11 (`ATTRIBUTED_SURVIVING`):**

- **Surviving campaign-attributed `CalendarEventMeta` rows:** 65
- **Date range:** earliest attributed event `start_time` = `2025-07-03 22:00:47+00:00`; latest = `2026-07-21 07:27:08+00:00`
- **Plain statement for 33-11's operator:** a campaign-attributed calendar entry **does** still exist (65 of them, spanning July 2025 through July 2026) -- 33-11's browser check does **not** need to seed one; any of the 65 surviving `CalendarEventMeta` rows with `run` set is a valid click target.

## Decisions Made

- `FOMO_DATABASE_PATH` resolved with `os.getenv('FOMO_DATABASE_PATH') or os.path.join(BASE_DIR, 'fomo_db.sqlite3')` -- the `or` form (not a two-arg `os.getenv`) so an empty-string value also falls back to the default, per the plan's explicit instruction.
- Event pk 335 was **deleted**, not re-attached to run 59, matching the plan's explicit decision: it is a reconciler-minted entry for a demo campaign that is itself being removed.
- Split `campaign_lifecycle_demo.ipynb`'s combined Task 1 + Task 2 edits into two separate edit/execute/verify/commit cycles (rather than committing both tasks' changes to that file in one commit) to preserve one commit per task.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Plan verify script over-scoped] Task 2's whole-file grep acceptance criteria cannot reach 0 without breaking the public submission form**

- **Found during:** Task 2 verification
- **Issue:** Task 2's acceptance criteria and `<verify>` block specify `grep -c 'contact_person' docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` (and same for `contact_email`) must output `0` -- a whole-file check. However, the task's own `<action>`, `<read_first>` (which names only cell 36, the public-table cell, and cell 35's markdown above it) and the plan's `must_haves.truths` ("campaign_lifecycle_demo.ipynb's **public-campaign-table cell** prints only...") all scope the fix to that one cell. The notebook's cells 9/10/22 legitimately pass `contact_person=CONTACT_PERSON`/`contact_email=CONTACT_EMAIL` as POST data to the real `CampaignRunSubmissionForm` -- these are the actual field names the form requires to submit contact info, not committed output, and removing them would break the notebook's own submission flow (out of `files_modified` scope to fix the form itself, and unrelated to the 33-05 P1/33-08 P5 prohibition this task closes, which is specifically about **committed output**, not source code using real field names).
- **Fix:** Removed `contact_person`/`contact_email` from cell 36's print loop and its committed output only (the change described in the task's `<action>`), leaving cells 9/10/22's legitimate form-submission code untouched. Verified narrowly: `grep -c 'contact_person\|contact_email'` scoped to cell 36's own source+output is `0`; the notebook-wide grep is `11` (all 11 matches are in cells 9/10/22's submission-form code, confirmed by line-by-line inspection).
- **Files modified:** `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` (cell 35 markdown, cell 36 code)
- **Verification:** Manual grep against cell 36 specifically (0 matches); manual grep against the whole file (11 matches, all attributed to cells 9/10/22 and confirmed legitimate); the cell-output-completeness check and the five-run visibility assertion both still pass.
- **Committed in:** `1f1b89d`

---

**Total deviations:** 1 auto-fixed (plan verify script over-scoped relative to its own stated task action and must-haves).
**Impact on plan:** No scope creep -- the actual must-have (public-table cell free of contact fields, in both code and output) is fully satisfied and verified. The over-broad whole-file grep in the plan's `<verify>` block would, if taken literally, have required deleting real form-submission code the task never asked to touch and that lies outside this plan's `files_modified`.

## Issues Encountered

None beyond the deviation documented above.

## User Setup Required

None - no external service configuration required.

## Requirements Status

This plan carries `requirements: [ANNOT-01, ANNOT-02]`. Both IDs are also declared by sibling plans still in flight in this phase's gap-closure wave (per `requirements.ready-ids`, 0/2 are currently markable complete because a sibling plan declaring the same ID has not yet produced its own SUMMARY). Per the shared-ID gate (issue #2388), neither ID is marked complete here -- they will be marked once the last plan declaring each ID finishes.

## Next Phase Readiness

- Plan 33-10 (wave 3, `depends_on` this plan via sequencing note) can now re-execute both notebooks without re-seeding the residue this plan just cleaned, since the scratch-copy mechanism is in place.
- Plan 33-11 (wave 2, `depends_on: [33-09]`) can read this plan's fixture receipt directly: 65 surviving campaign-attributed `CalendarEventMeta` rows exist (July 2025 - July 2026 date range), so its Task 2 browser check has a real click target and does not need to seed one.
- No blockers.

---
*Phase: 33-series-identity-reconciler-inversion*
*Completed: 2026-09-10*

## Self-Check: PASSED

- FOUND: src/fomo/settings.py
- FOUND: docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
- FOUND: docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
- FOUND: src/fomo_db.sqlite3 (untracked)
- FOUND: backup at /tmp/fomo-db-backups/33-09-backup-oWQ6P9/fomo_db.sqlite3.bak
- FOUND: commit 0219ccd (Task 1)
- FOUND: commit 1f1b89d (Task 2)
