---
phase: 16-submission-form-approval-queue-calendar-projection-write-pat
plan: 05
subsystem: ui
tags: [django-tables2, campaign-approval-queue, gap-closure]

# Dependency graph
requires:
  - phase: 16-submission-form-approval-queue-calendar-projection-write-pat (Plan 03)
    provides: ApprovalQueueTable and ApprovalQueueView (staff approval queue)
provides:
  - ApprovalQueueTable.Meta with exclude + sequence overrides trimming/reordering columns for triage
  - Column-contract regression test proving CampaignRunTable stays spreadsheet-parity
affects: [16-UAT.md Test 14 gap]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "django-tables2 Meta.exclude + Meta.sequence (with the '...' ellipsis token) used to
      produce a purpose-specific column view on a Table subclass without touching the parent
      Table's Meta.fields."

key-files:
  created: []
  modified:
    - solsys_code/campaign_tables.py
    - solsys_code/tests/test_campaign_approval.py

key-decisions:
  - "Fix scoped entirely to ApprovalQueueTable.Meta (exclude + sequence) — CampaignRunTable is untouched, preserving Phase 15's D-09 spreadsheet-parity read path."
  - "sequence uses the django-tables2 '...' ellipsis token rather than exhaustively enumerating all remaining columns, staying robust if CampaignRunTable's field set changes later."

patterns-established:
  - "Triage-focused Table subclass: exclude structurally-blank columns (no corresponding form field) and front-load the action column via Meta.sequence, rather than adding CSS sticky-positioning."

requirements-completed: [SUBMIT-03]

coverage:
  - id: D1
    description: "ApprovalQueueTable renders `actions` as its first column, so Approve/Reject is reachable without horizontal scrolling."
    requirement: "SUBMIT-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueColumns.test_actions_leads_approval_queue_table"
        status: pass
    human_judgment: false
  - id: D2
    description: "ApprovalQueueTable no longer renders weather/observation_outcome/publication_plans (structurally-blank post-observation columns with no submission-form field)."
    requirement: "SUBMIT-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueColumns.test_approval_queue_table_excludes_post_observation_columns"
        status: pass
    human_judgment: false
  - id: D3
    description: "CampaignRunTable is unchanged — still renders all 16 spreadsheet-parity columns including weather/observation_outcome/publication_plans, and has no actions column (Phase 15 D-09 preserved)."
    requirement: "SUBMIT-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueColumns.test_campaign_run_table_unchanged_by_approval_queue_trim"
        status: pass
    human_judgment: false

duration: 16min
completed: 2026-07-04
status: complete
---

# Phase 16 Plan 05: Approval Queue Column Trim & Reorder (Gap Closure) Summary

**`ApprovalQueueTable.Meta` gains `exclude`/`sequence` so Approve/Reject leads column 1 and three structurally-blank post-observation columns are dropped, while `CampaignRunTable` stays byte-for-byte spreadsheet-parity.**

## Performance

- **Duration:** 16 min
- **Started:** 2026-07-04T15:36:00Z (approx, based on prior commit timestamp)
- **Completed:** 2026-07-04T15:52:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- `ApprovalQueueTable.Meta` now overrides `exclude = ('weather', 'observation_outcome', 'publication_plans')` and `sequence = ('actions', 'approval_status', 'telescope_instrument', 'site', 'obs_date', 'ut_start', 'ut_end', '...')`, closing UAT Test 14's gap (scrolling past 16 mostly-blank columns to reach Approve/Reject)
- `CampaignRunTable` (Phase 15's D-09 spreadsheet-parity read path) is completely untouched — confirmed both by the diff (only `ApprovalQueueTable` lines changed) and by a new regression test
- Added `TestApprovalQueueColumns` with 3 tests: actions-leads, triage-trim, and a D-09 regression guard on `CampaignRunTable`

## Task Commits

Each task was committed atomically:

1. **Task 1: Trim and reorder ApprovalQueueTable columns via Meta.exclude + Meta.sequence** - `ed7b12a` (fix)
2. **Task 2: Column-contract test for the trimmed, reordered approval queue** - `eb9bdaa` (test)

**Plan metadata:** (this commit, following SUMMARY)

## Files Created/Modified
- `solsys_code/campaign_tables.py` - `ApprovalQueueTable.Meta` adds `exclude`/`sequence`; `CampaignRunTable` unchanged
- `solsys_code/tests/test_campaign_approval.py` - new `TestApprovalQueueColumns` class (3 tests)

## Decisions Made
- Fix scoped entirely to `ApprovalQueueTable.Meta` — no template, CSS, or `CampaignRunTable` changes, per the plan's scope guardrails and Phase 15's D-09 requirement.
- Used the django-tables2 `'...'` ellipsis token in `sequence` instead of enumerating all remaining columns, so the fix stays robust if `CampaignRunTable`'s field set changes in the future.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- UAT Test 14 gap closed: the pending-review approval queue now leads with Approve/Reject and hides the three always-blank post-observation columns.
- Phase 16 (all 5 plans, including this gap closure) is complete. Phase 17 (Coverage-Gap Analysis, deferrable to v2.1) remains the next milestone step.

---
*Phase: 16-submission-form-approval-queue-calendar-projection-write-pat*
*Completed: 2026-07-04*

## Self-Check: PASSED

- FOUND: solsys_code/campaign_tables.py
- FOUND: solsys_code/tests/test_campaign_approval.py
- FOUND: .planning/phases/16-submission-form-approval-queue-calendar-projection-write-pat/16-05-SUMMARY.md
- FOUND commit: ed7b12a
- FOUND commit: eb9bdaa
