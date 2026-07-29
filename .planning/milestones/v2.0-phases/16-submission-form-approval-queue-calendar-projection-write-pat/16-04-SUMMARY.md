---
phase: 16-submission-form-approval-queue-calendar-projection-write-pat
plan: 04
subsystem: web
tags: [django, django-tables2, visibility-gating, ui-discoverability]

# Dependency graph
requires:
  - phase: 16-02
    provides: campaigns:submit URL (public submission form)
  - phase: 16-03
    provides: campaigns:approval_queue URL (staff-only queue)
provides:
  - "D-09/SUBMIT-02: non-staff CampaignRunTableView queryset excludes pending_review rows (approved + rejected still visible)"
  - "CampaignListView.pending_count context value for the staff banner"
  - "Submit a Run entry buttons on campaign_list.html and campaignrun_table.html"
  - "Staff-only N pending review banner on campaign_list.html linking to campaigns:approval_queue"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "D-09 visibility change is a queryset-level .exclude(), never a template conditional -- mirrors the existing D-13 'restrict the queryset, not just the rendered table' discipline"

key-files:
  created: []
  modified:
    - solsys_code/campaign_views.py
    - src/templates/campaigns/campaign_list.html
    - src/templates/campaigns/campaignrun_table.html
    - solsys_code/tests/test_campaign_views.py

key-decisions:
  - "Three pre-existing Phase 15 anonymous-client tests (pagination row count, full run_status coverage, run_status multiselect semantics) switched to the staff client -- they test generic table mechanics unrelated to approval-status visibility and were written against the old fully-visible-to-anonymous assumption, which D-09 legitimately changes"

patterns-established: []

requirements-completed: [SUBMIT-01, SUBMIT-02]

coverage:
  - id: D1
    description: "Anonymous client GET the per-campaign table excludes every pending_review run from the queryset/paginator count; approved and rejected rows remain visible"
    requirement: "SUBMIT-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestNonStaffPendingReviewHidden.test_anonymous_queryset_excludes_pending_review"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestNonStaffPendingReviewHidden.test_anonymous_queryset_still_shows_approved_and_rejected"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestNonStaffPendingReviewHidden.test_anonymous_total_row_count_excludes_pending"
        status: pass
    human_judgment: false
  - id: D2
    description: "Staff client GET the same table sees every approval_status, including pending_review (unchanged)"
    requirement: "SUBMIT-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestNonStaffPendingReviewHidden.test_staff_sees_all_approval_statuses_including_pending"
        status: pass
    human_judgment: false
  - id: D3
    description: "CampaignListView context exposes pending_count = number of pending_review runs across all campaigns"
    requirement: "SUBMIT-01"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignListView.test_pending_count_in_context"
        status: pass
    human_judgment: false
  - id: D4
    description: "A 'Submit a Run' entry button appears on the campaigns list page and per-campaign table page; a staff-only 'N pending review' banner on the campaigns list links to the approval queue when is_staff and pending rows exist"
    requirement: "SUBMIT-01"
    verification:
      - kind: unit
        ref: "grep -n \"campaigns:submit\" src/templates/campaigns/campaign_list.html src/templates/campaigns/campaignrun_table.html"
        status: pass
      - kind: unit
        ref: "grep -n \"campaigns:approval_queue\" src/templates/campaigns/campaign_list.html (inside {% if request.user.is_staff and pending_count %})"
        status: pass
    human_judgment: false
  - id: D5
    description: "Manual UI check: banner/button layout, spacing, and visual hierarchy in a real browser (light/dark, staff vs anonymous view)"
    human_judgment: true
    rationale: "Visual layout of the header row, badge/banner placement, and Bootstrap alert styling requires rendering the page and looking at it -- cannot be asserted meaningfully by an integration test, per 16-VALIDATION.md's Manual-Only Verifications."
    verification: []

# Metrics
duration: 8min
completed: 2026-07-04
status: complete
---

# Phase 16 Plan 04: Non-Staff Visibility Filter & Entry Points Summary

**Non-staff visitors to a per-campaign table now see approved and rejected runs but never `pending_review` ones (queryset-level `.exclude()`, mirroring the existing D-13 discipline); "Submit a Run" buttons and a staff-only "N pending review" banner close the discoverability loop for the form (Plan 02) and approval queue (Plan 03).**

## Performance

- **Duration:** 8 min
- **Started:** 2026-07-04T11:54:45Z
- **Completed:** 2026-07-04T12:01:59Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- `CampaignRunTableView.get_queryset` non-staff branch now inserts `.exclude(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)` before `.values(*ALLOWED_FIELDS_FOR_NON_STAFF)` -- pending rows never enter the non-staff SELECT (D-09/SUBMIT-02, T-16-07). The staff branch and `ALLOWED_FIELDS_FOR_NON_STAFF` are both untouched, per the plan's explicit "don't touch column selection" instruction.
- `CampaignListView.get_context_data` adds `pending_count` (total `pending_review` `CampaignRun` rows across all campaigns), computed unconditionally since the template gates its display on `request.user.is_staff` (D-01/D-10: list membership itself unchanged).
- `campaign_list.html`: heading wrapped in a `d-flex justify-content-between align-items-center` header row with a `Submit a Run` button; a staff-only `{% if request.user.is_staff and pending_count %}` warning banner links to `campaigns:approval_queue`, exact markup per UI-SPEC.
- `campaignrun_table.html`: same header-row idiom with a `Submit a Run` button beside the per-campaign heading (no banner here -- list page only, per UI-SPEC).
- `solsys_code/tests/test_campaign_views.py`: new `TestNonStaffPendingReviewHidden` (4 tests: anonymous excludes pending, keeps approved+rejected, correct total count; staff sees all) plus `TestCampaignListView.test_pending_count_in_context`. Full `test_campaign_views` module: 21/21 passing. Full `solsys_code` suite: 300/300 passing.

## Task Commits

Each task was committed atomically, following RED-then-GREEN for Task 1's `tdd="true"` tag:

1. **Task 1 RED: failing D-09/pending_count tests** - `40c6ee7` (test)
2. **Task 1 GREEN: queryset exclude + pending_count context, plus updated pre-existing tests** - `3199771` (feat)
3. **Task 2: entry-point buttons + staff banner (templates)** - `af5b3f5` (feat)

**Plan metadata:** commit to follow (docs: complete plan)

## Files Created/Modified

- `solsys_code/campaign_views.py` - `CampaignRunTableView.get_queryset` non-staff branch gains the `.exclude()`; `CampaignListView.get_context_data` added for `pending_count`
- `src/templates/campaigns/campaign_list.html` - header row + `Submit a Run` button + staff pending banner
- `src/templates/campaigns/campaignrun_table.html` - header row + `Submit a Run` button
- `solsys_code/tests/test_campaign_views.py` - new `TestNonStaffPendingReviewHidden` class, `test_pending_count_in_context`, and 3 pre-existing tests switched to the staff client (see Decisions)

## Decisions Made

- Switched three pre-existing Phase 15 tests (`test_first_page_shows_25_rows_and_second_page_exists`, `test_default_load_shows_every_seeded_run_status_value`, `test_default_unfiltered_shows_all_rows`, `test_run_status_multiselect_or_semantics`) from the anonymous client to `self.client.force_login(self.staff_user)`. These tests assert generic table mechanics (pagination row counts, full `run_status` coverage, filter semantics) against the full 30-row fixture -- none are about approval-status visibility. D-09 legitimately drops the anonymous-visible row count from 30 to 20 (10 seeded `pending_review` rows, including the only row carrying `run_status=OBSERVED`), which broke these tests' old assumption that anonymous == full fixture. D-09's actual visibility behavior is proven separately and exhaustively by the new `TestNonStaffPendingReviewHidden` class.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Pre-existing Phase 15 tests broke under the new D-09 anonymous-visibility filter**
- **Found during:** Task 1 GREEN (running `test_campaign_views` after implementing the `.exclude()`)
- **Issue:** `test_first_page_shows_25_rows_and_second_page_exists`, `test_default_load_shows_every_seeded_run_status_value`, `test_default_unfiltered_shows_all_rows`, and `test_run_status_multiselect_or_semantics` all used the anonymous client and asserted against the full 30-row fixture. Once the non-staff queryset excludes `pending_review` (10 of 30 rows, including the fixture's only `run_status=OBSERVED` row), these counts/coverage assertions failed -- not because of a bug in the new filter, but because the tests encoded an assumption (anonymous sees everything) that D-09 intentionally invalidates.
- **Fix:** Switched these four tests to `self.client.force_login(self.staff_user)` since their actual intent (pagination mechanics, run_status coverage, filter OR-semantics) is orthogonal to approval-status visibility gating, which is now covered by its own dedicated test class.
- **Files modified:** `solsys_code/tests/test_campaign_views.py`
- **Verification:** Full `test_campaign_views` module (21/21) and full `solsys_code` suite (300/300) pass.
- **Committed in:** `3199771` (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug -- test-assumption update required by the plan's own intentional behavior change)
**Impact on plan:** No scope creep -- all four updated tests already existed in Phase 15; only their client/login state changed to keep them testing what they were actually meant to test.

## Issues Encountered

None beyond the auto-fixed test-assumption update above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- This was the final wave (Plan 04) of Phase 16. `campaigns:submit` (Plan 02) and `campaigns:approval_queue` (Plan 03) are now both linked from the public-facing pages, closing the SUBMIT-01/D-01 discoverability loop.
- Phase 16 as a whole (Plans 01-04) delivers SUBMIT-01..05 and CAL-01..03. Phase 17 (coverage-gap analysis, GAP-01/02) is next per the v2.0 roadmap and is explicitly deferrable to v2.1.
- No blockers.

---
*Phase: 16-submission-form-approval-queue-calendar-projection-write-pat*
*Completed: 2026-07-04*

## Self-Check: PASSED

- FOUND: solsys_code/campaign_views.py
- FOUND: src/templates/campaigns/campaign_list.html
- FOUND: src/templates/campaigns/campaignrun_table.html
- FOUND: solsys_code/tests/test_campaign_views.py
- FOUND: commit 40c6ee7
- FOUND: commit 3199771
- FOUND: commit af5b3f5
