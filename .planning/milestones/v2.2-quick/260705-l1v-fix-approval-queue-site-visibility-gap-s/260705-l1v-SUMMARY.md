---
phase: quick-260705-l1v
plan: 01
subsystem: campaign-coordination
tags: [django-tables2, django, orm, testing]

# Dependency graph
requires:
  - phase: 16
    provides: Submission form, approval queue, and CampaignRunDecisionView built in Phase 16
provides:
  - render_site fallback that surfaces site_raw for pending (site_needs_review=False) runs
  - resolve_site(create_placeholder=False) opt-out preventing placeholder Observatory fabrication on approve
affects: [campaign-coordination, observatory]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Keyword-only opt-out parameter (create_placeholder) on a shared helper so an unvetted caller can opt out of a side effect while the vetted caller's default behavior is untouched"

key-files:
  created: []
  modified:
    - solsys_code/campaign_tables.py
    - solsys_code/campaign_utils.py
    - solsys_code/campaign_views.py
    - solsys_code/tests/test_campaign_approval.py

key-decisions:
  - "render_site now falls back to site_raw whenever the resolved short_name is empty (regardless of site_needs_review), with two distinct presentations: muted-italic 'pending review' (no icon) when site_needs_review is False, and the pre-existing warning-triangle failure styling when it is True."
  - "resolve_site's new create_placeholder keyword defaults True so the CSV-import caller (import_campaign_csv, positional call) is completely unaffected; only the approval endpoint opts out with create_placeholder=False."

patterns-established:
  - "Shared data-resolution helpers exposed to both a vetted batch-import path and an unvetted public-submission path should gate risky side effects (row fabrication) behind an explicit keyword-only parameter defaulting to the existing/vetted behavior."

requirements-completed: [SUBMIT-03, D-07]

coverage:
  - id: D1
    description: "Pending approval-queue rows show the submitted site_raw text even though site_needs_review is False pre-approval, while the existing failure-indicator presentation is preserved when resolution genuinely ran and failed."
    requirement: "SUBMIT-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueSiteVisibility.test_pending_unresolved_site_shows_site_raw"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueSiteVisibility.test_pending_blank_site_raw_renders_empty_cell"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueSiteVisibility.test_resolution_failed_site_still_shows_site_raw_with_failure_indicator"
        status: pass
    human_judgment: false
  - id: D2
    description: "Approving a run whose site_raw matches no Observatory and no MPC obscode leaves site=None + site_needs_review=True, creates zero placeholder Observatory rows, and still transitions the run to APPROVED (site failure never blocks approval, D-07)."
    requirement: "D-07"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalSiteResolution.test_approving_unresolvable_free_text_site_creates_no_placeholder_observatory"
        status: pass
    human_judgment: false
  - id: D3
    description: "resolve_site(create_placeholder=False) returns (None, True) and creates no Observatory, while the default resolve_site(...) call (CSV-import path) is unchanged and still creates a tier-3 placeholder Observatory."
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalSiteResolution.test_resolve_site_create_placeholder_false_creates_no_observatory"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalSiteResolution.test_resolve_site_default_still_creates_placeholder_observatory"
        status: pass
    human_judgment: false

# Metrics
duration: ~20min
completed: 2026-07-05
status: complete
---

# Quick Task 260705-l1v: Fix Approval Queue Site-Visibility Gap Summary

**Pending-review runs now show their submitted site text in the approval queue, and approving an unresolvable free-text site no longer fabricates a fake Observatory row.**

## Performance

- **Duration:** ~20 min
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- `CampaignRunTable.render_site()` falls back to `site_raw` whenever the resolved `site__short_name` is empty and `site_raw` is non-empty, regardless of `site_needs_review` -- fixing the blank Site column staff saw for every pending public submission (D-07 leaves `site_needs_review` False pre-approval).
- Added a distinct not-yet-resolved presentation (muted italic, no warning icon) for the pending case, keeping the existing warning-triangle "could not be automatically resolved" styling exactly for the genuine-failure case (`site_needs_review=True`).
- `resolve_site()` gained a keyword-only `create_placeholder: bool = True` parameter; when `False`, tier 3's placeholder-`Observatory`-creation is skipped and the site is flagged for manual review with no row created.
- `CampaignRunDecisionView.post()` now calls `resolve_site(run.site_raw, create_placeholder=False)` on approve, so unresolvable free-text facility nicknames (e.g. `'DCT'`) no longer pollute the `Observatory` table. The run still transitions to `APPROVED` with `site=None`/`site_needs_review=True` -- site resolution failure never blocks approval (D-07).
- `import_campaign_csv`'s existing positional `resolve_site(site_raw)` call is completely unaffected -- the CSV path still creates tier-3 placeholders as before.

## Task Commits

Each task was committed atomically:

1. **Task 1: Show site_raw in the approval queue for unresolved-but-typed pending runs** - `63ea77b` (fix)
2. **Task 2: Stop the approval endpoint from fabricating placeholder Observatory rows** - `959a78d` (fix)

_Plan metadata commit will be added separately by the orchestrator._

## Files Created/Modified
- `solsys_code/campaign_tables.py` - `render_site()` fallback now keys off empty `site__short_name` + non-empty `site_raw`, with separate pending-vs-failed presentations.
- `solsys_code/campaign_utils.py` - `resolve_site()` gained `create_placeholder: bool = True` keyword-only param; tier 3 skipped and `(None, True)` returned when `False`.
- `solsys_code/campaign_views.py` - `CampaignRunDecisionView.post()` approval call site passes `create_placeholder=False`; comment updated to explain the D-07/CAL-01 rationale.
- `solsys_code/tests/test_campaign_approval.py` - added `TestApprovalQueueSiteVisibility` (3 tests) and `TestApprovalSiteResolution` (3 tests), covering both defects and the CSV-path non-regression.

## Decisions Made
- `render_site` distinguishes "not yet attempted" (site_needs_review=False, site_raw present) from "attempted and failed" (site_needs_review=True) with two different presentations, rather than collapsing both into the existing failure-icon styling -- avoids implying resolution already failed for a run that hasn't been decided yet.
- `create_placeholder` defaults to `True` (not `False`) so the change is purely additive for the existing CSV-import caller; only the approval endpoint's call site was touched to opt out.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Both approval-flow defects (visibility gap, placeholder fabrication) are fixed and covered by DB-backed tests.
- Full `solsys_code` suite (332 tests) and `ruff check`/`ruff format --check` on touched files are clean.
- No follow-on work identified; this quick task is self-contained.

---
*Phase: quick-260705-l1v*
*Completed: 2026-07-05*

## Self-Check: PASSED
