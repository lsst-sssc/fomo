---
phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
plan: 05
subsystem: ui
tags: [django-templates, bootstrap4, django-tables2, campaigns]

# Dependency graph
requires:
  - phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
    provides: "ApprovalQueueView / approval_queue.html (pending / decided / review sections, D-07 order)"
provides:
  - "approval_queue.html Sites Needing Review section wrapped in a distinguishing border-warning Bootstrap 4 card with an 'action required' header and helper line"
  - "Rendering regression test class proving the card wrapper renders and D-07's document order (decided before review) is preserved"
affects: [22-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Presentation-only gap-closure plan: restyle a template section via Bootstrap 4 card/utility classes without touching the view, queryset, or table classes feeding it"

key-files:
  created: []
  modified:
    - src/templates/campaigns/approval_queue.html
    - solsys_code/tests/test_campaign_approval.py

key-decisions:
  - "Wrapped only the Sites Needing Review section in a card (border-warning header + helper line + render_table); left Pending Review and Recently Decided render_table calls untouched, per plan scope"
  - "Preserved D-07's locked document order exactly (pending / decided / sites-needing-review) — this is a visual-grouping fix, not a reorder"

patterns-established:
  - "Action-required Bootstrap 4 card pattern (border-warning + bg-warning card-header + helper <p> + render_table) for distinguishing actionable work-queue tables from read-only historical tables in campaign templates"

requirements-completed: [D-07]

coverage:
  - id: D1
    description: "Sites Needing Review section renders inside a visually distinct border-warning card with an 'action required' header and helper line, differentiating it from the plain Recently Decided table"
    requirement: "D-07"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueSitesNeedingReviewGrouping.test_sites_needing_review_renders_as_distinguishing_action_required_card"
        status: pass
    human_judgment: false
  - id: D2
    description: "D-07's locked document order (pending / decided / sites-needing-review) is preserved — Recently Decided still renders before Sites Needing Review"
    requirement: "D-07"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueSitesNeedingReviewGrouping.test_d07_order_preserved_decided_precedes_sites_needing_review"
        status: pass
    human_judgment: false

# Metrics
duration: 11min
completed: 2026-07-15
status: complete
---

# Phase 22 Plan 05: Sites Needing Review Card Grouping Summary

**Wrapped the Sites Needing Review section in a border-warning Bootstrap 4 card with an "action required" header, without reordering D-07's locked pending/decided/review document order**

## Performance

- **Duration:** 11 min
- **Started:** 2026-07-15T20:13:00+01:00 (approx, first commit 20:14:28)
- **Completed:** 2026-07-15T20:24:47+01:00
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- The Sites Needing Review section in `approval_queue.html` now renders inside a `card border-warning` with a bold `bg-warning` header reading "Sites Needing Review — action required" and a one-line helper describing the staff action, so it reads as an actionable work queue distinct from the plain "Recently Decided" table above it.
- Added a rendering regression test class (`TestApprovalQueueSitesNeedingReviewGrouping`) asserting both the new card wrapper markup and that D-07's document order (Recently Decided before Sites Needing Review) is preserved — verified against empty tables, so it tests presentation, not data.
- Confirmed presentation-only scope: `ApprovalQueueView.get_context_data()` and `ApprovalQueueTable` were not touched; only the template's HTML markup changed.

## Task Commits

Each task was committed atomically:

1. **Task 1: Give 'Sites Needing Review' distinct actionable visual weight (presentation-only, D-07 order preserved)** - `936f565` (feat)
2. **Task 2: Rendering regression test for the grouping + preserved D-07 order** - `13604eb` (test)

**Plan metadata:** (this commit, docs: complete plan)

## Files Created/Modified
- `src/templates/campaigns/approval_queue.html` - Wrapped the Sites Needing Review `<h5>` + `{% render_table review_table %}` block in a `card border-warning` with an "action required" header and helper `<p>`; `decided_table`'s render call and document order left untouched.
- `solsys_code/tests/test_campaign_approval.py` - Added `TestApprovalQueueSitesNeedingReviewGrouping` with two tests: card-wrapper presence and D-07 order preservation (decided heading index < review heading index).

## Decisions Made
- Only the "Sites Needing Review" section was wrapped in a card; the plan's optional suggestion to also lift "Pending Review" into consistent card styling was left out of scope (the plan said "at minimum" the review card is required, and no UAT gap was raised against Pending Review's presentation).
- The card header text is "Sites Needing Review — action required" (em dash), which still satisfies the existing `test_unresolved_review_row_renders_live_search_widget_and_resolve_button` assertion `assertIn('Sites Needing Review', content)` since it's a substring match.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Full `solsys_code` suite (477 tests, including the 2 new tests here) passes.
- `ruff check` / `ruff format --check` clean on both touched files (the template is not part of ruff's Python lint domain, and `ruff check .` across the repo does not flag it).
- No blockers for subsequent gap-closure plans (22-06) in this phase.

---
*Phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo*
*Completed: 2026-07-15*

## Self-Check: PASSED

- FOUND: src/templates/campaigns/approval_queue.html
- FOUND: solsys_code/tests/test_campaign_approval.py
- FOUND: .planning/phases/22-site-matching-at-submission-and-unmatched-site-resolution-wo/22-05-SUMMARY.md
- FOUND commit: 936f565
- FOUND commit: 13604eb
