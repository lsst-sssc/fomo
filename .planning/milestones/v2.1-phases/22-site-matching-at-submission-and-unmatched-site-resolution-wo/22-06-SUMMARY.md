---
phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
plan: 06
subsystem: ui
tags: [django, django-tables2, campaign-approval, observatory, gap-closure]

# Dependency graph
requires:
  - phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo (22-04, 22-05)
    provides: "SiteSearchView q/site_raw/site_selection fallback resolution; Sites Needing Review card grouping (D-07 order preserved)"
provides:
  - "NEEDS_REVIEW_NAME_PREFIX + is_placeholder_observatory() shared placeholder-detection helper in campaign_utils.py"
  - "ApprovalQueueTable.render_site() correction-widget branch for a resolve-mode placeholder-site row"
  - "CampaignRunDecisionView._resolve_site() placeholder-replacement path, D-06 racing protection keyed on previous_site_id"
affects: [campaign-approval-queue, observatory-resolution]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Shared prefix-constant + pure DB-free detection helper (is_placeholder_observatory) as the single source of truth for tier-3 placeholder detection, consumed by both the render layer and the write-path guard"
    - "Conditional queryset-update racing guard keyed on the exact pre-read state value (site_id=previous_site_id) rather than a hard-coded null check, so the same guard shape covers both the unresolved and placeholder-replacement cases"

key-files:
  created: []
  modified:
    - solsys_code/campaign_utils.py
    - solsys_code/campaign_tables.py
    - solsys_code/campaign_views.py
    - solsys_code/tests/test_campaign_approval.py

key-decisions:
  - "NEEDS_REVIEW_NAME_PREFIX is a new public constant in campaign_utils.py so both the tier-3 create call and the detection helper read from a single source of truth (no ad-hoc string literal duplication)."
  - "The D-06 conditional-claim filter is keyed on site_id=previous_site_id (captured before any write) instead of the old hard-coded site__isnull=True -- Django treats site_id=None as IS NULL, so the unresolved case is byte-equivalent, while the placeholder case gets the same racing protection."

requirements-completed: [D-06, D-08, D-09]

coverage:
  - id: D1
    description: "Shared NEEDS_REVIEW_NAME_PREFIX constant + is_placeholder_observatory() helper; tier-3 create uses the constant (no behavioral change to resolve_site())"
    requirement: "D-09"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py::TestIsPlaceholderObservatory"
        status: pass
    human_judgment: false
  - id: D2
    description: "ApprovalQueueTable.render_site() shows the live-search correction widget for a resolve-mode, actionable row whose site is a tier-3 placeholder; a genuine-site retry row (CR-01) and any show_actions=False row (WR-01) still render plain text"
    requirement: "D-08"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py::TestPlaceholderSiteReplacement::test_placeholder_row_renders_live_search_widget_not_plain_text"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py::TestPlaceholderSiteReplacement::test_placeholder_row_read_only_table_never_renders_widget"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py::TestSitesNeedingReview::test_retry_row_renders_plain_text_site_and_resolve_button_no_input"
        status: pass
    human_judgment: false
  - id: D3
    description: "CampaignRunDecisionView._resolve_site() replaces a placeholder site via the conditional claim keyed on previous_site_id, never re-resolves a genuine site, never fabricates a second placeholder on failure, and preserves the non-reverting projection/flag-clear discipline"
    requirement: "D-06"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py::TestPlaceholderSiteReplacement::test_placeholder_replacement_repoints_site_and_clears_review_flag"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py::TestPlaceholderSiteReplacement::test_placeholder_replacement_failure_fabricates_no_second_placeholder"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py::TestPlaceholderSiteReplacement::test_genuine_site_still_never_re_resolved_when_replacing_placeholder_would_apply"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py::TestPlaceholderSiteReplacement::test_racing_second_resolve_after_placeholder_replacement_does_not_double_write"
        status: pass
    human_judgment: false

duration: 44min
completed: 2026-07-15
status: complete
---

# Phase 22 Plan 06: Placeholder-Site Correction Escape Hatch Summary

**Closes UAT gap 2B: a Sites Needing Review row for a tier-3 PLACEHOLDER Observatory (e.g. `Observatory(obscode='DCT')`, name `NEEDS REVIEW: DCT`, blank timezone) now shows the live-search correction widget and can be replaced through the UI, while D-06 racing/never-re-resolve protection, CR-01's genuine-site retry state, WR-01's read-only-table suppression, and D-09's never-fabricate invariant all stay intact.**

## Performance

- **Duration:** 44 min
- **Started:** 2026-07-15T20:49:48+01:00
- **Completed:** 2026-07-15T21:33:41+01:00
- **Tasks:** 3/3 completed
- **Files modified:** 4

## Accomplishments

- Added `NEEDS_REVIEW_NAME_PREFIX` + `is_placeholder_observatory()` in `campaign_utils.py` as the single source of truth for tier-3 placeholder detection; `resolve_site()`'s tier-3 create now builds its name from the constant instead of an ad-hoc string literal.
- `ApprovalQueueTable.render_site()` now falls through to the existing live-search correction widget when a resolve-mode, actionable row's site is a placeholder Observatory -- while a genuinely-resolved site (the CR-01 projection-failed retry state) and every `show_actions=False` row (WR-01) still render plain text exactly as before.
- `CampaignRunDecisionView._resolve_site()`'s D-06 never-re-resolve guard now also enters resolution for a placeholder site (not just `site is None`), and its conditional site-claim is keyed on the exact pre-read site state (`site_id=previous_site_id`) rather than a hard-coded `site__isnull=True` -- preserving racing-POST protection for both the unresolved and placeholder-replacement cases, and never fabricating a second placeholder on a failed replacement (D-09).

## Task Commits

Each task was committed atomically:

1. **Task 1: Shared placeholder-Observatory prefix constant + detection helper** - `ef97bd2` (feat)
2. **Task 2: render_site() shows the correction widget for a placeholder-site resolve row** - `03bb0e9` (feat)
3. **Task 3: _resolve_site() replaces a placeholder site (D-06 racing + never-re-resolve preserved) + tests** - `7bd649e` (feat)

**Plan metadata:** (final docs commit follows this SUMMARY)

## Files Created/Modified

- `solsys_code/campaign_utils.py` - `NEEDS_REVIEW_NAME_PREFIX` constant + `is_placeholder_observatory()` helper; tier-3 create now uses the constant.
- `solsys_code/campaign_tables.py` - `ApprovalQueueTable.render_site()` placeholder-detection branch (resolve mode + `show_actions` only), falling through to the existing `_render_site_search_widget()`.
- `solsys_code/campaign_views.py` - `CampaignRunDecisionView._resolve_site()` placeholder-replacement path; conditional claim keyed on `previous_site_id`.
- `solsys_code/tests/test_campaign_approval.py` - `TestIsPlaceholderObservatory` (helper unit tests) and `TestPlaceholderSiteReplacement` (render + view tests: widget rendering, WR-01 suppression, successful replacement, D-09 no-fabrication-on-failure, D-06 never-re-resolve, racing-guard shape after replacement).

## Decisions Made

- `NEEDS_REVIEW_NAME_PREFIX = 'NEEDS REVIEW: '` is exported (no leading underscore) from `campaign_utils.py` so `campaign_tables.py` and `campaign_views.py` can both import it alongside `is_placeholder_observatory()`.
- `is_placeholder_observatory()` is a pure, DB-free string check on the already-loaded `name` field (never triggers a query) -- callers already pass `select_related('site')` instances.
- The conditional-claim filter in `_resolve_site()` was changed from `site__isnull=True` to `site_id=previous_site_id`, where `previous_site_id` is captured (`run.site_id`) before any write. Django's ORM treats `site_id=None` as `IS NULL`, so the unresolved case is byte-equivalent to the old behavior; the placeholder case gets identical racing protection because a competing POST that already changed the site away from `previous_site_id` matches zero rows.
- No new database migration was needed -- the placeholder Observatory shape (`Observatory.objects.create(obscode=..., name=..., short_name=...)`, relying on model defaults for `timezone`/`altitude`/`observations_type`) was already produced by `resolve_site()`'s existing tier-3 fallback; this plan only adds detection and a correction path around it.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

While writing `TestPlaceholderSiteReplacement`, the shared `_make_placeholder_run()` test helper's default `kwargs` dict unconditionally called `self._make_placeholder_observatory()` even when a test passed its own `site=` override, creating two Observatory rows with the same `obscode`/`name` and tripping the DB's `UNIQUE constraint failed: ...observatory.name` in three tests. Fixed by only constructing the default placeholder when `'site'` is absent from the caller's `overrides` (test-file-only fix, not part of the shipped `solsys_code` application code, so not tracked as a Rule 1/2/3 deviation against the plan's `files_modified`).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

UAT gap 2B is closed: a placeholder-site Sites Needing Review row is now correctable through the UI, matching the treatment already given to gaps 2A (22-05) and the SiteSearchView term-resolution gap (22-04). All 487 tests in the `solsys_code` Django test suite pass (up from 477 as of 22-05, +10 new tests from this plan); `ruff check` and `ruff format --check` are clean on every touched file. No blockers for the next phase-22 gap-closure plan or for closing out Phase 22 UAT overall.

---
*Phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo*
*Completed: 2026-07-15*

## Self-Check: PASSED

All 4 modified source/test files and the SUMMARY.md itself exist on disk; all 3 task commits (`ef97bd2`, `03bb0e9`, `7bd649e`) verified present in `git log --oneline --all`.
