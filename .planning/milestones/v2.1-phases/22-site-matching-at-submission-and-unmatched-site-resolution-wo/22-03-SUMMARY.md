---
phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
plan: 03
subsystem: api
tags: [django, htmx, django-tables2, campaigns, calendar-projection]

requires:
  - phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo-plan-02
    provides: corrected hx-trigger grammar, three-way click-to-fill id wiring, campaigns:site_search live-search endpoint already wired into the pending row
provides:
  - "_project_calendar_event(run) -> bool module-level helper (extracted from the approve branch, returns True/False so callers can distinguish a projected event from a by-design skip)"
  - "CampaignRunDecisionView.post() resolve_site action -- resolves an approved run's still-unmatched site via the existing display-string->obscode pool mapping + resolve_site(create_placeholder=False), honoring D-06's never-re-resolve guard through a concurrency-safe conditional site-claim update, then retroactively fires the deferred calendar projection"
  - "ApprovalQueueView third 'Sites Needing Review' table (review_qs/review_table) listing APPROVED+site_needs_review=True runs, reusing the once-per-request candidate_pool"
  - "ApprovalQueueTable mode='pending'|'resolve' constructor flag; resolve-mode render_site()/render_actions() (live-search widget or plain-text retry fallback + single Resolve button)"
affects: []

tech-stack:
  added: []
  patterns:
    - "Shared, revert-agnostic helper extraction: _project_calendar_event() carries no error handling of its own -- callers (approve's reverting except block, resolve_site's non-reverting except block) each own their own failure semantics"
    - "Flag-clears-only-after-success ordering: site_needs_review is set False in code that runs strictly after the projection call returns without raising, never before it and never in the same statement as the site write"
    - "Conditional-update site claim: .filter(pk=pk, approval_status=APPROVED, site_needs_review=True, site__isnull=True).update(site=site) as the concurrency guard, mirroring the pre-existing approve/reject updated_count==1 discipline instead of transaction.atomic()+select_for_update()"
    - "Shared widget-markup factoring: _render_site_search_widget(site_raw, input_id, form_id) lets pending and resolve rows share identical htmx markup, differing only in which form id the input's form= attribute targets"

key-files:
  created: []
  modified:
    - solsys_code/campaign_views.py
    - solsys_code/campaign_tables.py
    - src/templates/campaigns/approval_queue.html
    - solsys_code/tests/test_campaign_approval.py

key-decisions:
  - "_project_calendar_event()'s bool return (True = insert_or_create_calendar_event() was called, False = skipped by design) drives resolve_site's two distinct success messages, per 22-REVIEWS.md finding 6 -- the approve branch ignores the return entirely, preserving its existing behavior byte-for-byte"
  - "resolve-mode rows submit into their own resolve-form-{pk} (distinct from the pending row's decide-form-{pk}), matching the plan's explicit form-id naming rather than reusing decide-form-{pk} for both modes"
  - "The projection-failed retry state (run.site already set, site_needs_review still True) is handled by review_qs's filter alone -- no special-case query branch needed, since filtering on site_needs_review=True naturally includes it"
  - "resolve_site's projection failure uses its own non-reverting try/except (never approve's revert-to-PENDING_REVIEW block) -- reverting an already-APPROVED run would resurrect it into the pending queue"

requirements-completed: [D-06, D-07, D-08, D-10]

coverage:
  - id: D1
    description: "_project_calendar_event() extracted as a module-level bool-returning helper; approve branch calls it unchanged (ignoring the return), with its existing revert-on-failure except block untouched"
    requirement: "D-08"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApproval and TestCalendarProjection (all pre-existing tests, unchanged, still pass)"
        status: pass
    human_judgment: false
  - id: D2
    description: "resolve_site action resolves an approved run's still-unmatched site via the existing pool-mapping + resolve_site(create_placeholder=False), then projects the deferred CalendarEvent for a single-night ground run"
    requirement: "D-08"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_resolve_success_single_night_ground_run_projects_calendar_event"
        status: pass
    human_judgment: false
  - id: D3
    description: "site_needs_review clears only after a successful projection; a projection failure keeps approval_status APPROVED, keeps the resolved site, and keeps the flag True so the row stays visible in review_table for retry"
    requirement: "D-08"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_resolve_retryable_projection_failure_stays_approved_site_saved_flag_stays_true"
        status: pass
    human_judgment: false
  - id: D4
    description: "The site write is a single conditional queryset update (pk + APPROVED + site_needs_review=True + site__isnull=True) so two racing staff resolve POSTs cannot double-write; the losing POST is a proven no-op"
    requirement: "D-06"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_resolve_lost_race_no_op_warns"
        status: pass
    human_judgment: false
  - id: D5
    description: "resolve_site never re-resolves a run whose site is already set (the projection-failed retry state); such a run skips resolution but still re-attempts and clears on success. A POST for a non-eligible run (PENDING_REVIEW, or already resolved) is rejected with a warning"
    requirement: "D-06"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_resolve_never_re_resolves_already_set_site_but_retries_projection"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_resolve_rejects_pending_review_run"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_resolve_rejects_already_resolved_run"
        status: pass
    human_judgment: false
  - id: D6
    description: "A third 'Sites Needing Review' table lists exactly the APPROVED+site_needs_review=True runs (including projection-failed retry rows), reusing the once-per-request candidate_pool (never a second build_site_candidates() call)"
    requirement: "D-07"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_review_table_context_lists_only_approved_needs_review_runs"
        status: pass
      - kind: unit
        ref: "grep -c 'build_site_candidates()' in ApprovalQueueView.get_context_data -- called exactly once"
        status: pass
    human_judgment: false
  - id: D7
    description: "Site-unresolved resolve rows render the same live-search widget (hx-get, corrected trigger, Create-new link) as pending rows; a retry row (site set, flag True) renders plain-text site with no input; every row has a single Resolve (btn-primary) action; empty state renders the configured copy"
    requirement: "D-10"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_unresolved_review_row_renders_live_search_widget_and_resolve_button"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_retry_row_renders_plain_text_site_and_resolve_button_no_input"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_review_table_empty_state_renders_configured_copy"
        status: pass
    human_judgment: false
  - id: D8
    description: "A range/TBD run resolves its site and clears the flag with no CalendarEvent created (helper returned False), surfacing the plain 'Site resolved.' message; an unresolvable selection leaves site None and the flag True with an error message"
    requirement: "D-08"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_resolve_range_tbd_run_clears_flag_with_no_calendar_event"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_resolve_unresolvable_selection_leaves_site_none_and_flag_true"
        status: pass
    human_judgment: false

duration: ~35min
completed: 2026-07-15
status: complete
---

# Phase 22 Plan 03: Sites Needing Review Table + resolve_site Decision Action Summary

**Closes the last Phase 21 gap: a third "Sites Needing Review" table on the approval-queue page lists approved runs with an unresolved site, and a new `resolve_site` decision action resolves the site via a concurrency-safe conditional claim and retroactively fires the deferred CalendarEvent projection only after it succeeds.**

## Performance

- **Duration:** ~35 min
- **Tasks:** 2 completed (both non-TDD-plan-level but Task 1 used TDD; 1 follow-up fix commit correcting a form-id naming deviation)
- **Files modified:** 4

## Accomplishments
- Extracted the approve branch's inline CalendarEvent projection block into a module-level `_project_calendar_event(run: CampaignRun) -> bool` helper, with the only logic change being the bool return (True = event projected, False = skipped by design) — the approve branch's revert-on-failure `except Exception` block is otherwise byte-for-byte unchanged.
- Added a `resolve_site` action to `CampaignRunDecisionView.post()`: re-fetches the run fresh, validates `approval_status == APPROVED and site_needs_review`, resolves the site (skipping resolution entirely if `run.site` is already set — the projection-failed retry state), claims the site write via a single conditional `.filter(...).update(site=site)` so two racing staff POSTs can't double-write, then projects the calendar event inside its own non-reverting `try/except` and clears `site_needs_review` **only** after that projection succeeds.
- Added a third `review_table` to `ApprovalQueueView.get_context_data()` — `CampaignRun.objects.filter(approval_status=APPROVED, site_needs_review=True)`, no row cap (unlike the 20-row-capped decided table), reusing the same `candidate_pool` already built for `pending_table`.
- `ApprovalQueueTable` gained a `mode='pending'|'resolve'` constructor flag extending the existing `show_actions` convention; `render_site()`/`render_actions()` render the live-search widget + Resolve button for resolve-mode rows (or the plain-text site + Resolve button for the projection-failed retry state), factored through a shared `_render_site_search_widget()` helper so pending and resolve markup stay in lockstep.
- `approval_queue.html` gained the "Sites Needing Review" section heading and `{% render_table review_table %}` block.

## Task Commits

Each task was committed atomically:

1. **Task 1: Extract `_project_calendar_event()` + add `resolve_site` decision action (D-08, D-06)** - `fe9be5a` (feat)
2. **Task 2: Third "Sites Needing Review" table (D-07) + resolve-mode rendering (D-10)** - `2d1e799` (feat)
3. **Fix: resolve-mode rows submit into their own `resolve-form-{pk}`** - `98d21ab` (fix, Rule 1 self-correction within Task 2's scope)

**Plan metadata:** (this commit, pending)

## Files Created/Modified
- `solsys_code/campaign_views.py` - `_project_calendar_event()` helper; `CampaignRunDecisionView.post()`/`_resolve_site()` resolve_site action; `ApprovalQueueView.get_context_data()` third table
- `solsys_code/campaign_tables.py` - `ApprovalQueueTable.__init__` `mode` flag; `_render_site_search_widget()` shared markup factoring; resolve-mode `render_site()`/`render_actions()` branches
- `src/templates/campaigns/approval_queue.html` - "Sites Needing Review" heading + `{% render_table review_table %}`
- `solsys_code/tests/test_campaign_approval.py` - `TestSitesNeedingReview` (12 tests: resolve success, D-06 never-re-resolve + retry, retryable projection failure preserving the retry surface, lost-race no-op, state validation, range/TBD, unresolvable selection, review_table filtering, widget/plain-text markup, empty state)

## Decisions Made
- `_project_calendar_event()`'s bool return distinguishes "event created" from "skipped by design" (22-REVIEWS.md finding 6) — the approve branch ignores it entirely, so its own behavior is provably unchanged.
- The site write is a single conditional `.filter(pk=pk, approval_status=APPROVED, site_needs_review=True, site__isnull=True).update(site=site)`, deliberately writing `site` only — never `site_needs_review` in the same statement (finding 5, concurrency) and never before the projection succeeds (finding 3, the retry-surface dead-end fix).
- Resolve-mode forms use their own `resolve-form-{pk}` id (distinct from the pending row's `decide-form-{pk}`) — an initial Task 2 implementation reused `decide-form-{pk}` for both, then was corrected to match the plan's explicit naming and its acceptance-criteria grep check.
- `review_qs`'s filter (`approval_status=APPROVED, site_needs_review=True`) naturally includes the projection-failed retry state (site already set, flag still True) with no special-case query branch — the D-07 table and the D-06 retry semantics compose without extra code.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Resolve-mode form id initially reused `decide-form-{pk}` instead of the plan's `resolve-form-{pk}`**
- **Found during:** Task 2 (self-review before final commit, cross-checking the plan's explicit `<form id="resolve-form-{pk}"...>` text and its acceptance-criteria `grep -n "resolve-form-"` check)
- **Issue:** `render_site()`'s resolve-mode widget and `render_actions()`'s resolve-mode form both used `decide-form-{pk}`, which happened to still work functionally (both forms exist and one targets the other via `form=`) but didn't match the plan's specified naming or its grep-based acceptance criterion.
- **Fix:** `render_site()`/`render_actions()` now derive `resolve-form-{pk}` when `self.mode == 'resolve'`, keeping `decide-form-{pk}` for pending mode; updated the three affected test assertions to match.
- **Files modified:** `solsys_code/campaign_tables.py`, `solsys_code/tests/test_campaign_approval.py`
- **Verification:** `grep -n "resolve-form-" solsys_code/campaign_tables.py` matches; full `test_campaign_approval` suite (60 tests) still green.
- **Committed in:** `98d21ab`

---

**Total deviations:** 1 auto-fixed (Rule 1 — self-correction against the plan's own explicit spec before final commit)
**Impact on plan:** Cosmetic naming correction only; no behavior change to the resolve flow itself. No scope creep.

## Issues Encountered
- The Task 1 RED test asserting the retry surface stays visible in `review_table` (22-REVIEWS.md finding 3) necessarily depends on Task 2's `review_table` context key. Resolved by writing the model-level assertions (approval_status/site/flag) in Task 1's commit, then extending that same test method with the `review_table` context assertion once Task 2 landed `review_table` — both tasks' tests pass at their respective commit points, and the full assertion is present by the time the plan completes.
- CLAUDE.md's demo-notebook-companion rule does not apply to this plan — it only touches `campaign_views.py`/`campaign_tables.py`/`approval_queue.html`/tests, none of which are in the four modules (`telescope_runs.py`, `load_telescope_runs.py`, `sync_lco_observation_calendar.py`, `sync_gemini_observation_calendar.py`) that rule scopes.

## User Setup Required
None - no external service configuration required. No new packages installed.

## Next Phase Readiness
- All four Phase 22 requirements (D-06, D-07, D-08, D-10) are now shipped: SITE-01..03 fuzzy-match resolution UI (Phase 21) plus this milestone's live-search widget wiring (Plan 02) and the Sites Needing Review retry workflow (this plan) close every outstanding site-matching gap from Phase 21.
- Full `solsys_code` test suite (470 tests) passes; `ruff check`/`ruff format --check` clean on all touched files.
- No blockers. This is the last plan of Phase 22 and of the v2.1 milestone's active work — `/gsd-complete-milestone` is the next step per STATE.md's "Active: None" note (once this plan's phase-completion bookkeeping lands).

---
*Phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo*
*Completed: 2026-07-15*

## Self-Check: PASSED

All 4 modified files verified present on disk; all 3 task/fix commit hashes
(`fe9be5a`, `2d1e799`, `98d21ab`) verified present in git history.
