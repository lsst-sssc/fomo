---
phase: 21-site-disambiguation-submitter-contact-opt-in
plan: 03
subsystem: ui
tags: [django-tables2, format_html, html5-datalist, difflib, stored-xss, django]

# Dependency graph
requires:
  - phase: 21-site-disambiguation-submitter-contact-opt-in
    provides: "Plan 21-01's build_site_candidates()/fuzzy_match_candidates() helpers"
provides:
  - "ApprovalQueueTable.render_site() inline site_selection input + fuzzy-matched <datalist> + 'Create new Observatory' link for unresolved actionable pending rows"
  - "ApprovalQueueTable.render_actions() single <form id=decide-form-{pk}> with two named submit buttons, replacing the prior two-<form> structure"
  - "ApprovalQueueView.get_context_data building the merged candidate pool once per request"
affects: [21-04 (decision-time D-06 clobber guard + reading site_selection + CreateObservatory ?obscode=/?next= handling)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "HTML5 <input list=...>/<datalist>/form=\"...\" cross-cell form targeting -- no JavaScript, no new endpoint (RESEARCH Pattern 4)"
    - "format_html_join for MPC-sourced <option> lists, mirroring render_window_start()'s T-20-03 positional-escaping precedent"

key-files:
  created: []
  modified:
    - solsys_code/campaign_tables.py
    - solsys_code/campaign_views.py
    - solsys_code/tests/test_campaign_approval.py

key-decisions:
  - "render_site() override lives on ApprovalQueueTable (not CampaignRunTable) since only ApprovalQueueTable carries self.show_actions/self.candidate_pool -- overriding on the parent would AttributeError for the per-campaign table"
  - "Only the candidate display string (not the obscode) is emitted as each <option value=...> -- the resolved obscode is looked up server-side from the submitted site_selection text in Plan 21-04, never from a hidden option attribute"
  - "Task 3's tests were written after Tasks 1-2's implementation (tdd=\"true\" on the task, but the plan's own task ordering -- build then prove -- means the RED phase collapses into 'already GREEN', same precedent as Plan 21-01's Tasks 2-3"

patterns-established:
  - "render_actions()'s two hidden-input-per-form action fields became two name=\"action\" submit buttons sharing one form -- request.POST.get('action') in CampaignRunDecisionView.post() needed no change"

requirements-completed: [SITE-01]

coverage:
  - id: D1
    description: "ApprovalQueueTable.render_site() renders an inline site_selection input + fuzzy-matched <datalist> + always-visible 'Create new Observatory' link for an unresolved actionable pending row; resolved and decided-table rows are unchanged"
    requirement: "SITE-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueSiteDisambiguationUI.test_unresolved_pending_row_renders_site_input_datalist_and_create_link"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueSiteDisambiguationUI.test_resolved_pending_row_renders_no_site_selection_input"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueSiteDisambiguationUI.test_decided_table_renders_no_site_selection_input"
        status: pass
    human_judgment: false
  - id: D2
    description: "All submitter-controlled site_raw and MPC-sourced candidate text is escaped via format_html/format_html_join -- a stored-XSS attempt in site_raw never reaches the response unescaped"
    requirement: "SITE-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueSiteDisambiguationUI.test_site_raw_script_injection_is_escaped_not_rendered_raw"
        status: pass
    human_judgment: false
  - id: D3
    description: "render_actions() collapses the row's two Approve/Reject forms into one <form id=decide-form-{pk}>, and the Site column's input submits into it via form=; existing approve/reject POST semantics are unchanged"
    requirement: "SITE-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py (full TestApproval/TestCalendarProjection/TestCalendarNoChurn coverage, 33 pre-existing tests all still passing)"
        status: pass
    human_judgment: false
  - id: D4
    description: "The merged candidate pool is built once per request (ApprovalQueueView.get_context_data), not per row"
    requirement: "SITE-01"
    verification:
      - kind: unit
        ref: "solsys_code manage.py test full suite (409 tests) -- no per-row MPC/cache calls observed, code inspection confirms single build_site_candidates() call site"
        status: pass
    human_judgment: false

duration: 21min
completed: 2026-07-11
status: complete
---

# Phase 21 Plan 03: Site Disambiguation UI (Approval Queue) Summary

**Inline `<input list=...>`/`<datalist>` site-disambiguation control wired into the staff approval queue's Site column, submitting into a single collapsed per-row `<form>` via the HTML5 `form=` attribute -- no new endpoint, no JavaScript.**

## Performance

- **Duration:** 21 min
- **Started:** 2026-07-11T14:48:xx Z (resumed from an interrupted prior attempt; no commits existed for this plan before this session)
- **Completed:** 2026-07-11T15:09:48Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- `ApprovalQueueTable.render_site()` override: unresolved actionable pending rows render an escaped `site_selection` input pre-filled with `site_raw`, a `<datalist>` of up to 5 `difflib`-fuzzy-matched candidates from Plan 21-01's `build_site_candidates()`/`fuzzy_match_candidates()`, and an always-visible "Create new Observatory" link (`?obscode=`/`?next=` pre-filled, handling lands in Plan 21-04); resolved rows and the read-only decided table delegate unchanged to `CampaignRunTable.render_site`
- `ApprovalQueueTable.render_actions()` refactored from two independent `<form>`s to one `<form id="decide-form-{pk}">` with two `name="action"` submit buttons, so the Site column's input can target it cross-cell via `form=`; `CampaignRunDecisionView.post()`'s `request.POST.get('action')` read is unchanged
- `ApprovalQueueTable.__init__` gained a `candidate_pool=None` kwarg; `ApprovalQueueView.get_context_data` calls `build_site_candidates()` exactly once per request and passes it into the pending table only (the decided table doesn't need it)
- `TestApprovalQueueSiteDisambiguationUI` (4 new tests): proves the datalist/input/create-link render for an unresolved row, proves `<script>` in `site_raw` is HTML-escaped (T-21-01 stored-XSS mitigation), and proves resolved/decided rows render no `site_selection` input

## Task Commits

Each task was committed atomically:

1. **Task 1: Single-form render_actions + candidate_pool kwarg + render_site override** - `e905297` (feat)
2. **Task 2: Build the merged candidate pool once per request and pass it to the table** - `1604632` (feat)
3. **Task 3: Datalist-rendering + stored-XSS tests** - `ae2d8e4` (test)

_Note: Task 3 was `tdd="true"`, but this plan's own task ordering builds the feature (Tasks 1-2) before proving it (Task 3) -- there was no meaningful RED state to establish separately (writing the tests before Tasks 1-2 landed would have meant reverting already-committed, already-verified implementation). Tests passed immediately on first run against the Task 1-2 implementation; same precedent as Plan 21-01's Tasks 2-3._

## Files Created/Modified
- `solsys_code/campaign_tables.py` - `ApprovalQueueTable.render_site()` override, `render_actions()` single-form refactor, `candidate_pool` kwarg, new imports (`format_html_join`, `urlencode`, `fuzzy_match_candidates`)
- `solsys_code/campaign_views.py` - `ApprovalQueueView.get_context_data` builds and passes `candidate_pool`
- `solsys_code/tests/test_campaign_approval.py` - `TestApprovalQueueSiteDisambiguationUI` (4 tests)

## Decisions Made
- `render_site()` override placed on `ApprovalQueueTable`, not `CampaignRunTable` -- only `ApprovalQueueTable` instances carry `self.show_actions`/`self.candidate_pool`, matching the plan's explicit guidance to avoid an `AttributeError` on the per-campaign table
- Only the candidate display string is emitted as each `<option value="...">` (not the obscode) -- the obscode resolution from the submitted `site_selection` text happens server-side in Plan 21-04, keeping this plan's markup minimal and avoiding leaking obscodes into a second hidden attribute
- Reused `campaign_utils._flatten_mpc_candidates()` (a "private" helper, leading underscore) directly in the new test class to build a deterministic candidate pool from `BULK_MPC_FIXTURE`, matching the existing `TestSiteFuzzyMatch` class's within-module convention of reaching into `campaign_utils` internals for fixture construction

## Deviations from Plan

None - plan executed exactly as written. The pre-existing uncommitted diff found in the working tree at session start (import-only additions to `campaign_tables.py`: `format_html_join`, `urlencode`, `fuzzy_match_candidates`) matched exactly what Task 1 needed and was folded into Task 1's commit.

## Issues Encountered
None. A prior session attempt was interrupted before any task commit; this session started fresh per the resume note and completed all three tasks without re-doing any prior work (none existed).

## User Setup Required

None - no external service configuration required. No new packages installed.

## Next Phase Readiness
- `site_selection` is now emitted as a POST field name by the inline input, ready for Plan 21-04's `CampaignRunDecisionView.post()` to read via `request.POST.get('site_selection', '')`
- The "Create new Observatory" link's `?obscode=`/`?next=` query string is already correctly formed; Plan 21-04 wires the `CreateObservatory` view to actually honor those params (pre-fill + redirect)
- Full `python manage.py test solsys_code` suite (409 tests) is green -- wave-merge gate satisfied before Plan 21-04 begins
- No blockers.

---
*Phase: 21-site-disambiguation-submitter-contact-opt-in*
*Completed: 2026-07-11*

## Self-Check: PASSED

- FOUND: solsys_code/campaign_tables.py
- FOUND: solsys_code/campaign_views.py
- FOUND: solsys_code/tests/test_campaign_approval.py
- FOUND commit: e905297 (feat)
- FOUND commit: 1604632 (feat)
- FOUND commit: ae2d8e4 (test)
