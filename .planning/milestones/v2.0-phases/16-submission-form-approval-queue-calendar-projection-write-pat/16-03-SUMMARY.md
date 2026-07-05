---
phase: 16-submission-form-approval-queue-calendar-projection-write-pat
plan: 03
subsystem: web
tags: [django, django-tables2, staff-gating, csrf, calendar-projection]

# Dependency graph
requires:
  - phase: 16-01
    provides: StaffRequiredMixin (dispatch-level is_staff gate)
  - phase: 16-02
    provides: CampaignRunSubmissionView creating PENDING_REVIEW CampaignRun rows (the queue's input)
provides:
  - ApprovalQueueView (StaffRequiredMixin + TemplateView) at campaigns:approval_queue -- staff-only
    two-section queue (pending actionable, recently-decided read-only, capped 20, -pk)
  - CampaignRunDecisionView (StaffRequiredMixin + View, POST-only) at campaigns:decide -- atomic
    conditional approve/reject transition, proven double-approve no-op
  - ApprovalQueueTable(CampaignRunTable) -- per-row Approve/Reject mini-forms with CSRF token
    minted via get_token(request)
  - CAMPAIGN:{pk} CalendarEvent projection on successful approve, routed through the shared
    insert_or_create_calendar_event() helper
affects: [16-04-calendar-projection-entry-points]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two independent Table instances built by hand in get_context_data (not MultiTableMixin) when the two querysets have asymmetric filtering, each given a distinct prefix= so pagination/sort query params don't collide"
    - "CSRF token for a per-row action form minted inside a Table's render_ method via django.middleware.csrf.get_token(request), with the request passed into the Table's __init__ as an explicit kwarg -- avoids needing to break out of {% render_table %} into a manual template row-loop"
    - "A sliced queryset ([:20]) handed to a django-tables2 Table must be materialized to a list first if the Table's Meta.order_by would otherwise trigger a second .order_by() call on the already-sliced queryset (Django raises 'Cannot reorder a query once a slice has been taken'); pass order_by=() explicitly to suppress the inherited default sort so the queryset's original ordering survives to first render"

key-files:
  created:
    - src/templates/campaigns/approval_queue.html
    - solsys_code/tests/test_campaign_approval.py
  modified:
    - solsys_code/campaign_views.py
    - solsys_code/campaign_tables.py
    - solsys_code/campaign_urls.py

key-decisions:
  - "CSRF token for the Actions column's per-row Approve/Reject mini-forms is minted via get_token(self.request) inside ApprovalQueueTable.render_actions, with request passed as an explicit __init__ kwarg -- kept {% render_table %} intact rather than switching to a manual template row-loop, per the plan's stated CSRF-handling alternative"
  - "decided_qs is materialized to a list (list(decided_qs)) before being handed to ApprovalQueueTable, with order_by=() passed explicitly, to fix a 'Cannot reorder a query once a slice has been taken' crash -- ApprovalQueueTable inherits CampaignRunTable.Meta.order_by=('-obs_date',), and django-tables2 re-applies that default sort at Table construction time by calling .order_by() again on the queryset, which Django refuses once [:20] has been applied"

patterns-established:
  - "ApprovalQueueTable(show_actions, request) kwargs on a Table subclass: pop non-django-tables2 kwargs before super().__init__ so a single Table class can render both an actionable and a read-only variant of the same columns"

requirements-completed: [SUBMIT-03, CAL-01, CAL-02, CAL-03]

coverage:
  - id: D1
    description: "ApprovalQueueView renders a two-section staff-only queue (pending with Approve/Reject actions, recently-decided read-only capped at 20 ordered -pk); anonymous/non-staff GET is redirected, never 200 with pending content"
    requirement: "SUBMIT-03"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestStaffGating.test_anonymous_get_approval_queue_redirects"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestStaffGating.test_non_staff_get_approval_queue_redirects"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestStaffGating.test_staff_get_approval_queue_succeeds"
        status: pass
    human_judgment: false
  - id: D2
    description: "CampaignRunDecisionView is POST-only and staff-gated; anonymous/non-staff POST is redirected with no state change; an invalid action value returns 400"
    requirement: "SUBMIT-03"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestStaffGating.test_anonymous_post_decide_redirects_and_makes_no_change"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestStaffGating.test_non_staff_post_decide_redirects_and_makes_no_change"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestApproval.test_invalid_action_returns_bad_request"
        status: pass
    human_judgment: false
  - id: D3
    description: "Approve/reject is an atomic conditional update; a double-approve is a proven no-op (approval_status stays APPROVED, CalendarEvent.count() unchanged, second POST surfaces the 'already decided' warning)"
    requirement: "SUBMIT-03"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestApproval.test_double_approve_is_noop"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestApproval.test_second_approve_surfaces_already_decided_warning"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestApproval.test_reject_path_sets_rejected_and_creates_no_event"
        status: pass
    human_judgment: false
  - id: D4
    description: "Approving a run with telescope_instrument + ut_start + ut_end creates a CalendarEvent keyed CAMPAIGN:{pk} via insert_or_create_calendar_event, target_list set to the campaign; missing any one of the three creates zero events"
    requirement: "CAL-01"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestCalendarProjection.test_approve_with_full_window_creates_calendar_event"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestCalendarProjection.test_approve_without_ut_end_creates_no_calendar_event"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestCalendarProjection.test_approve_without_ut_start_creates_no_calendar_event"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestCalendarProjection.test_approve_without_telescope_instrument_creates_no_calendar_event"
        status: pass
    human_judgment: false
  - id: D5
    description: "The projected CalendarEvent's target_list is the campaign's TargetList"
    requirement: "CAL-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestCalendarProjection.test_approve_with_full_window_creates_calendar_event"
        status: pass
    human_judgment: false
  - id: D6
    description: "Re-approving an already-approved run creates no duplicate CalendarEvent and no modified-timestamp churn"
    requirement: "CAL-03"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestCalendarNoChurn.test_second_approve_leaves_event_count_and_modified_unchanged"
        status: pass
    human_judgment: false
  - id: D7
    description: "Manual UI check: approval-queue layout legibility (two sections, badge colors, Actions column spacing) and the native confirm() Reject dialog copy/behavior in a real browser"
    human_judgment: true
    rationale: "Visual layout and a native browser confirm() dialog require rendering the page and clicking through it -- cannot be asserted meaningfully by an integration test, per 16-VALIDATION.md's Manual-Only Verifications."
    verification: []

# Metrics
duration: 21min
completed: 2026-07-04
status: complete
---

# Phase 16 Plan 03: Approval Queue, Decision Endpoint & Calendar Projection Summary

**Staff-gated two-section approval queue (pending actionable / recently-decided read-only), a POST-only atomic approve/reject endpoint whose double-approve is a proven no-op, and a `CAMPAIGN:{pk}` `CalendarEvent` projection on successful approve routed through the shared `insert_or_create_calendar_event()` helper.**

## Performance

- **Duration:** 21 min
- **Started:** 2026-07-04T11:46:20Z
- **Completed:** 2026-07-04T12:07:43Z
- **Tasks:** 2
- **Files modified:** 5 (2 created, 3 modified)

## Accomplishments

- `ApprovalQueueView` (`StaffRequiredMixin` + `TemplateView`) builds two independent `ApprovalQueueTable` instances by hand (16-RESEARCH.md Pattern 5): a `pending_qs` (actionable, Approve/Reject buttons) and a `decided_qs` (read-only, `show_actions=False`, capped 20 rows, ordered `-pk` since `CampaignRun` has no timestamp field)
- `CampaignRunDecisionView` (`StaffRequiredMixin` + `View`, `http_method_names = ['post']`) implements the single conditional `.filter(pk=pk, approval_status=PENDING_REVIEW).update(...)` that makes a double-approve/double-reject a proven no-op, with the calendar projection gated on `updated_count == 1 and action == 'approve'`
- `ApprovalQueueTable(CampaignRunTable)` adds an `Actions` column rendering side-by-side Approve/Reject POST mini-forms per row, with the CSRF token minted inline via `get_token(request)` (request passed in as a constructor kwarg) and the Reject button carrying the `confirm()` destructive-confirmation dialog from UI-SPEC
- Calendar projection: on a successful approve with `telescope_instrument`/`ut_start`/`ut_end` all present, `insert_or_create_calendar_event({'url': f'CAMPAIGN:{run.pk}'}, fields={...})` creates/updates the event -- never constructs `CalendarEvent` directly, keeping the `CAMPAIGN:` namespace collision-safe against the LCO/Gemini/classical sync commands
- `src/templates/campaigns/approval_queue.html` (new) -- Page 5 of the UI-SPEC, `{% render_table %}` for both sections, messages already wired via `tom_common/base.html`
- 14 new tests in `solsys_code/tests/test_campaign_approval.py`, all green; full `solsys_code` suite (295 tests) passes

## Task Commits

Each task was committed atomically:

1. **Task 1: ApprovalQueueTable + ApprovalQueueView + decision endpoint + URLs + template** - `9d9aff6` (feat)
2. **Task 2: test_campaign_approval.py (TDD-adjacent; includes a bug fix found while running the tests)** - `b132b35` (test)

**Plan metadata:** commit to follow (docs: complete plan)

## Files Created/Modified

- `solsys_code/campaign_views.py` - added `ApprovalQueueView(StaffRequiredMixin, TemplateView)` and `CampaignRunDecisionView(StaffRequiredMixin, View)`; fixed the sliced-queryset/order_by bug (see Deviations)
- `solsys_code/campaign_tables.py` - added `ApprovalQueueTable(CampaignRunTable)` with `show_actions`/`request` constructor kwargs and `render_actions`
- `solsys_code/campaign_urls.py` - added `approval_queue` (`approval-queue/`) and `decide` (`<int:pk>/decide/`) URL names
- `src/templates/campaigns/approval_queue.html` - new approval-queue template (UI-SPEC Page 5)
- `solsys_code/tests/test_campaign_approval.py` - 14 tests: `TestStaffGating` (5), `TestApproval` (4), `TestCalendarProjection` (4), `TestCalendarNoChurn` (1)

## Decisions Made

- CSRF token for the Actions column's per-row mini-forms is minted via `get_token(self.request)` inside `ApprovalQueueTable.render_actions`, with `request` passed as an explicit `__init__` kwarg -- kept `{% render_table %}` intact per the plan's stated alternative, rather than switching to a manual template row-loop.
- `decided_qs` (`[:20]`, ordered `-pk`) is materialized to `list(decided_qs)` before being handed to `ApprovalQueueTable`, with `order_by=()` passed explicitly to suppress the inherited `CampaignRunTable.Meta.order_by=('-obs_date',)` default sort. See Deviations for why this was necessary.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `ApprovalQueueView`'s decided-runs table crashed with "Cannot reorder a query once a slice has been taken"**
- **Found during:** Task 2 (writing `TestStaffGating.test_staff_get_approval_queue_succeeds`)
- **Issue:** `ApprovalQueueTable` inherits `CampaignRunTable.Meta.order_by = ('-obs_date',)`. Passing the already-sliced `decided_qs` (`.order_by('-pk')[:20]`) straight into the `Table` constructor triggers django-tables2's default-ordering machinery, which calls `.order_by()` again on the queryset to apply the inherited `-obs_date` sort -- Django raises `TypeError: Cannot reorder a query once a slice has been taken` because a slice was already applied.
- **Fix:** Materialize `decided_qs` to a plain `list(...)` before constructing the table (django-tables2 sorts Python lists via `TableListData.order_by`'s `list.sort()`, which has no such restriction), and pass `order_by=()` explicitly so the table doesn't re-apply the inherited `-obs_date` default sort on first render -- the query's original `-pk` selection order survives untouched. A later user-initiated column sort (via `RequestConfig`/query params) still works correctly against the materialized list.
- **Files modified:** `solsys_code/campaign_views.py`
- **Verification:** `TestStaffGating.test_staff_get_approval_queue_succeeds` (and the full `test_campaign_approval` module) passes; full `solsys_code` suite (295/295) passes.
- **Committed in:** `b132b35` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary for `ApprovalQueueView` to render at all for staff users. No scope creep -- fix is scoped entirely to the two lines that construct `decided_table`.

## Issues Encountered

- The plan's Task 1 acceptance-criteria grep (`grep -n "filter(pk=pk, approval_status" solsys_code/campaign_views.py`) doesn't match literally because `ruff format` wraps the conditional `.filter(...)` call across two lines (the single-line form is 151 characters, over the project's 120-column limit). The semantic requirement -- a single conditional `.filter(pk=pk, approval_status=PENDING_REVIEW).update(...)` -- is present and is exercised directly by `TestApproval.test_double_approve_is_noop`; verified via `grep -n "pk=pk, approval_status" solsys_code/campaign_views.py` instead. Not a functional gap, just a line-wrapping mismatch against the plan's literal grep string.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `campaigns:approval_queue` and `campaigns:decide` are live; Plan 02's `_notify_staff`'s `try/except NoReverseMatch` fallback around `reverse('campaigns:approval_queue')` is now dead code on the primary path (confirmed: `reverse()` succeeds directly, per the URL-resolution one-liner in this plan's verification) -- no follow-up edit is required since the fallback is harmless once the primary `reverse()` succeeds.
- `CAMPAIGN:{pk}` events are now live in `CalendarEvent` alongside the LCO/Gemini/classical-schedule events; Plan 04's calendar-projection entry points can rely on this namespace being populated for approved runs with a full telescope+date-range.
- No blockers for Plan 04.

---
*Phase: 16-submission-form-approval-queue-calendar-projection-write-pat*
*Completed: 2026-07-04*

## Self-Check: PASSED

- FOUND: src/templates/campaigns/approval_queue.html
- FOUND: solsys_code/tests/test_campaign_approval.py
- FOUND: commit 9d9aff6
- FOUND: commit b132b35
