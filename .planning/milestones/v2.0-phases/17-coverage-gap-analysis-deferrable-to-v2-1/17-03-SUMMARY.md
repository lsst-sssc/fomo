---
phase: 17-coverage-gap-analysis-deferrable-to-v2-1
plan: 03
subsystem: ui
tags: [django, templates, crispy-forms, bootstrap4, campaign, coverage-gap]

# Dependency graph
requires:
  - phase: 17-coverage-gap-analysis-deferrable-to-v2-1 (Plan 02)
    provides: CampaignGapAnalysisView, CampaignGapAnalysisForm, gap_analysis_available(campaign) D-14 helper, campaigns:gap_analysis URL
  - phase: 15-per-campaign-table-view-read-path
    provides: campaignrun_table.html header row / Bootstrap 4 conventions this plan extends
provides:
  - "campaignrun_gap_analysis.html: full UI-SPEC page (wait-state notice, Last computed caption, gap-date list / No gaps found empty state, D-08 undated/unattributed Needs review alert, D-03 observability-unknown caveat, IDOR alert-danger, back link)"
  - "campaignrun_table.html: Show Coverage Gaps btn-primary trigger button, disabled with D-14 helper text when unavailable"
  - "TestGapAnalysisButton: D-14 button-gating integration tests at the rendered-template level"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "D-14 button gating: view supplies a boolean availability flag to context; template renders either a live <a> or a disabled button + helper text, never a dead clickable control"
    - "IDOR/400 case is rendered by re-invoking self.render_to_response(context, status=400) with idor_error=True in context, so the same template shows the alert-danger and the still-usable form, instead of a bare HttpResponseBadRequest"

key-files:
  created: []
  modified:
    - src/templates/campaigns/campaignrun_gap_analysis.html
    - src/templates/campaigns/campaignrun_table.html
    - solsys_code/campaign_views.py
    - solsys_code/tests/test_campaign_gap.py

key-decisions:
  - "Deviation (Rule 2 - missing critical functionality): CampaignRunTableView.get_context_data() now supplies gap_analysis_available to context by calling the existing gap_analysis_available(campaign) helper from Plan 02 -- required by this plan's own Task 1 action text (button gating needs the flag from somewhere) and reuses the helper rather than duplicating its target-count/resolved-site logic."
  - "Deviation (Rule 2 - missing critical functionality): CampaignGapAnalysisView's two IDOR branches (out-of-scope target, out-of-scope site) now call self.render_to_response(context, status=400) with idor_error=True in context instead of a bare HttpResponseBadRequest(). This plan's must_haves require 'The IDOR 400 case re-renders the selection form with the error copy, never a raw Django debug/400 page' (UI-SPEC), which is impossible without rendering the template. Verified this does not break Plan 02's test_rejects_out_of_scope_target_and_site (it only asserts status_code == 400 and that no computation ran, both still true) -- full solsys_code suite re-run green at 326/326 after this plan's changes (see below)."

patterns-established:
  - "Disabled-with-explanation over fully-hidden for gated controls whose absence needs to stay legible (D-14), matching UI-SPEC Interaction States -- template renders a disabled btn-primary plus 14px muted helper text rather than omitting the button."

requirements-completed: [GAP-02]

coverage:
  - id: D1
    description: "campaignrun_gap_analysis.html renders the full UI-SPEC contract: title, scoped selection form with wait-state notice, Last computed caption, gap-date list / No gaps found empty state, D-08 Needs-review alert, D-03 observability-unknown caveat (only when count>0), IDOR alert-danger, and Back to Campaign Table link"
    requirement: "GAP-02"
    verification:
      - kind: other
        ref: "./manage.py shell -c \"import django.template.loader as l; l.get_template('campaigns/campaignrun_gap_analysis.html')\" (loads without TemplateSyntaxError)"
        status: pass
      - kind: manual_procedural
        ref: "Human-verify checkpoint (Task 3) -- reviewer visited /campaigns/<pk>/gaps/, confirmed copy/color/spacing/states against UI-SPEC"
        status: pass
    human_judgment: true
    rationale: "Visual/copy fidelity to the UI-SPEC (exact copywriting contract, color discipline, spacing) requires human eyes; the plan scoped this as an explicit checkpoint:human-verify task, now approved."
  - id: D2
    description: "campaignrun_table.html's Show Coverage Gaps button is a live btn-primary link when D-14 availability is true, and a disabled button with explanatory helper text when false"
    requirement: "GAP-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_gap.py#TestGapAnalysisButton.test_button_enabled_when_available"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_gap.py#TestGapAnalysisButton.test_button_disabled_no_targets"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_gap.py#TestGapAnalysisButton.test_button_disabled_no_resolved_site"
        status: pass
    human_judgment: false
  - id: D3
    description: "The IDOR 400 case re-renders the selection form with the alert-danger error copy, never a raw Django debug/400 page"
    requirement: "GAP-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_gap.py#TestGapAnalysisView.test_rejects_out_of_scope_target_and_site (re-verified green after switching to render_to_response(status=400))"
        status: pass
      - kind: manual_procedural
        ref: "Human-verify checkpoint (Task 3) -- covered by the general page review; no separate IDOR-specific manual step was scoped in how-to-verify"
        status: pass
    human_judgment: false

duration: ~15min (active work across two sessions; excludes checkpoint wait time for human review)
completed: 2026-07-04
status: complete
---

# Phase 17 Plan 03: Coverage-Gap Analysis Page + Trigger Button Summary

**Gap-analysis page (`campaignrun_gap_analysis.html`) and D-14-gated "Show Coverage Gaps" button on the campaign table, rendered verbatim to the UI-SPEC copywriting contract, human-verified and approved.**

## Performance

- **Duration:** ~15 min active work (Tasks 1-2 executed in a prior session; this continuation session verified the checkpoint approval, re-ran the full test suite, and finalized the plan)
- **Started:** 2026-07-04T22:04:00Z (approx., Task 1 commit)
- **Completed:** 2026-07-04T22:44:37Z (approx.)
- **Tasks:** 3 completed (2 auto + 1 checkpoint:human-verify)
- **Files modified:** 4 (3 declared in `files_modified` + `solsys_code/campaign_views.py` via Rule 2 deviation)

## Accomplishments
- `campaignrun_gap_analysis.html` fully replaces Plan 02's placeholder with the UI-SPEC contract:
  page title, crispy `CampaignGapAnalysisForm` in a `card bg-light p-3` panel, wait-state help text
  under "Update Results", "Last computed" caption, gap-date list / "No gaps found" empty state, D-08
  "Needs review: undated runs" alert-warning (covering both undated and target-unattributed runs),
  D-03 observability-unknown caveat (shown only when count > 0), IDOR `alert-danger`, and a
  "Back to Campaign Table" `btn-outline-primary` link.
- `campaignrun_table.html`'s header row gained a "Show Coverage Gaps" `btn-primary` button, gated on
  the D-14 `gap_analysis_available` context flag: a live link when available, a `disabled`
  `btn-primary` plus explanatory helper text when not (no campaign targets, or no run with a
  resolved site).
- `TestGapAnalysisButton` (3 new integration tests in `test_campaign_gap.py`) proves the D-14 gating
  at the rendered-HTML level: enabled-link case, zero-targets-disabled case, all-runs-site-None-
  disabled case.
- Human-verify checkpoint (Task 3) presented for visual/interaction review against the UI-SPEC and
  **approved** by the user ("approved", no issues raised).

## Task Commits

Each task was committed atomically:

1. **Task 1: Gap-analysis page template + table-page trigger button (GAP-02, D-08/D-09/D-14, UI-SPEC)** - `e5d870c` (feat)
2. **Task 2: TestGapAnalysisButton -- D-14 gating integration test (GAP-02)** - `a8da62f` (test)
3. **Task 3: Human-verify the gap-analysis page and trigger button** - checkpoint, no code changes; approved by user in this continuation session

**Plan metadata:** this SUMMARY.md's commit (docs: complete plan)

## Files Created/Modified
- `src/templates/campaigns/campaignrun_gap_analysis.html` - Full UI-SPEC gap-analysis page (form, wait notice, last-computed caption, gap list/empty state, D-08/D-03 status blocks, IDOR alert, back link)
- `src/templates/campaigns/campaignrun_table.html` - "Show Coverage Gaps" button, D-14-gated (live link vs. disabled + helper text)
- `solsys_code/campaign_views.py` - `CampaignRunTableView` context now supplies `gap_analysis_available`; `CampaignGapAnalysisView`'s IDOR branches re-render the template with `status=400` instead of a bare `HttpResponseBadRequest` (Rule 2 deviations, see below)
- `solsys_code/tests/test_campaign_gap.py` - Added `TestGapAnalysisButton` (3 tests) proving D-14 gating at the rendered-template level

## Decisions Made

- Reused the existing `gap_analysis_available(campaign)` module-level helper (added in Plan 02) from
  `CampaignRunTableView.get_context_data()` rather than re-deriving the availability logic inline in
  the view or the template -- keeps the target-count/resolved-site rule in exactly one place.
- Both IDOR branches in `CampaignGapAnalysisView.get()` now call `self.render_to_response(context,
  status=400)` with `idor_error=True` in context, rather than the bare `HttpResponseBadRequest()`
  Plan 02 originally used. This was required to satisfy this plan's own must_haves truth ("The IDOR
  400 case re-renders the selection form with the error copy, never a raw Django debug/400 page") --
  a bare `HttpResponseBadRequest` cannot render a template. `HttpResponseBadRequest` remains imported
  in `campaign_views.py` for `CampaignRunDecisionView`'s unrelated POST-action validation, which is
  untouched by this plan.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] `CampaignRunTableView` did not yet supply the D-14 availability flag to context**
- **Found during:** Task 1 (gap-analysis page template + table-page trigger button)
- **Issue:** The plan's own Task 1 action text requires the button to be gated on a D-14
  availability flag "supplied by the view," but `CampaignRunTableView.get_context_data()` did not
  yet add it -- Plan 02 only defined the `gap_analysis_available(campaign)` helper function, it
  never wired it into the table view's context.
- **Fix:** Added `context['gap_analysis_available'] = gap_analysis_available(context['campaign'])`
  to `CampaignRunTableView.get_context_data()`, importing/reusing the Plan 02 helper rather than
  duplicating its target-count/resolved-site logic.
- **Files modified:** `solsys_code/campaign_views.py`
- **Verification:** `TestGapAnalysisButton`'s three cases exercise all branches of this flag at the
  rendered-template level; `./manage.py test solsys_code.tests.test_campaign_gap` green.
- **Committed in:** `e5d870c` (Task 1 commit)

**2. [Rule 2 - Missing Critical Functionality] IDOR 400 responses were a bare `HttpResponseBadRequest`, not a re-rendered form**
- **Found during:** Task 1 (gap-analysis page template + table-page trigger button)
- **Issue:** This plan's must_haves explicitly require: "The IDOR 400 case re-renders the selection
  form with the error copy, never a raw Django debug/400 page (D-08/D-09, UI-SPEC)." Plan 02's
  `CampaignGapAnalysisView` returned a bare `HttpResponseBadRequest()` for both the out-of-scope
  target and out-of-scope site branches, which cannot show the UI-SPEC's `alert-danger` copy or the
  still-usable form.
- **Fix:** Both IDOR branches now set `context['idor_error'] = True` and return
  `self.render_to_response(context, status=400)`, so the same template renders the form plus the
  `alert-danger` block, still with a 400 status code (preserving the "never trust the dropdown"
  server-side contract from T-17-01).
- **Files modified:** `solsys_code/campaign_views.py`, `src/templates/campaigns/campaignrun_gap_analysis.html`
- **Verification:** Re-ran Plan 02's `TestGapAnalysisView.test_rejects_out_of_scope_target_and_site`
  (only asserts `status_code == 400` and that no computation ran -- both still hold) and the full
  `solsys_code.tests.test_campaign_gap` suite (23/23 green) and the full `solsys_code` app suite
  (**326/326 green**, re-verified in this finalization session -- includes this plan's own 3 new
  `TestGapAnalysisButton` tests).
- **Committed in:** `e5d870c` (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 2 - missing critical functionality, both required to
satisfy this plan's own stated must_haves/acceptance criteria, not scope creep).
**Impact on plan:** Both changes were necessary for the plan's declared truths to hold. No
unrelated logic was duplicated or touched; `CampaignRunDecisionView`'s use of
`HttpResponseBadRequest` for its own unrelated POST-action validation is untouched.

## Issues Encountered
None beyond the two deviations documented above.

## User Setup Required
None - no external service configuration required.

## Checkpoint Resolution

Task 3 (`checkpoint:human-verify`, gate="blocking") was presented to the human reviewer with the
gap-analysis page and the D-14-gated trigger button running against a seeded demo campaign
(`pk=7`) in the dev SQLite DB. The human responded **"approved"** with no issues raised. No
follow-up fixes were required; the checkpoint is resolved as-is.

## Verification Re-run (this finalization session)

- `./manage.py test solsys_code.tests.test_campaign_gap -v 1` -- 23/23 tests pass (all classes
  across Plans 01-03, including the 3 new `TestGapAnalysisButton` tests).
- `./manage.py test solsys_code` -- full app suite, **326/326 tests pass**. (This is the accurate
  current total post-Plan-03; it already includes `TestGapAnalysisButton`'s 3 tests.)
- `ruff check solsys_code/campaign_views.py solsys_code/tests/test_campaign_gap.py` -- clean, no
  findings.
- `ruff format --check` on the same two `.py` files -- both already formatted. (Ruff does not
  parse Django HTML templates; the two modified `.html` files are outside ruff's scope and were
  instead verified via Django's template loader, which confirmed no `TemplateSyntaxError`.)
- Both templates load via `django.template.loader.get_template()` without error.

## Next Phase Readiness

GAP-02's full user-facing surface (server-side view from Plan 02, page + trigger button from this
plan) is complete and human-approved. Phase 17 (Coverage-Gap Analysis, deferrable to v2.1) has no
further plans -- this is the last plan in the phase. Per this plan's own `<output>` instructions and
the orchestrator contract, this executor does not mark the phase itself complete; phase-level
verification and completion are handled by the orchestrator after this SUMMARY.md lands.

---
*Phase: 17-coverage-gap-analysis-deferrable-to-v2-1*
*Completed: 2026-07-04*

## Self-Check: PASSED

- FOUND: src/templates/campaigns/campaignrun_gap_analysis.html
- FOUND: src/templates/campaigns/campaignrun_table.html
- FOUND: solsys_code/tests/test_campaign_gap.py
- FOUND: solsys_code/campaign_views.py
- FOUND: .planning/phases/17-coverage-gap-analysis-deferrable-to-v2-1/17-03-SUMMARY.md
- FOUND commit: e5d870c
- FOUND commit: a8da62f
