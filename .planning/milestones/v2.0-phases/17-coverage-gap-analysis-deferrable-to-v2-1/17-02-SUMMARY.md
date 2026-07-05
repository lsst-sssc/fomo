---
phase: 17-coverage-gap-analysis-deferrable-to-v2-1
plan: 02
subsystem: api
tags: [django, forms, views, idor, campaign, coverage-gap]

# Dependency graph
requires:
  - phase: 17-coverage-gap-analysis-deferrable-to-v2-1 (Plan 01)
    provides: campaign_gap.py's get_or_compute_gap/clamp_date_range/build_gap_cache_key pure-logic core
  - phase: 14-campaign-data-model-bootstrap-import
    provides: CampaignRun model (approval_status/run_status, site FK, target FK)
  - phase: 15-per-campaign-table-view-read-path
    provides: CampaignRunTableView / campaign_urls.py conventions this plan extends
provides:
  - "CampaignGapAnalysisForm: campaign-scoped target/site/end_date selection form (D-12/D-13), GET-submitted"
  - "campaigns:gap_analysis URL (<int:pk>/gaps/)"
  - "CampaignGapAnalysisView: GET-triggered, cached, server-side IDOR-validated coverage-gap endpoint (GAP-02)"
  - "gap_analysis_available(campaign) D-14 gating helper for Plan 03's button"
  - "TestGapAnalysisView integration tests: no-inline-compute, cache-hit skip, IDOR rejection, single-target auto-select"
affects: [17-03-campaign-gap-analysis-template]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Campaign-scoped ModelChoiceField querysets built in Form.__init__(campaign=...), never class-level unscoped querysets (D-12/D-13)"
    - "View re-derives allowed target/site sets server-side and validates submitted pks against them before any query/cache-key use -- HttpResponseBadRequest() on mismatch, mirroring CampaignRunDecisionView (T-17-01/Pitfall 3)"
    - "D-09: gap computation lives entirely behind a separate GET-triggered view/URL, never inline in CampaignRunTableView"

key-files:
  created:
    - src/templates/campaigns/campaignrun_gap_analysis.html
  modified:
    - solsys_code/campaign_forms.py
    - solsys_code/campaign_views.py
    - solsys_code/campaign_urls.py
    - solsys_code/tests/test_campaign_gap.py

key-decisions:
  - "Deviation (Rule 2): added a minimal placeholder campaignrun_gap_analysis.html template (not in this plan's files_modified) because CampaignGapAnalysisView is a TemplateView and this plan's own integration tests (Django test Client GETs) require the view to render end-to-end -- Plan 03 fully replaces its content per UI-SPEC."
  - "Target/site membership validation is done via raw request.GET.get() + explicit queryset .filter(pk=...).exists() checks in the view, not solely via the form's ModelChoiceField validation -- keeps the 400 Bad Request semantics required by T-17-01 independent of the form's own (200-with-errors) validation behavior."
  - "Order of validation in the view is target first, then site -- matches CampaignRunDecisionView's single-condition-at-a-time HttpResponseBadRequest shape and lets IDOR tests exercise each branch independently."

patterns-established:
  - "CampaignGapAnalysisView's docstring explicitly states it imports campaign_gap (safe: only depends on telescope_runs.sun_event) but never solsys_code.views/ephem_utils, extending campaign_views.py's existing module-docstring discipline."

requirements-completed: [GAP-02]

coverage:
  - id: D1
    description: "CampaignGapAnalysisForm scopes target/site ModelChoiceField querysets to the campaign at __init__ time; target is optional for single-target campaigns, required for multi-target (D-12/D-13)"
    requirement: "GAP-02"
    verification:
      - kind: other
        ref: "./manage.py shell -c \"from django.urls import reverse; print(reverse('campaigns:gap_analysis', kwargs={'pk': 1}))\" (resolves) + ruff check"
        status: pass
    human_judgment: false
  - id: D2
    description: "The per-campaign table view GET never triggers gap computation (D-09)"
    requirement: "GAP-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_gap.py#TestGapAnalysisView.test_table_view_does_not_trigger_computation"
        status: pass
    human_judgment: false
  - id: D3
    description: "A cache hit serves the stored result with its original computed_at, skipping recomputation (D-10)"
    requirement: "GAP-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_gap.py#TestGapAnalysisView.test_cache_hit_skips_recomputation"
        status: pass
    human_judgment: false
  - id: D4
    description: "A submitted target_pk/site_pk outside the campaign's scope is rejected with 400, never queried (T-17-01/Pitfall 3)"
    requirement: "GAP-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_gap.py#TestGapAnalysisView.test_rejects_out_of_scope_target_and_site"
        status: pass
    human_judgment: false
  - id: D5
    description: "A single-target campaign auto-selects its sole target when no target_pk is submitted (D-12)"
    requirement: "GAP-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_gap.py#TestGapAnalysisView.test_single_target_autoselects"
        status: pass
    human_judgment: false
  - id: D6
    description: "campaign_views.py still imports no heavy ephemeris/views module at module scope"
    requirement: "GAP-01"
    verification:
      - kind: other
        ref: "grep -rnE forbidden-import-pattern solsys_code/campaign_views.py (zero matches)"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-07-04
status: complete
---

# Phase 17 Plan 02: Coverage-Gap Analysis View (Read Path Wiring) Summary

**`CampaignGapAnalysisView` wires Plan 01's `get_or_compute_gap` into a public, GET-triggered, cached endpoint with a campaign-scoped selection form and server-side IDOR re-validation of target/site pks.**

## Performance

- **Duration:** 25 min (approx.)
- **Started:** 2026-07-04T21:25:00Z (approx.)
- **Completed:** 2026-07-04T21:49:17Z
- **Tasks:** 3 completed
- **Files modified:** 4 (3 modified, 1 new)

## Accomplishments
- `CampaignGapAnalysisForm` (`campaign_forms.py`): campaign-scoped `target`/`site` `ModelChoiceField`s
  populated in `__init__(campaign=...)`, `target` required only for multi-target campaigns (D-12), an
  optional `end_date`, GET-method crispy layout with an "Update Results" submit button.
- `campaigns:gap_analysis` URL (`<int:pk>/gaps/`) added to `campaign_urls.py`.
- `CampaignGapAnalysisView` (`campaign_views.py`): resolves campaign, gates on the new
  `gap_analysis_available(campaign)` D-14 helper, resolves + re-validates target/site pks
  server-side (400 on mismatch, T-17-01), clamps the date range via Plan 01's
  `clamp_date_range` (D-11), and calls `get_or_compute_gap` (D-10) only once both target and
  site are validated.
- `TestGapAnalysisView`: 4 new integration tests proving no-inline-compute on the table view
  (D-09), cache-hit skipping with a stable `computed_at` (D-10), 400 rejection of an
  out-of-scope target *and* an out-of-scope site with zero computation calls (T-17-01/Pitfall
  3), and single-target auto-selection (D-12).
- A minimal placeholder `campaignrun_gap_analysis.html` template so the view renders
  end-to-end for this plan's own tests (see Deviations) -- Plan 03 replaces its content with
  the full UI-SPEC page.

## Task Commits

Each task was committed atomically:

1. **Task 1: CampaignGapAnalysisForm (campaign-scoped) + gap_analysis URL (D-11/D-12/D-13)** - `c2bfaff` (feat)
2. **Task 2: CampaignGapAnalysisView with server-side IDOR + clamp + cache (GAP-02, T-17-01)** - `f1a06b6` (feat)
3. **Task 3: Integration tests -- no-inline-compute, cache-hit, IDOR rejection** - `632d356` (test)

**Plan metadata:** _(this commit, docs: complete plan)_

## Files Created/Modified
- `solsys_code/campaign_forms.py` - `CampaignGapAnalysisForm`: campaign-scoped target/site querysets, GET-method crispy layout
- `solsys_code/campaign_urls.py` - `campaigns:gap_analysis` route added
- `solsys_code/campaign_views.py` - `gap_analysis_available()` helper + `CampaignGapAnalysisView` (IDOR-validated, clamped, cached)
- `src/templates/campaigns/campaignrun_gap_analysis.html` - minimal placeholder template (Rule 2 deviation; Plan 03 replaces content)
- `solsys_code/tests/test_campaign_gap.py` - `TestGapAnalysisView` (4 integration tests)

## Decisions Made
- Target/site IDOR validation is implemented as explicit `request.GET.get()` + `.filter(pk=...).exists()`
  checks in the view (not solely relying on the form's `ModelChoiceField` validation), so an
  out-of-scope submission always gets a `400 Bad Request` rather than a `200` re-render with a
  form error -- matches T-17-01's required semantics and `CampaignRunDecisionView`'s existing
  `HttpResponseBadRequest()` precedent.
- Validation order is target first, then site, so the two IDOR test cases (out-of-scope target
  with a valid site; valid target with an out-of-scope site) each exercise a distinct branch.
- `gap_analysis_available(campaign)` is a plain module-level function (not a classmethod) so
  Plan 03's `CampaignRunTableView` can import and reuse it directly for the button-gating logic
  without instantiating `CampaignGapAnalysisView`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] Added a minimal placeholder gap-analysis template**
- **Found during:** Task 2/3 (view implementation + integration tests)
- **Issue:** `CampaignGapAnalysisView` is a `TemplateView` pointing at
  `campaigns/campaignrun_gap_analysis.html`, which per this plan's own notes and 17-03-PLAN.md
  is only fully built in Plan 03. Without *any* template file, every `self.render_to_response(...)`
  call in this plan's own `TestGapAnalysisView` tests (which use the real Django test Client,
  per the plan's Task 3 instructions) would raise `TemplateDoesNotExist` and fail immediately --
  the view could not be proven to work end-to-end as the plan's acceptance criteria require.
- **Fix:** Added a minimal placeholder template (page title, availability gate, crispy form,
  gap-date list, "Back to Campaign Table" link) sufficient for the view to render successfully.
  It is explicitly commented as a Plan-02-only placeholder that Plan 03 will fully replace to
  match the UI-SPEC contract (empty state, D-08/D-03 caveats, IDOR alert, wait-state notice).
- **Files modified:** `src/templates/campaigns/campaignrun_gap_analysis.html` (new)
- **Verification:** `TestGapAnalysisView` (4/4) passes; full `./manage.py test solsys_code` (323/323) passes.
- **Committed in:** `f1a06b6` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical functionality)
**Impact on plan:** Necessary for this plan's own integration tests to exercise the view
end-to-end; no scope creep into Plan 03's UI-SPEC work (the placeholder is explicitly minimal
and documented as such for the next plan to replace).

## Issues Encountered

None. `ruff check`/`ruff format --check` clean on all four modified/created source files (the
new template is not ruff-scoped). Full `./manage.py test solsys_code` suite (323 tests, up from
319 pre-plan) passes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

`CampaignGapAnalysisView`, `CampaignGapAnalysisForm`, `gap_analysis_available()`, and the
`campaigns:gap_analysis` URL are all in place and tested. Plan 03 can now:
- Replace `campaignrun_gap_analysis.html` with the full UI-SPEC page (empty state, D-08 undated/
  unattributed-runs alert, D-03 unknown-date caveat, IDOR alert-danger, wait-state notice).
- Add the "Show Coverage Gaps" button to `campaignrun_table.html`, gated via the already-built
  `gap_analysis_available(campaign)` helper (import from `solsys_code.campaign_views`).
- Add `TestGapAnalysisButton` per 17-VALIDATION.md.

No blockers. Note the context keys already supplied by the view for Plan 03's template to
consume: `campaign`, `form`, `gap_analysis_available`, and (once target+site are resolved)
`target`, `site`, `start`, `end`, `result` (a dict with `gap_dates`, `claimed_dates`,
`observable_dates`, `undated_runs`, `unattributed_runs`, `unknown_date_count`, `computed_at`).

---
*Phase: 17-coverage-gap-analysis-deferrable-to-v2-1*
*Completed: 2026-07-04*

## Self-Check: PASSED

All 5 created/modified files verified present on disk; all 3 task commits (`c2bfaff`, `f1a06b6`,
`632d356`) verified present in git log.
