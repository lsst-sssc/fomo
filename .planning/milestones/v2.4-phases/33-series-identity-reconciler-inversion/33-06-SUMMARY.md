---
phase: 33-series-identity-reconciler-inversion
plan: 06
subsystem: ui
tags: [django-templates, campaign-attribution, calendar, accessibility, xss-escaping, gap-closure]

# Dependency graph
requires:
  - phase: 33-series-identity-reconciler-inversion
    provides: "campaign_decoration() template tag, CampaignRunTable row-id anchors, and the D-09/D-10/D-13 visibility/PII gates from plans 33-01/33-02"
provides:
  - "CR-01 fixed: the D-13 tr:target row-highlight rule is actually served to a browser, for staff and anonymous readers, for a campaign with runs and for one with zero runs"
  - "WR-08 pinned as a tested, documented limitation: a run sorting past page 1 has no id=\"run-{pk}\" anchor on the unpaginated page"
  - "One campaign-chip template partial (campaign_chip.html) with role=\"img\" plus a title/aria-label accessible name, included from both month-grid loops"
  - "campaign_decoration()'s visibility rule is now the only gate on event_form.html's attributed-run block"
  - "Three WR-05 month-view tests rewritten to fail for the reason they claim, plus the ANNOT-02 campaign-name escaping edge"
affects: ["33-07", "33-08"]

actuals:
  tokens: 5700
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Django's ExtendsNode discards top-level template nodes in a child template that {% extends %} -- CSS/JS meant to render must live inside a declared parent block (additional_css here)"
    - "A single included partial (campaign_chip.html) as the one definition for markup rendered from two call sites, gated once on the tag's own return value rather than duplicated per call site"
    - "table_url is None used as an existing, already-computed campaign-presence signal inside a template partial, avoiding a new key on a documented 'exactly these keys' tag contract"

key-files:
  created:
    - src/templates/tom_calendar/partials/campaign_chip.html
  modified:
    - src/templates/campaigns/campaignrun_table.html
    - src/templates/tom_calendar/partials/calendar.html
    - src/templates/tom_calendar/partials/event_form.html
    - solsys_code/templatetags/calendar_display_extras.py
    - solsys_code/tests/test_campaign_views.py
    - solsys_code/tests/test_calendar_template.py

key-decisions:
  - "CR-01 root cause was Django's ExtendsNode silently discarding a top-level <style> node in a child template -- fixed by moving it inside tom_common/base.html's empty additional_css block rather than into block content, keeping a document-wide rule out of the per-view content region."
  - "WR-08's positional-page-resolution fix is explicitly deferred (would add a per-event ordered query, contradicting plan 33-02's no-per-event-query must-have) -- pinned instead as a tested, documented limitation."
  - "table_url is None (already computed by campaign_decoration() only when a campaign exists) is reused as the chip's campaign-presence signal rather than adding a new key to the tag's return contract."

requirements-completed: [ANNOT-02]

coverage:
  - id: D1
    description: "The D-13 tr:target highlight rule is actually rendered to staff and anonymous readers, including for an empty-run campaign (CR-01)"
    requirement: "ANNOT-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunRowAnchor.test_staff_get_contains_tr_target_highlight_rule"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunRowAnchor.test_anonymous_get_contains_tr_target_highlight_rule"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunRowAnchor.test_empty_campaign_still_serves_tr_target_highlight_rule"
        status: pass
    human_judgment: false
  - id: D2
    description: "WR-08's page-1-only #run-{pk} anchor gap is pinned as a known, tested constraint"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunAnchorPagination.test_oldest_run_has_no_anchor_on_page_1"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunAnchorPagination.test_oldest_run_anchor_present_on_page_2"
        status: pass
    human_judgment: false
  - id: D3
    description: "One campaign chip definition with role=img and an accessible name (title + aria-label), distinct for a no-campaign run (WR-07/IN-01/IN-03/IN-05)"
    requirement: "ANNOT-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#DecorationSurvivalAndGuardsTest.test_no_campaign_run_renders_marker_and_no_table_href"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#MonthCellCampaignMarkerTest.test_month_view_shows_campaign_chip_and_name_tooltip"
        status: pass
    human_judgment: false
  - id: D4
    description: "event_form.html's attributed-run block is gated only by campaign_decoration() -- no second is_publicly_visible gate in the template"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#DecorationSurvivalAndGuardsTest.test_pending_review_run_shows_no_marker_for_staff_and_anonymous"
        status: pass
    human_judgment: false
  - id: D5
    description: "Three WR-05 tests rewritten so each fails when its own named subject regresses -- verified against real mutations, not just re-reading the assertions"
    verification:
      - kind: other
        ref: "scripted mutation proof in 33-06-PLAN.md Task 3 verify block -- neutralised is_publicly_visible guard, confirmed test_pending_review_run_shows_no_marker_for_staff_and_anonymous went red, then restored the file byte-for-byte"
        status: pass
    human_judgment: false
  - id: D6
    description: "ANNOT-02 campaign-name encoding edge: & < \" escape identically in the chip's title and aria-label"
    requirement: "ANNOT-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#DecorationSurvivalAndGuardsTest.test_campaign_name_encoding_edge_escapes_consistently_in_title_and_aria_label"
        status: pass
    human_judgment: false

duration: 32min
completed: 2026-09-06
status: complete
---

# Phase 33 Plan 06: Chip Accessibility, Visibility-Gate Consolidation, and the D-13 Highlight Fix Summary

**Moved a silently-discarded `<style>` block into Django's `additional_css` block so the D-13 row-highlight rule is actually served, gave the campaign chip a single accessible-name definition, and rewrote three month-view tests that previously could not fail for the reason they claimed.**

## Performance

- **Duration:** 32 min
- **Started:** 2026-09-05T23:40:00Z
- **Completed:** 2026-09-06T00:12:05Z
- **Tasks:** 3
- **Files modified:** 6 modified, 1 created

## Accomplishments

- **CR-01 closed:** the `tr:target` highlight rule was sitting as a top-level node in a template that `{% extends %}` — Django's `ExtendsNode` discards those silently, so the rule had never actually reached a browser. Moved it into `{% block additional_css %}`, the empty CSS hook `tom_common/base.html` declares, and added staff/anonymous/empty-campaign assertions proving the render (not just the source) contains the rule.
- **WR-08 pinned:** added `TestCampaignRunAnchorPagination`, a 26-run fixture that pins the known limitation that the calendar decoration's `#run-{pk}` link carries no page parameter, so a run sorting past page 1 has no anchor in the unpaginated document. Documented as a deliberate scope boundary (fixing it would add a per-event ordered query, breaking plan 33-02's no-per-event-query guard), not a silent dead link.
- **One chip definition:** extracted `campaign_chip.html`, included from both the all-day and timed month-grid loops. Carries `role="img"` and a `title`/`aria-label` pair — a campaign-bearing run gets `Campaign: {name}`, a no-campaign run gets a distinct `Attributed run #{pk} (no campaign)` naming the run itself (WR-07/IN-01/IN-03/IN-05).
- **One visibility gate:** removed the outer `{% with run=... %}` / `{% if run.is_publicly_visible %}` wrapper from `event_form.html`; `campaign_decoration()`'s own `run is None or not run.is_publicly_visible` rule is now the only gate on the attributed-run block.
- **WR-05 tests actually test something:** rewrote the pending-review test to discriminate on a campaign name only that fixture can produce (verified by a scripted mutation proof that neutralises the visibility gate and confirms the test goes red), sized the truncation-budget fixture titles to exactly the filter's own character budget, and retargeted the no-campaign test onto the chip's own tooltip string. Added the ANNOT-02 encoding edge test proving `&`, `<`, and `"` escape identically in both `title` and `aria-label`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Render the D-13 row-highlight rule end-to-end, and pin the paginated-anchor constraint** - `de9d184` (fix)
2. **Task 2: One chip definition with an accessible name, one visibility gate** - `d4c044f` (fix)
3. **Task 3: Make the month-view tests fail for the reason they claim** - `1a578f4` (test)

## Files Created/Modified

- `src/templates/campaigns/campaignrun_table.html` - moved the `tr:target` `<style>` into `{% block additional_css %}` so it actually renders
- `src/templates/tom_calendar/partials/campaign_chip.html` - new partial: the single campaign-chip definition with `role="img"` and title/aria-label
- `src/templates/tom_calendar/partials/calendar.html` - both inline chip spans replaced with `{% include campaign_chip.html %}`
- `src/templates/tom_calendar/partials/event_form.html` - removed the outer `run.is_publicly_visible` gate; `campaign_decoration()`'s own gate is now the only one
- `solsys_code/templatetags/calendar_display_extras.py` - docstring-only: documented `run_pk`/`table_url`'s rendered consumers
- `solsys_code/tests/test_campaign_views.py` - added tr:target rendering assertions and `TestCampaignRunAnchorPagination`
- `solsys_code/tests/test_calendar_template.py` - rewrote the three WR-05 assertions and added the campaign-name encoding edge test

## Decisions Made

- CR-01's root cause (a top-level template node silently discarded by `ExtendsNode`) is fixed by using the base template's existing, empty `additional_css` block rather than moving the rule into `block content`, keeping a document-wide CSS rule out of the per-view content region.
- WR-08's positional-page-resolution fix is deliberately deferred (recorded in the plan's "Explicitly deferred review findings"): computing a run's page inside `campaign_decoration()` would add a per-event ordered query, directly contradicting plan 33-02's must-have that guards against that N+1 pattern. The remaining half is pinned as a tested, documented limitation instead.
- `table_url is None` is reused as the chip's campaign-presence signal (it is only set by `campaign_decoration()` when `run.campaign_id is not None`), avoiding a new key on the tag's documented "exactly these keys" return contract.

## Deviations from Plan

None - plan executed exactly as written. Both scripted mutation proofs (Task 1's `additional_css` rename, Task 3's visibility-guard neutralisation) passed on the first attempt with no debugging required.

## Issues Encountered

One authoring slip caught and fixed inline before any commit: the first draft of Task 1's explanatory comment included the literal string `{% extends %}` inside an HTML comment, which Django's template parser tried to parse as a real tag, breaking template compilation. Rewrote the comment to describe the mechanism in prose instead of quoting Django tag syntax. Not logged as a deviation under the Rule 1-4 framework since it was corrected before any verification run or commit — no plan behavior changed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- CR-01, WR-05, WR-07, IN-01, IN-03 and IN-05 are closed; WR-08 is pinned by a test with the remaining positional-resolution half explicitly deferred and recorded in the plan.
- Plan 33-07 (wave 2, `depends_on: [33-06]`) can now proceed — this plan's two scripted mutation proofs, which required exclusive working-tree access, are complete and both source files were restored byte-for-byte.
- Plan 33-08 owns the operator-facing runbook prose for WR-06/WR-08; this plan wrote no documentation prose, only the pinning test that 33-08's runbook paragraph will refer to.

---
*Phase: 33-series-identity-reconciler-inversion*
*Completed: 2026-09-06*

## Self-Check: PASSED

- FOUND: src/templates/campaigns/campaignrun_table.html
- FOUND: src/templates/tom_calendar/partials/campaign_chip.html
- FOUND: src/templates/tom_calendar/partials/calendar.html
- FOUND: src/templates/tom_calendar/partials/event_form.html
- FOUND: solsys_code/templatetags/calendar_display_extras.py
- FOUND: solsys_code/tests/test_campaign_views.py
- FOUND: solsys_code/tests/test_calendar_template.py
- FOUND commit de9d184 (Task 1)
- FOUND commit d4c044f (Task 2)
- FOUND commit 1a578f4 (Task 3)
- `python manage.py test solsys_code.tests.test_campaign_views solsys_code.tests.test_calendar_template solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_null_campaign_guards` — 167 tests, OK
- Full project test suite (986 tests) — OK, no failures (the previously-known `data.minorplanetcenter.net` network flake did not occur this run)
- `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` — both Passed
