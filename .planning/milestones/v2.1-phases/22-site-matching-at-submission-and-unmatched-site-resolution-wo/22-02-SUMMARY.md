---
phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
plan: 02
subsystem: api
tags: [django, htmx, crispy-forms, django-tables2, campaigns]

requires:
  - phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo-plan-01
    provides: campaigns:site_search live-search endpoint, site_search_results.html suggestion fragment
provides:
  - CampaignRunSubmissionForm.site_raw live-search widget (hx-get to campaigns:site_search), no create-new-site link (D-09)
  - ApprovalQueueTable.render_site() pending-row live-search widget (hx-get to campaigns:site_search), replacing the static <=5-candidate datalist, keeping the Create-new-Observatory link (D-10)
  - Corrected htmx hx-trigger grammar (input[this.value.length >= 2] changed delay:300ms) applied consistently on both surfaces
affects: [22-site-matching-at-submission-and-unmatched-site-resolution-wo-plan-03]

tech-stack:
  added: []
  patterns:
    - "htmx hx-trigger event-filter placement: the [...] length-gate filter goes IMMEDIATELY after the event name (input[this.value.length >= 2] changed delay:300ms), never after the delay modifier -- 22-REVIEWS.md finding 1"
    - "Three-way click-to-fill id wiring: the input's id, the hx-target/container-div id suffix, and the hx-vals input_id value must all derive from one shared string, proven by a dedicated assertion per surface (22-REVIEWS.md finding 7)"
    - "Widget-attrs path (Django TextInput attrs) HTML-escapes hx-trigger/hx-vals; the format_html-literal path (table cell markup) does not -- test assertions differ accordingly (escaped substrings on the form, raw string on the table)"

key-files:
  created: []
  modified:
    - solsys_code/campaign_forms.py
    - solsys_code/campaign_tables.py
    - solsys_code/tests/test_campaign_submission.py
    - solsys_code/tests/test_campaign_approval.py

key-decisions:
  - "hx-trigger uses the input event (not keyup) so paste, browser autocomplete, and mobile IME input also fire the search -- adopted per 22-REVIEWS.md finding 1's reviewer-suggested variant, deliberately superseding the keyup-based string in 22-UI-SPEC.md/22-RESEARCH.md"
  - "fuzzy_match_candidates import removed from campaign_tables.py -- its sole caller (the datalist branch) was deleted; campaign_utils.py's own definition and test_campaign_approval.py's module-qualified uses are unaffected"
  - "Existing TestApprovalQueueSiteDisambiguationUI class (Plan 21-03's datalist-markup tests) renamed to TestApprovalQueueSiteSearchWidget and rewritten against the new widget markup, since the datalist assertions it made would otherwise fail against the swapped-in live-search widget"
  - "campaign_views.py's candidate_pool construction/passthrough to ApprovalQueueTable left untouched -- out of this plan's files_modified scope; the table no longer reads it in render_site(), a minor unused-computation note for a future cleanup, not a correctness issue"

requirements-completed: [D-04, D-05, D-06, D-09, D-10]

coverage:
  - id: D1
    description: "Public submission form's site_raw field renders an hx-get widget (campaigns:site_search) with the corrected hx-trigger grammar and a suggestions container, with NO Create-new-Observatory link (D-09)"
    requirement: "D-09"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestSubmissionFormSiteSearchWidget.test_form_renders_hx_get_and_corrected_trigger_grammar"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestSubmissionFormSiteSearchWidget.test_form_renders_suggestions_container"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestSubmissionFormSiteSearchWidget.test_form_has_no_create_new_observatory_link"
        status: pass
    human_judgment: false
  - id: D2
    description: "Public form's click-to-fill wiring: id_site_raw is simultaneously the input id, the hx-target/container suffix, and the hx-vals input_id value (22-REVIEWS.md finding 7)"
    requirement: "D-09"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestSubmissionFormSiteSearchWidget.test_click_to_fill_wiring_uses_one_consistent_id"
        status: pass
    human_judgment: false
  - id: D3
    description: "Approval-queue pending row's render_site() renders the live-search widget (hx-get to campaigns:site_search, corrected trigger grammar), no <datalist> element, and keeps the Create-new-Observatory link (D-10)"
    requirement: "D-10"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueSiteSearchWidget.test_unresolved_pending_row_renders_live_search_widget_and_create_link"
        status: pass
    human_judgment: false
  - id: D4
    description: "Approval-queue click-to-fill wiring: site-input-{pk} is simultaneously the input id, the hx-target value, the container div id, and the hx-vals input_id value"
    requirement: "D-10"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueSiteSearchWidget.test_click_to_fill_wiring_uses_one_consistent_id"
        status: pass
    human_judgment: false
  - id: D5
    description: "Resolved rows and the read-only decided table keep the plain-text render_site fallback unchanged; fuzzy_match_candidates import removed from campaign_tables.py (F401-clean)"
    requirement: "D-04/D-05/D-06"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueSiteSearchWidget.test_resolved_pending_row_renders_no_site_selection_input"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_approval.py#TestApprovalQueueSiteSearchWidget.test_decided_table_renders_no_site_selection_input"
        status: pass
      - kind: unit
        ref: "grep -c 'fuzzy_match_candidates' solsys_code/campaign_tables.py -- 0"
        status: pass
    human_judgment: false

duration: 15min
completed: 2026-07-15
status: complete
---

# Phase 22 Plan 02: Site-Entry Widget Wiring (Submission Form + Approval Queue) Summary

**Wired Plan 01's `campaigns:site_search` live-search endpoint into both site-entry surfaces: the public submission form's `site_raw` field (D-09, no create-new link) and the approval-queue pending row's inline site input (D-10, replacing the static datalist while keeping the Create-new-Observatory escape hatch), using the htmx-grammar-corrected `input[this.value.length >= 2] changed delay:300ms` trigger consistently on both.**

## Performance

- **Duration:** ~15 min
- **Tasks:** 2 completed (both standard, non-TDD)
- **Files modified:** 4

## Accomplishments
- `CampaignRunSubmissionForm.site_raw` re-declared with a `forms.TextInput` widget carrying `hx-get`/`hx-trigger`/`hx-target`/`hx-swap`/`hx-vals`/`autocomplete`/`placeholder`/`class` attrs, resolved via `reverse_lazy` (Pitfall 6) so importing the form module never raises `NoReverseMatch`.
- A crispy `HTML('<div id="site-suggestions-id_site_raw" class="mt-2"></div>')` container inserted directly below the `site_raw` field in the "Run details" fieldset, pushing later fields down in normal document flow.
- `ApprovalQueueTable.render_site()`'s pending-row branch rewritten: the static `fuzzy_match_candidates()`-backed `<datalist>` is gone, replaced by a live `hx-get="campaigns:site_search"` widget with its own suggestions `<div>`, still submitting into the row's `decide-form-{pk}` via the HTML5 `form=` attribute and still followed by the unchanged "Create new Observatory" link.
- Both widgets use the identical corrected htmx trigger grammar (`input[this.value.length >= 2] changed delay:300ms` -- the `[...]` filter directly after the event name, per 22-REVIEWS.md finding 1), with a source comment on the form field warning against regressing the ordering.
- `fuzzy_match_candidates` import removed from `campaign_tables.py` (its sole caller there was the deleted datalist branch); `format_html_join` import also removed as it became unused.
- Every markup test asserts the three-way click-to-fill id wiring (input id = hx-target/container suffix = hx-vals `input_id` value) per surface, per 22-REVIEWS.md finding 7.

## Task Commits

Each task was committed atomically:

1. **Task 1: Public submission form site_raw live-search widget (D-09)** - `907d459` (feat)
2. **Task 2: Approval-queue pending-row render_site() HTMX widget swap (D-10)** - `8bedff2` (feat)

**Plan metadata:** (this commit, pending)

## Files Created/Modified
- `solsys_code/campaign_forms.py` - `site_raw` widget with hx-* attrs, `reverse_lazy` + `HTML` imports, crispy suggestions container
- `solsys_code/campaign_tables.py` - `ApprovalQueueTable.render_site()` pending-row branch rewritten as the live-search widget; `fuzzy_match_candidates`/`format_html_join` imports dropped
- `solsys_code/tests/test_campaign_submission.py` - `TestSubmissionFormSiteSearchWidget` (4 tests: hx-get + corrected trigger, suggestions container, no create-link, click-to-fill wiring)
- `solsys_code/tests/test_campaign_approval.py` - `TestApprovalQueueSiteDisambiguationUI` renamed/rewritten to `TestApprovalQueueSiteSearchWidget` (6 tests: widget + create-link + no-datalist, click-to-fill wiring, XSS escaping retained, resolved-row/decided-table fallback retained)

## Decisions Made
- Adopted the `input` event (not `keyup`) for both widgets' `hx-trigger`, per 22-REVIEWS.md finding 1's reviewer-suggested variant -- this also fires on paste/autocomplete/IME input, a real staff flow (pasting an MPC code into the queue).
- Rewrote the pre-existing `TestApprovalQueueSiteDisambiguationUI` class (from Plan 21-03) rather than leaving it to fail: its assertions targeted the now-deleted `<datalist>`/`list=`/`<option>` markup, so they had to change in lockstep with the render_site() rewrite. Kept the stored-XSS and resolved/decided fallback tests structurally intact, only updating the markup they check for.
- Left `campaign_views.py`'s `build_site_candidates()`/`candidate_pool` construction and pass-through to `ApprovalQueueTable` untouched -- out of this plan's `files_modified` scope. The table constructor still accepts `candidate_pool` but `render_site()` no longer reads it; a minor unused-computation note, not a functional issue, and not touched per plan scope.

## Deviations from Plan

None - plan executed exactly as written, including the exact hx-trigger string, widget attrs, and id-wiring contract specified in the plan text (which itself supersedes 22-UI-SPEC.md/22-RESEARCH.md per the review disposition documented in the plan).

## Issues Encountered

Ruff's import-sort (`I001`) reordered `from crispy_forms.layout import Div, Fieldset, HTML, Layout, Submit` to `HTML, Div, Fieldset, Layout, Submit` (case-sensitive isort ordering places the all-caps `HTML` before `Div`) -- a cosmetic auto-fix applied by `ruff check --fix`, no functional impact.

## User Setup Required
None - no external service configuration required. No new packages installed.

## Next Phase Readiness
- Both public-facing and staff-facing site-entry surfaces now consume the shared `campaigns:site_search` endpoint from Plan 01 live-search on typed text different from the original `site_raw`.
- Plan 03 (sites-needing-review row) can reuse the same corrected `hx-trigger` grammar and the three-way id-wiring pattern established here.
- Full `solsys_code` test suite (458 tests) passes with no regressions; `ruff check`/`ruff format --check` clean on all 4 modified files.
- No blockers for Plan 03.

---
*Phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo*
*Completed: 2026-07-15*

## Self-Check: PASSED

All 4 modified files verified present on disk; both task commit hashes
(`907d459`, `8bedff2`) verified present in git history.
