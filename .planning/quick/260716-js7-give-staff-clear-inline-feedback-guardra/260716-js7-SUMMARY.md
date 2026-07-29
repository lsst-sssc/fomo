---
phase: quick-260716-js7
plan: 01
subsystem: campaign-approval-queue
tags: [ui, approval-queue, site-search, client-side-guard]
dependency-graph:
  requires: [Phase 22 site-search widget (_render_site_search_widget, site_search_results.html)]
  provides: [confirm-before-approve guard for unresolved site rows]
  affects: [ApprovalQueueTable rendering, site_search_results.html onclick]
tech-stack:
  added: []
  patterns:
    - "data-* attribute (data-site-resolved) tracked client-side via dataset.siteResolved, mirroring the existing hx-vals/hx-trigger static-literal format_html convention"
    - "onclick confirm-guard on a submit button, mirroring the existing Reject-button confirm() pattern"
key-files:
  created: []
  modified:
    - solsys_code/campaign_tables.py
    - src/templates/campaigns/partials/site_search_results.html
    - solsys_code/tests/test_campaign_approval.py
decisions:
  - "Known-resolved state tracked via a data-site-resolved attribute on the site_selection input (Claude's discretion per CONTEXT.md), set to 'true' only by the suggestion fragment's onclick, cleared to 'false' by an oninput handler on manual typing."
  - "Approve button's confirm-guard onclick short-circuits to return true when the paired input is missing, blank, or already known-resolved -- confirm() only fires for a non-blank, not-known-resolved value."
  - "No treatment added to the resolve-mode Resolve button (out of scope per CONTEXT.md Claude's Discretion)."
metrics:
  duration: ~20min
  completed: 2026-07-16
status: complete
---

# Quick Task 260716-js7: Confirm-before-approve guard for unresolved site rows Summary

Added a client-side confirm() nudge before staff can Approve a Pending Review row whose paired
site-search input hasn't been resolved via an actual dropdown-suggestion click, mirroring the
existing Reject-button confirmation pattern -- no server-side behavior changed (D-06 preserved).

## What Was Built

**Task 1 (commit `ee56256`):** Three coordinated markup edits, no backend/resolution-logic
changes:
1. `_render_site_search_widget()` in `solsys_code/campaign_tables.py` now emits
   `data-site-resolved="false"` plus `oninput="this.dataset.siteResolved = 'false';"` on the
   `site_selection` input -- any manual keystroke/paste clears the known-resolved flag.
2. `site_search_results.html`'s suggestion `onclick` now also sets
   `inputEl.dataset.siteResolved = 'true';` right after filling the combined display/obscode
   value -- since that fill is a programmatic `.value =` assignment (no native `input` event),
   the flag survives a suggestion click.
3. `render_actions()`'s pending-mode Approve button (only -- Reject and the resolve-mode
   Resolve button untouched) carries a new `onclick` guard: it looks up
   `document.getElementById('site-input-{pk}')`, and allows the submit (`return true`) when the
   element is missing, blank, or already known-resolved; otherwise it pops
   `confirm('This observing site does not look resolved yet. Approve anyway? It will land in
   Sites Needing Review.')` and returns that result.

**Task 2 (commit `24d1d94`):** Three new tests added to
`TestApprovalQueueSiteSearchWidget` in `solsys_code/tests/test_campaign_approval.py`:
- `test_pending_unresolved_row_renders_confirm_guard_and_flag` -- asserts the
  `data-site-resolved="false"` attribute, the `oninput` clear handler, the guard's
  `getElementById('site-input-{pk}')` lookup, and the confirm message all appear for an
  unresolved pending row.
- `test_decided_row_has_no_guard_and_no_flag` -- asserts neither `data-site-resolved` nor the
  confirm message appears for a read-only decided (APPROVED) row.
- `test_suggestion_fragment_sets_known_resolved_flag` -- renders
  `site_search_results.html` directly via `render_to_string` and asserts the rendered onclick
  sets `inputEl.dataset.siteResolved = 'true';`.

## Verification

- `python manage.py test solsys_code.tests.test_campaign_approval.TestApprovalQueueSiteSearchWidget`
  -- 8/8 pass (5 pre-existing + 3 new).
- `python manage.py test solsys_code` -- full suite, 518/518 pass, no regressions.
- `ruff check` / `ruff format --check` clean on both touched `.py` files
  (`solsys_code/campaign_tables.py`, `solsys_code/tests/test_campaign_approval.py`).
- Repo-wide `ruff check .` / `ruff format --check .` reports 5 pre-existing findings, all in
  `docs/notebooks/pre_executed/*.ipynb` files unrelated to and untouched by this task (out of
  scope per SCOPE BOUNDARY -- not fixed here).

## Deviations from Plan

None - plan executed exactly as written. `ruff format` reformatted the quote style inside the
new `render_actions()` onclick string (mixed single/double quotes) automatically as part of the
Task 1 verification step; no functional change.

## Known Stubs

None.

## Threat Flags

None -- all three threat-register entries (T-js7-01/02/03) were disposed as `mitigate`/`accept`
in the plan and no new surface was introduced beyond what the plan anticipated.

## Self-Check: PASSED

- FOUND: solsys_code/campaign_tables.py (data-site-resolved, oninput, and onclick guard present)
- FOUND: src/templates/campaigns/partials/site_search_results.html (dataset.siteResolved = 'true' present)
- FOUND: solsys_code/tests/test_campaign_approval.py (3 new tests present)
- FOUND commit ee56256 (feat(js7): confirm-before-approve guard for unresolved site rows)
- FOUND commit 24d1d94 (test(js7): cover confirm-guard markup presence/absence)
