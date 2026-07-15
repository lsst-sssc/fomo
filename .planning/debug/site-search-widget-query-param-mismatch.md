---
status: diagnosed
trigger: "Investigate issue: site-search-widget-query-param-mismatch — Live-search suggestion widgets (public submission form, and the \"Sites Needing Review\" row) never render suggestions despite the debounced hx-get firing correctly."
created: 2026-07-15T19:00:00Z
updated: 2026-07-15T19:09:00Z
---

## Current Focus
<!-- OVERWRITE on each update - reflects NOW -->

hypothesis: CONFIRMED. SiteSearchView.get() reads `request.GET.get('q', '')`, but neither the public-form widget (`name="site_raw"`) nor the approval-queue widget (`name="site_selection"`) sends a `q` param -- htmx's hx-get requests only serialize the triggering element's own `name`-keyed value plus `hx-vals` (never an enclosing form's other fields, per htmx docs), and `hx-vals` on both widgets only adds `input_id`, never `q`.
test: n/a -- root cause confirmed via direct code read + external doc verification.
expecting: n/a
next_action: n/a -- diagnose-only mode, returning ROOT CAUSE FOUND.

## Symptoms
<!-- Written during gathering, then IMMUTABLE -->

expected: Typing 2+ characters into the site-search input fires a debounced hx-get to /campaigns/site-search/ and a suggestion list (matching Faulkes/MPC sites etc.) renders below the field.
actual: GET requests fire correctly per Django runserver logs, properly debounced (e.g. progressively `GET /campaigns/site-search/?site_raw=fa&input_id=id_site_raw HTTP/1.1" 200 2`, then `...faulk...`, `...faulke...`, `...faulkes...`), but no suggestion list ever appears in the browser. Reproduced identically on two separate surfaces: (1) the public "Submit an Observing Run" form's Observing site field (UAT Test 1), and (2) an unresolved "Sites Needing Review" row's site-search widget for an injected CampaignRun (pk=30, site_raw='G96', site_needs_review=True) (UAT Test 3).
errors: None reported in the browser or server logs — each request returns HTTP 200 with a 2-byte response body (consistent with the view rendering its empty-fragment path).
reproduction: UAT Test 1 and Test 3 in .planning/phases/22-site-matching-at-submission-and-unmatched-site-resolution-wo/22-UAT.md.
started: Discovered during Phase 22 UAT (2026-07-15).

## Eliminated
<!-- APPEND only - prevents re-investigating -->

## Evidence
<!-- APPEND only - facts discovered -->

- timestamp: 2026-07-15T19:05:00Z
  checked: solsys_code/campaign_views.py SiteSearchView.get() (lines 755-813)
  found: "`query = request.GET.get('q', '')` (line 792); if `len(query.strip()) < 2` it renders the empty-fragment partial with `candidates: []` (lines 793-798) -- this is exactly the always-2-byte-body path the user observed."
  implication: The view's entire suggestion pipeline is gated on a GET param literally named `q`.

- timestamp: 2026-07-15T19:06:00Z
  checked: solsys_code/campaign_forms.py CampaignRunSubmissionForm.site_raw field widget (lines 31-47)
  found: "The widget is `forms.TextInput` on a form field named `site_raw`, so Django renders `<input name=\"site_raw\" ...>`. Its attrs carry `hx-get`, `hx-trigger='input[this.value.length >= 2] changed delay:300ms'`, `hx-target`, `hx-swap`, and `hx-vals='{\"input_id\": \"id_site_raw\"}'` -- hx-vals adds only `input_id`, never `q`, and there is no `name` override."
  implication: The public-form widget's own name attribute is `site_raw`, not `q` -- matches the user-reported request `?site_raw=fa&input_id=id_site_raw` exactly.

- timestamp: 2026-07-15T19:07:00Z
  checked: solsys_code/campaign_tables.py ApprovalQueueTable._render_site_search_widget() (lines 212-245) and render_site() (lines 247-281)
  found: "Hand-built `format_html` markup renders `<input type=\"text\" name=\"site_selection\" ... hx-get=\"{site_search_url}\" hx-trigger=\"input[this.value.length >= 2] changed delay:300ms\" ... hx-vals='{{\"input_id\": \"{1}\"}}'>` for BOTH the pending-mode row and the resolve-mode (Sites Needing Review) row -- same shared helper, same missing `q` key. This is the widget exercised by UAT Test 3 (CampaignRun pk=30)."
  implication: The Sites Needing Review row's widget has the identical structural defect as the public form's widget (different name value, same missing 'q'), confirming this is one shared root cause affecting both UAT failures (Test 1 and Test 3), not two separate bugs.

- timestamp: 2026-07-15T19:08:00Z
  checked: htmx official documentation (hx-get / hx-params attribute pages) via web search
  found: "For hx-get requests specifically, htmx does NOT gather values from an enclosing <form> the way it does for POST -- it only includes the triggering element's own value (serialized under its own `name` attribute) plus anything added via `hx-vals`/`hx-vars`, unless `hx-include`/`hx-params` is used to change that."
  implication: This confirms, from htmx's documented behavior (not just inference), that the observed requests (`?site_raw=fa&input_id=...`, and by the same mechanism `?site_selection=<value>&input_id=...` for the queue widget) are htmx behaving exactly as designed -- the bug is entirely in the query-param-name mismatch between the widgets (name=site_raw / name=site_selection) and the view (reads 'q'), not any misconfigured hx-trigger, hx-target, or debounce logic (all of which were separately verified correct by the existing Django TestCase markup tests per the symptom timeline).

## Resolution
<!-- OVERWRITE as understanding evolves -->

root_cause: |
  SiteSearchView.get() (solsys_code/campaign_views.py:792) reads the search term exclusively via
  `request.GET.get('q', '')`. Neither site-search widget's rendered `<input>` element sends a `q`
  param: the public submission form's field (solsys_code/campaign_forms.py `site_raw`, TextInput
  widget) has `name="site_raw"` (Django derives it from the form field name), and the approval-queue
  / Sites-Needing-Review widget (solsys_code/campaign_tables.py
  `ApprovalQueueTable._render_site_search_widget()`, shared by both pending and resolve-mode rows)
  hand-renders `name="site_selection"`. Both widgets' `hx-vals` attribute only injects `input_id`,
  never `q`, and htmx's documented hx-get behavior does NOT auto-gather values from an enclosing
  `<form>` for GET requests (unlike POST) -- it serializes only the triggering element's own
  name-keyed value plus hx-vals/hx-vars. So the actual fired requests are
  `?site_raw=<text>&input_id=id_site_raw` (public form) and `?site_selection=<text>&input_id=...`
  (queue widgets) -- `request.GET.get('q', '')` is therefore always `''`, which trips the
  `len(query.strip()) < 2` gate (line 793) and renders the empty-fragment partial (`candidates: []`)
  on every keystroke, regardless of what was typed. This is a single shared root cause manifesting
  identically on both UAT-reported surfaces (Test 1 public form, Test 3 Sites Needing Review row),
  since both widgets independently retain their real form/POST field name rather than sending the
  value under the key the view expects.
fix:
verification:
files_changed: []
