---
status: diagnosed
phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
source: [22-VERIFICATION.md]
started: 2026-07-15T18:00:00Z
updated: 2026-07-15T19:10:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Public form live-search fires, debounces, and fills in a real browser
expected: On the public 'Submit an Observing Run' form, typing 'faulkes' (2+ chars) into the
  Observing site field and waiting ~300ms fires an hx-get to /campaigns/site-search/; a
  suggestion list appears below the field showing both Faulkes sites as 'Display Name
  (obscode)'; clicking one fills the input with that exact text. Typing 1 character does
  nothing — no request fires before 2 characters.
result: issue
reported: |
  Something is happening when I type into the 'Observing Site' on /campaigns/submit/ as these lines:
  [15/Jul/2026 15:15:50] "GET /campaigns/site-search/?site_raw=fa&input_id=id_site_raw HTTP/1.1" 200 2
  [15/Jul/2026 15:15:52] "GET /campaigns/site-search/?site_raw=faulk&input_id=id_site_raw HTTP/1.1" 200 2
  [15/Jul/2026 15:15:53] "GET /campaigns/site-search/?site_raw=faulke&input_id=id_site_raw HTTP/1.1" 200 2
  [15/Jul/2026 15:15:53] "GET /campaigns/site-search/?site_raw=faulkes&input_id=id_site_raw HTTP/1.1" 200 2
  appear in the runserver logs but no suggestion list appears in the form
severity: major

### 2. Multi-row approval-queue widgets don't cross-fill between rows
expected: In the staff approval queue, using the pending-row site input and the Sites Needing
  Review row's site input to search and pick a suggestion fills the correct row's input (not a
  different row's); the 'Create new Observatory' link still works; submitting resolves/approves
  as expected.
result: issue
reported: |
  The Sites Needing Review appears but at the bottom after the approved runs and not under the
  Approval Queue banner which is unhelpful. There is no option to correct the placeholder 'DCT'
  entry that is there to select the proper site code - see screenshot.
  Screenshot shows two Sites Needing Review rows: row 1 (DCT telescope) has Site rendered as
  plain text "DCT" with only a Resolve button (no search input) - this is the
  already-resolved / projection-failed-retry rendering path (site_short_name set), by design
  per 22-03-PLAN.md, but the user expects a way to correct it if "DCT" is a placeholder/wrong
  value rather than a genuinely resolved Observatory. Row 2 (Generic 1m robotic telescope)
  correctly shows the live-search widget ("MPC code" input + Create new Observatory link) since
  its site is still unresolved.
  Screenshot saved at: /home/tlister/.claude/image-cache/945378df-779f-43c5-8119-fb584930b1b2/1.png
severity: major

### 3. Sites Needing Review resolve UX for the CR-01 blank-timezone fix
expected: Resolving a Sites Needing Review row end-to-end against a real (or realistic Tier-2)
  MPC obscode with a blank Observatory.timezone shows a warning, keeps the row in the table, and
  a later Resolve retries — the banner and row persistence behave as CR-01's fix describes when
  driven through the actual UI, not just the regression test's direct POST.
result: issue
reported: |
  Typing 'G96' or 'Mt. Lemmon' into Site box for that row also fails to produce any
  suggestions the same way as it behaved on the "Submit run" form.
  Tested against injected CampaignRun pk=30 (Test Campaign, telescope_instrument='CR-01 UAT
  Test 1m', site=None, site_raw='G96', site_needs_review=True, window 2026-08-01, APPROVED).
  User could not proceed to actually resolving/observing the CR-01 warning-and-retry banner
  behavior because the live-search widget on this row never renders suggestions to click,
  same symptom as test 1.
severity: major

## Summary

total: 3
passed: 0
issues: 3
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "On the public 'Submit an Observing Run' form, typing 'faulkes' (2+ chars) into the Observing site field fires an hx-get to /campaigns/site-search/ and a suggestion list appears below the field showing both Faulkes sites, with click-to-fill working."
  status: failed
  reason: "User reported: the hx-get fires correctly and is debounced (GET requests visible in runserver logs at fa -> faulk -> faulke -> faulkes), each returning HTTP 200 with a 2-byte body, but the query string key is `site_raw` (e.g. `?site_raw=faulkes&input_id=id_site_raw`) rather than `q` — SiteSearchView.get() reads `request.GET.get('q', '')`, so the server sees an empty query, hits the len<2 empty-fragment path, and renders nothing. No suggestion list appears."
  severity: major
  test: 1
  root_cause: "SiteSearchView.get() (campaign_views.py:792) reads the search term exclusively via request.GET.get('q', ''). Neither widget's <input> ever sends a `q` param: the public form's site_raw TextInput renders name=\"site_raw\" (the Django form field name) with hx-vals adding only `input_id`; htmx's hx-get does NOT auto-include enclosing-form values for GET (unlike POST) — it serializes only the triggering element's own name-keyed value plus hx-vals/hx-vars. So the actual request is `?site_raw=<text>&input_id=id_site_raw`, never `?q=<text>`, so request.GET.get('q','') is always empty and trips the len<2 empty-fragment gate regardless of what was typed."
  artifacts:
    - path: "solsys_code/campaign_views.py"
      issue: "SiteSearchView.get() line ~792 hard-codes request.GET.get('q', '') as its only input source"
    - path: "solsys_code/campaign_forms.py"
      issue: "CampaignRunSubmissionForm.site_raw widget (lines ~31-47) renders name=\"site_raw\"; hx-vals carries only input_id, never q"
  missing:
    - "Align the query-param key the view reads with what the widget actually sends, without breaking the real POST field name (site_raw must stay site_raw for form submission)"
    - "Either: view accepts q OR site_raw OR site_selection as fallback param names, OR each widget sends the typed value under q explicitly via dynamic hx-vals='js:{\"q\": event.target.value, \"input_id\": \"...\"}\'"
  debug_session: .planning/debug/site-search-widget-query-param-mismatch.md

- truth: "In the staff approval queue, the Sites Needing Review row's site input supports searching and picking a suggestion; submitting resolves as expected."
  status: failed
  reason: "User reported two problems from a screenshot: (1) placement — the Sites Needing Review table renders at the bottom of the approval-queue page after the approved/decided runs, not grouped clearly under the Approval Queue heading, which reads as unhelpful; (2) a projection-failed-retry row (site already resolved to plain-text 'DCT') offers only a Resolve button with no way to correct the site if 'DCT' is a placeholder/wrong value rather than a genuinely resolved Observatory — by design (22-03-PLAN.md), retry rows intentionally render site as read-only text with no search input, but the user expected an escape hatch to fix a wrong/placeholder resolution rather than being limited to a projection-retry-only action. NOTE: orchestrator confirmed via Django shell that the local Observatory row with obscode 'DCT' has name 'NEEDS REVIEW: DCT' and blank timezone — it IS a placeholder record (likely created via an earlier create_placeholder=True path, e.g. Phase 21's approve flow), not a genuinely resolved site, corroborating the user's suspicion."
  severity: major
  test: 2
  root_cause: "Two distinct, independently-confirmed causes, both faithful implementations of Phase 22's own planning docs rather than coding slips: (A) Placement — 22-CONTEXT.md decision D-07 explicitly locked table order as 'pending / decided / sites-needing-review'; 22-UI-SPEC.md and 22-03-PLAN.md Task 2 carried this into approval_queue.html verbatim, rendering all three tables as plain DOM siblings with no visual differentiation between the actionable work queue and the historical audit table above it — never checked against a real rendered page before UAT. (B) No correction path for a placeholder site — ApprovalQueueTable.render_site() (campaign_tables.py:266-268) gates its live-search widget solely on whether run.site is set at all, with no way to distinguish a genuinely-resolved Observatory from a placeholder one created by resolve_site()'s tier-3 fallback (name=f'NEEDS REVIEW: {code}', blank timezone, campaign_utils.py:196-213). This is compounded server-side: CampaignRunDecisionView.post()'s resolve_site action only resolves when run.site is None (the D-06 never-re-resolve guard), so a UI fix alone is insufficient — the view also refuses to replace an already-set site. No model field/helper exists to mark/detect a placeholder Observatory; the only signal is the ad-hoc 'NEEDS REVIEW: ' name-string prefix, unread anywhere at render/resolve time."
  artifacts:
    - path: "src/templates/campaigns/approval_queue.html"
      issue: "Table order (Pending Review -> Recently Decided -> Sites Needing Review) with no visual grouping distinguishing actionable-vs-historical sections, per locked D-07 ordering"
    - path: "solsys_code/campaign_tables.py"
      issue: "ApprovalQueueTable.render_site()/_render_site_search_widget() early-returns on any truthy site, with no placeholder detection"
    - path: "solsys_code/campaign_views.py"
      issue: "CampaignRunDecisionView.post() resolve_site branch only resolves when run.site is None; no path to replace an already-set (placeholder) site"
    - path: "solsys_code/campaign_utils.py"
      issue: "resolve_site() tier-3 (lines ~196-213) creates the placeholder Observatory with no model-level flag marking it as such"
  missing:
    - "Reorder/regroup the approval-queue page so Sites Needing Review has distinct visual weight as an actionable section, not lumped after the historical Recently Decided table"
    - "Add a queryable way to detect a placeholder Observatory (model field, or shared helper checking the 'NEEDS REVIEW: ' prefix)"
    - "Have render_site() show the live-search widget (not just plain text) when the row's site is a placeholder"
    - "Extend the resolve_site view action to allow replacing a placeholder site, distinguishing 'never re-resolve a genuinely-resolved site' from 'allow correcting a known-placeholder site'"
  debug_session: .planning/debug/sites-needing-review-placement-and-placeholder-correction.md

- truth: "Resolving a Sites Needing Review row's unresolved site (site=None) via its live-search widget renders suggestions the same way the public form and pending-queue widgets do."
  status: failed
  reason: "User reported: typing 'G96' or 'Mt. Lemmon' into the Site search box for an unresolved Sites Needing Review row (tested against orchestrator-injected CampaignRun pk=30 on 'Test Campaign', site=None, site_raw='G96', site_needs_review=True, single-night window 2026-08-01, APPROVED) produces no suggestions, same symptom as test 1. This is very likely the SAME root cause as test 1 (query-string key mismatch: the widget's GET request is keyed by the input's `name` attribute rather than `q`, which SiteSearchView.get() reads) — all three widget instances (public form, pending-queue row, Sites Needing Review row) share the same underlying markup-construction pattern per 22-02-PLAN.md/22-03-PLAN.md, so a fix to the shared widget construction should resolve all three occurrences at once. Could not proceed to observe the actual CR-01 warning-and-retry banner behavior because no suggestion could be picked to drive a resolution."
  severity: major
  test: 3
  root_cause: "Confirmed same root cause as test 1's gap: ApprovalQueueTable._render_site_search_widget() (campaign_tables.py, shared by pending and resolve-mode rows via render_site()) hand-renders name=\"site_selection\" with hx-vals carrying only input_id — never a `q` key — so SiteSearchView.get()'s request.GET.get('q', '') is always empty for this widget too, identical mechanism to the public form's widget."
  artifacts:
    - path: "solsys_code/campaign_tables.py"
      issue: "ApprovalQueueTable._render_site_search_widget() (lines ~212-245), called from render_site() (lines ~247-281) for both pending and resolve-mode rows — input name is site_selection, no q-keyed value"
  missing:
    - "Same fix as test 1's gap — a single shared widget-construction fix should resolve both test 1 and test 3 simultaneously"
  debug_session: .planning/debug/site-search-widget-query-param-mismatch.md
