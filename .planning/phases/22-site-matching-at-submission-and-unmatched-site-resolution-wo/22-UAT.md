---
status: complete
phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
source: [22-VERIFICATION.md]
started: 2026-07-15T18:00:00Z
updated: 2026-07-15T18:50:00Z
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
  artifacts: []
  missing: []

- truth: "In the staff approval queue, the Sites Needing Review row's site input supports searching and picking a suggestion; submitting resolves as expected."
  status: failed
  reason: "User reported two problems from a screenshot: (1) placement — the Sites Needing Review table renders at the bottom of the approval-queue page after the approved/decided runs, not grouped clearly under the Approval Queue heading, which reads as unhelpful; (2) a projection-failed-retry row (site already resolved to plain-text 'DCT') offers only a Resolve button with no way to correct the site if 'DCT' is a placeholder/wrong value rather than a genuinely resolved Observatory — by design (22-03-PLAN.md), retry rows intentionally render site as read-only text with no search input, but the user expected an escape hatch to fix a wrong/placeholder resolution rather than being limited to a projection-retry-only action. NOTE: orchestrator confirmed via Django shell that the local Observatory row with obscode 'DCT' has name 'NEEDS REVIEW: DCT' and blank timezone — it IS a placeholder record (likely created via an earlier create_placeholder=True path, e.g. Phase 21's approve flow), not a genuinely resolved site, corroborating the user's suspicion."
  severity: major
  test: 2
  artifacts: []
  missing: []

- truth: "Resolving a Sites Needing Review row's unresolved site (site=None) via its live-search widget renders suggestions the same way the public form and pending-queue widgets do."
  status: failed
  reason: "User reported: typing 'G96' or 'Mt. Lemmon' into the Site search box for an unresolved Sites Needing Review row (tested against orchestrator-injected CampaignRun pk=30 on 'Test Campaign', site=None, site_raw='G96', site_needs_review=True, single-night window 2026-08-01, APPROVED) produces no suggestions, same symptom as test 1. This is very likely the SAME root cause as test 1 (query-string key mismatch: the widget's GET request is keyed by the input's `name` attribute rather than `q`, which SiteSearchView.get() reads) — all three widget instances (public form, pending-queue row, Sites Needing Review row) share the same underlying markup-construction pattern per 22-02-PLAN.md/22-03-PLAN.md, so a fix to the shared widget construction should resolve all three occurrences at once. Could not proceed to observe the actual CR-01 warning-and-retry banner behavior because no suggestion could be picked to drive a resolution."
  severity: major
  test: 3
  artifacts: []
  missing: []
