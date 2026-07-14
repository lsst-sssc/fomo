---
status: complete
phase: 21-site-disambiguation-submitter-contact-opt-in
source: [21-01-SUMMARY.md, 21-02-SUMMARY.md, 21-03-SUMMARY.md, 21-04-SUMMARY.md]
started: 2026-07-14T09:23:05Z
updated: 2026-07-14T09:31:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Cold Start Smoke Test
expected: Kill any running server/service. Clear ephemeral state (temp DBs, caches, lock files). Start the application from scratch. Server boots without errors, migration 0007 (contact_public_opt_in) applies cleanly, and a primary query (approval queue page load or homepage) returns live data.
result: pass

### 2. Bulk MPC obscode fetch preserves query() contract
expected: MPCObscodeFetcher.query_all() bulk-fetches the full MPC obscode list (json={} body) and stores it on self.obs_data without disturbing query()'s single-code contract
result: pass
source: automated
coverage_id: D1

### 3. Candidate pool merges local + cached MPC data
expected: build_site_candidates() merges local Observatory rows with a 24h-cached bulk MPC fetch into a flattened {string: obscode} pool, falling back to local-only on MPC failure
result: pass
source: automated
coverage_id: D2

### 4. Fuzzy match wrapper returns ranked candidates
expected: fuzzy_match_candidates() wraps difflib.get_close_matches to return ranked (display_string, obscode) pairs, correctly returning [] for an acronym/nickname (e.g. 'DCT') that difflib cannot bridge
result: pass
source: automated
coverage_id: D3

### 5. Submitter contact opt-in checkbox persists
expected: Submitter can tick a single 'show my contact info publicly' checkbox on the submission form (default unchecked); value persists onto the created CampaignRun
result: pass
source: automated
coverage_id: D1

### 6. Contact info visibility gated by opt-in
expected: An opted-in run's contact_person/contact_email are visible to anonymous visitors on the per-campaign table; an opted-out run's contact fields are never present in the non-staff queryset SELECT (blanked at SQL, not template)
result: pass
source: automated
coverage_id: D2

### 7. Approval queue renders inline site disambiguation control
expected: ApprovalQueueTable.render_site() renders an inline site_selection input + fuzzy-matched <datalist> + always-visible 'Create new Observatory' link for an unresolved actionable pending row; resolved and decided-table rows are unchanged
result: pass
source: automated
coverage_id: D1

### 8. Site disambiguation input escapes submitter-controlled text
expected: All submitter-controlled site_raw and MPC-sourced candidate text is escaped via format_html/format_html_join -- a stored-XSS attempt in site_raw never reaches the response unescaped
result: pass
source: automated
coverage_id: D2

### 9. Approve/Reject collapsed into a single form
expected: render_actions() collapses the row's two Approve/Reject forms into one <form id=decide-form-{pk}>, and the Site column's input submits into it via form=; existing approve/reject POST semantics are unchanged
result: pass
source: automated
coverage_id: D3

### 10. Candidate pool built once per request
expected: The merged candidate pool is built once per request (ApprovalQueueView.get_context_data), not per row
result: pass
source: automated
coverage_id: D4

### 11. Already-resolved site is never clobbered on approve
expected: Approving a run whose site is already set skips resolve_site() entirely and never overwrites the existing resolution
result: pass
source: automated
coverage_id: D1

### 12. Staff site_selection resolves via the 3-tier resolver
expected: Approving a run with site=None reads the staff-submitted site_selection (falling back to site_raw) and resolves it via the existing 3-tier resolver, never fabricating a placeholder Observatory
result: pass
source: automated
coverage_id: D2

### 13. Oversized site_selection is flagged, not processed
expected: An oversized/malformed site_selection is flagged for review via resolve_site's existing _MAX_OBSCODE_LEN guard, not processed or crashed, with no network call
result: pass
source: automated
coverage_id: D3

### 14. Create-new-Observatory round-trip prefill + safe redirect
expected: Staff can create a new Observatory from an unresolved row and be returned to the approval queue afterward via ?obscode= prefill + a validated ?next= redirect, with an open-redirect fallback for an unsafe next
result: pass
source: automated
coverage_id: D4

## Summary

total: 14
passed: 14
issues: 0
pending: 0
skipped: 0

## Gaps

[none yet]
