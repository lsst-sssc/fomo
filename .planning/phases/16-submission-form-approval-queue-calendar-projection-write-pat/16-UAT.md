---
status: complete
phase: 16-submission-form-approval-queue-calendar-projection-write-pat
source: [16-01-SUMMARY.md, 16-02-SUMMARY.md, 16-03-SUMMARY.md, 16-04-SUMMARY.md]
started: 2026-07-04T14:45:00Z
updated: 2026-07-04T14:58:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Plan 16-01 auto-covered confirmation
expected: |
  All 5 deliverables (D1-D5) covered by passing automated tests — see Current Test for detail.
result: pass

### 2. D1 - CampaignRunSubmissionView minimal valid submission
expected: A minimal valid POST creates exactly one PENDING_REVIEW CampaignRun and redirects to campaigns:submission_thanks.
result: pass
source: automated
coverage_id: D1

### 3. D2 - Duplicate natural-key submission shows friendly error
expected: A duplicate campaign+telescope_instrument+ut_start submission surfaces a friendly non_field_errors banner, never a 500.
result: pass
source: automated
coverage_id: D2

### 4. D3 - Honeypot trip creates no run, sends no email
expected: A honeypot-tripped submission creates zero CampaignRun rows, sends zero emails, and redirects to the identical thanks page as a genuine submission.
result: pass
source: automated
coverage_id: D3

### 5. D4 - Thanks page markup identical for genuine vs honeypot
expected: submission_thanks.html renders identical markup for genuine and honeypot-tripped submissions, no conditional branch keyed on honeypot/status.
result: pass
source: automated
coverage_id: D4

### 6. D5 - Staff notification email, no PII
expected: A genuine submission emails every is_staff user with a non-empty email; subject/body contain no PII.
result: pass
source: automated
coverage_id: D5

### 7. Honeypot field is genuinely invisible to a sighted user
expected: |
  Open the campaign-run submission form (campaigns:submit) in a browser. The alt_contact_info
  honeypot field should not be visible or perceivable to a sighted user filling out the form
  normally — no visible label, no visible input box, nothing that looks like a real field to skip.
result: pass

### 8. D1 - Approval queue two-section layout + staff gating (GET)
expected: ApprovalQueueView renders a two-section staff-only queue (pending with Approve/Reject, recently-decided read-only capped at 20, -pk order); anonymous/non-staff GET is redirected, never 200 with pending content.
result: pass
source: automated
coverage_id: D1

### 9. D2 - Decision endpoint POST-only and staff-gated
expected: CampaignRunDecisionView is POST-only and staff-gated; anonymous/non-staff POST is redirected with no state change; an invalid action value returns 400.
result: pass
source: automated
coverage_id: D2

### 10. D3 - Double-approve is a proven no-op
expected: Approve/reject is an atomic conditional update; a double-approve is a proven no-op (status stays APPROVED, CalendarEvent count unchanged, second POST surfaces "already decided" warning).
result: pass
source: automated
coverage_id: D3

### 11. D4 - Calendar projection on approval
expected: Approving a run with telescope_instrument + ut_start + ut_end creates a CalendarEvent keyed CAMPAIGN:{pk} via insert_or_create_calendar_event, target_list set to the campaign; missing any one of the three creates zero events.
result: pass
source: automated
coverage_id: D4

### 12. D5 - Projected event's target_list is the campaign
expected: The projected CalendarEvent's target_list is the campaign's TargetList.
result: pass
source: automated
coverage_id: D5

### 13. D6 - Re-approve creates no duplicate/churn
expected: Re-approving an already-approved run creates no duplicate CalendarEvent and no modified-timestamp churn.
result: pass
source: automated
coverage_id: D6

### 14. Approval queue layout legibility + Reject confirm() dialog
expected: |
  Open the approval queue (campaigns:approval_queue) as a staff user in a browser. The two
  sections (pending / recently-decided) should read clearly at a glance — legible badge
  colors for approval/run status, sensible spacing in the Actions column. Clicking "Reject"
  should show a native browser confirm() dialog with clear, sensible copy before the decision
  is submitted.
result: issue
reported: "For the pending review entries, is there a way to hide blank entries/columns so that Actions column appears with less scrolling"
severity: minor

### 15. D1 - Non-staff table excludes pending_review
expected: Anonymous client GET of the per-campaign table excludes every pending_review run from the queryset/paginator count; approved and rejected rows remain visible.
result: pass
source: automated
coverage_id: D1

### 16. D2 - Staff table sees all statuses
expected: Staff client GET of the same table sees every approval_status, including pending_review (unchanged).
result: pass
source: automated
coverage_id: D2

### 17. D3 - pending_count in campaign-list context
expected: CampaignListView context exposes pending_count = number of pending_review runs across all campaigns.
result: pass
source: automated
coverage_id: D3

### 18. D4 - Submit-a-Run entry points + staff banner wiring
expected: A "Submit a Run" entry button appears on the campaigns list page and per-campaign table page; a staff-only "N pending review" banner on the campaigns list links to the approval queue when is_staff and pending rows exist.
result: pass
source: automated
coverage_id: D4

### 19. Banner/button layout and visual hierarchy
expected: |
  Open the campaigns list page in a browser, once as a staff user and once anonymously (or
  non-staff). As staff with pending runs: the "N pending review" banner and "Submit a Run"
  button should read clearly, positioned sensibly relative to the page header, with reasonable
  spacing (Bootstrap alert styling, not cramped or overlapping). As anonymous/non-staff: the
  banner should not appear at all, only the "Submit a Run" button.
result: pass

## Summary

total: 19
passed: 18
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "The two sections (pending / recently-decided) should read clearly at a glance — legible badge colors for approval/run status, sensible spacing in the Actions column."
  status: failed
  reason: "User reported: For the pending review entries, is there a way to hide blank entries/columns so that Actions column appears with less scrolling"
  severity: minor
  test: 14
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""
