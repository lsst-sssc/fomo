---
status: complete
phase: 15-per-campaign-table-view-read-path
source: [15-01-SUMMARY.md, 15-02-SUMMARY.md]
started: 2026-07-03T19:30:34Z
updated: 2026-07-03T19:33:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Anonymous GET /campaigns/<pk>/ lists all CampaignRun rows for that campaign, 25/page, default-sorted obs_date descending
expected: Anonymous GET /campaigns/<pk>/ lists all CampaignRun rows for that campaign, 25/page, default-sorted obs_date descending
result: pass
source: automated
coverage_id: D1

### 2. contact_person/contact_email are excluded from SQL SELECT and response.context for anonymous/non-staff requests; present for staff
expected: contact_person/contact_email are excluded from SQL SELECT and response.context for anonymous/non-staff requests; present for staff
result: pass
source: automated
coverage_id: D2

### 3. run_status multi-select filter (OR semantics) and open_to_collaboration boolean filter narrow rows; unfiltered default shows every row
expected: run_status multi-select filter (OR semantics) and open_to_collaboration boolean filter narrow rows; unfiltered default shows every row
result: pass
source: automated
coverage_id: D3

### 4. Campaigns list page (/campaigns/) lists only TargetLists with >= 1 CampaignRun
expected: Campaigns list page (/campaigns/) lists only TargetLists with >= 1 CampaignRun (D-03)
result: pass
source: automated
coverage_id: D4

### 5. Target-detail page for a Target that belongs to a campaign shows a 'View {campaign.name} Runs' link, linking to that campaign's table
expected: Target-detail page for a Target that belongs to a campaign shows a 'View {campaign.name} Runs' link, linking to that campaign's table
result: pass
source: automated
coverage_id: D1

### 6. Target-detail page for a Target in zero campaigns shows no campaign link
expected: Target-detail page for a Target in zero campaigns shows no campaign link (empty partial output, no placeholder)
result: pass
source: automated
coverage_id: D2

### 7. Campaign discovery is via TargetList membership, not CampaignRun's target FK
expected: Campaign discovery is via TargetList membership, not CampaignRun's target FK -- proven by a fixture where the member target is never set as any CampaignRun's target
result: pass
source: automated
coverage_id: D3

### 8. Every page's navbar shows a single-word 'Campaigns' entry linking to campaigns:list
expected: Every page's navbar shows a single-word 'Campaigns' entry linking to campaigns:list
result: pass
source: automated
coverage_id: D4

### 9. Approval/run-status badges render correctly for staff and anonymous users
expected: |
  Open a campaign's table at /campaigns/<pk>/ while logged out (anonymous).
  The `approval_status` and `run_status` columns render as colored Bootstrap
  badges (not raw text like "planned" or "pending_review"), with the badge
  color/style matching the UI-SPEC contract (e.g. muted for pending/planned,
  more prominent colors for approved/completed/rejected states). Now log in
  as a staff user and view the same page — the badges look identical in
  content and styling to the anonymous view (no raw codes leaking through
  for either).
result: pass

## Summary

total: 9
passed: 9
issues: 0
pending: 0
skipped: 0

## Gaps

[none yet]
