---
status: testing
phase: 25-range-window-calendarevent-projection-allow-approved-site-re
source: [25-VERIFICATION.md]
started: 2026-07-18T00:00:00Z
updated: 2026-07-18T00:00:00Z
---

## Current Test

number: 1
name: Execute the real backfill for CampaignRun pk=34 (and decide on pk=27/29)
expected: |
  CampaignRun pk=34 (GS-2026A-FT-115) gets its 4 per-night CalendarEvents;
  CalendarEvent.objects.filter(Q(url='CAMPAIGN:34') | Q(url__startswith='CAMPAIGN:34:')).count()
  becomes 4, closing the exact symptom (count() returns 0) that the debug report
  (.planning/debug/range-window-calendar-event.md) opened with.
awaiting: user response

## Tests

### 1. Execute the real backfill for CampaignRun pk=34 (and decide on pk=27/29)
expected: |
  Run `./manage.py backfill_range_calendar_events` (without `--dry-run`) against
  `src/fomo_db.sqlite3`, after deciding whether pk=27 (3I/ATLAS (demo): FTN/FLOYDS)
  and pk=29 (Crash Test Campaign: FTN/MuSCAT3) — both also surfaced as qualifying
  candidates by the dry-run — should be included, or whether the run should be
  scoped to pk=34 only. CampaignRun pk=34 gets its 4 per-night CalendarEvents;
  the count() check for CAMPAIGN:34 becomes 4.
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
