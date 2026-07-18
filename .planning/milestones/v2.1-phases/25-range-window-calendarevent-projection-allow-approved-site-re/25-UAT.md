---
status: complete
phase: 25-range-window-calendarevent-projection-allow-approved-site-re
source: [25-VERIFICATION.md]
started: 2026-07-18T00:00:00Z
updated: 2026-07-18T06:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Execute the real backfill for CampaignRun pk=34 (and decide on pk=27/29)
expected: |
  Run `./manage.py backfill_range_calendar_events` (without `--dry-run`) against
  `src/fomo_db.sqlite3`, after deciding whether pk=27 (3I/ATLAS (demo): FTN/FLOYDS)
  and pk=29 (Crash Test Campaign: FTN/MuSCAT3) — both also surfaced as qualifying
  candidates by the dry-run — should be included, or whether the run should be
  scoped to pk=34 only. CampaignRun pk=34 gets its 4 per-night CalendarEvents;
  the count() check for CAMPAIGN:34 becomes 4.
result: pass
notes: |
  User ran the real (non-dry-run) backfill against all 3 candidates. pk=34
  (GS-2026A-FT-115, Gemini South/GMOS-s) backfilled successfully — verified
  directly against src/fomo_db.sqlite3: 4 CalendarEvent rows
  (CAMPAIGN:34:2026-07-13 .. :07-16), correct dip-corrected per-night times,
  correct window-suffixed title. This closes the original debug symptom.

  pk=27 and pk=29 failed with "Observatory 'FTN' (obscode=F65) has no timezone
  set" — this is the command's designed graceful-skip behavior (FIX-08 must-have:
  a per-candidate ValueError is logged and skipped, never aborting the run), not
  a phase-25 code defect. Root cause is a pre-existing data gap in the FTN
  Observatory record (missing IANA timezone), out of scope for this phase.

  Observed UTC day-crossing in the calendar display (each night's event start ~22:07
  UTC, end ~11:30 UTC next day) is expected dip-corrected sunset->sunrise behavior
  from Phase 25-01, not a defect.

## Summary

total: 1
passed: 1
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

none
