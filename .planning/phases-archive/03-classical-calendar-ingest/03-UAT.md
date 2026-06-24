---
status: complete
phase: 03-classical-calendar-ingest
source:
  - .planning/phases/03-classical-calendar-ingest/03-01-SUMMARY.md
  - .planning/phases/03-classical-calendar-ingest/03-02-SUMMARY.md
started: 2026-06-16T17:30:00Z
updated: 2026-06-16T19:00:00Z
---

## Tests

### 1. Ingest a schedule file — correct event count
expected: |
  Write a schedule file with an NTT run of 5 nights:
    echo "NTT EFOSC2 allocation 9-13 July" > /tmp/test_schedule.txt
    ./manage.py load_telescope_runs /tmp/test_schedule.txt
  The summary line reports: created: 5, updated: 0, unchanged: 0, skipped: 0
result: PASS
notes: |
  Output: Done. lines processed: 1, created: 5, updated: 0, unchanged: 0, skipped: 0

### 2. CalendarEvent times — no sub-second precision
expected: |
  Query the events just created:
    ./manage.py shell -c "from tom_calendar.models import CalendarEvent; e=CalendarEvent.objects.first(); print(e.start_time.isoformat())"
  The timestamp has no decimal microseconds — looks like 2026-07-09T22:06:35+00:00,
  NOT 2026-07-09T22:06:35.917969+00:00
result: PASS
notes: |
  start_time: 2026-07-09T22:06:35+00:00
  end_time:   2026-07-10T11:29:46+00:00
  No microseconds present in either field.

### 3. CalendarEvent fields — title and description
expected: |
  For the first NTT event:
    ./manage.py shell -c "from tom_calendar.models import CalendarEvent; e=CalendarEvent.objects.filter(telescope='NTT').first(); print(e.title); print(e.description)"
  title is "NTT EFOSC2"
  description contains three lines:
    - "Dark window (-15 deg, UTC): ..." with ISO datetimes
    - "Status: allocation"
    - "Source line: NTT EFOSC2 allocation 9-13 July"
result: PASS
notes: |
  title: NTT EFOSC2
  description:
    Dark window (-15 deg, UTC): 2026-07-09T23:09:10+00:00 to 2026-07-10T10:27:15+00:00
    Status: allocation
    Source line: NTT EFOSC2 allocation 9-13 July

### 4. Ambiguous telescope line is skipped
expected: |
  Add a Magellan line with the bare telescope name (known to be ambiguous):
    printf "NTT EFOSC2 allocation 9-13 July\nMagellan IMACS 14-16 July (proposed)\n" > /tmp/test_sched2.txt
    ./manage.py load_telescope_runs /tmp/test_sched2.txt
  stderr shows a message about Line 2 being ambiguous ('Magellan' matches multiple sites).
  stdout summary reports skipped: 1. NTT events are still created (command does not abort).
result: PASS
notes: |
  stderr: Line 2: Ambiguous telescope 'Magellan': matches multiple SITES keys
    ['Magellan-Clay', 'Magellan-Baade']; use a more specific telescope name
    (e.g. "Magellan-Clay" or "Magellan-Baade"). (line text: 'Magellan IMACS 14-16 July (proposed)')
  stdout: Done. lines processed: 2, created: 0, updated: 0, unchanged: 5, skipped: 1
  (NTT events were already present from test 1, hence unchanged: 5)

### 5. Idempotent re-run
expected: |
  Run the same command a second time against the same file:
    ./manage.py load_telescope_runs /tmp/test_schedule.txt
  Summary reports: created: 0, updated: 0, unchanged: 5, skipped: 0
  Total CalendarEvent count does not increase.
result: PASS
notes: |
  Output: Done. lines processed: 1, created: 0, updated: 0, unchanged: 5, skipped: 0
  CalendarEvent.objects.count() = 5 (unchanged)

### 6. File not found — clean error message
expected: |
    ./manage.py load_telescope_runs /nonexistent/path/schedule.txt
  Exits with a formatted CommandError message like:
    "CommandError: Cannot open schedule file '/nonexistent/path/schedule.txt': ..."
  NOT a raw Python traceback. Exit code is non-zero.
result: PASS
notes: |
  Output: CommandError: Cannot open schedule file '/nonexistent/path/schedule.txt':
    [Errno 2] No such file or directory: '/nonexistent/path/schedule.txt'
  Exit code: 1. No traceback.

## Summary

total: 6
passed: 6
issues: 0
pending: 0
skipped: 0

## Gaps

[none]
