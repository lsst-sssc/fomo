---
status: diagnosed
phase: 27-the-canonical-run-record
source: 27-01-SUMMARY.md, 27-02-SUMMARY.md, 27-03-SUMMARY.md, 27-04-SUMMARY.md, 27-05-SUMMARY.md, 27-06-SUMMARY.md, quick/260730-jty-SUMMARY.md
started: 2026-07-30T22:40:00Z
updated: 2026-07-30T23:10:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Cold Start Smoke Test
expected: Stop any running dev server. Start fresh with `python manage.py runserver`. Server boots with no errors, `python manage.py migrate` reports no pending migrations (0008-0012 all applied), and the home page loads with live data.
result: pass

### 2. Sites Needing Review queue no longer lists class-wide or space runs
expected: Open the campaign approval queue. The "Sites Needing Review" card is empty (0 rows). Specifically, the four runs that carry a telescope class -- pk=26 JUICE (SPACE), pk=29 LCO 1m, pk=30 LCO 2m, pk=37 Generic 1m -- do NOT appear, because a known telescope class is an answer to "why is there no site", not a resolution failure.
result: issue
reported: "no approval queue visible in any of the 4 campaigns"
severity: major
note: |
  The data-level assertion was verified programmatically and HOLDS -- GET
  /campaigns/approval-queue/ returns 200 with 0 site-review rows, and none of
  pk=26/29/30/37 appear. The failure is navigational: there is no clickable
  path to the page.

### 3. Telescope class visible on the public run table
expected: Open the campaign runs table at /campaigns/<pk>/ as a NON-staff user (or logged out). A "Telescope class allocation" column is present and shows 1m0 / 2m0 / SPACE for the four classed runs, blank for the rest. The "Ingest source" column is NOT visible to non-staff.
result: pass
note: confirmed on /campaigns/3/

### 4. Admin change-list: filter by telescope class, source is staff-visible
expected: In Django admin, open CampaignRun. A "Telescope class allocation" filter appears in the right-hand sidebar; selecting 1m0 narrows the list to the 1m0 runs only. The "Ingest source" column/filter IS visible here (staff surface), showing legacy for 33 rows and csv_import for 10.
result: pass

### 5. Attribute an observation record to a run via the admin inline
expected: In Django admin, open any CampaignRun. A "Campaign run observations" inline is present and empty. Add a row selecting any ObservationRecord and save. The row persists, and its confirmed_by and confirmed_at are stamped with YOUR user automatically -- you never typed them.
result: pass

### 6. Calendar event modal links back to its owning run
expected: Setup first -- in Django admin, set a CalendarEventMeta row's "Run" FK to a publicly-visible CampaignRun. Then open that event in the calendar and click it. The modal shows a link back to the owning run, and following it lands on that run.
result: issue
reported: "That sort of worked but selecting the right Run is difficult as almost all of the 11 events have the save name with no date. Clicking on the event on July 7 in the calendar which was linked into CalendarEventMeta, reveals a whole bunch of what might be template comments: {# FOMO override of the upstream tom_calendar partial ... #}"
severity: major
note: |
  Two distinct defects in one test. The link itself works (setup succeeded and
  the modal did resolve the run), so CANON-05/D-08 is functionally delivered --
  but the modal is disfigured by leaked template source, and the admin picker
  that phase 27 makes staff depend on is not usable in practice.

### 7. CSV import records source and telescope class
expected: Run `python manage.py import_campaign_csv --campaign "<name>" <file.csv>` against a small CSV containing one row with a resolvable site code and one row with a blank Site Code but an instrument like "LCO 1m". The summary prints site_needs_review: 0 for the classed row. In admin, the first row has source=csv_import with blank telescope class; the second has source=csv_import, telescope_class=1m0, and is NOT flagged for site review.
result: pass
note: |
  Command reported "created: 1, updated: 0, unchanged: 1, skipped: 0,
  site_needs_review: 0, window_needs_review: 0"; the classed row landed as
  pk=44 with telescope_class=1m0 and source=csv_import.

## Summary

total: 7
passed: 5
issues: 2
pending: 0
skipped: 0
blocked: 0
gaps: 3

## Gaps

- truth: "Staff can reach the Sites Needing Review queue to see that class-wide and space runs are no longer flagged"
  status: failed
  reason: "User reported: no approval queue visible in any of the 4 campaigns"
  severity: major
  test: 2
  root_cause: "src/templates/campaigns/campaign_list.html:10 gates the only link to the approval queue on `{% if request.user.is_staff and pending_count %}`. The approval-queue page hosts TWO independent work queues -- pending submissions AND the D-07 'Sites Needing Review' card added by Phase 27 -- but only pending_count drives the link. With pending_count == 0, staff have no navigation path to the site-review queue even when it has actionable rows. Phase 27 added the second queue without widening the entry condition."
  artifacts:
    - path: "src/templates/campaigns/campaign_list.html"
      issue: "link gated on pending_count alone; ignores site_needs_review rows"
    - path: "solsys_code/campaign_views.py"
      issue: "CampaignListView supplies pending_count but no site-review count for the template to gate on"
  missing:
    - "Expose a site-review count in CampaignListView's context alongside pending_count"
    - "Widen the campaign_list.html condition so the link shows when EITHER queue has rows"
    - "Consider a persistent staff nav entry for the approval queue, independent of row counts"
  debug_session: ""

- truth: "The calendar event modal shows a clean link back to its owning run, with no template internals visible"
  status: failed
  reason: "User reported: clicking the July 7 event reveals a whole bunch of what might be template comments, starting {# FOMO override of the upstream tom_calendar partial ... #}"
  severity: major
  test: 6
  root_cause: "src/templates/tom_calendar/partials/event_form.html opens with a 16-line `{# ... #}` block (lines 1-16). Django's `{# #}` comment syntax is SINGLE-LINE ONLY -- a multi-line `{# ... #}` is not recognised as a comment and renders as literal text. Verified directly: Template('{# a\\nb #}VISIBLE').render() returns the literal '{# a\\nb #}VISIBLE', while the same content in `{% comment %}...{% endcomment %}` renders only 'VISIBLE'. The 27-REVIEW deep pass byte-diffed this file against the upstream tomtoolkit 3.0.0a9 partial and cleared it as 'a clean copy plus the inserted block' -- true of the diff, but it never checked what the file RENDERS, so a structural comparison could not catch it."
  artifacts:
    - path: "src/templates/tom_calendar/partials/event_form.html"
      issue: "lines 1-16 use multi-line {# #} which Django renders literally into the event modal"
  missing:
    - "Convert the lines 1-16 header to {% comment %}...{% endcomment %}"
    - "Grep the repo for other multi-line {# #} blocks introduced by the same authoring habit"
    - "Add a render-level assertion (not just a byte-diff) for FOMO's tom_calendar template overrides"
  debug_session: ""

- truth: "Staff can identify and select the correct run when hand-linking a CalendarEventMeta row"
  status: failed
  reason: "User reported: selecting the right Run is difficult as almost all of the 11 events have the save [same] name with no date"
  severity: minor
  test: 6
  root_cause: "CalendarEventMeta / CampaignRun admin relies on the default __str__ for its FK picker labels, which renders near-identical text for the 11 companion rows -- no date, no site, no disambiguator. Phase 27 (WR-03) makes this picker the ONLY way a run-to-event link can be created, since no production writer exists until the Phase 29 reconciler, so the picker's legibility is load-bearing rather than cosmetic."
  artifacts:
    - path: "solsys_code/models.py"
      issue: "CampaignRun.__str__ / CalendarEventMeta.__str__ produce non-distinguishing labels"
    - path: "solsys_code/admin.py"
      issue: "no raw_id_fields, autocomplete_fields, or custom label_from_instance to disambiguate the picker"
  missing:
    - "Give CampaignRun.__str__ (or the admin field's label_from_instance) a date and site/telescope discriminator"
    - "Consider autocomplete_fields for the run FK so staff can search rather than scan a flat list"
  debug_session: ""
