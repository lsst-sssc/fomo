---
status: diagnosed
phase: 27-the-canonical-run-record
source: 27-01-SUMMARY.md, 27-02-SUMMARY.md, 27-03-SUMMARY.md, 27-04-SUMMARY.md, 27-05-SUMMARY.md, 27-06-SUMMARY.md, quick/260730-jty-SUMMARY.md
started: 2026-07-30T22:40:00Z
updated: 2026-08-06T16:05:00Z
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

### 8. Re-verify — approval queue nav reachable via site-review count alone
expected: Open /campaigns/ as a staff user on a campaign that has runs needing site review but
  zero pending submissions. A warning banner appears at the top showing "N run(s) needing
  site review" with a "Review queue" button. Clicking it lands on /campaigns/approval-queue/
  and shows those flagged runs.
result: issue
reported: "Yes, I would still like the 'Sites Needing Review - action required' section to be at the top not the bottom"
severity: minor
note: |
  The originally reported navigation gap IS fixed -- the banner appears and the link
  reaches the queue. This is a new, separate finding: on /campaigns/approval-queue/
  (src/templates/campaigns/approval_queue.html), "Sites Needing Review" renders as the
  THIRD section, below "Pending Review" and "Recently Decided". When pending_count is 0
  (the exact scenario this gap covers), Sites Needing Review is the only actionable table
  on the page but is buried below two others, one of which (Recently Decided) is purely
  informational.

### 9. Re-verify — calendar event modal renders no leaked template source
expected: Open /calendar/, click one of the near-identical companion events (e.g. the July
  2026 "[EXPIRED] 2m0 2M0-SCICAM-MUSCAT" events) that is linked to a CampaignRun via
  CalendarEventMeta. The modal shows only real form fields and the run-link block back to
  the owning run — no visible template comment text such as "{# FOMO override...".
result: issue
reported: "Yes but one calendarevent for 2026-7-16 (which has '[EXPIRED]  2m0 2M0-SCICAM-MUSCAT' in the title) doesn't have a link to the CampaignRun but the other one for FTS/MuSCAT4 (which is the same as 2m0 2M0-SCICAM-MUSCAT; FTS is a '2m0', MuSCAT4 is an instance of a 2M0-SCICAM-MUSCAT class of instruments) does have the link"
severity: major
note: |
  The originally reported template-leak gap IS fixed -- the modal for the linked event
  renders cleanly, no visible {# #} comment text. This is a new, separate finding: the
  2026-07-16 companion event (title "[EXPIRED] 2m0 2M0-SCICAM-MUSCAT", a telescope-CLASS
  label) has no CalendarEventMeta.run link, while the FTS/MuSCAT4 event (a specific
  telescope INSTANCE of that same class) does. Not yet diagnosed whether this is a
  reconciler resolution gap (Phase 29) for class-level vs instance-level event titles, or
  simply an unlinked row awaiting manual/attribution-queue action.

### 10. Re-verify — admin picker disambiguates near-identical companion events
expected: In Django admin, open CalendarEventMeta (standalone list or the CampaignRun
  inline's run-link picker). The 11 near-identical companion events (same title, different
  dates) are each labeled distinctly — e.g. "Verified/Fallback + event title + start
  date-time" — so picking the correct one for a given date is straightforward, and the
  field uses an autocomplete/search widget rather than a flat dropdown.
result: pass
note: |
  Confirmed on the "Owning campaign run" autocomplete picker: entries are distinctly
  labeled, e.g. "#38 3I/ATLAS (demo) | FTN/FLOYDS | 2025-08-01..2025-08-15 | F65"
  (CampaignRun.__str__: #pk | campaign | telescope/instrument | window | site). Initial
  confusion was with a different screen (repeated calendar EVENT titles across nights of
  the same multi-night run, which is expected, not a bug) -- the admin run-picker itself
  disambiguates correctly.

## Summary

total: 10
passed: 6
issues: 4
pending: 0
skipped: 0
blocked: 0
gaps: 5

## Gaps

- truth: "Staff can reach the Sites Needing Review queue to see that class-wide and space runs are no longer flagged"
  status: resolved
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
  status: resolved
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
  status: resolved
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

- truth: "Sites Needing Review is the first thing staff see on the approval queue when it's the only actionable table"
  status: failed
  reason: "User reported: Yes, I would still like the 'Sites Needing Review - action required' section to be at the top not the bottom"
  severity: minor
  test: 8
  root_cause: "src/templates/campaigns/approval_queue.html hardcodes Pending Review, then Recently Decided, then Sites Needing Review in the order the features were built, with no {% if %} conditional tying visual order to actionability. ApprovalQueueView.get_context_data (solsys_code/campaign_views.py:402-416) doesn't even pass pending_count/site_review_count into this view's context (those only exist on the separate CampaignListView, for the /campaigns/ nav banner) -- so there is no count-based adaptation logic to preserve. This is an unreviewed ordering oversight from incremental feature addition (D-07/27.1-03 appended the card at the bottom), not a functional constraint."
  artifacts:
    - path: "src/templates/campaigns/approval_queue.html"
      issue: "Sites Needing Review card hardcoded as third block (lines 14-23), below Pending Review (8-9) and Recently Decided (11-12); all three always render unconditionally"
  missing:
    - "Move the Sites Needing Review card to the top of {% block content %}, above Pending Review, in approval_queue.html"
  debug_session: ".planning/debug/approval-queue-section-order.md"

- truth: "An operator looking at an unlinked calendar event can tell why it's unlinked and what to do about it"
  status: failed
  reason: "User reported: one calendarevent for 2026-7-16 ('[EXPIRED] 2m0 2M0-SCICAM-MUSCAT') doesn't have a link to the CampaignRun but the FTS/MuSCAT4 event (a specific instance of the same 2m0 2M0-SCICAM-MUSCAT class) does"
  severity: minor
  test: 9
  root_cause: "NOT a linking bug -- diagnosis confirmed the reconciler and attribution scorer both work correctly. The two events are not reconciler companions of each other: pk=59 ('[EXPIRED] 2m0 2M0-SCICAM-MUSCAT') was created by sync_lco_observation_calendar.py directly from the raw LCO API request, a pipeline with no concept of CampaignRun that never sets CalendarEventMeta.run -- attribution is a human-confirmed step (Phase 28 attribution queue) that just hasn't happened for this row yet. pk=72 ('Didymos 2026: FTS/MuSCAT4...', url='RUN:1:2026-07-16') is a Phase 29 reconciler-owned event for the SAME CampaignRun (pk=1, telescope_instrument='FTS/MuSCAT4', site=E10) -- both of reconcile_run()'s branches call _link_event_to_run() unconditionally, so reconciler-owned events are always linked by construction; there is no class-level-vs-instance-level dispatch gap. Verified live: campaign_attribution.candidates_for_event() already scores pk=59 as a HIGH-band (0.8) candidate for run pk=1 (instrument similarity 0.92, 100% date overlap) -- the match is already available, just not yet confirmed by staff. The real gap is UX/discoverability: event_form.html's run-link block renders nothing for an unlinked event and gives no hint that a high-confidence attribution-queue candidate already exists, and its WR-03 template comment is stale (still claims no production code writes CalendarEventMeta.run, which stopped being true once Phase 29 shipped)."
  artifacts:
    - path: "src/templates/tom_calendar/partials/event_form.html"
      issue: "run-link block renders nothing for an orphan event even when a HIGH-band attribution-queue candidate already exists for it; header {% comment %} is stale post-Phase-29"
  missing:
    - "Surface a HIGH-band attribution candidate (with a link to confirm it in the attribution queue) in the modal's run-link block when CalendarEventMeta.run is unset"
    - "Refresh the stale WR-03 template comment to reflect that Phase 29's reconciler now writes CalendarEventMeta.run automatically for its own events"
    - "No action needed on campaign_reconciler.py or campaign_attribution.py -- both confirmed correct"
  debug_session: ".planning/debug/calendar-event-run-link-inconsistent.md"
