---
status: resolved
trigger: "UAT gap diagnosis for Phase 16 Test 14: 'For the pending review entries, is there a way to hide blank entries/columns so that Actions column appears with less scrolling'"
created: 2026-07-04T15:30:32Z
updated: 2026-07-04T16:40:00Z
---

## Current Focus

hypothesis: CONFIRMED - see Resolution
test: n/a
expecting: n/a
next_action: none - diagnosis complete, hand off to /gsd-plan-phase --gaps

## Symptoms

expected: Approval queue (campaigns:approval_queue) reads clearly with sensible spacing in the Actions column.
actual: |
  User (staff, browser session): "For the pending review entries, is there a way to hide blank
  entries/columns so that Actions column appears with less scrolling". Reported during UAT Test 14
  (16-UAT.md), severity minor.
errors: None reported
reproduction: Open campaigns:approval_queue as staff user in a browser; look at Pending Review section.
started: Discovered during Phase 16 UAT (conversational testing session, 2026-07-04)

## Eliminated

(none - first hypothesis confirmed)

## Evidence

- timestamp: 2026-07-04T15:20:00Z
  checked: solsys_code/campaign_tables.py (CampaignRunTable, ApprovalQueueTable)
  found: |
    ApprovalQueueTable subclasses CampaignRunTable and adds one new column `actions`
    (orderable=False). Meta(CampaignRunTable.Meta): pass -- inherits the full 16-field
    Meta.fields tuple unchanged: telescope_instrument, site, obs_date, ut_start, ut_end,
    filters_bandpass, run_status, approval_status, open_to_collaboration,
    observation_details, weather, observation_outcome, publication_plans, comments,
    contact_person, contact_email. No `exclude`, no `sequence` override for the queue view.
  implication: ApprovalQueueTable renders the same 16 spreadsheet-parity columns as the
    Phase 15 read-path table, plus Actions -- nothing in this class limits/reorders columns
    for the approval-queue use case.

- timestamp: 2026-07-04T15:24:00Z
  checked: Empirical column order via `ApprovalQueueTable.base_columns.keys()` (Django shell,
    settings module src.fomo.settings)
  found: |
    CampaignRunTable columns (16, in order): telescope_instrument, site, obs_date, ut_start,
    ut_end, filters_bandpass, run_status, approval_status, open_to_collaboration,
    observation_details, weather, observation_outcome, publication_plans, comments,
    contact_person, contact_email.
    ApprovalQueueTable columns (17, in order): <same 16> + actions (last).
  implication: Confirms empirically (not just inferred from source) that Actions is the
    17th and LAST column. A user must scroll past all 16 data columns to reach Approve/Reject.

- timestamp: 2026-07-04T15:26:00Z
  checked: solsys_code/campaign_forms.py (CampaignRunSubmissionForm)
  found: |
    Only `campaign`, `contact_person`, `contact_email` are `required=True` (D-06 comment at
    line 6 confirms this is deliberate -- "Explicit required=False on every non-campaign
    field"). telescope_instrument, site_raw, obs_date, ut_start, ut_end, filters_bandpass,
    observation_details, open_to_collaboration, comments are all `required=False`.
    Additionally, `weather`, `observation_outcome`, `publication_plans` are not present in
    the submission form at all -- they are post-observation fields with no way to be
    populated until a run is actually observed.
  implication: A real-world PENDING_REVIEW row will structurally always have `weather`,
    `observation_outcome`, `publication_plans` blank (run hasn't happened yet), and commonly
    has several more of the 16 columns blank too (site/telescope/obs_date/times/filters/
    comments/observation_details left empty by the submitter). django-tables2 does not
    hide or collapse columns that are blank across all pending rows -- every declared column
    renders an (empty) cell of roughly its usual width for every row regardless of content.

- timestamp: 2026-07-04T15:28:00Z
  checked: django_tables2/templates/django_tables2/bootstrap4-responsive.html (installed
    package, CampaignRunTable.Meta.template_name, inherited unchanged by ApprovalQueueTable)
  found: |
    {% block table-wrapper %} wraps the rendered <table> in
    <div class="table-container table-responsive"> -- Bootstrap's standard horizontal-scroll
    container for tables wider than the viewport. There is no column-hiding/collapsing logic
    in this template; the "fix" for an overflowing table is exactly the horizontal scrollbar
    the user is complaining about.
  implication: The scrolling reported by the user is the intended fallback behavior of the
    inherited bootstrap4-responsive template when the table is wider than the viewport --
    not a bug in scroll mechanics, but a direct consequence of rendering 16 (often
    mostly-blank) columns plus Actions last.

## Resolution

root_cause: |
  ApprovalQueueTable (solsys_code/campaign_tables.py) reuses CampaignRunTable's full 16-column
  Meta.fields set verbatim (inherited via `class Meta(CampaignRunTable.Meta): pass`) and appends
  a 17th "actions" column at the very end -- confirmed empirically via
  `ApprovalQueueTable.base_columns.keys()`. That 16-column set was designed for Phase 15's
  spreadsheet-parity read path (D-09), where showing every field is the point. For the
  approval-queue's Pending Review section, several of those columns are structurally guaranteed
  blank for any not-yet-observed run (`weather`, `observation_outcome`, `publication_plans` --
  post-observation fields absent from CampaignRunSubmissionForm entirely), and most of the rest
  are merely optional on the submission form (`required=False` on everything except `campaign`,
  `contact_person`, `contact_email` per D-05/D-06), so real submissions commonly leave many of
  them empty too. django-tables2 renders every declared column for every row regardless of
  whether it's blank across the whole table -- there is no built-in or added logic in
  ApprovalQueueTable to omit/collapse columns that are all-blank for the pending queryset. The
  inherited `template_name = 'django_tables2/bootstrap4-responsive.html'` wraps the resulting
  wide table in a Bootstrap `table-responsive` div, which is the standard fallback for
  overflow-width tables -- it produces exactly the horizontal scrollbar the user experienced.
  Because Actions is the last of 17 columns, reaching Approve/Reject requires scrolling past
  all 16 (often mostly-blank) data columns first.
fix: |
  ApprovalQueueTable.Meta now sets exclude = ('weather', 'observation_outcome', 'publication_plans')
  and sequence = ('actions', 'approval_status', 'telescope_instrument', 'site', 'obs_date',
  'ut_start', 'ut_end', '...') so Actions leads the table and the three structurally-blank
  post-observation columns are dropped. CampaignRunTable is untouched (Phase 15 D-09 preserved).
verification: |
  TestApprovalQueueColumns (solsys_code/tests/test_campaign_approval.py) proves actions-first
  ordering, the triage column trim, and a D-09 regression guard confirming CampaignRunTable is
  unchanged. Full solsys_code suite (303 tests) and pytest suite pass; ruff clean.
files_changed:
  - solsys_code/campaign_tables.py
  - solsys_code/tests/test_campaign_approval.py
