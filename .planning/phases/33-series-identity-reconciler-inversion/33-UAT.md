---
status: diagnosed
phase: 33-series-identity-reconciler-inversion
source: [33-VERIFICATION.md]
started: 2026-09-04T18:13:21Z
updated: 2026-09-09T23:03:12Z
---

## Current Test

[testing complete]

## Tests

### 1. Month-cell campaign chip legibility across proposal fill colours
expected: Open the month calendar (`/calendar/`) on a month containing at least one campaign-attributed all-day entry AND one attributed timed entry, across several different proposal fill colours. The ⚑ campaign chip is legible against every proposal fill (inherits the entry's foreground via `color: currentColor`), does not compress or clip in the timed entry's flex row (`flex-shrink: 0`), and hovering it shows the campaign name as a tooltip.
result: pass
note: |
  Verified in-browser on September 2026 (http://tlister-thinkmate.lco.gtn:8000/calendar/, screenshots
  2026-09-09 145308 and 152216). White ⚑ is legible on the neutral classical fill (#5a6268) beside every
  telescope stripe in use (#f8bfce, #5dea3e, #ffb370, #ffb09e, #8ac9ff); dark ⚑ is legible on timed
  entries against the white day cell (e.g. "⚑ Y22 1m0-SciCam… 07:55", Sep 5-7). No clipping in the timed
  flex row. Caveat: the "several proposal fill colours" clause is not exercisable in this DB — every
  attributed event has a blank proposal (all 17 September attributions are demo-campaign classical-schedule
  entries), so only the neutral fill was observed; the chip inherits the fill's WCAG-gated foreground via
  `color: currentColor`, so that clause holds by construction rather than by observation. Tooltip
  hover not checked because the click/hover path is blocked by the issue in test 2.

### 2. "View campaign ↗" lands on the highlighted run row
expected: Click a campaign-attributed calendar entry to open its pop-up, then click the "View campaign ↗" link in the "Attributed campaign run" block. The campaign table page loads scrolled to that run's own row (`id="run-{pk}"`), and the row is visibly highlighted by the `tr:target` rule in `src/templates/campaigns/campaignrun_table.html`. (The CR-01 note from the 2026-09-04 session is superseded: 33-06 moved the rule into `{% block additional_css %}` and it is now served.)
result: issue
reported: "None of the ⚑ LCO 1m0 Network (…) entries bring up anything when clicked on except when clicking on the colored swatch at the far left which links to either /targets/?targetlist__name=6 (light blue swatch) or /targets/?targetlist__name=9 (red swatch)."
severity: blocker

### 3. Decide on the abstained backstop truth (observation_group reverse-manager ordering)
expected: Review the `insufficient_spec` item in 33-VERIFICATION.md. Either add a held-out/property-based test (shuffle insertion order of several `CalendarEventMeta` rows sharing one `ObservationGroup`; assert the consuming code's outcome is unchanged) before Phase 34's projector writes these links, or explicitly accept the absence-by-grep evidence (no `Meta.ordering` on `CalendarEventMeta`, no production reader of `group.calendar_event_metas`).
result: pass
decision: accepted (option A)
note: |
  Absence evidence accepted: no production reader of `group.calendar_event_metas` exists and
  `CalendarEventMeta` declares no `Meta.ordering`, so a shuffled-insertion test would assert nothing
  today. Carried forward as a Phase 34 context requirement: if the projector adds any reader of that
  reverse accessor, it must either set an explicit ordering or add the shuffled-insertion test.

### 4. Review the 11 judgment-tier prohibitions
expected: Review the Prohibitions section of 33-VERIFICATION.md (LLM-judge verdicts, NON-AUTHORITATIVE). Pay particular attention to 33-05 P1 (`campaign_lifecycle_demo.ipynb` prints `contact_person=''` / `contact_email=''` for demo runs in a pre-existing cell — empty values only) and 33-05 P2 (`reconcile_campaign_runs_demo.ipynb` writes to and deletes rows in the real developer database `src/fomo_db.sqlite3`). Each prohibition is confirmed as still not violated, or the deviation is accepted.
result: issue
reported: "fix both prohibitions"
severity: minor
note: |
  Both flagged prohibitions are confirmed violated in practice and are to be corrected in the gap-closure
  plan rather than accepted:
  - 33-05 P2 / 33-08 P4 (notebook residue): both demo notebooks execute against the real dev DB. Observed
    residue in src/fomo_db.sqlite3: event pk 335 (RUN:59:2026-09-02) detached from run 59 and now offered in
    the staff attribution queue as a HIGH-band candidate; 19 September 2026 events across the "Reconciler
    Demo Campaign" (created 2026-09-04) and "Campaign Lifecycle Demo" (created 2026-09-09) campaigns, all
    with a blank proposal, rendering under the "Classical schedule" legend slot and cluttering the month view.
    Remedy: notebooks must execute against a scratch copy of the DB (copy the sqlite file to a temp path
    before execution and point DATABASES at it), and the existing residue must be cleaned up.
  - 33-05 P1 / 33-08 P5 (contact fields in output): campaign_lifecycle_demo.ipynb cell 36 prints
    contact_person='' / contact_email=''. Values are empty (no PII leaked) but the fields must be dropped
    from the print and the notebook re-executed.

## Decisions

- test: 3
  decision: "Accept absence-by-grep evidence for observation_group reverse-manager ordering (option A); carry a 'set ordering or add shuffled-insertion test if a reader is added' requirement into Phase 34 context."
  decided_at: 2026-09-09
- test: 4
  decision: "Correct both flagged prohibitions in the gap-closure plan: demo notebooks run against a scratch DB and residue is cleaned; contact_person/contact_email removed from campaign_lifecycle_demo.ipynb output."
  decided_at: 2026-09-09
- gap: CR-04 remedy (33-VERIFICATION.md Gap 1)
  decision: "Option B — human outranks machine. The reconciler sweep detaches only rows with no confirmed_by; a human-confirmed attribution is never cleared by an automated sweep. Leftover RUN:{pk}:{date} duplicates on a night remain Phase 35 SC 5's responsibility. Do not write CalendarEventDismissal rows on automated detach."
  decided_at: 2026-09-09

## Summary

total: 4
passed: 2
issues: 2
pending: 0
skipped: 0
blocked: 0

## Deferred Follow-Ups

- test: 1
  idea: "Legend label 'Classical schedule' really means 'no proposal' (NEUTRAL_SLOT_COLOR, Phase 9 D-05/D-06) and is misleading for queue-scheduled LCO network runs that simply lack a proposal code — rename to 'No proposal' or similar."
  deferred_at: 2026-09-09
- test: 1
  idea: "Telescope legend entries (.cal-legend-telescope) are display-only; only the two proposal swatches respond to the spotlight filter, which reads as 'toggling only works on some proposals'. Consider making telescope entries filterable, and making the single-select spotlight behaviour discoverable."
  deferred_at: 2026-09-09

## Gaps

- gap_id: G-33-2
  truth: "Clicking a campaign-attributed calendar entry opens its pop-up; 'View campaign ↗' in the 'Attributed campaign run' block loads the campaign table scrolled to that run's row with the row highlighted by tr:target."
  status: failed
  reason: "User reported: None of the ⚑ LCO 1m0 Network (…) entries bring up anything when clicked on except when clicking on the colored swatch at the far left which links to either /targets/?targetlist__name=6 (light blue swatch) or /targets/?targetlist__name=9 (red swatch)."
  severity: blocker
  test: 2
  root_cause: "FOMO's override of tom_calendar's month partial (src/templates/tom_calendar/partials/calendar.html) opens the modal with jQuery — hx-on::after-request=\"$('#cal-modal').modal('show');\" in three handlers (event entries, day cells, '+ New Event') — but tomtoolkit 3.0.1's tom_common/base.html loads Bootstrap 5.3.3 bundle + htmx + Alpine and no jQuery, so $ is undefined and the handler throws silently after every hx-get. Upstream tom_calendar 3.0.1's partial already uses a Bootstrap-5-native showModal(). Regressed with the tomtoolkit 3.0 pin (036d96e). The server side is correct: GET /calendar/update/314/ returns 200 with the 'View campaign ↗' link to /campaigns/9/#run-59. Tests (test_calendar_template.py:492-535) only assert the server-rendered modal HTML; nothing exercises the client-side open. Affects every click target on the calendar, not only attributed entries."
  artifacts:
    - path: "src/templates/tom_calendar/partials/calendar.html"
      issue: "three hx-on::after-request handlers call $('#cal-modal').modal('show') — jQuery API on a page with no jQuery"
  missing:
    - "Replace the three handlers with bootstrap.Modal.getOrCreateInstance(document.getElementById('cal-modal')).show() (or adopt upstream tom_calendar 3.0.1's showModal() helper)"
    - "A rendered-template assertion that the served partial contains no jQuery $( call (regression guard for the tomtoolkit 3.x / Bootstrap 5 base)"
    - "Re-run this UAT test in a browser: pop-up opens, 'View campaign ↗' scrolls to id=run-{pk} with the tr:target highlight visible"
  debug_session: ""
- gap_id: G-33-4
  truth: "Demo notebooks leave no residue in the developer database and print no contact fields (33-05 P1/P2, 33-08 P4/P5)."
  status: failed
  reason: "User decision: fix both flagged prohibitions rather than accept them. Observed residue: pk 335 detached from run 59 and offered as a HIGH-band queue candidate; 19 blank-proposal September 2026 demo events across two demo campaigns in src/fomo_db.sqlite3."
  severity: minor
  test: 4
  root_cause: "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb and campaign_lifecycle_demo.ipynb use the project's default DATABASES setting, so nbconvert --execute writes through to src/fomo_db.sqlite3; campaign_lifecycle_demo.ipynb cell 36 prints contact_person/contact_email."
  artifacts:
    - path: "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb"
      issue: "executes against the real dev DB; detaches pre-existing RUN:59:2026-09-02 and leaves demo rows behind"
    - path: "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb"
      issue: "executes against the real dev DB; cell 36 prints contact_person/contact_email"
  missing:
    - "Execute both notebooks against a scratch copy of the DB (copy src/fomo_db.sqlite3 to a temp path in a setup cell and point DATABASES at it, or gate on an env var honoured by settings/local_settings)"
    - "Remove contact_person/contact_email from cell 36's print"
    - "Clean the existing residue from src/fomo_db.sqlite3 (re-attach or delete pk 335 as appropriate; remove the demo-campaign events/runs seeded 2026-09-04 and 2026-09-09)"
    - "Re-execute both notebooks and commit outputs"
  debug_session: ""
