---
status: testing
phase: 33-series-identity-reconciler-inversion
source: [33-VERIFICATION.md]
started: 2026-09-04T18:13:21Z
updated: 2026-09-04T18:13:21Z
---

## Current Test

number: 1
name: Month-cell campaign chip legibility across proposal fill colours
expected: |
  Open the month calendar (`/calendar/`) on a month containing at least one campaign-attributed
  all-day entry AND one attributed timed entry, across several different proposal fill colours.
  The ⚑ campaign chip is legible against every proposal fill (it inherits the entry's own
  foreground via `color: currentColor`), does not compress or clip in the timed entry's flex row
  (`flex-shrink: 0`), and hovering it shows the campaign name as a tooltip.
awaiting: user response

## Tests

### 1. Month-cell campaign chip legibility across proposal fill colours
expected: Open the month calendar (`/calendar/`) on a month containing at least one campaign-attributed all-day entry AND one attributed timed entry, across several different proposal fill colours. The ⚑ campaign chip is legible against every proposal fill (inherits the entry's foreground via `color: currentColor`), does not compress or clip in the timed entry's flex row (`flex-shrink: 0`), and hovering it shows the campaign name as a tooltip.
result: [pending]

### 2. "View campaign ↗" lands on the highlighted run row
expected: Click a campaign-attributed calendar entry to open its pop-up, then click the "View campaign ↗" link in the "Attributed campaign run" block. The campaign table page loads scrolled to that run's own row (`id="run-{pk}"`), and the row is visibly highlighted by the `tr:target` rule in `src/templates/campaigns/campaignrun_table.html`. Note: code review finding CR-01 reports that this `<style>` block sits outside every `{% block %}` in the extending template and is discarded by Django — expect this test to FAIL until CR-01 is fixed.
result: [pending]

### 3. Decide on the abstained backstop truth (observation_group reverse-manager ordering)
expected: Review the `insufficient_spec` item in 33-VERIFICATION.md. Either add a held-out/property-based test (shuffle insertion order of several `CalendarEventMeta` rows sharing one `ObservationGroup`; assert the consuming code's outcome is unchanged) before Phase 34's projector writes these links, or explicitly accept the absence-by-grep evidence (no `Meta.ordering` on `CalendarEventMeta`, no production reader of `group.calendar_event_metas`).
result: [pending]

### 4. Review the 11 judgment-tier prohibitions
expected: Review the Prohibitions section of 33-VERIFICATION.md (LLM-judge verdicts, NON-AUTHORITATIVE). Pay particular attention to 33-05 P1 (`campaign_lifecycle_demo.ipynb` prints `contact_person=''` / `contact_email=''` for demo runs in a pre-existing cell — empty values only) and 33-05 P2 (`reconcile_campaign_runs_demo.ipynb` writes to and deletes rows in the real developer database `src/fomo_db.sqlite3`). Each prohibition is confirmed as still not violated, or the deviation is accepted.
result: [pending]

## Summary

total: 4
passed: 0
issues: 0
pending: 4
skipped: 0
blocked: 0

## Gaps
