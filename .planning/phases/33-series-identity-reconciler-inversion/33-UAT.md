---
status: testing
phase: 33-series-identity-reconciler-inversion
source: [33-VERIFICATION.md]
started: 2026-09-10T06:57:12Z
updated: 2026-09-10T06:57:12Z
---

## Current Test

number: 1
name: "View campaign ↗" lands on the highlighted run row (re-run of the 2026-09-09 test 2 after G-33-2 closure)
expected: |
  Open http://<dev-server>/calendar/?year=2026&month=7 (July 2026 — 15 campaign-attributed
  entries, all belonging to run pk=1; verified present in `src/fomo_db.sqlite3` by direct
  query, so NO fixture seeding is needed). Click one of the ⚑ entries, then click
  'View campaign ↗' in the 'Attributed campaign run' block.
  The pop-up opens (this half is now machine-proven by the Playwright tests), and the
  campaign table page then loads SCROLLED to that run's own row with the row visibly
  highlighted by the `tr:target` rule.
awaiting: user response

## Tests

### 1. "View campaign ↗" lands on the highlighted run row
expected: Open http://<dev-server>/calendar/?year=2026&month=7 (July 2026 — 15 campaign-attributed entries, all belonging to run pk=1; verified present in `src/fomo_db.sqlite3` by direct query, so NO fixture seeding is needed). Click one of the ⚑ entries, then click 'View campaign ↗' in the 'Attributed campaign run' block. The pop-up opens (this half is now machine-proven by the Playwright tests, so it should just work), and the campaign table page then loads SCROLLED to that run's own row with the row visibly highlighted by the `tr:target` rule. Note from the verifier: 33-09's fixture receipt says July 2025 – July 2026; the surviving attributed months are 2025-07 (26), 2025-08 (21), 2025-11 (2), 2026-01 (1), 2026-07 (15) — none in the current month, so navigate deliberately. Why human: browser anchor-scroll plus `:target` highlight rendering is real-browser behaviour no server-side or headless-assertion test observes (33-11 Task 2's deferred `<human-check>`; UAT G-33-2's third `missing:` item).
result: [pending]

### 2. Flagged judgment-tier prohibition: `source` printed in the lifecycle notebook's committed output
expected: Review the flagged prohibition from 33-VERIFICATION.md: `campaign_lifecycle_demo.ipynb` prints each run's `source` enum value into committed output, and the 33-09/33-05 prohibition text names `source` alongside the two contact fields. Either the `source` clause of that prohibition is narrowed (it is provenance metadata, not contact data, and the cells exist to demonstrate it), or the prints are dropped and the notebook re-executed. Why human: unverified-prohibition — a judgment-tier must-NOT carries no wired enforcement test; a model verdict is never authoritative. Pre-existing (byte-identical to the pre-wave notebook), so this does not block the phase on its own.
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

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

## Deferred Follow-Ups

- test: 1
  idea: "Legend label 'Classical schedule' really means 'no proposal' (NEUTRAL_SLOT_COLOR, Phase 9 D-05/D-06) and is misleading for queue-scheduled LCO network runs that simply lack a proposal code — rename to 'No proposal' or similar."
  deferred_at: 2026-09-09
- test: 1
  idea: "Telescope legend entries (.cal-legend-telescope) are display-only; only the two proposal swatches respond to the spotlight filter, which reads as 'toggling only works on some proposals'. Consider making telescope entries filterable, and making the single-select spotlight behaviour discoverable."
  deferred_at: 2026-09-09

## Gaps
