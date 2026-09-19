---
status: testing
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
source: [37-VERIFICATION.md]
started: 2026-09-19T17:20:00Z
updated: 2026-09-19T17:20:00Z
---

## Current Test

number: 1
name: Calendar reads correctly by eye — unused night muted and `[U]`-prefixed, cancelled night keeps `[C]` plus its ring
expected: |
  On the calendar, an awarded night that came and went with nothing scheduled or observed
  renders visibly different from a realised night: muted (opacity 0.55) with a dashed border,
  and its title text carries a leading `[U]` token. A staff-cancelled run night still renders
  `[C]` and keeps its status ring. The two states are distinguishable at a glance without
  hovering.

  All markup, CSS, template tags and JS are verified present and unit-tested (395 scoped tests
  green). Only the by-eye half is outstanding — this is the deferred `<human-check>` block from
  37-07-PLAN.md:307.
awaiting: user response

## Tests

### 1. Unused vs cancelled night render distinctly on the calendar
expected: An unused awarded night is muted/dashed with a `[U]` title token; a cancelled night shows `[C]` and retains its status ring; the two are distinguishable at a glance.
result: [pending]

### 2. `[U]` legend swatch toggles the unused filter on and off
expected: Clicking the `[U]` entry in the calendar legend filters the view to unused nights; clicking it again clears the filter. Behaves like the existing proposal swatches, and does not break the single-active-filter behaviour those already have.
result: [pending]

### 3. Calendar pop-up tally line reads sensibly
expected: Opening an event pop-up for an attributed run shows a tally line inside the attributed-run block, reading naturally (counts of linked groups/records and nights observed / scheduled / expired-or-failed / unused). A not-yet-known unused figure reads as a word, never a bare `0`.
result: [pending]

### 4. Product decision — roll-up strip's unused figure staleness
expected: |
  A decision, not a visual check. The campaign-list roll-up strip's `[U]` total can disagree
  with the sum of the Progress cells directly beneath it on the same page, for up to 1 hour.

  Scope is narrower than first reported: `build_rollup_cache_key()` DOES fold in
  `campaign_records_version()`, so the strip is fully live to a projector narrowing — the
  roadmap's own trigger for "updating as the projector narrows". The residual is confined to
  the `unused_*` sub-figure under a purely time-driven or staff-`run_status`-driven change,
  neither of which moves `records_version`.

  Choose one:
  (a) Accept the hour-bounded disagreement — no code change.
  (b) Give `campaign_rollup()`/`get_or_compute_rollup()` the same live-unused split the
      per-run tally now has. This requires relaxing
      `test_campaign_list_query_count_bound_with_three_campaigns`'s zero-marginal-query
      bound on the anonymous campaign list page.
result: [pending]

## Summary

total: 4
passed: 0
issues: 0
pending: 4
skipped: 0
blocked: 0

## Gaps

None. The verifier scored 5/5 must-haves with no gaps or blockers; these four items are
human-judgement checks, not defects.

## Notes carried forward (not blocking this phase)

- `solsys_code/management/commands/backfill_lco_observations.py:303` still holds a bare
  `block.get('state') == 'COMPLETED'` — the same concept `calendar_utils.resolve_placement_block()`
  converted to `OCSState.COMPLETED`. That file is in no Phase 37 plan's `files_modified`, so it is
  outside declared scope: a residual of STATUS-01's "exactly one spelling" ambition rather than a
  Phase 37 regression. Worth a follow-up quick task.
- Plan 37-07's commit `df40929` says "regenerate the four pre-executed notebooks" but three
  changed. The verifier confirmed via `git diff` that Phase 37's `observation_projector.py` change
  is a pure refactor for the facilities `project_observation_calendar_demo.ipynb` exercises
  (`PROJECTED_FACILITIES = ('LCO', 'SOAR')`; marker values byte-identical, only their source
  moved), and CLAUDE.md's paired-docs rule explicitly carves out pure refactors. A commit-log
  inaccuracy, not a stale artifact.
