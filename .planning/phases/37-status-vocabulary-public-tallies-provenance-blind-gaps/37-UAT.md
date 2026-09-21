---
status: diagnosed
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
source: [37-VERIFICATION.md]
started: 2026-09-19T17:20:00Z
updated: 2026-09-20T00:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Unused vs cancelled night render distinctly on the calendar
expected: An unused awarded night is muted/dashed with a `[U]` title token; a cancelled night shows `[C]` and retains its status ring; the two are distinguishable at a glance.
result: pass

### 2. `[U]` legend swatch toggles the unused filter on and off
expected: Clicking the `[U]` entry in the calendar legend filters the view to unused nights; clicking it again clears the filter. Behaves like the existing proposal swatches, and does not break the single-active-filter behaviour those already have.
result: pass

### 3. Calendar pop-up tally line reads sensibly
expected: Opening an event pop-up for an attributed run shows a tally line inside the attributed-run block, reading naturally (counts of linked groups/records and nights observed / scheduled / expired-or-failed / unused). A not-yet-known unused figure reads as a word, never a bare `0`.
result: pass

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
result: issue
reported: "Option (b)"
severity: major

## Summary

total: 4
passed: 3
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

- gap_id: G-37-4
  truth: "The campaign-list roll-up strip's [U] total agrees with the sum of the Progress cells directly beneath it on the same page, with no time-bounded disagreement."
  status: failed
  reason: "User reported: Option (b)"
  severity: major
  test: 4
  root_cause: "get_or_compute_rollup() (campaign_tally.py:613) caches the WHOLE campaign_rollup() dict, unused_* fields included, so the roll-up's unused figure is frozen for TALLY_CACHE_TTL_SECONDS. The per-run path deliberately does not: tallies_for_runs()/get_or_compute_tally() cache WITHOUT the unused_* keys and re-apply them live through _apply_unused_fields() on every call, including on a cache hit (campaign_tally.py:269/278/303/308/343/349). build_rollup_cache_key() folds in campaign_records_version(), so the strip IS live to a record-driven narrowing; the residual is confined to the two unused_* drivers that do not move records_version -- an awarded night elapsing past its projected sunrise (is_unused_allocation_night() compares end_time against timezone.now()) and a staff run_status edit into a RUN_STATUS_MARKER status."
  artifacts:
    - path: "solsys_code/campaign_tally.py"
      issue: "get_or_compute_rollup() caches unused_* with the rest of the roll-up instead of recomputing that split live, diverging from get_or_compute_tally()'s cache-without-unused contract"
    - path: "solsys_code/campaign_tally.py"
      issue: "campaign_rollup() computes the unused sum inline; the live split needs it factored out so the cached and live halves can be assembled separately"
    - path: "solsys_code/tests/test_campaign_views.py:1170"
      issue: "test_campaign_list_query_count_bound_with_three_campaigns asserts a zero-marginal-query bound that a live per-run allocation-event count on the anonymous campaign list page will exceed"
  missing:
    - "Factor campaign_rollup()'s unused sum into a campaign-level counterpart of _apply_unused_fields() that adds each run's exact still-standing allocation-night count plus the D-06 proposal-derived estimate once per distinct non-blank proposal_code"
    - "Cache the roll-up WITHOUT the unused_* keys in get_or_compute_rollup() and apply the live split on every call, on cache hits too -- mirroring get_or_compute_tally()"
    - "Relax test_campaign_list_query_count_bound_with_three_campaigns to the new bound, keeping it a bound (asserting it does not scale per campaign) rather than deleting it"
    - "Update the docstrings on campaign_rollup()/get_or_compute_rollup() that currently promise the TALLY_CACHE_TTL_SECONDS staleness this change removes"
  debug_session: ""

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
- Segment labels in the tally line are not pluralized. `event_form.html:198` pluralizes the
  `groups`/`records` prefix but the segment loop at `event_form.html:211` renders
  `{{ segment.count }} {{ segment.label }}` with no `|pluralize`. Three of the four labels are
  state adjectives so it never shows; `Unused awarded night`
  (`status_vocabulary.py:83`) is the only count noun, so a count of 15 reads
  `15 Unused awarded night`. Same label feeds the campaign table's Progress column and the
  campaign-list roll-up strip, so the singular reads wrong in all three surfaces. Observed
  during UAT test 3 and accepted as non-blocking by the developer; worth a follow-up quick task.
- The pop-up's attributed-run header ends in `run.get_run_status_display()` (the staff-set
  `CampaignRun.run_status`, `calendar_display_extras.py:546`) while the tally line beneath it
  carries a derived `[O] N Observed` count. A run whose staff status is `Observed` can sit above
  `[O] 0 Observed`, so one line uses the same word for an editorial status and a computed count.
  Not a data inconsistency; noted as a readability observation, accepted as-is during UAT test 3.
