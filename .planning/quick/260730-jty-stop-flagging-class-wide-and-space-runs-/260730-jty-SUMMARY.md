---
quick_id: 260730-jty
description: Stop flagging class-wide and space runs as needing site review
date: 2026-07-30
status: complete
tasks_completed: 3
commits:
  - 945c507
  - 9371cb1
  - da2d166
merge_commit: 44d46d4
key_files:
  modified:
    - solsys_code/management/commands/import_campaign_csv.py
    - solsys_code/campaign_views.py
    - solsys_code/management/commands/repair_stale_campaign_run_sites.py
    - solsys_code/models.py
    - solsys_code/calendar_utils.py
    - solsys_code/tests/test_import_campaign_csv.py
    - solsys_code/tests/test_campaign_approval.py
    - solsys_code/tests/test_canonical_record_migration.py
    - solsys_code/tests/test_repair_stale_campaign_run_sites.py
    - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
    - docs/notebooks/pre_executed/fixtures/campaign_sample.csv
    - docs/runbooks/telescope_runs_calendar.rst
    - .planning/phases/27-the-canonical-run-record/27-REVIEW-FIX.md
  created:
    - solsys_code/migrations/0012_unflag_class_wide_campaignrun_site_review.py
---

# Quick Task 260730-jty — Summary

> **Note on provenance:** this SUMMARY.md was reconstructed by the orchestrator from the
> three task commits after the executor's original copy was lost — it was left uncommitted
> in the isolated worktree (per the workflow's "do not commit docs artifacts" constraint)
> and discarded when the worktree was force-removed during cleanup. The content below is
> derived from the commit messages and the merged diff, not from the executor's own prose.

## What changed

`CampaignRun.site_needs_review` now means only "site resolution failed **and** there is no
`telescope_class`". A non-blank `telescope_class` answers the question "why is there no
site" (D-06), so it is an answer, not a failure, and must never put a run in the staff
"Sites Needing Review" queue.

**Writers updated (all four):**

- `import_campaign_csv.py` — computes `telescope_class` *before* `needs_review`, using
  `site_resolution_failed and not telescope_class`. Deliberately **not** the literal
  `site is None` form: `resolve_site()` runs with `create_placeholder=True` and can return a
  placeholder `Observatory` (non-None site) with `needs_review=True`, which the literal form
  would have silently unflagged.
- `campaign_views.py` — the approve branch ANDs the flag with `not run.telescope_class`, and
  never adds `telescope_class` to `update_fields`. `_resolve_site()` needed no code change:
  its eligibility guard and conditional claim both key on `site_needs_review=True`, so
  classed rows become ineligible automatically once the flag is off.
- `repair_stale_campaign_run_sites.py` — skips class-carrying candidates entirely under a new
  `skipped_class_wide` counter (there is no site to repair).
- Migration `0012_unflag_class_wide_campaignrun_site_review.py` — unflags the four live
  classed rows (pk=26 JUICE/SPACE, 29 LCO 1m, 30 LCO 2m, 37 Generic 1m). One-way, with
  `reverse_code=noop`, matching the 0004/0005/0011 precedent. Needs no live-code import (the
  rule is a plain queryset), so phase 27's WR-06 constraint is satisfied trivially.

**Docstrings corrected** in `models.py` and `calendar_utils.py`: both previously read as a
mutual-exclusivity invariant ("a site-resolved run must never carry a telescope_class"), which
is what misled phase-27 code review into filing CR-01. They now restate D-06's framing and say
outright that `telescope_class` is **never cleared** when a site resolves.

## CR-01 recorded as REJECTED

`27-REVIEW-FIX.md` moves CR-01 from "skipped by explicit instruction" to **REJECTED**. Its
premise was invalid: a class-wide campaign (e.g. "LOOK Project Comet Followup 2026B" across
the LCO 1m0 network) legitimately keeps `telescope_class='1m0'` and `site=None` permanently,
with per-site detail on the linked `ObservationRecord`s via `CampaignRunObservation`
(CANON-04). CR-01's suggested fix would have destroyed a true campaign-level fact. The real
defect — classed runs sitting in the site-review queue — is what this task fixed instead.

## Tests

Regression coverage added at all four writers: the importer (classed row unflagged plus
summary count, class-less row still flagged, re-import stability), the approve path (classed
run stays unflagged and out of the review table, class-less control still flags and appears,
and the real `approve()` path proving `telescope_class` is never cleared when a site later
resolves), the repair command (class-carrying candidate skipped under its own counter), and
migration 0012 (aperture-class and SPACE rows unflagged, blank-class row still flagged).

Two pre-existing tests asserted the old buggy behaviour and were **revised in place, not
deleted**: `test_siteless_row_derives_telescope_class_from_instrument` and
`test_unresolvable_site_flags_needs_review_without_skipping_row`. The latter's `500@-999`
fixture derives `telescope_class='SPACE'` under D-11, so it was already an instance of the
flagging bug — a new sibling test covers the genuine no-class-signal failure case it had been
incorrectly standing in for.

**Verified after merge:** 238 tests pass across `test_import_campaign_csv`,
`test_campaign_approval`, `test_canonical_record_migration`,
`test_repair_stale_campaign_run_sites`, and `test_campaign_views`.

## Paired docs

- `import_campaign_csv_demo.ipynb` regenerated via `jupyter nbconvert --execute` against the
  migrated dev DB. One synthetic class-less fixture row ("Uma Unresolved") was added to
  `fixtures/campaign_sample.csv` so the notebook keeps evidence for the genuine
  site-resolution-failure branch — the existing class-carrying row ("Fay Review") is no longer
  flagged, which would otherwise have left the D-08/D-09 evidence row asserting nothing.
- `docs/runbooks/telescope_runs_calendar.rst` updated in three places for the corrected
  flagging rule and the repair command's new `skipped_class_wide` counter.

No other pre-executed notebook was re-executed (phase 27's IN-03 finding: the others were run
against the maintainer's populated dev DB, and a clean-DB re-run rewrites unrelated narrative
output).

## Outstanding

Migration 0012 is **not yet applied** to the local dev DB (`src/fomo_db.sqlite3`) — all four
rows are still flagged there. It is one-way, so applying it is the maintainer's call:

```bash
python manage.py migrate solsys_code
```

## Quality gates

`ruff check .` and `ruff format --check .` are unchanged from their pre-task baseline: 1
pre-existing D103 in `sync_gemini_observation_calendar_demo.ipynb`, and 3 pre-existing format
failures (`src/fomo/settings.py`, which carries an unrelated uncommitted local edit, plus two
`.planning/quick/260619-f7u/` scripts). No new violations in scope.
