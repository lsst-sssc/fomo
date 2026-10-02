---
created: 2026-10-02T22:09:12.279Z
title: Run pre-executed demo notebooks against a scratch DB copy, never the live dev DB
area: docs
severity: major
files:
  - docs/notebooks/pre_executed/*.ipynb
  - docs/notebooks/pre_executed/fixtures/campaign_sample.csv
  - src/fomo/settings.py:126-134
  - docs/runbooks/telescope_runs_calendar.rst
  - CLAUDE.md (paired-docs map)
---

## Problem

The pre-executed demo notebooks under `docs/notebooks/pre_executed/` are regenerated with
`jupyter nbconvert --to notebook --execute --inplace`, and (apart from the 261001-smo notebook,
whose setup cell `9084663a` sets `FOMO_DATABASE_PATH`) they run against the live dev database
`src/fomo_db.sqlite3`. Each regeneration therefore writes its fixtures into the real DB and never
removes them.

Concrete damage found on 2026-10-02: `import_campaign_csv_demo.ipynb` (loading
`fixtures/campaign_sample.csv`) and the Phase 27.1-04 / campaign-lifecycle runs left three
test-only campaigns behind — TargetList #4 "3I/ATLAS (demo)" (11 CampaignRuns, demo Target #143),
#5 "3I/ATLAS leading-comment demo" (2 runs) and #10 "WR06 tmp campaign" (1 run) — plus their 24
projected CalendarEvents (pks 104-126, 357) on the real calendar, including 15 fake "FTN FLOYDS"
nights in Aug 2025. Run #45 ("Uma Unresolved") was the single row behind the "Sites needing
review" banner on `/campaigns/approval-queue/`. Quick task 261002-l04 retires those rows; this
todo is the root-cause fix so it cannot happen again.

## Solution

Make every notebook in the CLAUDE.md paired-docs map re-executable any number of times from a
known starting state, leaving nothing behind in `src/fomo_db.sqlite3`:

- A shared setup cell (one helper, imported or copied into each notebook) that either snapshots a
  scratch copy of the DB — sqlite online `.backup` from a `mode=ro` URI, pointed at via
  `FOMO_DATABASE_PATH` (the single-setting `os.getenv()` hook already in `settings.py:126-134`,
  used by the 261001-smo notebook) — or builds a known fixture DB from scratch
  (`migrate` + the fixtures the notebook needs).
- A teardown cell at the end that removes the copy, so the notebook cleans up after itself.
- A guard in the setup cell that refuses to run when the resolved database is the live file.
- Update the runbook text that tells people how to regenerate the notebooks
  (`docs/runbooks/telescope_runs_calendar.rst` and the CLAUDE.md "Notebooks are regenerated via
  ..." sentence) so the scratch-copy step is the documented path.

Cover all notebooks in the CLAUDE.md paired-docs map, not just the campaign ones.
