---
phase: 20-range-tbd-import-asset-aware-coverage-gap
plan: 04
subsystem: docs
tags: [jupyter, nbconvert, csv-import, demo-notebook, campaign-run]

# Dependency graph
requires:
  - phase: 20-range-tbd-import-asset-aware-coverage-gap
    plan: 02
    provides: "CampaignRun.original_obs_date_raw/window_needs_review fields + migration 0006, applied to the dev DB"
  - phase: 20-range-tbd-import-asset-aware-coverage-gap
    plan: 03
    provides: "parse_obs_window() 7-tuple never-raise range/TBD parsing; import_campaign_csv's window_needs_review counter and resolved-vs-TBD natural-key branching"
provides:
  - "campaign_sample.csv fixture extended with one date-range row (FTN/FLOYDS) and one genuinely-unparseable TBD row (VLT/X-shooter, non-blank Contact Person for a stable TBD natural key)"
  - "import_campaign_csv_demo.ipynb demonstration cell showing window_start/window_end for the range row (IMPORT-01) and window_needs_review + original_obs_date_raw for the TBD row (IMPORT-02), with real executed output committed"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Demo-notebook fixture additions keep the natural-key discipline explicit: a TBD row must carry a non-blank Contact Person so its (campaign, telescope_instrument, contact_person) key stays stable across re-executions of the notebook"

key-files:
  created: []
  modified:
    - docs/notebooks/pre_executed/fixtures/campaign_sample.csv
    - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb

key-decisions:
  - "Placed the new demonstration cell after the existing 'Inspect the imported CampaignRun rows' cell (not immediately after the import cell) so it reuses the notebook's already-established query style before introducing new rows"
  - "Row-count wording in three markdown cells (6 -> 8) and the requirement-coverage table (added IMPORT-01/IMPORT-02 rows) updated to match the new fixture; the import-cell narrative was written to describe both the fresh-DB (all created) and persistent-DB (unchanged) cases so it stays accurate regardless of which state the dev DB is in when a future re-run happens"

requirements-completed: [IMPORT-01, IMPORT-02]

coverage:
  - id: D1
    description: "campaign_sample.csv contains a date-range Obs. Date row and a genuinely-unparseable TBD Obs. Date row, both synthetic and matching the 14-column header"
    requirement: "IMPORT-01"
    verification:
      - kind: manual_procedural
        ref: "docs/notebooks/pre_executed/fixtures/campaign_sample.csv rows 7-8 (Gia Range / Ike Pending)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Demo notebook has an executed cell displaying window_start/window_end for the range row and window_needs_review=True + original_obs_date_raw for the TBD row"
    requirement: "IMPORT-02"
    verification:
      - kind: manual_procedural
        ref: "jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb (cell 10 committed output: window_start=2025-08-01, window_end=2025-08-15, window_needs_review=False for FTN/FLOYDS; window_start=None, window_end=None, window_needs_review=True, original_obs_date_raw='TBD pending Cycle 2' for VLT/X-shooter)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Notebook re-executes cleanly end-to-end against the migrated dev DB and stays idempotent on repeat runs"
    requirement: "IMPORT-01, IMPORT-02"
    verification:
      - kind: manual_procedural
        ref: "jupyter nbconvert --to notebook --execute --inplace exits 0 (run twice); import cell and idempotency-check cell both report 'created: 0, updated: 0, unchanged: 8, window_needs_review: 1' on the second and third executions"
        status: pass
    human_judgment: false

duration: 22min
completed: 2026-07-10
status: complete
---

# Phase 20 Plan 4: Range/TBD Import Demo Notebook Summary

**Extended `campaign_sample.csv` with a date-range and a TBD row, then regenerated `import_campaign_csv_demo.ipynb` with a new committed-output cell demonstrating IMPORT-01's resolved multi-night window and IMPORT-02's flagged-TBD-with-preserved-raw-text import path end-to-end against the migrated dev DB.**

## Performance

- **Duration:** 22 min
- **Started:** 2026-07-10T19:36:00Z
- **Completed:** 2026-07-10T19:57:52Z
- **Tasks:** 1
- **Files modified:** 2 (campaign_sample.csv, import_campaign_csv_demo.ipynb)

## Accomplishments

- Added `Gia Range` (`FTN/FLOYDS`, `Obs. Date = '2025-08-01 to 2025-08-15'`, `Site Code = F65`) exercising Plan 03's `_DATE_RANGE_FULL` full-date-range parsing, and `Ike Pending` (`VLT/X-shooter`, `Obs. Date = 'TBD pending Cycle 2'`, `Site Code = 309`, non-blank Contact Person) exercising the never-raise TBD catch-all, to `campaign_sample.csv` — both use distinct `Telescope / Instrument` values so they don't collide with the existing 6 rows' natural keys, and both reuse already-seeded, resolvable Site Codes (`F65`/`309`)
- Added a markdown+code cell pair after the existing "Inspect the imported CampaignRun rows" cell that fetches the two new `CampaignRun` rows by `(campaign, telescope_instrument)` and prints `window_start`/`window_end`/`window_needs_review` for the range row and the same plus `original_obs_date_raw` for the TBD row
- Updated three markdown cells' hardcoded row-count wording (`6` -> `8` rows/created/unchanged) and the fixture-description bullet list to mention the new range/TBD rows; added IMPORT-01/IMPORT-02 rows to the Summary cell's requirement-coverage table
- Regenerated the notebook via `jupyter nbconvert --to notebook --execute --inplace` against the already-migrated dev DB (`src/fomo_db.sqlite3`, migration 0006 confirmed applied via `showmigrations`) and committed it with real executed output — import-summary line confirms `window_needs_review: 1` (only the TBD row counts; the range row resolves to a window) and, on repeat execution, `created: 0, updated: 0, unchanged: 8` (idempotent)

## Task Commits

Each task was committed atomically:

1. **Task 1: Demonstrate range/TBD import in the demo notebook** - `38d88ea` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `docs/notebooks/pre_executed/fixtures/campaign_sample.csv` - added a date-range row and a TBD row, synthetic Name/Email style matching the existing 6 rows
- `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` - new demonstration cell for the range/TBD import path; updated row-count wording; requirement-coverage table extended; regenerated with committed executed output

## Decisions Made

- New demonstration cell placed after the existing generic inspection cell (not immediately after the import `call_command` cell) so it reuses the notebook's already-established `CampaignRun.objects.get(...)` query style rather than introducing a new pattern
- Import-cell narrative in the "Inspect the imported CampaignRun rows" markdown describes both the fresh-DB (`created: 8`) and persistent-DB (`unchanged: 8`) cases, since the actual first-execution-after-fixture-change output on this dev DB was a transient mixed `created: 2, unchanged: 6` that would go stale as soon as the notebook is re-run again — the final committed output (after a second `nbconvert --execute` pass) shows the steady-state `unchanged: 8` case, matching the wording

## Deviations from Plan

None - plan executed exactly as written. One iteration was needed on the markdown wording (see Decisions Made) after the first `nbconvert --execute` pass produced a transient `created: 2, unchanged: 6` result rather than the steady-state `unchanged: 8` — this was resolved by re-running `nbconvert --execute --inplace` a second time against the now-populated dev DB and rewriting the wording to match the steady-state output, not a plan deviation.

## Issues Encountered

None. `ruff check .` and `ruff format --check .` both show pre-existing findings in unrelated files (`sync_lco_observation_calendar_demo.ipynb`, `src/fomo/settings.py`, quick-task scripts) confirmed present on the base commit via `git stash`; no new findings introduced by this plan's two changed files.

## User Setup Required

None - no external service configuration required. Migration 0006 was already applied to the dev DB by Plan 02.

## Next Phase Readiness

- Phase 20 (all 4 plans) complete: asset-aware coverage-gap analysis (Plan 01), window-review schema fields (Plan 02), range/TBD parsing and import (Plan 03), and this plan's demo-notebook sync are all shipped and committed
- `import_campaign_csv_demo.ipynb` now truthfully demonstrates every `Obs. Date` shape the command handles (exact date, full range, compact range, blank, `'YYYY-MM-?'`, garbage free text) via its fixture and executed output
- No blockers for Phase 21 (site disambiguation + submitter contact opt-in)

---
*Phase: 20-range-tbd-import-asset-aware-coverage-gap*
*Completed: 2026-07-10*

## Self-Check: PASSED

Both modified files (`docs/notebooks/pre_executed/fixtures/campaign_sample.csv`,
`docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`) and both commits
(`38d88ea` task commit, `9aaa221` summary commit) verified present on disk and in
git log.
