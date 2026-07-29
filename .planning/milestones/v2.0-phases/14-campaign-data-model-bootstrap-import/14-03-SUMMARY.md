---
phase: 14-campaign-data-model-bootstrap-import
plan: 03
subsystem: docs
tags: [jupyter, nbconvert, csv-fixture, campaign-coordination, pii-safety, demo-notebook]

# Dependency graph
requires:
  - phase: 14-02-csv-bootstrap-import
    provides: "campaign_utils.py (resolve_site/parse_obs_window/map_observation_status/insert_or_create_campaign_run) and the import_campaign_csv management command"
provides:
  - "Synthetic, PII-free campaign CSV fixture (docs/notebooks/pre_executed/fixtures/campaign_sample.csv) -- first fixtures/ subdirectory in the repo"
  - "Paired demo notebook for import_campaign_csv (docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb), executed end-to-end offline and committed with output"
affects: [15-per-campaign-table-view]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PII-safe synthetic fixture: hand-built CSV mirroring a real external data source's exact column shape, but every free-text value obviously fake, colocated in a new fixtures/ subdirectory under pre_executed/"
    - "Demo-notebook cell constructing model instances directly (bypassing the CLI command) to exercise a status transition the command itself never produces (approval_status lifecycle)"

key-files:
  created:
    - docs/notebooks/pre_executed/fixtures/campaign_sample.csv
    - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
  modified: []

key-decisions:
  - "Site Codes limited to 3 real MPC obscodes (F65/FTN, 309/ESO Paranal, 705/APO) the notebook seeds locally via update_or_create, so tier-1 resolution always hits and no live MPC API call happens anywhere in the notebook (D-11)"
  - "Fixture uses 6 rows (within the 5-10 spec) covering every required code path: clean multi-band imaging (griz), spectroscopy, a terminal Observation Status (cancelled), both of parse_obs_window's fallback branches (approximate-hour '~1 am' and fully-blank), Open to collaboration=yes, and a blank Site Code (site_needs_review)"
  - "Approval-lifecycle demo cell constructs CampaignRun rows directly via .objects.create() rather than through the CSV import (which always writes approved per D-03), since that is the only way to exercise pending_review -> approved and pending_review -> rejected"
  - "fixture_path is built from repo_root_path (already validated by the manage.py-existence assert in the Django-setup cell) rather than a CWD-relative literal, avoiding a hard dependency on the Jupyter kernel's working directory (see Deviations)"

patterns-established:
  - "Pattern: synthetic PII-free CSV fixture colocated at docs/notebooks/pre_executed/fixtures/, matching a real external data source's column shape exactly"
  - "Pattern: demonstrate a status lifecycle a management command never itself produces via direct model construction in a dedicated notebook cell"

requirements-completed: [CAMP-05]

coverage:
  - id: D1
    description: "The paired demo notebook executes end-to-end via jupyter nbconvert --execute against the synthetic fixture with no live network call, showing the real created/updated/unchanged/skipped/site_needs_review summary"
    requirement: "CAMP-05"
    verification:
      - kind: integration
        ref: "jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb (committed run: created=6, updated=0, unchanged=0, skipped=0, site_needs_review=1)"
        status: pass
    human_judgment: false
  - id: D2
    description: "No real PII is committed to git history -- fixture and notebook use only @example.com/@example.org contact info"
    requirement: "CAMP-05"
    verification:
      - kind: other
        ref: "grep/regex email scan of the committed fixture CSV and notebook JSON: only grace.lifecycle@example.com / hal.lifecycle@example.com and the fixture's @example.com/@example.org addresses found, no other addresses"
        status: pass
    human_judgment: false
  - id: D3
    description: "The notebook demonstrates the created/updated/skipped summary AND the pending_review -> approved/rejected approval_status lifecycle on synthetic data (D-03)"
    requirement: "CAMP-05"
    verification:
      - kind: integration
        ref: "docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb cells: import summary, inspection table, approval-lifecycle before/after prints, idempotent re-run (unchanged=6)"
        status: pass
    human_judgment: false

duration: ~25min
completed: 2026-07-03
status: complete
---

# Phase 14 Plan 03: Campaign CSV Bootstrap Import Demo Notebook Summary

**Synthetic, PII-free `campaign_sample.csv` fixture plus a paired, executed `import_campaign_csv_demo.ipynb` demonstrating the bootstrap import's created/updated/skipped summary, auto-target resolution, and the `pending_review` -> `approved`/`rejected` approval lifecycle -- all offline, satisfying CAMP-05.**

## Performance

- **Duration:** ~25 min
- **Tasks:** 2 completed
- **Files modified:** 2 created (fixture CSV, notebook), plus 1 phase-scoped `deferred-items.md`

## Accomplishments

- `docs/notebooks/pre_executed/fixtures/campaign_sample.csv` -- 6 hand-built synthetic rows in the real 3I/ATLAS sheet's exact 14-column order, every contact email `@example.com`/`@example.org`. Covers clean multi-band imaging (`griz`), a spectroscopy row (`Open to collaboration? = yes`), a terminal-mapping `Observation Status` (`cancelled - weather`), an approximate UT time (`~1 am`) and a fully blank UT time (both of `parse_obs_window`'s fallback branches), and a blank `Site Code` (the `site_needs_review` path with no Observatory match attempted). New `fixtures/` subdirectory confirmed not gitignored.
- `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` -- executed end-to-end via `jupyter nbconvert --to notebook --execute --inplace` and committed with real output. Seeds 3 `Observatory` rows (F65/FTN, 309/ESO Paranal, 705/APO) locally so every fixture row resolves at tier 1 with zero live MPC API calls; seeds a single-`Target` campaign `TargetList` via `NonSiderealTargetFactory` to demonstrate D-07 auto-target resolution; runs the import showing `created: 6, updated: 0, unchanged: 0, skipped: 0, site_needs_review: 1`; inspects the resulting rows (no PII columns printed); demonstrates the `pending_review` -> `approved` and `pending_review` -> `rejected` `approval_status` transitions on two synthetic rows built outside the CSV import; re-runs the import to prove idempotency (`unchanged: 6`).
- `.planning/phases/14-campaign-data-model-bootstrap-import/deferred-items.md` -- logs pre-existing `ruff check`/`ruff format --check` findings in unrelated files (out of scope per the deviation-rules scope boundary), following the Phase 04 precedent.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build the synthetic, PII-free campaign fixture CSV** - `b90ae05` (feat)
2. **Task 2: Author and execute the import_campaign_csv demo notebook** - `87eae88` (feat)

_Note: Task 2's first commit attempt was rejected by the `ruff-format` pre-commit hook (version skew between the dev venv's ruff 0.12.9 and pre-commit's pinned ruff v0.2.1 disagreed on one line-wrap); the hook's auto-fix was re-staged and the commit re-run successfully -- see Issues Encountered._

## Files Created/Modified

- `docs/notebooks/pre_executed/fixtures/campaign_sample.csv` - synthetic 6-row campaign CSV fixture (D-10)
- `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` - paired demo notebook (CAMP-05), executed with committed output
- `.planning/phases/14-campaign-data-model-bootstrap-import/deferred-items.md` - out-of-scope ruff findings log

## Decisions Made

- Reused real MPC obscodes (F65, 309, 705) with locally-seeded `Observatory` rows rather than inventing fictional codes, matching the existing `load_telescope_runs_demo.ipynb` convention of seeding accurate real-site data via `update_or_create` -- keeps the demo grounded in the same domain as the real 3I/ATLAS sheet while still making zero live network calls.
- Approval-lifecycle cell uses plain `CampaignRun.objects.create()` + direct field mutation (not `insert_or_create_campaign_run`) since the point is to demonstrate a state transition the create-or-update helper's own callers (the import command) never trigger.
- Re-executed the notebook fresh after `ruff format` touched the source, first deleting the `CampaignRun`/`TargetList`/`Target`/`Observatory` rows the earlier run had written to the local dev DB, so the committed output's "created: 6" narrative in the markdown cells matches an actual first-run execution rather than a stale "unchanged" result from a second run against already-seeded data.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Built the import's CSV path from `repo_root_path`, not the plan's illustrative CWD-relative literal**
- **Found during:** Task 2 (authoring the "run the import" cell)
- **Issue:** The plan's action text illustrates `call_command('import_campaign_csv', '--campaign', '3I/ATLAS (demo)', 'docs/notebooks/pre_executed/fixtures/campaign_sample.csv')`. The notebook's own Django-setup cell (mirroring `load_telescope_runs_demo.ipynb`) documents that the Jupyter kernel's working directory is `docs/notebooks/pre_executed/` itself (`parents[2]` comment). Passing that literal relative path as written would resolve to a doubled, nonexistent path (`docs/notebooks/pre_executed/docs/notebooks/pre_executed/fixtures/campaign_sample.csv`) and raise `CommandError`/`FileNotFoundError`, breaking `nbconvert --execute`.
- **Fix:** Built `fixture_path = repo_root_path / 'docs' / 'notebooks' / 'pre_executed' / 'fixtures' / 'campaign_sample.csv'` from the already-validated `repo_root_path` variable (asserted to exist via the `manage.py`-presence check two cells earlier) and passed `str(fixture_path)` to `call_command`, used for both the initial import and the idempotency re-run.
- **Files modified:** `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`
- **Verification:** `jupyter nbconvert --to notebook --execute --inplace` completes without error; `fixture_path.exists()` assertion passes; import summary shows `created: 6`.
- **Committed in:** `87eae88` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary to make the notebook actually executable; no scope creep -- the plan's snippet was illustrative shorthand, not a literal requirement.

## Issues Encountered

- **Pre-commit `ruff-format` version skew:** the dev venv's `ruff` (0.12.9) and pre-commit's pinned `ruff-pre-commit` hook (`rev: v0.2.1`) disagreed on how to wrap one multi-line `assert` statement in the notebook's Django-setup cell. The first `git commit` for Task 2 was rejected (hook modified the file, per the standard pre-commit "hook failed = no commit happened" behavior); the file was re-verified (valid JSON, outputs intact, `ruff format --check` clean against the hook's own pinned version after the auto-fix) and re-staged, and the commit succeeded on the next attempt. No `--no-verify` was used. Not logged as a plan deviation since it is routine pre-commit-hook remediation, not a functional bug.
- **`django.db.models.signals.post_save` on `Target`** already prints `Target post save hook: <name> created: <bool>` (fires twice per `NonSiderealTargetFactory.create()` call) -- confirmed pre-existing codebase behavior (also visible in the plain Django test run), not introduced by this plan; appears as expected stderr noise in the notebook's seed-cell output.

## User Setup Required

None - no external service configuration required. The real 3I/ATLAS CSV import (CAMP-04's live run) remains an explicit operator follow-up outside this plan, per Plan 02's summary.

## Next Phase Readiness

- CAMP-01 through CAMP-05 are all now satisfied: `CampaignRun` model + status vocabulary (Plan 01), `campaign_utils.py` + `import_campaign_csv` command (Plan 02), and this plan's synthetic-fixture-backed demo notebook (Plan 03) closing out CAMP-05.
- `./manage.py test solsys_code` (227 tests) passes; `ruff check .` / `ruff format --check .` are clean for every file this plan touches (pre-existing findings in unrelated files logged to `deferred-items.md`, not fixed, per scope boundary).
- Phase 15 (per-campaign table view) can now build against a `CampaignRun` schema whose end-to-end import path (including edge cases: blank site, blank/approximate UT time, terminal status mapping) has been exercised live, not just unit-tested in isolation.
- No blockers for Phase 15.

---
*Phase: 14-campaign-data-model-bootstrap-import*
*Completed: 2026-07-03*

## Self-Check: PASSED

All created files (`docs/notebooks/pre_executed/fixtures/campaign_sample.csv`,
`docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`,
`.planning/phases/14-campaign-data-model-bootstrap-import/deferred-items.md`, this
SUMMARY.md) confirmed present on disk. Both task commit hashes (`b90ae05`, `87eae88`)
confirmed present in git log.
