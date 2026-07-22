---
phase: quick-260722-hpw
plan: 01
subsystem: import
tags: [django, csv, management-command, campaign-run, tdd]

requires: []
provides:
  - "import_campaign_csv scans up to 10 leading rows for the real CSV header instead of
    assuming DictReader's row 1 is the header"
  - "Regression tests covering leading-comment-row skip, no-header-within-cap fast-fail,
    and header-beyond-cap fast-fail"
  - "Notebook demonstration of the leading-comment-row fix against a synthetic inline CSV"
affects: [import_campaign_csv, campaign-csv-import]

tech-stack:
  added: []
  patterns:
    - "Header discovery scan: read raw lines with f.readlines(), scan the first
      _MAX_HEADER_SCAN via csv.reader for the row containing every required column, then
      build csv.DictReader from lines[header_idx:] -- rather than assuming DictReader's
      row 1 is the header"

key-files:
  created: []
  modified:
    - solsys_code/management/commands/import_campaign_csv.py
    - solsys_code/tests/test_import_campaign_csv.py
    - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb

key-decisions:
  - "Split Task 1's tdd=true RED/GREEN cycle to drive out one representative regression
    test (test_skips_leading_comment_and_blank_rows_before_header) before implementing
    the fix, then added the other two regression tests specified by Task 2
    (test_no_header_row_within_scan_cap_raises_command_error,
    test_header_beyond_scan_cap_fails_fast) as their own commit -- avoids duplicating the
    same test across both the RED commit and Task 2's commit while still following the
    plan's TDD execution flow for Task 1's `tdd=\"true\"` frontmatter."
  - "Copied the gitignored, setuptools_scm-generated src/fomo/_version.py from the main
    repo checkout into this worktree so `./manage.py test`/`jupyter nbconvert` could
    import `fomo.__init__` (which reads `__version__` from that file) -- a worktree-local
    build artifact, not committed (already gitignored)."
  - "Ran `./manage.py migrate` in this worktree's dev SQLite DB (also gitignored) before
    executing the demo notebook, since it hadn't been migrated yet in this fresh
    worktree."

requirements-completed: [QUICK-260722-hpw]

duration: ~20min
completed: 2026-07-22
---

# Quick Task 260722-hpw: Fix import_campaign_csv leading-row header skip Summary

**`import_campaign_csv` now scans up to 10 leading rows via `csv.reader` for the real
14-column header before building the `csv.DictReader`, instead of assuming row 1 is the
header -- letting it consume the real 3I/ATLAS sheet export's free-text attribution row
and blank row unchanged.**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-07-22
- **Tasks:** 3/3 completed
- **Files modified:** 3

## Accomplishments

- `import_campaign_csv` no longer raises a false "missing required column(s)"
  `CommandError` when the real sheet export's leading attribution/blank rows precede the
  header -- it scans up to `_MAX_HEADER_SCAN` (10) leading rows for the first one
  containing all of `_REQUIRED_HEADERS`.
- The header-on-row-1 common case is unchanged: same fast-fail `CommandError` message
  shape, same "Row 2" first-data-row logging, all 50 pre-existing tests still pass.
- Fast-fail is preserved for genuinely malformed/wrong files: no header anywhere within
  the scan cap, or a valid header more than 10 rows in, both still raise `CommandError`
  with zero `CampaignRun` rows created (T-hpw-01/T-hpw-02 from the plan's threat model).
- The paired demo notebook (`import_campaign_csv_demo.ipynb`, required by CLAUDE.md's
  demo-notebook-companion convention) now has an executed cell demonstrating the fix
  against a synthetic inline CSV shaped like the real export.

## Task Commits

Each task was committed atomically:

1. **Task 1: Scan for the real header row before building the DictReader** (tdd) -
   `3dc3ebd` (test, RED) + `83d024c` (feat, GREEN)
2. **Task 2: Add regression tests for header discovery and no-header fast-fail** -
   `3b076b7` (test)
3. **Task 3: Add a notebook demonstration cell for the leading-comment-row skip** -
   `200f23f` (docs)

_Task 1 was `tdd="true"`: RED (`3dc3ebd`) added
`test_skips_leading_comment_and_blank_rows_before_header` and confirmed it failed against
the old row-1-only `DictReader` logic (`CommandError: ... missing required column(s):
['Telescope / Instrument', 'Obs. Date', 'UT Time Range']`); GREEN (`83d024c`) implemented
the header-scan fix and confirmed all 50 existing + 1 new test passed._

## Files Created/Modified

- `solsys_code/management/commands/import_campaign_csv.py` - Added `_MAX_HEADER_SCAN = 10`
  constant; replaced the row-1 `DictReader` assumption with a `csv.reader` scan over
  `lines[:_MAX_HEADER_SCAN]` that locates the first row containing all
  `_REQUIRED_HEADERS`, builds `csv.DictReader` from `lines[header_idx:]`, and raises
  `CommandError` (naming `_REQUIRED_HEADERS`) if none is found; updated row-number
  enumeration to `start=header_idx + 2` so "Row N" logging stays correct.
- `solsys_code/tests/test_import_campaign_csv.py` - Added 3 regression tests:
  `test_skips_leading_comment_and_blank_rows_before_header`,
  `test_no_header_row_within_scan_cap_raises_command_error`,
  `test_header_beyond_scan_cap_fails_fast`; imports `_MAX_HEADER_SCAN` from the command
  module.
- `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` - Added a markdown+code
  cell pair (after the range/TBD demonstration cells, before the approval-lifecycle
  section) building a self-contained inline CSV with a leading attribution row, a blank
  row, the real header, and two synthetic PII-free data rows; imports it under
  `'3I/ATLAS leading-comment demo'` and prints the resulting `CampaignRun` count.
  Regenerated executed output via `jupyter nbconvert --to notebook --execute --inplace`;
  the new cell's committed output shows `created: 2` / 2 `CampaignRun` rows.

## Decisions Made

- Split the TDD cycle so Task 1's RED/GREEN drove out one of the three planned regression
  tests, and Task 2 added the remaining two -- keeps each task's commit scoped to what it
  actually adds rather than re-adding the same test in both commits, while still
  satisfying Task 1's `tdd="true"` RED-then-GREEN requirement.
- Copied the worktree-local, gitignored `src/fomo/_version.py` (setuptools_scm build
  artifact) from the main repo checkout so Django/pytest could import `fomo` at all in
  this fresh worktree -- not a plan deviation, just fixing a broken test-runner
  environment (the file is gitignored, never committed).

## Deviations from Plan

None - plan executed exactly as written (see "Decisions Made" above for the one
TDD-sequencing adjustment, which stays within the plan's intent).

## Issues Encountered

- **Fresh worktree test environment:** `./manage.py test` initially failed with
  `ModuleNotFoundError: No module named 'src.fomo._version'` because this worktree's
  `src/fomo/` lacks the gitignored, setuptools_scm-generated `_version.py` that the main
  repo checkout already has. Fixed by copying that file from the main repo into the
  worktree (build artifact only, not a code change, not committed).
- **Fresh worktree dev DB:** the worktree's dev SQLite DB (also gitignored) hadn't been
  migrated yet, so the demo notebook's first `nbconvert --execute` run failed with
  `OperationalError: no such table: solsys_code_observatory_observatory`. Ran
  `./manage.py migrate` and re-executed successfully. As a side effect, the regenerated
  notebook's pre-existing cells now show `created: 8` (first run against a fresh DB)
  instead of the previously-committed `unchanged: 8` (which reflected a dev DB that
  already had accumulated data from earlier manual notebook runs) -- the idempotency-check
  cell (cell 16, unchanged) still correctly shows `unchanged: 8` on the second run,
  confirming idempotency end-to-end. This is expected re-execution behavior per the
  Task 3 `<verify>` command, not a functional regression.
- **Repo-wide `ruff format --check .` drift:** running the full quality gate at the repo
  root (`ruff check .` / `ruff format --check .`) surfaces pre-existing issues in 7 files
  unrelated to this task (other pre_executed notebooks, `src/fomo/settings.py`, two
  `.planning/quick/260619-f7u-.../verify_*.py` scripts) plus 5 pre-existing `ruff check`
  errors in `sync_lco_observation_calendar_demo.ipynb`. Confirmed via `git show
  97b5587:...import_campaign_csv_demo.ipynb | ruff format --check -` that this
  notebook's own drift (cell 3, the untouched Django-setup cell) predates this task.
  Logged in `deferred-items.md` per the scope-boundary rule, not fixed here. The 3 files
  this plan actually modified are clean under both `ruff check` and `ruff format --check`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `import_campaign_csv` is ready to consume the real 3I/ATLAS sheet export unchanged
  (leading attribution row + blank row before the header).
- No blockers. The repo-wide `ruff format` drift noted above is tracked in
  `deferred-items.md` for a future cleanup task, unrelated to this fix.

---
*Phase: quick-260722-hpw*
*Completed: 2026-07-22*

## Self-Check: PASSED

- FOUND: solsys_code/management/commands/import_campaign_csv.py
- FOUND: solsys_code/tests/test_import_campaign_csv.py
- FOUND: docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
- FOUND: .planning/quick/260722-hpw-fix-import-campaign-csv-to-skip-leading-/deferred-items.md
- FOUND commit: 3dc3ebd
- FOUND commit: 83d024c
- FOUND commit: 3b076b7
- FOUND commit: 200f23f
