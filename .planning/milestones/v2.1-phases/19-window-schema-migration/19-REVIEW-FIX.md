---
phase: 19-window-schema-migration
fixed_at: 2026-07-10T09:45:00Z
review_path: .planning/phases/19-window-schema-migration/19-REVIEW.md
iteration: 2
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 19: Code Review Fix Report

**Fixed at:** 2026-07-10T09:45:00Z
**Source review:** .planning/phases/19-window-schema-migration/19-REVIEW.md
**Iteration:** 2

**Note on iteration numbering:** this is the second `--fix` pass for this phase. Iteration 1
(documented in this same file's prior revision, superseded below) closed CR-01/WR-01/WR-02/WR-03
from the original `19-REVIEW.md`. This iteration fixes the 4 findings from the **re-review**
(`19-REVIEW.md`, reviewed `2026-07-10T09:00:00Z`), which independently confirmed all 4 iteration-1
fixes are correct and re-uses `WR-01`/`WR-02` as new finding IDs for two different, newly-surfaced
issues — these are NOT the same findings as iteration 1's `WR-01`/`WR-02`. `fix_scope: all` for
this pass, so both carried-over Info findings (`IN-01`, `IN-02`) are also in scope.

**Summary:**
- Findings in scope: 4 (`all` scope: WR-01, WR-02, IN-01, IN-02 — this review's numbering)
- Fixed: 4
- Skipped: 0

## Fixed Issues

### WR-01: New CheckConstraint (migration 0005) is completely untested — the exact crash scenario WR-02 was fixed to prevent has no regression coverage

**Files modified:** `solsys_code/tests/test_campaign_models.py`
**Commit:** `68d52fa`
**Applied fix:** Added `test_mismatched_window_start_end_pair_rejected_by_db` to
`TestCampaignRunWindowSchema`, asserting `CampaignRun.objects.create(...)` with `window_start` set
and `window_end=None` raises `IntegrityError` inside `transaction.atomic()` — directly exercising
the `campaign_run_window_start_end_null_together` `CheckConstraint` added by migration `0005`
(the WR-02 fix from iteration 1). Used the file's existing string-literal date convention
(`'2025-07-04'`) rather than the review's suggested `date(2025, 7, 4)` import, to match the
surrounding tests and avoid an unnecessary extra import. Did not additionally extend
`test_window_schema_migration.py`'s `migrate_to` to `0005` (the review's "optionally also" second
suggestion) — the new DB-level test above already gives direct, simpler coverage of the
constraint's actual runtime behavior at the model layer, which is where `claimed_dates()` (the
WR-02 fix this constraint protects) actually reads the data. The migration-history test class
specifically covers `0004`'s `RunPython` data-transform steps against the pre-0004 historical
schema; `0005`'s own new `RunPython` step (added below for this pass's WR-02) is a no-op against
current data and doesn't need the same historical-schema treatment to be meaningfully tested.
`python manage.py test solsys_code.tests.test_campaign_models.TestCampaignRunWindowSchema` (5/5)
and the full `python manage.py test solsys_code` suite (360/360, up from 359) pass.

### WR-02: Migration 0005 adds a CheckConstraint with no defensive data-cleanup step, unlike migration 0004's own CR-01 precedent in this same changeset

**Files modified:**
`solsys_code/migrations/0005_campaignrun_campaign_run_window_start_end_null_together.py`
**Commit:** `c24efdf`
**Applied fix:** Chose the review's first option (defensive `RunPython` cleanup, not just a
comment) to fully close the gap rather than merely document it, mirroring migration `0004`'s
`dedupe_resolved_window_collisions`/`dedupe_tbd_collisions` pattern exactly: added
`normalize_mismatched_window_pairs`, a `RunPython` step that finds any row with `window_start`
XOR `window_end` null (`Q(window_start__isnull=True, window_end__isnull=False) |
Q(window_start__isnull=False, window_end__isnull=True)`), logs a warning per row, and sets both
fields to `None` (collapsing it to fully-TBD) before the `AddConstraint` runs. Inserted it as the
first operation in `0005`, ahead of `AddConstraint`, matching the "dedup/cleanup RunPython before
the AddConstraint it protects" ordering already established as this changeset's own convention.
Added a module comment explaining the step is a no-op in the current codebase (0004's
`backfill_window_fields` already guarantees the invariant in the same deploy; no write path sets
the two fields independently) and exists purely as a safety net for a squashed/backported/
separate-deploy scenario. `python manage.py makemigrations solsys_code --check --dry-run` stays
clean (no schema drift from the added `RunPython`). `python manage.py test
solsys_code.tests.test_window_schema_migration solsys_code.tests.test_campaign_models` (15/15) and
the full `python manage.py test solsys_code` suite (360/360) pass.

### IN-01: Notebook narrative text still contradicts its own executed output

**Files modified:** `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`
**Commit:** `8b3942c`
**Applied fix:** Markdown-only wording fix, no re-execution needed (no code cell or output touched
— confirmed via `git diff --stat`, which shows only the two markdown `source` arrays changed).
Reworded markdown cell `c37e3856` to describe both possible outcomes ("Against a fresh, empty dev
DB this reports all 6 as `created`; against this notebook's persistent dev DB ... it instead
reports `unchanged: 6`") instead of asserting only the `created` case, matching the tone of the
existing "Idempotency note" cell `c32cae1e`. Reworded the CAMP-04 row of the summary table in cell
`90d785d4` similarly, replacing the inaccurate "Import cell summary (6 created) and re-run cell
summary (6 unchanged)" with wording that matches the actual executed output of code cell
`1093927e` (`unchanged: 6` on both the first and second `call_command` invocations against this
notebook's shared, already-populated dev DB). Verified the notebook JSON is still valid
(`json.load` succeeds, `nbformat: 4`) and that the diff is scoped to exactly the two markdown
cells' `source` fields.

### IN-02: Ground-vs-space calendar-projection branch still treats OCCULTATION/RADAR sites identically to OPTICAL

**Files modified:** `solsys_code/campaign_views.py`
**Commit:** `391b6bb`
**Applied fix:** Chose the review's second option (explicit code comment, not a behavior change) —
added a comment inside `CampaignRunDecisionView.post`'s ground-based `else` branch (lines ~360-367
pre-fix) documenting that the branch also catches `OCCULTATION_OBSTYPE` and `RADAR_OBSTYPE`, not
just `OPTICAL_OBSTYPE`, and that this is a deliberate simplification for this milestone (no
OCCULTATION/RADAR `Observatory` fixtures exist yet in `test_campaign_approval.py`, and those site
types' observing windows aren't governed by local darkness), naming the concrete follow-up
(`Observatory.OPTICAL_OBSTYPE` explicit scoping with no-projection fallback for
OCCULTATION/RADAR) for when real support is added. Deliberately did **not** change the actual
routing condition: this is Info-level (no crash, no data corruption — it silently produces the
wrong `CalendarEvent` window shape for a site type with zero fixtures/tests in this milestone), and
a real behavior change would require new `OCCULTATION_OBSTYPE`/`RADAR_OBSTYPE` `Observatory`
fixtures and a dedicated regression test per the review's own note — disproportionate scope for a
carried-over Info finding relative to a documentation fix that fully closes the "silent, undocumented"
part of the gap. `python -c "import ast; ast.parse(...)"`, `ruff check`, and `ruff format --check`
pass on the modified file; `python manage.py test solsys_code.tests.test_campaign_approval` (25/25,
unchanged) confirms the comment-only change causes no regression.

## Post-fix full-suite verification

- `python manage.py test solsys_code` — **360/360 passing** (359 pre-existing baseline + 1 new
  from WR-01).
- `python -m pytest` — 1/1 passing (unchanged; this suite only covers `tests/`, `src/`, `docs/`).
- `python manage.py makemigrations solsys_code --check --dry-run` — clean, no drift, after the new
  `RunPython` step added to migration `0005`.
- `ruff check .` / `ruff format --check .` — clean on every file touched by this fix pass. A
  pre-existing, unrelated baseline gap (5 `ruff check` findings in
  `solsys_code/management/commands/dump_obs_records.py`/settings/notebooks, 7 files needing
  `ruff format`) was independently confirmed present at the pre-fix commit (`65b74c4`) via direct
  comparison — none of it is new, and none of it is in a file this fix pass touched.
- Each commit above passed the project's full pre-commit hook chain (ruff lint, ruff format,
  Sphinx docs build, `manage.py test` via the `Run unit tests` hook) before being created.

## Skipped Issues

None — all 4 in-scope findings (WR-01, WR-02, IN-01, IN-02) were fixed.

---

_Fixed: 2026-07-10T09:45:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
