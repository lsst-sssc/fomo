---
phase: 19-window-schema-migration
fixed_at: 2026-07-10T08:30:00Z
review_path: .planning/phases/19-window-schema-migration/19-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 19: Code Review Fix Report

**Fixed at:** 2026-07-10T08:30:00Z
**Source review:** .planning/phases/19-window-schema-migration/19-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 4 (critical_warning scope: CR-01, WR-01, WR-02, WR-03; IN-01/IN-02 excluded as Info-level)
- Fixed: 4
- Skipped: 0

## Fixed Issues

### CR-01: Migration's dedup step only covers the TBD branch, not the structurally identical resolved-window collision the backfill also creates

**Files modified:** `solsys_code/migrations/0004_campaignrun_window_schema.py`
**Commit:** `4b1fc5c`
**Applied fix:** Added a new `dedupe_resolved_window_collisions` `RunPython` step, symmetric to the
existing `dedupe_tbd_collisions`, that deletes duplicate resolved-window rows (keeping the lowest
pk, logging a warning for each deletion) keyed on `(campaign, telescope_instrument, window_start,
window_end)`. Inserted it into `operations` immediately after `dedupe_tbd_collisions` and before
`RemoveConstraint` — i.e. after `backfill_window_fields` populates `window_start`/`window_end` but
before the `unique_campaign_run_resolved_window` `AddConstraint` that would otherwise raise
`IntegrityError` against real leftover same-night, different-`ut_start` rows. Renumbered the
in-file step comments (module docstring and inline `# N:` markers) to stay accurate with three
`RunPython` steps instead of two.

**Verification note:** The suggested manual "copy the dev DB and re-run migrate" dry run
(previously used for Plan 01/04's smoke tests) was not repeated here: migration 0004 has already
been applied for real to the shared dev DB (`src/fomo_db.sqlite3`, per 19-04-SUMMARY.md), no
pre-migration snapshot/backup of that file was left behind to restore from, and reversing 0004 in
place would be lossy (Django's `RemoveField` reversal re-adds empty columns; it does not restore
the original `obs_date`/`ut_start`/`ut_end` data). Rather than risk mutating the team's shared,
unbacked-up dev DB, verification was done instead via the new automated migration test added for
WR-03 below (`solsys_code/tests/test_window_schema_migration.py`), which seeds the exact CR-01
collision scenario (two rows sharing `campaign`/`telescope_instrument`/`obs_date` with distinct
non-null `ut_start`) against a real pre-0004 historical schema state and asserts the new dedup step
collapses them to one survivor without raising `IntegrityError` — a permanent, repeatable
regression check rather than a one-off manual dry run. Also ran `python manage.py makemigrations
solsys_code --check --dry-run` (clean, no drift) and the full `python manage.py test solsys_code`
suite (359/359 passing, up from 355 — see WR-03) after this change.

### WR-01: CSV importer's natural-key lookup omits `window_end`, contradicting the model's own documented resolved-window contract

**Files modified:** `solsys_code/management/commands/import_campaign_csv.py`
**Commit:** `089e0a0`
**Applied fix:** Added `'window_end': obs_date` to the `lookup` dict passed to
`insert_or_create_campaign_run` (matching `models.py`'s documented resolved-window natural key) and
removed the now-redundant `'window_end': obs_date` entry from `fields`, per the review's suggested
fix. `python manage.py test solsys_code.tests.test_import_campaign_csv` (33/33) still passes
unchanged.

### WR-02: No DB-level invariant that `window_start`/`window_end` are null together; `claimed_dates()` will crash on a mismatched row

**Files modified:** `solsys_code/models.py`, `solsys_code/campaign_gap.py`,
`solsys_code/migrations/0005_campaignrun_campaign_run_window_start_end_null_together.py` (new)
**Commit:** `d1adf4e`
**Applied fix:** Applied both halves of the review's suggested fix, not just one: (1) added a
`CheckConstraint` (`campaign_run_window_start_end_null_together`) to `CampaignRun.Meta.constraints`
enforcing `window_start`/`window_end` are both `NULL` or both set, generated via `python manage.py
makemigrations solsys_code` into a new migration `0005`; and (2) defensively guarded
`campaign_gap.claimed_dates()`'s loop to also bucket a mismatched (`window_start` set,
`window_end` NULL, or vice versa) row into `undated_runs` instead of raising `TypeError` on the
`(run.window_end - run.window_start)` subtraction, matching `render_window_start()`'s existing
graceful-degradation behavior. `python manage.py makemigrations solsys_code --check --dry-run` is
clean after generating migration `0005`. `python manage.py test solsys_code.tests.test_campaign_models
solsys_code.tests.test_campaign_gap` (33/33) passes.

### WR-03: Migration's `RunPython` backfill/dedup logic has zero automated regression coverage

**Files modified:** `solsys_code/tests/test_window_schema_migration.py` (new)
**Commit:** `32093cf`
**Applied fix:** Added `TestWindowSchemaMigrationDataTransform`, a hand-rolled
`django.test.TransactionTestCase` using `django.db.migrations.executor.MigrationExecutor` to migrate
to the pre-0004 historical schema (`apps.get_model` for both `CampaignRun` and `TargetList`), seed
rows exercising both the CR-01 resolved-window collision scenario (two rows sharing
`campaign`+`telescope_instrument`+`obs_date` with distinct non-null `ut_start`), a non-colliding
resolved row (regression guard: dedup must not delete rows that don't actually collide), and a
TBD-branch duplicate pair (regression guard: the pre-existing `dedupe_tbd_collisions` step keeps
working alongside the new one) — then migrates forward through 0004 and asserts the outcome
directly against the migration's own `RunPython` functions (4 new tests, all passing). This
directly follows the review's suggested pattern ("`MigratorTestCase`-style ... using historical
model state via `apps.get_model` inside a `django.test.TransactionTestCase`") and also serves as the
CR-01 verification artifact described above.

## Post-fix full-suite verification

- `python manage.py test solsys_code` — **359/359 passing** (355 pre-existing + 4 new from WR-03).
- `python manage.py makemigrations solsys_code --check --dry-run` — clean, no drift, after both new
  migrations (`0004` edit is data-only/no schema drift; `0005` is a new schema migration for WR-02's
  `CheckConstraint`).
- `ruff check` / `ruff format --check` — clean on every file touched by this fix pass.
- Each commit above passed the project's full pre-commit hook chain (ruff lint, ruff format, Sphinx
  docs build, `manage.py test` via the `Run unit tests` hook) before being created.

## Skipped Issues

None — all 4 in-scope findings (CR-01, WR-01, WR-02, WR-03) were fixed. IN-01 and IN-02 were
Info-level and out of scope for this `critical_warning` fix pass.

---

_Fixed: 2026-07-10T08:30:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
