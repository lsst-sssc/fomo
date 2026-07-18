---
phase: 19-window-schema-migration
reviewed: 2026-07-10T09:00:00Z
depth: deep
files_reviewed: 16
files_reviewed_list:
  - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
  - solsys_code/campaign_forms.py
  - solsys_code/campaign_gap.py
  - solsys_code/campaign_tables.py
  - solsys_code/campaign_utils.py
  - solsys_code/campaign_views.py
  - solsys_code/management/commands/import_campaign_csv.py
  - solsys_code/migrations/0004_campaignrun_window_schema.py
  - solsys_code/migrations/0005_campaignrun_campaign_run_window_start_end_null_together.py
  - solsys_code/models.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/tests/test_campaign_gap.py
  - solsys_code/tests/test_campaign_models.py
  - solsys_code/tests/test_campaign_submission.py
  - solsys_code/tests/test_campaign_views.py
  - solsys_code/tests/test_import_campaign_csv.py
  - solsys_code/tests/test_window_schema_migration.py
findings:
  critical: 0
  warning: 2
  info: 2
  total: 4
status: issues_found
---

# Phase 19: Code Review Report (re-review after --fix)

**Reviewed:** 2026-07-10T09:00:00Z
**Depth:** deep
**Files Reviewed:** 16
**Status:** issues_found

## Summary

This is a re-review after the `--fix` pass (commits `4b1fc5c`, `089e0a0`, `d1adf4e`, `32093cf`)
closed CR-01, WR-01, WR-02, and WR-03 from the prior `19-REVIEW.md`. All four fixes were
independently traced through the current source (not just trusted from `19-REVIEW-FIX.md`), and
each does what it claims:

- **CR-01** (migration dedup gap): `solsys_code/migrations/0004_campaignrun_window_schema.py` now
  contains `dedupe_resolved_window_collisions`, correctly sequenced after
  `backfill_window_fields`/`dedupe_tbd_collisions` and before `RemoveConstraint`/`RemoveField`/
  `AddConstraint`, so it runs before the `unique_campaign_run_resolved_window` `AddConstraint` that
  would otherwise choke on real leftover same-night, different-`ut_start` rows.
- **WR-01** (CSV importer lookup key): `import_campaign_csv.py`'s `insert_or_create_campaign_run`
  call now includes `'window_end': obs_date` in the `lookup` dict (removed from `fields`), matching
  `models.py`'s documented resolved-window natural key.
- **WR-02** (missing DB invariant): `models.py` now has the
  `campaign_run_window_start_end_null_together` `CheckConstraint`, mirrored in the new migration
  `0005`, and `campaign_gap.py:claimed_dates()` now buckets a mismatched (`window_start` XOR
  `window_end` null) row into `undated_runs` instead of crashing on the date-arithmetic
  subtraction.
- **WR-03** (no migration regression coverage): `test_window_schema_migration.py` is new, uses
  `MigrationExecutor` against the pre-0004 historical schema, and asserts the CR-01
  resolved-window collision scenario collapses to one survivor, a non-colliding row survives, and
  the pre-existing TBD dedup keeps working.

The two files new to this changeset since the last review — migration `0005` and
`test_window_schema_migration.py` — were reviewed independently for the first time. Both are
functionally correct, but together they leave a real coverage gap: the DB-level invariant WR-02
just added (the exact thing that stops `claimed_dates()` from crashing) is not exercised by any
test, and the migration that adds it has no defensive data-cleanup step of the kind CR-01 just
established as this changeset's own precedent. Neither is a functional bug in the shipped
behavior today; both are robustness/coverage gaps. The two carried-over Info findings from the
prior review were independently re-confirmed against current file contents and still hold,
unchanged.

## Narrative Findings (AI reviewer)

### Warnings

#### WR-01: New CheckConstraint (migration 0005) is completely untested — the exact crash scenario WR-02 was fixed to prevent has no regression coverage

**File:** `solsys_code/tests/test_window_schema_migration.py` (whole file); also
`solsys_code/tests/test_campaign_models.py`, `solsys_code/tests/test_campaign_gap.py`
**Issue:**

`test_window_schema_migration.py` (new, added for WR-03) only exercises migration `0004`'s three
`RunPython` steps (`backfill_window_fields`, `dedupe_tbd_collisions`,
`dedupe_resolved_window_collisions`) against the pre-0004 historical schema; `migrate_to =
[('solsys_code', '0004_campaignrun_window_schema')]` never advances as far as `0005`, and no test
asserts anything about the new `campaign_run_window_start_end_null_together` `CheckConstraint`.

I grepped every test file in scope (`test_campaign_models.py`, `test_campaign_gap.py`, and the
rest) for a test that creates a `CampaignRun` with `window_start` set and `window_end=None` (or
vice versa) and expects an `IntegrityError` — there is none. `TestCampaignRunWindowSchema` in
`test_campaign_models.py` covers the two `UniqueConstraint`s (TBD collision, resolved-window
collision) but not the new `CheckConstraint`.

The WR-02 fix report's own rationale for why `campaign_gap.claimed_dates()`'s defensive `run.
window_start is None or run.window_end is None` guard is enough — i.e. that the DB now makes the
mismatched-pair state genuinely unreachable in practice — is currently unverified by any automated
test. If a future change accidentally drops or weakens the constraint (e.g. during a migration
squash, a schema refactor, or a typo in the `Q` expression), nothing in the suite would catch it,
and the application-level fallback (`undated_runs` bucketing) would remain the only defense against
a state the code otherwise assumes can't occur.

**Fix:** Add one test alongside `TestCampaignRunWindowSchema` in `test_campaign_models.py` that
asserts the DB rejects a partial-null row:

```python
def test_mismatched_window_start_end_pair_rejected_by_db(self):
    """WR-02: window_start/window_end must be null together at the DB level."""
    with self.assertRaises(IntegrityError):
        with transaction.atomic():
            CampaignRun.objects.create(
                campaign=self.campaign,
                telescope_instrument='FTN/MuSCAT3',
                window_start=date(2025, 7, 4),
                window_end=None,
            )
```

Optionally also extend `test_window_schema_migration.py`'s `migrate_to` to `0005` (or add a second
migration test class) so the historical-schema coverage includes the `CheckConstraint`'s own
`AddConstraint` step, for symmetry with the `0004` coverage already there.

#### WR-02: Migration 0005 adds a CheckConstraint with no defensive data-cleanup step, unlike migration 0004's own CR-01 precedent in this same changeset

**File:** `solsys_code/migrations/0005_campaignrun_campaign_run_window_start_end_null_together.py:13-24`
**Issue:**

Migration `0004` (this same changeset) went out of its way to add
`dedupe_resolved_window_collisions` (the CR-01 fix) specifically because adding a constraint
against real, potentially-violating pre-existing data raises an unhandled `IntegrityError` with no
recovery path. Migration `0005` adds `campaign_run_window_start_end_null_together` as a bare
`AddConstraint` with no equivalent guard or cleanup `RunPython` step:

```python
operations = [
    migrations.AddConstraint(
        model_name='campaignrun',
        constraint=models.CheckConstraint(
            condition=models.Q(
                models.Q(('window_end__isnull', True), ('window_start__isnull', True)),
                models.Q(('window_end__isnull', False), ('window_start__isnull', False)),
                _connector='OR',
            ),
            name='campaign_run_window_start_end_null_together',
        ),
    ),
]
```

In this specific case the invariant already holds for every row at the point `0005` runs —
`0004`'s `backfill_window_fields` sets `window_start`/`window_end` from the same source column via
`F('obs_date')`, so the two fields are always either both `NULL` or both equal, and no write path
in this changeset (`CampaignRunSubmissionView`, `import_campaign_csv`) ever sets them
independently. Today's `0004`→`0005` sequence is therefore safe in practice. But that safety is
entirely incidental to `0005` itself — it depends on `0004` having already run in the same deploy.
If `0004` and `0005` were ever applied as separate deploys (not unusual for a squashed or
backported migration set), and any out-of-band write in the gap between them (a fixture load, a
data-migration bug elsewhere, a direct DB edit) created a mismatched-pair row, `0005` would fail
with an opaque `IntegrityError` and no documented cleanup step — contradicting the discipline this
same author just established one migration earlier, in this same file set, for the exact same
class of risk.

**Fix:** Either add a small defensive `RunPython` step ahead of the `AddConstraint` in `0005`
that normalizes any orphaned single-sided value (e.g. `CampaignRun.objects.filter(
window_start__isnull=True, window_end__isnull=False).update(window_end=None)` and the mirror
case for `window_start__isnull=False, window_end__isnull=True`), or at minimum add a comment in
`0005` explicitly documenting the "safe only because 0004 guarantees this in the same deploy"
assumption, so a future migration-squash or backport doesn't silently reintroduce the risk CR-01
already taught this codebase to guard against.

## Info

#### IN-01: Notebook narrative text still contradicts its own executed output (carried over, unfixed — expected, was out of scope for the `--fix` pass)

**File:** `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` (markdown cell `c37e3856`
and the summary table in cell `90d785d4`, vs. the executed output of code cell `1093927e`)
**Issue:**

Re-confirmed still present against the notebook's current committed JSON. The markdown cell above
the row-inspection cell (`c37e3856`) says *"all 6 should be `created` on a first run,"* and the
final summary table's CAMP-04 row (`90d785d4`) says *"Demonstrated by: Import cell summary (6
created) and re-run cell summary (6 unchanged)."* The actual executed output of the import cell
(`1093927e`) is `Done. created: 0, updated: 0, unchanged: 6, skipped: 0, site_needs_review: 1` for
**both** the first and second `call_command` invocations — the fixture rows were already imported
into the shared dev DB from a prior notebook run, so neither run ever reports `created: 6`. The
narrative text was never updated to match the executed output committed right next to it.

**Fix:** Update the `c37e3856` markdown cell and the `90d785d4` summary-table row to describe what
the executed output actually shows (`unchanged: 6` on both runs against this notebook's
persistent dev-DB state, matching the tone the later-added "Idempotency note" cell `c32cae1e`
already uses), or reset the target dev DB and re-execute+recommit the notebook so a fresh
`created: 6` first run genuinely matches the narrative claim.

#### IN-02: Ground-vs-space calendar-projection branch still treats OCCULTATION/RADAR sites identically to OPTICAL (carried over, unfixed — expected, was out of scope for the `--fix` pass)

**File:** `solsys_code/campaign_views.py:339-375` (`CampaignRunDecisionView.post`)
**Issue:**

Re-confirmed still present against current file contents. The branch is `if
run.site.observations_type == Observatory.SATELLITE_OBSTYPE: ... else: ...
sun_event(run.site, run.window_start, kind='sun')`. `Observatory.OBSTYPE_CHOICES`
(`solsys_code/solsys_code_observatory/models.py:17-25`) has four members —
`OPTICAL_OBSTYPE`, `OCCULTATION_OBSTYPE`, `SATELLITE_OBSTYPE`, `RADAR_OBSTYPE` — and every
non-`SATELLITE` value, including `OCCULTATION` and `RADAR`, falls into the `else` branch commented
`# Ground-based observatory:`, unconditionally requiring a dip-corrected dark window via
`sun_event(kind='sun')`. That's a reasonable default for optical sites but a real semantic
mismatch for occultation campaigns (whose observing window is the predicted occultation timing,
not local darkness) and for radar (which routinely operates in daylight). No test in
`test_campaign_approval.py` exercises an `OCCULTATION_OBSTYPE` or `RADAR_OBSTYPE` site through
this path — both fixtured `Observatory` records used across the approval tests are
`OPTICAL_OBSTYPE` or `SATELLITE_OBSTYPE` only.

**Fix:** Unchanged from the prior review's recommendation — either scope the calendar-projection
branch condition to `observations_type == Observatory.OPTICAL_OBSTYPE` explicitly (falling back to
no projection, same as an unresolved site or TBD run, for `OCCULTATION`/`RADAR`), or add an
explicit code comment acknowledging this is a deliberate simplification deferred to a later phase.
As before, this is Info-level because it doesn't crash or corrupt data — it silently produces the
wrong kind of `CalendarEvent` window for a site type this milestone has no real fixtures for yet.

---

_Reviewed: 2026-07-10T09:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
