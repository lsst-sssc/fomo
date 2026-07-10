---
phase: 19-window-schema-migration
reviewed: 2026-07-09T23:30:00Z
depth: deep
files_reviewed: 15
files_reviewed_list:
  - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
  - solsys_code/campaign_forms.py
  - solsys_code/campaign_gap.py
  - solsys_code/campaign_tables.py
  - solsys_code/campaign_utils.py
  - solsys_code/campaign_views.py
  - solsys_code/management/commands/import_campaign_csv.py
  - solsys_code/migrations/0004_campaignrun_window_schema.py
  - solsys_code/models.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/tests/test_campaign_gap.py
  - solsys_code/tests/test_campaign_models.py
  - solsys_code/tests/test_campaign_submission.py
  - solsys_code/tests/test_campaign_views.py
  - solsys_code/tests/test_import_campaign_csv.py
findings:
  critical: 1
  warning: 3
  info: 1
  total: 5
status: issues_found
---

# Phase 19: Code Review Report

**Reviewed:** 2026-07-09T23:30:00Z
**Depth:** deep
**Files Reviewed:** 15
**Status:** issues_found

## Summary

Phase 19 collapses `CampaignRun`'s `obs_date`/`ut_start`/`ut_end` triple into a
`window_start`/`window_end` `DateField` pair, backed by a single hard-cutover migration
(`0004_campaignrun_window_schema.py`) plus two partial `UniqueConstraint`s. The consumer-side
rewrite (`campaign_gap.py`, `campaign_views.py`, `campaign_tables.py`, `campaign_forms.py`,
`import_campaign_csv.py`) is generally careful and well-documented, and the TBD-branch
(`window_start IS NULL`) dedup step is real, tested against known production duplicates, and
correctly ordered ahead of its `AddConstraint`.

However, the migration's data-preserving dedup logic has an **asymmetry that the resolved-window
branch does not share with the TBD branch**: the old `unique_campaign_run_natural_key` constraint
was keyed on `(campaign, telescope_instrument, ut_start)`, so two pre-migration rows sharing the
same `obs_date` but a *different* `ut_start` (two genuinely distinct sessions the same night) were
legal and did not collide. After `backfill_window_fields` collapses `window_start = window_end =
obs_date` (dropping the time-of-night distinction entirely), those same two rows now share an
identical `(campaign, telescope_instrument, window_start, window_end)` tuple — and only the TBD
branch has a dedup step ahead of its `AddConstraint`; the resolved branch has none. This is a
correctness/robustness gap in exactly the area this review was asked to focus on, and the planning
artifacts (`19-RESEARCH.md` Wave 0 Gaps, `19-VALIDATION.md`) confirm this scenario was never
identified, dry-run tested, or defended against — only the TBD branch was validated against real
dev-DB duplicate pairs before this migration shipped.

A second, related gap: `import_campaign_csv.py`'s natural-key lookup (`insert_or_create_campaign_run`'s
`lookup` dict) keys only on `window_start`, not `window_end`, even though the model's own resolved-window
constraint — and its own `Meta` docstring's stated rationale — requires both. This is currently
unreachable (every write path in this changeset always sets `window_end == window_start`), but it
directly contradicts the invariant the model's Meta comment documents, and would either silently
collapse a future multi-night range's `window_end` back to a single date, or crash the import command
with an uncaught `MultipleObjectsReturned`, the moment window ranges become reachable (Phase 20, or
any direct-ORM row).

A third gap: nothing in the model or in `campaign_gap.claimed_dates()` guards against a
`window_start`-set/`window_end`-`None` (or vice versa) mismatched row — a state no current write path
produces, but one no `CheckConstraint` prevents either, and one that would raise an uncaught
`TypeError` inside the coverage-gap computation if it ever occurred.

## Critical Issues

### CR-01: Migration's dedup step only covers the TBD branch, not the structurally identical resolved-window collision it creates

**File:** `solsys_code/migrations/0004_campaignrun_window_schema.py:17-47,93-100`
**Issue:**

`backfill_window_fields` (lines 17-21) sets `window_start = window_end = obs_date` for **every**
row, discarding the time-of-night distinction the old constraint relied on. The old constraint
(`solsys_code/migrations/0003_campaignrun_natural_key_unique_constraint.py:14-19`) was keyed on
`(campaign, telescope_instrument, ut_start)` — not `obs_date` — so two pre-migration rows for the
same campaign/telescope/night but two different UT times (e.g. two separate observing blocks the
same night) were legal, non-colliding rows under the old schema.

After the backfill collapses both rows to the same `(window_start, window_end)` pair, they become
an exact duplicate under the new `unique_campaign_run_resolved_window` constraint
(`AddConstraint` at lines 93-100, mirrored in `models.py:124-128`). `dedupe_tbd_collisions`
(lines 23-47) only deletes duplicates among `window_start IS NULL` rows — it does nothing for
this resolved-window case. There is no equivalent dedup step before the resolved-window
`AddConstraint` runs.

Concretely: if the pre-migration `CampaignRun` table contains **any** two rows sharing
`(campaign, telescope_instrument, obs_date)` with distinct, non-null `ut_start` values (a
perfectly legal state under the schema this migration is replacing), this non-reversible,
single-file migration (D-02) will raise `IntegrityError` when it reaches the
`unique_campaign_run_resolved_window` `AddConstraint` step and fail to apply — on whatever
environment first encounters that data shape (a fresh deploy re-importing full historical CSV
data before migrating, a staging DB seeded from a richer historical export, etc.), not just the
one dev DB this phase's manual dry-run happened to check.

This is precisely the gap `19-RESEARCH.md`'s own "Wave 0 Gaps" section flags and never resolves:
*"No migration-level test/dry-run convention exists in this repo yet for verifying `RunPython`
backfill+dedup correctness against realistic pre-migration data shapes"* — the phase-gate dry run
that *was* run only characterized and fixed the two known TBD-branch duplicate pairs (pks
15/17, 16/18 per `19-VALIDATION.md`), never the resolved-window case.

**Fix:** Add a second, symmetric dedup `RunPython` step (mirroring `dedupe_tbd_collisions`) for the
resolved-window branch, inserted anywhere between the backfill and the resolved-window
`AddConstraint`:

```python
def dedupe_resolved_window_collisions(apps, schema_editor):
    """Analogous to dedupe_tbd_collisions, but for the resolved-window branch: two
    pre-migration rows that shared campaign+telescope_instrument+obs_date but differed
    only by the now-dropped ut_start collapse onto an identical (window_start, window_end)
    tuple after backfill_window_fields runs. Must run before the resolved-window
    UniqueConstraint is added below, or that AddConstraint can fail against real
    pre-existing same-night, different-ut_start rows.
    """
    CampaignRun = apps.get_model('solsys_code', 'CampaignRun')
    seen: dict[tuple, int] = {}
    qs = CampaignRun.objects.filter(window_start__isnull=False).order_by('pk')
    for run in qs:
        key = (run.campaign_id, run.telescope_instrument, run.window_start, run.window_end)
        if key in seen:
            logger.warning(
                'Deleting duplicate resolved-window CampaignRun pk=%s (kept pk=%s) for '
                'campaign=%s telescope_instrument=%r window=%s..%s',
                run.pk, seen[key], run.campaign_id, run.telescope_instrument,
                run.window_start, run.window_end,
            )
            run.delete()
        else:
            seen[key] = run.pk
```

and insert `migrations.RunPython(dedupe_resolved_window_collisions, reverse_code=migrations.RunPython.noop)`
immediately after `dedupe_tbd_collisions` (before `RemoveConstraint`). Before shipping, re-run the
manual `manage.py migrate` dry run against a copy of the real dev DB with this step added, and
confirm whether it actually deletes anything — if it does, this finding was live, not
hypothetical.

## Warnings

### WR-01: CSV importer's natural-key lookup omits `window_end`, contradicting the model's own documented resolved-window contract

**File:** `solsys_code/management/commands/import_campaign_csv.py:170-173`
**Issue:**

```python
run, action = insert_or_create_campaign_run(
    {'campaign': campaign, 'telescope_instrument': telescope_instrument, 'window_start': obs_date},
    fields,
)
```

`fields` (built at line 149-168) sets `window_end` but the `lookup` dict passed to
`insert_or_create_campaign_run` — and hence to `CampaignRun.objects.get_or_create(**lookup, ...)`
in `campaign_utils.py:301` — only constrains `window_start`. `models.py:120-124`'s own comment
is explicit about why this is wrong: *"window_end is included (not just window_start) so a range
starting on the same day as an existing single-night entry is not treated as the same row."* The
importer's lookup doesn't follow that rule.

This is currently unreachable — every write path shipped in this phase (`CampaignRunSubmissionView`,
this importer) always sets `window_end == window_start`, so no real range row exists yet to collide
with. But it is a landmine for the very next feature that creates a range (Phase 20, or a direct
admin/ORM edit): re-running this importer over a campaign that already has a resolved multi-night
`CampaignRun` starting on the CSV row's `obs_date` would either
- match that range row via `get()` (since only `window_start` is filtered) and silently overwrite
  its `window_end` back down to `obs_date` via `insert_or_create_campaign_run`'s field-diff/update
  path (`campaign_utils.py:304-309`) — a silent data-loss collapse of a real range into a single
  night — or
- raise an uncaught `django.core.exceptions.MultipleObjectsReturned` from `get_or_create()` if more
  than one range shares that `window_start` — a crash the command's per-row `try/except ValueError`
  (`import_campaign_csv.py:108-126`) does not catch, aborting the whole batch import instead of
  skip-and-log per row.

**Fix:** Include `window_end` in the lookup, matching the model's actual key:

```python
run, action = insert_or_create_campaign_run(
    {
        'campaign': campaign,
        'telescope_instrument': telescope_instrument,
        'window_start': obs_date,
        'window_end': obs_date,
    },
    fields,  # drop 'window_end' from fields; it's now part of the lookup key
)
```

### WR-02: No DB-level invariant that `window_start`/`window_end` are null together; `claimed_dates()` will crash on a mismatched row

**File:** `solsys_code/models.py:113-140` (missing `CheckConstraint`); `solsys_code/campaign_gap.py:176-183`
**Issue:**

Every reader of `window_start`/`window_end` in this changeset (`campaign_tables.render_window_start`,
`campaign_views.CampaignRunDecisionView.post`, `campaign_gap.claimed_dates`) assumes the two fields
are either both `NULL` (TBD) or both set (resolved). Nothing enforces this at the model or DB layer —
only the two partial `UniqueConstraint`s exist, and neither one prevents `window_start` set with
`window_end` NULL (or vice versa); that combination isn't NULL on `window_start`, so it falls into
the resolved-window constraint's `condition=Q(window_start__isnull=False)` branch and is simply
allowed to be a unique (if odd) row.

`campaign_gap.claimed_dates()` (`campaign_gap.py:176-183`) only guards the fully-TBD case:

```python
for run in qs:
    if run.window_start is None:
        undated_runs.append(run)
        continue
    n_days = (run.window_end - run.window_start).days + 1   # TypeError if window_end is None
```

If a `CampaignRun` ever exists with `window_start` set and `window_end` `None` (currently
unreachable via any shipped write path, but reachable via direct ORM/admin/future code, e.g. a
partially-completed Phase 20 range-entry form, or a bad manual fixup), this raises an uncaught
`TypeError`, which propagates out of `_compute_gap()` / `get_or_compute_gap()` and 500s the entire
`CampaignGapAnalysisView` for that campaign+site+target combination — not a graceful per-run skip.

**Fix:** Either add a DB-level `CheckConstraint` enforcing the pairing:

```python
models.CheckConstraint(
    condition=(
        models.Q(window_start__isnull=True, window_end__isnull=True)
        | models.Q(window_start__isnull=False, window_end__isnull=False)
    ),
    name='campaign_run_window_start_end_null_together',
),
```

and/or defensively guard `claimed_dates()`:

```python
if run.window_start is None or run.window_end is None:
    undated_runs.append(run)
    continue
```

### WR-03: Migration's `RunPython` backfill/dedup logic has zero automated regression coverage

**File:** `solsys_code/migrations/0004_campaignrun_window_schema.py:17-47`; `solsys_code/tests/test_campaign_models.py`
**Issue:**

`backfill_window_fields` and `dedupe_tbd_collisions` are non-reversible, data-deleting functions
with real production consequences (see CR-01), but no test in this changeset exercises them
directly — `test_campaign_models.py`'s `TestCampaignRunWindowSchema` only asserts the *post-migration
model shape* (constraints, nullability), never the migration's own data-transformation functions.
`19-RESEARCH.md`'s own "Wave 0 Gaps" acknowledges this: *"No migration-level test/dry-run convention
exists in this repo yet for verifying `RunPython` backfill+dedup correctness against realistic
pre-migration data shapes"* — and the mitigation adopted (a manual, undocumented-in-code dry run
against a copy of the dev DB) leaves no artifact in the repository that this logic was ever
exercised, let alone that CR-01's gap was checked for.

**Fix:** At minimum, add a `TransactionTestCase`-based migration test (Django's
`django.test.migrations`/`MigrationTestCase`-style pattern, or a hand-rolled one using
`connection.schema_editor()` against the pre-0004 historical model state via
`django.apps.apps.get_model` from a frozen migration state) that seeds two rows sharing
`obs_date` with distinct `ut_start` values, runs `backfill_window_fields`/the migration, and
asserts the outcome (dedup or a documented, deliberately-accepted failure mode) rather than
relying entirely on an unrepeatable manual dry run.

## Info

### IN-01: Demo notebook's committed output contradicts its own narrative text

**File:** `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` (cell `48b89ebf` markdown vs. cell `1093927e`/`16d14cf8` output)
**Issue:**

The "synthetic fixture" markdown cell (unchanged by this phase's regeneration) says *"all 6 should
be `created` on a first run"*, but the regenerated executed output (per the `19-04` regeneration
commit) shows `created: 0, updated: 0, unchanged: 6` and a total row count of **10**, not 6 — because
the notebook was regenerated against the shared, already-populated dev DB (`src/fomo_db.sqlite3`)
rather than a clean one, as the commit message for `a295b3e` explicitly acknowledges. This is a
reasonable, documented trade-off (the notebook is explicitly designed to be idempotent/re-runnable
against a dirty dev DB per the newly-added "Idempotency note" cell), but the older, unedited
"6 should be created" sentence now reads as factually wrong next to the actual printed output a
reader sees immediately below it.

**Fix:** Update the "synthetic fixture" markdown cell's wording to acknowledge the notebook may show
`unchanged` instead of `created` on a rerun against a pre-populated dev DB, matching the tone of the
"Idempotency note" cell added later in the same notebook.

---

_Reviewed: 2026-07-09T23:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
